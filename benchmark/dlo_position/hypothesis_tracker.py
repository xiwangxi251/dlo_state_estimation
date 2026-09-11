from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from .temporal_tracker import TemporalDLOTracker, TemporalEstimate


@dataclass
class _ScoredHypothesis:
    tracker: TemporalDLOTracker
    score: float
    result: TemporalEstimate


class MultiHypothesisDLOTracker:
    """Small beam search over competing arc-identity continuations.

    The ordinary tracker commits to one arc interval in every frame.  During a
    short/occluded observation, a second plausible explanation is deliberately
    retained: one child follows the normal matcher and one child strongly
    preserves the previous interval.  The lower-residual child wins on the next
    frame, while the beam prevents one ambiguous frame from irreversibly
    changing point identity.  The state is still a 14-point vector and the
    default beam width is two, keeping the extra work bounded.
    """

    def __init__(self, *, beam_width: int = 2, hold_weight: float = 20.0, **kwargs) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be at least one")
        self.beam_width = int(beam_width)
        self.hold_weight = float(max(hold_weight, 1.0))
        self._kwargs = dict(kwargs)
        self._hypotheses: list[_ScoredHypothesis] = []

    @property
    def initialized(self) -> bool:
        return bool(self._hypotheses and self._hypotheses[0].tracker.initialized)

    @property
    def points_world(self) -> np.ndarray | None:
        if not self._hypotheses:
            return None
        return self._hypotheses[0].tracker.points_world

    def reset(self) -> None:
        self._hypotheses = []

    def update(self, observed_world: np.ndarray, **kwargs) -> TemporalEstimate:
        start_ns = perf_counter_ns()
        if not self._hypotheses:
            tracker = TemporalDLOTracker(**self._kwargs)
            result = tracker.update(observed_world, **kwargs)
            self._hypotheses = [_ScoredHypothesis(tracker, 0.0, result)]
            return _with_hypothesis_timing(result, start_ns, 1, 0.0)

        children: list[_ScoredHypothesis] = []
        for parent in self._hypotheses:
            # The normal child follows all existing motion/arc cues.
            normal_tracker = copy.deepcopy(parent.tracker)
            normal_result = normal_tracker.update(observed_world, **kwargs)
            children.append(
                _ScoredHypothesis(
                    normal_tracker,
                    _child_score(parent, normal_result),
                    normal_result,
                )
            )

            # A short observation is where an arc identity can be ambiguous.
            # Keep a second child anchored to the previous interval.  Do not
            # create this branch for a nearly complete observation, where the
            # direct measurement is already identity-safe.
            observed_length = kwargs.get("observed_length_m")
            if observed_length is None:
                observed_length = float(
                    np.linalg.norm(np.diff(observed_world, axis=0), axis=1).sum()
                )
            ratio = float(observed_length) / max(normal_tracker.expected_length_m, 1e-9)
            previous_interval = parent.tracker._last_observed_interval
            if ratio < 0.65 and previous_interval is not None:
                hold_tracker = copy.deepcopy(parent.tracker)
                hold_tracker.arc_continuity_weight = max(
                    hold_tracker.arc_continuity_weight, self.hold_weight
                )
                hold_result = hold_tracker.update(observed_world, **kwargs)
                children.append(
                    _ScoredHypothesis(
                        hold_tracker,
                        _child_score(parent, hold_result),
                        hold_result,
                    )
                )

        children.sort(key=lambda item: item.score)
        # Remove near-identical children so the beam represents distinct
        # explanations instead of copies produced by an unambiguous frame.
        selected: list[_ScoredHypothesis] = []
        for child in children:
            if child.tracker.points_world is None:
                continue
            if any(
                np.mean(
                    np.linalg.norm(
                        child.tracker.points_world - other.tracker.points_world,
                        axis=1,
                    )
                )
                < 1e-4
                for other in selected
            ):
                continue
            selected.append(child)
            if len(selected) >= self.beam_width:
                break
        if not selected:
            selected = children[:1]
        self._hypotheses = selected
        best = self._hypotheses[0]
        return _with_hypothesis_timing(
            best.result,
            start_ns,
            len(self._hypotheses),
            best.score,
        )


def _child_score(parent: _ScoredHypothesis, result: TemporalEstimate) -> float:
    residual = result.matched_distance_m
    if not np.isfinite(residual):
        residual = 0.25
    # Keep a weak memory of the parent's quality, but let a new observation
    # overturn a stale hypothesis after one or two frames.
    return 0.25 * parent.score + float(residual)


def _with_hypothesis_timing(
    result: TemporalEstimate,
    start_ns: int,
    count: int,
    score: float,
) -> TemporalEstimate:
    timings = dict(result.timings_ms)
    timings["hypothesis_count"] = float(count)
    timings["hypothesis_score"] = float(score)
    timings["hypothesis_total"] = (perf_counter_ns() - start_ns) * 1e-6
    return TemporalEstimate(
        points_world=result.points_world,
        observed_mask=result.observed_mask,
        coverage=result.coverage,
        confidence=result.confidence,
        initialized=result.initialized,
        used_prediction=result.used_prediction,
        matched_distance_m=result.matched_distance_m,
        timings_ms=timings,
    )
