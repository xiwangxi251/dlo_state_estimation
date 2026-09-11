from __future__ import annotations

from time import perf_counter_ns

import numpy as np

from .geometry import resample_polyline
from .temporal_tracker import TemporalEstimate, _match_arc_parameters


class UPETrackTracker:
    """Lightweight reproduction of the geometry part of UPETrack.

    The published method uses GMM/EM for visible-node registration and the
    closed-form UPE equations for hidden nodes.  This implementation keeps the
    benchmark's existing RGB-D segmentation front-end and applies those
    equations to its ordered visible centerline samples.  It is therefore a
    reproducible UPETrack-style baseline, not the authors' unreleased code.
    """

    def __init__(
        self,
        *,
        sample_count: int = 14,
        expected_length_m: float = 0.78,
        gamma: float = 0.8,
        historical_weight: float = 0.6,
        bending_resistance: float = 0.8,
        blend_visible_prediction: float = 0.75,
        visibility_radius_m: float = 0.020,
        # The benchmark passes an ordered 14-point centerline rather than the
        # dense segmented point cloud used in the paper, so one nearby sample
        # is the appropriate visibility support here (the dense-cloud version
        # would use a larger neighbour count).
        visibility_count: int = 1,
        **_: object,
    ) -> None:
        if sample_count < 4:
            raise ValueError("sample_count must be at least four")
        self.sample_count = int(sample_count)
        self.expected_length_m = float(expected_length_m)
        self.gamma = float(np.clip(gamma, 0.0, 1.0))
        self.historical_weight = float(historical_weight)
        self.bending_resistance = float(max(bending_resistance, 0.0))
        self.blend_visible_prediction = float(np.clip(blend_visible_prediction, 0.0, 1.0))
        self.visibility_radius_m = float(max(visibility_radius_m, 1e-6))
        self.visibility_count = max(int(visibility_count), 1)
        self._state: np.ndarray | None = None
        self._previous_state: np.ndarray | None = None
        self._last_observed_interval: tuple[float, float] | None = None

    @property
    def initialized(self) -> bool:
        return self._state is not None

    @property
    def points_world(self) -> np.ndarray | None:
        return None if self._state is None else self._state.copy()

    def reset(self) -> None:
        self._state = None
        self._previous_state = None
        self._last_observed_interval = None

    def update(self, observed_world: np.ndarray, *, dt_s: float = 0.04, **_: object) -> TemporalEstimate:
        start = perf_counter_ns()
        observed = np.asarray(observed_world, dtype=np.float64)
        if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < 2:
            raise ValueError("observed_world must have shape (N, 3), N >= 2")
        if not np.isfinite(observed).all():
            raise ValueError("observed_world contains non-finite values")

        if self._state is None:
            state = resample_polyline(observed, self.sample_count)
            self._state = state.copy()
            self._previous_state = None
            return self._result(
                state,
                np.ones(self.sample_count, dtype=bool),
                coverage=1.0,
                used_prediction=False,
                matched_distance=0.0,
                start=start,
            )

        previous = self._state.copy()
        older = previous if self._previous_state is None else self._previous_state

        observed_length_m = float(np.linalg.norm(np.diff(observed, axis=0), axis=1).sum())
        # The paper performs GMM/EM registration against a dense point cloud.
        # This benchmark supplies an ordered centerline, so use the same
        # contiguous arc-interval objective as a deterministic compact proxy;
        # it prevents the visible nodes from drifting when the cable is fully
        # visible and still returns the arc interval needed by UPE.
        params, distance, orientation = _match_arc_parameters(
            previous,
            observed,
            observed_length_m=observed_length_m,
            expected_length_m=self.expected_length_m,
            preferred_interval=self._last_observed_interval,
            # Preserve the previously registered arc interval when the compact
            # 14-point observation is nearly full-length.  The dense GMM in
            # the paper supplies this identity cue implicitly; with a sparse
            # centerline we make it explicit to avoid static-frame jumps.
            continuity_weight=10.0,
            allow_rotation=observed_length_m / max(self.expected_length_m, 1e-9) >= 0.50,
        )
        if orientation:
            observed = observed[::-1].copy()
        keep = np.concatenate(([True], np.diff(params) > 1e-4))
        params = params[keep]
        observed = observed[keep]
        if len(params) >= 2:
            lower, upper = float(params[0]), float(params[-1])
            self._last_observed_interval = (lower, upper)
            state_params = np.linspace(0.0, 1.0, self.sample_count)
            visible = (state_params >= lower - 1e-6) & (state_params <= upper + 1e-6)
            visible_state = previous.copy()
            for axis in range(3):
                visible_state[visible, axis] = np.interp(
                    state_params[visible], params, observed[:, axis]
                )
        else:
            visible = np.zeros(self.sample_count, dtype=bool)
            visible_state = previous.copy()

        # Start hidden nodes from the old state and fill each contiguous
        # hidden interval from both directions when possible.
        current = visible_state.copy()
        hidden = ~visible
        available = visible.copy()
        remaining = set(map(int, np.flatnonzero(hidden)))
        # Once a hidden node is estimated, UPE treats it as available support
        # and propagates in the same direction until the entire gap is filled.
        while remaining:
            progressed = False
            for index in sorted(remaining):
                left = self._upe_from_side(index, -1, current, previous, older, available)
                right = self._upe_from_side(index, +1, current, previous, older, available)
                if left is None and right is None:
                    continue
                if left is not None and right is not None:
                    current[index] = 0.5 * (left + right)
                elif left is not None:
                    current[index] = left
                else:
                    current[index] = right
                available[index] = True
                remaining.remove(index)
                progressed = True
            if not progressed:
                break
        # A gap with fewer than three consecutive supporting nodes cannot be
        # resolved by the published UPE equations; expose the historical value
        # as a low-confidence fallback for those rare residual indices.
        for index in sorted(remaining):
            current[index] = previous[index]

        # Published UPETrack redistributes nodes by geodesic distance after
        # the occlusion update.  Keep the observed endpoints and re-sample.
        current = resample_polyline(current, self.sample_count)
        self._previous_state = previous
        self._state = current.copy()
        matched = float(distance)
        return self._result(
            current,
            visible,
            coverage=float(np.mean(visible)),
            used_prediction=bool(np.any(hidden)),
            matched_distance=matched,
            start=start,
        )

    def _upe_from_side(
        self,
        index: int,
        direction: int,
        current: np.ndarray,
        previous: np.ndarray,
        older: np.ndarray,
        visible: np.ndarray,
    ) -> np.ndarray | None:
        support = [index + direction * k for k in (1, 2, 3)]
        if any(i < 0 or i >= len(previous) or not visible[i] for i in support):
            return None
        # Re-index so p1,p2,p3 are ordered from the nearest visible node away
        # from the hidden node, matching the directional UPE recursion.
        p1, p2, p3 = support
        d1 = current[p1] - previous[p1]
        d2 = current[p2] - previous[p2]
        d3 = current[p3] - previous[p3]
        delta = (d1 + self.gamma * d2 + self.gamma * self.gamma * d3) / (
            1.0 + self.gamma + self.gamma * self.gamma
        )
        local = previous[index] + delta

        tangent = current[p1] - current[p2]
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-8:
            tangent = previous[p1] - previous[p2]
            tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-8:
            return local
        segment_length = float(np.linalg.norm(previous[index] - previous[p1]))
        proximal = current[p1] + self.bending_resistance * segment_length * tangent / tangent_norm
        historical = (older[p1] - older[index]) - (older[p2] - older[p1])
        curvature = proximal + self.historical_weight * historical
        return self.blend_visible_prediction * local + (1.0 - self.blend_visible_prediction) * curvature

    def _result(
        self,
        state: np.ndarray,
        visible: np.ndarray,
        *,
        coverage: float,
        used_prediction: bool,
        matched_distance: float,
        start: int,
    ) -> TemporalEstimate:
        elapsed = (perf_counter_ns() - start) * 1e-6
        return TemporalEstimate(
            points_world=state.copy(),
            observed_mask=np.asarray(visible, dtype=bool).copy(),
            coverage=float(coverage),
            confidence=float(coverage),
            initialized=True,
            used_prediction=bool(used_prediction),
            matched_distance_m=float(matched_distance),
            timings_ms={"total": float(elapsed)},
        )
