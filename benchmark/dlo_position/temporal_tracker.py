from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
import cv2

from .geometry import project_points, resample_polyline, transform_points


@dataclass
class TemporalEstimate:
    """A complete DLO state reconstructed from a partial observation."""

    points_world: np.ndarray | None
    observed_mask: np.ndarray
    coverage: float
    confidence: float
    initialized: bool
    used_prediction: bool
    matched_distance_m: float
    timings_ms: dict[str, float]


class TemporalDLOTracker:
    """Lightweight motion-coherent completion for a fixed-length DLO.

    The tracker keeps a complete ordered node state.  A partial observation is
    matched to the previous curve in arc-length coordinates; visible nodes are
    fused with a constant-velocity prediction while hidden nodes retain the
    prediction.  The implementation deliberately operates on the small state
    representation (14 nodes by default), so it can run inside a 50 Hz loop.
    """

    def __init__(
        self,
        *,
        sample_count: int = 14,
        expected_length_m: float = 0.8,
        min_initial_length_ratio: float = 0.80,
        observation_gain: float = 1.0,
        velocity_gain: float = 0.15,
        velocity_decay: float = 0.80,
        length_regularization_gain: float = 0.30,
        low_coverage_gain_scale: float = 0.50,
        confidence_gain_scale: float = 0.0,
        confidence_gain_floor: float = 0.25,
        confidence_gain_reference: float = 0.15,
        confidence_gain_low_coverage_only: bool = False,
        rigid_transform_gain: float = 1.0,
        rigid_residual_threshold_m: float = 0.025,
        rigid_min_prior_step_m: float = 0.004,
        rigid_max_translation_m: float = 0.30,
        rigid_max_angle_deg: float = 45.0,
        arc_continuity_weight: float = 1.0,
        centroid_motion_weight: float = 0.50,
        centroid_motion_slack_m: float = 0.15,
        sequence_min_translation_m: float = 0.004,
        rigid_min_match_confidence: float = 0.0,
        sequence_motion_weight: float = 2.0,
        sequence_motion_min_prior_step_m: float = 0.01,
        sequence_disagreement_translation_m: float = 1.4,
        sequence_disagreement_slack_m: float = 0.15,
        sequence_disagreement_persistence: int = 1,
        sequence_disagreement_hold_previous: bool = False,
        sequence_disagreement_use_centroid_translation: bool = True,
        sequence_disagreement_low_coverage_only: bool = True,
        sequence_disagreement_velocity_gain: float = 0.5,
        sequence_disagreement_velocity_min_centroid_step_m: float = 0.03,
        low_coverage_deformation_gain: float = 0.5,
        low_coverage_deformation_residual_m: float = 0.012,
        low_coverage_deformation_min_centroid_step_m: float = 0.02,
        low_coverage_deformation_persistence: int = 2,
        low_coverage_deformation_rigid_gain: float = 0.5,
        low_coverage_rigid_observation_gain: float = 1.0,
        hidden_motion_gain: float = 0.70,
        reacquisition_min_observed_ratio: float = 1.0,
        reacquisition_min_coverage: float = 0.85,
        reacquisition_max_match_confidence: float = -1.0,
        arc_hysteresis_jump: float = 0.0,
        arc_hysteresis_margin_m: float = 0.0,
        allow_partial_initialization: bool = False,
    ) -> None:
        if sample_count < 2:
            raise ValueError("sample_count must be at least two")
        if expected_length_m <= 0.0:
            raise ValueError("expected_length_m must be positive")
        self.sample_count = int(sample_count)
        self.expected_length_m = float(expected_length_m)
        self.min_initial_length_ratio = float(min_initial_length_ratio)
        self.observation_gain = float(np.clip(observation_gain, 0.0, 1.0))
        self.velocity_gain = float(np.clip(velocity_gain, 0.0, 1.0))
        self.velocity_decay = float(np.clip(velocity_decay, 0.0, 1.0))
        self.length_regularization_gain = float(
            np.clip(length_regularization_gain, 0.0, 1.0)
        )
        self.low_coverage_gain_scale = float(
            np.clip(low_coverage_gain_scale, 0.0, 1.0)
        )
        self.confidence_gain_scale = float(np.clip(confidence_gain_scale, 0.0, 1.0))
        self.confidence_gain_floor = float(np.clip(confidence_gain_floor, 0.0, 1.0))
        self.confidence_gain_reference = float(max(confidence_gain_reference, 1e-6))
        self.confidence_gain_low_coverage_only = bool(confidence_gain_low_coverage_only)
        self.rigid_transform_gain = float(np.clip(rigid_transform_gain, 0.0, 1.0))
        self.rigid_residual_threshold_m = float(max(rigid_residual_threshold_m, 0.0))
        self.rigid_min_prior_step_m = float(max(rigid_min_prior_step_m, 0.0))
        self.rigid_max_translation_m = float(max(rigid_max_translation_m, 0.0))
        self.rigid_max_angle_rad = float(max(np.deg2rad(rigid_max_angle_deg), 0.0))
        self.arc_continuity_weight = float(max(arc_continuity_weight, 0.0))
        self.centroid_motion_weight = float(max(centroid_motion_weight, 0.0))
        self.centroid_motion_slack_m = float(max(centroid_motion_slack_m, 0.0))
        self.sequence_min_translation_m = float(max(sequence_min_translation_m, 0.0))
        self.rigid_min_match_confidence = float(
            np.clip(rigid_min_match_confidence, 0.0, 1.0)
        )
        self.sequence_motion_weight = float(max(sequence_motion_weight, 0.0))
        self.sequence_motion_min_prior_step_m = float(
            max(sequence_motion_min_prior_step_m, 0.0)
        )
        self.sequence_disagreement_translation_m = float(
            max(sequence_disagreement_translation_m, 0.0)
        )
        self.sequence_disagreement_slack_m = float(
            max(sequence_disagreement_slack_m, 0.0)
        )
        self.sequence_disagreement_persistence = max(
            int(sequence_disagreement_persistence), 1
        )
        self.sequence_disagreement_hold_previous = bool(
            sequence_disagreement_hold_previous
        )
        self.sequence_disagreement_use_centroid_translation = bool(
            sequence_disagreement_use_centroid_translation
        )
        self.sequence_disagreement_low_coverage_only = bool(
            sequence_disagreement_low_coverage_only
        )
        self.sequence_disagreement_velocity_gain = float(
            np.clip(sequence_disagreement_velocity_gain, 0.0, 1.0)
        )
        self.sequence_disagreement_velocity_min_centroid_step_m = float(
            max(sequence_disagreement_velocity_min_centroid_step_m, 0.0)
        )
        self.low_coverage_deformation_gain = float(
            np.clip(low_coverage_deformation_gain, 0.0, 1.0)
        )
        self.low_coverage_deformation_residual_m = float(
            max(low_coverage_deformation_residual_m, 0.0)
        )
        self.low_coverage_deformation_min_centroid_step_m = float(
            max(low_coverage_deformation_min_centroid_step_m, 0.0)
        )
        self.low_coverage_deformation_persistence = max(
            int(low_coverage_deformation_persistence), 1
        )
        self.low_coverage_deformation_rigid_gain = float(
            np.clip(low_coverage_deformation_rigid_gain, 0.0, 1.0)
        )
        self.low_coverage_rigid_observation_gain = float(
            np.clip(low_coverage_rigid_observation_gain, 0.0, 1.0)
        )
        self.hidden_motion_gain = float(np.clip(hidden_motion_gain, 0.0, 1.0))
        self.reacquisition_min_observed_ratio = float(
            max(reacquisition_min_observed_ratio, 0.0)
        )
        self.reacquisition_min_coverage = float(
            np.clip(reacquisition_min_coverage, 0.0, 1.0)
        )
        self.reacquisition_max_match_confidence = float(
            reacquisition_max_match_confidence
        )
        self.arc_hysteresis_jump = float(max(arc_hysteresis_jump, 0.0))
        self.arc_hysteresis_margin_m = float(max(arc_hysteresis_margin_m, 0.0))
        self.allow_partial_initialization = bool(allow_partial_initialization)
        self._points_world: np.ndarray | None = None
        self._velocity_world: np.ndarray | None = None
        self._last_observed_interval: tuple[float, float] | None = None
        self._last_rigid_residual_m = float("nan")
        self._last_rigid_translation_m = float("nan")
        self._last_rigid_angle_rad = float("nan")
        self._last_rigid_applied = False
        self._last_prior_step_m = float("nan")
        self._last_observed_centroid: np.ndarray | None = None
        self._last_centroid_delta: np.ndarray | None = None
        self._last_centroid_step_m = float("nan")
        self._last_observed_points: np.ndarray | None = None
        self._last_sequence_residual_m = float("nan")
        self._last_sequence_translation_m = float("nan")
        self._last_sequence_angle_rad = float("nan")
        self._last_sequence_applied = False
        self._sequence_disagreement_count = 0
        self._deformation_motion_count = 0
        self._last_observed_ratio: float | None = None
        self._partial_identity_hold = False
        self._last_robot_occlusion_fraction = float("nan")

    def reset(self) -> None:
        self._points_world = None
        self._velocity_world = None
        self._last_observed_interval = None
        self._last_rigid_residual_m = float("nan")
        self._last_rigid_translation_m = float("nan")
        self._last_rigid_angle_rad = float("nan")
        self._last_rigid_applied = False
        self._last_prior_step_m = float("nan")
        self._last_observed_centroid = None
        self._last_centroid_delta = None
        self._last_centroid_step_m = float("nan")
        self._last_observed_points = None
        self._last_sequence_residual_m = float("nan")
        self._last_sequence_translation_m = float("nan")
        self._last_sequence_angle_rad = float("nan")
        self._last_sequence_applied = False
        self._sequence_disagreement_count = 0
        self._deformation_motion_count = 0
        self._last_observed_ratio = None
        self._partial_identity_hold = False
        self._last_robot_occlusion_fraction = float("nan")

    @property
    def initialized(self) -> bool:
        return self._points_world is not None

    @property
    def points_world(self) -> np.ndarray | None:
        return None if self._points_world is None else self._points_world.copy()

    def update(
        self,
        observed_world: np.ndarray,
        *,
        observed_length_m: float | None = None,
        dt_s: float = 0.04,
        observed_pixels: np.ndarray | None = None,
        observed_image_mask: np.ndarray | None = None,
        camera_from_world: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        robot_occlusion_mask: np.ndarray | None = None,
        image_match_weight: float = 0.0,
        image_match_min_improvement_m: float = 0.02,
    ) -> TemporalEstimate:
        start_ns = perf_counter_ns()
        observed = np.asarray(observed_world, dtype=np.float64)
        if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < 2:
            raise ValueError("observed_world must have shape (N, 3), N >= 2")
        if not np.isfinite(observed).all():
            raise ValueError("observed_world contains non-finite values")
        measurement_pixels = None
        if observed_pixels is not None:
            measurement_pixels = np.asarray(observed_pixels, dtype=np.float64)
            if measurement_pixels.shape != (len(observed), 2) or not np.isfinite(
                measurement_pixels
            ).all():
                raise ValueError(
                    "observed_pixels must have shape (N, 2) and contain finite values"
                )
            if camera_from_world is None or intrinsics is None:
                raise ValueError(
                    "camera_from_world and intrinsics are required with observed_pixels"
                )
        image_mask = None
        if observed_image_mask is not None:
            image_mask = np.asarray(observed_image_mask)
            if image_mask.ndim != 2 or image_mask.size == 0:
                raise ValueError("observed_image_mask must be a non-empty HxW array")
            image_mask = image_mask.astype(bool, copy=False)
        robot_mask = None
        if robot_occlusion_mask is not None:
            robot_mask = np.asarray(robot_occlusion_mask).astype(bool, copy=False)
            if robot_mask.ndim != 2 or robot_mask.size == 0:
                raise ValueError("robot_occlusion_mask must be a non-empty HxW array")
            if camera_from_world is None or intrinsics is None:
                raise ValueError(
                    "camera_from_world and intrinsics are required with robot_occlusion_mask"
                )
        dt = float(np.clip(dt_s, 1e-3, 0.25))
        if observed_length_m is None:
            observed_length_m = float(
                np.linalg.norm(np.diff(observed, axis=0), axis=1).sum()
            )
        observed_length_m = float(observed_length_m)
        observed_centroid = observed.mean(axis=0)
        centroid_step_m = (
            None
            if self._last_observed_centroid is None
            else float(np.linalg.norm(observed_centroid - self._last_observed_centroid))
        )
        centroid_delta = (
            None
            if self._last_observed_centroid is None
            else observed_centroid - self._last_observed_centroid
        )
        self._last_observed_centroid = observed_centroid.copy()
        self._last_centroid_delta = (
            None if centroid_delta is None else centroid_delta.copy()
        )
        self._last_centroid_step_m = (
            float("nan") if centroid_step_m is None else centroid_step_m
        )

        sequence_motion = None
        self._last_sequence_residual_m = float("nan")
        self._last_sequence_translation_m = float("nan")
        self._last_sequence_angle_rad = float("nan")
        self._last_sequence_applied = False
        if self._last_observed_points is not None:
            sequence_motion = _fit_sequence_transform(
                self._last_observed_points,
                observed,
                allow_rotation=(
                    observed_length_m / max(self.expected_length_m, 1e-9) >= 0.50
                ),
            )
            if sequence_motion[3]:
                observed = observed[::-1].copy()
                if measurement_pixels is not None:
                    measurement_pixels = measurement_pixels[::-1].copy()
        self._last_observed_points = observed.copy()
        sequence_disagreement = False
        if sequence_motion is not None:
            sequence_translation_norm = float(np.linalg.norm(sequence_motion[1]))
            sequence_disagreement = bool(
                sequence_motion[2] < 0.012
                and sequence_translation_norm
                > self.sequence_disagreement_translation_m
                and centroid_step_m is not None
                and sequence_translation_norm
                > centroid_step_m + self.sequence_disagreement_slack_m
            )
        if sequence_disagreement:
            self._sequence_disagreement_count += 1
        else:
            self._sequence_disagreement_count = 0
        deformation_motion = bool(
            sequence_motion is not None
            and sequence_motion[2] > self.low_coverage_deformation_residual_m
            and centroid_step_m is not None
            and centroid_step_m
            >= self.low_coverage_deformation_min_centroid_step_m
            and observed_length_m / max(self.expected_length_m, 1e-9) < 0.50
        )
        if deformation_motion:
            self._deformation_motion_count += 1
        else:
            self._deformation_motion_count = 0
        deformation_motion_active = (
            self._deformation_motion_count
            >= self.low_coverage_deformation_persistence
        )
        sequence_disagreement_active = (
            self._sequence_disagreement_count >= self.sequence_disagreement_persistence
            and (
                not self.sequence_disagreement_low_coverage_only
                or observed_length_m / max(self.expected_length_m, 1e-9) < 0.50
            )
        )
        if self._points_world is None:
            ratio = observed_length_m / self.expected_length_m
            if ratio < self.min_initial_length_ratio and not self.allow_partial_initialization:
                return TemporalEstimate(
                    points_world=None,
                    observed_mask=np.zeros(self.sample_count, dtype=bool),
                    coverage=0.0,
                    confidence=0.0,
                    initialized=False,
                    used_prediction=False,
                    matched_distance_m=float("nan"),
                    timings_ms=_timing_dict(start_ns, self),
                )
            partial_initialization = ratio < self.min_initial_length_ratio
            if partial_initialization:
                self._points_world = _extend_polyline_to_length(
                    observed, self.expected_length_m, self.sample_count
                )
            else:
                self._points_world = resample_polyline(observed, self.sample_count)
            self._velocity_world = np.zeros_like(self._points_world)
            if partial_initialization:
                visible_span = float(np.clip(ratio, 0.05, 1.0))
                lower = 0.5 * (1.0 - visible_span)
                upper = 0.5 * (1.0 + visible_span)
                self._last_observed_interval = (lower, upper)
                observed_mask = (
                    np.linspace(0.0, 1.0, self.sample_count) >= lower - 1e-6
                ) & (
                    np.linspace(0.0, 1.0, self.sample_count) <= upper + 1e-6
                )
                confidence = float(np.clip(0.5 * ratio, 0.0, 0.5))
            else:
                self._last_observed_interval = (0.0, 1.0)
                observed_mask = np.ones(self.sample_count, dtype=bool)
                confidence = 1.0
            observed_mask = self._apply_robot_occlusion_mask(
                observed_mask,
                self._points_world,
                robot_mask,
                camera_from_world,
                intrinsics,
                image_mask,
            )
            return TemporalEstimate(
                points_world=self._points_world.copy(),
                observed_mask=observed_mask,
                coverage=float(np.mean(observed_mask)),
                confidence=confidence,
                initialized=True,
                used_prediction=False,
                matched_distance_m=0.0,
                timings_ms=_timing_dict(start_ns, self),
            )

        previous = self._points_world
        velocity = (
            np.zeros_like(previous)
            if self._velocity_world is None
            else self._velocity_world
        )
        predicted = previous + velocity * dt
        if sequence_motion is not None:
            sequence_rotation, sequence_translation, sequence_residual, _ = sequence_motion
            sequence_translation_norm = float(np.linalg.norm(sequence_translation))
            sequence_angle = _rotation_angle(sequence_rotation)
            self._last_sequence_residual_m = sequence_residual
            self._last_sequence_translation_m = sequence_translation_norm
            self._last_sequence_angle_rad = sequence_angle
            if (
                sequence_residual < 0.012
                and sequence_translation_norm >= self.sequence_min_translation_m
                and sequence_translation_norm < 0.50
                and sequence_angle < np.deg2rad(45.0)
                # Once a short fragment has been tracked for at least one
                # frame, a low-residual consecutive fit is still a useful
                # global translation cue.  The old 0.35 cutoff disabled this
                # exactly in the 0.30--0.34 coverage regime seen after a
                # gripper occludes half of the cable, leaving hidden nodes
                # frozen while the visible cable moved.
                and observed_length_m / max(self.expected_length_m, 1e-9) >= 0.25
                and not sequence_disagreement_active
            ):
                transported = previous @ sequence_rotation.T + sequence_translation
                predicted = (
                    (1.0 - self.rigid_transform_gain) * predicted
                    + self.rigid_transform_gain * transported
                )
                self._last_sequence_applied = True
        prior_step_m = float(np.median(np.linalg.norm(velocity * dt, axis=1)))
        self._last_prior_step_m = prior_step_m
        params, distance, orientation = _match_arc_parameters(
            predicted,
            observed,
            observed_length_m=observed_length_m,
            expected_length_m=self.expected_length_m,
            preferred_interval=self._last_observed_interval,
            continuity_weight=self.arc_continuity_weight,
            motion_translation_m=centroid_step_m,
            motion_translation_weight=self.centroid_motion_weight,
            motion_translation_slack_m=self.centroid_motion_slack_m,
            motion_rotation=(
                sequence_motion[0]
                if sequence_motion is not None
                and sequence_motion[2] < 0.012
                and _rotation_angle(sequence_motion[0]) < np.deg2rad(45.0)
                and np.linalg.norm(sequence_motion[1]) < 0.50
                and prior_step_m >= self.sequence_motion_min_prior_step_m
                and not sequence_disagreement_active
                else None
            ),
            motion_translation_vector=(
                sequence_motion[1]
                if sequence_motion is not None
                and sequence_motion[2] < 0.012
                and _rotation_angle(sequence_motion[0]) < np.deg2rad(45.0)
                and np.linalg.norm(sequence_motion[1]) < 0.50
                and prior_step_m >= self.sequence_motion_min_prior_step_m
                and not sequence_disagreement_active
                else None
            ),
            motion_transform_weight=self.sequence_motion_weight,
            observed_pixels=measurement_pixels,
            observed_image_mask=image_mask,
            camera_from_world=camera_from_world,
            intrinsics=intrinsics,
            image_match_weight=image_match_weight,
            image_match_min_improvement_m=image_match_min_improvement_m,
            # A short, nearly straight fragment has an intrinsically ambiguous
            # 3-D Kabsch rotation: unrelated arc intervals can be rotated into
            # the same shape.  The sequence/camera motion is already accounted
            # for in ``predicted`` above, so use translation-only matching for
            # low coverage and keep the free rigid fit for longer observations.
            allow_rotation=(
                observed_length_m / max(self.expected_length_m, 1e-9) >= 0.50
            ),
            hysteresis_jump=self.arc_hysteresis_jump,
            hysteresis_margin_m=self.arc_hysteresis_margin_m,
        )
        if orientation:
            observed = observed[::-1].copy()

        # Remove duplicate arc coordinates before interpolation.  A short
        # observation can map several observed samples to the same state node.
        keep = np.concatenate(([True], np.diff(params) > 1e-4))
        params = params[keep]
        observed = observed[keep]
        candidate_coverage = (
            float(np.clip(params[-1] - params[0], 0.0, 1.0))
            if len(params) >= 2
            else 0.0
        )
        candidate_confidence = float(
            np.clip(candidate_coverage * np.exp(-distance / 0.02), 0.0, 1.0)
        )
        if (
            len(params) >= 2
            and self.reacquisition_max_match_confidence >= 0.0
            and observed_length_m / max(self.expected_length_m, 1e-9)
            >= self.reacquisition_min_observed_ratio
            and candidate_coverage >= self.reacquisition_min_coverage
            and candidate_confidence <= self.reacquisition_max_match_confidence
        ):
            # A nearly complete observation with a very weak match to the
            # stored state is more trustworthy than an ambiguous historical
            # arc assignment.  Reorder only by proximity to the predicted
            # endpoints, then use the observation as a full-state measurement.
            direct = resample_polyline(observed, self.sample_count)
            reverse = direct[::-1].copy()
            if float(np.mean(np.linalg.norm(reverse - predicted, axis=1))) < float(
                np.mean(np.linalg.norm(direct - predicted, axis=1))
            ):
                direct = reverse
            observed = direct
            params = np.linspace(0.0, 1.0, self.sample_count)
            distance = float(np.mean(np.linalg.norm(observed - predicted, axis=1)))
        self._last_prior_step_m = prior_step_m
        self._last_rigid_residual_m = float("nan")
        self._last_rigid_translation_m = float("nan")
        self._last_rigid_angle_rad = float("nan")
        self._last_rigid_applied = False
        if len(params) < 2:
            updated = predicted
            observed_mask = np.zeros(self.sample_count, dtype=bool)
            coverage = 0.0
        else:
            state_params = np.linspace(0.0, 1.0, self.sample_count)
            lower, upper = float(params[0]), float(params[-1])
            coverage = float(np.clip(upper - lower, 0.0, 1.0))
            self._last_observed_interval = (lower, upper)
            observed_mask = (
                (state_params >= lower - 1e-6)
                & (state_params <= upper + 1e-6)
            )
            observed_mask = self._apply_robot_occlusion_mask(
                observed_mask,
                predicted,
                robot_mask,
                camera_from_world,
                intrinsics,
                image_mask,
            )
            match_confidence = float(
                np.clip(coverage * np.exp(-distance / 0.02), 0.0, 1.0)
            )
            updated = predicted.copy()
            # If the observed fragment is consistent with a rigid motion,
            # transport the complete previous curve with that transform. This
            # is especially useful for the rigid-l1 scenario: updating only
            # visible nodes would leave the hidden continuation behind.
            rigid_rotation, rigid_translation, rigid_residual = _fit_rigid_transform(
                predicted, observed, params
            )
            self._last_rigid_residual_m = rigid_residual
            self._last_rigid_translation_m = float(np.linalg.norm(rigid_translation))
            self._last_rigid_angle_rad = _rotation_angle(rigid_rotation)
            if (
                self.rigid_transform_gain > 0.0
                and (
                    (
                        prior_step_m > self.rigid_min_prior_step_m
                        # For ordinary frame-to-frame motion require a tight
                        # fit.  Looser residuals are commonly caused by an
                        # ambiguous arc match, especially while the cable is
                        # static.
                        and rigid_residual < min(0.012, self.rigid_residual_threshold_m)
                    )
                    # A long camera-visible rigid jump can occur between
                    # recorded control frames before the velocity estimate has
                    # caught up.  Allow that case only for a large translation
                    # with a good geometric fit; this does not unlock ordinary
                    # static RGB-D jitter.
                    or (
                        self._last_rigid_translation_m > 0.20
                        and rigid_residual < 0.90 * self.rigid_residual_threshold_m
                        and coverage >= 0.30
                    )
                )
                and coverage >= 0.35
                and rigid_residual < self.rigid_residual_threshold_m
                and match_confidence >= self.rigid_min_match_confidence
                and not sequence_disagreement_active
                and (
                    self._last_rigid_translation_m < self.rigid_max_translation_m
                    or (
                        centroid_step_m is not None
                        and centroid_step_m > 0.20
                        and self._last_rigid_translation_m < 1.0
                    )
                )
                and self._last_rigid_angle_rad < self.rigid_max_angle_rad
            ):
                rigid_predicted = predicted @ rigid_rotation.T + rigid_translation
                predicted = (
                    (1.0 - self.rigid_transform_gain) * predicted
                    + self.rigid_transform_gain * rigid_predicted
                )
                updated = predicted.copy()
                self._last_rigid_applied = True
            # The observed part is a measurement, not a prediction.  Motion
            # gating belongs to the hidden completion only; applying it here
            # would make a low-coverage frame ignore an otherwise accurate
            # RGB-D fragment and visibly pull the red state away from the
            # cable.  The default gain is therefore one (exact measurement
            # replacement), while callers can still request smoothing via
            # ``observation_gain``.
            effective_gain = self.observation_gain
            confidence_factor = float(
                np.clip(
                    match_confidence / self.confidence_gain_reference,
                    self.confidence_gain_floor,
                    1.0,
                )
            )
            if not self.confidence_gain_low_coverage_only or coverage < 0.50:
                effective_gain *= (
                    (1.0 - self.confidence_gain_scale)
                    + self.confidence_gain_scale * confidence_factor
                )
            for axis in range(3):
                interpolated = np.interp(
                    state_params, params, observed[:, axis]
                )
                updated[observed_mask, axis] = (
                    (1.0 - effective_gain)
                    * predicted[observed_mask, axis]
                    + effective_gain
                    * interpolated[observed_mask]
                )

            # Keep the unobserved curve smooth without pulling observed nodes
            # away from the RGB-D measurement.
            for _ in range(2):
                for index in range(1, self.sample_count - 1):
                    if not observed_mask[index]:
                        updated[index] = (
                            0.25 * updated[index - 1]
                            + 0.50 * updated[index]
                            + 0.25 * updated[index + 1]
                        )

            # Propagate the measured motion of visible nodes to hidden nodes.
            # This is the low-dimensional analogue of motion coherence: a
            # local fragment may move while its occluded continuation is not
            # directly observed, but neighboring cable material should still
            # have correlated displacement.
            observed_indices = np.flatnonzero(observed_mask)
            state_params = np.linspace(0.0, 1.0, self.sample_count)
            observed_motion = updated[observed_indices] - previous[observed_indices]
            common_motion = np.median(observed_motion, axis=0)
            local_motion = observed_motion - common_motion
            sigma = max(0.10, 0.5 * coverage)
            for index in np.flatnonzero(~observed_mask):
                distances = np.abs(
                    state_params[index] - state_params[observed_indices]
                )
                weights = np.exp(-0.5 * (distances / sigma) ** 2)
                weights /= max(float(weights.sum()), 1e-9)
                coherent_motion = common_motion + np.sum(
                    weights[:, None] * local_motion, axis=0
                )
                coherent_target = previous[index] + coherent_motion
                nearest_distance = float(distances.min())
                blend = self.hidden_motion_gain * np.exp(-nearest_distance / 0.35)
                updated[index] = (
                    (1.0 - blend) * predicted[index]
                    + blend * coherent_target
                )

            # A simulated cable is inextensible.  Detector noise and repeated
            # interpolation can otherwise make the hidden continuation grow
            # without bound.  Keep measured visible nodes fixed and scale the
            # two hidden sides around their nearest visible anchors so the
            # complete polyline stays close to the known cable length.
            updated = _regularize_hidden_length(
                updated,
                observed_mask,
                target_length_m=self.expected_length_m,
                gain=self.length_regularization_gain,
            )

            # If the motion prior is effectively static and less than half of
            # the cable is visible, trust the last complete state.  This is a
            # deliberate anti-drift gate: a tiny fragment often has enough
            # RGB segmentation noise to move the inferred arc interval even
            # though the real cable did not move.
            if coverage < 0.45 and prior_step_m < 0.002:
                if not deformation_motion_active:
                    # Hold only the unobserved continuation.  The visible
                    # nodes must remain at the current RGB-D measurement.
                    updated[~observed_mask] = previous[~observed_mask]
            if (
                deformation_motion_active
                and self.low_coverage_deformation_gain > 0.0
                and self._last_centroid_delta is not None
            ):
                centroid_target = previous + self._last_centroid_delta
                if self.low_coverage_deformation_rigid_gain > 0.0:
                    rigid_target = previous @ rigid_rotation.T + rigid_translation
                    rigid_gain = self.low_coverage_deformation_rigid_gain
                    centroid_target = (
                        (1.0 - rigid_gain) * centroid_target
                        + rigid_gain * rigid_target
                    )
                gain = self.low_coverage_deformation_gain
                hidden_target = (1.0 - gain) * updated + gain * centroid_target
                updated[~observed_mask] = hidden_target[~observed_mask]
            if sequence_disagreement_active:
                # A low-residual, very large centerline motion that disagrees
                # with the measured centroid is usually an arc-identity swap.
                # Do not inject that fragment into the state; retain only the
                # motion prediction for this frame and let the next observation
                # reacquire the cable.
                if (
                    self.sequence_disagreement_use_centroid_translation
                    and self._last_centroid_delta is not None
                ):
                    disagreement_target = previous + self._last_centroid_delta
                else:
                    disagreement_target = (
                        previous.copy()
                        if self.sequence_disagreement_hold_previous
                        else predicted.copy()
                    )
                updated[~observed_mask] = disagreement_target[~observed_mask]

        displacement = (updated - previous) / dt
        if coverage < 0.50:
            # A short visible fragment is insufficient to estimate global
            # velocity.  Damp the old velocity instead of integrating detector
            # noise into an ever-growing drift during static occlusion.
            if (
                sequence_disagreement_active
                and self.sequence_disagreement_velocity_gain > 0.0
                and self._last_centroid_delta is not None
                and centroid_step_m is not None
                and centroid_step_m
                >= self.sequence_disagreement_velocity_min_centroid_step_m
            ):
                centroid_velocity = self._last_centroid_delta / dt
                self._velocity_world = (
                    (1.0 - self.sequence_disagreement_velocity_gain) * velocity
                    + self.sequence_disagreement_velocity_gain * centroid_velocity
                )
            else:
                self._velocity_world = velocity * self.velocity_decay
        else:
            self._velocity_world = (
                (1.0 - self.velocity_gain) * velocity
                + self.velocity_gain * displacement
            )
        self._points_world = updated
        return TemporalEstimate(
            points_world=updated.copy(),
            observed_mask=observed_mask,
            coverage=coverage,
            confidence=float(
                np.clip(coverage * np.exp(-distance / 0.02), 0.0, 1.0)
            ),
            initialized=True,
            used_prediction=bool(np.any(~observed_mask)),
            matched_distance_m=distance,
            timings_ms=_timing_dict(start_ns, self),
        )

    def _apply_robot_occlusion_mask(
        self,
        observed_mask: np.ndarray,
        points_world: np.ndarray,
        robot_mask: np.ndarray | None,
        camera_from_world: np.ndarray | None,
        intrinsics: np.ndarray | None,
        observed_image_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Remove state nodes whose predicted projections are robot-occluded.

        A cable observation is represented as an arc interval.  If a gripper
        hides the middle of that interval, treating every node between the two
        visible ends as measured creates a false interpolation across the
        gripper.  A projected robot mask turns those nodes back into hidden
        state, so the temporal predictor fills them instead.  If the mask would
        remove the entire interval, retain the original observation as a safe
        fallback for imperfect calibration or a stale robot pose.
        """

        mask = np.asarray(observed_mask, dtype=bool).copy()
        self._last_robot_occlusion_fraction = float("nan")
        if (
            robot_mask is None
            or camera_from_world is None
            or intrinsics is None
            or points_world is None
            or len(points_world) != len(mask)
        ):
            return mask
        projected = project_points(
            transform_points(camera_from_world, np.asarray(points_world)),
            np.asarray(intrinsics),
        )
        expanded = cv2.dilate(
            robot_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)
        ).astype(bool)
        cable_expanded = None
        if observed_image_mask is not None:
            cable_expanded = cv2.dilate(
                observed_image_mask.astype(np.uint8),
                np.ones((7, 7), dtype=np.uint8),
            ).astype(bool)
        height, width = expanded.shape
        columns = np.rint(projected[:, 0]).astype(np.int64)
        rows = np.rint(projected[:, 1]).astype(np.int64)
        valid = (
            np.isfinite(projected).all(axis=1)
            & (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        occluded = np.zeros(len(mask), dtype=bool)
        occluded[valid] = expanded[rows[valid], columns[valid]]
        if cable_expanded is not None:
            cable_visible = np.zeros(len(mask), dtype=bool)
            cable_visible[valid] = cable_expanded[rows[valid], columns[valid]]
            # A robot pixel next to a visible cable pixel is not evidence that
            # this state node is hidden; this guard removes the dilation false
            # positives at gripper/cable boundaries.
            occluded &= ~cable_visible
        occluded_observed = mask & occluded
        self._last_robot_occlusion_fraction = float(
            np.mean(occluded_observed) if np.any(mask) else 0.0
        )
        visible_after = mask & ~occluded
        if np.count_nonzero(visible_after) >= 2:
            return visible_after
        return mask


    def update_fragments(
        self,
        observed_fragments: list[np.ndarray],
        *,
        observed_lengths_m: list[float] | None = None,
        dt_s: float = 0.04,
        robot_occlusion_mask: np.ndarray | None = None,
        camera_from_world: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        enforce_nonoverlap: bool = True,
        observed_pixels_list: list[np.ndarray] | None = None,
        observed_image_masks: list[np.ndarray] | None = None,
        camera_from_worlds: list[np.ndarray] | None = None,
        intrinsics_list: list[np.ndarray] | None = None,
        image_match_weight: float = 0.0,
        image_match_min_improvement_m: float = 0.02,
    ) -> TemporalEstimate:
        """Fuse several disconnected visible fragments into one full state."""

        start_ns = perf_counter_ns()
        fragments = [
            np.asarray(fragment, dtype=np.float64)
            for fragment in observed_fragments
            if np.asarray(fragment).ndim == 2 and len(fragment) >= 2
        ]
        if not fragments:
            raise ValueError("observed_fragments must contain a valid fragment")
        if any(fragment.shape[1] != 3 for fragment in fragments):
            raise ValueError("each observed fragment must have shape (N, 3)")
        if any(not np.isfinite(fragment).all() for fragment in fragments):
            raise ValueError("observed_fragments contains non-finite values")
        metadata_lists = (
            observed_pixels_list,
            observed_image_masks,
            camera_from_worlds,
            intrinsics_list,
        )
        if any(values is not None and len(values) != len(fragments) for values in metadata_lists):
            raise ValueError("fragment image metadata must match observed_fragments")
        lengths = (
            [float(np.linalg.norm(np.diff(fragment, axis=0), axis=1).sum()) for fragment in fragments]
            if observed_lengths_m is None
            else [float(value) for value in observed_lengths_m]
        )
        if len(lengths) != len(fragments):
            raise ValueError("observed_lengths_m must match observed_fragments")
        observed_centroid = np.mean(
            np.concatenate(fragments, axis=0), axis=0
        )
        centroid_step_m = (
            None
            if self._last_observed_centroid is None
            else float(np.linalg.norm(observed_centroid - self._last_observed_centroid))
        )
        self._last_observed_centroid = observed_centroid.copy()
        self._last_centroid_step_m = (
            float("nan") if centroid_step_m is None else centroid_step_m
        )
        if self._points_world is None:
            longest = int(np.argmax(lengths))
            result = self.update(
                fragments[longest],
                observed_length_m=lengths[longest],
                dt_s=dt_s,
                robot_occlusion_mask=robot_occlusion_mask,
                camera_from_world=camera_from_world,
                intrinsics=intrinsics,
            )
            result.timings_ms["fragment_count"] = float(len(fragments))
            return result

        previous = self._points_world
        velocity = np.zeros_like(previous) if self._velocity_world is None else self._velocity_world
        dt = float(np.clip(dt_s, 1e-3, 0.25))
        predicted = previous + velocity * dt
        prior_step_m = float(np.median(np.linalg.norm(velocity * dt, axis=1)))
        self._last_prior_step_m = prior_step_m
        self._last_rigid_residual_m = float("nan")
        self._last_rigid_translation_m = float("nan")
        self._last_rigid_angle_rad = float("nan")
        self._last_rigid_applied = False
        state_params = np.linspace(0.0, 1.0, self.sample_count)
        robot_visible_nodes = self._apply_robot_occlusion_mask(
            np.ones(self.sample_count, dtype=bool),
            predicted,
            robot_occlusion_mask,
            camera_from_world,
            intrinsics,
            None,
        )
        measurements = []
        best_quality = -1.0
        best_interval = self._last_observed_interval
        best_rigid = None
        assigned_intervals: list[tuple[float, float]] = []
        for fragment_index, (fragment, length) in enumerate(zip(fragments, lengths)):
            # The first/largest component is the continuity anchor.  Additional
            # disconnected pieces may lie on a different arc interval; forcing
            # every piece to the previous interval makes the fusion collapse all
            # fragments onto the same cable section.  Match secondary fragments
            # globally, then let their geometric residual decide their weight.
            preferred_interval = (
                self._last_observed_interval if fragment_index == 0 else None
            )
            params, distance, orientation = _match_arc_parameters(
                predicted,
                fragment,
                observed_length_m=length,
                expected_length_m=self.expected_length_m,
                preferred_interval=preferred_interval,
                continuity_weight=self.arc_continuity_weight,
                motion_translation_m=centroid_step_m,
                motion_translation_weight=self.centroid_motion_weight,
                motion_translation_slack_m=self.centroid_motion_slack_m,
                excluded_intervals=(
                    assigned_intervals if enforce_nonoverlap and fragment_index else None
                ),
                observed_pixels=(
                    None
                    if observed_pixels_list is None
                    else observed_pixels_list[fragment_index]
                ),
                observed_image_mask=(
                    None
                    if observed_image_masks is None
                    else observed_image_masks[fragment_index]
                ),
                camera_from_world=(
                    None
                    if camera_from_worlds is None
                    else camera_from_worlds[fragment_index]
                ),
                intrinsics=(
                    None
                    if intrinsics_list is None
                    else intrinsics_list[fragment_index]
                ),
                image_match_weight=image_match_weight,
                image_match_min_improvement_m=image_match_min_improvement_m,
            )
            if not np.isfinite(distance) and fragment_index:
                # If the mask produced duplicate/split pieces from the same
                # physical arc, the non-overlap prior may reject every option.
                # Fall back to the unconstrained match rather than discarding
                # the entire frame.
                params, distance, orientation = _match_arc_parameters(
                    predicted,
                    fragment,
                    observed_length_m=length,
                    expected_length_m=self.expected_length_m,
                    preferred_interval=preferred_interval,
                    continuity_weight=self.arc_continuity_weight,
                    motion_translation_m=centroid_step_m,
                    motion_translation_weight=self.centroid_motion_weight,
                    motion_translation_slack_m=self.centroid_motion_slack_m,
                    observed_pixels=(
                        None
                        if observed_pixels_list is None
                        else observed_pixels_list[fragment_index]
                    ),
                    observed_image_mask=(
                        None
                        if observed_image_masks is None
                        else observed_image_masks[fragment_index]
                    ),
                    camera_from_world=(
                        None
                        if camera_from_worlds is None
                        else camera_from_worlds[fragment_index]
                    ),
                    intrinsics=(
                        None
                        if intrinsics_list is None
                        else intrinsics_list[fragment_index]
                    ),
                    image_match_weight=image_match_weight,
                    image_match_min_improvement_m=image_match_min_improvement_m,
                )
            if orientation:
                fragment = fragment[::-1].copy()
            keep = np.concatenate(([True], np.diff(params) > 1e-4))
            params = params[keep]
            fragment = fragment[keep]
            if len(params) < 2:
                continue
            lower, upper = float(params[0]), float(params[-1])
            coverage = float(np.clip(upper - lower, 0.0, 1.0))
            assigned_intervals.append((lower, upper))
            mask = (state_params >= lower - 1e-6) & (state_params <= upper + 1e-6)
            mask &= robot_visible_nodes
            interpolated = np.column_stack(
                [np.interp(state_params, params, fragment[:, axis]) for axis in range(3)]
            )
            quality = float(
                np.clip(np.exp(-distance / 0.02) * min(1.0, coverage / 0.15), 0.0, 1.0)
            )
            measurements.append((mask, interpolated, quality, distance, coverage))
            if quality > best_quality:
                best_quality = quality
                # Keep the temporal arc prior anchored to the largest/first
                # fragment.  A short secondary fragment is useful for filling
                # an occluded interval, but should not make the next frame jump
                # to a visually similar section of the cable.
                if fragment_index == 0 or best_interval is None:
                    best_interval = (lower, upper)
                best_rigid = _fit_rigid_transform(predicted, fragment, params)

        if not measurements:
            updated = predicted.copy()
            observed_mask = np.zeros(self.sample_count, dtype=bool)
            coverage = 0.0
            distance = float("nan")
        else:
            self._last_observed_interval = best_interval
            distance = float(min(item[3] for item in measurements))
            observed_mask = np.zeros(self.sample_count, dtype=bool)
            weighted_sum = np.zeros_like(predicted)
            weights = np.zeros(self.sample_count, dtype=np.float64)
            for mask, interpolated, quality, _, _ in measurements:
                observed_mask |= mask
                weighted_sum[mask] += quality * interpolated[mask]
                weights[mask] += quality
            # Coverage is the union of matched state nodes, not the mean length
            # of disconnected intervals.  Using the mean underestimates how
            # much of the cable is actually observed and unnecessarily damps
            # the correction/velocity gains when two fragments are available.
            coverage = float(np.mean(observed_mask))
            match_confidence = float(
                np.clip(coverage * np.exp(-distance / 0.02), 0.0, 1.0)
            )
            if best_rigid is not None:
                rotation, translation, residual = best_rigid
                self._last_rigid_residual_m = residual
                self._last_rigid_translation_m = float(np.linalg.norm(translation))
                self._last_rigid_angle_rad = _rotation_angle(rotation)
                if (
                    self.rigid_transform_gain > 0.0
                    and (
                        (
                            prior_step_m > self.rigid_min_prior_step_m
                            and residual < min(0.012, self.rigid_residual_threshold_m)
                        )
                        or (
                            self._last_rigid_translation_m > 0.20
                            and residual < 0.90 * self.rigid_residual_threshold_m
                            and coverage >= 0.30
                        )
                    )
                    and coverage >= 0.35
                    and residual < self.rigid_residual_threshold_m
                    and match_confidence >= self.rigid_min_match_confidence
                    and (
                        self._last_rigid_translation_m < self.rigid_max_translation_m
                        or (
                            centroid_step_m is not None
                            and centroid_step_m > 0.20
                            and self._last_rigid_translation_m < 1.0
                        )
                    )
                    and self._last_rigid_angle_rad < self.rigid_max_angle_rad
                ):
                    rigid_predicted = predicted @ rotation.T + translation
                    predicted = (
                        (1.0 - self.rigid_transform_gain) * predicted
                        + self.rigid_transform_gain * rigid_predicted
                    )
                    self._last_rigid_applied = True
            updated = predicted.copy()
            # Every matched fragment is a current RGB-D measurement.  The
            # low-coverage motion scale is reserved for hidden completion;
            # applying it to the visible nodes makes a static scene ignore
            # accurate measurements merely because the mask is split.
            effective_gain = self.observation_gain
            valid_weights = weights > 1e-9
            fused = weighted_sum / np.maximum(weights[:, None], 1e-9)
            # Measurement quality is used primarily to choose between competing
            # fragments.  Do not multiply the correction by the raw residual
            # weight: a perfectly valid fragment with a 2--3 cm RGB-D residual
            # would otherwise be almost ignored and the state would drift.  A
            # moderate floor keeps a single valid fragment equivalent to the
            # original update() path, while still down-weighting weak fragments
            # when several components overlap the same node.
            max_weight = max(float(weights.max()), 1e-9)
            relative_weights = weights / max_weight
            node_gain = effective_gain * np.clip(relative_weights, 0.50, 1.0)
            updated[valid_weights] = (
                (1.0 - node_gain[valid_weights, None]) * predicted[valid_weights]
                + node_gain[valid_weights, None] * fused[valid_weights]
            )
            for _ in range(2):
                for index in range(1, self.sample_count - 1):
                    if not observed_mask[index]:
                        updated[index] = (
                            0.25 * updated[index - 1]
                            + 0.50 * updated[index]
                            + 0.25 * updated[index + 1]
                        )
            observed_indices = np.flatnonzero(observed_mask)
            if len(observed_indices):
                observed_motion = updated[observed_indices] - previous[observed_indices]
                common_motion = np.median(observed_motion, axis=0)
                local_motion = observed_motion - common_motion
                sigma = max(0.10, 0.5 * coverage)
                for index in np.flatnonzero(~observed_mask):
                    distances = np.abs(state_params[index] - state_params[observed_indices])
                    local_weights = np.exp(-0.5 * (distances / sigma) ** 2)
                    local_weights /= max(float(local_weights.sum()), 1e-9)
                    coherent_motion = common_motion + np.sum(
                        local_weights[:, None] * local_motion, axis=0
                    )
                    coherent_target = previous[index] + coherent_motion
                    blend = self.hidden_motion_gain * np.exp(
                        -float(distances.min()) / 0.35
                    )
                    updated[index] = (
                        (1.0 - blend) * predicted[index] + blend * coherent_target
                    )
            updated = _regularize_hidden_length(
                updated,
                observed_mask,
                target_length_m=self.expected_length_m,
                gain=self.length_regularization_gain,
            )
            if coverage < 0.45 and prior_step_m < 0.002:
                # Hold only the hidden continuation.  Keeping the visible
                # fragment at its current measurement prevents the fragment
                # fusion path from reintroducing the visible-point regression
                # that motivated the hard-observation update in update().
                updated[~observed_mask] = previous[~observed_mask]

        displacement = (updated - previous) / dt
        if coverage < 0.50:
            self._velocity_world = velocity * self.velocity_decay
        else:
            self._velocity_world = (
                (1.0 - self.velocity_gain) * velocity
                + self.velocity_gain * displacement
            )
        self._points_world = updated
        return TemporalEstimate(
            points_world=updated.copy(),
            observed_mask=observed_mask,
            coverage=coverage,
            confidence=float(np.clip(coverage * np.exp(-distance / 0.02), 0.0, 1.0))
            if np.isfinite(distance)
            else 0.0,
            initialized=True,
            used_prediction=bool(np.any(~observed_mask)),
            matched_distance_m=distance,
            timings_ms={
                "total": _elapsed_ms(start_ns),
                "rigid_residual": self._last_rigid_residual_m,
                "rigid_translation": self._last_rigid_translation_m,
                "rigid_angle": self._last_rigid_angle_rad,
                "rigid_applied": float(self._last_rigid_applied),
                "prior_step": self._last_prior_step_m,
                "centroid_step": self._last_centroid_step_m,
                "sequence_residual": self._last_sequence_residual_m,
                "sequence_translation": self._last_sequence_translation_m,
                "sequence_angle": self._last_sequence_angle_rad,
                "sequence_applied": float(self._last_sequence_applied),
                "fragment_count": float(len(fragments)),
            },
        )


def _match_arc_parameters(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    observed_length_m: float,
    expected_length_m: float,
    preferred_interval: tuple[float, float] | None = None,
    continuity_weight: float = 1.0,
    motion_translation_m: float | None = None,
    motion_translation_weight: float = 0.50,
    motion_translation_slack_m: float = 0.15,
    motion_rotation: np.ndarray | None = None,
    motion_translation_vector: np.ndarray | None = None,
    motion_transform_weight: float = 0.0,
    excluded_intervals: list[tuple[float, float]] | None = None,
    observed_pixels: np.ndarray | None = None,
    observed_image_mask: np.ndarray | None = None,
    camera_from_world: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    image_match_weight: float = 0.0,
    image_match_min_improvement_m: float = 0.02,
    allow_rotation: bool = True,
    hysteresis_jump: float = 0.0,
    hysteresis_margin_m: float = 0.0,
) -> tuple[np.ndarray, float, bool]:
    """Match an ordered partial curve to a translated reference interval.

    Nearest-neighbour matching collapses when the whole DLO moves between
    frames.  Instead, use the measured length ratio to enumerate contiguous
    intervals of the previous curve, estimate a translation for each interval,
    and choose the lowest residual.  This preserves the arc-length semantics
    needed to update a full state while remaining small enough for 50 Hz.
    """

    dense = _densify(reference, max(240, 16 * len(reference)))
    if observed_pixels is not None:
        observed_pixels = np.asarray(observed_pixels, dtype=np.float64)
        if observed_pixels.shape != (len(observed), 2):
            raise ValueError("observed_pixels shape must match observed points")
        if camera_from_world is None or intrinsics is None:
            raise ValueError(
                "camera_from_world and intrinsics are required with observed_pixels"
            )
    if observed_image_mask is not None:
        observed_image_mask = np.asarray(observed_image_mask).astype(bool, copy=False)
        if observed_image_mask.ndim != 2 or observed_image_mask.size == 0:
            raise ValueError("observed_image_mask must be a non-empty HxW array")
    observed_ratio = float(
        np.clip(observed_length_m / max(expected_length_m, 1e-9), 0.05, 1.0)
    )
    window = max(2, int(round(observed_ratio * (len(dense) - 1))) + 1)
    max_start = max(0, len(dense) - window)
    starts = np.unique(np.linspace(0, max_start, min(96, max_start + 1)).astype(int))
    sample_positions = np.linspace(0.0, 1.0, len(observed))
    index_grid = np.rint(
        starts[:, None] + sample_positions[None, :] * (window - 1)
    ).astype(int)
    candidates = dense[index_grid]
    best_cost = float("inf")
    best_start = int(starts[0])
    best_reverse = False
    candidate_records: list[tuple[float, int, bool]] = []
    for use_reverse in (False, True):
        candidate_observed = observed[::-1] if use_reverse else observed
        candidate_observed_pixels = (
            None
            if observed_pixels is None
            else (observed_pixels[::-1] if use_reverse else observed_pixels)
        )
        candidate_centered = candidates - candidates.mean(axis=1, keepdims=True)
        observed_centered = candidate_observed - candidate_observed.mean(
            axis=0, keepdims=True
        )
        covariance = np.einsum(
            "smi,mj->sij", candidate_centered, observed_centered
        )
        u, singular_values, vh = np.linalg.svd(covariance)
        v = np.swapaxes(vh, 1, 2)
        rotation = np.matmul(v, np.swapaxes(u, 1, 2))
        determinant = np.linalg.det(rotation)
        v[:, :, -1] *= np.where(determinant < 0.0, -1.0, 1.0)[:, None]
        rotation = np.matmul(v, np.swapaxes(u, 1, 2))
        if not allow_rotation:
            # Preserve the ordered 3-D shape and use only the fitted
            # translation.  This is the key ambiguity guard for short visible
            # fragments; allowing a free rotation can swap distant, locally
            # straight arc intervals with nearly zero residual.
            rotation = np.broadcast_to(
                np.eye(3, dtype=np.float64), rotation.shape
            ).copy()
        # A straight (or nearly straight) fragment has an ambiguous 3-D
        # Kabsch rotation: a 180-degree flip produces exactly the same
        # residual and can silently reverse the arc assignment.  In this
        # degenerate case use translation-only matching, where the measured
        # point order disambiguates the two orientations.
        rank_ratio = singular_values[:, 1] / np.maximum(singular_values[:, 0], 1e-12)
        # Keep the threshold tight: real cable fragments can be locally
        # almost straight while still carrying enough 3-D curvature for a
        # rigid alignment.  Only exact numerical rank-one cases need the
        # translation-only fallback.
        degenerate = rank_ratio < 1e-6
        if np.any(degenerate):
            rotation[degenerate] = np.eye(3)
        translations = candidate_observed.mean(axis=0) - np.einsum(
            "si,sji->sj", candidates.mean(axis=1), rotation
        )
        aligned = np.einsum("smi,sji->smj", candidates, rotation)
        residual = aligned + translations[:, None, :] - candidate_observed[None, :, :]
        costs = np.mean(np.linalg.norm(residual, axis=2), axis=1)
        base_costs = costs.copy()
        if (
            candidate_observed_pixels is not None
            and camera_from_world is not None
            and intrinsics is not None
            and image_match_weight > 0.0
            and observed_ratio < 0.50
        ):
            # The predicted state is already expressed in the current world
            # frame (sequence/camera motion is handled before matching).  Its
            # projected interval therefore gives an identity cue that a free
            # 3-D Kabsch fit cannot provide for a short, nearly straight
            # fragment.  Convert pixel residual to metres at the observed
            # median depth so it is commensurate with the 3-D residual.
            candidate_camera = transform_points(
                camera_from_world, candidates.reshape(-1, 3)
            ).reshape(candidates.shape)
            candidate_pixels = project_points(
                candidate_camera.reshape(-1, 3), intrinsics
            ).reshape(candidate_camera.shape[0], candidate_camera.shape[1], 2)
            # Compare the *predicted* projected interval with the measured
            # pixels before fitting a candidate-specific 3-D transform.  The
            # latter would make every locally straight interval look equally
            # good because its fitted translation is explicitly chosen to
            # overlap the measurement.  The predicted image location retains
            # the arc-identity cue needed at a full-to-partial transition.
            pixel_residual = np.linalg.norm(
                candidate_pixels - candidate_observed_pixels[None, :, :], axis=2
            )
            focal = max(
                0.5 * (float(intrinsics[0, 0]) + float(intrinsics[1, 1])),
                1.0,
            )
            observed_camera = transform_points(
                camera_from_world, candidate_observed
            )
            depth_scale = float(
                np.median(np.maximum(observed_camera[:, 2], 1e-3))
            )
            image_residual_m = np.mean(pixel_residual, axis=1) * depth_scale / focal
            if observed_image_mask is not None:
                # A segmentation mask is a stronger identity cue than a
                # candidate-specific 3-D fit: the correct predicted arc
                # should project onto the currently visible cable pixels,
                # while a geometrically similar but wrong arc generally falls
                # on the background.  Dilate by a few pixels to tolerate the
                # cable radius and RGB-D centreline bias.
                kernel = np.ones((9, 9), dtype=np.uint8)
                mask_u8 = observed_image_mask.astype(np.uint8)
                dilated = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)
                distance_px = cv2.distanceTransform(
                    (~dilated).astype(np.uint8), cv2.DIST_L2, 3
                )
                height, width = dilated.shape
                projected_x = np.where(
                    np.isfinite(candidate_pixels[..., 0]),
                    np.rint(candidate_pixels[..., 0]),
                    -1.0,
                ).astype(np.int64)
                projected_y = np.where(
                    np.isfinite(candidate_pixels[..., 1]),
                    np.rint(candidate_pixels[..., 1]),
                    -1.0,
                ).astype(np.int64)
                valid_projection = (
                    np.isfinite(candidate_pixels).all(axis=2)
                    & (projected_x >= 0)
                    & (projected_x < width)
                    & (projected_y >= 0)
                    & (projected_y < height)
                )
                mask_distance_px = np.full(
                    projected_x.shape, max(width, height), dtype=np.float64
                )
                mask_distance_px[valid_projection] = distance_px[
                    projected_y[valid_projection], projected_x[valid_projection]
                ]
                mask_residual_m = (
                    np.mean(mask_distance_px, axis=1) * depth_scale / focal
                )
                # Blend the mask distance with direct pixel agreement.  The
                # direct term retains sub-pixel precision; the mask term is
                # what rejects a wrong arc that happens to have similar 3-D
                # shape.
                image_residual_m = 0.25 * image_residual_m + 0.75 * mask_residual_m
            # Only invoke the cue when the currently preferred interval is
            # visibly inconsistent with the image.  This avoids allowing a
            # deformed predicted curve to jump to an image-similar but wrong
            # interval while still recovering a lost identity after an
            # occlusion transition.
            if preferred_interval is not None:
                scale = max(len(dense) - 1, 1)
                candidate_lower = starts / scale
                candidate_upper = (starts + window - 1) / scale
                interval_distance = np.abs(
                    candidate_lower - preferred_interval[0]
                ) + np.abs(candidate_upper - preferred_interval[1])
                preferred_index = int(np.argmin(interval_distance))
                image_improvement = float(
                    image_residual_m[preferred_index] - np.min(image_residual_m)
                )
            else:
                image_improvement = float(np.max(image_residual_m))
            if image_improvement >= max(float(image_match_min_improvement_m), 0.0):
                # Do not let the image cue select an interval whose 3-D shape
                # fit is substantially worse.  This keeps the recovery cue
                # from destabilising genuinely deforming combined motions.
                best_3d = float(np.min(base_costs))
                allowable = base_costs <= best_3d + max(0.08, 1.50 * best_3d)
                costs = base_costs + float(image_match_weight) * image_residual_m
                costs[~allowable] = float("inf")
        if motion_translation_m is not None and np.isfinite(motion_translation_m):
            # The raw fragment centroid provides a weak, order-independent
            # motion cue.  Use it only to suppress implausibly large candidate
            # translations; short occlusion changes can still alter the
            # centroid, so a 15 cm slack is deliberately retained.
            translation_norm = np.linalg.norm(translations, axis=1)
            excess = np.maximum(
                translation_norm
                - (motion_translation_m + max(motion_translation_slack_m, 0.0)),
                0.0,
            )
            costs = costs + max(motion_translation_weight, 0.0) * excess
        if (
            motion_rotation is not None
            and motion_translation_vector is not None
            and motion_transform_weight > 0.0
        ):
            # Consecutive visible centerlines provide an independent estimate of
            # the frame-to-frame rigid motion.  Penalize candidate arc matches
            # whose fitted transform disagrees with that cue; this resolves
            # locally straight/similar arc ambiguities without forcing a match
            # when the sequence fit is unavailable.
            motion_rotation = np.asarray(motion_rotation, dtype=np.float64)
            motion_translation_vector = np.asarray(
                motion_translation_vector, dtype=np.float64
            )
            relative = np.matmul(rotation, motion_rotation.T)
            traces = np.trace(relative, axis1=1, axis2=2)
            rotation_error = np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0))
            translation_error = np.linalg.norm(
                translations - motion_translation_vector[None, :], axis=1
            )
            costs = costs + float(motion_transform_weight) * (
                translation_error + 0.02 * rotation_error
            )
        if excluded_intervals:
            # Treat a second visible component as a different arc interval
            # unless it is only a small boundary overlap.  Independent
            # fragment matching otherwise maps several pieces onto the same
            # visually similar section and the union-of-masks looks better
            # than the actual 3-D identity assignment.
            scale = max(len(dense) - 1, 1)
            candidate_lower = starts / scale
            candidate_upper = (starts + window - 1) / scale
            candidate_span = np.maximum(candidate_upper - candidate_lower, 1e-6)
            conflict = np.zeros(len(starts), dtype=bool)
            for excluded_lower, excluded_upper in excluded_intervals:
                overlap = np.maximum(
                    np.minimum(candidate_upper, float(excluded_upper))
                    - np.maximum(candidate_lower, float(excluded_lower)),
                    0.0,
                )
                overlap_ratio = overlap / np.minimum(
                    candidate_span,
                    max(float(excluded_upper) - float(excluded_lower), 1e-6),
                )
                conflict |= overlap_ratio > 0.65
            costs[conflict] = float("inf")
        if preferred_interval is not None and preferred_interval[1] - preferred_interval[0] < 0.85:
            candidate_lower = starts / max(len(dense) - 1, 1)
            candidate_upper = (starts + window - 1) / max(len(dense) - 1, 1)
            interval_penalty = np.abs(candidate_lower - preferred_interval[0])
            interval_penalty += np.abs(candidate_upper - preferred_interval[1])
            # During strong occlusion the fragment geometry is ambiguous;
            # preserving the last arc interval is safer than jumping to a
            # visually similar but wrong section.  Relax this prior again
            # when most of the cable is visible so genuine material motion
            # can move the interval.
            prior_weight = float(continuity_weight) * (
                1.00 if observed_ratio < 0.60 else 0.05
            )
            costs = costs + prior_weight * interval_penalty
        index = int(np.argmin(costs))
        candidate_records.extend(
            (float(cost), int(start), bool(use_reverse))
            for cost, start in zip(costs, starts)
        )
        if float(costs[index]) < best_cost:
            best_cost = float(costs[index])
            best_start = int(starts[index])
            best_reverse = use_reverse
    if (
        preferred_interval is not None
        and hysteresis_jump > 0.0
        and hysteresis_margin_m > 0.0
        and candidate_records
    ):
        scale = max(len(dense) - 1, 1)
        best_lower = best_start / scale
        best_upper = (best_start + window - 1) / scale
        best_interval_distance = abs(best_lower - preferred_interval[0]) + abs(
            best_upper - preferred_interval[1]
        )
        if best_interval_distance > hysteresis_jump:
            near_cost, near_start, near_reverse = min(
                candidate_records,
                key=lambda item: abs(item[1] / scale - preferred_interval[0])
                + abs((item[1] + window - 1) / scale - preferred_interval[1]),
            )
            if near_cost <= best_cost + hysteresis_margin_m:
                best_start = near_start
                best_reverse = near_reverse
                best_cost = near_cost
    best_params = (
        best_start + sample_positions * (window - 1)
    ) / max(len(dense) - 1, 1)
    return best_params, best_cost, best_reverse


def _densify(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points.copy()
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    if cumulative[-1] <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    queries = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(queries, cumulative, points[:, axis]) for axis in range(3)]
    )


def _extend_polyline_to_length(
    points: np.ndarray, target_length_m: float, sample_count: int
) -> np.ndarray:
    """Symmetrically extend a short observed fragment along endpoint tangents."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("at least two points are required for extension")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    observed_length = float(segment_lengths.sum())
    if observed_length <= 1e-9:
        return np.repeat(points[:1], sample_count, axis=0)
    missing_each_side = max(float(target_length_m) - observed_length, 0.0) * 0.5
    first_tangent = points[1] - points[0]
    last_tangent = points[-1] - points[-2]
    first_tangent /= max(float(np.linalg.norm(first_tangent)), 1e-9)
    last_tangent /= max(float(np.linalg.norm(last_tangent)), 1e-9)
    extended = np.vstack(
        [
            points[0] - missing_each_side * first_tangent,
            points,
            points[-1] + missing_each_side * last_tangent,
        ]
    )
    return resample_polyline(extended, sample_count)


def _fit_rigid_transform(
    reference: np.ndarray,
    observed: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a row-vector rigid transform from a reference arc to observations."""

    reference = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    state_params = np.linspace(0.0, 1.0, len(reference))
    matched = np.column_stack(
        [np.interp(params, state_params, reference[:, axis]) for axis in range(3)]
    )
    reference_mean = matched.mean(axis=0)
    observed_mean = observed.mean(axis=0)
    covariance = (matched - reference_mean).T @ (observed - observed_mean)
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1] *= -1.0
        rotation = vh.T @ u.T
    translation = observed_mean - reference_mean @ rotation.T
    aligned = matched @ rotation.T + translation
    residual = float(np.mean(np.linalg.norm(aligned - observed, axis=1)))
    return rotation, translation, residual


def _fit_sequence_transform(
    previous_observed: np.ndarray,
    observed: np.ndarray,
    *,
    allow_rotation: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Fit a rigid transform between consecutive visible centerline samples.

    Unlike the state-to-fragment fit, this correspondence is independent of
    the unknown global arc interval.  It is therefore useful as a motion cue
    when the same visible cable section persists through several frames.
    """

    reference = np.asarray(previous_observed, dtype=np.float64)
    current = np.asarray(observed, dtype=np.float64)
    if reference.shape != current.shape or len(reference) < 3:
        return np.eye(3), np.zeros(3), float("inf"), False

    def fit(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        reference_mean = reference.mean(axis=0)
        candidate_mean = candidate.mean(axis=0)
        covariance = (reference - reference_mean).T @ (candidate - candidate_mean)
        u, _, vh = np.linalg.svd(covariance)
        rotation = vh.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vh[-1] *= -1.0
            rotation = vh.T @ u.T
        if not allow_rotation:
            rotation = np.eye(3, dtype=np.float64)
        translation = candidate_mean - reference_mean @ rotation.T
        aligned = reference @ rotation.T + translation
        residual = float(np.mean(np.linalg.norm(aligned - candidate, axis=1)))
        return rotation, translation, residual

    forward = fit(current)
    reverse = fit(current[::-1])
    if reverse[2] < forward[2]:
        return reverse[0], reverse[1], reverse[2], True
    return forward[0], forward[1], forward[2], False


def _rotation_angle(rotation: np.ndarray) -> float:
    trace = float(np.trace(np.asarray(rotation, dtype=np.float64)))
    return float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def _timing_dict(start_ns: int, tracker: TemporalDLOTracker) -> dict[str, float]:
    return {
        "total": _elapsed_ms(start_ns),
        "rigid_residual": tracker._last_rigid_residual_m,
        "rigid_translation": tracker._last_rigid_translation_m,
        "rigid_angle": tracker._last_rigid_angle_rad,
        "rigid_applied": float(tracker._last_rigid_applied),
        "prior_step": tracker._last_prior_step_m,
        "centroid_step": tracker._last_centroid_step_m,
        "sequence_residual": tracker._last_sequence_residual_m,
        "sequence_translation": tracker._last_sequence_translation_m,
        "sequence_angle": tracker._last_sequence_angle_rad,
        "sequence_applied": float(tracker._last_sequence_applied),
        "robot_occlusion_fraction": tracker._last_robot_occlusion_fraction,
    }


def _regularize_hidden_length(
    points: np.ndarray,
    observed_mask: np.ndarray,
    *,
    target_length_m: float,
    gain: float,
) -> np.ndarray:
    """Scale only hidden sides so an inextensible cable keeps its length.

    The visible interval is supplied by RGB-D and should not be moved by a
    global scale.  Hidden points are therefore scaled around the first and
    last visible state nodes.  This preserves the observed geometry while
    preventing the temporal predictor from accumulating an unrealistically
    long continuation during sustained occlusion.
    """

    points = np.asarray(points, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    if len(points) < 2 or mask.shape != (len(points),) or not np.any(~mask):
        return points
    visible_indices = np.flatnonzero(mask)
    first_visible = int(visible_indices[0])
    last_visible = int(visible_indices[-1])
    visible_length = float(
        np.linalg.norm(
            np.diff(points[first_visible : last_visible + 1], axis=0), axis=1
        ).sum()
    )
    left_length = float(
        np.linalg.norm(np.diff(points[: first_visible + 1], axis=0), axis=1).sum()
    )
    right_length = float(
        np.linalg.norm(
            np.diff(points[last_visible:], axis=0), axis=1
        ).sum()
    )
    hidden_length = left_length + right_length
    if hidden_length <= 1e-9:
        return points
    desired_hidden = max(float(target_length_m) - visible_length, 1e-6)
    target_scale = desired_hidden / hidden_length
    # Avoid a single bad depth frame producing a very large extrapolation.
    target_scale = float(np.clip(target_scale, 0.25, 4.0))
    scale = 1.0 + float(np.clip(gain, 0.0, 1.0)) * (target_scale - 1.0)
    regularized = points.copy()
    if first_visible > 0:
        anchor = regularized[first_visible]
        regularized[:first_visible] = anchor + scale * (
            regularized[:first_visible] - anchor
        )
    if last_visible + 1 < len(points):
        anchor = regularized[last_visible]
        regularized[last_visible + 1 :] = anchor + scale * (
            regularized[last_visible + 1 :] - anchor
        )
    return regularized


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) * 1e-6
