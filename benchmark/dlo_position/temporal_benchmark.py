from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns

import cv2
import mujoco
import numpy as np

from .estimator import DLOPositionEstimator
from .geometry import (
    project_points,
    resample_polyline,
    reversal_invariant_errors,
    transform_points,
)
from .recorded_benchmark import (
    camera_matrix,
    choose_evenly_spaced,
    discover_episode_dirs,
    world_from_camera_optical,
)
from .temporal_tracker import TemporalDLOTracker
from .hypothesis_tracker import MultiHypothesisDLOTracker
from .upe_tracker import UPETrackTracker


@dataclass
class TemporalFrameRecord:
    camera: str
    scenario: str
    episode: str
    frame: int
    state_index: int
    ok: bool
    failure: str
    target_in_frame_fraction: float
    whole_target_in_frame: bool
    observed_length_ratio: float
    completion_coverage: float
    confidence: float
    completed: bool
    visible_point_error_m: float
    occluded_point_error_m: float
    image_visible_fraction: float
    image_visible_point_error_m: float
    image_occluded_point_error_m: float
    mean_point_error_m: float
    rmse_point_error_m: float
    endpoint_error_m: float
    length_ratio: float
    render_ms: float
    segmentation_ms: float
    skeleton_ordering_ms: float
    depth_geometry_ms: float
    tracker_ms: float
    algorithm_ms: float
    rigid_residual_m: float
    rigid_translation_m: float
    rigid_angle_deg: float
    rigid_transform_applied: bool
    prior_step_m: float
    centroid_step_m: float
    sequence_residual_m: float
    sequence_translation_m: float
    sequence_angle_deg: float
    sequence_transform_applied: bool
    component_count: float
    selected_component_area: float
    total_component_area: float
    # Diagnostics for the two failure modes this benchmark is intended to
    # expose.  Defaults keep older callers that construct records manually
    # source-compatible.
    ground_truth_crossing: bool = False
    order_inversion_fraction: float = float("nan")
    crossing_candidate_count: float = float("nan")
    skeleton_had_crossing: bool = False
    hypothesis_count: float = float("nan")
    hypothesis_score: float = float("nan")
    robot_occlusion_fraction: float = float("nan")
    robot_mask_ms: float = float("nan")
    processing_ms: float = float("nan")
    # Two-view fusion diagnostics.  These are fractions of the 14 output
    # nodes supported by each camera's current RGB mask, measured after the
    # shared world-space state has been updated.
    opst_support_fraction: float = float("nan")
    wrist_support_fraction: float = float("nan")
    opst_only_fraction: float = float("nan")
    wrist_only_fraction: float = float("nan")
    both_support_fraction: float = float("nan")
    hidden_from_both_fraction: float = float("nan")


def _render_rgb_depth(renderer, data, camera_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    start = perf_counter_ns()
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render().copy()
    return rgb, depth, (perf_counter_ns() - start) * 1e-6


def _render_robot_mask(
    renderer: mujoco.Renderer,
    data,
    camera_name: str,
    robot_geom_ids: set[int],
) -> tuple[np.ndarray, float]:
    """Render a boolean robot-geometry mask for occlusion-aware tracking."""

    start = perf_counter_ns()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name)
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    geom_ids = np.asarray(segmentation[..., 0], dtype=np.int64)
    return np.isin(geom_ids, np.asarray(sorted(robot_geom_ids), dtype=np.int64)), (
        perf_counter_ns() - start
    ) * 1e-6


def run_temporal_benchmark(
    *,
    run_root: Path,
    project_src: Path,
    output: Path,
    scenarios: list[str],
    cameras: list[str],
    episodes_per_scenario: int | None,
    frame_stride: int,
    max_frames_per_episode: int | None,
    sample_count: int = 14,
    # The rendered depth is a near-surface depth, so the empirically calibrated
    # effective centerline correction is slightly larger than the nominal cable
    # radius used by the scene model.
    cable_radius_m: float = 0.020,
    surface_to_center_mode: str = "adaptive",
    adaptive_normal_residual_m: float = 0.006,
    adaptive_normal_weight: float = 0.5,
    spline_smoothing: float = 0.0005,
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
    image_match_weight: float = 0.0,
    allow_partial_initialization: bool = False,
    use_fragments: bool = False,
    use_crossing_hypotheses: bool = False,
    use_multi_hypothesis: bool = False,
    tracker_method: str = "temporal",
    hypothesis_beam_width: int = 2,
    hypothesis_hold_weight: float = 20.0,
    use_robot_occlusion_mask: bool = False,
) -> dict:
    project_src = project_src.resolve()
    if tracker_method not in {"temporal", "upetrack"}:
        raise ValueError(f"unsupported tracker_method: {tracker_method}")
    if tracker_method == "upetrack" and (use_fragments or use_multi_hypothesis):
        raise ValueError("UPETrack reproduction currently supports a single centerline observation")
    if use_multi_hypothesis and use_fragments:
        raise ValueError("multi-hypothesis tracking currently supports single centerline observations only")
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario

    output.mkdir(parents=True, exist_ok=True)
    records: list[TemporalFrameRecord] = []
    selected_manifest: list[dict] = []

    for scenario_name in scenarios:
        episode_dirs = choose_evenly_spaced(
            discover_episode_dirs(run_root, scenario_name), episodes_per_scenario
        )
        if not episode_dirs:
            continue
        with (episode_dirs[0] / "episode.json").open("r", encoding="utf-8") as stream:
            first_metadata = json.load(stream)
        seed = int(first_metadata["result"]["requested_seed"])
        scenario = get_scenario(scenario_name)
        config = env_config_for_scenario(scenario, seed=seed, episode_seconds=15.0)
        config.dynamicvla_cameras_enabled = True
        env = CableGraspEnv(config)
        renderer = mujoco.Renderer(env.model, height=360, width=480)
        robot_geom_ids = {
            int(index)
            for index, body_id in enumerate(env.model.geom_bodyid)
            if int(body_id) in set(range(1, 12))
        }
        try:
            camera_specs = {}
            for short_name in cameras:
                if short_name == "opst":
                    camera_id = int(env.dynamicvla_opst_camera_id)
                    model_name = env.config.dynamicvla_opst_camera_name
                elif short_name == "wrist":
                    camera_id = int(env.dynamicvla_wrist_camera_id)
                    model_name = env.config.dynamicvla_wrist_camera_name
                else:
                    raise ValueError(f"unsupported camera: {short_name}")
                intrinsics = camera_matrix(
                    480, 360, float(env.model.cam_fovy[camera_id])
                )
                camera_specs[short_name] = (camera_id, model_name, intrinsics)

            for episode_dir in episode_dirs:
                trajectory = np.load(
                    episode_dir / "trajectory.npz", allow_pickle=False
                )
                state_spec = mujoco.mjtState(int(trajectory["state_spec"]))
                expected_state_size = mujoco.mj_stateSize(env.model, state_spec)
                if trajectory["states"].shape[1] != expected_state_size:
                    raise RuntimeError(
                        f"state size mismatch in {episode_dir}: "
                        f"{trajectory['states'].shape[1]} versus {expected_state_size}"
                    )
                frame_indices = np.arange(
                    0,
                    len(trajectory["frame_state_indices"]),
                    max(1, frame_stride),
                    dtype=np.int64,
                )
                if max_frames_per_episode is not None:
                    frame_indices = frame_indices[:max_frames_per_episode]
                selected_manifest.append(
                    {
                        "scenario": scenario_name,
                        "episode": episode_dir.name,
                        "frames": int(len(frame_indices)),
                    }
                )
                estimators = {
                    camera: DLOPositionEstimator(
                        camera_specs[camera][2],
                        sample_count=sample_count,
                        surface_to_center_offset_m=cable_radius_m,
                        surface_to_center_mode=surface_to_center_mode,
                        adaptive_normal_residual_m=adaptive_normal_residual_m,
                        adaptive_normal_weight=adaptive_normal_weight,
                        use_crossing_hypotheses=use_crossing_hypotheses,
                        spline_smoothing=spline_smoothing,
                    )
                    for camera in cameras
                }
                if tracker_method == "upetrack":
                    tracker_class = UPETrackTracker
                else:
                    tracker_class = (
                        MultiHypothesisDLOTracker if use_multi_hypothesis else TemporalDLOTracker
                    )
                trackers = {
                    camera: tracker_class(
                        sample_count=sample_count,
                        expected_length_m=expected_length_m,
                        min_initial_length_ratio=min_initial_length_ratio,
                        observation_gain=observation_gain,
                        velocity_gain=velocity_gain,
                        velocity_decay=velocity_decay,
                        length_regularization_gain=length_regularization_gain,
                        low_coverage_gain_scale=low_coverage_gain_scale,
                        confidence_gain_scale=confidence_gain_scale,
                        confidence_gain_floor=confidence_gain_floor,
                        confidence_gain_reference=confidence_gain_reference,
                        confidence_gain_low_coverage_only=confidence_gain_low_coverage_only,
                        rigid_transform_gain=rigid_transform_gain,
                        rigid_residual_threshold_m=rigid_residual_threshold_m,
                        rigid_min_prior_step_m=rigid_min_prior_step_m,
                        rigid_max_translation_m=rigid_max_translation_m,
                        rigid_max_angle_deg=rigid_max_angle_deg,
                        arc_continuity_weight=arc_continuity_weight,
                        centroid_motion_weight=centroid_motion_weight,
                        centroid_motion_slack_m=centroid_motion_slack_m,
                        sequence_min_translation_m=sequence_min_translation_m,
                        rigid_min_match_confidence=rigid_min_match_confidence,
                        sequence_motion_weight=sequence_motion_weight,
                        sequence_motion_min_prior_step_m=sequence_motion_min_prior_step_m,
                        sequence_disagreement_translation_m=sequence_disagreement_translation_m,
                        sequence_disagreement_slack_m=sequence_disagreement_slack_m,
                        sequence_disagreement_persistence=sequence_disagreement_persistence,
                        sequence_disagreement_hold_previous=sequence_disagreement_hold_previous,
                        sequence_disagreement_use_centroid_translation=sequence_disagreement_use_centroid_translation,
                        sequence_disagreement_low_coverage_only=sequence_disagreement_low_coverage_only,
                        sequence_disagreement_velocity_gain=sequence_disagreement_velocity_gain,
                        sequence_disagreement_velocity_min_centroid_step_m=sequence_disagreement_velocity_min_centroid_step_m,
                        low_coverage_deformation_gain=low_coverage_deformation_gain,
                        low_coverage_deformation_residual_m=low_coverage_deformation_residual_m,
                        low_coverage_deformation_min_centroid_step_m=low_coverage_deformation_min_centroid_step_m,
                        low_coverage_deformation_persistence=low_coverage_deformation_persistence,
                        low_coverage_deformation_rigid_gain=low_coverage_deformation_rigid_gain,
                        low_coverage_rigid_observation_gain=low_coverage_rigid_observation_gain,
                        hidden_motion_gain=hidden_motion_gain,
                        reacquisition_min_observed_ratio=reacquisition_min_observed_ratio,
                        reacquisition_min_coverage=reacquisition_min_coverage,
                        reacquisition_max_match_confidence=reacquisition_max_match_confidence,
                        arc_hysteresis_jump=arc_hysteresis_jump,
                        arc_hysteresis_margin_m=arc_hysteresis_margin_m,
                        allow_partial_initialization=allow_partial_initialization,
                         **(
                            {
                                "beam_width": hypothesis_beam_width,
                                "hold_weight": hypothesis_hold_weight,
                            }
                             if use_multi_hypothesis and tracker_method != "upetrack"
                             else {}
                         ),
                    )
                    for camera in cameras
                }
                last_times = {camera: None for camera in cameras}

                for frame_number in frame_indices:
                    state_index = int(trajectory["frame_state_indices"][frame_number])
                    mujoco.mj_setState(
                        env.model,
                        env.data,
                        trajectory["states"][state_index],
                        state_spec,
                    )
                    mujoco.mj_forward(env.model, env.data)
                    cable_world = env.data.xpos[env.cable_ids].copy()
                    current_time = float(env.data.time)

                    for camera in cameras:
                        camera_id, model_name, intrinsics = camera_specs[camera]
                        rgb, depth, render_ms = _render_rgb_depth(
                            renderer, env.data, model_name
                        )
                        robot_mask = None
                        robot_mask_ms = 0.0
                        if use_robot_occlusion_mask:
                            robot_mask, robot_mask_ms = _render_robot_mask(
                                renderer, env.data, model_name, robot_geom_ids
                            )
                        world_from_camera = world_from_camera_optical(
                            env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
                        )
                        camera_from_world = np.linalg.inv(world_from_camera)
                        target_camera = transform_points(camera_from_world, cable_world)
                        target_14 = resample_polyline(target_camera, sample_count)
                        target_pixels = project_points(target_14, intrinsics)
                        target_dense_pixels = project_points(target_camera, intrinsics)
                        ground_truth_crossing = _polyline_has_crossing(target_dense_pixels)
                        visible = (
                            np.isfinite(target_pixels).all(axis=1)
                            & (target_14[:, 2] > 1e-8)
                            & (target_pixels[:, 0] >= 0)
                            & (target_pixels[:, 0] < 480)
                            & (target_pixels[:, 1] >= 0)
                            & (target_pixels[:, 1] < 360)
                        )
                        visible_fraction = float(np.mean(visible))
                        whole_visible = bool(np.all(visible))
                        try:
                            if use_fragments:
                                estimates = estimators[camera].estimate_fragments(rgb, depth)
                                observed_world_fragments = [
                                    transform_points(world_from_camera, item.points_camera)
                                    for item in estimates
                                ]
                                observed_lengths = [
                                    float(
                                        np.linalg.norm(
                                            np.diff(item, axis=0), axis=1
                                        ).sum()
                                    )
                                    for item in observed_world_fragments
                                ]
                                estimate = estimates[0]
                                observed_length = float(sum(observed_lengths))
                            else:
                                estimate = estimators[camera].estimate(rgb, depth)
                                observed_world = transform_points(
                                    world_from_camera, estimate.points_camera
                                )
                                observed_length = float(
                                    np.linalg.norm(
                                        np.diff(observed_world, axis=0), axis=1
                                    ).sum()
                                )
                            previous_time = last_times[camera]
                            dt = (
                                0.04 * max(1, frame_stride)
                                if previous_time is None
                                else max(1e-3, current_time - previous_time)
                            )
                            last_times[camera] = current_time
                            if use_fragments:
                                tracked = trackers[camera].update_fragments(
                                    observed_world_fragments,
                                    observed_lengths_m=observed_lengths,
                                    dt_s=dt,
                                    robot_occlusion_mask=robot_mask,
                                    camera_from_world=camera_from_world,
                                    intrinsics=intrinsics,
                                )
                            else:
                                tracked = trackers[camera].update(
                                    observed_world,
                                    observed_length_m=observed_length,
                                    dt_s=dt,
                                    observed_pixels=project_points(
                                        estimate.points_camera, intrinsics
                                    ),
                                    observed_image_mask=estimate.mask,
                                    camera_from_world=camera_from_world,
                                    intrinsics=intrinsics,
                                    robot_occlusion_mask=robot_mask,
                                    image_match_weight=image_match_weight,
                                )
                            tracker_ms = tracked.timings_ms["total"]
                            observed_ratio = observed_length / expected_length_m
                            if tracked.points_world is None:
                                raise RuntimeError("tracker_not_initialized")
                            predicted_camera = transform_points(
                                camera_from_world, tracked.points_world
                            )
                            errors, reverse = reversal_invariant_errors(
                                predicted_camera, target_14
                            )
                            mask = tracked.observed_mask.copy()
                            if reverse:
                                mask = mask[::-1]
                            visible_error = (
                                float(np.mean(errors[mask]))
                                if np.any(mask)
                                else float("nan")
                            )
                            occluded_error = (
                                float(np.mean(errors[~mask]))
                                if np.any(~mask)
                                else float("nan")
                            )
                            image_visible = _target_visible_from_mask(
                                target_pixels, estimate.mask
                            )
                            if reverse:
                                image_visible = image_visible[::-1]
                            order_inversion_fraction = _order_inversion_fraction(
                                predicted_camera, target_14
                            )
                            image_visible_error = (
                                float(np.mean(errors[image_visible]))
                                if np.any(image_visible)
                                else float("nan")
                            )
                            image_occluded_error = (
                                float(np.mean(errors[~image_visible]))
                                if np.any(~image_visible)
                                else float("nan")
                            )
                            predicted_length = float(
                                np.linalg.norm(
                                    np.diff(predicted_camera, axis=0), axis=1
                                ).sum()
                            )
                            target_length = float(
                                np.linalg.norm(
                                    np.diff(target_camera, axis=0), axis=1
                                ).sum()
                            )
                            timings = (
                                _aggregate_fragment_timings(estimates)
                                if use_fragments
                                else estimate.timings_ms
                            )
                            records.append(
                                TemporalFrameRecord(
                                    camera=camera,
                                    scenario=scenario_name,
                                    episode=episode_dir.name,
                                    frame=int(frame_number),
                                    state_index=state_index,
                                    ok=True,
                                    failure="",
                                    target_in_frame_fraction=visible_fraction,
                                    whole_target_in_frame=whole_visible,
                                    observed_length_ratio=observed_ratio,
                                    completion_coverage=tracked.coverage,
                                    confidence=tracked.confidence,
                                    completed=tracked.used_prediction,
                                    visible_point_error_m=visible_error,
                                    occluded_point_error_m=occluded_error,
                                    image_visible_fraction=float(np.mean(image_visible)),
                                    image_visible_point_error_m=image_visible_error,
                                    image_occluded_point_error_m=image_occluded_error,
                                    mean_point_error_m=float(np.mean(errors)),
                                    rmse_point_error_m=float(
                                        np.sqrt(np.mean(errors * errors))
                                    ),
                                    endpoint_error_m=float(np.mean(errors[[0, -1]])),
                                    length_ratio=predicted_length / max(target_length, 1e-9),
                                    render_ms=render_ms,
                                    segmentation_ms=timings["segmentation"],
                                    skeleton_ordering_ms=timings["skeleton_ordering"],
                                    depth_geometry_ms=timings["depth_geometry"],
                                    tracker_ms=tracker_ms,
                                     algorithm_ms=timings["total"] + tracker_ms + robot_mask_ms,
                                    rigid_residual_m=tracked.timings_ms.get("rigid_residual", float("nan")),
                                    rigid_translation_m=tracked.timings_ms.get("rigid_translation", float("nan")),
                                    rigid_angle_deg=tracked.timings_ms.get("rigid_angle", float("nan")) * 180.0 / np.pi,
                                    rigid_transform_applied=bool(tracked.timings_ms.get("rigid_applied", 0.0)),
                                    prior_step_m=tracked.timings_ms.get("prior_step", float("nan")),
                                    centroid_step_m=tracked.timings_ms.get("centroid_step", float("nan")),
                                    sequence_residual_m=tracked.timings_ms.get("sequence_residual", float("nan")),
                                    sequence_translation_m=tracked.timings_ms.get("sequence_translation", float("nan")),
                                    sequence_angle_deg=tracked.timings_ms.get("sequence_angle", float("nan")) * 180.0 / np.pi,
                                    sequence_transform_applied=bool(tracked.timings_ms.get("sequence_applied", 0.0)),
                                    component_count=timings.get("component_count", float("nan")),
                                     selected_component_area=timings.get("selected_component_area", float("nan")),
                                     total_component_area=timings.get("total_component_area", float("nan")),
                                     ground_truth_crossing=ground_truth_crossing,
                                     order_inversion_fraction=order_inversion_fraction,
                                     crossing_candidate_count=timings.get(
                                         "crossing_candidate_count",
                                         timings.get("crossing_candidates", float("nan")),
                                     ),
                                     skeleton_had_crossing=bool(
                                         any(item.had_crossing for item in estimates)
                                         if use_fragments
                                         else estimate.had_crossing
                                     ),
                                     hypothesis_count=tracked.timings_ms.get(
                                         "hypothesis_count", float("nan")
                                     ),
                                     hypothesis_score=tracked.timings_ms.get(
                                         "hypothesis_score", float("nan")
                                     ),
                                     robot_occlusion_fraction=tracked.timings_ms.get(
                                         "robot_occlusion_fraction", float("nan")
                                     ),
                                     robot_mask_ms=robot_mask_ms,
                                     processing_ms=timings["total"] + tracker_ms + robot_mask_ms,
                                 )
                            )
                        except Exception as exc:
                            nan = float("nan")
                            records.append(
                                TemporalFrameRecord(
                                    camera=camera,
                                    scenario=scenario_name,
                                    episode=episode_dir.name,
                                    frame=int(frame_number),
                                    state_index=state_index,
                                    ok=False,
                                    failure=f"{type(exc).__name__}:{exc}",
                                    target_in_frame_fraction=visible_fraction,
                                    whole_target_in_frame=whole_visible,
                                    observed_length_ratio=nan,
                                    completion_coverage=nan,
                                    confidence=0.0,
                                    completed=False,
                                    visible_point_error_m=nan,
                                    occluded_point_error_m=nan,
                                    image_visible_fraction=nan,
                                    image_visible_point_error_m=nan,
                                    image_occluded_point_error_m=nan,
                                    mean_point_error_m=nan,
                                    rmse_point_error_m=nan,
                                    endpoint_error_m=nan,
                                    length_ratio=nan,
                                    render_ms=render_ms,
                                    segmentation_ms=nan,
                                    skeleton_ordering_ms=nan,
                                    depth_geometry_ms=nan,
                                    tracker_ms=nan,
                                    algorithm_ms=nan,
                                    rigid_residual_m=nan,
                                    rigid_translation_m=nan,
                                    rigid_angle_deg=nan,
                                    rigid_transform_applied=False,
                                    prior_step_m=nan,
                                    centroid_step_m=nan,
                                    sequence_residual_m=nan,
                                    sequence_translation_m=nan,
                                    sequence_angle_deg=nan,
                                    sequence_transform_applied=False,
                                    component_count=nan,
                                     selected_component_area=nan,
                                     total_component_area=nan,
                                     ground_truth_crossing=ground_truth_crossing,
                                     order_inversion_fraction=nan,
                                     crossing_candidate_count=nan,
                                     skeleton_had_crossing=False,
                                     hypothesis_count=nan,
                                     hypothesis_score=nan,
                                 )
                            )
                trajectory.close()
        finally:
            renderer.close()
            env.close()

    if not records:
        raise RuntimeError("no recorded frames were evaluated")
    _write_records(output / "per_frame.csv", records)
    groups = _group_summaries(records)
    _write_dict_rows(output / "per_camera_scenario.csv", groups)
    summary = {
        "source_run": str(run_root.resolve()),
        "project_src": str(project_src),
        "method": (
            "HSV + skeleton + UPETrack-style compact arc registration + "
            "unidirectional position estimation + geodesic resampling"
            if tracker_method == "upetrack"
            else (
                "HSV + skeleton + temporal arc-length matching + adaptive "
                "motion-coherent completion + cable-length regularization + "
                "two-level rigid-motion gate + centroid-motion prior + "
                "motion-constrained arc matching + consecutive-centerline rigid "
                "transport"
                + (" + disconnected-fragment fusion" if use_fragments else "")
                + (" + projected robot occlusion mask" if use_robot_occlusion_mask else "")
            )
        ),
        "tracker_method": tracker_method,
        "sample_count": sample_count,
        "surface_to_center_mode": surface_to_center_mode,
        "adaptive_normal_residual_m": adaptive_normal_residual_m,
        "adaptive_normal_weight": adaptive_normal_weight,
        "frame_stride": frame_stride,
        "episodes_per_scenario": episodes_per_scenario,
        "expected_length_m": expected_length_m,
        "spline_smoothing": spline_smoothing,
        "min_initial_length_ratio": min_initial_length_ratio,
        "observation_gain": observation_gain,
        "velocity_gain": velocity_gain,
        "velocity_decay": velocity_decay,
        "length_regularization_gain": length_regularization_gain,
        "low_coverage_gain_scale": low_coverage_gain_scale,
        "confidence_gain_scale": confidence_gain_scale,
        "confidence_gain_floor": confidence_gain_floor,
        "confidence_gain_reference": confidence_gain_reference,
        "confidence_gain_low_coverage_only": confidence_gain_low_coverage_only,
        "rigid_transform_gain": rigid_transform_gain,
        "rigid_residual_threshold_m": rigid_residual_threshold_m,
        "rigid_min_prior_step_m": rigid_min_prior_step_m,
        "rigid_max_translation_m": rigid_max_translation_m,
        "rigid_max_angle_deg": rigid_max_angle_deg,
        "arc_continuity_weight": arc_continuity_weight,
        "centroid_motion_weight": centroid_motion_weight,
        "centroid_motion_slack_m": centroid_motion_slack_m,
        "sequence_min_translation_m": sequence_min_translation_m,
        "rigid_min_match_confidence": rigid_min_match_confidence,
        "sequence_motion_weight": sequence_motion_weight,
        "sequence_motion_min_prior_step_m": sequence_motion_min_prior_step_m,
        "sequence_disagreement_translation_m": sequence_disagreement_translation_m,
        "sequence_disagreement_slack_m": sequence_disagreement_slack_m,
        "sequence_disagreement_persistence": sequence_disagreement_persistence,
        "sequence_disagreement_hold_previous": sequence_disagreement_hold_previous,
        "sequence_disagreement_use_centroid_translation": sequence_disagreement_use_centroid_translation,
        "sequence_disagreement_low_coverage_only": sequence_disagreement_low_coverage_only,
        "sequence_disagreement_velocity_gain": sequence_disagreement_velocity_gain,
        "sequence_disagreement_velocity_min_centroid_step_m": sequence_disagreement_velocity_min_centroid_step_m,
        "low_coverage_deformation_gain": low_coverage_deformation_gain,
        "low_coverage_deformation_residual_m": low_coverage_deformation_residual_m,
        "low_coverage_deformation_min_centroid_step_m": low_coverage_deformation_min_centroid_step_m,
        "low_coverage_deformation_persistence": low_coverage_deformation_persistence,
        "low_coverage_deformation_rigid_gain": low_coverage_deformation_rigid_gain,
        "low_coverage_rigid_observation_gain": low_coverage_rigid_observation_gain,
        "hidden_motion_gain": hidden_motion_gain,
        "reacquisition_min_observed_ratio": reacquisition_min_observed_ratio,
        "reacquisition_min_coverage": reacquisition_min_coverage,
        "reacquisition_max_match_confidence": reacquisition_max_match_confidence,
        "arc_hysteresis_jump": arc_hysteresis_jump,
        "arc_hysteresis_margin_m": arc_hysteresis_margin_m,
        "image_match_weight": image_match_weight,
        "allow_partial_initialization": allow_partial_initialization,
        "use_fragments": use_fragments,
        "use_crossing_hypotheses": use_crossing_hypotheses,
        "use_multi_hypothesis": use_multi_hypothesis,
        "hypothesis_beam_width": hypothesis_beam_width,
        "hypothesis_hold_weight": hypothesis_hold_weight,
        "use_robot_occlusion_mask": use_robot_occlusion_mask,
        "selected_episodes": selected_manifest,
        "camera_results": {
            camera: _summary_for([record for record in records if record.camera == camera])
            for camera in cameras
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def _values(records: list[TemporalFrameRecord], field: str) -> np.ndarray:
    values = np.asarray([getattr(record, field) for record in records], dtype=float)
    return values[np.isfinite(values)]


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Return whether two 2-D segments intersect, including collinear overlap."""

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        first = q - p
        second = r - p
        return float(first[0] * second[1] - first[1] * second[0])

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return bool(
            min(float(p[0]), float(r[0])) - 1e-6 <= float(q[0]) <= max(float(p[0]), float(r[0])) + 1e-6
            and min(float(p[1]), float(r[1])) - 1e-6 <= float(q[1]) <= max(float(p[1]), float(r[1])) + 1e-6
        )

    ab_c = orient(a, b, c)
    ab_d = orient(a, b, d)
    cd_a = orient(c, d, a)
    cd_b = orient(c, d, b)
    eps = 1e-6
    if (ab_c > eps and ab_d < -eps or ab_c < -eps and ab_d > eps) and (
        cd_a > eps and cd_b < -eps or cd_a < -eps and cd_b > eps
    ):
        return True
    return bool(
        abs(ab_c) <= eps and on_segment(a, c, b)
        or abs(ab_d) <= eps and on_segment(a, d, b)
        or abs(cd_a) <= eps and on_segment(c, a, d)
        or abs(cd_b) <= eps and on_segment(c, b, d)
    )


def _polyline_has_crossing(pixels: np.ndarray) -> bool:
    """Detect a projected self-crossing in a dense centerline polyline."""

    points = np.asarray(pixels, dtype=np.float64)
    if len(points) < 4:
        return False
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    for first in range(len(points) - 3):
        for second in range(first + 2, len(points) - 1):
            # Consecutive segments share a vertex and are not a topological
            # crossing.  The loop starts at first+2, so that case is excluded.
            if _segments_intersect(
                points[first], points[first + 1], points[second], points[second + 1]
            ):
                return True
    return False


def _order_inversion_fraction(predicted: np.ndarray, target: np.ndarray) -> float:
    """Fraction of adjacent estimated nodes that move backwards in GT arc order."""

    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or len(predicted) < 3:
        return float("nan")
    forward = float(np.mean(np.linalg.norm(predicted - target, axis=1)))
    reverse = float(np.mean(np.linalg.norm(predicted[::-1] - target, axis=1)))
    aligned = predicted[::-1] if reverse < forward else predicted
    pairwise = np.linalg.norm(aligned[:, None, :] - target[None, :, :], axis=2)
    nearest = np.argmin(pairwise, axis=1)
    return float(np.mean(np.diff(nearest) < 0))


def _target_visible_from_mask(
    target_pixels: np.ndarray, mask: np.ndarray, *, radius_px: int = 5
) -> np.ndarray:
    """Approximate which GT nodes are visible in the RGB segmentation.

    The rendered target centerline need not fall exactly on the segmented
    cable center (surface thickness, rasterisation and depth correction all
    contribute a few pixels), so test a small dilation of the observed mask.
    This is deliberately an evaluation diagnostic; it does not feed the
    tracker and therefore cannot hide an identity-matching failure.
    """

    pixels = np.asarray(target_pixels, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        return np.zeros(len(pixels), dtype=bool)
    kernel_size = max(1, 2 * int(radius_px) + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    expanded = cv2.dilate((mask > 0).astype(np.uint8), kernel)
    visible = np.zeros(len(pixels), dtype=bool)
    for index, pixel in enumerate(pixels):
        if not np.isfinite(pixel).all():
            continue
        col, row = np.rint(pixel).astype(int)
        if 0 <= row < expanded.shape[0] and 0 <= col < expanded.shape[1]:
            visible[index] = bool(expanded[row, col])
    return visible


def _stats(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _summary_for(records: list[TemporalFrameRecord]) -> dict:
    successful = [record for record in records if record.ok]
    latency = _values(successful, "algorithm_ms")
    return {
        "frames_attempted": len(records),
        "frames_successful": len(successful),
        "full_state_success_rate": len(successful) / max(len(records), 1),
        "whole_target_in_frame_rate": sum(
            record.whole_target_in_frame for record in records
        )
        / max(len(records), 1),
        "mean_target_in_frame_fraction": float(
            np.nanmean([record.target_in_frame_fraction for record in records])
        ),
        "mean_observed_length_ratio": _stats(_values(records, "observed_length_ratio")),
        "mean_completion_coverage": _stats(_values(successful, "completion_coverage")),
        "mean_confidence": _stats(_values(successful, "confidence")),
        "completion_used_rate": float(
            np.mean([record.completed for record in successful])
        )
        if successful
        else 0.0,
        "mean_point_error_m": _stats(_values(successful, "mean_point_error_m")),
        "visible_point_error_m": _stats(_values(successful, "visible_point_error_m")),
        "occluded_point_error_m": _stats(_values(successful, "occluded_point_error_m")),
        "image_visible_fraction": _stats(
            _values(successful, "image_visible_fraction")
        ),
        "image_visible_point_error_m": _stats(
            _values(successful, "image_visible_point_error_m")
        ),
        "image_occluded_point_error_m": _stats(
            _values(successful, "image_occluded_point_error_m")
        ),
        "crossing_frame_rate": float(
            np.mean([record.ground_truth_crossing for record in successful])
        )
        if successful
        else 0.0,
        "crossing_mean_point_error_m": _stats(
            _values(
                [record for record in successful if record.ground_truth_crossing],
                "mean_point_error_m",
            )
        ),
        "noncrossing_mean_point_error_m": _stats(
            _values(
                [record for record in successful if not record.ground_truth_crossing],
                "mean_point_error_m",
            )
        ),
        "order_inversion_fraction": _stats(
            _values(successful, "order_inversion_fraction")
        ),
        "crossing_candidate_count": _stats(
            _values(successful, "crossing_candidate_count")
        ),
        "skeleton_branch_frame_rate": float(
            np.mean([record.skeleton_had_crossing for record in successful])
        )
        if successful
        else 0.0,
        "hypothesis_count": _stats(_values(successful, "hypothesis_count")),
        "robot_occlusion_fraction": _stats(
            _values(successful, "robot_occlusion_fraction")
        ),
        "robot_mask_ms": _stats(_values(successful, "robot_mask_ms")),
        "processing_ms": _stats(_values(successful, "processing_ms")),
        "opst_support_fraction": _stats(
            _values(successful, "opst_support_fraction")
        ),
        "wrist_support_fraction": _stats(
            _values(successful, "wrist_support_fraction")
        ),
        "opst_only_fraction": _stats(_values(successful, "opst_only_fraction")),
        "wrist_only_fraction": _stats(_values(successful, "wrist_only_fraction")),
        "both_support_fraction": _stats(_values(successful, "both_support_fraction")),
        "hidden_from_both_fraction": _stats(
            _values(successful, "hidden_from_both_fraction")
        ),
        "algorithm_ms": _stats(latency),
        "tracker_ms": _stats(_values(successful, "tracker_ms")),
        "algorithm_fps_from_mean": float(1000.0 / np.mean(latency))
        if len(latency)
        else 0.0,
        "within_20ms_rate": float(np.mean(latency <= 20.0)) if len(latency) else 0.0,
        "meets_50hz_p95": bool(len(latency) and np.percentile(latency, 95) <= 20.0),
    }


def _group_summaries(records: list[TemporalFrameRecord]) -> list[dict]:
    rows = []
    groups = sorted({(record.camera, record.scenario) for record in records})
    for camera, scenario in groups:
        subset = [
            record
            for record in records
            if record.camera == camera and record.scenario == scenario
        ]
        summary = _summary_for(subset)
        rows.append(
            {
                "camera": camera,
                "scenario": scenario,
                "frames": summary["frames_attempted"],
                "full_state_success_rate": summary["full_state_success_rate"],
                "whole_target_in_frame_rate": summary["whole_target_in_frame_rate"],
                "mean_observed_length_ratio": summary["mean_observed_length_ratio"]["mean"],
                "mean_completion_coverage": summary["mean_completion_coverage"]["mean"],
                "mean_confidence": summary["mean_confidence"]["mean"],
                "completion_used_rate": summary["completion_used_rate"],
                "mean_point_error_m": summary["mean_point_error_m"]["mean"],
                "visible_point_error_m": summary["visible_point_error_m"]["mean"],
                "occluded_point_error_m": summary["occluded_point_error_m"]["mean"],
                "image_visible_fraction": summary["image_visible_fraction"]["mean"],
                "image_visible_point_error_m": summary[
                    "image_visible_point_error_m"
                ]["mean"],
                "image_occluded_point_error_m": summary[
                    "image_occluded_point_error_m"
                ]["mean"],
                "crossing_frame_rate": summary["crossing_frame_rate"],
                "crossing_mean_point_error_m": summary[
                    "crossing_mean_point_error_m"
                ]["mean"],
                "noncrossing_mean_point_error_m": summary[
                    "noncrossing_mean_point_error_m"
                ]["mean"],
                "order_inversion_fraction": summary[
                    "order_inversion_fraction"
                ]["mean"],
                "crossing_candidate_count": summary[
                    "crossing_candidate_count"
                ]["mean"],
                "skeleton_branch_frame_rate": summary["skeleton_branch_frame_rate"],
                "hypothesis_count": summary["hypothesis_count"]["mean"],
                "robot_occlusion_fraction": summary["robot_occlusion_fraction"]["mean"],
                "robot_mask_mean_ms": summary["robot_mask_ms"]["mean"],
                "processing_mean_ms": summary["processing_ms"]["mean"],
                "processing_p95_ms": summary["processing_ms"]["p95"],
                "opst_support_fraction": summary["opst_support_fraction"]["mean"],
                "wrist_support_fraction": summary["wrist_support_fraction"]["mean"],
                "opst_only_fraction": summary["opst_only_fraction"]["mean"],
                "wrist_only_fraction": summary["wrist_only_fraction"]["mean"],
                "both_support_fraction": summary["both_support_fraction"]["mean"],
                "hidden_from_both_fraction": summary[
                    "hidden_from_both_fraction"
                ]["mean"],
                "algorithm_mean_ms": summary["algorithm_ms"]["mean"],
                "algorithm_p95_ms": summary["algorithm_ms"]["p95"],
                "tracker_mean_ms": summary["tracker_ms"]["mean"],
                "within_20ms_rate": summary["within_20ms_rate"],
            }
        )
    return rows


def _write_records(path: Path, records: list[TemporalFrameRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_dict_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_fragment_timings(estimates: list) -> dict[str, float]:
    """Combine per-fragment timings without counting shared segmentation twice."""

    if not estimates:
        return {"segmentation": float("nan"), "skeleton_ordering": float("nan"), "depth_geometry": float("nan"), "total": float("nan")}
    timings = [estimate.timings_ms for estimate in estimates]
    return {
        "segmentation": float(timings[0]["segmentation"]),
        "skeleton_ordering": float(sum(item["skeleton_ordering"] for item in timings)),
        "depth_geometry": float(sum(item["depth_geometry"] for item in timings)),
        "total": float(max(item["total"] for item in timings)),
        "component_count": float(timings[0].get("component_count", len(estimates))),
        "selected_component_area": float(timings[0].get("selected_component_area", 0.0)),
        "total_component_area": float(timings[0].get("total_component_area", 0.0)),
        "crossing_candidate_count": float(
            max(item.get("crossing_candidates", 1.0) for item in timings)
        ),
    }
