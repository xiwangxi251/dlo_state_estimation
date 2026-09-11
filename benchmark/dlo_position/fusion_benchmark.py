from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter_ns

import mujoco
import numpy as np

from .estimator import DLOPositionEstimator
from .geometry import project_points, resample_polyline, reversal_invariant_errors, transform_points
from .recorded_benchmark import camera_matrix, choose_evenly_spaced, discover_episode_dirs, world_from_camera_optical
from .temporal_benchmark import (
    TemporalFrameRecord,
    _group_summaries,
    _render_rgb_depth,
    _summary_for,
    _target_visible_from_mask,
    _polyline_has_crossing,
    _order_inversion_fraction,
    _write_dict_rows,
    _write_records,
)
from .temporal_tracker import TemporalDLOTracker


def run_temporal_fusion_benchmark(
    *,
    run_root: Path,
    project_src: Path,
    output: Path,
    scenarios: list[str],
    episodes_per_scenario: int | None,
    frame_stride: int,
    max_frames_per_episode: int | None,
    sample_count: int = 14,
    cable_radius_m: float = 0.014,
    expected_length_m: float = 0.78,
    min_initial_length_ratio: float = 0.80,
    observation_gain: float = 1.0,
    velocity_gain: float = 0.15,
    velocity_decay: float = 0.80,
    length_regularization_gain: float = 0.30,
    low_coverage_gain_scale: float = 0.50,
    rigid_transform_gain: float = 0.0,
    fusion_mode: str = "pointwise",
) -> dict:
    """Fuse independent opposite/wrist temporal tracks in world coordinates.

    The fixed opposite camera remains the reference frame.  A wrist estimate
    is used only at state nodes it directly observes; otherwise the opposite
    track supplies the complete continuation.  This makes the experiment
    conservative: an unreliable wrist fragment cannot overwrite hidden nodes.
    """

    if fusion_mode not in {"pointwise", "joint"}:
        raise ValueError("fusion_mode must be 'pointwise' or 'joint'")
    joint_mode = fusion_mode == "joint"
    project_src = project_src.resolve()
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario

    output.mkdir(parents=True, exist_ok=True)
    records = []
    selected_manifest = []
    for scenario_name in scenarios:
        episode_dirs = choose_evenly_spaced(
            discover_episode_dirs(run_root, scenario_name), episodes_per_scenario
        )
        if not episode_dirs:
            continue
        with (episode_dirs[0] / "episode.json").open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        config = env_config_for_scenario(
            get_scenario(scenario_name),
            seed=int(metadata["result"]["requested_seed"]),
            episode_seconds=15.0,
        )
        config.dynamicvla_cameras_enabled = True
        env = CableGraspEnv(config)
        renderer = mujoco.Renderer(env.model, height=360, width=480)
        try:
            camera_specs = {}
            for short_name in ("opst", "wrist"):
                if short_name == "opst":
                    camera_id = int(env.dynamicvla_opst_camera_id)
                    model_name = env.config.dynamicvla_opst_camera_name
                else:
                    camera_id = int(env.dynamicvla_wrist_camera_id)
                    model_name = env.config.dynamicvla_wrist_camera_name
                camera_specs[short_name] = (
                    camera_id,
                    model_name,
                    camera_matrix(480, 360, float(env.model.cam_fovy[camera_id])),
                )
            for episode_dir in episode_dirs:
                trajectory = np.load(episode_dir / "trajectory.npz", allow_pickle=False)
                state_spec = mujoco.mjtState(int(trajectory["state_spec"]))
                frame_indices = np.arange(
                    0,
                    len(trajectory["frame_state_indices"]),
                    max(1, frame_stride),
                    dtype=np.int64,
                )
                if max_frames_per_episode is not None:
                    frame_indices = frame_indices[:max_frames_per_episode]
                selected_manifest.append(
                    {"scenario": scenario_name, "episode": episode_dir.name, "frames": int(len(frame_indices))}
                )
                estimators = {
                    name: DLOPositionEstimator(
                        camera_specs[name][2],
                        sample_count=sample_count,
                        surface_to_center_offset_m=cable_radius_m,
                    )
                    for name in ("opst", "wrist")
                }
                trackers = {
                    name: TemporalDLOTracker(
                        sample_count=sample_count,
                        expected_length_m=expected_length_m,
                        min_initial_length_ratio=min_initial_length_ratio,
                        observation_gain=observation_gain,
                        velocity_gain=velocity_gain,
                        velocity_decay=velocity_decay,
                        length_regularization_gain=length_regularization_gain,
                        low_coverage_gain_scale=low_coverage_gain_scale,
                        rigid_transform_gain=rigid_transform_gain,
                    )
                    for name in ("opst", "wrist")
                }
                joint_tracker = TemporalDLOTracker(
                    sample_count=sample_count,
                    expected_length_m=expected_length_m,
                    min_initial_length_ratio=min_initial_length_ratio,
                    observation_gain=observation_gain,
                    velocity_gain=velocity_gain,
                    velocity_decay=velocity_decay,
                    length_regularization_gain=length_regularization_gain,
                    low_coverage_gain_scale=low_coverage_gain_scale,
                    rigid_transform_gain=rigid_transform_gain,
                ) if joint_mode else None
                last_times = {"opst": None, "wrist": None}
                joint_last_time = None
                for frame_number in frame_indices:
                    state_index = int(trajectory["frame_state_indices"][frame_number])
                    mujoco.mj_setState(env.model, env.data, trajectory["states"][state_index], state_spec)
                    mujoco.mj_forward(env.model, env.data)
                    cable_world = env.data.xpos[env.cable_ids].copy()
                    camera_data = {}
                    algorithm_start = perf_counter_ns()
                    render_ms = 0.0
                    for name in ("opst", "wrist"):
                        if name == "wrist":
                            opposite = camera_data.get("opst")
                            # The wrist camera is an event-driven fallback.
                            # Skip its expensive RGB-D pass while the fixed
                            # camera already provides a broad, confident
                            # fragment; trigger it for short/uncertain views.
                            if (
                                not joint_mode
                                and
                                opposite is not None
                                and opposite["tracked"].points_world is not None
                                and opposite["tracked"].coverage >= 0.55
                                and opposite["tracked"].confidence >= 0.15
                            ):
                                camera_data["wrist"] = None
                                continue
                        camera_id, model_name, intrinsics = camera_specs[name]
                        rgb, depth, current_render_ms = _render_rgb_depth(renderer, env.data, model_name)
                        render_ms += current_render_ms
                        world_from_camera = world_from_camera_optical(
                            env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
                        )
                        camera_from_world = np.linalg.inv(world_from_camera)
                        target_camera = transform_points(camera_from_world, cable_world)
                        target_14 = resample_polyline(target_camera, sample_count)
                        target_pixels = project_points(target_14, intrinsics)
                        visible = (
                            np.isfinite(target_pixels).all(axis=1)
                            & (target_14[:, 2] > 1e-8)
                            & (target_pixels[:, 0] >= 0)
                            & (target_pixels[:, 0] < 480)
                            & (target_pixels[:, 1] >= 0)
                            & (target_pixels[:, 1] < 360)
                        )
                        try:
                            estimate = estimators[name].estimate(rgb, depth)
                            observed_world = transform_points(world_from_camera, estimate.points_camera)
                            observed_length = float(np.linalg.norm(np.diff(observed_world, axis=0), axis=1).sum())
                            current_time = float(env.data.time)
                            previous_time = last_times[name]
                            dt = 0.04 * max(1, frame_stride) if previous_time is None else max(1e-3, current_time - previous_time)
                            last_times[name] = current_time
                            tracked = None
                            if not joint_mode:
                                tracked = trackers[name].update(
                                    observed_world,
                                    observed_length_m=observed_length,
                                    dt_s=dt,
                                )
                            camera_data[name] = {
                                "estimate": estimate,
                                "tracked": tracked,
                                "observed_world": observed_world,
                                "observed_length": observed_length,
                                "observed_pixels": project_points(
                                    estimate.points_camera, intrinsics
                                ),
                                "world_from_camera": world_from_camera,
                                "camera_from_world": camera_from_world,
                                "intrinsics": intrinsics,
                                "target_camera": target_camera,
                                "target_14": target_14,
                                "target_pixels": target_pixels,
                                "visible": visible,
                            }
                        except Exception:
                            camera_data[name] = None
                    # Keep raw observations for per-camera utilization
                    # diagnostics.  Joint mode later collapses camera_data to
                    # one shared state, so these references must be retained.
                    fusion_opst = camera_data.get("opst")
                    fusion_wrist = camera_data.get("wrist")
                    if joint_mode:
                        joint_observations = [
                            item
                            for item in (camera_data.get("opst"), camera_data.get("wrist"))
                            if item is not None and item.get("observed_world") is not None
                        ]
                        if joint_observations and joint_tracker is not None:
                            joint_dt = (
                                0.04 * max(1, frame_stride)
                                if joint_last_time is None
                                else max(1e-3, current_time - joint_last_time)
                            )
                            joint_last_time = current_time
                            try:
                                joint_state = joint_tracker.update_fragments(
                                    [item["observed_world"] for item in joint_observations],
                                    observed_lengths_m=[
                                        item["observed_length"] for item in joint_observations
                                    ],
                                    dt_s=joint_dt,
                                    enforce_nonoverlap=False,
                                    observed_pixels_list=[
                                        item["observed_pixels"] for item in joint_observations
                                    ],
                                    observed_image_masks=[
                                        item["estimate"].mask for item in joint_observations
                                    ],
                                    camera_from_worlds=[
                                        item["camera_from_world"] for item in joint_observations
                                    ],
                                    intrinsics_list=[
                                        item["intrinsics"] for item in joint_observations
                                    ],
                                    image_match_weight=0.8,
                                    image_match_min_improvement_m=0.01,
                                )
                                joint_reference_name = (
                                    "opst"
                                    if camera_data.get("opst") is not None
                                    else "wrist"
                                )
                                camera_data[joint_reference_name]["tracked"] = joint_state
                                camera_data[
                                    "wrist" if joint_reference_name == "opst" else "opst"
                                ] = None
                            except Exception:
                                camera_data["opst"] = None
                                camera_data["wrist"] = None
                        else:
                            camera_data["opst"] = None
                            camera_data["wrist"] = None
                    opst = camera_data["opst"]
                    wrist = camera_data["wrist"]
                    if opst is None and wrist is None:
                        records.append(_failure_record(scenario_name, episode_dir.name, int(frame_number), state_index, render_ms))
                        continue
                    reference = opst if opst is not None else wrist
                    reference_target = reference["target_14"]
                    reference_camera_from_world = reference["camera_from_world"]
                    states = [item["tracked"] for item in (opst, wrist) if item is not None and item["tracked"].points_world is not None]
                    if not states:
                        records.append(_failure_record(scenario_name, episode_dir.name, int(frame_number), state_index, render_ms))
                        continue
                    if opst is not None and opst["tracked"].points_world is not None:
                        fused_world = opst["tracked"].points_world.copy()
                        fused_mask = opst["tracked"].observed_mask.copy()
                        fused_coverage = opst["tracked"].coverage
                        fused_confidence = opst["tracked"].confidence
                    else:
                        fused_world = wrist["tracked"].points_world.copy()
                        fused_mask = wrist["tracked"].observed_mask.copy()
                        fused_coverage = wrist["tracked"].coverage
                        fused_confidence = wrist["tracked"].confidence
                    if wrist is not None and wrist["tracked"].points_world is not None:
                        wrist_state = wrist["tracked"]
                        use_wrist = wrist_state.observed_mask & (~fused_mask)
                        both = wrist_state.observed_mask & fused_mask
                        # Wrist observations are accepted on previously hidden
                        # nodes.  On overlap, use cross-view reprojection as a
                        # local identity check.  If both RGB-D tracks agree in
                        # 3-D, average them to reduce surface-depth noise; if
                        # they disagree, prefer the state whose points project
                        # onto the other camera's cable mask, then fall back to
                        # the scalar confidence comparison.
                        fused_world[use_wrist] = wrist_state.points_world[use_wrist]
                        if np.any(both):
                            disagreement = np.linalg.norm(
                                fused_world - wrist_state.points_world, axis=1
                            )
                            close = both & (disagreement <= 0.04)
                            if np.any(close):
                                weight_opst = max(float(fused_confidence), 0.05)
                                weight_wrist = max(float(wrist_state.confidence), 0.05)
                                fused_world[close] = (
                                    weight_opst * fused_world[close]
                                    + weight_wrist * wrist_state.points_world[close]
                                ) / (weight_opst + weight_wrist)
                            conflicting = both & ~close
                            if np.any(conflicting):
                                opst_support = _cross_view_mask_support(
                                    fused_world, wrist
                                )
                                wrist_support = _cross_view_mask_support(
                                    wrist_state.points_world, opst
                                )
                                choose_wrist = conflicting & wrist_support & ~opst_support
                                choose_opst = conflicting & opst_support & ~wrist_support
                                fused_world[choose_wrist] = wrist_state.points_world[
                                    choose_wrist
                                ]
                                unresolved = conflicting & ~(choose_wrist | choose_opst)
                                if wrist_state.confidence > fused_confidence:
                                    fused_world[unresolved] = wrist_state.points_world[
                                        unresolved
                                    ]
                        fused_mask |= wrist_state.observed_mask
                        fused_coverage = float(np.mean(fused_mask))
                        fused_confidence = max(fused_confidence, wrist_state.confidence)
                    # Per-node support in each camera's image mask.  These
                    # fractions expose whether wrist actually contributes
                    # nodes or is merely being rendered and then discarded.
                    opst_support = _cross_view_mask_support(fused_world, fusion_opst)
                    wrist_support = _cross_view_mask_support(fused_world, fusion_wrist)
                    both_support = opst_support & wrist_support
                    opst_only = opst_support & ~wrist_support
                    wrist_only = wrist_support & ~opst_support
                    hidden_from_both = ~(opst_support | wrist_support)
                    predicted_camera = transform_points(reference_camera_from_world, fused_world)
                    errors, reverse = reversal_invariant_errors(predicted_camera, reference_target)
                    mask = fused_mask[::-1] if reverse else fused_mask
                    visible_error = float(np.mean(errors[mask])) if np.any(mask) else float("nan")
                    occluded_error = float(np.mean(errors[~mask])) if np.any(~mask) else float("nan")
                    image_visible = _target_visible_from_mask(
                        reference["target_pixels"], reference["estimate"].mask
                    )
                    if reverse:
                        image_visible = image_visible[::-1]
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
                    ground_truth_crossing = _polyline_has_crossing(
                        project_points(reference["target_camera"], reference["intrinsics"])
                    )
                    order_inversion_fraction = _order_inversion_fraction(
                        predicted_camera, reference_target
                    )
                    predicted_length = float(np.linalg.norm(np.diff(predicted_camera, axis=0), axis=1).sum())
                    target_length = float(np.linalg.norm(np.diff(reference_target, axis=0), axis=1).sum())
                    total_ms = (perf_counter_ns() - algorithm_start) * 1e-6
                    records.append(
                        TemporalFrameRecord(
                            camera="fused_opst_wrist",
                            scenario=scenario_name,
                            episode=episode_dir.name,
                            frame=int(frame_number),
                            state_index=state_index,
                            ok=True,
                            failure="",
                            target_in_frame_fraction=float(np.mean(reference["visible"])),
                            whole_target_in_frame=bool(np.all(reference["visible"])),
                            observed_length_ratio=float(
                                np.mean([
                                    np.linalg.norm(np.diff(item["estimate"].points_camera, axis=0), axis=1).sum()
                                    / expected_length_m
                                    for item in (opst, wrist) if item is not None
                                ]
                            )),
                            completion_coverage=fused_coverage,
                            confidence=fused_confidence,
                            completed=bool(np.any(~fused_mask)),
                            visible_point_error_m=visible_error,
                            occluded_point_error_m=occluded_error,
                            image_visible_fraction=float(np.mean(image_visible)),
                            image_visible_point_error_m=image_visible_error,
                            image_occluded_point_error_m=image_occluded_error,
                            mean_point_error_m=float(np.mean(errors)),
                            rmse_point_error_m=float(np.sqrt(np.mean(errors * errors))),
                            endpoint_error_m=float(np.mean(errors[[0, -1]])),
                            length_ratio=predicted_length / max(target_length, 1e-9),
                            render_ms=render_ms,
                            segmentation_ms=float(sum(item["estimate"].timings_ms["segmentation"] for item in (opst, wrist) if item is not None)),
                            skeleton_ordering_ms=float(sum(item["estimate"].timings_ms["skeleton_ordering"] for item in (opst, wrist) if item is not None)),
                            depth_geometry_ms=float(sum(item["estimate"].timings_ms["depth_geometry"] for item in (opst, wrist) if item is not None)),
                             tracker_ms=float(sum(item["tracked"].timings_ms["total"] for item in (opst, wrist) if item is not None)),
                             algorithm_ms=total_ms,
                             processing_ms=float(
                                 sum(
                                     item["estimate"].timings_ms[stage]
                                     for item in (opst, wrist)
                                     if item is not None
                                     for stage in (
                                         "segmentation",
                                         "skeleton_ordering",
                                         "depth_geometry",
                                     )
                                 )
                                 + sum(
                                     item["tracked"].timings_ms["total"]
                                     for item in (opst, wrist)
                                     if item is not None
                                 )
                             ),
                            rigid_residual_m=float("nan"),
                            rigid_translation_m=float("nan"),
                            rigid_angle_deg=float("nan"),
                            rigid_transform_applied=False,
                            prior_step_m=float("nan"),
                            centroid_step_m=float("nan"),
                            sequence_residual_m=float("nan"),
                            sequence_translation_m=float("nan"),
                            sequence_angle_deg=float("nan"),
                            sequence_transform_applied=False,
                            component_count=float("nan"),
                             selected_component_area=float("nan"),
                             total_component_area=float("nan"),
                             ground_truth_crossing=ground_truth_crossing,
                             order_inversion_fraction=order_inversion_fraction,
                             crossing_candidate_count=reference["estimate"].timings_ms.get(
                                 "crossing_candidates", float("nan")
                             ),
                             skeleton_had_crossing=reference["estimate"].had_crossing,
                             opst_support_fraction=float(np.mean(opst_support)),
                             wrist_support_fraction=float(np.mean(wrist_support)),
                             opst_only_fraction=float(np.mean(opst_only)),
                             wrist_only_fraction=float(np.mean(wrist_only)),
                             both_support_fraction=float(np.mean(both_support)),
                             hidden_from_both_fraction=float(np.mean(hidden_from_both)),
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
        "method": (
            "opst + wrist shared-state joint arc fusion"
            if joint_mode
            else "opst + wrist pointwise temporal fusion"
        ),
        "fusion_mode": fusion_mode,
        "sample_count": sample_count,
        "frame_stride": frame_stride,
        "episodes_per_scenario": episodes_per_scenario,
        "expected_length_m": expected_length_m,
        "selected_episodes": selected_manifest,
        "camera_results": {"fused_opst_wrist": _summary_for(records)},
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def _cross_view_mask_support(points_world: np.ndarray, camera_data: dict | None) -> np.ndarray:
    """Check whether a world-space track reprojects onto another view's mask."""

    if camera_data is None:
        return np.zeros(len(points_world), dtype=bool)
    pixels = project_points(
        transform_points(camera_data["camera_from_world"], points_world),
        camera_data["intrinsics"],
    )
    return _target_visible_from_mask(
        pixels,
        camera_data["estimate"].mask,
        radius_px=5,
    )


def _failure_record(scenario, episode, frame, state_index, render_ms):
    nan = float("nan")
    return TemporalFrameRecord(
        camera="fused_opst_wrist",
        scenario=scenario,
        episode=episode,
        frame=frame,
        state_index=state_index,
        ok=False,
        failure="tracker_not_initialized",
        target_in_frame_fraction=nan,
        whole_target_in_frame=False,
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
    )
