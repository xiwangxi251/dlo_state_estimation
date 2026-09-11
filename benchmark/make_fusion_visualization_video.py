from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

from dlo_position.estimator import DLOPositionEstimator
from dlo_position.geometry import (
    project_points,
    resample_polyline,
    reversal_invariant_errors,
    transform_points,
)
from dlo_position.recorded_benchmark import (
    camera_matrix,
    choose_evenly_spaced,
    discover_episode_dirs,
    world_from_camera_optical,
)
from dlo_position.temporal_benchmark import (
    _render_rgb_depth,
    _target_visible_from_mask,
)
from dlo_position.temporal_tracker import TemporalDLOTracker


DEFAULT_RUN_ROOT = Path(
    r"C:\Users\27642\Desktop\dynamic_cable\linux_log\expert_grasp_fix_4x50\run_20260824_113325"
)
DEFAULT_PROJECT_SRC = Path(
    r"C:\Users\27642\Desktop\dynamic_cable\panda_cable_grasp\src"
)
DEFAULT_SCENARIOS = [
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
]


def _draw_curve(image, pixels, color, thickness=2, radius=3):
    pixels = np.asarray(pixels, dtype=np.float64)
    valid = np.isfinite(pixels).all(axis=1)
    points = np.rint(pixels[valid]).astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), radius, color, -1, cv2.LINE_AA)


def _draw_points(image, pixels, mask, color, radius=4):
    pixels = np.asarray(pixels, dtype=np.float64)
    valid = np.isfinite(pixels).all(axis=1) & np.asarray(mask, dtype=bool)
    for point in np.rint(pixels[valid]).astype(np.int32):
        cv2.circle(image, tuple(point), radius, color, -1, cv2.LINE_AA)


def _text(image, value, y, color=(255, 255, 255), x=8, scale=0.42):
    origin = (int(x), int(y))
    # A black outline keeps the diagnostic labels readable over the robot and cable.
    cv2.putText(
        image,
        str(value),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        str(value),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _make_overlay(rgb, target_pixels, observed_pixels, predicted_pixels, estimate, tracked,
                  title, scenario, frame_number, error_cm, visible_error_cm, proc_ms):
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Green: simulator cable centerline (the reference used only for evaluation).
    _draw_curve(overlay, target_pixels, (0, 220, 0), thickness=2, radius=3)
    # Blue: what the RGB-D centerline extractor can see in this camera.
    if observed_pixels is not None:
        _draw_curve(overlay, observed_pixels, (255, 180, 0), thickness=1, radius=2)
    if predicted_pixels is not None:
        # Red: the 14-node state supplied to RL.  Magenta marks nodes that are not
        # currently supported by this camera's mask (completion/occlusion).
        _draw_curve(overlay, predicted_pixels, (0, 0, 255), thickness=2, radius=3)
        if estimate is not None:
            visible_pred = _target_visible_from_mask(predicted_pixels, estimate.mask)
            _draw_points(overlay, predicted_pixels, ~visible_pred, (255, 0, 255), radius=4)

    if estimate is None:
        _text(overlay, "estimator failed", 20, (0, 0, 255))
    elif tracked is None or tracked.points_world is None:
        _text(overlay, "tracker warming up", 20, (0, 165, 255))
    else:
        _text(
            overlay,
            "err={:.1f}cm  visible={:.1f}cm  coverage={:.2f}  conf={:.2f}".format(
                error_cm, visible_error_cm, tracked.coverage, tracked.confidence
            ),
            20,
        )
    _text(overlay, "green=GT  blue=visible  red=14pt state  magenta=hidden", 40)
    _text(overlay, "{}  {}  frame={}  proc={:.1f}ms".format(title, scenario, frame_number, proc_ms), 60)
    return overlay


def _joint_state(tracker, camera_data, previous_state, dt_s):
    observations = [
        item for item in (camera_data.get("opst"), camera_data.get("wrist"))
        if item is not None and item.get("observed_world") is not None
    ]
    if not observations:
        return previous_state
    try:
        return tracker.update_fragments(
            [item["observed_world"] for item in observations],
            observed_lengths_m=[item["observed_length"] for item in observations],
            dt_s=dt_s,
            enforce_nonoverlap=False,
            observed_pixels_list=[item["observed_pixels"] for item in observations],
            observed_image_masks=[item["estimate"].mask for item in observations],
            camera_from_worlds=[item["camera_from_world"] for item in observations],
            intrinsics_list=[item["intrinsics"] for item in observations],
            image_match_weight=0.8,
            image_match_min_improvement_m=0.01,
        )
    except Exception:
        # A bad depth frame should not make the visualizer stop; the previous
        # complete state is exactly what the online tracker would hold.
        return previous_state


def make_video(
    *,
    run_root: Path,
    project_src: Path,
    output: Path,
    scenarios: list[str],
    episodes_per_scenario: int,
    frame_stride: int,
    max_frames_per_episode: int | None,
    sample_count: int,
    cable_radius_m: float,
    expected_length_m: float,
    fps: float,
):
    project_src = project_src.resolve()
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 480, 360
    separator_width = 4
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
        (2 * width + separator_width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("could not open video writer: {}".format(output))

    written = 0
    try:
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
            renderer = mujoco.Renderer(env.model, height=height, width=width)
            try:
                camera_specs = {}
                for short_name in ("opst", "wrist"):
                    if short_name == "opst":
                        camera_id = int(env.dynamicvla_opst_camera_id)
                        camera_name = env.config.dynamicvla_opst_camera_name
                    else:
                        camera_id = int(env.dynamicvla_wrist_camera_id)
                        camera_name = env.config.dynamicvla_wrist_camera_name
                    camera_specs[short_name] = {
                        "id": camera_id,
                        "name": camera_name,
                        "intrinsics": camera_matrix(
                            width, height, float(env.model.cam_fovy[camera_id])
                        ),
                    }

                for episode_dir in episode_dirs:
                    trajectory = np.load(episode_dir / "trajectory.npz", allow_pickle=False)
                    state_spec = mujoco.mjtState(int(trajectory["state_spec"]))
                    frame_indices = np.arange(
                        0, len(trajectory["frame_state_indices"]), max(1, frame_stride), dtype=np.int64
                    )
                    if max_frames_per_episode is not None:
                        frame_indices = frame_indices[:max_frames_per_episode]
                    estimators = {
                        name: DLOPositionEstimator(
                            camera_specs[name]["intrinsics"],
                            sample_count=sample_count,
                            surface_to_center_offset_m=cable_radius_m,
                        )
                        for name in ("opst", "wrist")
                    }
                    tracker = TemporalDLOTracker(
                        sample_count=sample_count,
                        expected_length_m=expected_length_m,
                        min_initial_length_ratio=0.80,
                        observation_gain=1.0,
                        velocity_gain=0.15,
                        velocity_decay=0.80,
                        length_regularization_gain=0.30,
                        low_coverage_gain_scale=0.50,
                        rigid_transform_gain=0.0,
                    )
                    previous_state = None
                    previous_time = None

                    for frame_number in frame_indices:
                        state_index = int(trajectory["frame_state_indices"][frame_number])
                        mujoco.mj_setState(
                            env.model, env.data, trajectory["states"][state_index], state_spec
                        )
                        mujoco.mj_forward(env.model, env.data)
                        cable_world = env.data.xpos[env.cable_ids].copy()
                        camera_data = {}
                        estimator_proc_ms = 0.0
                        for name in ("opst", "wrist"):
                            spec = camera_specs[name]
                            rgb, depth, _ = _render_rgb_depth(renderer, env.data, spec["name"])
                            world_from_camera = world_from_camera_optical(
                                env.data.cam_xpos[spec["id"]], env.data.cam_xmat[spec["id"]]
                            )
                            camera_from_world = np.linalg.inv(world_from_camera)
                            target_camera = transform_points(camera_from_world, cable_world)
                            target_14 = resample_polyline(target_camera, sample_count)
                            target_pixels = project_points(target_14, spec["intrinsics"])
                            try:
                                estimate = estimators[name].estimate(rgb, depth)
                                estimator_proc_ms += float(estimate.timings_ms.get("total", 0.0))
                                observed_world = transform_points(world_from_camera, estimate.points_camera)
                                observed_length = float(
                                    np.linalg.norm(np.diff(observed_world, axis=0), axis=1).sum()
                                )
                                camera_data[name] = {
                                    "rgb": rgb,
                                    "estimate": estimate,
                                    "observed_world": observed_world,
                                    "observed_length": observed_length,
                                    "observed_pixels": project_points(estimate.points_camera, spec["intrinsics"]),
                                    "camera_from_world": camera_from_world,
                                    "intrinsics": spec["intrinsics"],
                                    "target_pixels": target_pixels,
                                }
                            except Exception:
                                camera_data[name] = {
                                    "rgb": rgb,
                                    "estimate": None,
                                    "observed_world": None,
                                    "observed_length": 0.0,
                                    "observed_pixels": None,
                                    "camera_from_world": camera_from_world,
                                    "intrinsics": spec["intrinsics"],
                                    "target_pixels": target_pixels,
                                }

                        now = float(env.data.time)
                        dt = 0.04 * max(1, frame_stride) if previous_time is None else max(1e-3, now - previous_time)
                        previous_time = now
                        previous_state = _joint_state(tracker, camera_data, previous_state, dt)
                        tracker_proc_ms = 0.0 if previous_state is None else float(
                            previous_state.timings_ms.get("total", 0.0)
                        )
                        proc_ms = estimator_proc_ms + tracker_proc_ms

                        panels = []
                        for name in ("opst", "wrist"):
                            item = camera_data[name]
                            predicted_pixels = None
                            error_cm = float("nan")
                            visible_error_cm = float("nan")
                            if previous_state is not None and previous_state.points_world is not None:
                                predicted_camera = transform_points(
                                    item["camera_from_world"], previous_state.points_world
                                )
                                predicted_pixels = project_points(predicted_camera, item["intrinsics"])
                                errors, reverse = reversal_invariant_errors(
                                    predicted_camera,
                                    resample_polyline(
                                        transform_points(item["camera_from_world"], cable_world),
                                        sample_count,
                                    ),
                                )
                                error_cm = float(np.mean(errors)) * 100.0
                                if item["estimate"] is not None:
                                    visible = _target_visible_from_mask(
                                        project_points(
                                            resample_polyline(
                                                transform_points(item["camera_from_world"], cable_world),
                                                sample_count,
                                            ),
                                            item["intrinsics"],
                                        ),
                                        item["estimate"].mask,
                                    )
                                    if reverse:
                                        visible = visible[::-1]
                                    if np.any(visible):
                                        visible_error_cm = float(np.mean(errors[visible])) * 100.0
                            panels.append(
                                _make_overlay(
                                    item["rgb"], item["target_pixels"], item["observed_pixels"],
                                    predicted_pixels, item["estimate"], previous_state,
                                    name, scenario_name, int(frame_number), error_cm,
                                    visible_error_cm, proc_ms,
                                )
                            )
                        separator = np.zeros((height, separator_width, 3), dtype=np.uint8)
                        frame = np.concatenate([panels[0], separator, panels[1]], axis=1)
                        writer.write(frame)
                        written += 1
                    trajectory.close()
            finally:
                renderer.close()
                env.close()
    finally:
        writer.release()
    return written


def parse_args():
    parser = argparse.ArgumentParser(description="Create a two-camera joint DLO diagnostic video.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--project-src", type=Path, default=DEFAULT_PROJECT_SRC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument("--cable-radius-m", type=float, default=0.014)
    parser.add_argument("--expected-length-m", type=float, default=0.78)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def main():
    args = parse_args()
    written = make_video(
        run_root=args.run_root.resolve(),
        project_src=args.project_src.resolve(),
        output=args.output.resolve(),
        scenarios=args.scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        frame_stride=args.frame_stride,
        max_frames_per_episode=args.max_frames_per_episode,
        sample_count=args.samples,
        cable_radius_m=args.cable_radius_m,
        expected_length_m=args.expected_length_m,
        fps=args.fps,
    )
    print("wrote {} frames to {}".format(written, args.output.resolve()))


if __name__ == "__main__":
    main()
