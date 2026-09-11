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
from dlo_position.temporal_tracker import TemporalDLOTracker
from dlo_position.temporal_benchmark import _target_visible_from_mask


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
    valid = np.isfinite(pixels).all(axis=1)
    points = np.rint(pixels[valid]).astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), radius, color, -1, cv2.LINE_AA)


def _draw_points(image, pixels, mask, color, radius=4):
    valid = np.isfinite(pixels).all(axis=1) & np.asarray(mask, dtype=bool)
    for point in np.rint(pixels[valid]).astype(np.int32):
        cv2.circle(image, tuple(point), radius, color, -1, cv2.LINE_AA)


def _text(image, value, y, color=(255, 255, 255)):
    cv2.putText(
        image,
        value,
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        2,
        cv2.LINE_AA,
    )


def _render_rgb_depth(renderer, data, camera_name):
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render().copy()
    return rgb, depth


def make_video(
    *,
    run_root,
    project_src,
    output,
    camera,
    scenarios,
    episodes_per_scenario,
    frame_stride,
    max_frames_per_episode,
    sample_count,
    cable_radius_m,
    surface_to_center_mode,
    adaptive_normal_residual_m,
    adaptive_normal_weight,
    expected_length_m,
    length_regularization_gain,
    low_coverage_gain_scale,
    fps,
):
    project_src = project_src.resolve()
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario

    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 480, 360
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
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
            seed = int(metadata["result"]["requested_seed"])
            config = env_config_for_scenario(
                get_scenario(scenario_name), seed=seed, episode_seconds=15.0
            )
            config.dynamicvla_cameras_enabled = True
            env = CableGraspEnv(config)
            renderer = mujoco.Renderer(env.model, height=height, width=width)
            try:
                if camera == "opst":
                    camera_id = int(env.dynamicvla_opst_camera_id)
                    camera_name = env.config.dynamicvla_opst_camera_name
                else:
                    camera_id = int(env.dynamicvla_wrist_camera_id)
                    camera_name = env.config.dynamicvla_wrist_camera_name
                intrinsics = camera_matrix(
                    width, height, float(env.model.cam_fovy[camera_id])
                )
                for episode_dir in episode_dirs:
                    trajectory = np.load(
                        episode_dir / "trajectory.npz", allow_pickle=False
                    )
                    state_spec = mujoco.mjtState(int(trajectory["state_spec"]))
                    frame_indices = np.arange(
                        0,
                        len(trajectory["frame_state_indices"]),
                        max(1, frame_stride),
                        dtype=np.int64,
                    )
                    if max_frames_per_episode is not None:
                        frame_indices = frame_indices[:max_frames_per_episode]
                    estimator = DLOPositionEstimator(
                        intrinsics,
                        sample_count=sample_count,
                        surface_to_center_offset_m=cable_radius_m,
                        surface_to_center_mode=surface_to_center_mode,
                        adaptive_normal_residual_m=adaptive_normal_residual_m,
                        adaptive_normal_weight=adaptive_normal_weight,
                    )
                    tracker = TemporalDLOTracker(
                        sample_count=sample_count,
                        expected_length_m=expected_length_m,
                        length_regularization_gain=length_regularization_gain,
                        low_coverage_gain_scale=low_coverage_gain_scale,
                    )
                    last_time = None
                    for frame_number in frame_indices:
                        state_index = int(
                            trajectory["frame_state_indices"][frame_number]
                        )
                        mujoco.mj_setState(
                            env.model,
                            env.data,
                            trajectory["states"][state_index],
                            state_spec,
                        )
                        mujoco.mj_forward(env.model, env.data)
                        cable_world = env.data.xpos[env.cable_ids].copy()
                        rgb, depth = _render_rgb_depth(renderer, env.data, camera_name)
                        world_from_camera = world_from_camera_optical(
                            env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
                        )
                        camera_from_world = np.linalg.inv(world_from_camera)
                        target_camera = transform_points(camera_from_world, cable_world)
                        target_14 = resample_polyline(target_camera, sample_count)
                        target_pixels = project_points(target_14, intrinsics)
                        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        _draw_curve(overlay, target_pixels, (0, 220, 0), 2, 3)
                        try:
                            estimate = estimator.estimate(rgb, depth)
                            observed_world = transform_points(
                                world_from_camera, estimate.points_camera
                            )
                            current_time = float(env.data.time)
                            dt = (
                                0.04 * max(1, frame_stride)
                                if last_time is None
                                else max(1e-3, current_time - last_time)
                            )
                            last_time = current_time
                            observed_length = float(
                                np.linalg.norm(np.diff(observed_world, axis=0), axis=1).sum()
                            )
                            tracked = tracker.update(
                                observed_world,
                                observed_length_m=observed_length,
                                dt_s=dt,
                            )
                            observed_pixels = project_points(estimate.points_camera, intrinsics)
                            _draw_curve(overlay, observed_pixels, (255, 180, 0), 1, 2)
                            if tracked.points_world is None:
                                _text(overlay, "tracker warming up", 20, (0, 165, 255))
                                _text(overlay, "green=GT  blue=visible observation", 40)
                                _text(overlay, "coverage=0.00  confidence=0.00", 60)
                            else:
                                predicted_camera = transform_points(
                                    camera_from_world, tracked.points_world
                                )
                                predicted_pixels = project_points(
                                    predicted_camera, intrinsics
                                )
                                _draw_curve(overlay, predicted_pixels, (0, 0, 255), 2, 3)
                                hidden_mask = ~tracked.observed_mask
                                _draw_points(
                                    overlay,
                                    predicted_pixels,
                                    hidden_mask,
                                    (255, 0, 255),
                                    4,
                                )
                                # The cable has no intrinsic direction.  Use
                                # the same reversal-invariant correspondence
                                # as the benchmark; otherwise a perfectly
                                # overlapping curve with the opposite
                                # skeleton ordering is reported as a large
                                # point error in the video.
                                errors, reverse = reversal_invariant_errors(
                                    predicted_camera, target_14
                                )
                                if reverse:
                                    predicted_pixels = predicted_pixels[::-1]
                                    hidden_mask = hidden_mask[::-1]
                                image_visible = _target_visible_from_mask(
                                    target_pixels, estimate.mask
                                )
                                if reverse:
                                    image_visible = image_visible[::-1]
                                image_visible_error = (
                                    float(np.mean(errors[image_visible]))
                                    if np.any(image_visible)
                                    else float("nan")
                                )
                                _text(
                                    overlay,
                                    "green=GT  blue=visible  red=14pt state  magenta=hidden",
                                    20,
                                )
                                _text(
                                    overlay,
                                    "err={:.1f}cm  visible={:.1f}cm  coverage={:.2f}  conf={:.2f}".format(
                                        float(np.mean(errors)) * 100.0,
                                        image_visible_error * 100.0,
                                        tracked.coverage,
                                        tracked.confidence,
                                    ),
                                    40,
                                )
                                _text(
                                    overlay,
                                    "{}  {}  frame={}".format(
                                        camera, scenario_name, int(frame_number)
                                    ),
                                    60,
                                )
                        except Exception as exc:
                            _text(overlay, "estimation failed: {}".format(type(exc).__name__), 20, (0, 0, 255))
                        writer.write(overlay)
                        written += 1
                    trajectory.close()
            finally:
                renderer.close()
                env.close()
    finally:
        writer.release()
    return written


def parse_args():
    parser = argparse.ArgumentParser(description="Create a temporal DLO completion overlay video.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--project-src", type=Path, default=DEFAULT_PROJECT_SRC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", choices=["opst", "wrist"], default="opst")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-frames-per-episode", type=int, default=90)
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument(
        "--cable-radius-m",
        type=float,
        default=0.020,
        help="effective RGB-D surface-to-centerline correction in metres",
    )
    parser.add_argument(
        "--surface-to-center-mode",
        choices=["ray", "normal", "adaptive"],
        default="adaptive",
    )
    parser.add_argument("--adaptive-normal-residual-m", type=float, default=0.006)
    parser.add_argument("--adaptive-normal-weight", type=float, default=0.50)
    parser.add_argument("--expected-length-m", type=float, default=0.78)
    parser.add_argument("--length-regularization-gain", type=float, default=0.30)
    parser.add_argument("--low-coverage-gain-scale", type=float, default=0.50)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def main():
    args = parse_args()
    written = make_video(
        run_root=args.run_root.resolve(),
        project_src=args.project_src.resolve(),
        output=args.output.resolve(),
        camera=args.camera,
        scenarios=args.scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        frame_stride=args.frame_stride,
        max_frames_per_episode=args.max_frames_per_episode,
        sample_count=args.samples,
        cable_radius_m=args.cable_radius_m,
        surface_to_center_mode=args.surface_to_center_mode,
        adaptive_normal_residual_m=args.adaptive_normal_residual_m,
        adaptive_normal_weight=args.adaptive_normal_weight,
        expected_length_m=args.expected_length_m,
        length_regularization_gain=args.length_regularization_gain,
        low_coverage_gain_scale=args.low_coverage_gain_scale,
        fps=args.fps,
    )
    print("wrote {} frames to {}".format(written, args.output.resolve()))


if __name__ == "__main__":
    main()
