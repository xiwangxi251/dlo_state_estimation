from __future__ import annotations

import argparse
import os
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path(
    os.environ.get("DLO_RUN_ROOT", REPO_ROOT / "data" / "recorded_run")
)
DEFAULT_PROJECT_SRC = Path(
    os.environ.get("PANDA_CABLE_GRASP_SRC", REPO_ROOT.parent / "panda_cable_grasp" / "src")
)
DEFAULT_SCENARIOS = [
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
]


def _draw_curve(
    image: np.ndarray,
    pixels: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    radius: int,
) -> None:
    valid = np.isfinite(pixels).all(axis=1)
    points = np.rint(pixels[valid]).astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), radius, color, -1, cv2.LINE_AA)


def _text(image: np.ndarray, value: str, y: int, color=(255, 255, 255)) -> None:
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


def _render_rgb_depth(renderer, data, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render().copy()
    return rgb, depth


def make_video(
    *,
    run_root: Path,
    project_src: Path,
    output: Path,
    camera: str,
    scenarios: list[str],
    episodes_per_scenario: int,
    frame_stride: int,
    max_frames_per_episode: int | None,
    sample_count: int,
    cable_radius_m: float,
    fps: float,
) -> int:
    project_src = project_src.resolve()
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario

    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 480, 360
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {output}")

    written = 0
    try:
        for scenario_name in scenarios:
            episode_dirs = choose_evenly_spaced(
                discover_episode_dirs(run_root, scenario_name), episodes_per_scenario
            )
            if not episode_dirs:
                continue
            scenario = get_scenario(scenario_name)
            config = env_config_for_scenario(
                scenario, seed=20280804, episode_seconds=15.0
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
                estimator = DLOPositionEstimator(
                    intrinsics,
                    sample_count=sample_count,
                    surface_to_center_offset_m=cable_radius_m,
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
                    estimator.reset()
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
                        target_camera = transform_points(
                            np.linalg.inv(world_from_camera), cable_world
                        )
                        target_14 = resample_polyline(target_camera, sample_count)
                        target_pixels = project_points(target_14, intrinsics)
                        visible = (
                            np.isfinite(target_pixels).all(axis=1)
                            & (target_14[:, 2] > 1e-8)
                            & (target_pixels[:, 0] >= 0)
                            & (target_pixels[:, 0] < width)
                            & (target_pixels[:, 1] >= 0)
                            & (target_pixels[:, 1] < height)
                        )
                        visible_fraction = float(np.mean(visible))
                        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        _draw_curve(overlay, target_pixels, (0, 220, 0), 2, 3)
                        try:
                            estimate = estimator.estimate(rgb, depth)
                            errors, reverse = reversal_invariant_errors(
                                estimate.points_camera, target_14
                            )
                            predicted_pixels = project_points(
                                estimate.points_camera, intrinsics
                            )
                            _draw_curve(
                                overlay, predicted_pixels, (0, 0, 255), 2, 3
                            )
                            predicted_length = float(
                                np.sum(
                                    np.linalg.norm(
                                        np.diff(estimate.points_camera, axis=0), axis=1
                                    )
                                )
                            )
                            target_length = float(
                                np.sum(
                                    np.linalg.norm(
                                        np.diff(target_camera, axis=0), axis=1
                                    )
                                )
                            )
                            length_ratio = predicted_length / max(target_length, 1e-9)
                            _text(
                                overlay,
                                f"{camera}  {scenario_name}  frame={int(frame_number)}",
                                20,
                            )
                            _text(
                                overlay,
                                f"green=GT  red=prediction  err={np.mean(errors)*100:.1f}cm",
                                40,
                            )
                            _text(
                                overlay,
                                f"length={length_ratio:.2f}x  GT-in-frame={visible_fraction:.2f}  time={estimate.timings_ms['total']:.1f}ms",
                                60,
                            )
                            if reverse:
                                _text(overlay, "endpoint order reversed for comparison", 80)
                        except Exception as exc:
                            _text(
                                overlay,
                                f"{camera}  {scenario_name}  frame={int(frame_number)}",
                                20,
                            )
                            _text(overlay, f"estimation failed: {type(exc).__name__}", 42, (0, 0, 255))
                            _text(overlay, f"GT-in-frame={visible_fraction:.2f}", 64)
                        writer.write(overlay)
                        written += 1
                    trajectory.close()
            finally:
                renderer.close()
                env.close()
    finally:
        writer.release()
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an RGB-D DLO overlay preview video.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--project-src", type=Path, default=DEFAULT_PROJECT_SRC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", choices=["opst", "wrist"], required=True)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-frames-per-episode", type=int, default=90)
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument("--cable-radius-m", type=float, default=0.014)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
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
        fps=args.fps,
    )
    print(f"wrote {written} frames to {args.output.resolve()}")


if __name__ == "__main__":
    main()
