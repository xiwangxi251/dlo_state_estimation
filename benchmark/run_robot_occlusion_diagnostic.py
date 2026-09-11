"""Measure robot-vs-cable visibility using MuJoCo's segmentation buffer.

This is an evaluation oracle for the recorded simulation, not an input to the
RGB-D estimator.  It answers how often a projected GT DLO node is hidden by a
robot geometry (as opposed to another cable segment) and therefore sizes the
robot-mask part of the follow-up work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import mujoco
import numpy as np

from dlo_position.geometry import project_points, resample_polyline, transform_points
from dlo_position.recorded_benchmark import (
    camera_matrix,
    choose_evenly_spaced,
    discover_episode_dirs,
    world_from_camera_optical,
)


DEFAULT_RUN_ROOT = Path(
    r"C:\Users\27642\Desktop\dynamic_cable\linux_log\expert_grasp_fix_4x50\run_20260824_113325"
)
DEFAULT_PROJECT_SRC = Path(r"C:\Users\27642\Desktop\dynamic_cable\panda_cable_grasp\src")
DEFAULT_SCENARIOS = [
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
]


def _segmentation(renderer: mujoco.Renderer, data, camera_name: str) -> np.ndarray:
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name)
    result = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    return result


def _visible_labels(
    pixels: np.ndarray,
    segmentation: np.ndarray,
    cable_geom_ids: set[int],
    robot_geom_ids: set[int],
    radius_px: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return in-frame, robot-occluded and any-foreground-occluded node masks."""

    height, width = segmentation.shape[:2]
    in_frame = np.isfinite(pixels).all(axis=1)
    robot_occluded = np.zeros(len(pixels), dtype=bool)
    foreground_occluded = np.zeros(len(pixels), dtype=bool)
    geom_ids = np.asarray(segmentation[..., 0], dtype=np.int64)
    for index, pixel in enumerate(pixels):
        if not in_frame[index]:
            continue
        col, row = np.rint(pixel).astype(int)
        if not (0 <= row < height and 0 <= col < width):
            in_frame[index] = False
            continue
        patch = geom_ids[
            max(0, row - radius_px) : min(height, row + radius_px + 1),
            max(0, col - radius_px) : min(width, col + radius_px + 1),
        ].reshape(-1)
        patch = patch[patch >= 0]
        cable_hit = any(int(value) in cable_geom_ids for value in patch)
        robot_hit = any(int(value) in robot_geom_ids for value in patch)
        foreground_hit = any(int(value) not in cable_geom_ids for value in patch)
        robot_occluded[index] = not cable_hit and robot_hit
        foreground_occluded[index] = not cable_hit and foreground_hit
    return in_frame, robot_occluded, foreground_occluded


def run_diagnostic(
    *,
    run_root: Path,
    project_src: Path,
    output: Path,
    scenarios: list[str],
    camera: str,
    episodes_per_scenario: int | None,
    frame_stride: int,
    max_frames_per_episode: int | None,
    sample_count: int,
) -> dict:
    project_src = project_src.resolve()
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario_name in scenarios:
        episode_dirs = choose_evenly_spaced(
            discover_episode_dirs(run_root, scenario_name),
            episodes_per_scenario,
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
            if camera == "opst":
                camera_id = int(env.dynamicvla_opst_camera_id)
                camera_name = env.config.dynamicvla_opst_camera_name
            else:
                camera_id = int(env.dynamicvla_wrist_camera_id)
                camera_name = env.config.dynamicvla_wrist_camera_name
            intrinsics = camera_matrix(480, 360, float(env.model.cam_fovy[camera_id]))
            cable_geom_ids = {int(value) for value in env.cable_geom_ids}
            robot_body_ids = set(range(1, 12))
            robot_geom_ids = {
                int(index)
                for index, body_id in enumerate(env.model.geom_bodyid)
                if int(body_id) in robot_body_ids
            }
            attempted = successful = robot_nodes = foreground_nodes = in_frame_nodes = 0
            robot_frames = foreground_frames = 0
            seg_time_ms = []
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
                for frame_number in frame_indices:
                    attempted += 1
                    state_index = int(trajectory["frame_state_indices"][frame_number])
                    mujoco.mj_setState(
                        env.model, env.data, trajectory["states"][state_index], state_spec
                    )
                    mujoco.mj_forward(env.model, env.data)
                    cable_world = env.data.xpos[env.cable_ids].copy()
                    world_from_camera = world_from_camera_optical(
                        env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
                    )
                    camera_from_world = np.linalg.inv(world_from_camera)
                    target_camera = transform_points(camera_from_world, cable_world)
                    target_14 = resample_polyline(target_camera, sample_count)
                    target_pixels = project_points(target_14, intrinsics)
                    start = perf_counter_ns()
                    segmentation = _segmentation(renderer, env.data, camera_name)
                    seg_time_ms.append((perf_counter_ns() - start) * 1e-6)
                    in_frame, robot_mask, foreground_mask = _visible_labels(
                        target_pixels,
                        segmentation,
                        cable_geom_ids,
                        robot_geom_ids,
                    )
                    in_frame_nodes += int(np.count_nonzero(in_frame))
                    robot_nodes += int(np.count_nonzero(robot_mask & in_frame))
                    foreground_nodes += int(np.count_nonzero(foreground_mask & in_frame))
                    robot_frames += int(np.any(robot_mask & in_frame))
                    foreground_frames += int(np.any(foreground_mask & in_frame))
                    successful += 1
                trajectory.close()
        finally:
            renderer.close()
            env.close()
        rows.append(
            {
                "camera": camera,
                "scenario": scenario_name,
                "frames": attempted,
                "successful": successful,
                "mean_in_frame_nodes": in_frame_nodes / max(successful, 1),
                "robot_occluded_node_fraction": robot_nodes / max(in_frame_nodes, 1),
                "foreground_occluded_node_fraction": foreground_nodes / max(in_frame_nodes, 1),
                "frames_with_robot_occlusion": robot_frames / max(successful, 1),
                "frames_with_foreground_occlusion": foreground_frames / max(successful, 1),
                "segmentation_mean_ms": float(np.mean(seg_time_ms)) if seg_time_ms else float("nan"),
                "segmentation_p95_ms": float(np.percentile(seg_time_ms, 95)) if seg_time_ms else float("nan"),
            }
        )
    result = {
        "run_root": str(run_root.resolve()),
        "camera": camera,
        "sample_count": sample_count,
        "frame_stride": frame_stride,
        "episodes_per_scenario": episodes_per_scenario,
        "rows": rows,
        "note": "MuJoCo segmentation oracle; not supplied to the RGB-D estimator.",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--project-src", type=Path, default=DEFAULT_PROJECT_SRC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", choices=["opst", "wrist"], default="opst")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--samples", type=int, default=14)
    args = parser.parse_args()
    print(
        json.dumps(
            run_diagnostic(
                run_root=args.run_root.resolve(),
                project_src=args.project_src.resolve(),
                output=args.output.resolve(),
                scenarios=args.scenarios,
                camera=args.camera,
                episodes_per_scenario=args.episodes_per_scenario,
                frame_stride=args.frame_stride,
                max_frames_per_episode=args.max_frames_per_episode,
                sample_count=args.samples,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
