"""Save current-frame reconstruction overlays for a few diagnostic frames."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get("DLO_PROJECT_ROOT", repo_root.parent))
    scenario = "id_combined_l1_nominal"
    targets = {20, 60, 80, 100, 120, 160, 220, 300}
    run_root = Path(os.environ.get("DLO_RUN_ROOT", repo_root / "data" / "recorded_run"))
    episode = run_root / "episodes" / "expert" / scenario / "seed_20280804"
    sys.path.insert(0, str(root / "panda_cable_grasp" / "src"))
    sys.path.insert(0, str(repo_root / "trackdlo_standalone" / "src"))
    sys.path.insert(0, str(repo_root / "benchmark"))
    from dlo_position.recorded_benchmark import camera_matrix, world_from_camera_optical
    from trackdlo_standalone.current_frame import (
        CurrentFrameReconstructionConfig,
        extract_current_component_paths,
        reconstruct_current_polyline,
    )
    from trackdlo_standalone.geometry import depth_to_meters, project_camera_points, resample_polyline
    from trackdlo_standalone.initialization import segment_hsv
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario

    metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    config = env_config_for_scenario(
        get_scenario(scenario),
        seed=int(metadata["result"]["requested_seed"]),
        episode_seconds=15.0,
    )
    config.dynamicvla_cameras_enabled = True
    env = CableGraspEnv(config)
    renderer = mujoco.Renderer(env.model, height=360, width=480)
    camera_id = int(env.dynamicvla_opst_camera_id)
    camera_name = env.config.dynamicvla_opst_camera_name
    intrinsics = camera_matrix(480, 360, float(env.model.cam_fovy[camera_id]))
    trajectory = np.load(episode / "trajectory.npz", allow_pickle=False)
    spec = mujoco.mjtState(int(trajectory["state_spec"]))
    capture = cv2.VideoCapture(str(episode / "global.mp4"))
    output = repo_root / "results" / "current_frame_debug"
    output.mkdir(parents=True, exist_ok=True)
    for frame_index in range(max(targets) + 1):
        ok, bgr = capture.read()
        if not ok:
            break
        if frame_index not in targets:
            continue
        state_index = int(trajectory["frame_state_indices"][frame_index])
        mujoco.mj_setState(env.model, env.data, trajectory["states"][state_index], spec)
        mujoco.mj_forward(env.model, env.data)
        renderer.enable_depth_rendering()
        renderer.update_scene(env.data, camera=camera_name)
        depth = depth_to_meters(renderer.render().copy())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = segment_hsv(rgb, (112, 180, 80), (130, 255, 255))
        paths = extract_current_component_paths(mask, depth, intrinsics, surface_offset_m=0.014)
        current, info = reconstruct_current_polyline(
            paths, target_length_m=0.78, config=CurrentFrameReconstructionConfig()
        )
        canvas = bgr.copy()
        if current is not None:
            current = resample_polyline(current, 45)
            px = project_camera_points(current, intrinsics)
            for x, y in np.rint(px).astype(int):
                if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                    cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)
        world_from_camera = world_from_camera_optical(
            env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
        )
        truth = resample_polyline(
            transform_camera(env.data.xpos[env.cable_ids].copy(), world_from_camera), 45
        )
        px = project_camera_points(truth, intrinsics)
        for x, y in np.rint(px).astype(int):
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            f"current-frame green / truth red  frame={frame_index} paths={int(info.get('path_count', 0))}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output / f"{scenario}_frame_{frame_index:04d}.png"), canvas)
    capture.release()


def transform_camera(points_world: np.ndarray, world_from_camera: np.ndarray) -> np.ndarray:
    return points_world @ np.linalg.inv(world_from_camera)[:3, :3].T + np.linalg.inv(world_from_camera)[:3, 3]


if __name__ == "__main__":
    main()
