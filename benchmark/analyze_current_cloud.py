from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np


RUN_ROOT = Path(r"C:\Users\27642\Desktop\dynamic_cable\linux_log\expert_grasp_fix_4x50\run_20260824_113325")
PROJECT_SRC = Path(r"C:\Users\27642\Desktop\dynamic_cable\panda_cable_grasp\src")
TRACK_SRC = Path(r"C:\Users\27642\Desktop\dynamic_cable\trackdlo_standalone\src")


def camera_matrix(width: int, height: int, fovy: float) -> np.ndarray:
    focal = 0.5 * height / np.tan(np.deg2rad(fovy) * 0.5)
    return np.array([[focal, 0.0, (width - 1) * 0.5], [0.0, focal, (height - 1) * 0.5], [0.0, 0.0, 1.0]], dtype=float)


def world_from_camera_optical(position, rotation):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation).reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = position
    return transform


def transform(points, matrix):
    points = np.asarray(points, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="id_shape_nominal_current")
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(TRACK_SRC))
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario
    from trackdlo_standalone.geometry import backproject_mask, depth_to_meters, voxel_downsample
    from trackdlo_standalone.initialization import segment_hsv

    episode = RUN_ROOT / "episodes" / "expert" / args.scenario / "seed_20280804"
    metadata = __import__("json").loads((episode / "episode.json").read_text(encoding="utf-8"))
    config = env_config_for_scenario(get_scenario(args.scenario), seed=int(metadata["result"]["requested_seed"]), episode_seconds=15.0)
    config.dynamicvla_cameras_enabled = True
    env = CableGraspEnv(config)
    renderer = mujoco.Renderer(env.model, height=360, width=480)
    trajectory = np.load(episode / "trajectory.npz", allow_pickle=False)
    state_spec = mujoco.mjtState(int(trajectory["state_spec"]))
    state_index = int(trajectory["frame_state_indices"][args.frame])
    mujoco.mj_setState(env.model, env.data, trajectory["states"][state_index], state_spec)
    mujoco.mj_forward(env.model, env.data)
    all_clouds = []
    for camera, video_name, camera_id, camera_name in (
        ("opst", "global.mp4", int(env.dynamicvla_opst_camera_id), env.config.dynamicvla_opst_camera_name),
        ("wrist", "wrist.mp4", int(env.dynamicvla_wrist_camera_id), env.config.dynamicvla_wrist_camera_name),
    ):
        cap = cv2.VideoCapture(str(episode / video_name))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"could not read {video_name}")
        renderer.enable_depth_rendering()
        renderer.update_scene(env.data, camera=camera_name)
        depth = renderer.render().copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = segment_hsv(rgb, (112, 180, 80), (130, 255, 255))
        intrinsics = camera_matrix(480, 360, float(env.model.cam_fovy[camera_id]))
        points = backproject_mask(depth_to_meters(depth), mask, intrinsics)
        points = voxel_downsample(points, 0.004)
        wfc = world_from_camera_optical(env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id])
        world_points = transform(points, wfc)
        all_clouds.append(world_points)
        print(camera, "mask", int(mask.sum()), "cloud", len(points), "z", np.nanmin(points[:, 2]) if len(points) else None, np.nanmax(points[:, 2]) if len(points) else None)
    cloud = np.concatenate(all_clouds, axis=0) if all_clouds else np.empty((0, 3))
    truth = env.data.xpos[env.cable_ids].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, cloud_world=cloud, truth_world=truth)
    print("saved", args.output, "cloud", len(cloud), "truth", len(truth))


if __name__ == "__main__":
    main()
