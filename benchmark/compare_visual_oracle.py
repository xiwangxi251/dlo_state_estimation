"""Compare visual and simulator DLO inputs during PPO rollouts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(os.environ.get(
    "DLO_PROJECT_ROOT", r"C:\Users\27642\Desktop\dynamic_cable"
))
PANDA_SRC = (
    PROJECT_ROOT / "src"
    if (PROJECT_ROOT / "src" / "panda_cable_grasp").is_dir()
    else PROJECT_ROOT / "panda_cable_grasp" / "src"
)
TRACKDLO_ROOT = Path(os.environ.get(
    "DLO_TRACKDLO_ROOT", str(PROJECT_ROOT / "trackdlo_standalone")
))
BENCHMARK_ROOT = Path(os.environ.get(
    "DLO_BENCHMARK_ROOT", str(PROJECT_ROOT / "dlo_position_benchmark")
))
for _path in (PANDA_SRC, TRACKDLO_ROOT / "src", BENCHMARK_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
from panda_cable_grasp.rl.environment import RLCableGraspEnv
from panda_cable_grasp.scenarios.registry import get_scenario
from run_ppo_visual import VisualRLCableGraspEnv
from run_scripted_vision import camera_matrix, world_from_camera_optical
from dlo_position.geometry import project_points, transform_points
from stable_baselines3 import PPO


SCENARIOS = (
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
)


def _segmentation(renderer, data, camera_name: str) -> np.ndarray:
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


def _state_pair(env: VisualRLCableGraspEnv):
    true_pos, true_vel = env._sample_cable_state()
    visual_pos = env._arc_length_samples(
        env._vision_nodes, env._vision_nodes, env.rl_config.cable_sample_count
    )
    visual_vel_nodes = (
        np.zeros_like(env._vision_nodes)
        if env._vision_velocity is None else env._vision_velocity
    )
    visual_vel = env._arc_length_samples(
        env._vision_nodes, visual_vel_nodes, env.rl_config.cable_sample_count
    )
    return true_pos, true_vel, visual_pos, visual_vel


def _image_visible_mask(env: VisualRLCableGraspEnv, true_pos: np.ndarray) -> np.ndarray:
    """MuJoCo segmentation oracle: visible if cable geometry is seen by either camera."""
    base = env.base_env
    cable_geom_ids = {int(value) for value in base.cable_geom_ids}
    robot_body_ids = set(range(1, 12))
    robot_geom_ids = {
        int(index)
        for index, body_id in enumerate(base.model.geom_bodyid)
        if int(body_id) in robot_body_ids
    }
    visible_union = np.zeros(len(true_pos), dtype=bool)
    for camera, camera_id, camera_name, intrinsics in (
        ("opst", env._vision.opst_id, env._vision.opst_name, env._vision.opst_k),
        ("wrist", env._vision.wrist_id, env._vision.wrist_name, env._vision.wrist_k),
    ):
        world_from_cam = world_from_camera_optical(
            env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
        )
        camera_points = transform_points(np.linalg.inv(world_from_cam), true_pos)
        pixels = project_points(camera_points, intrinsics)
        segmentation = _segmentation(env._vision_renderer, env.data, camera_name)
        in_frame, _, foreground_occluded = _visible_labels(
            pixels, segmentation, cable_geom_ids, robot_geom_ids
        )
        visible_union |= in_frame & ~foreground_occluded
    return visible_union


def run_one(
    checkpoint: str,
    scenario_name: str,
    seed: int,
    period: int,
    zero_action: bool = False,
    oracle_action: bool = False,
) -> dict:
    scenario = get_scenario(scenario_name)
    config = replace(
        env_config_for_scenario(scenario, seed=seed, episode_seconds=15.0),
        dynamicvla_cameras_enabled=True,
    )
    env = VisualRLCableGraspEnv(config, vision_period_steps=period)
    model = None if zero_action else PPO.load(checkpoint, device="cpu")
    position_errors = []
    velocity_errors = []
    true_velocity_norms = []
    update_position_errors = []
    update_velocity_errors = []
    oracle_obs_errors = []
    update_flags = []
    visible_position_errors = []
    occluded_position_errors = []
    visible_counts = 0
    occluded_counts = 0
    try:
        obs, _ = env.reset(seed=seed)
        control_dt = float(env.base_env.model.opt.timestep * max(1, env.base_env.config.frame_skip))
        max_steps = int(round(config.episode_seconds / control_dt)) + 2

        def measure(is_update: bool):
            tp, tv, vp, vv = _state_pair(env)
            pe = np.linalg.norm(vp - tp, axis=1)
            ve = np.linalg.norm(vv - tv, axis=1)
            position_errors.extend(pe.tolist())
            velocity_errors.extend(ve.tolist())
            true_velocity_norms.extend(np.linalg.norm(tv, axis=1).tolist())
            oracle = RLCableGraspEnv._observation(env)
            oracle_obs_errors.append(float(np.sqrt(np.mean((obs[:84] - oracle[:84]) ** 2))))
            update_flags.append(bool(is_update))
            visible = _image_visible_mask(env, tp)
            visible_position_errors.extend(pe[visible].tolist())
            occluded_position_errors.extend(pe[~visible].tolist())
            if is_update:
                update_position_errors.extend(pe.tolist())
                update_velocity_errors.extend(ve.tolist())

        measure(True)
        success = False
        for step in range(1, max_steps + 1):
            if zero_action:
                action = np.zeros(5, dtype=np.float32)
            else:
                # In oracle-action mode, the policy controls the same physical
                # rollout using simulator DLO state, while TrackDLO still runs
                # on every vision update for an apples-to-apples perception
                # comparison along a realistic moving trajectory.
                action_obs = RLCableGraspEnv._observation(env) if oracle_action else obs
                action, _ = model.predict(action_obs, deterministic=True)
            obs, _, success, truncated, _ = env.step(action)
            measure(step % period == 0)
            if success or truncated:
                break
        all_true = np.asarray(true_velocity_norms, dtype=float)
        all_vel = np.asarray(velocity_errors, dtype=float)
        return {
            "scenario": scenario_name,
            "seed": seed,
            "steps": step,
            "success": bool(success),
            "zero_action": bool(zero_action),
            "oracle_action": bool(oracle_action),
            "position_rmse_mm": float(np.sqrt(np.mean(np.square(position_errors))) * 1000.0),
            "position_mae_mm": float(np.mean(position_errors) * 1000.0),
            "position_p95_mm": float(np.percentile(position_errors, 95.0) * 1000.0),
            "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(velocity_errors)))),
            "velocity_mae_mps": float(np.mean(velocity_errors)),
            "velocity_p95_mps": float(np.percentile(velocity_errors, 95.0)),
            "true_velocity_rms_mps": float(np.sqrt(np.mean(np.square(all_true)))),
            "velocity_relative_rmse": float(np.sqrt(np.mean(np.square(all_vel))) / max(np.sqrt(np.mean(np.square(all_true))), 1e-6)),
            "update_position_rmse_mm": float(np.sqrt(np.mean(np.square(update_position_errors))) * 1000.0),
            "update_velocity_rmse_mps": float(np.sqrt(np.mean(np.square(update_velocity_errors)))),
            "oracle_dlo_obs_rmse": float(np.mean(oracle_obs_errors)),
            "visual_updates": int(env._vision.update_count),
            "visual_failures": int(env._vision.failure_count),
            "visible_node_count": int(visible_counts + len(visible_position_errors)),
            "occluded_node_count": int(occluded_counts + len(occluded_position_errors)),
            "visible_position_rmse_mm": float(np.sqrt(np.mean(np.square(visible_position_errors))) * 1000.0) if visible_position_errors else float("nan"),
            "occluded_position_rmse_mm": float(np.sqrt(np.mean(np.square(occluded_position_errors))) * 1000.0) if occluded_position_errors else float("nan"),
            "visible_position_mae_mm": float(np.mean(visible_position_errors) * 1000.0) if visible_position_errors else float("nan"),
            "occluded_position_mae_mm": float(np.mean(occluded_position_errors) * 1000.0) if occluded_position_errors else float("nan"),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=20260906)
    parser.add_argument("--vision-period-steps", type=int, default=10)
    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument(
        "--oracle-action",
        action="store_true",
        help="drive the robot with simulator DLO state while measuring visual state on the same rollout",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = [(s, args.seed_base + i) for s in SCENARIOS for i in range(args.episodes)]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                run_one,
                args.checkpoint,
                scenario,
                seed,
                args.vision_period_steps,
                args.zero_action,
                args.oracle_action,
            ): (scenario, seed)
            for scenario, seed in jobs
        }
        for future in as_completed(futures):
            scenario, seed = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"scenario": scenario, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
