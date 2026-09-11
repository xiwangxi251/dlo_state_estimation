"""Run the Panda scripted grasp policy from visual TrackDLO state.

The environment is still responsible for physics/contact-based success
qualification.  The scripted policy receives only the estimated DLO state for
target position, target velocity, and material-segment locking; it does not
read ``env.target_position()``, ``env.target_velocity()``, or cable positions
for control.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np


# Allow the same vision adapter to run on the local Windows checkout and on
# the Linux jump151 checkout.  The default assumes this repository sits next
# to panda_cable_grasp; remote evaluators can set DLO_PROJECT_ROOT explicitly.
REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get(
    "DLO_PROJECT_ROOT", str(REPO_ROOT.parent)
))
# The local checkout keeps the package under panda_cable_grasp/src, whereas
# jump151 uses the conventional repository-level src/ layout.
PANDA_SRC = (
    PROJECT_ROOT / "src"
    if (PROJECT_ROOT / "src" / "panda_cable_grasp").is_dir()
    else PROJECT_ROOT / "panda_cable_grasp" / "src"
)
TRACKDLO_ROOT = Path(os.environ.get(
    "DLO_TRACKDLO_ROOT", str(REPO_ROOT / "trackdlo_standalone")
))
BENCHMARK_ROOT = Path(os.environ.get(
    "DLO_BENCHMARK_ROOT", str(REPO_ROOT / "benchmark")
))
TRACKDLO_SRC = TRACKDLO_ROOT / "src"
for _path in (PANDA_SRC, TRACKDLO_SRC, BENCHMARK_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from panda_cable_grasp.env.environment import CableGraspEnv
from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig
from panda_cable_grasp.scenarios.registry import get_scenario
from dlo_position.geometry import transform_points
from trackdlo_standalone import TrackDLOConfig, TrackDLOTracker
from trackdlo_standalone.geometry import backproject_mask, depth_to_meters, voxel_downsample
from trackdlo_standalone.initialization import segment_hsv


def camera_matrix(width: int, height: int, fovy_degrees: float) -> np.ndarray:
    """Pinhole intrinsics matching the recorded benchmark renderer."""
    focal = 0.5 * height / np.tan(np.deg2rad(fovy_degrees) * 0.5)
    return np.array(
        [[focal, 0.0, (width - 1) * 0.5],
         [0.0, focal, (height - 1) * 0.5],
         [0.0, 0.0, 1.0]], dtype=np.float64,
    )


def world_from_camera_optical(
    camera_position: np.ndarray, camera_rotation: np.ndarray
) -> np.ndarray:
    """Return the world transform of MuJoCo's optical camera frame."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(camera_rotation, dtype=np.float64).reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = np.asarray(camera_position, dtype=np.float64)
    return transform


SCENARIOS = [
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
]


def _render_depth(renderer: mujoco.Renderer, env: CableGraspEnv, camera_name: str) -> np.ndarray:
    renderer.enable_depth_rendering()
    renderer.update_scene(env.data, camera=camera_name)
    return renderer.render().copy()


def _track_config() -> TrackDLOConfig:
    """The selected adaptive dual-camera TrackDLO configuration."""

    # Training/evaluation launchers can select the computational budget without
    # changing the benchmark's algorithmic defaults.  The established real-time
    # setting is a 100-point cloud and 10 CPD iterations; diagnostics can leave
    # these variables unset to retain the full cloud and 50-iteration default.
    max_iter = max(int(os.environ.get("DLO_TRACKDLO_MAX_ITER", "50")), 1)
    max_observed_points = max(
        int(os.environ.get("DLO_TRACKDLO_MAX_OBSERVED_POINTS", "0")), 0
    )

    return TrackDLOConfig(
        hsv_lower=(112, 180, 80),
        hsv_upper=(130, 255, 255),
        visibility_mode="neighborhood",
        visibility_threshold=0.008,
        visibility_weak_threshold=0.012,
        visibility_neighborhood_radius=0.015,
        visibility_min_neighbors=3,
        visibility_mask_radius_px=8,
        visibility_min_mask_pixels=3,
        alpha=3.0,
        max_iter=max_iter,
        dlo_pixel_width=6,
        downsample_leaf_size=0.008,
        min_visible_nodes=3,
        reinitialize_after_failures=0,
        hold_last_on_failure=True,
        accept_nonconverged=True,
        visible_observation_fusion=True,
        visible_observation_blend=1.0,
        visible_observation_radius=0.015,
        visible_observation_min_points=3,
        visible_observation_motion_threshold=0.0,
        visible_observation_residual_margin=0.0015,
        visible_observation_normal_only=True,
        visible_observation_tangent_blend=0.0,
        visible_observation_topology="previous",
        visible_observation_use_extended=True,
        hidden_velocity_prediction=True,
        hidden_velocity_decay=0.30,
        adaptive_history_fusion=True,
        adaptive_visible_alpha=12.0,
        adaptive_occluded_alpha=0.0,
        adaptive_history_deformation_switch=True,
        adaptive_history_deformation_threshold=0.012,
        adaptive_history_deformation_occluded_alpha=0.0,
        adaptive_history_deformation_visible_alpha=3.0,
        max_observed_points=max_observed_points,
        observed_point_sampling=os.environ.get(
            "DLO_TRACKDLO_POINT_SAMPLING", "farthest"
        ),
    )


class DualViewTrackDLO:
    """Render both cameras and return TrackDLO nodes in the world frame."""

    def __init__(self, env: CableGraspEnv, renderer: mujoco.Renderer):
        self.env = env
        self.renderer = renderer
        self.opst_name = env.config.dynamicvla_opst_camera_name
        self.wrist_name = env.config.dynamicvla_wrist_camera_name
        self.opst_id = int(env.dynamicvla_opst_camera_id)
        self.wrist_id = int(env.dynamicvla_wrist_camera_id)
        self.opst_k = camera_matrix(
            env.config.dynamicvla_camera_width,
            env.config.dynamicvla_camera_height,
            float(env.model.cam_fovy[self.opst_id]),
        )
        self.wrist_k = camera_matrix(
            env.config.dynamicvla_camera_width,
            env.config.dynamicvla_camera_height,
            float(env.model.cam_fovy[self.wrist_id]),
        )
        self.tracker = TrackDLOTracker(self.opst_k, _track_config())
        self.last_nodes_world: np.ndarray | None = None
        self.last_nodes_primary: np.ndarray | None = None
        self.last_time: float | None = None
        self.last_update_ms = float("nan")
        self.update_times_ms: list[float] = []
        self.update_count = 0
        self.failure_count = 0
        self.nonconverged_count = 0
        self.last_failure_reason: str | None = None
        # Per-node diagnostics consumed by the visual PPO adapter.  These are
        # deliberately kept separate from ``last_nodes_world``: a held pose
        # after a failed update must not be presented as a fresh, confident
        # observation.
        self.last_visible_mask: np.ndarray | None = None
        self.last_confidence: np.ndarray | None = None
        self.last_tracking_ok = False
        self.last_nonconverged = False
        self.last_observed_points = 0

    def reset(self) -> None:
        self.tracker.reset()
        self.last_nodes_world = None
        self.last_nodes_primary = None
        self.last_time = None
        self.last_update_ms = float("nan")
        self.update_times_ms = []
        self.update_count = 0
        self.failure_count = 0
        self.nonconverged_count = 0
        self.last_failure_reason = None
        self.last_visible_mask = None
        self.last_confidence = None
        self.last_tracking_ok = False
        self.last_nonconverged = False
        self.last_observed_points = 0

    def update(self) -> np.ndarray:
        start = perf_counter()
        images = self.env.dynamicvla_camera_rgb()
        opst_rgb = np.asarray(images["opst_cam"], dtype=np.uint8)
        wrist_rgb = np.asarray(images["wrist_cam"], dtype=np.uint8)
        opst_depth = _render_depth(self.renderer, self.env, self.opst_name)
        wrist_depth = _render_depth(self.renderer, self.env, self.wrist_name)

        opst_mask = segment_hsv(opst_rgb, (112, 180, 80), (130, 255, 255))
        wrist_mask = segment_hsv(wrist_rgb, (112, 180, 80), (130, 255, 255))
        opst_points = backproject_mask(
            depth_to_meters(opst_depth), opst_mask, self.opst_k
        )
        wrist_points = backproject_mask(
            depth_to_meters(wrist_depth), wrist_mask, self.wrist_k
        )
        opst_world = world_from_camera_optical(
            self.env.data.cam_xpos[self.opst_id],
            self.env.data.cam_xmat[self.opst_id],
        )
        wrist_world = world_from_camera_optical(
            self.env.data.cam_xpos[self.wrist_id],
            self.env.data.cam_xmat[self.wrist_id],
        )
        wrist_in_opst = transform_points(
            np.linalg.inv(opst_world) @ wrist_world, wrist_points
        )
        merged = np.concatenate((opst_points, wrist_in_opst), axis=0)
        merged = voxel_downsample(merged, 0.008)
        result = self.tracker.update(
            opst_rgb,
            opst_depth,
            observed_points_override=merged,
        )
        self.last_observed_points = int(len(result.observed_points_camera))
        if not result.tracking_ok:
            self.last_tracking_ok = False
            self.last_nonconverged = False
            if self.last_nodes_world is not None:
                node_count = len(self.last_nodes_world)
                self.last_visible_mask = np.zeros(node_count, dtype=bool)
                self.last_confidence = np.zeros(node_count, dtype=np.float64)
            self.failure_count += 1
            self.last_failure_reason = result.failure_reason
            if self.last_nodes_world is None:
                raise RuntimeError(
                    f"TrackDLO failed before a valid visual state: {result.failure_reason}"
                )
            self.last_update_ms = (perf_counter() - start) * 1000.0
            self.update_times_ms.append(self.last_update_ms)
            return self.last_nodes_world.copy()
        self.nonconverged_count += int(result.nonconverged)
        self.last_tracking_ok = True
        self.last_nonconverged = bool(result.nonconverged)
        visible = np.zeros(len(result.nodes_camera), dtype=bool)
        visible_indices = np.asarray(result.visible_nodes, dtype=np.int64).reshape(-1)
        visible_indices = visible_indices[
            (visible_indices >= 0) & (visible_indices < len(visible))
        ]
        visible[visible_indices] = True
        observed = np.asarray(result.observed_points_camera, dtype=np.float64)
        confidence = np.zeros(len(visible), dtype=np.float64)
        if len(observed) and np.any(visible):
            distances = np.linalg.norm(
                np.asarray(result.nodes_camera, dtype=np.float64)[:, None, :]
                - observed[None, :, :], axis=2
            ).min(axis=1)
            # A soft support score is more useful to PPO than a hard threshold:
            # points close to the RGB-D cloud are high-confidence, while hidden
            # or weakly supported nodes stay near zero.
            confidence = np.exp(-distances / 0.020) * visible.astype(np.float64)
        if result.nonconverged:
            confidence *= 0.5
        self.last_visible_mask = visible
        self.last_confidence = np.clip(confidence, 0.0, 1.0)
        self.last_nodes_primary = np.asarray(result.nodes_camera, dtype=np.float64).copy()
        self.last_nodes_world = transform_points(opst_world, self.last_nodes_primary)
        self.last_time = float(self.env.data.time)
        self.last_update_ms = (perf_counter() - start) * 1000.0
        self.update_times_ms.append(self.last_update_ms)
        self.update_count += 1
        return self.last_nodes_world.copy()


class VisionScriptedPolicy(DynamicCableGraspPolicy):
    """Scripted policy whose DLO target comes from a visual 45-node chain."""

    def __init__(self, env: CableGraspEnv, config: PolicyConfig | None = None):
        super().__init__(env, config)
        self._vision_nodes: np.ndarray | None = None
        self._vision_velocity: np.ndarray | None = None
        self._vision_time: float | None = None
        self.vision_state_updates = 0

    def reset(self) -> None:
        super().reset()
        self._vision_nodes = None
        self._vision_velocity = None
        self._vision_time = None
        self.vision_state_updates = 0

    def set_visual_state(self, nodes_world: np.ndarray, timestamp: float) -> None:
        nodes = np.asarray(nodes_world, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all():
            raise ValueError("visual DLO state must be a finite (N,3) array")
        if self._vision_nodes is not None and self._vision_time is not None:
            dt = float(timestamp) - float(self._vision_time)
            if dt > 1e-6:
                velocity = (nodes - self._vision_nodes) / dt
                velocity = np.clip(velocity, -0.8, 0.8)
                self._vision_velocity = velocity
        self._vision_nodes = nodes.copy()
        self._vision_time = float(timestamp)
        self.vision_state_updates += 1

    def _vision_at_s(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        assert self._vision_nodes is not None
        nodes = self._vision_nodes
        velocity = (
            np.zeros_like(nodes)
            if self._vision_velocity is None
            else self._vision_velocity
        )
        u = float(np.clip(s, 0.0, 1.0)) * (len(nodes) - 1)
        left = min(int(np.floor(u)), len(nodes) - 2)
        alpha = float(u - left)
        position = (1.0 - alpha) * nodes[left] + alpha * nodes[left + 1]
        speed = (1.0 - alpha) * velocity[left] + alpha * velocity[left + 1]
        return position, speed

    def _estimated_target(self) -> tuple[np.ndarray, np.ndarray]:
        target_index = self.env.cable_index[self.env.target_body_id]
        s = target_index / max(len(self.env.cable_ids) - 1, 1)
        return self._vision_at_s(s)

    def _predicted_segment(self, prediction_horizon: float | None = None) -> np.ndarray:
        if self._vision_nodes is None:
            return super()._predicted_segment(prediction_horizon)
        if self.locked_segment_index is None:
            position, velocity = self._estimated_target()
        else:
            env_s = (
                self.locked_segment_index + self.locked_segment_alpha
            ) / max(len(self.env.cable_ids) - 1, 1)
            position, velocity = self._vision_at_s(env_s)
        velocity = np.clip(velocity, -0.8, 0.8)
        if prediction_horizon is None:
            if self.phase.name in {"SETTLE", "APPROACH"}:
                prediction_horizon = self.config.approach_prediction_horizon
            elif self.phase.name == "CLOSE":
                prediction_horizon = self.config.close_prediction_horizon
            else:
                prediction_horizon = self.config.prediction_horizon
        predicted = position + float(prediction_horizon) * velocity
        predicted[0] = np.clip(predicted[0], *self.config.intercept_x_limits)
        predicted[1] = np.clip(predicted[1], *self.config.intercept_y_limits)
        predicted[2] = np.clip(predicted[2], *self.config.intercept_z_limits)
        self.filtered_target += self.config.target_filter_alpha * (
            predicted - self.filtered_target
        )
        return self.filtered_target.copy()

    def _nearest_cable_point(self, point: np.ndarray):
        if self._vision_nodes is None:
            return super()._nearest_cable_point(point)
        nodes = self._vision_nodes
        starts = nodes[:-1]
        vectors = nodes[1:] - starts
        denom = np.sum(vectors * vectors, axis=1)
        alpha = np.sum((point - starts) * vectors, axis=1) / np.maximum(denom, 1e-12)
        alpha = np.clip(alpha, 0.0, 1.0)
        projected = starts + alpha[:, None] * vectors
        distances = np.linalg.norm(projected - point, axis=1)
        index = int(np.argmin(distances))
        # Return the equivalent segment coordinate in the simulator chain so
        # inherited locking logic remains valid despite 45 vs. simulator-N.
        s = (index + float(alpha[index])) / max(len(nodes) - 1, 1)
        env_u = s * max(len(self.env.cable_ids) - 1, 1)
        env_index = min(int(np.floor(env_u)), len(self.env.cable_ids) - 2)
        env_alpha = float(env_u - env_index)
        return projected[index].copy(), float(distances[index]), env_index, env_alpha


def run_episode(scenario_name: str, seed: int, vision_period_steps: int) -> dict:
    scenario = get_scenario(scenario_name)
    config = env_config_for_scenario(scenario, seed=seed, episode_seconds=15.0)
    config = replace(
        config,
        robot="panda",
        target_selection="middle",
        dynamicvla_cameras_enabled=True,
    )
    env = CableGraspEnv(config)
    renderer = mujoco.Renderer(
        env.model,
        height=config.dynamicvla_camera_height,
        width=config.dynamicvla_camera_width,
    )
    vision = DualViewTrackDLO(env, renderer)
    policy = VisionScriptedPolicy(env)
    try:
        _, initial_info = env.reset(seed=seed)
        policy.reset()
        vision.reset()
        nodes = vision.update()
        policy.set_visual_state(nodes, float(env.data.time))
        steps = 0
        min_target_distance = float("inf")
        termination_reason = None
        while not policy.finished and env.data.time < env.config.episode_seconds:
            if steps > 0 and steps % max(1, vision_period_steps) == 0:
                nodes = vision.update()
                policy.set_visual_state(nodes, float(env.data.time))
            action = policy.action()
            _, _, terminated, truncated, step_info = env.step(action)
            steps += 1
            min_target_distance = min(
                min_target_distance,
                # Diagnostic only: do not call _predicted_segment here because
                # it updates the policy's low-pass filter and would alter the
                # next control command.
                float(np.linalg.norm(
                    (policy._estimated_target()[0]
                     if policy._vision_nodes is not None
                     else env.target_position())
                    - env.hand_position
                )),
            )
            if truncated:
                termination_reason = step_info.get("termination_reason")
                if policy.result == "running":
                    policy.result = (
                        "failed_motion_boundary"
                        if termination_reason == "rigid_motion_boundary_crossed"
                        else "failed_timeout"
                    )
                    policy.finished = True
            if terminated:
                break
        info = env.info()
        info["ever_pinched"] = env.last_grasped_body_id is not None
        info["base_success"] = env.ever_success
        info["success"] = policy.result == "success"
        if env.ever_success and policy.result == "running":
            policy.result = "success"
        return {
            "scenario": scenario_name,
            "seed": int(seed),
            "success": bool(env.ever_success),
            "policy_result": policy.result,
            "ever_confirmed_grasp": bool(env.ever_confirmed_grasp),
            "ever_bilateral_candidate": bool(env.ever_bilateral_candidate),
            "grasped_body_id": info.get("grasped_body_id"),
            "last_grasp_break_reason": info.get("last_grasp_break_reason"),
            "termination_reason": termination_reason,
            "sim_time": float(env.data.time),
            "steps": int(steps),
            "min_visual_target_distance": float(min_target_distance),
            "vision_updates": int(vision.update_count),
            "vision_failures": int(vision.failure_count),
            "vision_nonconverged": int(vision.nonconverged_count),
            "vision_mean_ms": float(np.mean(vision.update_times_ms)) if vision.update_times_ms else float("nan"),
            "vision_last_failure_reason": vision.last_failure_reason,
            "policy_retries": int(policy.retry_count),
        }
    finally:
        renderer.close()
        env.close()


def _run_episode_job(job: tuple[str, int, int]) -> dict:
    """Process-pool entry point; each worker owns its MuJoCo/TrackDLO state."""

    scenario_name, seed, vision_period_steps = job
    return run_episode(scenario_name, seed, vision_period_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20280804)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--seed-stride", type=int, default=1,
        help="increment between repeats of each scenario",
    )
    parser.add_argument("--vision-period-steps", type=int, default=10)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    jobs = [
        (scenario, args.seed + repeat * args.seed_stride, args.vision_period_steps)
        for scenario in args.scenarios
        for repeat in range(args.episodes)
    ]
    rows = []
    if args.workers == 1:
        for job in jobs:
            print(f"running {job[0]} seed={job[1]}", flush=True)
            row = _run_episode_job(job)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_episode_job, job): job for job in jobs}
            for completed, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                row = future.result()
                rows.append(row)
                print(
                    f"completed {completed}/{len(jobs)} "
                    f"{job[0]} seed={job[1]} success={row['success']}",
                    flush=True,
                )
        rows.sort(key=lambda row: (str(row["scenario"]), int(row["seed"])))
    summary = {
        "method": "scripted_policy_with_dual_view_trackdlo_state",
        "seed": int(args.seed),
        "episodes_per_scenario": int(args.episodes),
        "seed_stride": int(args.seed_stride),
        "workers": int(args.workers),
        "vision_period_steps": int(args.vision_period_steps),
        "control_frequency_hz": 50.0,
        "vision_nominal_frequency_hz": 50.0 / max(1, args.vision_period_steps),
        "episodes": len(rows),
        "successes": int(sum(bool(row["success"]) for row in rows)),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])) if rows else float("nan"),
        "by_scenario": {
            scenario: {
                "episodes": int(sum(row["scenario"] == scenario for row in rows)),
                "successes": int(sum(bool(row["success"]) for row in rows if row["scenario"] == scenario)),
                "success_rate": float(np.mean([bool(row["success"]) for row in rows if row["scenario"] == scenario])) if any(row["scenario"] == scenario for row in rows) else float("nan"),
            }
            for scenario in args.scenarios
        },
        "rows": rows,
        "state_source": "current adaptive dual-camera TrackDLO; wrist cloud transformed into opst frame",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        with (args.output / "per_episode.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
