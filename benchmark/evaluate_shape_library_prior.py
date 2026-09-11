"""Evaluate a current-visible / offline-shape-prior completion ablation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np


def _alignment(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    uu, _, vv = np.linalg.svd((src - src_mean).T @ (dst - dst_mean))
    rotation = vv.T @ uu.T
    if np.linalg.det(rotation) < 0.0:
        vv[-1] *= -1.0
        rotation = vv.T @ uu.T
    translation = dst_mean - src_mean @ rotation
    return rotation, translation


def _apply(shape: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(shape) @ rotation + translation


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="id_shape_nominal_current")
    parser.add_argument("--target-episode", default="seed_20280804")
    parser.add_argument("--sample-stride", type=int, default=20)
    parser.add_argument("--max-library", type=int, default=1000)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--fit-on-truth-visible", action="store_true")
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("DLO_RUN_ROOT", repo_root / "data" / "recorded_run")))
    parser.add_argument("--project-root", type=Path, default=Path(os.environ.get("DLO_PROJECT_ROOT", repo_root.parent)))
    args = parser.parse_args()
    root = args.project_root
    sys.path.insert(0, str(root / "panda_cable_grasp" / "src"))
    sys.path.insert(0, str(repo_root / "trackdlo_standalone" / "src"))
    sys.path.insert(0, str(repo_root / "benchmark"))
    from dlo_position.geometry import transform_points
    from dlo_position.recorded_benchmark import camera_matrix, world_from_camera_optical
    from trackdlo_standalone.geometry import resample_polyline
    from trackdlo_standalone.metrics import frame_metrics
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario

    trace_file = Path(
        os.environ.get(
            "DLO_TRACE_FILE",
            str(repo_root / "results" / "adaptive_dual_visible_history_final_20260906" / "node_traces.npz"),
        )
    )
    traces = np.load(trace_file, allow_pickle=False)
    scene = traces["scenario"].astype(str)
    select = scene == args.scenario
    predicted = np.asarray(traces["predicted"])[select]
    truth = np.asarray([resample_polyline(value, 45) for value in traces["truth"][select]])
    visible = np.asarray(traces["visible"], dtype=bool)[select]

    episodes_root = args.run_root / "episodes" / "expert" / args.scenario
    target = episodes_root / args.target_episode
    metadata = json.loads((target / "episode.json").read_text(encoding="utf-8"))
    config = env_config_for_scenario(
        get_scenario(args.scenario),
        seed=int(metadata["result"]["requested_seed"]),
        episode_seconds=15.0,
    )
    config.dynamicvla_cameras_enabled = True
    env = CableGraspEnv(config)
    camera_id = int(env.dynamicvla_opst_camera_id)
    target_traj = np.load(target / "trajectory.npz", allow_pickle=False)
    state_spec = mujoco.mjtState(int(target_traj["state_spec"]))

    library: list[np.ndarray] = []
    for episode in sorted(episodes_root.iterdir()):
        trajectory_file = episode / "trajectory.npz"
        if episode.name == args.target_episode or not trajectory_file.is_file() or trajectory_file.stat().st_size < 1000:
            continue
        trajectory = np.load(trajectory_file, allow_pickle=False)
        states = trajectory["states"]
        state_indices = np.asarray(trajectory["frame_state_indices"], dtype=np.int64)
        for state_index in state_indices[:: max(int(args.sample_stride), 1)]:
            mujoco.mj_setState(env.model, env.data, states[int(state_index)], state_spec)
            mujoco.mj_forward(env.model, env.data)
            world_from_camera = world_from_camera_optical(
                env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
            )
            points_camera = transform_points(
                np.linalg.inv(world_from_camera), env.data.xpos[env.cable_ids].copy()
            )
            library.append(resample_polyline(points_camera, 45))
    library = np.asarray(library, dtype=np.float64)
    if len(library) > int(args.max_library):
        rng = np.random.default_rng(0)
        library = library[rng.choice(len(library), int(args.max_library), replace=False)]
    print("library", library.shape, flush=True)

    def summarize(values):
        result = [frame_metrics(a, b) for a, b in values]
        return {
            key: float(np.mean([item[key] for item in result]))
            for key in ("frame_error_m", "ordered_error_m", "endpoint_error_m")
        }

    print("baseline", {key: round(value * 100.0, 3) for key, value in summarize(zip(predicted, truth)).items()}, flush=True)
    for k in args.top_k:
        values: list[tuple[np.ndarray, np.ndarray]] = []
        for estimate, target_truth, mask in zip(predicted, truth, visible):
            indices = np.flatnonzero(mask)
            if len(indices) < 4:
                values.append((estimate, target_truth))
                continue
            candidates: list[tuple[float, np.ndarray]] = []
            observed = target_truth if args.fit_on_truth_visible else estimate
            for shape in library:
                best: tuple[float, np.ndarray] | None = None
                for oriented in (shape, shape[::-1]):
                    rotation, translation = _alignment(oriented[indices], observed[indices])
                    aligned = _apply(oriented, rotation, translation)
                    residual = float(np.mean(np.linalg.norm(aligned[indices] - observed[indices], axis=1)))
                    if best is None or residual < best[0]:
                        best = (residual, aligned)
                if best is not None:
                    candidates.append(best)
            candidates.sort(key=lambda value: value[0])
            prior = np.mean([item[1] for item in candidates[: max(int(k), 1)]], axis=0)
            completed = estimate.copy()
            completed[~mask] = prior[~mask]
            values.append((completed, target_truth))
        print("k", k, {key: round(value * 100.0, 3) for key, value in summarize(values).items()}, flush=True)


if __name__ == "__main__":
    main()
