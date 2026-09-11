"""Evaluate crossing-aware Euler path candidates on recorded RGB-D data.

This is an offline diagnostic only: simulator truth is used to quantify the
candidate gap; no truth is used by the eventual tracker.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="id_shape_nominal_current")
    ap.add_argument("--camera", choices=["opst", "wrist"], default="opst")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, default=Path(r"C:\Users\27642\Desktop\dynamic_cable\linux_log\expert_grasp_fix_4x50\run_20260824_113325"))
    ap.add_argument("--project-src", type=Path, default=Path(r"C:\Users\27642\Desktop\dynamic_cable"))
    args = ap.parse_args()
    sys.path.insert(0, str(args.project_src / "panda_cable_grasp" / "src"))
    sys.path.insert(0, str(args.project_src / "dlo_position_benchmark"))
    sys.path.insert(0, str(args.project_src / "trackdlo_standalone" / "src"))
    from dlo_position.geometry import transform_points
    from dlo_position.recorded_benchmark import camera_matrix, world_from_camera_optical
    from dlo_position.estimator import _minimum_bending_crossing_path, _skeleton_graph
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario
    from trackdlo_standalone.geometry import depth_to_meters, resample_polyline
    from trackdlo_standalone.initialization import segment_hsv, largest_component
    from trackdlo_standalone.metrics import frame_metrics
    episode = args.run_root / "episodes" / "expert" / args.scenario / "seed_20280804"
    with (episode / "episode.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg = env_config_for_scenario(
        get_scenario(args.scenario), seed=int(meta["result"]["requested_seed"]), episode_seconds=15.0
    )
    cfg.dynamicvla_cameras_enabled = True
    env = CableGraspEnv(cfg)
    renderer = mujoco.Renderer(env.model, height=360, width=480)
    camera_id = int(env.dynamicvla_opst_camera_id if args.camera == "opst" else env.dynamicvla_wrist_camera_id)
    camera_name = env.config.dynamicvla_opst_camera_name if args.camera == "opst" else env.config.dynamicvla_wrist_camera_name
    K = camera_matrix(480, 360, float(env.model.cam_fovy[camera_id]))
    traj = np.load(episode / "trajectory.npz", allow_pickle=False)
    spec = mujoco.mjtState(int(traj["state_spec"]))
    cap = cv2.VideoCapture(str(episode / ("global.mp4" if args.camera == "opst" else "wrist.mp4")))
    rows: list[dict] = []
    for frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ok, bgr = cap.read()
        if not ok:
            break
        if frame < 20:
            continue
        si = int(traj["frame_state_indices"][frame])
        mujoco.mj_setState(env.model, env.data, traj["states"][si], spec)
        mujoco.mj_forward(env.model, env.data)
        renderer.enable_depth_rendering()
        renderer.update_scene(env.data, camera=camera_name)
        depth = depth_to_meters(renderer.render().copy())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = largest_component(segment_hsv(rgb, (112, 180, 80), (130, 255, 255)))
        truth = transform_points(
            np.linalg.inv(world_from_camera_optical(env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id])),
            env.data.xpos[env.cable_ids].copy(),
        )
        skeleton = __import__("skimage.morphology", fromlist=["skeletonize"]).skeletonize(mask > 0)
        graph = _skeleton_graph(skeleton)
        candidates = _minimum_bending_crossing_path(graph, skeleton.shape, 4, return_candidates=True)
        if candidates is None:
            candidates = []
        for index, pixels in enumerate(candidates):
            pts = []
            for row, col in np.rint(pixels).astype(np.int32):
                if not (0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]):
                    continue
                value = float(depth[row, col])
                if not np.isfinite(value) or value <= 0:
                    patch = depth[max(0, row - 2):row + 3, max(0, col - 2):col + 3]
                    valid = patch[np.isfinite(patch) & (patch > 0)]
                    if not len(valid):
                        continue
                    value = float(np.median(valid))
                pts.append(((col - K[0, 2]) * value / K[0, 0], (row - K[1, 2]) * value / K[1, 1], value))
            if len(pts) < 4:
                continue
            points = np.asarray(pts, dtype=float)
            points += 0.014 * points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-9)
            sampled = resample_polyline(points, len(truth))
            forward = frame_metrics(sampled, truth)
            reverse = frame_metrics(sampled[::-1], truth)
            nearest = np.linalg.norm(points[:, None, :] - truth[None, :, :], axis=2).min(axis=1).mean()
            steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
            tangent = np.diff(points, axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
            turn = np.mean(1.0 - np.clip(np.sum(tangent[:-1] * tangent[1:], axis=1), -1, 1)) if len(tangent) > 1 else 1.0
            rows.append(dict(frame=frame, candidate=index, candidate_count=len(candidates), length_m=float(steps.sum()), step_p95_m=float(np.percentile(steps, 95)), turn=float(turn), nearest_cm=float(nearest * 100), ordered_cm=float(min(forward["ordered_error_m"], reverse["ordered_error_m"]) * 100), frame_cm=float(min(forward["frame_error_m"], reverse["frame_error_m"]) * 100)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0]) if rows else ["frame"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    cap.release(); renderer.close(); env.close(); traj.close()


if __name__ == "__main__":
    main()
