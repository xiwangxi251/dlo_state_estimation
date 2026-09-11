"""Inspect current-frame RGB-D skeleton paths against MuJoCo ground truth.

This is a small diagnostic used while developing current-frame-first DLO
tracking.  It does not feed ground truth to the tracker; ground truth is only
used to score the extracted image path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np


def _main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="id_shape_nominal_current")
    parser.add_argument("--frame", type=int, nargs="+", default=[20, 50, 80, 110])
    parser.add_argument("--camera", choices=["opst", "wrist"], default="opst")
    parser.add_argument("--save-overlay-dir", type=Path, default=None)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(os.environ.get("DLO_RUN_ROOT", repo_root / "data" / "recorded_run")),
    )
    parser.add_argument(
        "--project-src", type=Path, default=Path(os.environ.get("DLO_PROJECT_ROOT", repo_root.parent))
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.project_src / "panda_cable_grasp" / "src"))
    sys.path.insert(0, str(repo_root / "trackdlo_standalone" / "src"))
    from dlo_position.geometry import transform_points
    from dlo_position.recorded_benchmark import camera_matrix, world_from_camera_optical
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario
    from trackdlo_standalone.initialization import (
        depth_skeleton_paths,
        depth_skeleton_paths_global,
        ordered_skeleton_pixels,
        ordered_skeleton_pixels_depth,
        segment_hsv,
    )
    from trackdlo_standalone.geometry import depth_to_meters
    from trackdlo_standalone.geometry import resample_polyline
    from trackdlo_standalone.metrics import frame_metrics

    episode = (
        args.run_root
        / "episodes"
        / "expert"
        / args.scenario
        / "seed_20280804"
    )
    with (episode / "episode.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    config = env_config_for_scenario(
        get_scenario(args.scenario),
        seed=int(metadata["result"]["requested_seed"]),
        episode_seconds=15.0,
    )
    config.dynamicvla_cameras_enabled = True
    env = CableGraspEnv(config)
    renderer = mujoco.Renderer(env.model, height=360, width=480)
    camera_id = int(
        env.dynamicvla_opst_camera_id
        if args.camera == "opst"
        else env.dynamicvla_wrist_camera_id
    )
    camera_name = (
        env.config.dynamicvla_opst_camera_name
        if args.camera == "opst"
        else env.config.dynamicvla_wrist_camera_name
    )
    intrinsics = camera_matrix(480, 360, float(env.model.cam_fovy[camera_id]))
    trajectory = np.load(episode / "trajectory.npz", allow_pickle=False)
    state_spec = mujoco.mjtState(int(trajectory["state_spec"]))
    capture = cv2.VideoCapture(str(episode / ("global.mp4" if args.camera == "opst" else "wrist.mp4")))
    targets = set(int(value) for value in args.frame)
    results = []
    for frame_index in range(max(targets) + 1):
        ok, bgr = capture.read()
        if not ok:
            break
        if frame_index not in targets:
            continue
        state_index = int(trajectory["frame_state_indices"][frame_index])
        mujoco.mj_setState(env.model, env.data, trajectory["states"][state_index], state_spec)
        mujoco.mj_forward(env.model, env.data)
        renderer.enable_depth_rendering()
        renderer.update_scene(env.data, camera=camera_name)
        depth_m = depth_to_meters(renderer.render().copy())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = segment_hsv(rgb, (112, 180, 80), (130, 255, 255))
        paths = []
        try:
            pixels = ordered_skeleton_pixels(mask)
            paths.append(("ordinary", pixels))
        except Exception as exc:
            paths.append(("ordinary_error", str(exc)))
        try:
            pixels = ordered_skeleton_pixels_depth(mask, depth_m, intrinsics)
            paths.append(("depth_longest", pixels))
        except Exception as exc:
            paths.append(("depth_longest_error", str(exc)))
        try:
            for index, pixels in enumerate(depth_skeleton_paths(mask, depth_m, intrinsics)):
                paths.append((f"depth_path_{index}", pixels))
        except Exception as exc:
            paths.append(("depth_paths_error", str(exc)))
        try:
            for index, pixels in enumerate(depth_skeleton_paths_global(mask, depth_m, intrinsics)):
                paths.append((f"global_depth_path_{index}", pixels))
        except Exception as exc:
            paths.append(("global_depth_paths_error", str(exc)))
        truth_world = env.data.xpos[env.cable_ids].copy()
        world_from_camera = world_from_camera_optical(
            env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
        )
        truth_camera = transform_points(np.linalg.inv(world_from_camera), truth_world)
        if args.save_overlay_dir is not None:
            args.save_overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay = bgr.copy()
            truth_pixels = (intrinsics @ truth_camera.T).T
            truth_pixels = truth_pixels[:, :2] / np.maximum(truth_pixels[:, 2:3], 1e-9)
            for px in truth_pixels.astype(np.int32):
                cv2.circle(overlay, (int(px[0]), int(px[1])), 2, (0, 255, 0), -1)
            try:
                draw_pixels = ordered_skeleton_pixels(mask)
                for row, col in np.asarray(draw_pixels, dtype=np.int32):
                    cv2.circle(overlay, (int(col), int(row)), 1, (0, 0, 255), -1)
            except Exception:
                pass
            cv2.imwrite(str(args.save_overlay_dir / f"{args.camera}_{frame_index:04d}.png"), overlay)
        print(f"frame={frame_index} truth_length={np.linalg.norm(np.diff(truth_camera,axis=0),axis=1).sum():.4f}")
        for name, item in paths:
            if isinstance(item, str):
                print(f"  {name}: {item}")
                continue
            points = []
            for row, col in np.asarray(item, dtype=np.int32):
                value = float(depth_m[row, col])
                if not np.isfinite(value) or value <= 0:
                    patch = depth_m[max(0,row-2):row+3, max(0,col-2):col+3]
                    valid = patch[np.isfinite(patch) & (patch > 0)]
                    if not len(valid):
                        continue
                    value = float(np.median(valid))
                points.append(((col-intrinsics[0,2])*value/intrinsics[0,0],
                               (row-intrinsics[1,2])*value/intrinsics[1,1], value))
            points = np.asarray(points, dtype=np.float64)
            if len(points) < 4:
                print(f"  {name}: valid_points={len(points)}")
                continue
            # A depth skeleton samples the camera-facing cable surface.  Add
            # the radius along the viewing ray as a first-order centreline
            # estimate for this diagnostic.
            norm = np.linalg.norm(points, axis=1, keepdims=True)
            points = points + 0.014 * points / np.maximum(norm, 1e-9)
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            step_values = np.linalg.norm(np.diff(points, axis=0), axis=1)
            pixel_length = float(
                np.linalg.norm(np.diff(np.asarray(item, dtype=np.float64), axis=0), axis=1).sum()
            )
            metric_pixel_length = pixel_length * float(np.median(points[:, 2])) / max(float(intrinsics[0, 0]), 1e-9)
            nearest = np.linalg.norm(points[:, None, :] - truth_camera[None, :, :], axis=2).min(axis=1)
            endpoint_truth = np.argmin(
                np.linalg.norm(points[:, None, :] - truth_camera[None, :, :], axis=2), axis=1
            )
            monotonic_span = (
                int(endpoint_truth.min()),
                int(endpoint_truth.max()),
                float(np.mean(np.abs(np.diff(endpoint_truth)))) if len(endpoint_truth) > 1 else 0.0,
            )
            try:
                sampled = resample_polyline(points, len(truth_camera))
                metrics_forward = frame_metrics(sampled, truth_camera)
                metrics_reverse = frame_metrics(sampled[::-1], truth_camera)
                best_metrics = metrics_forward
                orientation = "forward"
                if metrics_reverse["ordered_error_m"] < metrics_forward["ordered_error_m"]:
                    best_metrics = metrics_reverse
                    orientation = "reverse"
                current_state = (
                    f" resampled={best_metrics['ordered_error_m']*100:.2f}cm/"
                    f"{best_metrics['endpoint_error_m']*100:.2f}cm({orientation})"
                )
            except (ValueError, RuntimeError, FloatingPointError):
                current_state = ""
            print(
                f"  {name}: pixels={len(item)} valid={len(points)} length={length:.4f} "
                f"pixel_metric_length={metric_pixel_length:.4f} "
                f"step_med/max={np.median(step_values):.4f}/{np.max(step_values):.4f} "
                f"nearest_truth_mean={nearest.mean()*100:.2f}cm p95={np.percentile(nearest,95)*100:.2f}cm"
                f" endpoints_truth_idx={int(endpoint_truth[0])},{int(endpoint_truth[-1])} "
                f"nearest_idx_range={monotonic_span[0]}-{monotonic_span[1]} "
                f"idx_step={monotonic_span[2]:.2f}{current_state}"
            )
        print()
    trajectory.close()
    renderer.close()
    env.close()


if __name__ == "__main__":
    _main()
