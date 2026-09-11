"""Offline evaluation of the current-frame-first cable reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            r"C:\Users\27642\Desktop\dynamic_cable\linux_log\expert_grasp_fix_4x50\run_20260824_113325"
        ),
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(r"C:\Users\27642\Desktop\dynamic_cable")
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["id_shape_nominal_current", "id_combined_l1_nominal"],
    )
    parser.add_argument("--episode", default="seed_20280804")
    parser.add_argument("--num-nodes", type=int, default=45)
    parser.add_argument("--surface-offset", type=float, default=0.014)
    parser.add_argument("--dual-camera", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()

    project = args.project_root
    sys.path.insert(0, str(project / "panda_cable_grasp" / "src"))
    sys.path.insert(0, str(project / "trackdlo_standalone" / "src"))
    sys.path.insert(0, str(project / "dlo_position_benchmark"))
    from dlo_position.geometry import transform_points
    from dlo_position.recorded_benchmark import camera_matrix, world_from_camera_optical
    from trackdlo_standalone.current_frame import (
        CurrentFrameReconstructionConfig,
        extract_current_component_paths,
        reconstruct_current_polyline,
    )
    from trackdlo_standalone.geometry import depth_to_meters, resample_polyline
    from trackdlo_standalone.initialization import segment_hsv
    from trackdlo_standalone.metrics import frame_metrics
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario

    for scenario in args.scenarios:
        episode = args.run_root / "episodes" / "expert" / scenario / args.episode
        with (episode / "episode.json").open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
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
        if args.dual_camera:
            secondary_camera_id = int(env.dynamicvla_wrist_camera_id)
            secondary_camera_name = env.config.dynamicvla_wrist_camera_name
            secondary_intrinsics = camera_matrix(
                480, 360, float(env.model.cam_fovy[secondary_camera_id])
            )
            secondary_capture = cv2.VideoCapture(str(episode / "wrist.mp4"))
        else:
            secondary_camera_id = None
            secondary_camera_name = None
            secondary_intrinsics = None
            secondary_capture = None
        trajectory = np.load(episode / "trajectory.npz", allow_pickle=False)
        state_spec = mujoco.mjtState(int(trajectory["state_spec"]))
        capture = cv2.VideoCapture(str(episode / "global.mp4"))
        metrics: list[dict[str, float]] = []
        diagnostics: list[dict[str, float]] = []
        for frame_index in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index % max(int(args.frame_stride), 1) != 0:
                continue
            if secondary_capture is not None:
                secondary_ok, secondary_bgr = secondary_capture.read()
                if not secondary_ok:
                    break
            state_index = int(trajectory["frame_state_indices"][frame_index])
            mujoco.mj_setState(env.model, env.data, trajectory["states"][state_index], state_spec)
            mujoco.mj_forward(env.model, env.data)
            renderer.enable_depth_rendering()
            renderer.update_scene(env.data, camera=camera_name)
            depth_m = depth_to_meters(renderer.render().copy())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = segment_hsv(rgb, (112, 180, 80), (130, 255, 255))
            paths = extract_current_component_paths(
                mask,
                depth_m,
                intrinsics,
                surface_offset_m=float(args.surface_offset),
            )
            if secondary_capture is not None:
                renderer.update_scene(env.data, camera=secondary_camera_name)
                secondary_depth = depth_to_meters(renderer.render().copy())
                secondary_rgb = cv2.cvtColor(secondary_bgr, cv2.COLOR_BGR2RGB)
                secondary_mask = segment_hsv(
                    secondary_rgb, (112, 180, 80), (130, 255, 255)
                )
                secondary_paths = extract_current_component_paths(
                    secondary_mask,
                    secondary_depth,
                    secondary_intrinsics,
                    surface_offset_m=float(args.surface_offset),
                )
                secondary_world_from_camera = world_from_camera_optical(
                    env.data.cam_xpos[secondary_camera_id],
                    env.data.cam_xmat[secondary_camera_id],
                )
                primary_world_from_camera = world_from_camera_optical(
                    env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
                )
                secondary_to_primary = np.linalg.inv(primary_world_from_camera) @ secondary_world_from_camera
                paths.extend(
                    [transform_points(secondary_to_primary, path) for path in secondary_paths]
                )
            polyline, info = reconstruct_current_polyline(
                paths,
                target_length_m=0.78,
                config=CurrentFrameReconstructionConfig(),
            )
            if polyline is None:
                continue
            world_from_camera = world_from_camera_optical(
                env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
            )
            truth_camera = transform_points(
                np.linalg.inv(world_from_camera), env.data.xpos[env.cable_ids].copy()
            )
            truth = resample_polyline(truth_camera, args.num_nodes)
            metrics.append(frame_metrics(resample_polyline(polyline, args.num_nodes), truth))
            diagnostics.append(info)
        capture.release()
        if secondary_capture is not None:
            secondary_capture.release()
        if not metrics:
            print(scenario, "no_valid_reconstruction")
            continue
        keys = ("frame_error_m", "ordered_error_m", "endpoint_error_m")
        mean = {key: float(np.mean([item[key] for item in metrics])) for key in keys}
        print(
            scenario,
            "frames", len(metrics),
            "valid", len(metrics) / max(1, int(trajectory["frame_state_indices"].shape[0])),
            "frame_cm", round(mean["frame_error_m"] * 100.0, 3),
            "ordered_cm", round(mean["ordered_error_m"] * 100.0, 3),
            "endpoint_cm", round(mean["endpoint_error_m"] * 100.0, 3),
            "paths", round(float(np.mean([item["path_count"] for item in diagnostics])), 2),
            "output_len_cm", round(float(np.mean([item["output_length_m"] for item in diagnostics])) * 100.0, 2),
        )


if __name__ == "__main__":
    main()
