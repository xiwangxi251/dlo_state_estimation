from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns

import cv2
import mujoco
import numpy as np

from .estimator import DLOPositionEstimator, PositionEstimate
from .geometry import (
    project_points,
    resample_polyline,
    reversal_invariant_errors,
    transform_points,
)


@dataclass
class RecordedFrameRecord:
    camera: str
    scenario: str
    episode: str
    frame: int
    state_index: int
    ok: bool
    failure: str
    target_in_frame_fraction: float
    whole_target_in_frame: bool
    crossing: bool
    mean_point_error_m: float
    rmse_point_error_m: float
    endpoint_error_m: float
    length_ratio: float
    render_ms: float
    segmentation_ms: float
    skeleton_ordering_ms: float
    depth_geometry_ms: float
    algorithm_ms: float


def camera_matrix(width: int, height: int, fovy_degrees: float) -> np.ndarray:
    focal = 0.5 * height / np.tan(np.deg2rad(fovy_degrees) * 0.5)
    return np.array(
        [
            [focal, 0.0, (width - 1) * 0.5],
            [0.0, focal, (height - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def world_from_camera_optical(
    camera_position: np.ndarray, camera_rotation: np.ndarray
) -> np.ndarray:
    axis_conversion = np.diag([1.0, -1.0, -1.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        np.asarray(camera_rotation, dtype=np.float64).reshape(3, 3)
        @ axis_conversion
    )
    transform[:3, 3] = np.asarray(camera_position, dtype=np.float64)
    return transform


def discover_episode_dirs(run_root: Path, scenario: str) -> list[Path]:
    root = run_root / "episodes" / "expert" / scenario
    if not root.is_dir():
        return []
    return sorted(
        episode
        for episode in root.iterdir()
        if episode.is_dir()
        and (episode / "trajectory.npz").is_file()
        and (episode / "trajectory.npz").stat().st_size > 0
        and (episode / "episode.json").is_file()
    )


def choose_evenly_spaced(items: list[Path], count: int | None) -> list[Path]:
    if count is None or count >= len(items):
        return items
    if count <= 0:
        return []
    indices = np.rint(np.linspace(0, len(items) - 1, count)).astype(int)
    return [items[index] for index in np.unique(indices)]


def run_recorded_benchmark(
    *,
    run_root: Path,
    project_src: Path,
    output: Path,
    scenarios: list[str],
    cameras: list[str],
    episodes_per_scenario: int | None,
    frame_stride: int,
    max_frames_per_episode: int | None,
    sample_count: int = 14,
    width: int = 480,
    height: int = 360,
    cable_radius_m: float = 0.014,
    save_overlays_per_group: int = 1,
) -> dict:
    project_src = project_src.resolve()
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario

    output.mkdir(parents=True, exist_ok=True)
    overlay_dir = output / "overlays"
    if save_overlays_per_group:
        overlay_dir.mkdir(exist_ok=True)
    records: list[RecordedFrameRecord] = []
    selected_manifest: list[dict] = []

    for scenario_name in scenarios:
        episode_dirs = choose_evenly_spaced(
            discover_episode_dirs(run_root, scenario_name), episodes_per_scenario
        )
        if not episode_dirs:
            continue
        with (episode_dirs[0] / "episode.json").open("r", encoding="utf-8") as stream:
            first_metadata = json.load(stream)
        seed = int(first_metadata["result"]["requested_seed"])
        scenario = get_scenario(scenario_name)
        config = env_config_for_scenario(
            scenario, seed=seed, episode_seconds=15.0
        )
        config.dynamicvla_cameras_enabled = True
        env = CableGraspEnv(config)
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        try:
            camera_specs = {}
            for short_name in cameras:
                if short_name == "opst":
                    camera_id = int(env.dynamicvla_opst_camera_id)
                    model_name = env.config.dynamicvla_opst_camera_name
                elif short_name == "wrist":
                    camera_id = int(env.dynamicvla_wrist_camera_id)
                    model_name = env.config.dynamicvla_wrist_camera_name
                else:
                    raise ValueError(f"unsupported camera: {short_name}")
                intrinsics = camera_matrix(
                    width, height, float(env.model.cam_fovy[camera_id])
                )
                camera_specs[short_name] = (camera_id, model_name, intrinsics)

            for episode_dir in episode_dirs:
                trajectory = np.load(
                    episode_dir / "trajectory.npz", allow_pickle=False
                )
                state_spec = mujoco.mjtState(int(trajectory["state_spec"]))
                expected_state_size = mujoco.mj_stateSize(env.model, state_spec)
                if trajectory["states"].shape[1] != expected_state_size:
                    raise RuntimeError(
                        f"state size mismatch in {episode_dir}: "
                        f"{trajectory['states'].shape[1]} versus {expected_state_size}"
                    )
                frame_indices = np.arange(
                    0,
                    len(trajectory["frame_state_indices"]),
                    max(1, frame_stride),
                    dtype=np.int64,
                )
                if max_frames_per_episode is not None:
                    frame_indices = frame_indices[:max_frames_per_episode]
                selected_manifest.append(
                    {
                        "scenario": scenario_name,
                        "episode": episode_dir.name,
                        "frames": int(len(frame_indices)),
                    }
                )
                estimators = {
                    camera: DLOPositionEstimator(
                        camera_specs[camera][2],
                        sample_count=sample_count,
                        surface_to_center_offset_m=cable_radius_m,
                    )
                    for camera in cameras
                }
                overlays_saved = {camera: 0 for camera in cameras}

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

                    for camera in cameras:
                        camera_id, model_name, intrinsics = camera_specs[camera]
                        render_start = perf_counter_ns()
                        renderer.disable_depth_rendering()
                        renderer.update_scene(env.data, camera=model_name)
                        rgb = renderer.render().copy()
                        renderer.enable_depth_rendering()
                        renderer.update_scene(env.data, camera=model_name)
                        depth = renderer.render().copy()
                        render_ms = (perf_counter_ns() - render_start) * 1e-6

                        world_from_camera = world_from_camera_optical(
                            env.data.cam_xpos[camera_id],
                            env.data.cam_xmat[camera_id],
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
                        whole_visible = bool(np.all(visible))
                        try:
                            estimate = estimators[camera].estimate(rgb, depth)
                            errors, reverse = reversal_invariant_errors(
                                estimate.points_camera, target_14
                            )
                            aligned = (
                                estimate.points_camera[::-1]
                                if reverse
                                else estimate.points_camera
                            )
                            predicted_length = float(
                                np.sum(
                                    np.linalg.norm(
                                        np.diff(estimate.points_camera, axis=0),
                                        axis=1,
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
                            timings = estimate.timings_ms
                            records.append(
                                RecordedFrameRecord(
                                    camera=camera,
                                    scenario=scenario_name,
                                    episode=episode_dir.name,
                                    frame=int(frame_number),
                                    state_index=state_index,
                                    ok=True,
                                    failure="",
                                    target_in_frame_fraction=visible_fraction,
                                    whole_target_in_frame=whole_visible,
                                    crossing=estimate.had_crossing,
                                    mean_point_error_m=float(np.mean(errors)),
                                    rmse_point_error_m=float(
                                        np.sqrt(np.mean(errors * errors))
                                    ),
                                    endpoint_error_m=float(
                                        np.mean(errors[[0, -1]])
                                    ),
                                    length_ratio=predicted_length
                                    / max(target_length, 1e-9),
                                    render_ms=render_ms,
                                    segmentation_ms=timings["segmentation"],
                                    skeleton_ordering_ms=timings[
                                        "skeleton_ordering"
                                    ],
                                    depth_geometry_ms=timings["depth_geometry"],
                                    algorithm_ms=timings["total"],
                                )
                            )
                            if overlays_saved[camera] < save_overlays_per_group:
                                name = (
                                    f"{camera}__{scenario_name}__{episode_dir.name}"
                                    f"__{int(frame_number):06d}.png"
                                )
                                _write_overlay(
                                    overlay_dir / name,
                                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                    aligned,
                                    target_14,
                                    intrinsics,
                                    estimate,
                                    errors,
                                    visible_fraction,
                                )
                                overlays_saved[camera] += 1
                        except Exception as exc:
                            nan = float("nan")
                            records.append(
                                RecordedFrameRecord(
                                    camera=camera,
                                    scenario=scenario_name,
                                    episode=episode_dir.name,
                                    frame=int(frame_number),
                                    state_index=state_index,
                                    ok=False,
                                    failure=f"{type(exc).__name__}:{exc}",
                                    target_in_frame_fraction=visible_fraction,
                                    whole_target_in_frame=whole_visible,
                                    crossing=False,
                                    mean_point_error_m=nan,
                                    rmse_point_error_m=nan,
                                    endpoint_error_m=nan,
                                    length_ratio=nan,
                                    render_ms=render_ms,
                                    segmentation_ms=nan,
                                    skeleton_ordering_ms=nan,
                                    depth_geometry_ms=nan,
                                    algorithm_ms=nan,
                                )
                            )
        finally:
            renderer.close()
            env.close()

    if not records:
        raise RuntimeError("no recorded frames were evaluated")
    _write_records(output / "per_frame.csv", records)
    group_rows = _group_summaries(records)
    _write_dict_rows(output / "per_camera_scenario.csv", group_rows)
    summary = {
        "source_run": str(run_root.resolve()),
        "project_src": str(project_src),
        "method": "HSV + skeleton + minimum-bending traversal + rendered RGB-D + 3D arc-length sampling",
        "sample_count": sample_count,
        "frame_stride": frame_stride,
        "episodes_per_scenario": episodes_per_scenario,
        "selected_episodes": selected_manifest,
        "camera_results": {
            camera: _summary_for(
                [record for record in records if record.camera == camera]
            )
            for camera in cameras
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def _values(records: list[RecordedFrameRecord], field: str) -> np.ndarray:
    values = np.asarray([getattr(record, field) for record in records], dtype=float)
    return values[np.isfinite(values)]


def _stats(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _summary_for(records: list[RecordedFrameRecord]) -> dict:
    successful = [record for record in records if record.ok]
    whole_visible = [record for record in successful if record.whole_target_in_frame]
    complete = [
        record
        for record in successful
        if 0.75 <= record.length_ratio <= 1.25
    ]
    latency = _values(successful, "algorithm_ms")
    errors = _values(successful, "mean_point_error_m")
    return {
        "frames_attempted": len(records),
        "frames_successful": len(successful),
        "detection_success_rate": len(successful) / max(len(records), 1),
        "whole_target_in_frame_rate": sum(
            record.whole_target_in_frame for record in records
        )
        / max(len(records), 1),
        "mean_target_in_frame_fraction": float(
            np.mean([record.target_in_frame_fraction for record in records])
        ),
        "complete_curve_rate_length_75_to_125pct": len(complete)
        / max(len(records), 1),
        "mean_point_error_m": _stats(errors),
        "whole_target_visible_mean_point_error_m": _stats(
            _values(whole_visible, "mean_point_error_m")
        ),
        "complete_curve_mean_point_error_m": _stats(
            _values(complete, "mean_point_error_m")
        ),
        "algorithm_ms": _stats(latency),
        "stage_ms": {
            "segmentation": _stats(_values(successful, "segmentation_ms")),
            "skeleton_ordering": _stats(
                _values(successful, "skeleton_ordering_ms")
            ),
            "depth_geometry": _stats(
                _values(successful, "depth_geometry_ms")
            ),
        },
        "algorithm_fps_from_mean": float(1000.0 / np.mean(latency))
        if len(latency)
        else 0.0,
        "within_20ms_rate": float(np.mean(latency <= 20.0))
        if len(latency)
        else 0.0,
        "meets_50hz_p95": bool(
            len(latency) and np.percentile(latency, 95) <= 20.0
        ),
        "render_ms_not_counted_as_algorithm": _stats(
            _values(records, "render_ms")
        ),
    }


def _group_summaries(records: list[RecordedFrameRecord]) -> list[dict]:
    rows = []
    groups = sorted({(record.camera, record.scenario) for record in records})
    for camera, scenario in groups:
        subset = [
            record
            for record in records
            if record.camera == camera and record.scenario == scenario
        ]
        summary = _summary_for(subset)
        rows.append(
            {
                "camera": camera,
                "scenario": scenario,
                "frames": summary["frames_attempted"],
                "success_rate": summary["detection_success_rate"],
                "whole_target_in_frame_rate": summary[
                    "whole_target_in_frame_rate"
                ],
                "complete_curve_rate": summary[
                    "complete_curve_rate_length_75_to_125pct"
                ],
                "mean_point_error_m": summary["mean_point_error_m"]["mean"],
                "point_error_p95_m": summary["mean_point_error_m"]["p95"],
                "algorithm_mean_ms": summary["algorithm_ms"]["mean"],
                "algorithm_p95_ms": summary["algorithm_ms"]["p95"],
                "within_20ms_rate": summary["within_20ms_rate"],
            }
        )
    return rows


def _write_records(path: Path, records: list[RecordedFrameRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_dict_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_overlay(
    path: Path,
    bgr: np.ndarray,
    predicted_camera: np.ndarray,
    target_camera: np.ndarray,
    intrinsics: np.ndarray,
    estimate: PositionEstimate,
    errors: np.ndarray,
    target_visible_fraction: float,
) -> None:
    overlay = bgr.copy()
    predicted_pixels = project_points(predicted_camera, intrinsics)
    target_pixels = project_points(target_camera, intrinsics)
    _draw_curve(overlay, target_pixels, (0, 255, 0), 2)
    _draw_curve(overlay, predicted_pixels, (0, 128, 255), 2)
    for index, pixel in enumerate(predicted_pixels):
        if np.isfinite(pixel).all():
            center = tuple(np.rint(pixel).astype(int))
            cv2.circle(overlay, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                str(index),
                (center[0] + 2, center[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    text = (
        f"green=GT red=prediction err={np.mean(errors)*100:.2f}cm "
        f"time={estimate.timings_ms['total']:.2f}ms "
        f"GT-in-frame={target_visible_fraction:.2f}"
    )
    cv2.putText(
        overlay,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), overlay)


def _draw_curve(
    image: np.ndarray, pixels: np.ndarray, color: tuple[int, int, int], thickness: int
) -> None:
    valid = np.isfinite(pixels).all(axis=1)
    points = np.rint(pixels[valid]).astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
