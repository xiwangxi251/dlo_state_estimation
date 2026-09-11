from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .geometry import transform_points
from .metrics import frame_metrics, summarize
from .tracker import TrackDLOConfig, TrackDLOTracker
from .visualization import render_frame


def load_camera(sequence: Path) -> tuple[dict, np.ndarray]:
    with (sequence / "camera.json").open("r", encoding="utf-8") as stream:
        camera = json.load(stream)
    return camera, np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)


def run_sequence(
    sequence: Path,
    output: Path,
    config: TrackDLOConfig | None = None,
    max_frames: int | None = None,
    init_seconds: float = 0.8,
) -> dict:
    sequence = sequence.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera, intrinsics = load_camera(sequence)
    timestamps = np.load(sequence / "timestamps.npy")
    truth = np.load(sequence / "ground_truth_evaluation_only.npz", allow_pickle=False)
    init_frame = int(np.searchsorted(timestamps, float(timestamps[0]) + init_seconds, side="left"))
    stop_frame = len(timestamps) if max_frames is None else min(len(timestamps), init_frame + max_frames)
    frame_indices = np.arange(init_frame, stop_frame, dtype=np.int64)
    if len(frame_indices) < 2:
        raise ValueError("A tracking sequence needs at least two frames")
    truth_world_all = truth["cable_world"][frame_indices]
    margin = 0.12
    world_bounds = (
        float(truth_world_all[:, :, 0].min() - margin),
        float(truth_world_all[:, :, 0].max() + margin),
        float(truth_world_all[:, :, 1].min() - margin),
        float(truth_world_all[:, :, 1].max() + margin),
    )
    tracker = TrackDLOTracker(intrinsics, config)
    fps = 1.0 / float(np.median(np.diff(timestamps[frame_indices])))
    video_path = output / "trackdlo_overlay.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(camera["width"]) * 2, int(camera["height"])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {video_path}")
    rows = []
    nodes = []
    visible_masks = []
    tracking_ok = []
    reinitialized = []
    failure_reasons = []
    try:
        for output_index, frame_index in enumerate(frame_indices):
            bgr = cv2.imread(str(sequence / "rgb" / f"{frame_index:06d}.png"), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(sequence / "depth" / f"{frame_index:06d}.png"), cv2.IMREAD_UNCHANGED)
            if bgr is None or depth is None:
                raise FileNotFoundError(f"Missing RGB-D frame {frame_index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = tracker.update(rgb, depth)
            world_from_camera = truth["world_from_camera_optical"][frame_index]
            truth_world = truth["cable_world"][frame_index]
            truth_camera = transform_points(np.linalg.inv(world_from_camera), truth_world)
            metrics = frame_metrics(result.nodes_camera, truth_camera)
            row = {
                "frame": output_index,
                "source_frame": int(frame_index),
                "timestamp": float(timestamps[frame_index]),
                "tracking_ok": result.tracking_ok,
                "reinitialized": result.reinitialized,
                "failure_reason": result.failure_reason or "",
                "visible_nodes": len(result.visible_nodes),
                "observed_points": len(result.observed_points_camera),
                "preprocess_ms": result.preprocess_ms,
                "tracking_ms": result.tracking_ms,
                "total_ms": result.total_ms,
                **metrics,
            }
            rows.append(row)
            nodes.append(result.nodes_camera)
            visible = np.zeros(len(result.nodes_camera), dtype=bool)
            visible[result.visible_nodes] = True
            visible_masks.append(visible)
            tracking_ok.append(result.tracking_ok)
            reinitialized.append(result.reinitialized)
            failure_reasons.append(result.failure_reason or "")
            visual = render_frame(
                bgr,
                result,
                truth_camera,
                truth_world,
                world_from_camera,
                intrinsics,
                metrics,
                world_bounds,
            )
            writer.write(visual)
            if output_index == 0:
                cv2.imwrite(str(output / "trackdlo_overlay_first.png"), visual)
    finally:
        writer.release()

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    summary = summarize(rows)
    summary.update(
        {
            "sequence": str(sequence),
            "video": str(video_path),
            "fps": fps,
            "init_seconds": init_seconds,
            "init_source_frame": int(init_frame),
            "config": asdict(tracker.config),
            "evaluation_policy": "Simulation ground truth is loaded only after inference for metrics/overlay.",
        }
    )
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    np.savez_compressed(
        output / "trackdlo_results.npz",
        nodes_camera=np.asarray(nodes),
        visible_mask=np.asarray(visible_masks),
        tracking_ok=np.asarray(tracking_ok),
        reinitialized=np.asarray(reinitialized),
        failure_reason=np.asarray(failure_reasons),
        source_frame_indices=frame_indices,
        timestamps=timestamps[frame_indices],
    )
    return summary
