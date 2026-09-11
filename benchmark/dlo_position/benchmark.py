from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np

from .estimator import DLOPositionEstimator, PositionEstimate
from .geometry import (
    project_points,
    resample_polyline,
    reversal_invariant_errors,
    transform_points,
)


@dataclass
class FrameRecord:
    sequence: str
    frame: int
    ok: bool
    failure: str
    crossing: bool
    reversed_for_continuity: bool
    mean_point_error_m: float
    rmse_point_error_m: float
    max_point_error_m: float
    endpoint_error_m: float
    predicted_length_m: float
    target_length_m: float
    length_ratio: float
    decode_ms: float
    segmentation_ms: float
    skeleton_ordering_ms: float
    depth_geometry_ms: float
    algorithm_ms: float


def discover_sequences(data_root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in data_root.rglob("ground_truth_evaluation_only.npz")
        if (path.parent / "rgb").is_dir()
        and (path.parent / "depth").is_dir()
        and (path.parent / "camera.json").is_file()
    )


def run_benchmark(
    sequences: list[Path],
    output: Path,
    *,
    sample_count: int = 14,
    start_frame: int = 0,
    cable_radius_m: float = 0.014,
    stride: int = 1,
    max_frames_per_sequence: int | None = None,
    save_overlays: int = 1,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    overlay_dir = output / "overlays"
    if save_overlays:
        overlay_dir.mkdir(exist_ok=True)
    records: list[FrameRecord] = []

    for sequence in sequences:
        with (sequence / "camera.json").open("r", encoding="utf-8") as stream:
            camera = json.load(stream)
        intrinsics = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
        truth = np.load(sequence / "ground_truth_evaluation_only.npz", allow_pickle=False)
        cable_world = truth["cable_world"]
        world_from_camera = truth["world_from_camera_optical"]
        rgb_files = sorted((sequence / "rgb").glob("*.png"))
        depth_files = sorted((sequence / "depth").glob("*.png"))
        frame_count = min(len(rgb_files), len(depth_files), len(cable_world))
        indices = list(range(max(0, start_frame), frame_count, max(1, stride)))
        if max_frames_per_sequence is not None:
            indices = indices[:max_frames_per_sequence]
        estimator = DLOPositionEstimator(
            intrinsics,
            sample_count=sample_count,
            surface_to_center_offset_m=cable_radius_m,
        )
        sequence_name = "/".join(sequence.parts[-3:])
        saved = 0

        for frame_index in indices:
            decode_start = perf_counter_ns()
            bgr = cv2.imread(str(rgb_files[frame_index]), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_files[frame_index]), cv2.IMREAD_UNCHANGED)
            decode_ms = (perf_counter_ns() - decode_start) * 1e-6
            if bgr is None or depth is None:
                records.append(_failure_record(sequence_name, frame_index, decode_ms, "image_decode"))
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                estimate = estimator.estimate(rgb, depth)
                camera_from_world = np.linalg.inv(world_from_camera[frame_index])
                target_camera = transform_points(camera_from_world, cable_world[frame_index])
                target_14 = resample_polyline(target_camera, sample_count)
                errors, reversed_for_metric = reversal_invariant_errors(
                    estimate.points_camera, target_14
                )
                predicted_length = float(
                    np.sum(np.linalg.norm(np.diff(estimate.points_camera, axis=0), axis=1))
                )
                target_length = float(
                    np.sum(np.linalg.norm(np.diff(target_camera, axis=0), axis=1))
                )
                aligned_prediction = (
                    estimate.points_camera[::-1]
                    if reversed_for_metric
                    else estimate.points_camera
                )
                endpoint_error = float(np.mean(errors[[0, -1]]))
                timings = estimate.timings_ms
                records.append(
                    FrameRecord(
                        sequence=sequence_name,
                        frame=frame_index,
                        ok=True,
                        failure="",
                        crossing=estimate.had_crossing,
                        reversed_for_continuity=estimate.reversed_for_continuity,
                        mean_point_error_m=float(np.mean(errors)),
                        rmse_point_error_m=float(np.sqrt(np.mean(errors * errors))),
                        max_point_error_m=float(np.max(errors)),
                        endpoint_error_m=endpoint_error,
                        predicted_length_m=predicted_length,
                        target_length_m=target_length,
                        length_ratio=predicted_length / max(target_length, 1e-9),
                        decode_ms=decode_ms,
                        segmentation_ms=timings["segmentation"],
                        skeleton_ordering_ms=timings["skeleton_ordering"],
                        depth_geometry_ms=timings["depth_geometry"],
                        algorithm_ms=timings["total"],
                    )
                )
                if saved < save_overlays:
                    overlay_name = (
                        sequence_name.replace("/", "__") + f"__{frame_index:06d}.png"
                    )
                    _write_overlay(
                        overlay_dir / overlay_name,
                        bgr,
                        aligned_prediction,
                        target_14,
                        intrinsics,
                        estimate,
                        errors,
                    )
                    saved += 1
            except Exception as exc:
                records.append(
                    _failure_record(
                        sequence_name,
                        frame_index,
                        decode_ms,
                        f"{type(exc).__name__}:{exc}",
                    )
                )

    _write_frame_csv(output / "per_frame.csv", records)
    sequence_rows = _sequence_summaries(records)
    _write_dict_csv(output / "per_sequence.csv", sequence_rows)
    summary = _aggregate_summary(
        records, sequences, sample_count, stride, cable_radius_m
    )
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


def _failure_record(
    sequence: str, frame: int, decode_ms: float, reason: str
) -> FrameRecord:
    nan = float("nan")
    return FrameRecord(
        sequence=sequence,
        frame=frame,
        ok=False,
        failure=reason,
        crossing=False,
        reversed_for_continuity=False,
        mean_point_error_m=nan,
        rmse_point_error_m=nan,
        max_point_error_m=nan,
        endpoint_error_m=nan,
        predicted_length_m=nan,
        target_length_m=nan,
        length_ratio=nan,
        decode_ms=decode_ms,
        segmentation_ms=nan,
        skeleton_ordering_ms=nan,
        depth_geometry_ms=nan,
        algorithm_ms=nan,
    )


def _finite(records: list[FrameRecord], field: str) -> np.ndarray:
    values = np.asarray([getattr(record, field) for record in records], dtype=np.float64)
    return values[np.isfinite(values)]


def _statistics(values: np.ndarray, prefix: str) -> dict[str, float]:
    if not len(values):
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def _aggregate_summary(
    records: list[FrameRecord],
    sequences: list[Path],
    sample_count: int,
    stride: int,
    cable_radius_m: float,
) -> dict:
    successful = [record for record in records if record.ok]
    latency = _finite(successful, "algorithm_ms")
    deadline = latency <= 20.0
    result: dict = {
        "method": "HSV + skeleton + minimum-bending crossing traversal + RGB-D + 3D arc-length resampling",
        "coordinate_frame": "camera optical frame",
        "sample_count": sample_count,
        "surface_to_center_offset_m": cable_radius_m,
        "sequence_count": len(sequences),
        "stride": stride,
        "frames_attempted": len(records),
        "frames_successful": len(successful),
        "detection_success_rate": len(successful) / max(len(records), 1),
        "crossing_frame_rate": sum(record.crossing for record in successful)
        / max(len(successful), 1),
        "algorithm_timing_excludes_disk_decode": True,
    }
    result.update(_statistics(_finite(successful, "mean_point_error_m"), "mean_point_error_m"))
    result.update(_statistics(_finite(successful, "rmse_point_error_m"), "rmse_point_error_m"))
    result.update(_statistics(_finite(successful, "endpoint_error_m"), "endpoint_error_m"))
    result.update(_statistics(_finite(successful, "length_ratio"), "length_ratio"))
    result.update(_statistics(latency, "algorithm_ms"))
    result.update(_statistics(_finite(records, "decode_ms"), "disk_decode_ms"))
    combined_latency = np.asarray(
        [record.algorithm_ms + record.decode_ms for record in successful],
        dtype=np.float64,
    )
    result.update(_statistics(combined_latency, "disk_decode_plus_algorithm_ms"))
    result["algorithm_fps_from_mean"] = float(1000.0 / np.mean(latency)) if len(latency) else 0.0
    result["frames_within_20ms_rate"] = float(np.mean(deadline)) if len(deadline) else 0.0
    result["disk_decode_plus_algorithm_within_20ms_rate"] = (
        float(np.mean(combined_latency <= 20.0)) if len(combined_latency) else 0.0
    )
    result["meets_50hz_mean"] = bool(len(latency) and np.mean(latency) <= 20.0)
    result["meets_50hz_p95"] = bool(len(latency) and np.percentile(latency, 95) <= 20.0)
    length_ratios = _finite(successful, "length_ratio")
    complete = (length_ratios >= 0.75) & (length_ratios <= 1.25)
    result["complete_curve_rate_length_75_to_125pct"] = (
        float(np.sum(complete) / max(len(records), 1))
    )
    complete_records = [
        record for record in successful if 0.75 <= record.length_ratio <= 1.25
    ]
    result["complete_curve_accuracy"] = _statistics(
        _finite(complete_records, "mean_point_error_m"), "mean_point_error_m"
    )
    mean_errors = _finite(successful, "mean_point_error_m")
    result["frames_below_1cm_error_rate"] = float(
        np.sum(mean_errors <= 0.01) / max(len(records), 1)
    )
    result["frames_below_2cm_error_rate"] = float(
        np.sum(mean_errors <= 0.02) / max(len(records), 1)
    )
    result["stage_timing_ms"] = {}
    for field in ("segmentation_ms", "skeleton_ordering_ms", "depth_geometry_ms"):
        result["stage_timing_ms"].update(_statistics(_finite(successful, field), field))
    return result


def _sequence_summaries(records: list[FrameRecord]) -> list[dict]:
    rows = []
    for sequence in sorted({record.sequence for record in records}):
        subset = [record for record in records if record.sequence == sequence]
        successful = [record for record in subset if record.ok]
        latency = _finite(successful, "algorithm_ms")
        errors = _finite(successful, "mean_point_error_m")
        rows.append(
            {
                "sequence": sequence,
                "frames": len(subset),
                "successful": len(successful),
                "success_rate": len(successful) / max(len(subset), 1),
                "mean_point_error_m": float(np.mean(errors)) if len(errors) else float("nan"),
                "point_error_p95_m": float(np.percentile(errors, 95)) if len(errors) else float("nan"),
                "algorithm_mean_ms": float(np.mean(latency)) if len(latency) else float("nan"),
                "algorithm_p95_ms": float(np.percentile(latency, 95)) if len(latency) else float("nan"),
                "within_20ms_rate": float(np.mean(latency <= 20.0)) if len(latency) else 0.0,
            }
        )
    return rows


def _write_frame_csv(path: Path, records: list[FrameRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
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
) -> None:
    overlay = bgr.copy()
    predicted_pixels = project_points(predicted_camera, intrinsics)
    target_pixels = project_points(target_camera, intrinsics)
    _draw_curve(overlay, target_pixels, (0, 255, 0), 2)
    _draw_curve(overlay, predicted_pixels, (0, 128, 255), 2)
    for index, pixel in enumerate(predicted_pixels):
        if np.isfinite(pixel).all():
            center = tuple(np.rint(pixel).astype(int))
            cv2.circle(overlay, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                str(index),
                (center[0] + 3, center[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    text = (
        f"green=GT orange/red=prediction mean={np.mean(errors)*100:.2f}cm "
        f"time={estimate.timings_ms['total']:.2f}ms crossing={estimate.had_crossing}"
    )
    cv2.putText(
        overlay,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
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
