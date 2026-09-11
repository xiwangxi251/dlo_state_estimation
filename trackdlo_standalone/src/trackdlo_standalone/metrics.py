from __future__ import annotations

import numpy as np

from .geometry import resample_polyline


def point_to_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    starts = polyline[:-1]
    vectors = np.diff(polyline, axis=0)
    lengths_sq = np.sum(vectors * vectors, axis=1)
    relative = points[:, None, :] - starts[None, :, :]
    fractions = np.divide(
        np.sum(relative * vectors[None, :, :], axis=2),
        lengths_sq[None, :],
        out=np.zeros((len(points), len(vectors))),
        where=lengths_sq[None, :] > 1e-16,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts[None, :, :] + fractions[:, :, None] * vectors[None, :, :]
    return np.linalg.norm(points[:, None, :] - projections, axis=2).min(axis=1)


def frame_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred_to_truth = point_to_polyline_distance(predicted, truth)
    truth_to_pred = point_to_polyline_distance(truth, predicted)
    frame_error = 0.5 * (float(pred_to_truth.mean()) + float(truth_to_pred.mean()))
    count = max(len(predicted), len(truth), 100)
    pred_resampled = resample_polyline(predicted, count)
    truth_resampled = resample_polyline(truth, count)
    forward = np.linalg.norm(pred_resampled - truth_resampled, axis=1)
    reverse = np.linalg.norm(pred_resampled[::-1] - truth_resampled, axis=1)
    ordered = min(float(forward.mean()), float(reverse.mean()))
    endpoint_forward = 0.5 * (
        np.linalg.norm(predicted[0] - truth[0]) + np.linalg.norm(predicted[-1] - truth[-1])
    )
    endpoint_reverse = 0.5 * (
        np.linalg.norm(predicted[-1] - truth[0]) + np.linalg.norm(predicted[0] - truth[-1])
    )
    return {
        "frame_error_m": frame_error,
        "ordered_error_m": ordered,
        "endpoint_error_m": float(min(endpoint_forward, endpoint_reverse)),
        "predicted_length_m": float(np.linalg.norm(np.diff(predicted, axis=0), axis=1).sum()),
        "truth_length_m": float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum()),
    }


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if row["tracking_ok"]]
    native = [
        row
        for row in successful
        if not row.get("reinitialized", False) and int(row.get("frame", 0)) > 0
    ]
    longest_failure_streak = 0
    current_failure_streak = 0
    for row in rows:
        current_failure_streak = 0 if row["tracking_ok"] else current_failure_streak + 1
        longest_failure_streak = max(longest_failure_streak, current_failure_streak)
    summary = {
        "frames": len(rows),
        "initializations": 1 if rows else 0,
        "tracking_failures": sum(not row["tracking_ok"] for row in rows),
        "nonconverged_frames": sum(bool(row.get("nonconverged", False)) for row in rows),
        "reinitializations": sum(bool(row.get("reinitialized", False)) for row in rows),
        "native_tracking_updates": len(native),
        "tracking_success_rate": len(successful) / len(rows) if rows else float("nan"),
        "native_update_rate": len(native) / (len(rows) - 1) if len(rows) > 1 else float("nan"),
        "longest_failure_streak": longest_failure_streak,
    }
    for key in (
        "frame_error_m",
        "ordered_error_m",
        "endpoint_error_m",
        "visible_geometric_error_m",
        "preprocess_ms",
        "tracking_ms",
        "total_ms",
    ):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_median"] = float(np.nanmedian(values))
        summary[f"{key}_p95"] = float(np.nanpercentile(values, 95))
        if successful:
            successful_values = np.asarray([row[key] for row in successful], dtype=np.float64)
            summary[f"{key}_successful_mean"] = float(np.nanmean(successful_values))
        if native:
            native_values = np.asarray([row[key] for row in native], dtype=np.float64)
            summary[f"{key}_native_mean"] = float(np.nanmean(native_values))
    return summary
