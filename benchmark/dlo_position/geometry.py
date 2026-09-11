from __future__ import annotations

import numpy as np


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-8)
    pixels[valid, 0] = (
        intrinsics[0, 0] * points[valid, 0] / points[valid, 2]
        + intrinsics[0, 2]
    )
    pixels[valid, 1] = (
        intrinsics[1, 1] * points[valid, 1] / points[valid, 2]
        + intrinsics[1, 2]
    )
    return pixels


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if count < 2:
        raise ValueError("count must be at least two")
    if len(points) == 0:
        raise ValueError("cannot resample an empty polyline")
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    keep = np.concatenate(
        ([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9)
    )
    points = points[keep]
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    queries = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(queries, cumulative, points[:, axis]) for axis in range(points.shape[1])]
    )


def reversal_invariant_errors(
    predicted: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, bool]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} versus {target.shape}")
    forward = np.linalg.norm(predicted - target, axis=1)
    reverse = np.linalg.norm(predicted[::-1] - target, axis=1)
    if np.mean(reverse) < np.mean(forward):
        return reverse, True
    return forward, False

