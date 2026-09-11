from __future__ import annotations

import numpy as np


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_camera_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-8)
    pixels[valid, 0] = intrinsics[0, 0] * points[valid, 0] / points[valid, 2] + intrinsics[0, 2]
    pixels[valid, 1] = intrinsics[1, 1] * points[valid, 1] / points[valid, 2] + intrinsics[1, 2]
    return pixels


def project_world_points(
    points_world: np.ndarray, world_from_camera: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    return project_camera_points(transform_points(np.linalg.inv(world_from_camera), points_world), intrinsics)


def backproject_mask(depth_m: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0.0)
    rows, cols = np.nonzero(valid)
    z = depth_m[rows, cols]
    x = (cols - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.column_stack((x, y, z)).astype(np.float64, copy=False)


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.dtype == np.uint16:
        return depth.astype(np.float64) * 0.001
    result = depth.astype(np.float64, copy=False)
    # Float depth is part of the public API and is always interpreted as metres.
    return result


def voxel_downsample(points: np.ndarray, leaf_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if not len(points) or leaf_size <= 0.0:
        return points.copy()
    keys = np.floor(points / leaf_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    centroids = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )
    return centroids


def farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    """Select a spatially covering subset with deterministic farthest-point sampling.

    The input is already voxel-downsampled in the benchmark.  Unlike taking
    evenly spaced rows from the voxel hash order, FPS keeps coverage at both
    ends and around bends of the cable, which is important when the native
    CPD cloud is capped for speed.
    """
    points = np.asarray(points, dtype=np.float64)
    if count <= 0 or len(points) <= count:
        return points.copy()
    if not np.isfinite(points).all():
        raise ValueError("farthest_point_sample requires finite points")
    selected = np.empty(int(count), dtype=np.int64)
    selected[0] = 0
    min_distance_sq = np.sum((points - points[0]) ** 2, axis=1)
    min_distance_sq[0] = -1.0
    for index in range(1, int(count)):
        next_index = int(np.argmax(min_distance_sq))
        selected[index] = next_index
        distance_sq = np.sum((points - points[next_index]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
        min_distance_sq[selected[: index + 1]] = -1.0
    return points[selected]


def chain_guided_sample(points: np.ndarray, chain: np.ndarray, count: int) -> np.ndarray:
    """Keep a small cloud with coverage distributed along a reference chain.

    The chain is used only to assign samples to arc-length bins; the returned
    points are still measured points and are never moved toward the chain.
    This prevents a hash-ordered cap from dropping an entire bend while
    keeping the CPD matrix small enough for real-time experiments.
    """
    points = np.asarray(points, dtype=np.float64)
    chain = np.asarray(chain, dtype=np.float64)
    if count <= 0 or len(points) <= count or len(chain) < 2:
        return points.copy()
    if not np.isfinite(points).all() or not np.isfinite(chain).all():
        raise ValueError("chain_guided_sample requires finite points")
    starts = chain[:-1]
    vectors = np.diff(chain, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    lengths_sq = lengths * lengths
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    relative = points[:, None, :] - starts[None, :, :]
    fractions = np.divide(
        np.sum(relative * vectors[None, :, :], axis=2),
        lengths_sq[None, :],
        out=np.zeros((len(points), len(vectors)), dtype=np.float64),
        where=lengths_sq[None, :] > 1e-12,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts[None, :, :] + fractions[:, :, None] * vectors[None, :, :]
    distance_sq = np.sum((points[:, None, :] - projections) ** 2, axis=2)
    segments = np.argmin(distance_sq, axis=1)
    arc = cumulative[segments] + fractions[np.arange(len(points)), segments] * lengths[segments]
    total = max(float(cumulative[-1]), 1e-9)
    bins = np.minimum((arc / total * int(count)).astype(np.int64), int(count) - 1)
    # Prefer points close to the reference chain within each arc bin.  This
    # keeps the measured cable centre/surface while avoiding isolated outliers.
    order = np.argsort(distance_sq[np.arange(len(points)), segments], kind="stable")
    selected: list[int] = []
    used_bins: set[int] = set()
    for point_index in order:
        bin_index = int(bins[point_index])
        if bin_index in used_bins:
            continue
        selected.append(int(point_index))
        used_bins.add(bin_index)
        if len(selected) == int(count):
            break
    if len(selected) < int(count):
        selected_set = set(selected)
        for point_index in order:
            point_index = int(point_index)
            if point_index in selected_set:
                continue
            selected.append(point_index)
            selected_set.add(point_index)
            if len(selected) == int(count):
                break
    return points[np.asarray(selected, dtype=np.int64)]


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("Cannot resample an empty polyline")
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    keep = np.concatenate(([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9))
    points = points[keep]
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    samples = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(samples, cumulative, points[:, axis]) for axis in range(points.shape[1])]
    )


def projection_matrix(intrinsics: np.ndarray) -> np.ndarray:
    return np.column_stack((np.asarray(intrinsics, dtype=np.float64), np.zeros(3)))
