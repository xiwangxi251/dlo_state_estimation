"""Current-frame-first RGB-D cable reconstruction helpers.

The native TrackDLO update is intentionally temporal: it iterates a deformable
model from the previous node state.  That is useful for a rigid object, but it
can keep a wrong branch during a fast cable deformation.  This module contains
small, state-free helpers for the complementary experiment used by the
benchmark: order the visible RGB-D cable segments in the *current* frame and
join their gaps with a smooth 3-D polyline.

The functions do not know the previous TrackDLO chain and do not use ground
truth.  A total cable length is only a geometric regularizer; it is not a
historical pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import cv2
import numpy as np

from .geometry import depth_to_meters, resample_polyline
from .initialization import (
    depth_skeleton_paths,
    depth_skeleton_paths_global,
    ordered_skeleton_pixels,
)


@dataclass(frozen=True)
class CurrentFrameReconstructionConfig:
    """Tolerances for joining current-frame visible centerline segments."""

    max_paths: int = 5
    min_path_length_m: float = 0.025
    max_connector_m: float = 0.16
    tangent_weight: float = 0.025
    connector_weight: float = 1.0
    length_weight: float = 2.0
    endpoint_extension: bool = True
    extension_fraction_each_end: float = 0.5


def _backproject_path_pixels(
    pixels: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray | None:
    """Back-project a skeleton path, filling isolated depth holes locally."""

    depth_m = depth_to_meters(depth_m)
    height, width = depth_m.shape[:2]
    fx = max(float(intrinsics[0, 0]), 1e-9)
    fy = max(float(intrinsics[1, 1]), 1e-9)
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    points: list[tuple[float, float, float]] = []
    for row, col in np.asarray(pixels, dtype=np.int32):
        if not (0 <= int(row) < height and 0 <= int(col) < width):
            continue
        row, col = int(row), int(col)
        value = float(depth_m[row, col])
        if not np.isfinite(value) or value <= 0.0:
            patch = depth_m[
                max(0, row - 2) : min(height, row + 3),
                max(0, col - 2) : min(width, col + 3),
            ]
            valid = patch[np.isfinite(patch) & (patch > 0.0)]
            if not len(valid):
                continue
            value = float(np.median(valid))
        points.append(((col - cx) * value / fx, (row - cy) * value / fy, value))
    if len(points) < 4:
        return None
    path = np.asarray(points, dtype=np.float64)
    keep = np.concatenate(([True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-7))
    path = path[keep]
    if len(path) < 4:
        return None
    # The recorded workspace is within a few metres of either camera.  A
    # skeleton pixel that samples the renderer's far plane is not a cable
    # centerline; accepting it would create a smooth-looking but metre-scale
    # false route, especially in the wrist view.
    finite_depth = path[:, 2][np.isfinite(path[:, 2])]
    if not len(finite_depth) or float(np.nanmedian(finite_depth)) > 3.0:
        return None
    # A single invalid depth pixel can create an artificial long jump through
    # the scene.  Such a trace is rejected instead of being used to fill a
    # hidden interval.
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    median_step = max(float(np.median(steps)), 1e-6)
    if float(np.max(steps)) > max(0.04, 12.0 * median_step):
        return None
    return path


def _surface_to_center(path: np.ndarray, offset_m: float) -> np.ndarray:
    if offset_m <= 0.0 or len(path) == 0:
        return path
    norms = np.linalg.norm(path, axis=1)
    valid = np.isfinite(path).all(axis=1) & (norms > 1e-9)
    result = path.copy()
    result[valid] += float(offset_m) * path[valid] / norms[valid, None]
    return result


def extract_current_component_paths(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    *,
    surface_offset_m: float = 0.014,
    min_component_area: int = 30,
    max_paths_per_component: int = 1,
    max_component_path_length_m: float = 1.20,
) -> list[np.ndarray]:
    """Extract depth-aware centerline paths from every current mask component.

    ``largest_component`` is appropriate for initialization, but it silently
    discards cable pieces separated by a gripper or a self-occlusion.  This
    routine keeps each sufficiently large component and traces it independently
    before the route solver joins the pieces.
    """

    mask = np.asarray(mask)
    if mask.ndim != 2:
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    paths: list[np.ndarray] = []
    for label in range(1, int(count)):
        if int(stats[label, cv2.CC_STAT_AREA]) < int(min_component_area):
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        candidates: list[np.ndarray] = []
        for extractor in (
            lambda: depth_skeleton_paths_global(component, depth_m, intrinsics),
            lambda: depth_skeleton_paths(component, depth_m, intrinsics),
            lambda: [ordered_skeleton_pixels(component)],
        ):
            try:
                raw_paths = extractor()
            except Exception:
                continue
            if isinstance(raw_paths, np.ndarray):
                raw_paths = [raw_paths]
            for raw in raw_paths:
                path = _backproject_path_pixels(raw, depth_m, intrinsics)
                if path is not None:
                    if _path_length(path) > float(max_component_path_length_m):
                        continue
                    candidates.append(_surface_to_center(path, float(surface_offset_m)))
        if not candidates:
            continue
        # Opposite endpoint traces are usually duplicates.  Retain only the
        # longest representative per component for route reconstruction.
        candidates.sort(key=lambda value: _path_length(value), reverse=True)
        kept: list[np.ndarray] = []
        for candidate in candidates:
            if any(_same_segment(candidate, existing) for existing in kept):
                continue
            kept.append(candidate)
            if len(kept) >= max(int(max_paths_per_component), 1):
                break
        paths.extend(kept)
    return deduplicate_paths(paths)


def _clean_path(path: np.ndarray) -> np.ndarray | None:
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        return None
    finite = np.isfinite(path).all(axis=1)
    path = path[finite]
    if len(path) < 2:
        return None
    keep = np.concatenate(([True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-7))
    path = path[keep]
    return path if len(path) >= 2 else None


def _path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _endpoint_tangent(path: np.ndarray, at_start: bool) -> np.ndarray:
    """Return an outward unit tangent at a path endpoint."""

    count = min(max(len(path) - 1, 1), 8)
    if at_start:
        delta = path[0] - path[count]
    else:
        delta = path[-1] - path[-1 - count]
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return delta / norm


def _same_segment(first: np.ndarray, second: np.ndarray) -> bool:
    """Reject duplicate traces returned from opposite skeleton endpoints."""

    if len(first) < 2 or len(second) < 2:
        return False
    # Compare a short resampled signature in both orientations.  This is
    # deliberately permissive because depth holes can shorten one trace.
    count = 12
    a = resample_polyline(first, count)
    b = resample_polyline(second, count)
    direct = float(np.mean(np.linalg.norm(a - b, axis=1)))
    reverse = float(np.mean(np.linalg.norm(a - b[::-1], axis=1)))
    # Independent cameras observe opposite sides of the same cylindrical
    # cable.  After a nominal radius correction their centerline estimates can
    # still differ by roughly 2--3 cm, so a tighter duplicate threshold would
    # count the same visible segment twice in a dual-view route.
    if min(direct, reverse) < 0.030:
        return True
    # Paths from two viewpoints can cover different fractions of the same
    # cable, so corresponding samples may be phase-shifted.  A symmetric
    # nearest-neighbour Chamfer check catches that duplicate without merging
    # two branches that merely cross at one point.
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    chamfer = 0.5 * (float(np.mean(np.min(distances, axis=1))) + float(np.mean(np.min(distances, axis=0))))
    return chamfer < 0.022


def deduplicate_paths(
    paths: list[np.ndarray] | tuple[np.ndarray, ...],
    config: CurrentFrameReconstructionConfig | None = None,
) -> list[np.ndarray]:
    """Clean, length-filter, and deduplicate current-frame path candidates."""

    cfg = config or CurrentFrameReconstructionConfig()
    cleaned: list[np.ndarray] = []
    for raw in paths:
        path = _clean_path(raw)
        if path is None or _path_length(path) < float(cfg.min_path_length_m):
            continue
        if any(_same_segment(path, existing) for existing in cleaned):
            continue
        cleaned.append(path)
    cleaned.sort(key=_path_length, reverse=True)
    return cleaned[: max(int(cfg.max_paths), 1)]


def _connector(
    first: np.ndarray,
    second: np.ndarray,
    samples_per_m: float = 80.0,
) -> np.ndarray:
    """Create a straight hidden connector, excluding duplicate endpoints."""

    distance = float(np.linalg.norm(second[0] - first[-1]))
    count = max(2, int(np.ceil(distance * samples_per_m)) + 1)
    return np.linspace(first[-1], second[0], count, dtype=np.float64)[1:-1]


def _route_score(
    route: list[np.ndarray],
    target_length_m: float | None,
    config: CurrentFrameReconstructionConfig,
) -> tuple[float, float, float]:
    visible_length = sum(_path_length(path) for path in route)
    connector_length = 0.0
    tangent_cost = 0.0
    for first, second in zip(route[:-1], route[1:]):
        connector = float(np.linalg.norm(second[0] - first[-1]))
        connector_length += connector
        outgoing = _endpoint_tangent(first, at_start=False)
        incoming = -_endpoint_tangent(second, at_start=True)
        if np.linalg.norm(outgoing) > 0.0 and np.linalg.norm(incoming) > 0.0:
            tangent_cost += 1.0 - float(np.clip(np.dot(outgoing, incoming), -1.0, 1.0))
    total = visible_length + connector_length
    length_cost = (
        abs(total - float(target_length_m)) / max(float(target_length_m), 1e-6)
        if target_length_m is not None
        else 0.0
    )
    # Maximize visible coverage while penalizing implausible jumps and sharp
    # turns.  The returned tuple keeps diagnostics useful to callers.
    score = (
        -visible_length
        + float(config.connector_weight) * connector_length
        + float(config.tangent_weight) * tangent_cost
        + float(config.length_weight) * length_cost
    )
    return score, visible_length, connector_length


def _orient(path: np.ndarray, reverse: bool) -> np.ndarray:
    return path[::-1].copy() if reverse else path.copy()


def _extend_to_length(
    route: np.ndarray,
    target_length_m: float,
    config: CurrentFrameReconstructionConfig,
) -> np.ndarray:
    """Extend both current-frame ends along their measured tangent.

    This is used only when visible segments leave less than the known cable
    length.  It adds an explicit low-confidence hidden tail rather than
    moving the measured samples toward a stale previous pose.
    """

    if not config.endpoint_extension or len(route) < 3:
        return route
    current_length = _path_length(route)
    deficit = float(target_length_m) - current_length
    if deficit <= 1e-6:
        return route
    first_fraction = float(np.clip(config.extension_fraction_each_end, 0.0, 1.0))
    first_extra = deficit * first_fraction
    last_extra = deficit - first_extra
    start_tangent = _endpoint_tangent(route, at_start=True)
    end_tangent = _endpoint_tangent(route, at_start=False)
    if np.linalg.norm(start_tangent) <= 1e-9:
        start_tangent = np.zeros(3, dtype=np.float64)
        first_extra = 0.0
        last_extra = deficit
    if np.linalg.norm(end_tangent) <= 1e-9:
        end_tangent = np.zeros(3, dtype=np.float64)
        last_extra = 0.0
        first_extra = deficit
    spacing = max(float(np.median(np.linalg.norm(np.diff(route, axis=0), axis=1))), 0.004)
    first_count = int(np.ceil(first_extra / spacing)) if first_extra > 0.0 else 0
    last_count = int(np.ceil(last_extra / spacing)) if last_extra > 0.0 else 0
    start = (
        route[0][None, :]
        - start_tangent[None, :] * np.linspace(first_extra, spacing, first_count)[:, None]
        if first_count
        else np.empty((0, 3), dtype=np.float64)
    )
    end = (
        route[-1][None, :]
        + end_tangent[None, :] * np.linspace(spacing, last_extra, last_count)[:, None]
        if last_count
        else np.empty((0, 3), dtype=np.float64)
    )
    return np.concatenate((start, route, end), axis=0)


def reconstruct_current_polyline(
    paths: list[np.ndarray] | tuple[np.ndarray, ...],
    target_length_m: float | None = None,
    config: CurrentFrameReconstructionConfig | None = None,
) -> tuple[np.ndarray | None, dict[str, float]]:
    """Join current-frame visible segments without a previous-frame pose.

    Returns ``(polyline, diagnostics)``.  The route search is intentionally
    small (at most five segments), so an exhaustive orientation/permutation
    search is reliable and still negligible compared with CPD.
    """

    cfg = config or CurrentFrameReconstructionConfig()
    unique = deduplicate_paths(paths, cfg)
    if not unique:
        return None, {"path_count": 0.0, "visible_length_m": 0.0, "connector_length_m": 0.0}
    # At a crossing a skeleton extractor can produce a very long spurious
    # path.  Keep a route with all plausible components, but never allow a
    # single connector to bridge an unrelated branch.
    best: tuple[float, list[np.ndarray], float, float] | None = None
    count = len(unique)
    for order in permutations(range(count)):
        ordered = [unique[index] for index in order]
        for orientation in product((False, True), repeat=count):
            route = [_orient(path, reverse) for path, reverse in zip(ordered, orientation)]
            connectors = [
                float(np.linalg.norm(second[0] - first[-1]))
                for first, second in zip(route[:-1], route[1:])
            ]
            if connectors and max(connectors) > float(cfg.max_connector_m):
                continue
            score, visible_length, connector_length = _route_score(route, target_length_m, cfg)
            if best is None or score < best[0]:
                best = (score, route, visible_length, connector_length)
    if best is None:
        route = [unique[0]]
        best = (*_route_score(route, target_length_m, cfg)[:1], route, _path_length(route[0]), 0.0)
    _, route, visible_length, connector_length = best
    pieces: list[np.ndarray] = [route[0]]
    for first, second in zip(route[:-1], route[1:]):
        bridge = _connector(first, second)
        if len(bridge):
            pieces.append(bridge)
        pieces.append(second)
    polyline = np.concatenate(pieces, axis=0)
    if target_length_m is not None and _path_length(polyline) < float(target_length_m):
        polyline = _extend_to_length(polyline, float(target_length_m), cfg)
    diagnostics = {
        "path_count": float(len(route)),
        "candidate_count": float(len(unique)),
        "visible_length_m": float(visible_length),
        "connector_length_m": float(connector_length),
        "output_length_m": float(_path_length(polyline)),
    }
    return polyline, diagnostics
