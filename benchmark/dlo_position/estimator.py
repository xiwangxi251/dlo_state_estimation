from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter_ns

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize

from .geometry import resample_polyline


Pixel = tuple[int, int]


@dataclass
class PositionEstimate:
    points_camera: np.ndarray
    ordered_pixels_rc: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    had_crossing: bool
    reversed_for_continuity: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


class DLOPositionEstimator:
    """Single-DLO RGB-D centerline estimator with no simulator dependency."""

    def __init__(
        self,
        intrinsics: np.ndarray,
        *,
        sample_count: int = 14,
        hsv_lower: tuple[int, int, int] = (100, 180, 150),
        hsv_upper: tuple[int, int, int] = (135, 255, 255),
        minimum_component_area: int = 80,
        junction_radius_px: int = 4,
        spline_smoothing: float = 0.0005,
        surface_to_center_offset_m: float = 0.014,
        surface_to_center_mode: str = "ray",
        adaptive_normal_residual_m: float = 0.006,
        adaptive_normal_weight: float = 0.5,
        use_crossing_hypotheses: bool = False,
    ) -> None:
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        self.sample_count = int(sample_count)
        self.hsv_lower = np.asarray(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.asarray(hsv_upper, dtype=np.uint8)
        self.minimum_component_area = int(minimum_component_area)
        self.junction_radius_px = int(junction_radius_px)
        self.spline_smoothing = float(spline_smoothing)
        self.surface_to_center_offset_m = float(surface_to_center_offset_m)
        if surface_to_center_mode not in {"ray", "normal", "adaptive"}:
            raise ValueError(
                "surface_to_center_mode must be 'ray', 'normal', or 'adaptive'"
            )
        self.surface_to_center_mode = surface_to_center_mode
        self.adaptive_normal_residual_m = float(max(adaptive_normal_residual_m, 0.0))
        self.adaptive_normal_weight = float(np.clip(adaptive_normal_weight, 0.0, 1.0))
        self.use_crossing_hypotheses = bool(use_crossing_hypotheses)
        self._previous_points: np.ndarray | None = None
        self._last_component_count = 0
        self._last_selected_component_area = 0
        self._last_total_component_area = 0

    def reset(self) -> None:
        self._previous_points = None
        self._last_component_count = 0
        self._last_selected_component_area = 0
        self._last_total_component_area = 0

    def estimate(self, rgb: np.ndarray, depth: np.ndarray) -> PositionEstimate:
        total_start = perf_counter_ns()

        stage_start = perf_counter_ns()
        mask = self._segment(rgb)
        segment_ms = _elapsed_ms(stage_start)

        stage_start = perf_counter_ns()
        candidate_paths, skeleton, had_crossing = self._ordered_centerline_candidates(mask)
        ordering_ms = _elapsed_ms(stage_start)

        stage_start = perf_counter_ns()
        previous_points = self._previous_points
        candidate_results: list[tuple[float, np.ndarray, np.ndarray, bool]] = []
        for path_index, candidate_path in enumerate(candidate_paths):
            try:
                dense_points = self._backproject_ordered_pixels(candidate_path, depth)
                candidate_points = self._smooth_and_resample(dense_points)
            except RuntimeError:
                continue
            reversed_for_continuity = False
            if previous_points is None:
                score = float(path_index)
            else:
                forward_cost = float(
                    np.mean(np.linalg.norm(candidate_points - previous_points, axis=1))
                )
                reverse_cost = float(
                    np.mean(
                        np.linalg.norm(candidate_points[::-1] - previous_points, axis=1)
                    )
                )
                if reverse_cost < forward_cost:
                    candidate_points = candidate_points[::-1].copy()
                    candidate_path = candidate_path[::-1].copy()
                    reversed_for_continuity = True
                    score = reverse_cost
                else:
                    score = forward_cost
            candidate_results.append(
                (score, candidate_points, candidate_path, reversed_for_continuity)
            )
        if not candidate_results:
            raise RuntimeError("no candidate centerline has valid depth")
        candidate_results.sort(key=lambda item: item[0])
        _, points, ordered_pixels, reversed_for_continuity = candidate_results[0]
        geometry_ms = _elapsed_ms(stage_start)
        score_gap = float("nan")
        if len(candidate_results) > 1 and np.isfinite(candidate_results[1][0]):
            score_gap = float(
                max(candidate_results[1][0] - candidate_results[0][0], 0.0)
            )
        self._previous_points = points.copy()

        return PositionEstimate(
            points_camera=points,
            ordered_pixels_rc=ordered_pixels,
            mask=mask,
            skeleton=skeleton,
            had_crossing=had_crossing,
            reversed_for_continuity=reversed_for_continuity,
            timings_ms={
                "segmentation": segment_ms,
                "skeleton_ordering": ordering_ms,
                "depth_geometry": geometry_ms,
                "total": _elapsed_ms(total_start),
                "component_count": float(self._last_component_count),
                "selected_component_area": float(self._last_selected_component_area),
                "total_component_area": float(self._last_total_component_area),
                "crossing_candidates": float(len(candidate_paths)),
                "crossing_score_gap_m": score_gap,
            },
        )

    def estimate_fragments(self, rgb: np.ndarray, depth: np.ndarray) -> list[PositionEstimate]:
        """Estimate every sufficiently large visible cable component.

        Occlusion can split one DLO into several disconnected blue mask
        components.  The legacy :meth:`estimate` method intentionally keeps
        the largest component for the single-curve baseline; this method
        exposes all fragments so a temporal tracker can place them on the
        common arc-length state instead of discarding the smaller pieces.
        """

        total_start = perf_counter_ns()
        stage_start = perf_counter_ns()
        masks = self._segment_components(rgb)
        segment_ms = _elapsed_ms(stage_start)
        fragments: list[PositionEstimate] = []
        for mask in masks:
            try:
                stage_start = perf_counter_ns()
                ordered_pixels, skeleton, had_crossing = self._ordered_centerline(mask)
                ordering_ms = _elapsed_ms(stage_start)
                stage_start = perf_counter_ns()
                dense_points = self._backproject_ordered_pixels(ordered_pixels, depth)
                points = self._smooth_and_resample(dense_points)
                geometry_ms = _elapsed_ms(stage_start)
            except RuntimeError:
                # A small false-positive component can pass the area filter but
                # still lack a valid skeleton/depth trace.  Keep other cable
                # fragments rather than failing the entire frame.
                continue
            fragments.append(
                PositionEstimate(
                    points_camera=points,
                    ordered_pixels_rc=ordered_pixels,
                    mask=mask,
                    skeleton=skeleton,
                    had_crossing=had_crossing,
                    reversed_for_continuity=False,
                    timings_ms={
                        "segmentation": segment_ms,
                        "skeleton_ordering": ordering_ms,
                        "depth_geometry": geometry_ms,
                        "total": _elapsed_ms(total_start),
                        "component_count": float(self._last_component_count),
                        "selected_component_area": float(
                            self._last_selected_component_area
                        ),
                        "total_component_area": float(
                            self._last_total_component_area
                        ),
                    },
                )
            )
        if not fragments:
            raise RuntimeError("no cable fragments found")
        # Keep the largest component first for deterministic initialization.
        fragments.sort(
            key=lambda item: int(np.count_nonzero(item.mask)), reverse=True
        )
        self._previous_points = fragments[0].points_camera.copy()
        return fragments

    def _segment(self, rgb: np.ndarray) -> np.ndarray:
        return self._segment_components(rgb)[0]

    def _segment_components(self, rgb: np.ndarray) -> list[np.ndarray]:
        rgb = np.asarray(rgb, dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), 8
        )
        if count <= 1:
            raise RuntimeError("no cable component found")
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        valid_components = np.flatnonzero(
            component_areas >= self.minimum_component_area
        )
        self._last_component_count = int(len(valid_components))
        self._last_total_component_area = int(component_areas.sum())
        if not len(valid_components):
            raise RuntimeError(
                "largest cable component is too small: {} pixels".format(
                    int(component_areas.max()) if len(component_areas) else 0
                )
            )
        ordered_components = valid_components[
            np.argsort(component_areas[valid_components])[::-1]
        ]
        component = 1 + int(ordered_components[0])
        area = int(stats[component, cv2.CC_STAT_AREA])
        self._last_selected_component_area = area
        return [
            np.where(labels == (1 + int(index)), 255, 0).astype(np.uint8)
            for index in ordered_components
        ]

    def _ordered_centerline(
        self, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        candidates, skeleton, had_crossing = self._ordered_centerline_candidates(mask)
        return candidates[0], skeleton, had_crossing

    def _ordered_centerline_candidates(
        self, mask: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray, bool]:
        skeleton = skeletonize(mask > 0)
        graph = _skeleton_graph(skeleton)
        if len(graph) < 2:
            raise RuntimeError("cable skeleton is too small")
        has_branch = any(len(neighbours) > 2 for neighbours in graph.values())
        if has_branch:
            crossing_paths = _minimum_bending_crossing_path(
                graph,
                skeleton.shape,
                self.junction_radius_px,
                return_candidates=self.use_crossing_hypotheses,
            )
            if crossing_paths is not None and not self.use_crossing_hypotheses:
                if len(crossing_paths) >= 8:
                    return [crossing_paths], skeleton, True
            elif crossing_paths is not None and len(crossing_paths):
                valid = [path for path in crossing_paths if len(path) >= 8]
                if len(valid) > 1:
                    return valid, skeleton, True
        return [_longest_geodesic_path(graph)], skeleton, False

    def _backproject_ordered_pixels(
        self, pixels_rc: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        depth_m = np.asarray(depth)
        if depth_m.dtype == np.uint16:
            depth_m = depth_m.astype(np.float64) * 0.001
        else:
            depth_m = depth_m.astype(np.float64, copy=False)
        height, width = depth_m.shape
        points: list[tuple[float, float, float]] = []
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        for row_value, col_value in pixels_rc:
            row = int(np.clip(round(float(row_value)), 0, height - 1))
            col = int(np.clip(round(float(col_value)), 0, width - 1))
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
            points.append(
                ((col - cx) * value / fx, (row - cy) * value / fy, value)
            )
        result = np.asarray(points, dtype=np.float64)
        if len(result) < max(8, self.sample_count):
            raise RuntimeError(f"only {len(result)} valid centerline depth points")
        # An RGB-D camera observes the near cable surface, while the RL state is
        # defined at the capsule centerline. Move each point away from the
        # camera by the known cable radius along its viewing ray. Set the offset
        # to zero for applications whose target is the observed surface itself.
        if self.surface_to_center_offset_m > 0.0:
            ray_norms = np.linalg.norm(result, axis=1, keepdims=True)
            rays = result / np.maximum(ray_norms, 1e-9)
            # Fragmented masks are usually caused by self-occlusion.  The
            # local tangent on a short visible fragment is less stable, so
            # adaptive mode retains the calibrated ray correction whenever
            # multiple components are present and uses the normal correction
            # only for a single connected component.
            use_normal = self.surface_to_center_mode == "normal"
            if self.surface_to_center_mode == "adaptive":
                use_normal = (
                    self._last_component_count <= 1
                    and self._nonrigid_motion_residual(result)
                    > self.adaptive_normal_residual_m
                )
            if use_normal and len(result) >= 3:
                tangent = np.empty_like(result)
                tangent[1:-1] = result[2:] - result[:-2]
                tangent[0] = result[1] - result[0]
                tangent[-1] = result[-1] - result[-2]
                tangent /= np.maximum(
                    np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9
                )
                perpendicular = rays - np.sum(
                    rays * tangent, axis=1, keepdims=True
                ) * tangent
                perpendicular_norms = np.linalg.norm(
                    perpendicular, axis=1, keepdims=True
                )
                valid = perpendicular_norms[:, 0] > 1e-4
                correction = rays.copy()
                correction[valid] = perpendicular[valid] / perpendicular_norms[valid]
                if self.surface_to_center_mode == "adaptive":
                    correction = (
                        (1.0 - self.adaptive_normal_weight) * rays
                        + self.adaptive_normal_weight * correction
                    )
                    correction /= np.maximum(
                        np.linalg.norm(correction, axis=1, keepdims=True), 1e-9
                    )
                result = result + self.surface_to_center_offset_m * correction
            else:
                result = result + self.surface_to_center_offset_m * rays
        return result

    def _nonrigid_motion_residual(self, surface_points: np.ndarray) -> float:
        """Return residual after aligning the current visible curve rigidly.

        The residual is a cheap shape-change cue.  Low values indicate a
        static/rigidly moving cable, for which the calibrated ray correction
        is safer; high values indicate non-rigid deformation, where the local
        tangent-normal correction helps remove surface bias.
        """
        if self._previous_points is None or len(surface_points) < self.sample_count:
            return 0.0
        current = resample_polyline(surface_points, self.sample_count)
        previous = np.asarray(self._previous_points, dtype=np.float64)
        if previous.shape != current.shape:
            return 0.0
        source = previous - previous.mean(axis=0, keepdims=True)
        target = current - current.mean(axis=0, keepdims=True)
        covariance = source.T @ target
        u, _, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1] *= -1.0
            rotation = vt.T @ u.T
        translation = current.mean(axis=0) - previous.mean(axis=0) @ rotation.T
        aligned = previous @ rotation.T + translation
        return float(np.mean(np.linalg.norm(aligned - current, axis=1)))

    def _smooth_and_resample(self, points: np.ndarray) -> np.ndarray:
        keep = np.concatenate(
            ([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7)
        )
        points = points[keep]
        if len(points) < 4:
            return resample_polyline(points, self.sample_count)
        try:
            spline, _ = splprep(
                points.T,
                s=self.spline_smoothing,
                k=min(3, len(points) - 1),
            )
            dense = np.column_stack(
                splev(np.linspace(0.0, 1.0, max(300, len(points))), spline)
            )
            if np.isfinite(dense).all():
                points = dense
        except (ValueError, TypeError):
            pass
        return resample_polyline(points, self.sample_count)


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) * 1e-6


def _skeleton_graph(skeleton: np.ndarray) -> dict[Pixel, list[tuple[Pixel, float]]]:
    pixels = [tuple(map(int, value)) for value in np.argwhere(skeleton)]
    pixel_set = set(pixels)
    graph: dict[Pixel, list[tuple[Pixel, float]]] = defaultdict(list)
    for row, col in pixels:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbour = (row + dr, col + dc)
                diagonal_shortcut = (
                    dr != 0
                    and dc != 0
                    and ((row + dr, col) in pixel_set or (row, col + dc) in pixel_set)
                )
                if neighbour in pixel_set and not diagonal_shortcut:
                    graph[(row, col)].append((neighbour, float(np.hypot(dr, dc))))
    return dict(graph)


def _dijkstra_farthest(
    graph: dict[Pixel, list[tuple[Pixel, float]]], start: Pixel
) -> tuple[Pixel, float, dict[Pixel, Pixel | None]]:
    distances = {start: 0.0}
    parents: dict[Pixel, Pixel | None] = {start: None}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate
                parents[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    end = max(distances, key=distances.get)
    return end, distances[end], parents


def _longest_geodesic_path(
    graph: dict[Pixel, list[tuple[Pixel, float]]]
) -> np.ndarray:
    endpoints = [node for node, neighbours in graph.items() if len(neighbours) == 1]
    candidates = endpoints if endpoints else [next(iter(graph))]
    best: tuple[float, Pixel, Pixel, dict[Pixel, Pixel | None]] | None = None
    for start in candidates:
        end, distance, parents = _dijkstra_farthest(graph, start)
        if best is None or distance > best[0]:
            best = (distance, start, end, parents)
    assert best is not None
    _, start, end, parents = best
    path = []
    node: Pixel | None = end
    while node is not None:
        path.append(node)
        if node == start:
            break
        node = parents.get(node)
    if not path or path[-1] != start:
        raise RuntimeError("could not order cable skeleton")
    return np.asarray(path[::-1], dtype=np.float64)


def _minimum_bending_crossing_path(
    graph: dict[Pixel, list[tuple[Pixel, float]]],
    image_shape: tuple[int, int],
    junction_radius: int,
    *,
    return_candidates: bool = False,
) -> np.ndarray | list[np.ndarray] | None:
    """Return an Euler trail with minimum discrete bending at crossing choices.

    This is an independent, single-DLO simplification of the mBEST idea. The
    branch-pixel cluster is collapsed, skeleton segments are extracted, and the
    few possible segment trails are enumerated. The complete trail with the
    smallest cumulative turning energy is selected.
    ``return_candidates=True`` exposes all feasible trails in increasing
    bending-energy order so the temporal estimator can select a candidate using
    3-D continuity as well as the image-only bending heuristic.
    """

    branch_mask = np.zeros(image_shape, dtype=np.uint8)
    for node, neighbours in graph.items():
        if len(neighbours) > 2:
            branch_mask[node] = 1
    if not np.any(branch_mask):
        return None
    size = max(3, 2 * int(junction_radius) + 1)
    junction_region = cv2.dilate(branch_mask, np.ones((size, size), np.uint8))
    count, labels = cv2.connectedComponents(junction_region, 8)

    root_of: dict[Pixel, object] = {}
    members: dict[object, list[Pixel]] = defaultdict(list)
    for node in graph:
        label = int(labels[node])
        root: object = ("junction", label) if label > 0 else node
        root_of[node] = root
        members[root].append(node)

    adjacency: dict[object, set[object]] = {root: set() for root in members}
    for node, neighbours in graph.items():
        root = root_of[node]
        for neighbour, _ in neighbours:
            other = root_of[neighbour]
            if root != other:
                adjacency[root].add(other)
    centers = {
        root: np.asarray(nodes, dtype=np.float64).mean(axis=0)
        for root, nodes in members.items()
    }
    special = {root for root, neighbours in adjacency.items() if len(neighbours) != 2}
    endpoints = [root for root in special if len(adjacency[root]) == 1]
    if len(endpoints) != 2:
        return None

    visited_links: set[frozenset[object]] = set()
    abstract_edges: list[tuple[object, object, list[object]]] = []
    for start in special:
        for first in adjacency[start]:
            link = frozenset((start, first))
            if link in visited_links:
                continue
            path = [start, first]
            visited_links.add(link)
            previous, current = start, first
            while current not in special:
                choices = adjacency[current] - {previous}
                if len(choices) != 1:
                    return None
                following = next(iter(choices))
                visited_links.add(frozenset((current, following)))
                path.append(following)
                previous, current = current, following
            abstract_edges.append((start, current, path))

    if not abstract_edges or len(abstract_edges) > 12:
        return None
    incident: dict[object, list[int]] = defaultdict(list)
    for index, (start, end, _) in enumerate(abstract_edges):
        incident[start].append(index)
        if end != start:
            incident[end].append(index)

    candidates: list[np.ndarray] = []
    candidate_limit = 128

    def search(current: object, used: frozenset[int], roots: list[object]) -> None:
        if len(candidates) >= candidate_limit:
            return
        if len(used) == len(abstract_edges):
            if current == endpoints[1]:
                candidates.append(np.asarray([centers[root] for root in roots]))
            return
        for edge_index in incident[current]:
            if edge_index in used:
                continue
            start, end, path = abstract_edges[edge_index]
            orientations: list[tuple[object, list[object]]] = []
            if current == start:
                orientations.append((end, path))
            if current == end:
                reverse_path = list(reversed(path))
                if not orientations or reverse_path != orientations[0][1]:
                    orientations.append((start, reverse_path))
            for destination, oriented in orientations:
                search(destination, used | {edge_index}, roots + oriented[1:])

    search(endpoints[0], frozenset(), [endpoints[0]])
    if not candidates:
        return None

    def bending_energy(path: np.ndarray) -> float:
        vectors = np.diff(path, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 1e-6
        vectors = vectors[valid]
        lengths = lengths[valid]
        if len(vectors) < 2:
            return float("inf")
        directions = vectors / lengths[:, None]
        return float(np.sum(1.0 - np.sum(directions[:-1] * directions[1:], axis=1)))

    ordered = sorted(candidates, key=bending_energy)
    return ordered if return_candidates else ordered[0]
