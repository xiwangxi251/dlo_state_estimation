from __future__ import annotations

import heapq
from collections import defaultdict

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize

from .geometry import depth_to_meters, resample_polyline


def segment_hsv(rgb: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> np.ndarray:
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.asarray(lower, np.uint8), np.asarray(upper, np.uint8))
    # Match the intent of the official mode filter while rejecting isolated render noise.
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if count <= 1:
        raise RuntimeError("HSV mask contains no connected cable component")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == component, 255, 0).astype(np.uint8)


def _skeleton_graph(skeleton: np.ndarray):
    pixels = [tuple(value) for value in np.argwhere(skeleton)]
    pixel_set = set(pixels)
    graph = defaultdict(list)
    for row, col in pixels:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                neighbour = (row + dr, col + dc)
                # A diagonal edge beside an orthogonal skeleton edge is only a
                # raster shortcut around a one-pixel bend. Keeping it creates
                # artificial triangular junctions and breaks cable ordering.
                diagonal_shortcut = (
                    dr != 0
                    and dc != 0
                    and ((row + dr, col) in pixel_set or (row, col + dc) in pixel_set)
                )
                if neighbour in pixel_set and not diagonal_shortcut:
                    graph[(row, col)].append((neighbour, np.hypot(dr, dc)))
    return graph


def _farthest(graph, start):
    distances = {start: 0.0}
    parents = {start: None}
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


def _short_path(graph, start, end, limit: int) -> list[tuple[int, int]] | None:
    """Find a short unweighted graph path, used only to group crossing pixels."""

    queue = [(start, [start])]
    visited = {start}
    while queue:
        node, path = queue.pop(0)
        if len(path) > limit + 1:
            continue
        for neighbour, _ in graph[node]:
            if neighbour == end:
                return path + [neighbour]
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, path + [neighbour]))
    return None


def _ordered_crossing_trail(
    graph,
    *,
    turn_minimized: bool = False,
    coordinates: dict[tuple[int, int], np.ndarray] | None = None,
) -> np.ndarray | None:
    """Recover a complete cable trail when its 2-D projection self-crosses.

    Skeletonization represents one crossing as a small cluster of degree-three
    pixels. Collapsing that cluster produces one degree-four junction. The
    resulting graph has two odd endpoints and therefore an Euler trail that
    traverses every visible cable branch exactly once.
    """

    branches = [node for node, neighbours in graph.items() if len(neighbours) > 2]
    if not branches:
        return None

    parent = {node: node for node in graph}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    # The simulated cable is 6-8 px wide, so branch pixels belonging to the
    # same projected crossing remain within a ten-pixel skeleton path.
    for index, first in enumerate(branches):
        for second in branches[index + 1 :]:
            path = _short_path(graph, first, second, limit=10)
            if path is not None:
                for node in path[1:]:
                    union(path[0], node)

    members = defaultdict(list)
    for node in graph:
        members[find(node)].append(node)

    adjacency = {root: set() for root in members}
    for node, neighbours in graph.items():
        root = find(node)
        for neighbour, _ in neighbours:
            other = find(neighbour)
            if root != other:
                adjacency[root].add(other)

    edge_count = sum(len(value) for value in adjacency.values()) // 2
    odd = [node for node, neighbours in adjacency.items() if len(neighbours) % 2 == 1]
    endpoints = [node for node, neighbours in adjacency.items() if len(neighbours) == 1]
    if len(odd) != 2 or len(endpoints) != 2:
        return None

    centers = {
        root: np.asarray(
            [coordinates.get(node, node) if coordinates is not None else node for node in points],
            dtype=np.float64,
        ).mean(axis=0)
        for root, points in members.items()
    }

    if turn_minimized:
        # At a projected crossing the graph admits multiple Euler trails.  The
        # default initializer retains its historical deterministic traversal,
        # while current-frame reconstruction can ask for the smoothest trail:
        # locally continuous cable tangents are a useful self-supervised cue.
        edge_total = int(edge_count)
        start = endpoints[0]
        best: tuple[float, list[object]] | None = None
        visited_states = 0

        def edge_key(first, second):
            return tuple(sorted((first, second)))

        def transition_cost(previous, current, next_node) -> float:
            if previous is None:
                return 0.0
            incoming = centers[current] - centers[previous]
            outgoing = centers[next_node] - centers[current]
            in_norm = float(np.linalg.norm(incoming))
            out_norm = float(np.linalg.norm(outgoing))
            if in_norm <= 1e-9 or out_norm <= 1e-9:
                return 2.0
            cosine = float(np.dot(incoming, outgoing) / (in_norm * out_norm))
            return 1.0 - float(np.clip(cosine, -1.0, 1.0))

        def search(current, previous, used_edges, trail, cost):
            nonlocal best, visited_states
            visited_states += 1
            if visited_states > 200_000:
                return
            if len(used_edges) == edge_total:
                if current in endpoints and (best is None or cost < best[0]):
                    best = (float(cost), trail.copy())
                return
            candidates = [
                neighbour for neighbour in adjacency[current]
                if edge_key(current, neighbour) not in used_edges
            ]
            candidates.sort(key=lambda node: transition_cost(previous, current, node))
            for neighbour in candidates:
                edge = edge_key(current, neighbour)
                used_edges.add(edge)
                trail.append(neighbour)
                search(neighbour, current, used_edges, trail,
                       cost + transition_cost(previous, current, neighbour))
                trail.pop()
                used_edges.remove(edge)

        search(start, None, set(), [start], 0.0)
        if best is not None:
            return np.asarray([centers[node] for node in best[1]], dtype=np.float64)

    # Keep the original deterministic Euler traversal for the public
    # initializer.  A projected crossing has multiple valid Euler trails;
    # current-frame experiments may request the turn-minimised variant below,
    # but changing the default here would alter TrackDLO initialization.
    remaining = {node: set(neighbours) for node, neighbours in adjacency.items()}
    stack = [endpoints[0]]
    circuit = []
    while stack:
        node = stack[-1]
        if remaining[node]:
            neighbour = next(iter(remaining[node]))
            remaining[node].remove(neighbour)
            remaining[neighbour].remove(node)
            stack.append(neighbour)
        else:
            circuit.append(stack.pop())
    if len(circuit) != edge_count + 1:
        return None
    return np.asarray([centers[node] for node in circuit[::-1]], dtype=np.float64)


def ordered_skeleton_pixels(mask: np.ndarray) -> np.ndarray:
    component = largest_component(mask)
    skeleton = skeletonize(component > 0)
    graph = _skeleton_graph(skeleton)
    if len(graph) < 2:
        raise RuntimeError("Cable skeleton is too small")
    crossing_trail = _ordered_crossing_trail(graph)
    if crossing_trail is not None:
        return np.rint(crossing_trail).astype(np.int32)
    endpoints = [node for node, neighbours in graph.items() if len(neighbours) == 1]
    if endpoints:
        best = None
        for start in endpoints:
            end, distance, parents = _farthest(graph, start)
            if best is None or distance > best[0]:
                best = (distance, start, end, parents)
        _, start, end, parents = best
    else:
        arbitrary = next(iter(graph))
        start, _, _ = _farthest(graph, arbitrary)
        end, _, parents = _farthest(graph, start)
    path = []
    node = end
    while node is not None:
        path.append(node)
        if node == start:
            break
        node = parents.get(node)
    if not path or path[-1] != start:
        raise RuntimeError("Could not traverse the cable skeleton")
    return np.asarray(path[::-1], dtype=np.int32)


def ordered_skeleton_pixels_turn_minimized(mask: np.ndarray) -> np.ndarray:
    """Return an Euler trail whose projected crossing turns are minimized.

    This is an opt-in current-frame candidate.  The public
    :func:`ordered_skeleton_pixels` remains unchanged for compatibility with
    the original TrackDLO initializer.
    """
    component = largest_component(mask)
    skeleton = skeletonize(component > 0)
    graph = _skeleton_graph(skeleton)
    if len(graph) < 2:
        raise RuntimeError("Cable skeleton is too small")
    crossing_trail = _ordered_crossing_trail(graph, turn_minimized=True)
    if crossing_trail is not None:
        return np.rint(crossing_trail).astype(np.int32)
    return ordered_skeleton_pixels(mask)


def ordered_skeleton_pixels_simple(mask: np.ndarray) -> np.ndarray:
    """Return the longest simple skeleton path without Euler crossing joins.

    At a 2-D self-crossing the crossing-aware Euler trail visits both
    projected branches, but their raster order is not the physical cable
    order.  For current-frame observation, a single smooth endpoint-to-
    endpoint path is a safer visible component; hidden nodes are completed by
    the tracker.  The original ``ordered_skeleton_pixels`` remains unchanged
    for initialization and the default TrackDLO path.
    """
    component = largest_component(mask)
    skeleton = skeletonize(component > 0)
    graph = _skeleton_graph(skeleton)
    if len(graph) < 2:
        raise RuntimeError("Cable skeleton is too small")
    endpoints = [node for node, neighbours in graph.items() if len(neighbours) == 1]
    if endpoints:
        best = None
        for start in endpoints:
            end, distance, parents = _farthest(graph, start)
            if best is None or distance > best[0]:
                best = (distance, start, end, parents)
        _, start, end, parents = best
    else:
        arbitrary = next(iter(graph))
        start, _, _ = _farthest(graph, arbitrary)
        end, _, parents = _farthest(graph, start)
    path = []
    node = end
    while node is not None:
        path.append(node)
        if node == start:
            break
        node = parents.get(node)
    if not path or path[-1] != start:
        raise RuntimeError("Could not traverse the cable skeleton")
    return np.asarray(path[::-1], dtype=np.int32)


def depth_skeleton_paths(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Trace a smooth endpoint-to-endpoint skeleton path using RGB-D depth.

    A 2-D crossing appears as a four-way junction.  The ordinary raster
    shortest path has no way to know which two branches belong to the same
    3-D cable segment.  During a walk, choose the continuation whose 3-D
    tangent is most aligned with the incoming tangent and whose depth changes
    smoothly.  This deliberately returns one physical visible component;
    unsupported components remain available to the temporal completion step.
    """
    component = largest_component(mask)
    skeleton = skeletonize(component > 0)
    graph = _skeleton_graph(skeleton)
    if len(graph) < 2:
        raise RuntimeError("Cable skeleton is too small")
    height, width = depth_m.shape[:2]
    points: dict[tuple[int, int], np.ndarray] = {}
    for row, col in graph:
        value = float(depth_m[row, col])
        if not np.isfinite(value) or value <= 0.0:
            patch = depth_m[
                max(0, row - 2) : min(height, row + 3),
                max(0, col - 2) : min(width, col + 3),
            ]
            valid = patch[np.isfinite(patch) & (patch > 0.0)]
            if len(valid):
                value = float(np.median(valid))
        if np.isfinite(value) and value > 0.0:
            points[(row, col)] = np.asarray(
                [
                    (float(col) - intrinsics[0, 2]) * value / intrinsics[0, 0],
                    (float(row) - intrinsics[1, 2]) * value / intrinsics[1, 1],
                    value,
                ],
                dtype=np.float64,
            )
    if len(points) < 4:
        raise RuntimeError("Cable depth skeleton is too small")
    endpoints = [node for node, neighbours in graph.items() if len(neighbours) == 1]
    if not endpoints:
        endpoints = list(graph)[: min(8, len(graph))]

    def trace(start):
        path = [start]
        visited = {start}
        previous = None
        current = start
        while True:
            candidates = [
                neighbour
                for neighbour, _ in graph[current]
                if neighbour not in visited and neighbour in points
            ]
            if not candidates:
                break
            if previous is None or previous not in points:
                # At an endpoint, prefer the direction with the longest
                # immediate 3-D step; this avoids stepping into a tiny raster
                # spur before the real cable segment.
                next_node = max(
                    candidates,
                    key=lambda node: float(np.linalg.norm(points[node] - points[current])),
                )
            else:
                incoming = points[current] - points[previous]
                incoming_norm = float(np.linalg.norm(incoming))
                if incoming_norm <= 1e-9:
                    next_node = candidates[0]
                else:
                    incoming /= incoming_norm

                    def continuation_score(node):
                        outgoing = points[node] - points[current]
                        outgoing_norm = float(np.linalg.norm(outgoing))
                        if outgoing_norm <= 1e-9:
                            return -1e9
                        outgoing /= outgoing_norm
                        tangent_score = float(np.dot(incoming, outgoing))
                        depth_step = abs(float(outgoing[2] - incoming[2]))
                        return tangent_score - 0.5 * depth_step

                    next_node = max(candidates, key=continuation_score)
            previous, current = current, next_node
            path.append(current)
            visited.add(current)
            if len(graph[current]) == 1:
                break
            if len(path) > len(graph):
                break
        return path

    traces = [trace(start) for start in endpoints]
    traces = [path for path in traces if len(path) >= 4]
    if not traces:
        raise RuntimeError("Could not trace a depth-aware cable skeleton")
    # Tracing from both ends of the same physical segment yields two nearly
    # identical paths in opposite directions.  Keep one representative but
    # retain distinct segments separated by a depth-disambiguated crossing.
    unique = []
    for candidate in traces:
        candidate_set = set(candidate)
        duplicate = False
        for existing in unique:
            overlap = len(candidate_set & set(existing)) / max(
                1, min(len(candidate_set), len(existing))
            )
            if overlap >= 0.65:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return [np.asarray(path, dtype=np.int32) for path in unique]


def depth_skeleton_paths_global(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    turn_weight: float = 4.0,
    depth_weight: float = 18.0,
) -> list[np.ndarray]:
    """Trace current RGB-D skeleton paths with global tangent continuity.

    ``depth_skeleton_paths`` uses a greedy continuation at every junction.
    At a projected crossing that local choice can jump from one physical cable
    branch to another even when depth separates the branches.  This variant
    runs Dijkstra on directed skeleton edges: the state contains the previous
    pixel, so each transition is scored by 3-D tangent continuity and depth
    slope, then the lowest-cost endpoint-to-endpoint path is recovered.
    """
    component = largest_component(mask)
    skeleton = skeletonize(component > 0)
    graph = _skeleton_graph(skeleton)
    if len(graph) < 2:
        raise RuntimeError("Cable skeleton is too small")
    height, width = depth_m.shape[:2]
    points: dict[tuple[int, int], np.ndarray] = {}
    for row, col in graph:
        value = float(depth_m[row, col])
        if not np.isfinite(value) or value <= 0.0:
            patch = depth_m[
                max(0, row - 2) : min(height, row + 3),
                max(0, col - 2) : min(width, col + 3),
            ]
            valid = patch[np.isfinite(patch) & (patch > 0.0)]
            if len(valid):
                value = float(np.median(valid))
        if np.isfinite(value) and value > 0.0:
            points[(row, col)] = np.asarray(
                [
                    (float(col) - intrinsics[0, 2]) * value / intrinsics[0, 0],
                    (float(row) - intrinsics[1, 2]) * value / intrinsics[1, 1],
                    value,
                ],
                dtype=np.float64,
            )
    if len(points) < 4:
        raise RuntimeError("Cable depth skeleton is too small")
    endpoints = [node for node, neighbours in graph.items() if len(neighbours) == 1]
    if len(endpoints) < 2:
        endpoints = list(graph)[: min(8, len(graph))]

    # The skeleton is small (typically <500 pixels), so a directed-edge
    # shortest path is cheap enough for every frame.  Edge costs are kept in
    # pixel units; depth and turn terms are dimensionless penalties.
    import heapq as _heapq

    def edge_cost(previous, current, next_node):
        incoming = points[current] - points[previous]
        outgoing = points[next_node] - points[current]
        in_norm = float(np.linalg.norm(incoming))
        out_norm = float(np.linalg.norm(outgoing))
        if in_norm <= 1e-9 or out_norm <= 1e-9:
            return float("inf")
        incoming /= in_norm
        outgoing /= out_norm
        turn = 1.0 - float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
        incoming_depth = float(points[current][2] - points[previous][2]) / in_norm
        outgoing_depth = float(points[next_node][2] - points[current][2]) / out_norm
        depth_change = abs(outgoing_depth - incoming_depth)
        pixel_step = float(np.linalg.norm(np.asarray(next_node, dtype=np.float64) - np.asarray(current, dtype=np.float64)))
        return pixel_step * (1.0 + turn_weight * turn) + depth_weight * depth_change

    paths: list[np.ndarray] = []
    for start in endpoints:
        for first, _ in graph[start]:
            if first not in points:
                continue
            first_state = (start, first)
            distances = {first_state: 0.0}
            parents: dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]] | None] = {
                first_state: None
            }
            queue = [(0.0, first_state)]
            target_state = None
            while queue:
                distance, state = _heapq.heappop(queue)
                if distance != distances.get(state):
                    continue
                previous, current = state
                if current in endpoints and current != start:
                    target_state = state
                    break
                for next_node, _ in graph[current]:
                    if next_node == previous or next_node not in points:
                        continue
                    transition = edge_cost(previous, current, next_node)
                    if not np.isfinite(transition):
                        continue
                    next_state = (current, next_node)
                    new_distance = distance + transition
                    if new_distance < distances.get(next_state, float("inf")):
                        distances[next_state] = new_distance
                        parents[next_state] = state
                        _heapq.heappush(queue, (new_distance, next_state))
            if target_state is None:
                continue
            states = [target_state]
            while parents[states[-1]] is not None:
                states.append(parents[states[-1]])
                if len(states) > len(graph) * 3:
                    break
            if states[-1] != first_state:
                continue
            nodes = [start] + [state[1] for state in states[::-1]]
            if len(nodes) >= 4:
                paths.append(np.asarray(nodes, dtype=np.int32))

    unique: list[np.ndarray] = []
    for candidate in paths:
        candidate_set = set(map(tuple, candidate.tolist()))
        duplicate = False
        for existing in unique:
            existing_set = set(map(tuple, existing.tolist()))
            overlap = len(candidate_set & existing_set) / max(1, min(len(candidate_set), len(existing_set)))
            if overlap >= 0.65:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique


def ordered_skeleton_pixels_depth(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Return the longest depth-aware visible path (legacy single-path API)."""
    paths = depth_skeleton_paths(mask, depth_m, intrinsics)
    if not paths:
        raise RuntimeError("Could not trace a depth-aware cable skeleton")
    return max(paths, key=len)


def initialize_nodes(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    num_nodes: int,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    mask = segment_hsv(rgb, hsv_lower, hsv_upper)
    pixels_rc = ordered_skeleton_pixels(mask)
    depth_m = depth_to_meters(depth)
    points = []
    for row, col in pixels_rc:
        value = depth_m[row, col]
        if not np.isfinite(value) or value <= 0.0:
            patch = depth_m[max(0, row - 2) : row + 3, max(0, col - 2) : col + 3]
            valid = patch[np.isfinite(patch) & (patch > 0.0)]
            if not len(valid):
                continue
            value = float(np.median(valid))
        x = (col - intrinsics[0, 2]) * value / intrinsics[0, 0]
        y = (row - intrinsics[1, 2]) * value / intrinsics[1, 1]
        points.append((x, y, value))
    points = np.asarray(points, dtype=np.float64)
    if len(points) < max(8, num_nodes // 3):
        raise RuntimeError(f"Only {len(points)} valid 3-D skeleton points were found")
    # A single missing/invalid depth sample can be replaced by a far-plane
    # value even though the RGB pixel belongs to the cable.  If it is left in
    # the ordered chain, the initial arc length becomes metres long and all
    # later visible/hidden interval decisions are corrupted.  Remove isolated
    # 3-D jumps before spline fitting; this uses only spatial continuity of the
    # current skeleton, never a previous frame.
    for _ in range(3):
        if len(points) < 3:
            break
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        median_step = max(float(np.median(steps)), 1e-6)
        jump_limit = max(0.08, 20.0 * median_step)
        jump_indices = np.flatnonzero(steps > jump_limit) + 1
        if not len(jump_indices):
            break
        for index in jump_indices:
            if 0 < int(index) < len(points) - 1:
                points[int(index)] = 0.5 * (points[int(index) - 1] + points[int(index) + 1])
    if len(points) >= 3:
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        keep = np.concatenate(([True], steps <= max(0.08, 20.0 * max(float(np.median(steps)), 1e-6))))
        points = points[keep]
    if len(points) < max(8, num_nodes // 3):
        raise RuntimeError(f"Only {len(points)} continuous 3-D skeleton points were found")
    # Follow the official initializer: smooth the ordered 3-D chain before uniform sampling.
    try:
        spline, _ = splprep(points.T, s=0.0005, k=min(3, len(points) - 1))
        spline_dense = np.column_stack(
            splev(np.linspace(0.0, 1.0, max(300, len(points))), spline)
        )
        raw_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        spline_steps = np.linalg.norm(np.diff(spline_dense, axis=0), axis=1)
        spline_length = float(spline_steps.sum())
        # At an occluded/crossing skeleton, cubic splines can overshoot a
        # single noisy depth sample and create a many-metre loop.  Such a
        # curve is worse than the piecewise-linear current-frame path; reject
        # it using only the current path's own length and step statistics.
        if (
            not np.isfinite(spline_dense).all()
            or spline_length > max(1.5 * raw_length, raw_length + 0.15)
            or float(np.max(spline_steps))
            > max(0.08, 20.0 * max(float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1))), 1e-6))
        ):
            dense = points
        else:
            dense = spline_dense
    except (ValueError, TypeError):
        dense = points
    nodes = resample_polyline(dense, num_nodes)
    return nodes, mask
