from __future__ import annotations

import cv2
import numpy as np

from .geometry import project_camera_points, transform_points


def _edge_color(index: int, visible: set[int]) -> tuple[int, int, int]:
    """Colour an edge according to endpoint support, not optical visibility.

    The tracker makes its visibility decision per *node*.  Using a different
    colour for one-supported-endpoint edges prevents the old green/red rule
    from making a whole segment look supported when only one endpoint was.
    Colours are BGR (OpenCV order).
    """
    left = index in visible
    right = index + 1 in visible
    if left and right:
        return (0, 220, 0)       # both endpoints supported by RGB-D
    if left or right:
        return (0, 190, 255)     # only one endpoint supported
    return (0, 0, 255)           # neither endpoint supported


def _world_view_bounds(
    observed_world: np.ndarray,
    predicted_world: np.ndarray,
    truth_world: np.ndarray,
    fallback: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Return a padded, aspect-preserving view that keeps the cable on screen.

    The old panel always used [-0.8, 0.8] in both axes.  That is a convenient
    workspace box, but it can leave a cable near an edge (or clip it) when the
    camera/world origin is not the centre of the cable.  We fit the current
    truth and tracked curve, using robust observed-cloud percentiles so a few
    bad depth points cannot make the panel jump wildly.
    """
    core_arrays = []
    for points in (predicted_world, truth_world):
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr):
            finite = np.isfinite(arr[:, :2]).all(axis=1)
            if np.any(finite):
                core_arrays.append(arr[finite, :2])

    observed = np.asarray(observed_world, dtype=np.float64)
    if observed.ndim == 2 and observed.shape[1] >= 2 and len(observed):
        finite = np.isfinite(observed[:, :2]).all(axis=1)
        if np.any(finite):
            # Keep the cloud influence robust; always include the model curve
            # extrema below so a valid node can never be cropped out.
            obs_xy = observed[finite, :2]
            core_arrays.append(np.percentile(obs_xy, [2, 98], axis=0))

    if not core_arrays:
        return fallback or (-0.8, 0.8, -0.8, 0.8)

    xy = np.concatenate(core_arrays, axis=0)
    low = np.nanmin(xy, axis=0)
    high = np.nanmax(xy, axis=0)
    # A very small span makes the line fill the whole panel.  Keep a modest
    # minimum physical extent while still centring the current cable.
    span = np.maximum(high - low, np.array([0.12, 0.12], dtype=np.float64))
    centre = 0.5 * (low + high)
    low = centre - 0.5 * span
    high = centre + 0.5 * span
    margin = np.maximum(0.04, 0.15 * span)
    low -= margin
    high += margin

    drawable_ratio = max(width - 41, 1) / max(height - 41, 1)
    span = high - low
    current_ratio = span[0] / max(span[1], 1e-9)
    if current_ratio < drawable_ratio:
        extra = 0.5 * (drawable_ratio * span[1] - span[0])
        low[0] -= extra
        high[0] += extra
    elif current_ratio > drawable_ratio:
        extra = 0.5 * (span[0] / drawable_ratio - span[1])
        low[1] -= extra
        high[1] += extra
    return (float(low[0]), float(high[0]), float(low[1]), float(high[1]))


def _draw_camera_overlay(
    bgr: np.ndarray,
    predicted_camera: np.ndarray,
    truth_camera: np.ndarray,
    observed_camera: np.ndarray,
    visible_nodes: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    image = bgr.copy()
    # Draw the actual RGB-D observations first.  These are the downsampled
    # points passed to TrackDLO, not an independently reconstructed curve.
    # Drawing them below the model overlays lets the user check whether the
    # point cloud itself follows the visible cable.
    observed_pixels = project_camera_points(observed_camera, intrinsics)
    height, width = image.shape[:2]
    valid_observed = np.isfinite(observed_pixels).all(axis=1)
    if np.any(valid_observed):
        for pixel in np.rint(observed_pixels[valid_observed]).astype(np.int32):
            col, row = int(pixel[0]), int(pixel[1])
            if 0 <= col < width and 0 <= row < height:
                cv2.circle(image, (col, row), 1, (255, 255, 255), -1, cv2.LINE_AA)

    predicted_pixels = project_camera_points(predicted_camera, intrinsics)
    truth_pixels = project_camera_points(truth_camera, intrinsics)
    visible = set(map(int, visible_nodes))
    valid_truth = np.isfinite(truth_pixels).all(axis=1)
    if valid_truth.sum() >= 2:
        cv2.polylines(image, [np.rint(truth_pixels[valid_truth]).astype(np.int32)], False, (255, 255, 0), 2, cv2.LINE_AA)
    for index in range(len(predicted_pixels) - 1):
        if not np.isfinite(predicted_pixels[index : index + 2]).all():
            continue
        color = _edge_color(index, visible)
        cv2.line(
            image,
            tuple(np.rint(predicted_pixels[index]).astype(int)),
            tuple(np.rint(predicted_pixels[index + 1]).astype(int)),
            color,
            3,
            cv2.LINE_AA,
        )
    for index, pixel in enumerate(predicted_pixels):
        if not np.isfinite(pixel).all():
            continue
        col, row = np.rint(pixel).astype(int)
        if 0 <= col < width and 0 <= row < height:
            color = (0, 150, 255) if index in visible else (0, 0, 255)
            cv2.circle(image, (col, row), 4, color, -1, cv2.LINE_AA)
    return image


def _world_panel(
    shape: tuple[int, int],
    observed_camera: np.ndarray,
    predicted_camera: np.ndarray,
    truth_world: np.ndarray,
    world_from_camera: np.ndarray,
    visible_nodes: np.ndarray,
    world_bounds: tuple[float, float, float, float] | None,
) -> np.ndarray:
    height, width = shape
    panel = np.full((height, width, 3), 36, dtype=np.uint8)

    observed_world = transform_points(world_from_camera, observed_camera)
    predicted_world = transform_points(world_from_camera, predicted_camera)
    x_min, x_max, y_min, y_max = _world_view_bounds(
        observed_world, predicted_world, truth_world, world_bounds, width, height
    )

    def pixels(points_world):
        result = np.empty((len(points_world), 2), dtype=np.float64)
        result[:, 0] = (points_world[:, 0] - x_min) / max(x_max - x_min, 1e-9) * (width - 41) + 20
        result[:, 1] = height - 21 - (points_world[:, 1] - y_min) / max(y_max - y_min, 1e-9) * (height - 41)
        return result

    for fraction in np.linspace(0.0, 1.0, 6):
        x = int(20 + fraction * (width - 41))
        y = int(20 + fraction * (height - 41))
        cv2.line(panel, (x, 20), (x, height - 21), (60, 60, 60), 1)
        cv2.line(panel, (20, y), (width - 21, y), (60, 60, 60), 1)
    observed_pixels = pixels(observed_world)
    for point in np.rint(observed_pixels[:: max(1, len(observed_pixels) // 1500)]).astype(int):
        cv2.circle(panel, tuple(point), 1, (115, 115, 115), -1)
    truth_pixels = np.rint(pixels(truth_world)).astype(np.int32)
    predicted_pixels = np.rint(pixels(predicted_world)).astype(np.int32)
    cv2.polylines(panel, [truth_pixels], False, (255, 255, 0), 2, cv2.LINE_AA)
    visible = set(map(int, visible_nodes))
    for index in range(len(predicted_pixels) - 1):
        color = _edge_color(index, visible)
        cv2.line(panel, tuple(predicted_pixels[index]), tuple(predicted_pixels[index + 1]), color, 3, cv2.LINE_AA)
    for index, point in enumerate(predicted_pixels):
        cv2.circle(panel, tuple(point), 4, (0, 150, 255) if index in visible else (0, 0, 255), -1)
    cv2.putText(panel, "world XY (auto-fit): cloud + tracked nodes", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _pointcloud_panel(
    shape: tuple[int, int], observed_camera: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Render only the RGB-D cloud in a standalone, auto-fit 2-D plot.

    Keeping this panel free of the RGB image, model curve and ground truth
    makes holes, blobs and spatial offsets in the actual observation easy to
    inspect.  Point colour encodes camera depth (z); it is not a confidence
    score.  The ``in-frame`` count reports how many projected points would
    fall inside the original camera image before auto-fitting the plot.
    """
    height, width = shape
    panel = np.full((height, width, 3), 22, dtype=np.uint8)
    points = np.asarray(observed_camera, dtype=np.float64)
    pixels = project_camera_points(points, intrinsics)
    valid = np.isfinite(pixels).all(axis=1) & np.isfinite(points).all(axis=1)
    valid &= points[:, 2] > 1e-8
    if np.any(valid):
        depths = points[valid, 2]
        z_min = float(np.min(depths))
        z_max = float(np.max(depths))
        scale = max(z_max - z_min, 1e-9)
        colors = cv2.applyColorMap(
            np.rint(np.clip((depths - z_min) / scale, 0.0, 1.0) * 255.0)
            .astype(np.uint8)
            .reshape(-1, 1),
            cv2.COLORMAP_TURBO,
        )[:, 0, :]
        projected = pixels[valid]
        in_frame = (
            (projected[:, 0] >= 0.0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < height)
        )
        # Plot all finite projected points, rather than silently clipping the
        # out-of-frame majority.  This is a diagnostic view, not a pixel-true
        # overlay; the in-frame count above tells us whether calibration is
        # consistent with the RGB image.
        low = np.percentile(projected, 1.0, axis=0)
        high = np.percentile(projected, 99.0, axis=0)
        span = np.maximum(high - low, np.array([1.0, 1.0]))
        margin = np.maximum(20.0, 0.12 * span)
        low -= margin
        high += margin
        plot_width = max(width - 30, 1)
        plot_height = max(height - 75, 1)
        display = np.empty_like(projected)
        display[:, 0] = 15.0 + (projected[:, 0] - low[0]) / max(high[0] - low[0], 1e-9) * plot_width
        display[:, 1] = 60.0 + (projected[:, 1] - low[1]) / max(high[1] - low[1], 1e-9) * plot_height
        for pixel, color in zip(np.rint(display).astype(np.int32), colors):
            col, row = int(pixel[0]), int(pixel[1])
            if 0 <= col < width and 60 <= row < height:
                cv2.circle(panel, (col, row), 2, tuple(int(v) for v in color), -1, cv2.LINE_AA)
        depth_text = f"z={z_min:.3f}-{z_max:.3f} m"
        in_frame_text = f"in-frame={int(in_frame.sum())}/{int(valid.sum())}"
    else:
        depth_text = "no valid points"
        in_frame_text = "in-frame=0/0"
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (100, 100, 100), 1)
    cv2.putText(panel, "RGB-D observed point cloud only (auto-fit)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"points={int(valid.sum())}  {in_frame_text}  {depth_text}  colour=depth", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (220, 220, 220), 1, cv2.LINE_AA)
    return panel


def render_pointcloud_frame(
    bgr: np.ndarray, observed_camera: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Return a raw-RGB + standalone point-cloud diagnostic frame."""
    left = bgr.copy()
    cv2.putText(left, "raw RGB (no TrackDLO overlay)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    right = _pointcloud_panel(bgr.shape[:2], observed_camera, intrinsics)
    return np.concatenate((left, right), axis=1)


def render_correspondence_frame(
    bgr: np.ndarray,
    predicted_camera: np.ndarray,
    observed_camera: np.ndarray,
    visible_nodes: np.ndarray,
    self_occluded_nodes: np.ndarray,
    intrinsics: np.ndarray,
    visibility_threshold: float = 0.008,
) -> np.ndarray:
    """Visualise nearest-cloud correspondences for every tracked node.

    TrackDLO does not receive an explicit one-to-one node/point assignment;
    the Python gate only computes each node's nearest observed point distance.
    This diagnostic draws those nearest-point links and the distances used by
    the 8 mm visibility test.
    """
    height, width = bgr.shape[:2]
    left = bgr.copy()
    observed = np.asarray(observed_camera, dtype=np.float64)
    predicted = np.asarray(predicted_camera, dtype=np.float64)
    visible = set(map(int, visible_nodes))
    self_occluded = set(map(int, self_occluded_nodes))
    observed_pixels = project_camera_points(observed, intrinsics)
    predicted_pixels = project_camera_points(predicted, intrinsics)
    valid_observed = np.isfinite(observed_pixels).all(axis=1)
    for pixel in np.rint(observed_pixels[valid_observed]).astype(np.int32):
        col, row = int(pixel[0]), int(pixel[1])
        if 0 <= col < width and 0 <= row < height:
            cv2.circle(left, (col, row), 1, (255, 255, 255), -1, cv2.LINE_AA)

    if len(observed):
        distances = np.linalg.norm(predicted[:, None, :] - observed[None, :, :], axis=2)
        nearest_index = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(len(predicted)), nearest_index]
    else:
        nearest_index = np.full(len(predicted), -1, dtype=np.int32)
        nearest_distance = np.full(len(predicted), np.inf, dtype=np.float64)

    # Draw links first, then nodes on top.  Green links pass the 8 mm support
    # test; red links show the same nearest-point calculation failing it.
    for index, node_pixel in enumerate(predicted_pixels):
        point_index = int(nearest_index[index])
        if point_index < 0 or not np.isfinite(node_pixel).all():
            continue
        point_pixel = observed_pixels[point_index]
        if not np.isfinite(point_pixel).all():
            continue
        node_xy = tuple(np.rint(node_pixel).astype(int))
        point_xy = tuple(np.rint(point_pixel).astype(int))
        in_image = all(
            (0 <= xy[0] < width and 0 <= xy[1] < height) for xy in (node_xy, point_xy)
        )
        if not in_image:
            continue
        supported = index in visible and nearest_distance[index] <= visibility_threshold
        color = (0, 220, 0) if supported else (0, 0, 255)
        cv2.line(left, node_xy, point_xy, color, 1, cv2.LINE_AA)
        cv2.drawMarker(left, point_xy, color, cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)

    for index, node_pixel in enumerate(predicted_pixels):
        if not np.isfinite(node_pixel).all():
            continue
        col, row = np.rint(node_pixel).astype(int)
        if not (0 <= col < width and 0 <= row < height):
            continue
        color = (0, 150, 255) if index in visible else (0, 0, 255)
        cv2.circle(left, (col, row), 4, color, -1, cv2.LINE_AA)
        if index % 5 == 0 or index in visible:
            cv2.putText(left, str(index), (col + 4, row - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    right = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.rectangle(right, (0, 0), (width - 1, height - 1), (100, 100, 100), 1)
    chart_left, chart_right = 38, width - 18
    chart_top, chart_bottom = 65, height - 35
    finite_distances = nearest_distance[np.isfinite(nearest_distance)]
    max_mm = max(
        visibility_threshold * 1000.0 * 2.5,
        float(np.percentile(finite_distances, 95) * 1000.0 * 1.15)
        if len(finite_distances)
        else 20.0,
    )
    max_mm = max(max_mm, 20.0)
    cv2.line(right, (chart_left, chart_bottom), (chart_right, chart_bottom), (180, 180, 180), 1)
    cv2.line(right, (chart_left, chart_top), (chart_left, chart_bottom), (180, 180, 180), 1)
    threshold_y = int(chart_bottom - (visibility_threshold * 1000.0 / max_mm) * (chart_bottom - chart_top))
    cv2.line(right, (chart_left, threshold_y), (chart_right, threshold_y), (0, 220, 255), 2)
    cv2.putText(right, f"{visibility_threshold * 1000:.0f} mm threshold", (chart_left + 5, max(chart_top - 8, threshold_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    count = max(len(nearest_distance), 1)
    bar_width = max((chart_right - chart_left) // count - 2, 2)
    for index, distance in enumerate(nearest_distance):
        x0 = chart_left + int(index * (chart_right - chart_left) / count) + 1
        x1 = min(x0 + bar_width, chart_right)
        if not np.isfinite(distance):
            continue
        value_mm = float(distance) * 1000.0
        y = int(chart_bottom - min(value_mm, max_mm) / max_mm * (chart_bottom - chart_top))
        if index in visible:
            color = (0, 220, 0)
        elif index in self_occluded:
            color = (120, 120, 120)
        else:
            color = (0, 0, 220)
        cv2.rectangle(right, (x0, y), (x1, chart_bottom - 1), color, -1)
    for tick in (0.0, max_mm):
        y = int(chart_bottom - tick / max_mm * (chart_bottom - chart_top))
        cv2.putText(right, f"{tick:.0f}", (5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
    for index in range(0, len(nearest_distance), 5):
        x = chart_left + int(index * (chart_right - chart_left) / count)
        cv2.putText(right, str(index), (x, chart_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(right, "node -> nearest observed point distance", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, f"visible={len(visible)}/{len(predicted)}  green<=threshold  gray=self-occluded  red>threshold", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(left, "white=observed cloud; line=node-to-nearest point", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(left, f"visible={len(visible)}/{len(predicted)}  threshold={visibility_threshold * 1000:.0f} mm", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate((left, right), axis=1)


def render_frame(
    bgr: np.ndarray,
    result,
    truth_camera: np.ndarray,
    truth_world: np.ndarray,
    world_from_camera: np.ndarray,
    intrinsics: np.ndarray,
    metrics: dict,
    world_bounds: tuple[float, float, float, float] | None,
) -> np.ndarray:
    # Show the actual RGB segmentation used to build the point cloud.  A
    # translucent magenta overlay makes it possible to distinguish true robot
    # occlusion/depth loss from an HSV segmentation failure.
    camera_bgr = bgr.copy()
    mask = np.asarray(getattr(result, "mask", np.zeros(bgr.shape[:2], dtype=bool)))
    if mask.shape == bgr.shape[:2] and np.any(mask):
        tint = np.zeros_like(camera_bgr)
        tint[:, :, 0] = 180
        tint[:, :, 2] = 180
        camera_bgr[mask.astype(bool)] = cv2.addWeighted(
            camera_bgr[mask.astype(bool)], 0.55, tint[mask.astype(bool)], 0.45, 0
        )
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(camera_bgr, contours, -1, (255, 0, 255), 1, cv2.LINE_AA)
    left = _draw_camera_overlay(
        camera_bgr,
        result.nodes_camera,
        truth_camera,
        result.observed_points_camera,
        result.visible_nodes,
        intrinsics,
    )
    right = _world_panel(
        bgr.shape[:2],
        result.observed_points_camera,
        result.nodes_camera,
        truth_world,
        world_from_camera,
        result.visible_nodes,
        world_bounds,
    )
    cv2.putText(left, "orange=supported; red=no support; white=RGB-D cloud", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(left, "green edge=2; amber=1; red=0; cyan=simulation GT", (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        left,
        f"visible nodes={len(result.visible_nodes)}  observed points={len(result.observed_points_camera)}",
        (12, 141),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        left,
        f"frame error={metrics['frame_error_m']*100:.2f}cm  total={result.total_ms:.1f}ms",
        (12, 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not result.tracking_ok:
        status = "status=HOLD LAST (tracking failed)"
        cv2.putText(left, result.failure_reason or "tracking failure", (12, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)
    elif result.reinitialized:
        status = "status=REINITIALIZED"
        cv2.putText(left, "RGB-D REINITIALIZATION", (12, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 180, 255), 2, cv2.LINE_AA)
    elif result.nonconverged:
        status = "status=TRACK UPDATE (NONCONVERGED)"
        cv2.putText(left, "accepted final CPD iterate", (12, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 2, cv2.LINE_AA)
    else:
        status = "status=TRACK UPDATE"
    cv2.putText(left, status, (12, 117), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate((left, right), axis=1)
