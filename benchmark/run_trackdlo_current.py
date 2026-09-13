from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter_ns

import cv2
import mujoco
import numpy as np
from scipy.spatial import cKDTree

from dlo_position.geometry import transform_points
from dlo_position.recorded_benchmark import (
    camera_matrix,
    choose_evenly_spaced,
    discover_episode_dirs,
    world_from_camera_optical,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path(
    os.environ.get("DLO_RUN_ROOT", REPO_ROOT / "data" / "recorded_run")
)
DEFAULT_PROJECT_SRC = Path(
    os.environ.get("PANDA_CABLE_GRASP_SRC", REPO_ROOT.parent / "panda_cable_grasp" / "src")
)
DEFAULT_TRACKDLO_ROOT = REPO_ROOT / "trackdlo_standalone"
DEFAULT_SCENARIOS = [
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_depth(renderer, data, camera_name: str) -> np.ndarray:
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    return renderer.render().copy()


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _cloud_rigid_fit_residual(
    current: np.ndarray,
    previous: np.ndarray,
    max_samples: int = 400,
) -> float | None:
    """Estimate non-rigid RGB-D motion after a cheap nearest-neighbour ICP step.

    The centroid displacement used by the first motion-gate experiment is
    blind to a cable that bends around a nearly fixed centroid.  This metric
    removes the best rigid transform between consecutive visible clouds and
    returns the robust residual.  It is only a confidence cue; no tracker
    state or ground truth is used.
    """
    current = np.asarray(current, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    if current.ndim != 2 or previous.ndim != 2 or current.shape[1:] != (3,) or previous.shape[1:] != (3,):
        return None
    current = current[np.isfinite(current).all(axis=1)]
    previous = previous[np.isfinite(previous).all(axis=1)]
    if len(current) < 12 or len(previous) < 12:
        return None
    if len(current) > max_samples:
        current = current[:: int(np.ceil(len(current) / max_samples))]
    if len(previous) > max_samples:
        previous = previous[:: int(np.ceil(len(previous) / max_samples))]
    tree = cKDTree(previous)
    distances, indices = tree.query(current, k=1)
    # Reject unrelated mask fragments before fitting the rigid transform.
    keep = np.isfinite(distances) & (distances < 0.08)
    if int(np.count_nonzero(keep)) < 8:
        return None
    target = current[keep]
    source = previous[np.asarray(indices[keep], dtype=np.int64)]
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    try:
        left, _, right_t = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    prediction = (source - source_center) @ rotation.T + target_center
    residual = np.linalg.norm(prediction - target, axis=1)
    if not len(residual):
        return None
    return float(np.median(residual))


def _surface_to_center_points(
    points: np.ndarray,
    offset_m: float,
    normal_only: bool = False,
) -> np.ndarray:
    """Move RGB-D surface samples approximately one cable radius inward.

    A depth camera observes the camera-facing surface of a cylindrical cable,
    while TrackDLO nodes represent the cable centreline.  The correction is
    applied only to the separate visible-fusion cloud; the raw cloud remains
    the input to the native CPD optimizer.  Points are in an optical camera
    frame, so the ray from the camera centre to each point is simply the point
    direction.
    """
    points = np.asarray(points, dtype=np.float64)
    if offset_m <= 0.0 or not len(points):
        return points.copy()
    norms = np.linalg.norm(points, axis=1)
    valid = np.isfinite(points).all(axis=1) & (norms > 1e-9)
    corrected = points.copy()
    directions = np.zeros_like(points, dtype=np.float64)
    directions[valid] = points[valid] / norms[valid, None]
    if normal_only and int(valid.sum()) >= 8:
        # The ray to the camera is not exactly radial on a bent cable.  A
        # small neighbourhood PCA estimates the cable tangent; removing the
        # tangent component keeps the correction from shifting a node along
        # the material coordinate (the source of ordering drift).
        finite_indices = np.flatnonzero(valid)
        finite_points = points[finite_indices]
        try:
            neighbours = cKDTree(finite_points).query(
                finite_points, k=min(12, len(finite_points)), workers=1
            )[1]
            local = finite_points[np.asarray(neighbours, dtype=np.int64)]
            local_zero = local - local.mean(axis=1, keepdims=True)
            covariance = np.einsum("nki,nkj->nij", local_zero, local_zero)
            _, vectors = np.linalg.eigh(covariance)
            tangents = vectors[:, :, -1]
            rays = directions[finite_indices]
            radial = rays - tangents * np.einsum("ni,ni->n", rays, tangents)[:, None]
            radial_norm = np.linalg.norm(radial, axis=1)
            good = radial_norm > 1e-6
            if np.any(good):
                directions[finite_indices[good]] = radial[good] / radial_norm[good, None]
        except (np.linalg.LinAlgError, ValueError, IndexError):
            pass
    corrected[valid] += float(offset_m) * directions[valid]
    return corrected


def _secondary_visible_node_indices(
    nodes_primary: np.ndarray | None,
    world_from_primary: np.ndarray,
    secondary_world_from_camera: np.ndarray,
    secondary_intrinsics: np.ndarray,
    secondary_mask: np.ndarray | None,
    secondary_points: np.ndarray | None,
    *,
    mask_radius_px: int = 8,
    min_mask_pixels: int = 3,
    support_radius_m: float = 0.030,
    min_neighbors: int = 3,
) -> np.ndarray:
    """Return material-node indices supported by the secondary RGB-D view.

    The primary tracker normally projects its visibility test into only the
    primary image.  In a dual-camera run this would discard a wrist-supported
    node that is outside the global view.  This helper performs the same two
    conservative checks in the secondary camera: a cable-mask patch and local
    3-D cloud support.  It returns *indices in the primary node ordering*; the
    caller uses them only for post-CPD visible fusion.
    """
    if nodes_primary is None or secondary_mask is None or secondary_points is None:
        return np.empty(0, dtype=np.int32)
    nodes_primary = np.asarray(nodes_primary, dtype=np.float64)
    points = np.asarray(secondary_points, dtype=np.float64)
    if nodes_primary.ndim != 2 or nodes_primary.shape[1] != 3 or not len(nodes_primary):
        return np.empty(0, dtype=np.int32)
    finite_points = points[np.isfinite(points).all(axis=1)]
    if not len(finite_points):
        return np.empty(0, dtype=np.int32)
    # Primary-camera nodes -> world -> secondary optical camera frame.
    nodes_world = transform_points(world_from_primary, nodes_primary)
    nodes_secondary = transform_points(
        np.linalg.inv(secondary_world_from_camera), nodes_world
    )
    intrinsics = np.asarray(secondary_intrinsics, dtype=np.float64).reshape(3, 3)
    pixels = np.full((len(nodes_secondary), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(nodes_secondary).all(axis=1) & (nodes_secondary[:, 2] > 1e-8)
    pixels[valid, 0] = (
        intrinsics[0, 0] * nodes_secondary[valid, 0] / nodes_secondary[valid, 2]
        + intrinsics[0, 2]
    )
    pixels[valid, 1] = (
        intrinsics[1, 1] * nodes_secondary[valid, 1] / nodes_secondary[valid, 2]
        + intrinsics[1, 2]
    )
    mask = np.asarray(secondary_mask) > 0
    height, width = mask.shape[:2]
    mask_support = np.zeros(len(nodes_secondary), dtype=bool)
    radius = max(int(mask_radius_px), 0)
    for index, pixel in enumerate(pixels):
        if not np.isfinite(pixel).all():
            continue
        col, row = np.rint(pixel).astype(int)
        if not (0 <= row < height and 0 <= col < width):
            continue
        row_min, row_max = max(0, row - radius), min(height, row + radius + 1)
        col_min, col_max = max(0, col - radius), min(width, col + radius + 1)
        mask_support[index] = int(mask[row_min:row_max, col_min:col_max].sum()) >= int(
            min_mask_pixels
        )
    # A cylindrical surface sample is up to roughly one radius away from its
    # centreline.  Use a 30-mm gate and a local-count fallback so voxelisation
    # does not reject a valid wrist-supported segment.
    distances = np.linalg.norm(nodes_secondary[:, None, :] - finite_points[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    counts = np.sum(distances <= float(support_radius_m), axis=1)
    cloud_support = (nearest <= float(support_radius_m)) | (counts >= int(min_neighbors))
    return np.flatnonzero(valid & mask_support & cloud_support).astype(np.int32)


def _current_skeleton_centerline_paths(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    offset_m: float = 0.014,
) -> list[np.ndarray]:
    """Extract current-frame RGB-D centerline paths.

    This deliberately does not use the previous TrackDLO chain.  The HSV
    mask is skeletonized in the current image, depth is sampled at the
    skeleton pixels, and the camera-facing surface samples are shifted toward
    the cable centre.  It is used as a *visible-observation* cloud; the raw
    RGB-D cloud remains available to native CPD for temporal completion.
    """
    from trackdlo_standalone.geometry import depth_to_meters
    from trackdlo_standalone.initialization import (
        depth_skeleton_paths,
        depth_skeleton_paths_global,
        ordered_skeleton_pixels,
        ordered_skeleton_pixels_depth,
        segment_hsv,
    )

    rgb = np.asarray(rgb)
    depth_m = depth_to_meters(depth)
    mask = segment_hsv(rgb, (112, 180, 80), (130, 255, 255))
    path_candidates: list[np.ndarray] = []
    for extractor in (
        lambda: depth_skeleton_paths_global(mask, depth_m, intrinsics),
        lambda: depth_skeleton_paths(mask, depth_m, intrinsics),
        lambda: [ordered_skeleton_pixels(mask)],
    ):
        try:
            path_candidates.extend(extractor())
        except Exception:
            continue
    if not path_candidates:
        try:
            path_candidates = [ordered_skeleton_pixels_depth(mask, depth_m, intrinsics)]
        except Exception:
            return []
    height, width = depth_m.shape[:2]
    collected: list[np.ndarray] = []
    for pixels in path_candidates:
        points: list[tuple[float, float, float]] = []
        for row, col in np.asarray(pixels, dtype=np.int32):
            if not (0 <= row < height and 0 <= col < width):
                continue
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
                (
                    (float(col) - intrinsics[0, 2]) * value / intrinsics[0, 0],
                    (float(row) - intrinsics[1, 2]) * value / intrinsics[1, 1],
                    value,
                )
            )
        if len(points) < 4:
            continue
        path = np.asarray(points, dtype=np.float64)
        keep = np.concatenate(
            ([True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-7)
        )
        path = path[keep]
        if len(path) >= 4:
            steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
            # A depth discontinuity at a projected crossing can make a
            # skeleton path jump tens of centimetres in one pixel.  Such a
            # path is geometrically close to the cable but not an ordered
            # physical branch, so do not use it for current-frame fusion.
            median_step = max(float(np.median(steps)), 1e-6)
            if float(np.max(steps)) > max(0.04, 12.0 * median_step):
                continue
            collected.append(path)
    if not collected:
        return []
    # Estimate local tangents independently for each disconnected path.  If
    # all paths are concatenated before the PCA correction, a neighbour query
    # can connect two unrelated branches at a crossing and shift one path
    # tangentially.
    return [
        _surface_to_center_points(path, float(offset_m), normal_only=True)
        for path in collected
    ]


def _current_skeleton_centerline_points(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    offset_m: float = 0.014,
) -> np.ndarray:
    """Flatten current-frame centerline paths into an observation cloud."""
    paths = _current_skeleton_centerline_paths(rgb, depth, intrinsics, offset_m)
    return (
        np.concatenate(paths, axis=0)
        if paths
        else np.empty((0, 3), dtype=np.float64)
    )


def _current_component_centerline_paths(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    offset_m: float = 0.014,
) -> list[np.ndarray]:
    """Extract all current-frame RGB-D cable components as ordered paths."""

    from trackdlo_standalone.current_frame import extract_current_component_paths
    from trackdlo_standalone.geometry import depth_to_meters
    from trackdlo_standalone.initialization import segment_hsv

    mask = segment_hsv(rgb, (112, 180, 80), (130, 255, 255))
    return extract_current_component_paths(
        mask,
        depth_to_meters(depth),
        intrinsics,
        surface_offset_m=float(offset_m),
    )


def _annotate_video_frame(frame: np.ndarray, scenario: str, episode: str, frame_index: int) -> np.ndarray:
    out = frame.copy()
    cv2.putText(
        out,
        f"current run  {scenario}  {episode}  source_frame={frame_index}",
        (12, out.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _fuse_independent_camera_results(
    primary,
    secondary,
    secondary_to_primary: np.ndarray,
    combined_points: np.ndarray,
):
    """Fuse two independently tracked ordered chains by visibility.

    Each camera keeps its own CPD history and node ordering.  The secondary
    chain is transformed into the primary camera frame and oriented by its
    endpoints.  A node is replaced by the secondary estimate only when the
    primary camera does not support it; when both cameras support it, the
    estimates are averaged.  Nodes unsupported by either view remain the
    primary chain, which is the explicit history-completion branch.
    """
    from trackdlo_standalone import TrackResult

    primary_nodes = np.asarray(primary.nodes_camera, dtype=np.float64)
    secondary_nodes = (
        np.asarray(secondary.nodes_camera, dtype=np.float64)
        if secondary is not None
        else None
    )
    if secondary_nodes.shape == primary_nodes.shape and len(secondary_nodes):
        secondary_nodes = transform_points(secondary_to_primary, secondary_nodes)
        direct = float(
            np.linalg.norm(secondary_nodes[0] - primary_nodes[0])
            + np.linalg.norm(secondary_nodes[-1] - primary_nodes[-1])
        )
        reverse = float(
            np.linalg.norm(secondary_nodes[-1] - primary_nodes[0])
            + np.linalg.norm(secondary_nodes[0] - primary_nodes[-1])
        )
        if reverse < direct:
            secondary_nodes = secondary_nodes[::-1]
    else:
        secondary_nodes = None

    primary_visible = set(int(i) for i in np.asarray(primary.visible_nodes).ravel())
    secondary_visible = (
        set(int(i) for i in np.asarray(secondary.visible_nodes).ravel())
        if secondary is not None and secondary_nodes is not None
        else set()
    )
    fused = primary_nodes.copy()
    fused_count = 0
    if secondary_nodes is not None:
        for index in range(len(fused)):
            p_supported = index in primary_visible
            s_supported = index in secondary_visible
            if not s_supported:
                continue
            # Independent chains can occasionally choose different branches
            # at a crossing.  Reject a grossly inconsistent secondary node;
            # the primary/history estimate is safer in that case.
            disagreement = float(np.linalg.norm(secondary_nodes[index] - fused[index]))
            if disagreement > 0.08:
                continue
            if p_supported:
                fused[index] = 0.5 * (fused[index] + secondary_nodes[index])
            else:
                fused[index] = secondary_nodes[index]
            fused_count += 1

    visible_union = np.asarray(
        sorted(primary_visible | secondary_visible), dtype=np.int32
    )
    hidden_intersection = np.asarray(
        sorted(set(range(len(fused))) - set(int(i) for i in visible_union)),
        dtype=np.int32,
    )
    reasons = [
        value
        for value in (
            primary.failure_reason,
            None if secondary is None else secondary.failure_reason,
        )
        if value
    ]
    return TrackResult(
        nodes_camera=fused,
        mask=primary.mask,
        observed_points_camera=np.asarray(combined_points, dtype=np.float64),
        visible_nodes=visible_union,
        self_occluded_nodes=hidden_intersection,
        initialized=bool(primary.initialized or (secondary is not None and secondary.initialized)),
        tracking_ok=bool(primary.tracking_ok or (secondary is not None and secondary.tracking_ok)),
        nonconverged=bool(primary.nonconverged or (secondary is not None and secondary.nonconverged)),
        reinitialized=bool(primary.reinitialized or (secondary is not None and secondary.reinitialized)),
        failure_reason=";".join(reasons) if reasons else None,
        preprocess_ms=float(primary.preprocess_ms + (0.0 if secondary is None else secondary.preprocess_ms)),
        tracking_ms=float(primary.tracking_ms + (0.0 if secondary is None else secondary.tracking_ms)),
        total_ms=float(primary.total_ms + (0.0 if secondary is None else secondary.total_ms)),
        observation_fused_nodes=int(fused_count),
    )


def run_current_trackdlo(
    *,
    run_root: Path,
    project_src: Path,
    trackdlo_root: Path,
    output: Path,
    robot: str,
    nero_base_offset_x: float,
    nero_tcp_dx: float,
    nero_tcp_dy: float,
    nero_tcp_dz: float,
    camera: str,
    dual_camera: bool,
    dual_independent: bool,
    dual_visible_primary: bool,
    dual_visible_secondary: bool,
    dual_visible_secondary_all: bool,
    surface_to_center_offset_m: float,
    surface_to_center_normal_only: bool,
    surface_to_center_for_cpd: bool,
    scenarios: list[str],
    episode_names: list[str] | None,
    episodes_per_scenario: int | None,
    frame_stride: int,
    init_seconds: float,
    video_episodes_per_scenario: int,
    no_reinitialize: bool,
    reinitialize_grace_seconds: float,
    force_sparse_update: bool,
    hold_last_on_failure: bool,
    accept_nonconverged: bool,
    min_visible_nodes: int,
    visibility_mode: str,
    visibility_threshold: float,
    visibility_weak_threshold: float,
    visibility_neighborhood_radius: float,
    visibility_min_neighbors: int,
    alpha: float,
    max_iter: int,
    visible_observation_fusion: bool,
    visible_observation_blend: float,
    visible_observation_radius: float,
    visible_observation_min_points: int,
    visible_observation_adaptive_radius: bool,
    visible_observation_fallback_radius: float,
    visible_observation_support_target: int,
    visible_observation_arc_window_scale: float,
    visible_observation_arc_window_min: float,
    visible_observation_motion_threshold: float,
    visible_observation_residual_margin: float,
    visible_observation_normal_only: bool,
    visible_observation_tangent_blend: float,
    visible_observation_topology: str,
    visible_observation_centerline_fit: bool,
    visible_observation_use_extended: bool,
    visible_pixel_snap_fusion: bool,
    visible_pixel_snap_blend: float,
    visible_pixel_snap_radius_px: float,
    visible_pixel_snap_center_offset_m: float,
    monotonic_visible_fusion: bool,
    monotonic_tangent_blend: float,
    refine_visibility_after_cpd: bool,
    hidden_history_completion: bool,
    hidden_history_blend: float,
    hidden_history_deformation_threshold: float,
    hidden_piecewise_completion: bool,
    hidden_piecewise_blend: float,
    hidden_piecewise_max_anchor_disagreement: float,
    hidden_piecewise_max_gap_nodes: int,
    hidden_hold_previous: bool,
    hidden_velocity_prediction: bool,
    hidden_velocity_decay: float,
    hidden_velocity_deformation_threshold_m: float,
    hidden_velocity_max_step_m: float,
    pointcloud_path_fusion: bool,
    pointcloud_path_blend: float,
    pointcloud_arc_fusion: bool,
    pointcloud_arc_anchor_only: bool,
    max_observed_points: int,
    observed_point_sampling: str,
    cpd_stride: int,
    hold_hidden_on_nonconverged: bool,
    fast_observation_only: bool,
    adaptive_history_fusion: bool,
    adaptive_visible_alpha: float,
    adaptive_occluded_alpha: float,
    adaptive_history_deformation_switch: bool,
    adaptive_history_deformation_threshold: float,
    adaptive_history_deformation_occluded_alpha: float,
    adaptive_history_deformation_visible_alpha: float,
    adaptive_history_support_switch: bool,
    adaptive_history_support_threshold: float,
    adaptive_history_support_visible_alpha: float,
    adaptive_cloud_motion_switch: bool,
    adaptive_cloud_motion_icp: bool,
    adaptive_cloud_motion_threshold_m: float,
    adaptive_motion_visible_alpha: float,
    adaptive_motion_occluded_alpha: float,
    adaptive_motion_surface_offset_m: float,
    current_skeleton_fusion: bool,
    current_skeleton_blend: float,
    current_skeleton_simple_path: bool,
    current_skeleton_depth_path: bool,
    current_skeleton_cloud: bool,
    current_path_fusion: bool,
    current_path_blend: float,
    current_path_use_candidate: bool,
    current_path_run_locked: bool,
    pre_cpd_visible_seed: bool,
    pre_cpd_seed_blend: float,
    post_cpd_current_spline: bool,
    current_component_paths: bool,
    save_node_traces: bool,
    dlo_pixel_width: int,
    downsample_leaf_size: float,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
) -> dict:
    trackdlo_src = trackdlo_root.resolve() / "src"
    project_src = project_src.resolve()
    for path in (trackdlo_src, project_src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from panda_cable_grasp.env.environment import CableGraspEnv, ROBOT_SPECS
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario
    from trackdlo_standalone import TrackDLOConfig, TrackDLOTracker
    from trackdlo_standalone.geometry import backproject_mask, depth_to_meters, voxel_downsample
    from trackdlo_standalone.initialization import segment_hsv
    from trackdlo_standalone.metrics import frame_metrics, point_to_polyline_distance, summarize
    from trackdlo_standalone.visualization import (
        render_correspondence_frame,
        render_frame,
        render_pointcloud_frame,
    )

    robot = str(robot).lower()
    if robot == "nero":
        # The panda_like NERO videos were rendered with the calibration used
        # by the 2026-09-12 dataset: base x=0.35 m and TCP x offset=+0.01 m.
        nominal = ROBOT_SPECS["nero"]
        ROBOT_SPECS["nero"] = replace(
            nominal,
            base_offset=(float(nero_base_offset_x), 0.0, 0.0),
            grasp_center_local=(
                0.1733 + float(nero_tcp_dx),
                float(nero_tcp_dy),
                -0.0235 + float(nero_tcp_dz),
            ),
        )

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / f"trackdlo_{camera}_4scenes.mp4"
    pointcloud_video_path = output / f"trackdlo_{camera}_pointcloud_4scenes.mp4"
    correspondence_video_path = output / f"trackdlo_{camera}_correspondence_4scenes.mp4"
    video_writer = None
    pointcloud_video_writer = None
    correspondence_video_writer = None
    rows: list[dict] = []
    selected_manifest: list[dict] = []
    sequence_summaries: list[dict] = []
    video_written = 0
    node_trace_predictions: list[np.ndarray] = []
    node_trace_truth: list[np.ndarray] = []
    node_trace_visible: list[np.ndarray] = []
    node_trace_scenarios: list[str] = []
    node_trace_frames: list[int] = []

    for scenario_name in scenarios:
        episode_dirs = discover_episode_dirs(run_root, scenario_name)
        if episode_names:
            episode_dirs = [episode for episode in episode_dirs if episode.name in episode_names]
        episode_dirs = choose_evenly_spaced(episode_dirs, episodes_per_scenario)
        if not episode_dirs:
            continue
        with (episode_dirs[0] / "episode.json").open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        config = env_config_for_scenario(
            get_scenario(scenario_name),
            seed=int(metadata["result"]["requested_seed"]),
            episode_seconds=15.0,
            robot=robot,
        )
        config.dynamicvla_cameras_enabled = True
        env = CableGraspEnv(config)
        renderer = mujoco.Renderer(env.model, height=360, width=480)
        try:
            if camera == "opst":
                camera_id = int(env.dynamicvla_opst_camera_id)
                camera_name = env.config.dynamicvla_opst_camera_name
                video_name = "global.mp4"
            else:
                camera_id = int(env.dynamicvla_wrist_camera_id)
                camera_name = env.config.dynamicvla_wrist_camera_name
                video_name = "wrist.mp4"
            intrinsics = camera_matrix(480, 360, float(env.model.cam_fovy[camera_id]))
            secondary_camera_id = None
            secondary_camera_name = None
            secondary_video_name = None
            secondary_intrinsics = None
            if dual_camera:
                if camera == "opst":
                    secondary_camera_id = int(env.dynamicvla_wrist_camera_id)
                    secondary_camera_name = env.config.dynamicvla_wrist_camera_name
                    secondary_video_name = "wrist.mp4"
                else:
                    secondary_camera_id = int(env.dynamicvla_opst_camera_id)
                    secondary_camera_name = env.config.dynamicvla_opst_camera_name
                    secondary_video_name = "global.mp4"
                secondary_intrinsics = camera_matrix(
                    480, 360, float(env.model.cam_fovy[secondary_camera_id])
                )

            for episode_index, episode_dir in enumerate(episode_dirs):
                trajectory = np.load(episode_dir / "trajectory.npz", allow_pickle=False)
                state_spec = mujoco.mjtState(int(trajectory["state_spec"]))
                frame_count = min(
                    len(trajectory["frame_state_indices"]),
                    int(cv2.VideoCapture(str(episode_dir / video_name)).get(cv2.CAP_PROP_FRAME_COUNT)),
                )
                if dual_camera and secondary_video_name is not None:
                    frame_count = min(
                        frame_count,
                        int(
                            cv2.VideoCapture(
                                str(episode_dir / secondary_video_name)
                            ).get(cv2.CAP_PROP_FRAME_COUNT)
                        ),
                    )
                frame_times = np.asarray(trajectory["frame_times"], dtype=np.float64)[:frame_count]
                frame_state_indices = np.asarray(
                    trajectory["frame_state_indices"], dtype=np.int64
                )[:frame_count]
                if frame_count < 2:
                    trajectory.close()
                    continue
                init_frame = int(
                    np.searchsorted(
                        frame_times,
                        float(frame_times[0]) + float(init_seconds),
                        side="left",
                    )
                )
                selected_indices = np.arange(
                    init_frame, frame_count, max(1, int(frame_stride)), dtype=np.int64
                )
                if not len(selected_indices):
                    trajectory.close()
                    continue
                selected_manifest.append(
                    {
                        "scenario": scenario_name,
                        "episode": episode_dir.name,
                        "frames": int(len(selected_indices)),
                        "init_source_frame": init_frame,
                    }
                )

                track_config = TrackDLOConfig(
                    hsv_lower=tuple(int(value) for value in hsv_lower),
                    hsv_upper=tuple(int(value) for value in hsv_upper),
                    reinitialize_after_failures=0 if no_reinitialize else 3,
                    dlo_pixel_width=int(dlo_pixel_width),
                    downsample_leaf_size=float(downsample_leaf_size),
                    force_sparse_update=bool(force_sparse_update),
                    hold_last_on_failure=bool(hold_last_on_failure),
                    accept_nonconverged=bool(accept_nonconverged),
                    min_visible_nodes=int(min_visible_nodes),
                    visibility_mode=str(visibility_mode),
                    visibility_threshold=float(visibility_threshold),
                    visibility_weak_threshold=float(visibility_weak_threshold),
                    visibility_neighborhood_radius=float(visibility_neighborhood_radius),
                    visibility_min_neighbors=int(visibility_min_neighbors),
                    alpha=float(alpha),
                    max_iter=max(1, int(max_iter)),
                    visible_observation_fusion=bool(visible_observation_fusion),
                    visible_observation_blend=float(visible_observation_blend),
                    visible_observation_radius=float(visible_observation_radius),
                    visible_observation_min_points=int(visible_observation_min_points),
                    visible_observation_adaptive_radius=bool(
                        visible_observation_adaptive_radius
                    ),
                    visible_observation_fallback_radius=float(
                        visible_observation_fallback_radius
                    ),
                    visible_observation_support_target=int(
                        visible_observation_support_target
                    ),
                    visible_observation_arc_window_scale=float(
                        visible_observation_arc_window_scale
                    ),
                    visible_observation_arc_window_min=float(
                        visible_observation_arc_window_min
                    ),
                    visible_observation_motion_threshold=float(visible_observation_motion_threshold),
                    visible_observation_residual_margin=float(visible_observation_residual_margin),
                    visible_observation_normal_only=bool(visible_observation_normal_only),
                    visible_observation_tangent_blend=float(visible_observation_tangent_blend),
                    visible_observation_topology=str(visible_observation_topology),
                    visible_observation_centerline_fit=bool(visible_observation_centerline_fit),
                    visible_observation_use_extended=bool(visible_observation_use_extended),
                    visible_pixel_snap_fusion=bool(visible_pixel_snap_fusion),
                    visible_pixel_snap_blend=float(visible_pixel_snap_blend),
                    visible_pixel_snap_radius_px=float(visible_pixel_snap_radius_px),
                    visible_pixel_snap_center_offset_m=float(visible_pixel_snap_center_offset_m),
                    monotonic_visible_fusion=bool(monotonic_visible_fusion),
                    monotonic_tangent_blend=float(monotonic_tangent_blend),
                    refine_visibility_after_cpd=bool(refine_visibility_after_cpd),
                    hidden_history_completion=bool(hidden_history_completion),
                    hidden_history_blend=float(hidden_history_blend),
                    hidden_history_deformation_threshold=float(hidden_history_deformation_threshold),
                    hidden_piecewise_completion=bool(hidden_piecewise_completion),
                    hidden_piecewise_blend=float(hidden_piecewise_blend),
                    hidden_piecewise_max_anchor_disagreement=float(hidden_piecewise_max_anchor_disagreement),
                    hidden_piecewise_max_gap_nodes=int(hidden_piecewise_max_gap_nodes),
                    hidden_hold_previous=bool(hidden_hold_previous),
                    hidden_velocity_prediction=bool(hidden_velocity_prediction),
                    hidden_velocity_decay=float(hidden_velocity_decay),
                    hidden_velocity_deformation_threshold_m=float(hidden_velocity_deformation_threshold_m),
                    hidden_velocity_max_step_m=float(hidden_velocity_max_step_m),
                    pointcloud_path_fusion=bool(pointcloud_path_fusion),
                    pointcloud_path_blend=float(pointcloud_path_blend),
                    pointcloud_arc_fusion=bool(pointcloud_arc_fusion),
                    pointcloud_arc_anchor_only=bool(pointcloud_arc_anchor_only),
                    max_observed_points=int(max_observed_points),
                    observed_point_sampling=str(observed_point_sampling),
                    cpd_stride=max(1, int(cpd_stride)),
                    hold_hidden_on_nonconverged=bool(hold_hidden_on_nonconverged),
                    fast_observation_only=bool(fast_observation_only),
                    adaptive_history_fusion=bool(adaptive_history_fusion),
                    adaptive_visible_alpha=float(adaptive_visible_alpha),
                    adaptive_occluded_alpha=float(adaptive_occluded_alpha),
                    adaptive_history_deformation_switch=bool(adaptive_history_deformation_switch),
                    adaptive_history_deformation_threshold=float(adaptive_history_deformation_threshold),
                    adaptive_history_deformation_occluded_alpha=float(adaptive_history_deformation_occluded_alpha),
                    adaptive_history_deformation_visible_alpha=float(adaptive_history_deformation_visible_alpha),
                    adaptive_history_support_switch=bool(adaptive_history_support_switch),
                    adaptive_history_support_threshold=float(adaptive_history_support_threshold),
                    adaptive_history_support_visible_alpha=float(adaptive_history_support_visible_alpha),
                    adaptive_cloud_motion_switch=bool(adaptive_cloud_motion_switch),
                    adaptive_cloud_motion_icp=bool(adaptive_cloud_motion_icp),
                    adaptive_cloud_motion_threshold_m=float(adaptive_cloud_motion_threshold_m),
                    adaptive_motion_visible_alpha=float(adaptive_motion_visible_alpha),
                    adaptive_motion_occluded_alpha=float(adaptive_motion_occluded_alpha),
                    current_skeleton_fusion=bool(current_skeleton_fusion),
                    current_skeleton_blend=float(current_skeleton_blend),
                    current_skeleton_simple_path=bool(current_skeleton_simple_path),
                    current_skeleton_depth_path=bool(current_skeleton_depth_path),
                    current_path_fusion=bool(current_path_fusion),
                    current_path_blend=float(current_path_blend),
                    current_path_use_candidate=bool(current_path_use_candidate),
                    current_path_run_locked=bool(current_path_run_locked),
                    pre_cpd_visible_seed=bool(pre_cpd_visible_seed),
                    pre_cpd_seed_blend=float(pre_cpd_seed_blend),
                    post_cpd_current_spline=bool(post_cpd_current_spline),
                )
                tracker = TrackDLOTracker(intrinsics, track_config)
                secondary_tracker = (
                    TrackDLOTracker(secondary_intrinsics, track_config)
                    if dual_camera and dual_independent and secondary_intrinsics is not None
                    else None
                )
                capture = cv2.VideoCapture(str(episode_dir / video_name))
                if not capture.isOpened():
                    raise RuntimeError(f"could not open {episode_dir / video_name}")
                secondary_capture = None
                if dual_camera and secondary_video_name is not None:
                    secondary_capture = cv2.VideoCapture(
                        str(episode_dir / secondary_video_name)
                    )
                    if not secondary_capture.isOpened():
                        raise RuntimeError(
                            f"could not open {episode_dir / secondary_video_name}"
                        )
                make_video = episode_index < max(0, int(video_episodes_per_scenario))
                # Metrics may be sampled for a faster sweep, but a visualization
                # episode must still advance TrackDLO on every source frame.
                selected_set = set(map(int, selected_indices))
                video_set = set(range(init_frame, frame_count)) if make_video else set()
                process_set = selected_set | video_set
                if make_video and video_writer is None:
                    fps = float(
                        cv2.VideoCapture(str(episode_dir / video_name)).get(
                            cv2.CAP_PROP_FPS
                        )
                        or 25.0
                    )
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(fps, 1.0),
                        (960, 360),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"could not open {video_path}")
                    pointcloud_video_writer = cv2.VideoWriter(
                        str(pointcloud_video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(fps, 1.0),
                        (960, 360),
                    )
                    if not pointcloud_video_writer.isOpened():
                        raise RuntimeError(f"could not open {pointcloud_video_path}")
                    correspondence_video_writer = cv2.VideoWriter(
                        str(correspondence_video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(fps, 1.0),
                        (960, 360),
                    )
                    if not correspondence_video_writer.isOpened():
                        raise RuntimeError(f"could not open {correspondence_video_path}")

                sequence_rows: list[dict] = []
                previous_cloud_centroid: np.ndarray | None = None
                previous_motion_points: np.ndarray | None = None
                for frame_index in range(frame_count):
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    secondary_bgr = None
                    if secondary_capture is not None:
                        secondary_ok, secondary_bgr = secondary_capture.read()
                        if not secondary_ok:
                            break
                    if frame_index not in process_set:
                        continue
                    # Optionally allow recovery only during the initial
                    # tracking grace period.  The timer starts at the first
                    # processed frame (after init_seconds), rather than at
                    # source frame zero, so this means one second of actual
                    # tracking time.
                    if reinitialize_grace_seconds > 0.0 and not no_reinitialize:
                        grace_end = float(frame_times[init_frame]) + float(
                            reinitialize_grace_seconds
                        )
                        # TrackDLOConfig is intentionally frozen so that
                        # per-run settings cannot drift.  This one threshold
                        # is an explicit time-gated runtime switch, so bypass
                        # the frozen dataclass guard only for this field.
                        object.__setattr__(
                            track_config,
                            "reinitialize_after_failures",
                            3
                            if float(frame_times[frame_index]) <= grace_end
                            else 0,
                        )
                    state_index = int(frame_state_indices[frame_index])
                    mujoco.mj_setState(
                        env.model, env.data, trajectory["states"][state_index], state_spec
                    )
                    mujoco.mj_forward(env.model, env.data)
                    depth = _render_depth(renderer, env.data, camera_name)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    truth_world = env.data.xpos[env.cable_ids].copy()
                    world_from_camera = world_from_camera_optical(
                        env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id]
                    )
                    truth_camera = transform_points(
                        np.linalg.inv(world_from_camera), truth_world
                    )
                    algorithm_start = perf_counter_ns()
                    observed_points_override = None
                    secondary_points_for_fusion = None
                    primary_points = None
                    secondary_rgb = None
                    secondary_depth = None
                    skeleton_points_override = None
                    ordered_path_override = None
                    ordered_paths_override = None
                    secondary_visible_nodes = None
                    secondary_points_primary_for_fusion = None
                    cloud_motion_m: float | None = None
                    effective_surface_offset_m = float(surface_to_center_offset_m)
                    if (
                        secondary_bgr is not None
                        and secondary_camera_id is not None
                        and secondary_camera_name is not None
                        and secondary_intrinsics is not None
                    ):
                        secondary_rgb = cv2.cvtColor(
                            secondary_bgr, cv2.COLOR_BGR2RGB
                        )
                        secondary_depth = _render_depth(
                            renderer, env.data, secondary_camera_name
                        )
                        primary_mask = segment_hsv(rgb, hsv_lower, hsv_upper)
                        secondary_mask = segment_hsv(
                            secondary_rgb, hsv_lower, hsv_upper
                        )
                        primary_points = backproject_mask(
                            depth_to_meters(depth), primary_mask, intrinsics
                        )
                        secondary_points = backproject_mask(
                            depth_to_meters(secondary_depth),
                            secondary_mask,
                            secondary_intrinsics,
                        )
                        secondary_world_from_camera = world_from_camera_optical(
                            env.data.cam_xpos[secondary_camera_id],
                            env.data.cam_xmat[secondary_camera_id],
                        )
                        secondary_world = transform_points(
                            secondary_world_from_camera, secondary_points
                        )
                        secondary_in_primary = transform_points(
                            np.linalg.inv(world_from_camera), secondary_world
                        )
                        # Keep the separate wrist fusion cloud light.  The raw
                        # mask can contain several thousand depth pixels; a
                        # deterministic stride preserves coverage while
                        # avoiding a second full-size voxel hash in the tracker.
                        if len(secondary_in_primary) > 2000:
                            stride = int(np.ceil(len(secondary_in_primary) / 2000.0))
                            secondary_points_primary_for_fusion = secondary_in_primary[::stride]
                        else:
                            secondary_points_primary_for_fusion = secondary_in_primary
                        observed_points_override = voxel_downsample(
                            np.concatenate(
                                (primary_points, secondary_in_primary), axis=0
                            ),
                            float(downsample_leaf_size),
                        )
                        secondary_points_for_fusion = secondary_points
                        if (
                            dual_camera
                            and not dual_independent
                            and dual_visible_secondary
                            and tracker.nodes is not None
                        ):
                            secondary_visible_nodes = _secondary_visible_node_indices(
                                tracker.nodes,
                                world_from_camera,
                                secondary_world_from_camera,
                                secondary_intrinsics,
                                secondary_mask,
                                secondary_points,
                                mask_radius_px=track_config.visibility_mask_radius_px,
                                min_mask_pixels=track_config.visibility_min_mask_pixels,
                                support_radius_m=max(
                                    0.012,
                                    float(track_config.visibility_threshold),
                                ),
                                min_neighbors=track_config.visibility_min_neighbors,
                            )
                        if current_skeleton_cloud or current_path_fusion:
                            path_extractor = (
                                _current_component_centerline_paths
                                if current_component_paths
                                else _current_skeleton_centerline_paths
                            )
                            primary_skeleton_paths = path_extractor(rgb, depth, intrinsics)
                            secondary_skeleton_paths = path_extractor(
                                secondary_rgb, secondary_depth, secondary_intrinsics
                            )
                            primary_skeleton = (
                                np.concatenate(primary_skeleton_paths, axis=0)
                                if primary_skeleton_paths
                                else np.empty((0, 3), dtype=np.float64)
                            )
                            secondary_skeleton = (
                                np.concatenate(secondary_skeleton_paths, axis=0)
                                if secondary_skeleton_paths
                                else np.empty((0, 3), dtype=np.float64)
                            )
                            if len(secondary_skeleton):
                                secondary_skeleton_world = transform_points(
                                    secondary_world_from_camera, secondary_skeleton
                                )
                                secondary_skeleton_primary = transform_points(
                                    np.linalg.inv(world_from_camera),
                                    secondary_skeleton_world,
                                )
                            else:
                                secondary_skeleton_primary = np.empty(
                                    (0, 3), dtype=np.float64
                                )
                            skeleton_parts = [
                                item
                                for item in (primary_skeleton, secondary_skeleton_primary)
                                if len(item)
                            ]
                            if skeleton_parts:
                                skeleton_points_override = np.concatenate(
                                    skeleton_parts, axis=0
                                )
                            if current_path_fusion:
                                transformed_paths = list(primary_skeleton_paths)
                                for path in secondary_skeleton_paths:
                                    transformed_paths.append(
                                        transform_points(
                                            np.linalg.inv(world_from_camera)
                                            @ secondary_world_from_camera,
                                            path,
                                        )
                                    )
                                if transformed_paths:
                                    transformed_paths.sort(
                                        key=lambda value: float(
                                            np.linalg.norm(np.diff(value, axis=0), axis=1).sum()
                                        ),
                                        reverse=True,
                                    )
                                    ordered_paths_override = transformed_paths
                                    ordered_path_override = max(
                                        transformed_paths,
                                        key=lambda value: float(
                                            np.linalg.norm(np.diff(value, axis=0), axis=1).sum()
                                        )
                                    )
                    # Use only a scalar current-cloud centroid displacement
                    # as an optional motion cue.  This avoids feeding the
                    # previous DLO geometry back into the adaptive decision.
                    # Use the primary/global cloud for the scalar motion cue.
                    # The merged wrist cloud can have a different viewpoint
                    # and its median is not a stable scene-motion estimate.
                    motion_points = primary_points if primary_points is not None else observed_points_override
                    if motion_points is None:
                        motion_mask = segment_hsv(rgb, hsv_lower, hsv_upper)
                        motion_points = backproject_mask(
                            depth_to_meters(depth), motion_mask, intrinsics
                        )
                    finite_motion_points = np.asarray(motion_points, dtype=np.float64)
                    finite_motion_points = finite_motion_points[
                        np.isfinite(finite_motion_points).all(axis=1)
                    ]
                    if len(finite_motion_points):
                        cloud_centroid = np.median(finite_motion_points, axis=0)
                        if previous_cloud_centroid is not None:
                            cloud_motion_m = float(
                                np.linalg.norm(cloud_centroid - previous_cloud_centroid)
                            )
                        if adaptive_cloud_motion_icp and previous_motion_points is not None:
                            # A nearest-neighbour rigid fit removes camera/object
                            # translation and rotation.  The residual is therefore
                            # sensitive to bending while remaining independent of
                            # the estimated DLO state.
                            icp_motion_m = _cloud_rigid_fit_residual(
                                finite_motion_points, previous_motion_points
                            )
                            if icp_motion_m is not None and np.isfinite(icp_motion_m):
                                cloud_motion_m = float(icp_motion_m)
                        previous_cloud_centroid = cloud_centroid
                        previous_motion_points = finite_motion_points.copy()
                    if (
                        adaptive_cloud_motion_switch
                        and cloud_motion_m is not None
                        and cloud_motion_m
                        > max(float(adaptive_cloud_motion_threshold_m), 0.0)
                    ):
                        effective_surface_offset_m = float(adaptive_motion_surface_offset_m)

                    if (
                        dual_independent
                        and secondary_tracker is not None
                        and primary_points is not None
                        and secondary_points_for_fusion is not None
                        and secondary_rgb is not None
                        and secondary_depth is not None
                    ):
                        primary_result = tracker.update(
                            rgb,
                            depth,
                            observed_points_override=primary_points,
                            cloud_motion_m=cloud_motion_m,
                        )
                        secondary_world_from_camera = world_from_camera_optical(
                            env.data.cam_xpos[secondary_camera_id],
                            env.data.cam_xmat[secondary_camera_id],
                        )
                        if not secondary_tracker.initialized:
                            secondary_result = None
                            try:
                                primary_world_nodes = transform_points(
                                    world_from_camera, primary_result.nodes_camera
                                )
                                secondary_nodes = transform_points(
                                    np.linalg.inv(secondary_world_from_camera),
                                    primary_world_nodes,
                                )
                                secondary_result = secondary_tracker.initialize_from_nodes(
                                    secondary_nodes, secondary_rgb, secondary_depth
                                )
                            except (RuntimeError, ValueError, FloatingPointError):
                                secondary_result = None
                        else:
                            secondary_result = secondary_tracker.update(
                                secondary_rgb,
                                secondary_depth,
                                observed_points_override=secondary_points_for_fusion,
                                cloud_motion_m=cloud_motion_m,
                            )
                        secondary_to_primary = (
                            np.linalg.inv(world_from_camera) @ secondary_world_from_camera
                        )
                        result = _fuse_independent_camera_results(
                            primary_result,
                            secondary_result,
                            secondary_to_primary,
                            observed_points_override,
                        )
                        tracker.adopt_nodes(result.nodes_camera)
                    else:
                        if current_skeleton_cloud or current_path_fusion:
                            path_extractor = (
                                _current_component_centerline_paths
                                if current_component_paths
                                else _current_skeleton_centerline_paths
                            )
                            primary_paths = path_extractor(rgb, depth, intrinsics)
                            if primary_paths:
                                ordered_paths_override = primary_paths
                                ordered_path_override = max(
                                    primary_paths,
                                    key=lambda value: float(
                                        np.linalg.norm(np.diff(value, axis=0), axis=1).sum()
                                    ),
                                )
                                skeleton_points_override = np.concatenate(
                                    primary_paths, axis=0
                                )
                        visible_points_override = None
                        if skeleton_points_override is not None and len(skeleton_points_override):
                            # This branch is intentionally current-frame only:
                            # the skeleton samples become the visible cloud,
                            # while raw RGB-D points still drive native CPD.
                            visible_points_override = skeleton_points_override
                        elif float(effective_surface_offset_m) > 0.0:
                            if primary_points is not None and secondary_points_for_fusion is not None:
                                corrected_primary = _surface_to_center_points(
                                    primary_points,
                                    float(effective_surface_offset_m),
                                    normal_only=bool(surface_to_center_normal_only),
                                )
                                corrected_secondary = _surface_to_center_points(
                                    secondary_points_for_fusion,
                                    float(effective_surface_offset_m),
                                    normal_only=bool(surface_to_center_normal_only),
                                )
                                secondary_world = transform_points(
                                    secondary_world_from_camera, corrected_secondary
                                )
                                corrected_secondary_in_primary = transform_points(
                                    np.linalg.inv(world_from_camera), secondary_world
                                )
                                visible_points_override = np.concatenate(
                                    (corrected_primary, corrected_secondary_in_primary), axis=0
                                )
                            elif observed_points_override is not None:
                                visible_points_override = _surface_to_center_points(
                                    observed_points_override,
                                    float(effective_surface_offset_m),
                                    normal_only=bool(surface_to_center_normal_only),
                                )
                        if (
                            bool(surface_to_center_for_cpd)
                            and visible_points_override is not None
                        ):
                            observed_points_override = visible_points_override
                        result = tracker.update(
                            rgb,
                            depth,
                            observed_points_override=observed_points_override,
                            visible_points_override=(
                                visible_points_override
                                if visible_points_override is not None
                                else (
                                    primary_points
                                    if observed_points_override is not None and dual_visible_primary
                                    else None
                                )
                            ),
                            ordered_visible_path_override=ordered_path_override,
                            ordered_visible_paths_override=ordered_paths_override,
                            additional_fusion_visible_nodes=secondary_visible_nodes,
                            additional_visible_points_override=secondary_points_primary_for_fusion,
                            additional_fusion_all_nodes=bool(dual_visible_secondary_all),
                            cloud_motion_m=cloud_motion_m,
                        )
                    algorithm_ms = (perf_counter_ns() - algorithm_start) * 1e-6
                    metrics = frame_metrics(result.nodes_camera, truth_camera)
                    visible_indices = np.asarray(result.visible_nodes, dtype=np.int64)
                    if save_node_traces:
                        visible_mask = np.zeros(len(result.nodes_camera), dtype=np.uint8)
                        valid_visible = visible_indices[
                            (visible_indices >= 0) & (visible_indices < len(visible_mask))
                        ]
                        visible_mask[valid_visible] = 1
                        node_trace_predictions.append(
                            np.asarray(result.nodes_camera, dtype=np.float64).copy()
                        )
                        node_trace_truth.append(np.asarray(truth_camera, dtype=np.float64).copy())
                        node_trace_visible.append(visible_mask)
                        node_trace_scenarios.append(str(scenario_name))
                        node_trace_frames.append(int(frame_index))
                    try:
                        visible_geometric_error_m = float(
                            point_to_polyline_distance(
                                result.nodes_camera[visible_indices], truth_camera
                            ).mean()
                        ) if len(visible_indices) else float("nan")
                    except (IndexError, ValueError, FloatingPointError):
                        visible_geometric_error_m = float("nan")
                    row = {
                        "camera": camera,
                        "scenario": scenario_name,
                        "episode": episode_dir.name,
                        "frame": int(frame_index),
                        "state_index": state_index,
                        "tracking_ok": bool(result.tracking_ok),
                        "nonconverged": bool(result.nonconverged),
                        "reinitialized": bool(result.reinitialized),
                        "failure_reason": result.failure_reason or "",
                        "visible_nodes": int(len(result.visible_nodes)),
                        "secondary_visible_nodes": int(
                            0 if secondary_visible_nodes is None else len(secondary_visible_nodes)
                        ),
                        "observation_fused_nodes": int(result.observation_fused_nodes),
                        "observed_points": int(len(result.observed_points_camera)),
                        "frame_error_m": metrics["frame_error_m"],
                        "ordered_error_m": metrics["ordered_error_m"],
                        "endpoint_error_m": metrics["endpoint_error_m"],
                        "visible_geometric_error_m": visible_geometric_error_m,
                        "predicted_length_m": metrics["predicted_length_m"],
                        "truth_length_m": metrics["truth_length_m"],
                        "length_ratio": metrics["predicted_length_m"]
                        / max(metrics["truth_length_m"], 1e-9),
                        "preprocess_ms": result.preprocess_ms,
                        "tracking_ms": result.tracking_ms,
                        "total_ms": result.total_ms,
                        "benchmark_algorithm_ms": algorithm_ms,
                        "cloud_motion_m": (
                            float("nan") if cloud_motion_m is None else float(cloud_motion_m)
                        ),
                        "deformation_residual_m": float(
                            getattr(result, "deformation_residual_m", float("nan"))
                        ),
                        "deformation_detected": bool(
                            getattr(result, "deformation_detected", False)
                        ),
                        "adaptive_motion_gate": bool(
                            adaptive_cloud_motion_switch
                            and cloud_motion_m is not None
                            and cloud_motion_m > max(float(adaptive_cloud_motion_threshold_m), 0.0)
                        ),
                    }
                    if frame_index in selected_set:
                        rows.append(row)
                        sequence_rows.append(row)

                    if make_video and video_writer is not None and frame_index in video_set:
                        world_bounds = (-0.8, 0.8, -0.8, 0.8)
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
                        visual = _annotate_video_frame(
                            visual, scenario_name, episode_dir.name, int(frame_index)
                        )
                        video_writer.write(visual)
                        pointcloud_visual = render_pointcloud_frame(
                            bgr, result.observed_points_camera, intrinsics
                        )
                        pointcloud_visual = _annotate_video_frame(
                            pointcloud_visual, scenario_name, episode_dir.name, int(frame_index)
                        )
                        pointcloud_video_writer.write(pointcloud_visual)
                        correspondence_visual = render_correspondence_frame(
                            bgr,
                            result.nodes_camera,
                            result.observed_points_camera,
                            result.visible_nodes,
                            result.self_occluded_nodes,
                            intrinsics,
                        )
                        correspondence_visual = _annotate_video_frame(
                            correspondence_visual, scenario_name, episode_dir.name, int(frame_index)
                        )
                        correspondence_video_writer.write(correspondence_visual)
                        video_written += 1
                capture.release()
                if secondary_capture is not None:
                    secondary_capture.release()
                sequence_summary = summarize(sequence_rows)
                sequence_summary.update(
                    {
                        "camera": camera,
                        "scenario": scenario_name,
                        "episode": episode_dir.name,
                        "source_frames": frame_count,
                        "evaluated_frames": len(sequence_rows),
                    }
                )
                sequence_summaries.append(sequence_summary)
                trajectory.close()
        finally:
            renderer.close()
            env.close()

    if video_writer is not None:
        video_writer.release()
    if pointcloud_video_writer is not None:
        pointcloud_video_writer.release()
    if correspondence_video_writer is not None:
        correspondence_video_writer.release()
    if not rows:
        raise RuntimeError("no current-run frames were evaluated")
    _write_csv(output / "per_frame.csv", rows)
    _write_csv(output / "per_sequence.csv", sequence_summaries)
    if save_node_traces and node_trace_predictions:
        np.savez_compressed(
            output / "node_traces.npz",
            predicted=np.asarray(node_trace_predictions, dtype=np.float64),
            truth=np.asarray(node_trace_truth, dtype=np.float64),
            visible=np.asarray(node_trace_visible, dtype=np.uint8),
            scenario=np.asarray(node_trace_scenarios),
            frame=np.asarray(node_trace_frames, dtype=np.int64),
        )
    summary = summarize(rows)
    summary.update(
        {
            "method": "TrackDLO official C++ core via ROS-free wrapper",
            "robot": robot,
            "camera": camera,
            "dual_camera": bool(dual_camera),
            "dual_independent": bool(dual_independent),
            "dual_visible_secondary": bool(dual_visible_secondary),
            "dual_visible_secondary_all": bool(dual_visible_secondary_all),
            "source_run": str(run_root.resolve()),
            "project_src": str(project_src),
            "trackdlo_root": str(trackdlo_root.resolve()),
            "frame_stride": int(frame_stride),
            "episodes_per_scenario": episodes_per_scenario,
            "init_seconds": float(init_seconds),
            "reinitialize_grace_seconds": float(reinitialize_grace_seconds),
            "video": str(video_path) if video_writer is not None else None,
            "pointcloud_video": str(pointcloud_video_path)
            if pointcloud_video_writer is not None
            else None,
            "correspondence_video": str(correspondence_video_path)
            if correspondence_video_writer is not None
            else None,
            "video_frames": int(video_written),
            "selected_episodes": selected_manifest,
            "config": {
                **asdict(track_config),
                "surface_to_center_offset_m": float(surface_to_center_offset_m),
                "surface_to_center_normal_only": bool(surface_to_center_normal_only),
                "surface_to_center_for_cpd": bool(surface_to_center_for_cpd),
                "current_skeleton_cloud": bool(current_skeleton_cloud),
                "current_path_fusion": bool(current_path_fusion),
                "current_path_blend": float(current_path_blend),
                "current_component_paths": bool(current_component_paths),
                "adaptive_motion_surface_offset_m": float(adaptive_motion_surface_offset_m),
            },
            "evaluation_policy": "MuJoCo ground truth is read after TrackDLO inference for metrics and overlay only.",
        }
    )
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run TrackDLO on the current run's global/wrist RGB videos with replayed depth."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--project-src", type=Path, default=DEFAULT_PROJECT_SRC)
    parser.add_argument("--trackdlo-root", type=Path, default=DEFAULT_TRACKDLO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--robot",
        choices=["panda", "nero"],
        default="panda",
        help="robot model used to replay MuJoCo states and render depth",
    )
    parser.add_argument(
        "--nero-base-offset-x",
        type=float,
        default=0.35,
        help="NERO base-x calibration used by the panda_like rendered dataset",
    )
    parser.add_argument(
        "--nero-tcp-dx",
        type=float,
        default=0.01,
        help="NERO grasp-center x offset used by the panda_like rendered dataset",
    )
    parser.add_argument("--nero-tcp-dy", type=float, default=0.0)
    parser.add_argument("--nero-tcp-dz", type=float, default=0.0)
    parser.add_argument("--camera", choices=["opst", "wrist"], default="opst")
    parser.add_argument(
        "--dual-camera",
        action="store_true",
        help="Merge the primary and secondary camera RGB-D clouds in the primary camera frame.",
    )
    parser.add_argument(
        "--dual-independent",
        action="store_true",
        help="Track each camera independently, then fuse their visible ordered nodes in the primary frame.",
    )
    parser.add_argument(
        "--dual-visible-primary",
        action="store_true",
        help="Ablation: use only the primary camera cloud for visible-node fusion.",
    )
    parser.add_argument(
        "--dual-visible-secondary",
        action="store_true",
        help=(
            "Use secondary-camera mask+cloud support to add visible nodes to post-CPD "
            "current-cloud fusion (native primary CPD guide set is unchanged)."
        ),
    )
    parser.add_argument(
        "--dual-visible-secondary-all",
        action="store_true",
        help="When secondary support is enabled, allow its current observation to correct nodes also visible in the primary view.",
    )
    parser.add_argument(
        "--surface-to-center-offset",
        type=float,
        default=0.0,
        help=(
            "Optional inward correction (m) for the camera-facing RGB-D surface cloud "
            "used only by visible-node fusion; 0 keeps raw surface points."
        ),
    )
    parser.add_argument(
        "--surface-to-center-normal-only",
        action="store_true",
        help="Project the surface-to-centre correction onto the plane normal to a local cloud tangent.",
    )
    parser.add_argument(
        "--surface-to-center-for-cpd",
        action="store_true",
        help="Also feed the surface-to-centre corrected cloud to native CPD (experimental).",
    )
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--episode-names",
        nargs="+",
        default=None,
        help="Optional exact episode directory names to evaluate (useful for crash-isolated runs).",
    )
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--init-seconds", type=float, default=0.8)
    parser.add_argument("--video-episodes-per-scenario", type=int, default=1)
    parser.add_argument("--no-reinitialize", action="store_true")
    parser.add_argument(
        "--reinitialize-grace-seconds",
        type=float,
        default=0.0,
        help=(
            "Allow the normal three-failure reinitialization only for this "
            "many seconds after tracking starts; 0 keeps the legacy behavior."
        ),
    )
    parser.add_argument(
        "--force-sparse-update",
        action="store_true",
        help="Experimental: send sparse supported-node sets to TrackDLO instead of rejecting them.",
    )
    parser.add_argument(
        "--no-hold-last",
        action="store_true",
        help="Experimental: return NaN nodes on an update failure instead of the previous state.",
    )
    parser.add_argument(
        "--accept-nonconverged",
        action="store_true",
        help="Use the last finite CPD iterate when max_iter is reached instead of rejecting the frame.",
    )
    parser.add_argument(
        "--min-visible-nodes",
        type=int,
        default=6,
        help="Minimum visible guide nodes required before calling the native TrackDLO core.",
    )
    parser.add_argument(
        "--visibility-mode",
        choices=["strict", "neighborhood", "mask", "mask_all", "depth"],
        default="strict",
        help="Visibility support gate: strict nearest point, local neighborhood, RGB-mask-only, mask without line occlusion, or RGB-mask+depth experiment.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.008,
        help="Strict nearest-point support threshold in metres.",
    )
    parser.add_argument(
        "--visibility-weak-threshold",
        type=float,
        default=0.012,
        help="Weak nearest-point threshold in metres for neighborhood visibility mode.",
    )
    parser.add_argument(
        "--visibility-neighborhood-radius",
        type=float,
        default=0.015,
        help="3-D support radius in metres for neighborhood visibility mode.",
    )
    parser.add_argument(
        "--visibility-min-neighbors",
        type=int,
        default=3,
        help="Minimum points inside the support radius for neighborhood visibility mode.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=3.0,
        help=(
            "TrackDLO temporal/correspondence anchor weight. "
            "Use 3.0 for the current configuration; 0 disables this anchor "
            "for an observation-dominant ablation."
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Maximum CPD iterations per update; 50 is the native setting, lower values trade accuracy for speed.",
    )
    parser.add_argument(
        "--visible-observation-fusion",
        action="store_true",
        help="Fuse current RGB-D cloud observations into supported visible nodes only.",
    )
    parser.add_argument(
        "--visible-observation-blend",
        type=float,
        default=0.75,
        help="Blend fraction for current-cloud visible-node observations (0..1).",
    )
    parser.add_argument(
        "--visible-observation-radius",
        type=float,
        default=0.025,
        help="3-D radius in metres for local visible-node cloud support.",
    )
    parser.add_argument(
        "--visible-observation-min-points",
        type=int,
        default=3,
        help="Minimum local cloud points needed to fuse one visible node.",
    )
    parser.add_argument(
        "--visible-observation-adaptive-radius",
        action="store_true",
        help="Select between the base and fallback local cloud radius per visible node using robust support.",
    )
    parser.add_argument(
        "--visible-observation-fallback-radius",
        type=float,
        default=0.025,
        help="Expanded local cloud radius in metres for adaptive visible fusion.",
    )
    parser.add_argument(
        "--visible-observation-support-target",
        type=int,
        default=8,
        help="Preferred number of local cloud samples used in adaptive-radius scoring.",
    )
    parser.add_argument(
        "--visible-observation-arc-window-scale",
        type=float,
        default=1.5,
        help="Multiplier on adjacent-node arc spacing used to assign local cloud points.",
    )
    parser.add_argument(
        "--visible-observation-arc-window-min",
        type=float,
        default=0.018,
        help="Minimum arc-assignment window in metres for local visible fusion.",
    )
    parser.add_argument(
        "--visible-observation-motion-threshold",
        type=float,
        default=0.004,
        help="Minimum observed node motion in metres before visible fusion is applied.",
    )
    parser.add_argument(
        "--visible-observation-residual-margin",
        type=float,
        default=0.0015,
        help="Required point-cloud residual improvement (m) before replacing a visible CPD node.",
    )
    parser.add_argument(
        "--visible-observation-full-displacement",
        action="store_true",
        help="Allow visible point-cloud fusion to correct tangential as well as normal motion.",
    )
    parser.add_argument(
        "--visible-observation-extended",
        action="store_true",
        help="Also fuse nodes in the supported interval between visible anchors when local cloud support exists.",
    )
    parser.add_argument(
        "--visible-observation-tangent-blend",
        type=float,
        default=0.0,
        help="Allow this fraction of the visible-cloud correction along the chain tangent (0..1).",
    )
    parser.add_argument(
        "--visible-observation-topology",
        choices=["previous", "candidate", "adaptive", "history_guard"],
        default="previous",
        help="Chain used for current-cloud arc assignment: previous, candidate, adaptive per-node residual selection, or history_guard (candidate only after a bad historical chain).",
    )
    parser.add_argument(
        "--visible-observation-centerline-fit",
        action="store_true",
        help="Fit a local circle in the plane normal to the cable tangent to estimate the visible centreline.",
    )
    parser.add_argument(
        "--visible-pixel-snap-fusion",
        action="store_true",
        help="Snap visible nodes to the current RGB-mask skeleton and depth instead of using the previous-chain cloud arc.",
    )
    parser.add_argument(
        "--visible-pixel-snap-blend",
        type=float,
        default=1.0,
        help="Blend fraction for current pixel/depth snap observations (0..1).",
    )
    parser.add_argument(
        "--visible-pixel-snap-radius-px",
        type=float,
        default=18.0,
        help="Maximum image-space distance for a visible-node skeleton snap.",
    )
    parser.add_argument(
        "--visible-pixel-snap-center-offset",
        type=float,
        default=0.014,
        help="Approximate cable-radius correction (m) from depth surface to centreline.",
    )
    parser.add_argument(
        "--monotonic-visible-fusion",
        action="store_true",
        help=(
            "Use current-cloud arc displacements with a monotonic material-order "
            "constraint for visible nodes instead of the previous-arc-only fusion."
        ),
    )
    parser.add_argument(
        "--monotonic-tangent-blend",
        type=float,
        default=0.75,
        help="Fraction of the order-preserving current tangential displacement to apply (0..1).",
    )
    parser.add_argument(
        "--refine-visibility-after-cpd",
        action="store_true",
        help="Recompute visible/occluded nodes from the finite current CPD candidate before fusion.",
    )
    parser.add_argument(
        "--hidden-history-completion",
        action="store_true",
        help="Transport the previous hidden shape between current visible anchors.",
    )
    parser.add_argument(
        "--hidden-history-blend",
        type=float,
        default=0.80,
        help="Blend fraction for hidden-shape transport (0..1).",
    )
    parser.add_argument(
        "--hidden-history-deformation-threshold",
        type=float,
        default=0.012,
        help="Maximum visible rigid-fit residual (m) for history completion; larger motion is treated as deformation.",
    )
    parser.add_argument(
        "--hidden-piecewise-completion",
        action="store_true",
        help="Complete hidden intervals by interpolating current visible-anchor displacement along the previous material arc.",
    )
    parser.add_argument(
        "--hidden-piecewise-blend",
        type=float,
        default=1.0,
        help="Blend fraction for piecewise hidden displacement completion (0..1).",
    )
    parser.add_argument(
        "--hidden-piecewise-max-anchor-disagreement",
        type=float,
        default=0.020,
        help="Maximum difference (m) between neighboring visible-anchor displacements for piecewise completion.",
    )
    parser.add_argument(
        "--hidden-piecewise-max-gap-nodes",
        type=int,
        default=8,
        help="Maximum number of consecutive hidden nodes to fill with piecewise completion.",
    )
    parser.add_argument(
        "--hidden-hold-previous",
        action="store_true",
        help="Hold previous positions for hidden nodes after visible fusion.",
    )
    parser.add_argument(
        "--hidden-velocity-prediction",
        action="store_true",
        help="Predict hidden nodes from a decayed two-frame velocity after visible fusion.",
    )
    parser.add_argument(
        "--hidden-velocity-decay",
        type=float,
        default=0.60,
        help="Decay for hidden-node constant-velocity prediction (0..1).",
    )
    parser.add_argument(
        "--hidden-velocity-deformation-threshold",
        type=float,
        default=0.0,
        help=(
            "Only apply hidden velocity prediction when the visible-cloud "
            "rigid-fit residual exceeds this value in metres; 0 keeps it always on."
        ),
    )
    parser.add_argument(
        "--hidden-velocity-max-step",
        type=float,
        default=0.0,
        help=(
            "Reject a hidden velocity proposal when its largest adjacent node "
            "step exceeds this value in metres; 0 disables the sanity gate."
        ),
    )
    parser.add_argument(
        "--pointcloud-path-fusion",
        action="store_true",
        help="Order current RGB-D points with a 3-D neighbor graph for visible nodes.",
    )
    parser.add_argument(
        "--pointcloud-path-blend",
        type=float,
        default=0.75,
        help="Blend fraction for 3-D point-cloud path observations (0..1).",
    )
    parser.add_argument(
        "--pointcloud-arc-fusion",
        action="store_true",
        help="Map the current 3-D cloud path to visible nodes by reference arc length and endpoint anchors.",
    )
    parser.add_argument(
        "--pointcloud-arc-anchor-only",
        action="store_true",
        help="Use only the first visible-node anchor plus physical arc spacing when mapping a current cloud path.",
    )
    parser.add_argument(
        "--max-observed-points",
        type=int,
        default=0,
        help="Optional cap after voxel downsampling; zero keeps all points.",
    )
    parser.add_argument(
        "--observed-point-sampling",
        choices=["farthest", "chain"],
        default="farthest",
        help="Point-cloud cap policy: spatial farthest-point coverage or chain-arc bins.",
    )
    parser.add_argument(
        "--fast-observation-only",
        action="store_true",
        help=(
            "Ablation: skip native CPD; update supported visible nodes from the current cloud "
            "and keep occluded nodes at the previous state."
        ),
    )
    parser.add_argument(
        "--cpd-stride",
        type=int,
        default=1,
        help="Run native CPD every Nth frame; N=1 is the native per-frame update.",
    )
    parser.add_argument(
        "--hold-hidden-on-nonconverged",
        action="store_true",
        help="When CPD hits its iteration limit, keep unsupported hidden nodes from the previous valid state.",
    )
    parser.add_argument(
        "--adaptive-history-fusion",
        action="store_true",
        help="Use native per-node current-visible/history-occluded correspondence priors.",
    )
    parser.add_argument(
        "--adaptive-visible-alpha",
        type=float,
        default=3.0,
        help="Prior weight for visible nodes in native adaptive fusion.",
    )
    parser.add_argument(
        "--adaptive-occluded-alpha",
        type=float,
        default=1.0,
        help="Prior weight for occluded nodes in native adaptive fusion.",
    )
    parser.add_argument(
        "--adaptive-history-deformation-switch",
        action="store_true",
        help=(
            "When the current visible cloud is inconsistent with a rigid "
            "motion, switch the occluded-node prior to the deformation alpha."
        ),
    )
    parser.add_argument(
        "--adaptive-history-deformation-threshold",
        type=float,
        default=0.012,
        help="Visible-cloud rigid-fit residual (m) at which the adaptive history switch activates.",
    )
    parser.add_argument(
        "--adaptive-history-deformation-occluded-alpha",
        type=float,
        default=0.0,
        help="Occluded-node prior used after the deformation switch (0 disables the explicit history anchor).",
    )
    parser.add_argument(
        "--adaptive-history-deformation-visible-alpha",
        type=float,
        default=3.0,
        help=(
            "Visible-node prior used after the deformation switch; lowering it "
            "lets the current cloud move visible nodes during non-rigid motion."
        ),
    )
    parser.add_argument(
        "--adaptive-history-support-switch",
        action="store_true",
        help="Lower visible history weight when the current cloud supports a high fraction of visible nodes.",
    )
    parser.add_argument(
        "--adaptive-history-support-threshold",
        type=float,
        default=0.50,
        help="Visible-node support fraction at which the support-adaptive prior activates.",
    )
    parser.add_argument(
        "--adaptive-history-support-visible-alpha",
        type=float,
        default=3.0,
        help="Visible-node history weight while support-adaptive fusion is active.",
    )
    parser.add_argument(
        "--adaptive-cloud-motion-switch",
        action="store_true",
        help="Use current RGB-D cloud centroid motion to switch to the stronger visible/current prior.",
    )
    parser.add_argument(
        "--adaptive-cloud-motion-icp",
        action="store_true",
        help=(
            "Use a nearest-neighbour rigid-fit residual instead of centroid motion "
            "for the adaptive cloud-motion gate."
        ),
    )
    parser.add_argument(
        "--adaptive-cloud-motion-threshold",
        type=float,
        default=0.05,
        help="Centroid displacement in metres that activates the motion-adaptive prior.",
    )
    parser.add_argument(
        "--adaptive-motion-visible-alpha",
        type=float,
        default=12.0,
        help="Visible-node alpha while the cloud-motion gate is active.",
    )
    parser.add_argument(
        "--adaptive-motion-occluded-alpha",
        type=float,
        default=0.0,
        help="Occluded-node alpha while the cloud-motion gate is active.",
    )
    parser.add_argument(
        "--adaptive-motion-surface-offset",
        type=float,
        default=0.014,
        help="Surface-to-centre offset (m) while the cloud-motion gate is active.",
    )
    parser.add_argument(
        "--current-skeleton-fusion",
        action="store_true",
        help="Use current RGB-D skeleton positions for visible node runs.",
    )
    parser.add_argument(
        "--current-skeleton-blend",
        type=float,
        default=0.85,
        help="Blend fraction for current skeleton positions on visible nodes (0..1).",
    )
    parser.add_argument(
        "--current-skeleton-simple-path",
        action="store_true",
        help="At 2-D crossings, use a simple endpoint path instead of an Euler branch trail for current-frame fusion.",
    )
    parser.add_argument(
        "--current-skeleton-depth-path",
        action="store_true",
        help="Use RGB-D tangent/depth continuity to choose a single physical path through 2-D crossings.",
    )
    parser.add_argument(
        "--current-skeleton-cloud",
        action="store_true",
        help=(
            "Use current-frame RGB-D skeleton centerline samples as the visible-observation cloud "
            "(raw cloud remains the native CPD input)."
        ),
    )
    parser.add_argument(
        "--current-path-fusion",
        action="store_true",
        help=(
            "Replace the CPD estimate on the material interval selected by an "
            "ordered current-frame RGB-D skeleton path."
        ),
    )
    parser.add_argument(
        "--current-path-blend",
        type=float,
        default=0.75,
        help="Blend fraction for ordered current-path visible observations (0..1).",
    )
    parser.add_argument(
        "--current-path-use-candidate",
        action="store_true",
        help="Use the current CPD candidate, rather than the previous chain, only as the path interval cue.",
    )
    parser.add_argument(
        "--current-path-run-locked",
        action="store_true",
        help=(
            "In current-frame-first mode, assign each current path to the currently "
            "visible node run instead of searching all historical material intervals."
        ),
    )
    parser.add_argument(
        "--pre-cpd-visible-seed",
        action="store_true",
        help=(
            "Initialize native CPD from current visible-cloud anchors and a smooth hidden spline "
            "before each iteration; previous nodes provide only material indices."
        ),
    )
    parser.add_argument(
        "--pre-cpd-seed-blend",
        type=float,
        default=1.0,
        help="Blend of the current-frame seed versus the previous chain (0..1).",
    )
    parser.add_argument(
        "--post-cpd-current-spline",
        action="store_true",
        help="Replace hidden nodes after CPD with a smooth spline through current visible anchors.",
    )
    parser.add_argument(
        "--current-component-paths",
        action="store_true",
        help="Extract all current RGB-D mask components for ordered-path fusion instead of only the largest component.",
    )
    parser.add_argument(
        "--save-node-traces",
        action="store_true",
        help="Save per-frame predicted/ground-truth node arrays for visible-vs-hidden diagnostics.",
    )
    parser.add_argument("--dlo-pixel-width", type=int, default=6)
    parser.add_argument(
        "--downsample-leaf-size",
        type=float,
        default=0.008,
        help="Voxel leaf size in metres for the observed 3-D cable cloud.",
    )
    parser.add_argument(
        "--hsv-lower",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(112, 180, 80),
        help="RGB cable HSV lower bound; adapted to the current blue cable/background.",
    )
    parser.add_argument(
        "--hsv-upper",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(130, 255, 255),
        help="RGB cable HSV upper bound.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_current_trackdlo(
        run_root=args.run_root.resolve(),
        project_src=args.project_src.resolve(),
        trackdlo_root=args.trackdlo_root.resolve(),
        output=args.output.resolve(),
        robot=args.robot,
        nero_base_offset_x=args.nero_base_offset_x,
        nero_tcp_dx=args.nero_tcp_dx,
        nero_tcp_dy=args.nero_tcp_dy,
        nero_tcp_dz=args.nero_tcp_dz,
        camera=args.camera,
        dual_camera=args.dual_camera,
        dual_independent=args.dual_independent,
        dual_visible_primary=args.dual_visible_primary,
        dual_visible_secondary=args.dual_visible_secondary,
        dual_visible_secondary_all=args.dual_visible_secondary_all,
        surface_to_center_offset_m=args.surface_to_center_offset,
        surface_to_center_normal_only=args.surface_to_center_normal_only,
        surface_to_center_for_cpd=args.surface_to_center_for_cpd,
        scenarios=args.scenarios,
        episode_names=args.episode_names,
        episodes_per_scenario=args.episodes_per_scenario,
        frame_stride=args.frame_stride,
        init_seconds=args.init_seconds,
        video_episodes_per_scenario=args.video_episodes_per_scenario,
        no_reinitialize=args.no_reinitialize,
        reinitialize_grace_seconds=args.reinitialize_grace_seconds,
        force_sparse_update=args.force_sparse_update,
        hold_last_on_failure=not args.no_hold_last,
        accept_nonconverged=args.accept_nonconverged,
        min_visible_nodes=args.min_visible_nodes,
        visibility_mode=args.visibility_mode,
        visibility_threshold=args.visibility_threshold,
        visibility_weak_threshold=args.visibility_weak_threshold,
        visibility_neighborhood_radius=args.visibility_neighborhood_radius,
        visibility_min_neighbors=args.visibility_min_neighbors,
        alpha=args.alpha,
        max_iter=args.max_iter,
        visible_observation_fusion=args.visible_observation_fusion,
        visible_observation_blend=args.visible_observation_blend,
        visible_observation_radius=args.visible_observation_radius,
        visible_observation_min_points=args.visible_observation_min_points,
        visible_observation_adaptive_radius=args.visible_observation_adaptive_radius,
        visible_observation_fallback_radius=args.visible_observation_fallback_radius,
        visible_observation_support_target=args.visible_observation_support_target,
        visible_observation_arc_window_scale=args.visible_observation_arc_window_scale,
        visible_observation_arc_window_min=args.visible_observation_arc_window_min,
        visible_observation_motion_threshold=args.visible_observation_motion_threshold,
        visible_observation_residual_margin=args.visible_observation_residual_margin,
        visible_observation_normal_only=not args.visible_observation_full_displacement,
        visible_observation_tangent_blend=args.visible_observation_tangent_blend,
        visible_observation_topology=args.visible_observation_topology,
        visible_observation_centerline_fit=args.visible_observation_centerline_fit,
        visible_observation_use_extended=args.visible_observation_extended,
        visible_pixel_snap_fusion=args.visible_pixel_snap_fusion,
        visible_pixel_snap_blend=args.visible_pixel_snap_blend,
        visible_pixel_snap_radius_px=args.visible_pixel_snap_radius_px,
        visible_pixel_snap_center_offset_m=args.visible_pixel_snap_center_offset,
        monotonic_visible_fusion=args.monotonic_visible_fusion,
        monotonic_tangent_blend=args.monotonic_tangent_blend,
        refine_visibility_after_cpd=args.refine_visibility_after_cpd,
        hidden_history_completion=args.hidden_history_completion,
        hidden_history_blend=args.hidden_history_blend,
        hidden_history_deformation_threshold=args.hidden_history_deformation_threshold,
        hidden_piecewise_completion=args.hidden_piecewise_completion,
        hidden_piecewise_blend=args.hidden_piecewise_blend,
        hidden_piecewise_max_anchor_disagreement=args.hidden_piecewise_max_anchor_disagreement,
        hidden_piecewise_max_gap_nodes=args.hidden_piecewise_max_gap_nodes,
        hidden_hold_previous=args.hidden_hold_previous,
        hidden_velocity_prediction=args.hidden_velocity_prediction,
        hidden_velocity_decay=args.hidden_velocity_decay,
        hidden_velocity_deformation_threshold_m=args.hidden_velocity_deformation_threshold,
        hidden_velocity_max_step_m=args.hidden_velocity_max_step,
        pointcloud_path_fusion=args.pointcloud_path_fusion,
        pointcloud_path_blend=args.pointcloud_path_blend,
        pointcloud_arc_fusion=args.pointcloud_arc_fusion,
        pointcloud_arc_anchor_only=args.pointcloud_arc_anchor_only,
        max_observed_points=args.max_observed_points,
        observed_point_sampling=args.observed_point_sampling,
        cpd_stride=args.cpd_stride,
        hold_hidden_on_nonconverged=args.hold_hidden_on_nonconverged,
        fast_observation_only=args.fast_observation_only,
        adaptive_history_fusion=args.adaptive_history_fusion,
        adaptive_visible_alpha=args.adaptive_visible_alpha,
        adaptive_occluded_alpha=args.adaptive_occluded_alpha,
        adaptive_history_deformation_switch=args.adaptive_history_deformation_switch,
        adaptive_history_deformation_threshold=args.adaptive_history_deformation_threshold,
        adaptive_history_deformation_occluded_alpha=args.adaptive_history_deformation_occluded_alpha,
        adaptive_history_deformation_visible_alpha=args.adaptive_history_deformation_visible_alpha,
        adaptive_history_support_switch=args.adaptive_history_support_switch,
        adaptive_history_support_threshold=args.adaptive_history_support_threshold,
        adaptive_history_support_visible_alpha=args.adaptive_history_support_visible_alpha,
        adaptive_cloud_motion_switch=args.adaptive_cloud_motion_switch,
        adaptive_cloud_motion_icp=args.adaptive_cloud_motion_icp,
        adaptive_cloud_motion_threshold_m=args.adaptive_cloud_motion_threshold,
        adaptive_motion_visible_alpha=args.adaptive_motion_visible_alpha,
        adaptive_motion_occluded_alpha=args.adaptive_motion_occluded_alpha,
        adaptive_motion_surface_offset_m=args.adaptive_motion_surface_offset,
        current_skeleton_fusion=args.current_skeleton_fusion,
        current_skeleton_blend=args.current_skeleton_blend,
        current_skeleton_simple_path=args.current_skeleton_simple_path,
        current_skeleton_depth_path=args.current_skeleton_depth_path,
        current_skeleton_cloud=args.current_skeleton_cloud,
        current_path_fusion=args.current_path_fusion,
        current_path_blend=args.current_path_blend,
        current_path_use_candidate=args.current_path_use_candidate,
        current_path_run_locked=args.current_path_run_locked,
        pre_cpd_visible_seed=args.pre_cpd_visible_seed,
        pre_cpd_seed_blend=args.pre_cpd_seed_blend,
        post_cpd_current_spline=args.post_cpd_current_spline,
        current_component_paths=args.current_component_paths,
        save_node_traces=args.save_node_traces,
        dlo_pixel_width=args.dlo_pixel_width,
        downsample_leaf_size=args.downsample_leaf_size,
        hsv_lower=tuple(args.hsv_lower),
        hsv_upper=tuple(args.hsv_upper),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
