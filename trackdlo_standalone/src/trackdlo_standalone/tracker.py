from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from ._core import TrackDLOCore
from .geometry import (
    backproject_mask,
    chain_guided_sample,
    depth_to_meters,
    farthest_point_sample,
    project_camera_points,
    projection_matrix,
    resample_polyline,
    voxel_downsample,
)
from .initialization import (
    initialize_nodes,
    depth_skeleton_paths,
    ordered_skeleton_pixels,
    ordered_skeleton_pixels_depth,
    ordered_skeleton_pixels_simple,
    segment_hsv,
)


@dataclass(frozen=True)
class TrackDLOConfig:
    num_nodes: int = 45
    hsv_lower: tuple[int, int, int] = (90, 90, 80)
    hsv_upper: tuple[int, int, int] = (130, 255, 255)
    # ``strict`` preserves the original nearest-point gate.  ``neighborhood``
    # is an opt-in experiment that also considers local point support.
    visibility_mode: str = "strict"
    visibility_threshold: float = 0.008
    visibility_weak_threshold: float = 0.012
    visibility_neighborhood_radius: float = 0.015
    visibility_min_neighbors: int = 3
    visibility_mask_radius_px: int = 8
    visibility_min_mask_pixels: int = 3
    beta: float = 0.35
    lambda_: float = 50000.0
    alpha: float = 3.0
    k_vis: float = 50.0
    mu: float = 0.1
    max_iter: int = 50
    tol: float = 0.0002
    beta_pre_proc: float = 3.0
    lambda_pre_proc: float = 1.0
    lle_weight: float = 10.0
    d_vis: float = 0.06
    # The official parameter is the projected object width, not a universal constant.
    # This simulation cable is about 6-8 px wide at 640x480 (the demo rope was 40 px).
    dlo_pixel_width: int = 6
    downsample_leaf_size: float = 0.008
    # Two or three guide nodes cannot constrain a 45-node chain and expose
    # unsupported corner cases in the upstream both-ends-occluded traversal.
    min_visible_nodes: int = 6
    reinitialize_after_failures: int = 3
    reinitialize_length_ratio_min: float = 0.75
    reinitialize_length_ratio_max: float = 1.25
    # Experimental: pass sparse observations to the C++ optimizer instead of
    # rejecting the frame at the Python visibility gate.  This is disabled by
    # default because the upstream traversal is not defined for an empty guide
    # set or an empty point cloud.
    force_sparse_update: bool = False
    # Experimental evaluation policy.  When false, a failed update returns
    # NaN nodes instead of reusing the previous state.  This keeps the failure
    # explicit in metrics/control integration rather than hiding it as a hold.
    hold_last_on_failure: bool = True
    # Accept the last finite CPD iterate when max_iter is reached. Hard
    # numerical/input failures are still rejected.
    accept_nonconverged: bool = False
    # Experimental observation-dominant fusion. Supported visible nodes are
    # nudged toward the current RGB-D cloud; occluded nodes keep the native
    # TrackDLO temporal/CPD estimate.
    visible_observation_fusion: bool = False
    visible_observation_blend: float = 0.75
    visible_observation_radius: float = 0.025
    visible_observation_min_points: int = 3
    visible_observation_adaptive_radius: bool = False
    visible_observation_fallback_radius: float = 0.025
    visible_observation_support_target: int = 8
    visible_observation_arc_window_scale: float = 1.5
    visible_observation_arc_window_min: float = 0.018
    visible_observation_motion_threshold: float = 0.004
    visible_observation_residual_margin: float = 0.0015
    visible_observation_normal_only: bool = True
    # Fraction of the cloud correction allowed along the current chain
    # tangent when ``normal_only`` is enabled.  Zero is the conservative
    # normal-only mode; intermediate values reduce dependence on the CPD
    # tangential coordinate without allowing a full branch jump.
    visible_observation_tangent_blend: float = 0.0
    # Arc-coordinate reference for assigning current cloud points.  The
    # previous chain is safer at crossings; the current CPD chain is less
    # biased when the cable has visibly deformed since the last frame.
    visible_observation_topology: str = "previous"
    visible_observation_centerline_fit: bool = False
    visible_observation_use_extended: bool = False
    # Experimental frame-wise 2-D snap: project each visible CPD node onto
    # the current RGB mask skeleton and back-project its local depth patch.
    # This uses the previous/current chain only as a local pixel cue, rather
    # than iterating the whole visible segment from the previous 3-D state.
    visible_pixel_snap_fusion: bool = False
    visible_pixel_snap_blend: float = 1.0
    visible_pixel_snap_radius_px: float = 18.0
    visible_pixel_snap_center_offset_m: float = 0.014
    # Experimental current-frame visible fusion with a monotonic arc
    # constraint.  Tangential motion is estimated from the current cloud, but
    # the fitted material arc cannot go backwards at a crossing.
    monotonic_visible_fusion: bool = False
    monotonic_tangent_blend: float = 0.75
    # Re-evaluate support after the native CPD step.  The initial visibility
    # gate uses the previous chain for the CPD correspondence traversal; this
    # optional second pass lets the current-frame candidate decide which
    # nodes are actually supported before visible/hidden fusion.
    refine_visibility_after_cpd: bool = False
    hidden_history_completion: bool = False
    hidden_history_blend: float = 0.80
    # History completion is trusted only when the currently visible part is
    # consistent with a near-rigid motion.  A large residual indicates local
    # cable deformation, where the native CPD estimate is safer for hidden
    # nodes than transporting the stale previous shape.
    hidden_history_deformation_threshold: float = 0.012
    # Piecewise displacement completion for hidden intervals.  Unlike the
    # rigid-history completion above, this interpolates the measured motion
    # of the two nearest visible anchors along the previous material arc, so
    # a locally deforming visible segment can move its hidden neighbours
    # without transporting one stale global rigid shape.
    hidden_piecewise_completion: bool = False
    hidden_piecewise_blend: float = 1.0
    hidden_piecewise_max_anchor_disagreement: float = 0.020
    hidden_piecewise_max_gap_nodes: int = 8
    hidden_hold_previous: bool = False
    hidden_velocity_prediction: bool = False
    hidden_velocity_decay: float = 0.60
    hidden_velocity_deformation_threshold_m: float = 0.0
    hidden_velocity_max_step_m: float = 0.0
    pointcloud_path_fusion: bool = False
    pointcloud_path_blend: float = 0.75
    # Experimental current-frame arc matcher.  It uses the current cloud's
    # graph path and only endpoint/index anchors from the previous chain;
    # unlike CPD, it does not iteratively deform the visible segment.
    pointcloud_arc_fusion: bool = False
    pointcloud_arc_anchor_only: bool = False
    max_observed_points: int = 0
    observed_point_sampling: str = "farthest"
    cpd_stride: int = 1
    hold_hidden_on_nonconverged: bool = False
    fast_observation_only: bool = False
    # Experimental native adaptive prior: visible nodes use a current-guide
    # target, while occluded-node priors are replaced by the previous state.
    adaptive_history_fusion: bool = False
    adaptive_visible_alpha: float = 3.0
    adaptive_occluded_alpha: float = 1.0
    # Optional online switch for deforming scenes.  When the current visible
    # cloud is inconsistent with a rigid motion of the previous visible
    # chain, reduce the occluded-node history prior instead of forcing stale
    # history through the hidden interval.
    adaptive_history_deformation_switch: bool = False
    adaptive_history_deformation_threshold: float = 0.012
    adaptive_history_deformation_occluded_alpha: float = 0.0
    adaptive_history_deformation_visible_alpha: float = 3.0
    adaptive_history_support_switch: bool = False
    adaptive_history_support_threshold: float = 0.50
    adaptive_history_support_visible_alpha: float = 3.0
    # Optional scalar motion gate.  It uses only the current RGB-D cloud
    # centroid displacement (not the previous DLO shape) to choose a stronger
    # visible/current prior during fast global motion.
    adaptive_cloud_motion_switch: bool = False
    adaptive_cloud_motion_icp: bool = False
    adaptive_cloud_motion_threshold_m: float = 0.05
    adaptive_motion_visible_alpha: float = 12.0
    adaptive_motion_occluded_alpha: float = 0.0
    # Experimental current-frame skeleton fusion.  This is deliberately
    # separate from the native CPD adaptive prior above: it writes positions
    # for visible runs from the current RGB-D skeleton and leaves occluded
    # nodes untouched.
    current_skeleton_fusion: bool = False
    current_skeleton_blend: float = 0.85
    current_skeleton_simple_path: bool = False
    current_skeleton_depth_path: bool = False
    # Experimental current-frame path fusion.  The ordered path is extracted
    # from this image's RGB-D skeleton and only its visible interval replaces
    # the CPD candidate; nodes outside that interval keep native temporal
    # completion.
    current_path_fusion: bool = False
    current_path_blend: float = 0.75
    # Use the finite current CPD candidate as the weak interval/index cue for
    # the ordered RGB-D path.  This avoids letting a stale previous chain pick
    # the wrong branch after visible deformation; returned positions still
    # come directly from the current path.
    current_path_use_candidate: bool = False
    # In current-frame-first mode, lock a path to the currently supported
    # visible-node run instead of searching all material intervals with old
    # endpoint geometry.  This avoids branch swaps when the previous pose has
    # drifted; the previous chain still provides only the run's integer labels.
    current_path_run_locked: bool = False
    # Experimental current-frame-first CPD seed.  The visible nodes are
    # fitted to this frame's cloud first; hidden nodes are then initialized by
    # a smooth interpolation between those current anchors before CPD runs.
    # The previous chain is used only for material-index ordering, not as the
    # whole-chain geometric starting pose.
    pre_cpd_visible_seed: bool = False
    pre_cpd_seed_blend: float = 1.0
    # Replace unsupported nodes after CPD with a smooth curve through the
    # current-frame visible anchors.  This is the strict current-frame-first
    # completion ablation; it intentionally does not copy the previous hidden
    # shape.
    post_cpd_current_spline: bool = False


@dataclass
class TrackResult:
    nodes_camera: np.ndarray
    mask: np.ndarray
    observed_points_camera: np.ndarray
    visible_nodes: np.ndarray
    self_occluded_nodes: np.ndarray
    initialized: bool
    tracking_ok: bool
    nonconverged: bool
    reinitialized: bool
    failure_reason: str | None
    preprocess_ms: float
    tracking_ms: float
    total_ms: float
    observation_fused_nodes: int = 0
    # Diagnostics for frame-adaptive fusion.  They are populated only when
    # the deformation probe is enabled; defaults keep old callers compatible.
    deformation_residual_m: float = float("nan")
    deformation_detected: bool = False


class TrackDLOTracker:
    """Stateful, ROS-free RGB-D interface around the official TrackDLO core."""

    def __init__(self, intrinsics: np.ndarray, config: TrackDLOConfig | None = None):
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        self.projection = projection_matrix(self.intrinsics)
        self.config = config or TrackDLOConfig()
        self._core: TrackDLOCore | None = None
        self._nodes: np.ndarray | None = None
        self._history_nodes: np.ndarray | None = None
        self._geodesic: np.ndarray | None = None
        self._reference_length: float | None = None
        self._consecutive_failures = 0
        self._update_counter = 0
        self._last_current_path_debug: tuple[float, int, int] | None = None
        self._current_path_endpoints: tuple[np.ndarray, np.ndarray] | None = None
        # Last accepted current-frame ordered path.  This is deliberately
        # separate from the CPD node chain: during a deformation the CPD
        # history can stretch or jump branches, while consecutive RGB-D
        # skeleton paths still provide a reliable direction cue.
        self._current_path_memory: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self._core is not None

    @property
    def nodes(self) -> np.ndarray | None:
        return None if self._nodes is None else self._nodes.copy()

    def reset(self) -> None:
        self._core = None
        self._nodes = None
        self._history_nodes = None
        self._geodesic = None
        self._reference_length = None
        self._consecutive_failures = 0
        self._update_counter = 0
        self._last_current_path_debug = None
        self._current_path_endpoints = None
        self._current_path_memory = None

    def adopt_nodes(self, nodes: np.ndarray) -> None:
        """Make an externally fused chain the next temporal state.

        This is used by the independent dual-camera experiment after its
        visibility decision.  The native core is synchronized so the next
        update starts from the fused state rather than an unfused branch.
        """
        if self._core is None or self._nodes is None:
            raise RuntimeError("Cannot adopt nodes before tracker initialization")
        array = np.asarray(nodes, dtype=np.float64)
        if array.shape != self._nodes.shape or not np.isfinite(array).all():
            raise ValueError("Invalid externally fused node array")
        previous = self._nodes.copy()
        self._nodes = array.copy()
        self._history_nodes = previous
        self._core.initialize_nodes(self._nodes)
        self._core.set_sigma2(0.0)

    def initialize_from_nodes(
        self,
        nodes: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
    ) -> TrackResult:
        """Initialize a camera tracker from an externally ordered chain.

        This fallback is useful for the independent wrist view when the cable
        is fully hidden at that camera's first frame.  The primary view
        supplies only the initial geometry; subsequent wrist updates still
        use its own RGB-D observations and visibility decisions.
        """
        start = perf_counter()
        array = np.asarray(nodes, dtype=np.float64)
        if array.shape != (self.config.num_nodes, 3) or not np.isfinite(array).all():
            raise ValueError("Invalid external initialization nodes")
        config = self.config
        self._core = TrackDLOCore(
            config.num_nodes,
            config.visibility_threshold,
            config.beta,
            config.lambda_,
            config.alpha,
            config.k_vis,
            config.mu,
            config.max_iter,
            config.tol,
            config.beta_pre_proc,
            config.lambda_pre_proc,
            config.lle_weight,
        )
        if config.adaptive_history_fusion:
            self._core.set_adaptive_alpha(
                float(config.adaptive_visible_alpha),
                float(config.adaptive_occluded_alpha),
            )
        geodesic = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(array, axis=0), axis=1)))
        )
        self._core.initialize_nodes(array)
        self._core.initialize_geodesic_coord(geodesic.tolist())
        self._nodes = array.copy()
        self._history_nodes = None
        self._geodesic = geodesic
        self._reference_length = float(geodesic[-1])
        self._consecutive_failures = 0
        self._update_counter = 0
        mask = segment_hsv(rgb, config.hsv_lower, config.hsv_upper)
        points = voxel_downsample(
            backproject_mask(depth_to_meters(depth), mask, self.intrinsics),
            config.downsample_leaf_size,
        )
        elapsed = (perf_counter() - start) * 1000.0
        return TrackResult(
            nodes_camera=array.copy(),
            mask=mask,
            observed_points_camera=points,
            visible_nodes=np.empty(0, dtype=np.int32),
            self_occluded_nodes=np.arange(len(array), dtype=np.int32),
            initialized=True,
            tracking_ok=True,
            nonconverged=False,
            reinitialized=False,
            failure_reason=None,
            preprocess_ms=elapsed,
            tracking_ms=0.0,
            total_ms=elapsed,
            observation_fused_nodes=0,
        )

    def initialize(self, rgb: np.ndarray, depth: np.ndarray) -> TrackResult:
        start = perf_counter()
        nodes, mask = initialize_nodes(
            rgb,
            depth,
            self.intrinsics,
            self.config.num_nodes,
            self.config.hsv_lower,
            self.config.hsv_upper,
        )
        config = self.config
        self._core = TrackDLOCore(
            config.num_nodes,
            config.visibility_threshold,
            config.beta,
            config.lambda_,
            config.alpha,
            config.k_vis,
            config.mu,
            config.max_iter,
            config.tol,
            config.beta_pre_proc,
            config.lambda_pre_proc,
            config.lle_weight,
        )
        if config.adaptive_history_fusion:
            self._core.set_adaptive_alpha(
                float(config.adaptive_visible_alpha),
                float(config.adaptive_occluded_alpha),
            )
        geodesic = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))))
        self._core.initialize_nodes(nodes)
        self._core.initialize_geodesic_coord(geodesic.tolist())
        self._nodes = nodes
        self._history_nodes = None
        self._geodesic = geodesic
        length = float(geodesic[-1])
        if self._reference_length is None:
            self._reference_length = length
        self._consecutive_failures = 0
        self._update_counter = 0
        points = voxel_downsample(
            backproject_mask(depth_to_meters(depth), mask, self.intrinsics),
            config.downsample_leaf_size,
        )
        elapsed = (perf_counter() - start) * 1000.0
        return TrackResult(
            nodes_camera=nodes.copy(),
            mask=mask,
            observed_points_camera=points,
            visible_nodes=np.arange(len(nodes), dtype=np.int32),
            self_occluded_nodes=np.empty(0, dtype=np.int32),
            initialized=True,
            tracking_ok=True,
            nonconverged=False,
            reinitialized=False,
            failure_reason=None,
            preprocess_ms=elapsed,
            tracking_ms=0.0,
            total_ms=elapsed,
            observation_fused_nodes=0,
        )

    def update(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        observed_points_override: np.ndarray | None = None,
        visible_points_override: np.ndarray | None = None,
        ordered_visible_path_override: np.ndarray | None = None,
        ordered_visible_paths_override: list[np.ndarray] | None = None,
        additional_fusion_visible_nodes: np.ndarray | list[int] | None = None,
        additional_visible_points_override: np.ndarray | None = None,
        additional_fusion_all_nodes: bool = False,
        cloud_motion_m: float | None = None,
    ) -> TrackResult:
        if not self.initialized:
            result = self.initialize(rgb, depth)
            initial_path = ordered_visible_path_override
            if initial_path is None and ordered_visible_paths_override:
                initial_path = ordered_visible_paths_override[0]
            if initial_path is not None:
                initial_path = np.asarray(initial_path, dtype=np.float64)
                if initial_path.ndim == 2 and len(initial_path) >= 2 and np.isfinite(initial_path).all():
                    direct = float(
                        np.linalg.norm(initial_path[0] - result.nodes_camera[0])
                        + np.linalg.norm(initial_path[-1] - result.nodes_camera[-1])
                    )
                    reverse = float(
                        np.linalg.norm(initial_path[-1] - result.nodes_camera[0])
                        + np.linalg.norm(initial_path[0] - result.nodes_camera[-1])
                    )
                    if reverse < direct:
                        initial_path = initial_path[::-1]
                    # The first path orientation is arbitrary, but recording
                    # it lets subsequent current-frame paths keep a stable
                    # material direction without consulting every old node.
                    self._current_path_endpoints = (
                        initial_path[0].copy(),
                        initial_path[-1].copy(),
                    )
                    # Keep the first ordered RGB-D path as the material
                    # direction reference as well as its endpoints.  Without
                    # this initialization, the first short path accepted
                    # after startup can choose the opposite orientation and
                    # permanently swap all node indices at a crossing.
                    self._current_path_memory = initial_path.copy()
            return result
        total_start = perf_counter()
        mask = segment_hsv(rgb, self.config.hsv_lower, self.config.hsv_upper)
        depth_m = depth_to_meters(depth)
        if observed_points_override is None:
            points = voxel_downsample(
                backproject_mask(depth_m, mask, self.intrinsics), self.config.downsample_leaf_size
            )
        else:
            points = voxel_downsample(
                np.asarray(observed_points_override, dtype=np.float64),
                self.config.downsample_leaf_size,
            )
        max_points = int(self.config.max_observed_points)
        if max_points > 0 and len(points) > max_points:
            # Keep spatial coverage when capping the cloud.  The voxel hash
            # order is not a geometric order, so selecting evenly spaced rows
            # can discard a whole bend of the cable.
            if self.config.observed_point_sampling == "chain" and self._nodes is not None:
                points = chain_guided_sample(points, self._nodes, max_points)
            else:
                points = farthest_point_sample(points, max_points)
        visible_points = points
        if visible_points_override is not None:
            visible_points = voxel_downsample(
                np.asarray(visible_points_override, dtype=np.float64),
                self.config.downsample_leaf_size,
            )
        additional_visible_points = None
        if additional_visible_points_override is not None:
            additional_visible_points = voxel_downsample(
                np.asarray(additional_visible_points_override, dtype=np.float64),
                self.config.downsample_leaf_size,
            )
        visible, visible_extended, self_occluded = self._visibility(
            points, rgb.shape[:2], mask, depth_m
        )
        preprocess_ms = (perf_counter() - total_start) * 1000.0
        sparse_observation = len(points) < 4 or len(visible) < self.config.min_visible_nodes
        if (
            sparse_observation
            and not self.config.force_sparse_update
            and not self.config.fast_observation_only
        ):
            return self._failed_update(
                rgb,
                depth,
                mask,
                points,
                visible,
                self_occluded,
                "insufficient_visible_observations",
                preprocess_ms,
                0.0,
                total_start,
            )
        # The C++ core cannot form a probability matrix with N=0 points.  In
        # this case the honest no-hold result is an invalid frame, not a
        # fabricated update from the previous state.
        if len(points) == 0:
            return self._failed_update(
                rgb,
                depth,
                mask,
                points,
                visible,
                self_occluded,
                "empty_observation_cloud",
                preprocess_ms,
                0.0,
                total_start,
            )
        previous = self._nodes.copy()
        older = None if self._history_nodes is None else self._history_nodes.copy()
        self._update_counter += 1
        fused_count = 0
        tracking_start = perf_counter()
        deformation_residual = float("nan")
        deformation_detected = False
        if self.config.fast_observation_only:
            # Current-frame-first ablation: no temporal CPD iteration.  When an
            # ordered RGB-D path is available, visible nodes are sampled
            # directly from this frame and the hidden interval is completed
            # from the current anchors.  This is intentionally different from
            # the normal TrackDLO update, where CPD first deforms the whole
            # chain from the previous frame and only then receives a visible
            # correction.
            candidate = previous.copy()
            path_observations: dict[int, np.ndarray] = {}
            best_oriented_path: np.ndarray | None = None
            if self.config.current_path_fusion and (
                ordered_visible_path_override is not None
                or ordered_visible_paths_override
            ):
                path_list = (
                    ordered_visible_paths_override
                    if ordered_visible_paths_override
                    else [ordered_visible_path_override]
                )
                best_path_score = float("inf")
                for current_path in path_list:
                    if current_path is None:
                        continue
                    current_path = np.asarray(current_path, dtype=np.float64)
                    oriented_path, visual_cost = self._visual_path_orientation(
                        current_path, self._current_path_memory
                    )
                    orientation_hint = 1 if np.isfinite(visual_cost) else None
                    one_path = self._current_path_observations(
                        oriented_path,
                        previous,
                        visible,
                        orientation_hint=orientation_hint,
                    )
                    if not one_path:
                        continue
                    path_debug = self._last_current_path_debug
                    path_cost = float(path_debug[0]) if path_debug is not None else 1e9
                    span_fraction = len(one_path) / max(len(previous), 1)
                    visual_term = 1.5 * visual_cost if np.isfinite(visual_cost) else 0.0
                    score = path_cost + visual_term + 0.08 * (1.0 - span_fraction)
                    if score < best_path_score:
                        best_path_score = score
                        path_observations = one_path
                        best_oriented_path = oriented_path
                if path_observations:
                    visible_set = {
                        int(index)
                        for index in np.asarray(visible).ravel()
                        if 0 <= int(index) < len(previous)
                    }
                    path_observations = {
                        index: observation
                        for index, observation in path_observations.items()
                        if index in visible_set
                    }
                if path_observations:
                    for node_index, observation in path_observations.items():
                        candidate[node_index] = observation
                    # The current visible anchors, not the old hidden pose,
                    # define the interpolation geometry.  The old state is
                    # used only for the material index of each anchor.  A
                    # strict hold option is useful for separating visible
                    # path/index errors from hidden completion errors.
                    if not self.config.hidden_hold_previous:
                        candidate = self._complete_hidden_current_anchor_spline(
                            previous, candidate, visible
                        )
                    fused_count = len(path_observations)
                    if best_oriented_path is not None:
                        self._current_path_endpoints = (
                            best_oriented_path[0].copy(),
                            best_oriented_path[-1].copy(),
                        )
                        self._current_path_memory = best_oriented_path.copy()
            if not path_observations and self.config.current_skeleton_fusion:
                skeleton_observations = self._current_skeleton_observations(
                    mask, depth_m, previous, visible
                )
                blend = float(np.clip(self.config.current_skeleton_blend, 0.0, 1.0))
                for node_index, observation in skeleton_observations.items():
                    candidate[node_index] = (
                        (1.0 - blend) * candidate[node_index] + blend * observation
                    )
                fused_count = len(skeleton_observations)
            elif not path_observations:
                candidate, fused_count = self._fuse_visible_observations(
                    previous,
                    previous.copy(),
                    visible_points,
                    visible_extended if self.config.visible_observation_use_extended else visible,
                )
            previous_length = float(
                np.linalg.norm(np.diff(previous, axis=0), axis=1).sum()
            )
            candidate_length = float(
                np.linalg.norm(np.diff(candidate, axis=0), axis=1).sum()
            )
            ratio = candidate_length / max(previous_length, 1e-9)
            steps = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
            if (
                not 0.85 <= ratio <= 1.18
                or not np.isfinite(steps).all()
                or float(np.max(steps)) > max(
                    0.035,
                    2.5 * float(np.median(np.linalg.norm(np.diff(previous, axis=0), axis=1))),
                )
            ):
                candidate = previous.copy()
                fused_count = 0
            self._nodes = candidate
            self._history_nodes = previous
            tracking_ms = (perf_counter() - tracking_start) * 1000.0
            self._consecutive_failures = 0
            return TrackResult(
                nodes_camera=self._nodes.copy(),
                mask=mask,
                observed_points_camera=points,
                visible_nodes=np.asarray(visible, dtype=np.int32),
                self_occluded_nodes=np.asarray(self_occluded, dtype=np.int32),
                initialized=True,
                tracking_ok=True,
                nonconverged=False,
                reinitialized=False,
                failure_reason=None,
                preprocess_ms=preprocess_ms,
                tracking_ms=tracking_ms,
                total_ms=(perf_counter() - total_start) * 1000.0,
                observation_fused_nodes=int(fused_count),
                deformation_residual_m=float(deformation_residual),
                deformation_detected=bool(deformation_detected),
            )
        # The native adaptive prior is normally fixed for the whole run.  An
        # optional deformation-aware switch makes the policy genuinely
        # frame-adaptive: visible nodes remain strongly anchored to the
        # current guide, while the occluded prior is weakened when the
        # currently supported cable cannot be explained by a rigid motion of
        # the previous visible chain.  The probe is a cheap local cloud fit;
        # it does not use ground truth and does not alter tracker state.
        if self.config.adaptive_history_fusion and self.config.adaptive_history_deformation_switch:
            visible_alpha = float(self.config.adaptive_visible_alpha)
            occluded_alpha = float(self.config.adaptive_occluded_alpha)
            support_fraction = 0.0
            try:
                probe_visible = (
                    visible_extended
                    if self.config.visible_observation_use_extended
                    else visible
                )
                probe_nodes, probe_count = self._fuse_visible_observations(
                    previous,
                    previous.copy(),
                    visible_points,
                    probe_visible,
                )
                # Count only the nodes that were actually returned by the
                # current-cloud probe.  This is a frame confidence cue, not a
                # geometric update, and is independent of ground truth.
                support_fraction = min(
                    1.0,
                    float(probe_count) / max(float(len(probe_visible)), 1.0),
                )
                supported = [
                    int(index)
                    for index in visible
                    if 0 <= int(index) < len(previous)
                    and np.isfinite(probe_nodes[int(index)]).all()
                ]
                if len(supported) >= 3:
                    residual = self._rigid_fit_residual(
                        previous[np.asarray(supported, dtype=np.int64)],
                        probe_nodes[np.asarray(supported, dtype=np.int64)],
                    )
                    deformation_residual = float(residual)
                    if residual > max(
                        float(self.config.adaptive_history_deformation_threshold), 0.0
                    ):
                        deformation_detected = True
                        occluded_alpha = float(
                            self.config.adaptive_history_deformation_occluded_alpha
                        )
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                # Keep the configured conservative history prior if the
                # confidence probe is underconstrained or numerically invalid.
                pass
            if deformation_detected:
                # During non-rigid motion the previous chain is a less useful
                # visible-node guide.  Lower its regularization so the current
                # cloud/correspondence term can move visible nodes, while the
                # occluded interval follows the configured deformation prior.
                visible_alpha = float(
                    self.config.adaptive_history_deformation_visible_alpha
                )
            if (
                self.config.adaptive_history_support_switch
                and support_fraction
                >= max(float(self.config.adaptive_history_support_threshold), 0.0)
            ):
                visible_alpha = float(self.config.adaptive_history_support_visible_alpha)
            if (
                self.config.adaptive_cloud_motion_switch
                and cloud_motion_m is not None
                and float(cloud_motion_m)
                > max(float(self.config.adaptive_cloud_motion_threshold_m), 0.0)
            ):
                visible_alpha = float(self.config.adaptive_motion_visible_alpha)
                occluded_alpha = float(self.config.adaptive_motion_occluded_alpha)
            self._core.set_adaptive_alpha(visible_alpha, max(0.0, occluded_alpha))
        elif self.config.adaptive_history_fusion:
            visible_alpha = float(self.config.adaptive_visible_alpha)
            occluded_alpha = float(self.config.adaptive_occluded_alpha)
            if (
                self.config.adaptive_cloud_motion_switch
                and cloud_motion_m is not None
                and float(cloud_motion_m)
                > max(float(self.config.adaptive_cloud_motion_threshold_m), 0.0)
            ):
                visible_alpha = float(self.config.adaptive_motion_visible_alpha)
                occluded_alpha = float(self.config.adaptive_motion_occluded_alpha)
            self._core.set_adaptive_alpha(visible_alpha, max(0.0, occluded_alpha))
        cpd_stride = max(int(self.config.cpd_stride), 1)
        if cpd_stride > 1 and self._update_counter % cpd_stride != 0:
            # Periodic-CPD mode: most frames use only the current cloud for
            # supported visible nodes.  The preceding CPD frame supplies the
            # ordered state for occluded nodes and periodically refreshes the
            # topology, avoiding a history iteration on every image.
            candidate, fused_count = self._fuse_visible_observations(
                previous,
                previous.copy(),
                visible_points,
                visible_extended if self.config.visible_observation_use_extended else visible,
            )
            previous_length = float(np.linalg.norm(np.diff(previous, axis=0), axis=1).sum())
            candidate_length = float(np.linalg.norm(np.diff(candidate, axis=0), axis=1).sum())
            ratio = candidate_length / max(previous_length, 1e-9)
            steps = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
            if (
                not 0.85 <= ratio <= 1.18
                or not np.isfinite(steps).all()
                or float(np.max(steps)) > max(
                    0.035,
                    2.5 * float(np.median(np.linalg.norm(np.diff(previous, axis=0), axis=1))),
                )
            ):
                candidate = previous.copy()
                fused_count = 0
            self._nodes = candidate
            self._history_nodes = previous
            tracking_ms = (perf_counter() - tracking_start) * 1000.0
            self._consecutive_failures = 0
            return TrackResult(
                nodes_camera=self._nodes.copy(),
                mask=mask,
                observed_points_camera=points,
                visible_nodes=np.asarray(visible, dtype=np.int32),
                self_occluded_nodes=np.asarray(self_occluded, dtype=np.int32),
                initialized=True,
                tracking_ok=True,
                nonconverged=False,
                reinitialized=False,
                failure_reason=None,
                preprocess_ms=preprocess_ms,
                tracking_ms=tracking_ms,
                total_ms=(perf_counter() - total_start) * 1000.0,
                observation_fused_nodes=int(fused_count),
            )
        try:
            cpd_seed = previous.copy()
            seed_fused_count = 0
            if self.config.pre_cpd_visible_seed:
                cpd_seed, seed_fused_count = self._fuse_visible_observations(
                    previous,
                    previous.copy(),
                    visible_points,
                    visible_extended
                    if self.config.visible_observation_use_extended
                    else visible,
                )
                if seed_fused_count and len(visible) >= 2:
                    cpd_seed = self._complete_hidden_current_anchor_spline(
                        previous, cpd_seed, visible
                    )
                seed_blend = float(np.clip(self.config.pre_cpd_seed_blend, 0.0, 1.0))
                if seed_blend < 1.0:
                    cpd_seed = previous + seed_blend * (cpd_seed - previous)
                if seed_fused_count and np.isfinite(cpd_seed).all():
                    # This is the only state passed to the native iteration.
                    # If the CPD call fails, the exception handler below
                    # restores ``previous`` in the native core.
                    self._core.initialize_nodes(cpd_seed)
                    self._core.set_sigma2(0.0)
            self._core.tracking_step(
                np.ascontiguousarray(points, dtype=np.float64),
                list(map(int, visible)),
                list(map(int, visible_extended)),
                self.projection,
                int(rgb.shape[0]),
                int(rgb.shape[1]),
            )
            candidate = np.asarray(self._core.get_tracking_result(), dtype=np.float64)
            if candidate.shape != previous.shape or not np.isfinite(candidate).all():
                raise RuntimeError("TrackDLO core returned invalid nodes")
            native_candidate = candidate.copy()
            nonconverged = bool(self._core.get_last_nonconverged())
            if nonconverged and not self.config.accept_nonconverged:
                raise RuntimeError("TrackDLO nonconverged update rejected")
            fusion_visible = visible
            fusion_extended = visible_extended
            fusion_occluded = self_occluded
            if self.config.pre_cpd_visible_seed and seed_fused_count:
                # The current-frame seed already contains the observation
                # update.  Keep it as the base for hidden completion, while
                # still allowing CPD to refine the current visible geometry.
                candidate = candidate.copy()
            primary_fusion_visible = list(fusion_visible)
            primary_fusion_extended = list(fusion_extended)
            if self.config.refine_visibility_after_cpd:
                refined_visible, refined_extended, refined_occluded = self._visibility(
                    points,
                    rgb.shape[:2],
                    mask,
                    depth_m,
                    nodes_override=candidate,
                )
                # A very small refined set is less useful than the original
                # CPD guide set and can make a transient candidate hide the
                # whole chain.  Keep the initial classification in that case.
                if len(refined_visible) >= max(2, int(self.config.min_visible_nodes)):
                    fusion_visible = refined_visible
                    fusion_extended = refined_extended
                    fusion_occluded = refined_occluded
            # A second camera can support a node even when that node is outside
            # the primary image.  Keep the native CPD guide set unchanged (the
            # C++ traversal is defined in the primary projection), but include
            # the externally supported nodes in the *post-CPD* visible/current
            # fusion set.  Thus the wrist cloud can correct visible geometry
            # without pretending that the primary camera observed it.
            if additional_fusion_visible_nodes is not None:
                additional = {
                    int(index)
                    for index in np.asarray(additional_fusion_visible_nodes).ravel()
                    if 0 <= int(index) < len(previous)
                }
                if additional:
                    fusion_visible = sorted(set(int(i) for i in fusion_visible) | additional)
                    fusion_extended = sorted(set(int(i) for i in fusion_extended) | additional)
                    fusion_occluded = sorted(
                        set(range(len(previous))) - set(fusion_visible)
                    )
            if nonconverged and self.config.hold_hidden_on_nonconverged and fusion_occluded:
                # A finite but unfinished CPD iterate is often useful for the
                # supported visible segment, but can pull an occluded tail to
                # an arbitrary local minimum.  Keep only the previous hidden
                # state, then let visible-cloud fusion correct supported nodes.
                candidate[np.asarray(fusion_occluded, dtype=np.int64)] = previous[
                    np.asarray(fusion_occluded, dtype=np.int64)
                ]
            if self.config.current_skeleton_fusion:
                skeleton_observations = self._current_skeleton_observations(
                    mask, depth_m, previous, fusion_visible
                )
                blend = float(np.clip(self.config.current_skeleton_blend, 0.0, 1.0))
                for node_index, observation in skeleton_observations.items():
                    candidate[node_index] = (
                        (1.0 - blend) * candidate[node_index] + blend * observation
                    )
                fused_count = len(skeleton_observations)
            elif self.config.visible_pixel_snap_fusion:
                snap_observations = self._visible_pixel_snap_observations(
                    mask, depth_m, candidate, fusion_visible
                )
                snap_blend = float(np.clip(self.config.visible_pixel_snap_blend, 0.0, 1.0))
                for node_index, observation in snap_observations.items():
                    candidate[node_index] = (
                        (1.0 - snap_blend) * candidate[node_index]
                        + snap_blend * observation
                    )
                fused_count = len(snap_observations)
                # Pixel skeletons can be broken by a depth hole or a crossing.
                # For nodes without a validated snap, retain the ordinary
                # local point-cloud correction instead of leaving an entire
                # visible run at the stale CPD estimate.
                fallback_nodes = sorted(
                    set(int(index) for index in np.asarray(fusion_visible).ravel())
                    - set(int(index) for index in snap_observations)
                )
                if fallback_nodes:
                    candidate, fallback_count = self._fuse_visible_observations(
                        previous, candidate, visible_points, fallback_nodes
                    )
                    fused_count += int(fallback_count)
            elif self.config.pointcloud_arc_fusion:
                path_observations = self._pointcloud_arc_observations(
                    visible_points,
                    previous,
                    fusion_extended if self.config.visible_observation_use_extended else fusion_visible,
                )
                blend = float(np.clip(self.config.pointcloud_path_blend, 0.0, 1.0))
                for node_index, observation in path_observations.items():
                    candidate[node_index] = (
                        (1.0 - blend) * candidate[node_index] + blend * observation
                    )
                fused_count = len(path_observations)
            elif self.config.pointcloud_path_fusion:
                path_observations = self._pointcloud_path_observations(
                    visible_points,
                    previous,
                    fusion_extended if self.config.visible_observation_use_extended else fusion_visible,
                )
                blend = float(np.clip(self.config.pointcloud_path_blend, 0.0, 1.0))
                for node_index, observation in path_observations.items():
                    candidate[node_index] = (
                        (1.0 - blend) * candidate[node_index] + blend * observation
                    )
                fused_count = len(path_observations)
            elif self.config.current_path_fusion and (
                ordered_visible_path_override is not None
                or ordered_visible_paths_override
            ):
                path_list = (
                    ordered_visible_paths_override
                    if ordered_visible_paths_override
                    else [ordered_visible_path_override]
                )
                best_path_observations: dict[int, np.ndarray] = {}
                best_oriented_path: np.ndarray | None = None
                best_path_score = float("inf")
                for current_path in path_list:
                    if current_path is None:
                        continue
                    current_path = np.asarray(current_path, dtype=np.float64)
                    oriented_path, visual_cost = self._visual_path_orientation(
                        current_path, self._current_path_memory
                    )
                    orientation_hint = 1 if np.isfinite(visual_cost) else None
                    path_reference = (
                        candidate if self.config.current_path_use_candidate else previous
                    )
                    one_path = self._current_path_observations(
                        oriented_path,
                        path_reference,
                        fusion_visible,
                        orientation_hint=orientation_hint,
                    )
                    if one_path:
                        path_debug = self._last_current_path_debug
                        path_cost = float(path_debug[0]) if path_debug is not None else 1e9
                        # Prefer a path that explains a larger material
                        # interval when costs are close; tiny disconnected
                        # skeleton fragments should not win merely because
                        # their endpoint happens to be nearby.
                        span_fraction = len(one_path) / max(len(previous), 1)
                        # Once a visual path is available, continuity of that
                        # path is a stronger cue than stale CPD endpoints.
                        visual_term = 1.5 * visual_cost if np.isfinite(visual_cost) else 0.0
                        score = path_cost + visual_term + 0.08 * (1.0 - span_fraction)
                        if score < best_path_score:
                            best_path_score = score
                            best_path_observations = one_path
                            best_oriented_path = oriented_path
                path_observations = best_path_observations
                if path_observations:
                    # A current RGB-D path is evidence only where this frame
                    # actually supports a node.  Do not overwrite the
                    # occluded interval with a path extrapolation; that is
                    # precisely the stale-history failure this fusion mode is
                    # meant to avoid.
                    visible_set = {
                        int(index)
                        for index in np.asarray(fusion_visible).ravel()
                        if 0 <= int(index) < len(previous)
                    }
                    path_observations = {
                        index: observation
                        for index, observation in path_observations.items()
                        if index in visible_set
                    }
                if path_observations:
                    path_blend = float(np.clip(self.config.current_path_blend, 0.0, 1.0))
                    for node_index, observation in path_observations.items():
                        candidate[node_index] = (
                            (1.0 - path_blend) * candidate[node_index]
                            + path_blend * observation
                        )
                    fused_count = len(path_observations)
                    if best_oriented_path is not None:
                        self._current_path_endpoints = (
                            best_oriented_path[0].copy(),
                            best_oriented_path[-1].copy(),
                        )
                        self._current_path_memory = best_oriented_path.copy()
                elif self.config.visible_observation_fusion:
                    # A short or broken current skeleton is not enough to
                    # identify a material interval.  Retain the ordinary
                    # local cloud correction instead of silently disabling
                    # visible fusion for that frame.
                    candidate, fused_count = self._fuse_visible_observations(
                        previous,
                        candidate,
                        visible_points,
                        fusion_extended
                        if self.config.visible_observation_use_extended
                        else fusion_visible,
                    )
            elif self.config.visible_observation_fusion:
                fusion_fn = (
                    self._fuse_visible_observations_monotonic
                    if self.config.monotonic_visible_fusion
                    else self._fuse_visible_observations
                )
                candidate, fused_count = fusion_fn(
                    previous,
                    candidate,
                    visible_points,
                    primary_fusion_extended
                    if self.config.visible_observation_use_extended
                    else primary_fusion_visible,
                )
                # Keep the two views' observations separate for nodes that are
                # supported only by the secondary camera.  A merged cloud can
                # contain a nearby primary-view branch at a crossing; applying
                # the wrist cloud alone prevents that branch from winning the
                # local arc assignment.
                if (
                    additional_visible_points is not None
                    and additional_fusion_visible_nodes is not None
                ):
                    external_supported = {
                        int(index)
                        for index in np.asarray(additional_fusion_visible_nodes).ravel()
                        if 0 <= int(index) < len(previous)
                    }
                    external_only = sorted(
                        external_supported
                        if additional_fusion_all_nodes
                        else external_supported - set(int(index) for index in primary_fusion_visible)
                    )
                    if external_only:
                        external_fn = (
                            self._fuse_visible_observations_monotonic
                            if self.config.monotonic_visible_fusion
                            else self._fuse_visible_observations
                        )
                        candidate, external_count = external_fn(
                            previous,
                            candidate,
                            additional_visible_points,
                            external_only,
                        )
                        fused_count += int(external_count)
            if fused_count and len(fusion_visible) >= 2:
                if self.config.post_cpd_current_spline:
                    candidate = self._complete_hidden_current_anchor_spline(
                        previous, candidate, fusion_visible
                    )
                elif self.config.hidden_piecewise_completion:
                    candidate = self._complete_hidden_piecewise(
                        previous, candidate, fusion_visible
                    )
                elif self.config.hidden_history_completion:
                    candidate = self._complete_hidden_from_history(
                        previous, candidate, fusion_visible
                    )
            if fused_count and self.config.hidden_hold_previous:
                hidden_indices = sorted(
                    set(range(len(previous))) - set(int(i) for i in fusion_visible)
                )
                if hidden_indices:
                    candidate[hidden_indices] = previous[hidden_indices]
            if (
                fused_count
                and self.config.hidden_velocity_prediction
                and older is not None
                and older.shape == previous.shape
            ):
                velocity_gate = float(self.config.hidden_velocity_deformation_threshold_m)
                velocity_allowed = (
                    velocity_gate <= 0.0
                    or (
                        np.isfinite(deformation_residual)
                        and deformation_residual > velocity_gate
                    )
                )
                hidden_indices = sorted(
                    set(range(len(previous))) - set(int(i) for i in fusion_visible)
                )
                if hidden_indices and velocity_allowed:
                    decay = float(np.clip(self.config.hidden_velocity_decay, 0.0, 1.0))
                    prediction = previous + decay * (previous - older)
                    max_step = float(self.config.hidden_velocity_max_step_m)
                    trial = candidate.copy()
                    trial[hidden_indices] = prediction[hidden_indices]
                    prediction_steps = np.linalg.norm(np.diff(trial, axis=0), axis=1)
                    if (
                        max_step <= 0.0
                        or (
                            np.isfinite(prediction_steps).all()
                            and float(np.max(prediction_steps)) <= max_step
                        )
                    ):
                        candidate = trial
            if fused_count:
                # A local observation must not be allowed to change the
                # topology/length of the whole DLO.  This guard is especially
                # important near crossings, where a 2-D cable mask can be
                # assigned to the wrong branch for one frame.  In that case
                # keep the native CPD result and let the next frame recover.
                fused_steps = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
                if self.config.current_path_fusion:
                    # The current path is allowed to change the visible
                    # interval's total arc length during deformation.  Only
                    # reject numerically invalid or obviously discontinuous
                    # adjacent samples; the native whole-chain length gate
                    # would erase precisely the current-frame update we want.
                    invalid_path = (
                        not np.isfinite(fused_steps).all()
                        or float(np.max(fused_steps)) > 0.08
                    )
                else:
                    previous_length = float(
                        np.linalg.norm(np.diff(previous, axis=0), axis=1).sum()
                    )
                    fused_length = float(
                        np.linalg.norm(np.diff(candidate, axis=0), axis=1).sum()
                    )
                    length_ratio = fused_length / max(previous_length, 1e-9)
                    previous_steps = np.linalg.norm(np.diff(previous, axis=0), axis=1)
                    step_limit = max(0.035, 2.5 * float(np.median(previous_steps)))
                    invalid_path = (
                        not 0.85 <= length_ratio <= 1.18
                        or not np.isfinite(fused_steps).all()
                        or float(np.max(fused_steps)) > step_limit
                    )
                if invalid_path:
                    candidate = native_candidate
                    fused_count = 0
            if fused_count and (
                self.config.current_skeleton_fusion or self.config.current_path_fusion
            ):
                # Keep the native core synchronized with the fused state;
                # otherwise the next CPD call would still start from the
                # pre-fusion candidate held internally by C++.
                self._core.initialize_nodes(candidate)
                self._core.set_sigma2(0.0)
            self._nodes = candidate
            self._history_nodes = previous
            ok, reason = True, None
        except Exception as exc:  # retain last valid state for an analyzable failure frame
            self._nodes = previous
            # A guarded native failure may leave internal Y/guide matrices
            # partially updated.  Restore the last valid state before the next
            # frame so one bad sparse observation cannot poison the tracker.
            try:
                self._core.initialize_nodes(previous)
                self._core.set_sigma2(0.0)
            except Exception:
                pass
            ok, reason = False, f"core_error:{type(exc).__name__}:{exc}"
        tracking_ms = (perf_counter() - tracking_start) * 1000.0
        if not ok:
            return self._failed_update(
                rgb,
                depth,
                mask,
                points,
                visible,
                self_occluded,
                reason or "core_error",
                preprocess_ms,
                tracking_ms,
                total_start,
            )
        self._consecutive_failures = 0
        return TrackResult(
            nodes_camera=self._nodes.copy(),
            mask=mask,
            observed_points_camera=points,
            visible_nodes=np.asarray(fusion_visible, dtype=np.int32),
            self_occluded_nodes=np.asarray(fusion_occluded, dtype=np.int32),
            initialized=True,
            tracking_ok=True,
            nonconverged=nonconverged,
            reinitialized=False,
            failure_reason=None,
            preprocess_ms=preprocess_ms,
            tracking_ms=tracking_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
            observation_fused_nodes=int(fused_count),
            deformation_residual_m=float(deformation_residual),
            deformation_detected=bool(deformation_detected),
        )

    def _failed_update(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        points: np.ndarray,
        visible,
        self_occluded,
        reason: str,
        preprocess_ms: float,
        tracking_ms: float,
        total_start: float,
    ) -> TrackResult:
        self._consecutive_failures += 1
        threshold = self.config.reinitialize_after_failures
        if threshold > 0 and self._consecutive_failures >= threshold:
            old_core = self._core
            old_nodes = self._nodes.copy()
            old_geodesic = self._geodesic.copy()
            old_reference_length = self._reference_length
            failed_count = self._consecutive_failures
            try:
                recovered = self.initialize(rgb, depth)
                recovered_length = float(self._geodesic[-1])
                ratio = recovered_length / max(float(old_reference_length), 1e-9)
                if not (
                    self.config.reinitialize_length_ratio_min
                    <= ratio
                    <= self.config.reinitialize_length_ratio_max
                ):
                    raise RuntimeError(f"reinitialized_length_ratio={ratio:.3f}")
                recovered.reinitialized = True
                recovered.total_ms = (perf_counter() - total_start) * 1000.0
                return recovered
            except Exception as exc:
                self._core = old_core
                self._nodes = old_nodes
                self._geodesic = old_geodesic
                self._reference_length = old_reference_length
                self._consecutive_failures = failed_count
                reason = f"{reason};reinitialize_failed:{type(exc).__name__}:{exc}"
        returned_nodes = (
            self._nodes.copy()
            if self.config.hold_last_on_failure
            else np.full_like(self._nodes, np.nan, dtype=np.float64)
        )
        return TrackResult(
            nodes_camera=returned_nodes,
            mask=mask,
            observed_points_camera=points,
            visible_nodes=np.asarray(visible, dtype=np.int32),
            self_occluded_nodes=np.asarray(self_occluded, dtype=np.int32),
            initialized=True,
            tracking_ok=False,
            nonconverged=False,
            reinitialized=False,
            failure_reason=reason,
            preprocess_ms=preprocess_ms,
            tracking_ms=tracking_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
            observation_fused_nodes=0,
        )

    def _best_ordered_path_segment(
        self,
        path: np.ndarray,
        previous: np.ndarray,
        run: list[int],
        expected_length: float,
        min_ratio: float,
        max_ratio: float,
        metric_path: np.ndarray | None = None,
        metric_previous: np.ndarray | None = None,
        metric_scale: float = 1.0,
    ) -> tuple[np.ndarray, float] | None:
        """Match a current ordered path to one topological node run.

        Endpoint-only matching is fragile when the old chain has drifted or
        a crossing makes the nearest endpoint ambiguous.  This matcher
        searches contiguous current-path segments with the expected arc
        length and scores a monotonic, uniformly resampled segment against
        *all* nodes in the run.  It keeps the previous chain only as an index
        reference; the returned positions come entirely from this frame.
        """
        if len(path) < 4 or len(run) < 2 or expected_length <= 1e-6:
            return None
        # Skeletons can contain one pixel sample per row/column (hundreds or
        # thousands of points).  The node state has only 45 samples, so a
        # length-preserving reduction to a few hundred path samples retains
        # the geometry while avoiding a quadratic Python search per frame.
        metric = path if metric_path is None else np.asarray(metric_path, dtype=np.float64)
        if metric_path is not None:
            metric = metric * max(float(metric_scale), 1e-9)
        if len(metric) != len(path):
            return None
        if len(path) > 220:
            path = resample_polyline(path, 220)
            metric = resample_polyline(metric, 220)
        cumulative = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        )
        lower = max(float(min_ratio) * expected_length, 1e-6)
        upper = float(max_ratio) * expected_length + 0.08
        target = np.asarray(
            previous[run] if metric_previous is None else metric_previous[run],
            dtype=np.float64,
        )
        if metric_previous is not None:
            target = target * max(float(metric_scale), 1e-9)
        best: tuple[float, np.ndarray] | None = None

        # Evaluate a small set of length scales from every possible start.
        # Interpolating directly on the cumulative path replaces the old
        # start/end nested loop (and thousands of repeated resamplings) with
        # O(path_samples * scales * node_count) work.
        start_stride = 2 if len(path) > 150 else 1
        scales = (0.50, 0.75, 1.00, 1.25, 1.50, 1.80)
        for start in range(0, len(path) - 2, start_stride):
            start_arc = float(cumulative[start])
            for scale in scales:
                segment_length = float(expected_length) * scale
                if segment_length < lower or segment_length > upper:
                    continue
                end_arc = start_arc + segment_length
                if end_arc > float(cumulative[-1]):
                    continue
                samples_arc = np.linspace(start_arc, end_arc, len(run))
                sampled = np.column_stack(
                    [np.interp(samples_arc, cumulative, path[:, axis]) for axis in range(path.shape[1])]
                )
                sampled_metric = np.column_stack(
                    [np.interp(samples_arc, cumulative, metric[:, axis]) for axis in range(metric.shape[1])]
                )
                normalized_length_error = abs(scale - 1.0)
                direct_cost = float(np.mean(np.linalg.norm(sampled_metric - target, axis=1)))
                endpoint_cost = float(
                    0.25
                    * (
                        np.linalg.norm(sampled_metric[0] - target[0])
                        + np.linalg.norm(sampled_metric[-1] - target[-1])
                    )
                )
                cost = direct_cost + endpoint_cost + 0.02 * normalized_length_error
                if best is None or cost < best[0]:
                    best = (cost, sampled)
                # A projected cable can be traversed in either direction.
                reverse_cost = float(np.mean(np.linalg.norm(sampled_metric[::-1] - target, axis=1)))
                reverse_endpoint = float(
                    0.25
                    * (
                        np.linalg.norm(sampled_metric[-1] - target[0])
                        + np.linalg.norm(sampled_metric[0] - target[-1])
                    )
                )
                reverse_total = reverse_cost + reverse_endpoint + 0.02 * normalized_length_error
                if best is None or reverse_total < best[0]:
                    best = (reverse_total, sampled[::-1])
        if best is None:
            return None
        return best[1], float(best[0])

    def _current_skeleton_observations(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        previous: np.ndarray,
        visible: list[int] | np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Estimate current visible-node positions from this frame's image.

        The skeleton is extracted without using the CPD state.  Previous nodes
        are used only to choose the topological index interval occupied by the
        visible component, so crossings do not reorder the returned nodes.
        """
        if not len(visible):
            return {}
        try:
            if self.config.current_skeleton_depth_path:
                pixel_paths = depth_skeleton_paths(mask, depth_m, self.intrinsics)
            elif self.config.current_skeleton_simple_path:
                pixel_paths = [ordered_skeleton_pixels_simple(mask)]
            else:
                pixel_paths = [ordered_skeleton_pixels(mask)]
        except Exception:
            return {}
        height, width = depth_m.shape[:2]
        paths = []
        for pixels in pixel_paths:
            points = []
            valid_pixels = []
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
                        (float(col) - self.intrinsics[0, 2]) * value / self.intrinsics[0, 0],
                        (float(row) - self.intrinsics[1, 2]) * value / self.intrinsics[1, 1],
                        value,
                    )
                )
                valid_pixels.append((float(col), float(row)))
            if len(points) < 4:
                continue
            path = np.asarray(points, dtype=np.float64)
            keep = np.concatenate(([True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-7))
            path = path[keep]
            if len(path) >= 4:
                paths.append((path, np.asarray(valid_pixels, dtype=np.float64)[keep]))
        if not paths:
            return {}

        visible_sorted = sorted(set(int(i) for i in visible if 0 <= int(i) < len(previous)))
        runs: list[list[int]] = []
        for index in visible_sorted:
            if not runs or index != runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)
        if not runs:
            return {}

        reference_length = max(float(self._reference_length or 0.0), 1e-6)
        observations: dict[int, np.ndarray] = {}

        # If a single current path contains almost the complete cable, use its
        # current-frame uniform samples directly for all supported indices.
        if len(paths) == 1:
            path = paths[0][0]
            path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if (
                0.85 * reference_length <= path_length <= 1.30 * reference_length
                and len(path) >= 20
                and len(visible_sorted) >= int(0.75 * len(previous))
            ):
                current_nodes = resample_polyline(path, len(previous))
                direct_cost = np.linalg.norm(current_nodes[0] - previous[0]) + np.linalg.norm(
                    current_nodes[-1] - previous[-1]
                )
                reverse_cost = np.linalg.norm(current_nodes[-1] - previous[0]) + np.linalg.norm(
                    current_nodes[0] - previous[-1]
                )
                if reverse_cost < direct_cost:
                    current_nodes = current_nodes[::-1]
                for index in visible_sorted:
                    observations[index] = current_nodes[index]
                return observations

        # For a partially visible image, match the ordered component to the
        # visible topological run using all node positions, rather than only
        # the two endpoints.  This is more stable when the old chain has
        # drifted or the 2-D mask contains a crossing.
        previous_cumulative = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(previous, axis=0), axis=1)))
        )
        previous_pixels = project_camera_points(previous, self.intrinsics)
        for run in runs:
            segment_length = float(
                previous_cumulative[run[-1]] - previous_cumulative[run[0]]
            )
            best_run = None
            for path, path_pixels in paths:
                match = self._best_ordered_path_segment(
                    path,
                    previous,
                    run,
                    segment_length,
                    0.45,
                    1.8,
                    metric_path=path_pixels,
                    metric_previous=previous_pixels,
                    metric_scale=float(np.median(path[:, 2]) / max(self.intrinsics[0, 0], 1e-9)),
                )
                if match is None:
                    continue
                sampled, cost = match
                if best_run is None or cost < best_run[0]:
                    best_run = (cost, sampled)
            if best_run is None or best_run[0] > 0.08:
                continue
            for index, observation in zip(run, best_run[1]):
                observations[index] = observation
        return observations

    def _pointcloud_path_observations(
        self,
        points: np.ndarray,
        previous: np.ndarray,
        visible: list[int] | np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Order the current 3-D cloud with a radius-neighbour graph.

        Unlike a 2-D skeleton, the graph uses depth to separate crossing
        branches.  The longest shortest path in the largest connected cloud
        component is used only for visible-node observations; hidden nodes
        remain the native TrackDLO estimate.
        """
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 8 or not np.isfinite(points).all():
            return {}
        try:
            neighbour_count = min(10, len(points) - 1)
            distances, indices = cKDTree(points).query(points, k=neighbour_count + 1)
            rows: list[int] = []
            cols: list[int] = []
            weights: list[float] = []
            for index in range(len(points)):
                for distance, neighbour in zip(distances[index, 1:], indices[index, 1:]):
                    if np.isfinite(distance) and distance <= 0.045:
                        rows.append(index)
                        cols.append(int(neighbour))
                        weights.append(float(distance))
            if not weights:
                return {}
            graph = csr_matrix((weights, (rows, cols)), shape=(len(points), len(points)))
            component_count, labels = connected_components(graph, directed=False)
            if component_count <= 0:
                return {}
            component_sizes = np.bincount(labels)
            component = int(np.argmax(component_sizes))
            members = np.flatnonzero(labels == component)
            if len(members) < 8:
                return {}
            subgraph = graph[members][:, members]
            # Two sweeps provide a good graph-diameter approximation while
            # avoiding an O(N^2) all-pairs shortest-path calculation.
            dist0 = dijkstra(subgraph, indices=0)
            first = int(np.argmax(np.where(np.isfinite(dist0), dist0, -1.0)))
            dist1, predecessor = dijkstra(
                subgraph, indices=first, return_predecessors=True
            )
            second = int(np.argmax(np.where(np.isfinite(dist1), dist1, -1.0)))
            path_local = [second]
            while path_local[-1] != first:
                parent = int(predecessor[path_local[-1]])
                if parent < 0 or parent == path_local[-1]:
                    return {}
                path_local.append(parent)
                if len(path_local) > len(members):
                    return {}
            path = points[members[np.asarray(path_local[::-1], dtype=np.int64)]]
        except (ValueError, RuntimeError, IndexError):
            return {}
        if len(path) < 4:
            return {}
        keep = np.concatenate(([True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-7))
        path = path[keep]
        if len(path) < 4:
            return {}
        path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        previous_cumulative = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(previous, axis=0), axis=1)))
        )
        visible_sorted = sorted(set(int(i) for i in visible if 0 <= int(i) < len(previous)))
        runs: list[list[int]] = []
        for index in visible_sorted:
            if not runs or index != runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)
        if not runs:
            return {}
        reference_length = max(float(self._reference_length or 0.0), 1e-6)
        observations: dict[int, np.ndarray] = {}
        if (
            0.85 * reference_length <= path_length <= 1.25 * reference_length
            and len(visible_sorted) >= int(0.75 * len(previous))
        ):
            sampled = resample_polyline(path, len(previous))
            direct = np.linalg.norm(sampled[0] - previous[0]) + np.linalg.norm(
                sampled[-1] - previous[-1]
            )
            reverse = np.linalg.norm(sampled[-1] - previous[0]) + np.linalg.norm(
                sampled[0] - previous[-1]
            )
            if reverse < direct:
                sampled = sampled[::-1]
            for index in visible_sorted:
                observations[index] = sampled[index]
            return observations
        best = None
        for run in runs:
            expected = float(previous_cumulative[run[-1]] - previous_cumulative[run[0]])
            if expected <= 1e-6:
                continue
            match = self._best_ordered_path_segment(
                path, previous, run, expected, 0.55, 1.65
            )
            if match is None:
                continue
            sampled, cost = match
            if best is None or cost < best[0]:
                best = (cost, run, sampled)
        if best is None or best[0] > 0.08:
            return {}
        _, run, sampled = best
        for index, observation in zip(run, sampled):
            observations[index] = observation
        return observations

    def _current_path_observations(
        self,
        path: np.ndarray,
        previous: np.ndarray,
        visible: list[int] | np.ndarray,
        orientation_hint: int | None = None,
    ) -> dict[int, np.ndarray]:
        """Map an ordered current-frame path to a material interval.

        Unlike CPD, this routine never deforms the path toward the previous
        chain.  The previous state contributes only endpoint orientation and
        a weak interval-index cue; all returned visible positions are direct
        samples of the current RGB-D path.  Nodes outside the selected
        interval are deliberately left to the caller's native completion.
        """
        path = np.asarray(path, dtype=np.float64)
        previous = np.asarray(previous, dtype=np.float64)
        if (
            path.ndim != 2
            or path.shape[1] != 3
            or len(path) < 4
            or previous.ndim != 2
            or previous.shape[1] != 3
            or len(previous) < 2
            or not np.isfinite(path).all()
            or not np.isfinite(previous).all()
        ):
            return {}
        keep = np.concatenate(
            ([True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-7)
        )
        path = path[keep]
        if len(path) < 4:
            return {}
        cumulative = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        )
        path_length = float(cumulative[-1])
        if path_length <= 1e-5:
            return {}

        # Resample before a short moving-average pass so pixel-scale skeleton
        # jitter does not turn into large 3-D node oscillations.
        dense_count = max(80, min(300, len(path) * 2))
        dense_arcs = np.linspace(0.0, path_length, dense_count)
        dense = np.column_stack(
            [np.interp(dense_arcs, cumulative, path[:, axis]) for axis in range(3)]
        )
        if len(dense) >= 7:
            smooth = np.empty_like(dense)
            for index in range(len(dense)):
                left = max(0, index - 2)
                right = min(len(dense), index + 3)
                smooth[index] = np.mean(dense[left:right], axis=0)
            if smooth.shape == dense.shape and np.isfinite(smooth).all():
                dense = smooth
        dense_cumulative = np.linspace(0.0, path_length, len(dense))

        reference_length = max(float(self._reference_length or 0.0), 1e-6)
        if reference_length <= 1e-5:
            reference_length = float(
                np.linalg.norm(np.diff(previous, axis=0), axis=1).sum()
            )
        spacing = reference_length / max(len(previous) - 1, 1)
        if spacing <= 1e-6:
            return {}
        visible_set = sorted(
            set(int(i) for i in np.asarray(visible).ravel() if 0 <= int(i) < len(previous))
        )
        runs: list[list[int]] = []
        for index in visible_set:
            if not runs or index != runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)

        if self.config.current_path_run_locked and visible_set:
            # A visible run is already a material-index cue supplied by the
            # image/depth visibility test.  Do not search every historical
            # interval: after a bend, that search can select a geometrically
            # similar branch and swap the node order.  The current path is
            # sampled over the run itself; the previous chain is only used to
            # choose orientation and the run label.
            def sample_oriented(oriented_path: np.ndarray, count: int) -> np.ndarray:
                # Skeleton pixels are sampled at nearly uniform image arc
                # length, while a depth hole can create a large 3-D step.  Use
                # the projected 2-D arc as the parameter whenever projection
                # is valid; this keeps node spacing tied to the actual mask
                # path and avoids assigning a disproportionate material span
                # to a single noisy depth sample.
                projected = project_camera_points(oriented_path, self.intrinsics)
                projected_steps = np.linalg.norm(np.diff(projected, axis=0), axis=1)
                projected_cumulative = np.concatenate(
                    ([0.0], np.cumsum(projected_steps))
                )
                spatial_cumulative = np.concatenate(
                    ([0.0], np.cumsum(np.linalg.norm(np.diff(oriented_path, axis=0), axis=1)))
                )
                use_projected = (
                    np.isfinite(projected_cumulative).all()
                    and float(projected_cumulative[-1]) > 1e-6
                )
                path_cumulative = projected_cumulative if use_projected else spatial_cumulative
                path_len = float(path_cumulative[-1])
                arcs = np.linspace(0.0, path_len, max(int(count), 2))
                return np.column_stack(
                    [
                        np.interp(arcs, path_cumulative, oriented_path[:, axis])
                        for axis in range(3)
                    ]
                )

            reference_length_locked = max(float(self._reference_length or 0.0), 1e-6)
            full_visible_fraction = len(visible_set) / max(len(previous), 1)
            if (
                full_visible_fraction >= 0.70
                and 0.65 * reference_length_locked <= path_length <= 1.45 * reference_length_locked
            ):
                direct = sample_oriented(dense, len(previous))
                reverse = direct[::-1]
                direct_cost = float(
                    np.linalg.norm(direct[0] - previous[0])
                    + np.linalg.norm(direct[-1] - previous[-1])
                )
                reverse_cost = float(
                    np.linalg.norm(reverse[0] - previous[0])
                    + np.linalg.norm(reverse[-1] - previous[-1])
                )
                sampled = reverse if reverse_cost < direct_cost else direct
                cost = min(direct_cost, reverse_cost)
                self._last_current_path_debug = (float(cost), 0, len(previous) - 1)
                return {index: sampled[index] for index in visible_set}

            best_locked: tuple[float, list[int], np.ndarray] | None = None
            for run in runs:
                if len(run) < 2:
                    continue
                expected = reference_length_locked * float(run[-1] - run[0]) / max(
                    len(previous) - 1, 1
                )
                if expected <= 1e-6:
                    continue
                path_cumulative = np.concatenate(
                    ([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1)))
                )
                current_path_len = float(path_cumulative[-1])
                # A broken mask path may be shorter than the material run.  It
                # is still useful for visible-node geometry, but a path that is
                # orders of magnitude longer is likely a crossing jump.
                if current_path_len < 0.30 * expected or current_path_len > 2.8 * expected:
                    continue
                for orientation in (1, -1):
                    oriented = dense if orientation == 1 else dense[::-1]
                    sampled = sample_oriented(oriented, len(run))
                    target = previous[np.asarray(run, dtype=np.int64)]
                    endpoint_cost = float(
                        np.linalg.norm(sampled[0] - target[0])
                        + np.linalg.norm(sampled[-1] - target[-1])
                    )
                    shape_cost = float(np.mean(np.linalg.norm(sampled - target, axis=1)))
                    length_cost = abs(current_path_len - expected) / reference_length_locked
                    # Run locking makes the index cue discrete; use geometry
                    # only to reject a clearly wrong orientation/path.
                    score = endpoint_cost + 0.25 * shape_cost + 0.04 * length_cost
                    if best_locked is None or score < best_locked[0]:
                        best_locked = (score, run, sampled)
            if best_locked is not None:
                score, run, sampled = best_locked
                self._last_current_path_debug = (float(score), int(run[0]), int(run[-1]))
                return {index: observation for index, observation in zip(run, sampled)}

        # If the native chain has already stretched or jumped branches, using
        # its endpoints to assign the current path is self-reinforcing: a bad
        # history chooses a bad interval and the next frame gets worse.  In
        # that case use the visible-node run only as a coarse material anchor
        # and let the current ordered RGB-D path determine every position in
        # that run.  This is the intended current-visible/history-hidden
        # recovery path; it activates only for a near-complete visible cable.
        previous_length = float(np.linalg.norm(np.diff(previous, axis=0), axis=1).sum())
        previous_steps = np.linalg.norm(np.diff(previous, axis=0), axis=1)
        history_unreliable = (
            previous_length / reference_length < 0.82
            or previous_length / reference_length > 1.22
            or (len(previous_steps) and float(np.max(previous_steps)) > 0.055)
        )
        if history_unreliable and path_length >= 0.84 * reference_length and visible_set:
            start = int(min(visible_set))
            span = max(2, int(round(path_length / spacing)))
            end = min(len(previous) - 1, start + span)
            if end - start >= 2:
                samples_arc = np.linspace(0.0, path_length, end - start + 1)
                sampled = np.column_stack(
                    [
                        np.interp(samples_arc, dense_cumulative, dense[:, axis])
                        for axis in range(3)
                    ]
                )
                self._last_current_path_debug = (0.0, start, end)
                return {
                    index: observation
                    for index, observation in zip(range(start, end + 1), sampled)
                }

        # Candidate material intervals.  A path close to the complete cable
        # is allowed to populate all 45 nodes; a shorter path searches only
        # intervals whose physical arc length agrees with the observation.
        interval_candidates: list[tuple[int, int]] = []
        if 0.78 * reference_length <= path_length <= 1.38 * reference_length:
            interval_candidates.append((0, len(previous) - 1))
        expected_span = path_length / spacing
        for start in range(len(previous) - 1):
            centre_end = int(round(start + expected_span))
            for end in range(max(start + 2, centre_end - 2), min(len(previous), centre_end + 3)):
                if end <= start:
                    continue
                if (start, end) not in interval_candidates:
                    interval_candidates.append((start, end))
        if not interval_candidates:
            return {}

        previous_cumulative = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(previous, axis=0), axis=1)))
        )
        best: tuple[float, int, int, np.ndarray] | None = None
        orientations = (int(orientation_hint),) if orientation_hint in (1, -1) else (1, -1)
        for orientation in orientations:
            oriented = dense if orientation == 1 else dense[::-1]
            start_point = oriented[0]
            end_point = oriented[-1]
            for start, end in interval_candidates:
                expected_length = spacing * float(end - start)
                length_error = abs(path_length - expected_length) / max(reference_length, 1e-6)
                endpoint_cost = float(
                    np.linalg.norm(start_point - previous[start])
                    + np.linalg.norm(end_point - previous[end])
                )
                # Endpoint identity is useful for orientation but can become
                # misleading after a bend/crossing.  Add a trimmed whole-run
                # shape cost so the direction is selected from the complete
                # visible geometry, not from whichever endpoint happens to be
                # nearest the stale historical endpoint.  The previous chain
                # is used only as an index/orientation cue; the returned
                # positions are still sampled directly from the current path.
                if visible_set:
                    inside = sum(start <= index <= end for index in visible_set)
                    coverage_penalty = 0.012 * (len(visible_set) - inside) / len(visible_set)
                else:
                    coverage_penalty = 0.02
                samples_arc = np.linspace(0.0, path_length, end - start + 1)
                sampled = np.column_stack(
                    [
                        np.interp(samples_arc, dense_cumulative, oriented[:, axis])
                        for axis in range(3)
                    ]
                )
                node_cost = float(
                    np.mean(np.linalg.norm(sampled - previous[start : end + 1], axis=1))
                )
                cost = (
                    endpoint_cost
                    + 0.35 * node_cost
                    + 0.10 * length_error
                    + coverage_penalty
                )
                if best is None or cost < best[0]:
                    best = (cost, start, end, sampled)
        if best is None:
            return {}
        cost, start, end, sampled = best
        self._last_current_path_debug = (float(cost), int(start), int(end))
        # Reject an interval that is not plausibly connected to the previous
        # chain at either end.  The threshold is intentionally loose because
        # this is an orientation/index cue, not a geometric fit constraint.
        # The endpoint cue can be large during fast motion even when the
        # current path is internally smooth.  Scale the acceptance threshold
        # with the calibrated cable length instead of rejecting every useful
        # current-frame path after a sizeable deformation.
        if cost > max(0.18, 0.65 * reference_length):
            return {}
        return {
            index: observation
            for index, observation in zip(range(start, end + 1), sampled)
        }

    @staticmethod
    def _visual_path_orientation(path: np.ndarray, memory: np.ndarray | None) -> tuple[np.ndarray, float]:
        """Orient a current path against the previous visual path.

        Endpoint-only orientation is ambiguous at projected crossings.  A
        small arc-length-resampled whole-path distance preserves the material
        direction across frames while allowing the cable to deform.  The
        returned cost is only a selection cue; the path samples remain the
        current-frame observations.
        """
        path = np.asarray(path, dtype=np.float64)
        if memory is None or len(path) < 4 or len(memory) < 4:
            return path, float("inf")
        try:
            count = min(80, max(12, min(len(path), len(memory))))
            current = resample_polyline(path, count)
            previous = resample_polyline(memory, count)
            direct = float(np.mean(np.linalg.norm(current - previous, axis=1)))
            reverse = float(np.mean(np.linalg.norm(current[::-1] - previous, axis=1)))
            if reverse < direct:
                return path[::-1], reverse
            return path, direct
        except (ValueError, RuntimeError, FloatingPointError):
            return path, float("inf")

    def _pointcloud_arc_observations(
        self,
        points: np.ndarray,
        previous: np.ndarray,
        visible: list[int] | np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Map a current 3-D cloud path to visible node indices by arc length.

        The ordinary path fusion scores every path sample against every old
        node.  That is accurate only while the old shape is trustworthy; at a
        deformation or crossing it can select a geometrically similar but
        topologically wrong branch.  This variant uses the current cloud for
        the *entire* visible geometry and uses the previous chain only for the
        material interval endpoints and orientation.  The expected interval
        length comes from the reference cable length, not the deformed old
        polyline.
        """
        points = np.asarray(points, dtype=np.float64)
        previous = np.asarray(previous, dtype=np.float64)
        if len(points) < 8 or len(previous) < 2 or not np.isfinite(points).all():
            return {}
        try:
            neighbour_count = min(10, len(points) - 1)
            distances, indices = cKDTree(points).query(points, k=neighbour_count + 1)
            rows: list[int] = []
            cols: list[int] = []
            weights: list[float] = []
            for index in range(len(points)):
                for distance, neighbour in zip(distances[index, 1:], indices[index, 1:]):
                    if np.isfinite(distance) and distance <= 0.045:
                        rows.append(index)
                        cols.append(int(neighbour))
                        weights.append(float(distance))
            if not weights:
                return {}
            graph = csr_matrix((weights, (rows, cols)), shape=(len(points), len(points)))
            component_count, labels = connected_components(graph, directed=False)
            if component_count <= 0:
                return {}
            component_sizes = np.bincount(labels)
            component = int(np.argmax(component_sizes))
            members = np.flatnonzero(labels == component)
            if len(members) < 8:
                return {}
            subgraph = graph[members][:, members]
            dist0 = dijkstra(subgraph, indices=0)
            first = int(np.argmax(np.where(np.isfinite(dist0), dist0, -1.0)))
            dist1, predecessor = dijkstra(
                subgraph, indices=first, return_predecessors=True
            )
            second = int(np.argmax(np.where(np.isfinite(dist1), dist1, -1.0)))
            path_local = [second]
            while path_local[-1] != first:
                parent = int(predecessor[path_local[-1]])
                if parent < 0 or parent == path_local[-1]:
                    return {}
                path_local.append(parent)
                if len(path_local) > len(members):
                    return {}
            path = points[members[np.asarray(path_local[::-1], dtype=np.int64)]]
        except (ValueError, RuntimeError, IndexError):
            return {}
        if len(path) < 4:
            return {}
        keep = np.concatenate(([True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-7))
        path = path[keep]
        if len(path) < 4:
            return {}
        cumulative = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        )
        path_length = float(cumulative[-1])
        if path_length <= 1e-6:
            return {}

        visible_sorted = sorted(
            set(int(i) for i in visible if 0 <= int(i) < len(previous))
        )
        runs: list[list[int]] = []
        for index in visible_sorted:
            if not runs or index != runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)
        if not runs:
            return {}

        reference_length = max(float(self._reference_length or 0.0), 1e-6)
        node_spacing = reference_length / max(len(previous) - 1, 1)
        observations: dict[int, np.ndarray] = {}
        occupied_intervals: list[tuple[float, float]] = []
        for run in runs:
            if len(run) == 1:
                # A single visible node is under-constrained by an ordered
                # path; let the regular local fusion handle it instead.
                continue
            expected = node_spacing * float(run[-1] - run[0])
            if expected <= 1e-5:
                continue
            best: tuple[float, np.ndarray, float, float] | None = None
            # Endpoint anchors determine which material interval is visible;
            # the rest of the node locations are uniform samples of the
            # current path and therefore do not inherit the previous shape.
            target_start = previous[run[0]]
            target_end = previous[run[-1]]
            for orientation in (1, -1):
                oriented = path if orientation == 1 else path[::-1]
                oriented_cumulative = (
                    cumulative if orientation == 1 else path_length - cumulative[::-1]
                )
                for start_index in range(0, len(oriented) - 2, max(1, len(oriented) // 180)):
                    start_arc = float(oriented_cumulative[start_index])
                    for scale in (0.75, 0.90, 1.00, 1.10, 1.25):
                        segment_length = expected * scale
                        end_arc = start_arc + segment_length
                        if end_arc > path_length:
                            continue
                        samples_arc = np.linspace(start_arc, end_arc, len(run))
                        sampled = np.column_stack(
                            [
                                np.interp(samples_arc, oriented_cumulative, oriented[:, axis])
                                for axis in range(3)
                            ]
                        )
                        if self.config.pointcloud_arc_anchor_only:
                            # At a deformation the last visible node can
                            # already have jumped to another branch.  Anchor
                            # the current path at the first visible material
                            # index and use the known cable spacing to march
                            # forward; this prevents a bad endpoint from
                            # selecting a long, geometrically similar loop.
                            endpoint_cost = float(
                                np.linalg.norm(sampled[0] - target_start)
                            )
                            node_cost = 0.0
                        else:
                            endpoint_cost = float(
                                np.linalg.norm(sampled[0] - target_start)
                                + np.linalg.norm(sampled[-1] - target_end)
                            )
                            node_cost = float(
                                np.mean(np.linalg.norm(sampled - previous[run], axis=1))
                            )
                        length_cost = 0.025 * abs(scale - 1.0) * expected
                        # Use the old chain only as a weak material-index
                        # cue.  The current path still supplies every
                        # returned position; the node term prevents a path
                        # crossing from being selected solely because its
                        # endpoints happen to be close.
                        cost = endpoint_cost + 0.35 * node_cost + length_cost
                        if best is None or cost < best[0]:
                            best = (cost, sampled, start_arc, end_arc)
            if best is None or best[0] > 0.14:
                continue
            _, sampled, start_arc, end_arc = best
            # Do not let two disjoint visible runs consume the same current
            # path interval.  This is mainly relevant at 2-D crossings.
            overlap = any(
                max(start_arc, old_start) <= min(end_arc, old_end)
                for old_start, old_end in occupied_intervals
            )
            if overlap and len(runs) > 1:
                continue
            occupied_intervals.append((start_arc, end_arc))
            for index, observation in zip(run, sampled):
                observations[index] = observation
        return observations

    @staticmethod
    def _centerline_observation(
        selected: np.ndarray,
        anchor: np.ndarray,
        tangent: np.ndarray,
    ) -> np.ndarray:
        """Estimate a cable centre from points on its cylindrical surface.

        RGB-D returns visible surface points, so their median can sit on the
        near side of the cable.  A least-squares circle in the plane normal to
        the local tangent recovers the centre when the two views provide
        enough angular coverage; otherwise the robust 3-D median is safer.
        """
        selected = np.asarray(selected, dtype=np.float64)
        anchor = np.asarray(anchor, dtype=np.float64)
        tangent = np.asarray(tangent, dtype=np.float64)
        if len(selected) < 8 or not np.isfinite(selected).all():
            return np.median(selected, axis=0)
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(reference, tangent))) > 0.90:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(tangent, reference)
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 1e-9:
            return np.median(selected, axis=0)
        u /= u_norm
        v = np.cross(tangent, u)
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1e-9:
            return np.median(selected, axis=0)
        v /= v_norm
        offsets = selected - anchor
        coordinates = np.column_stack((offsets @ u, offsets @ v))
        # Quantise angular support into bins.  If the visible surface covers
        # only a small arc, a circle fit is under-constrained and the median
        # has lower bias/variance than an extrapolated centre.
        angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
        bins = np.floor((angles + np.pi) / (2.0 * np.pi) * 12.0).astype(np.int64)
        coverage = len(np.unique(np.clip(bins, 0, 11)))
        if coverage < 5:
            return np.median(selected, axis=0)
        design = np.column_stack((coordinates[:, 0], coordinates[:, 1], np.ones(len(coordinates))))
        rhs = -(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2)
        try:
            coefficients, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return np.median(selected, axis=0)
        centre_2d = -0.5 * coefficients[:2]
        radius = float(np.linalg.norm(centre_2d))
        radial = np.linalg.norm(coordinates - centre_2d[None, :], axis=1)
        fitted_radius = float(np.median(radial))
        # The rendered cable radius is about 4--10 mm.  Reject numerically
        # unstable fits that jump outside the local point neighbourhood.
        if (
            not np.isfinite(centre_2d).all()
            or radius > 0.030
            or fitted_radius < 0.001
            or fitted_radius > 0.030
            or float(np.median(np.abs(radial - fitted_radius))) > 0.012
        ):
            return np.median(selected, axis=0)
        return anchor + u * float(centre_2d[0]) + v * float(centre_2d[1])

    def _fuse_visible_observations_monotonic(
        self,
        previous: np.ndarray,
        candidate: np.ndarray,
        points: np.ndarray,
        visible: list[int] | np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Fuse current-cloud observations while preserving material order.

        The ordinary fusion keeps the previous chain as an arc-coordinate
        lookup.  During a fast bend this makes tangential motion look like a
        branch swap because all node windows are anchored to stale positions.
        This variant still uses those windows to reject unrelated cloud
        points, but estimates each supported node's arc displacement from the
        current observations and projects the displacements through a weighted
        isotonic (non-decreasing) fit.  Consequently the visible part can move
        along the cable in the current frame without ever reversing node order;
        nodes with no current support remain the native CPD/history result.
        """
        previous = np.asarray(previous, dtype=np.float64)
        candidate = np.asarray(candidate, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        if (
            previous.ndim != 2
            or candidate.shape != previous.shape
            or points.ndim != 2
            or previous.shape[1] != 3
            or len(points) == 0
            or not np.isfinite(previous).all()
            or not np.isfinite(candidate).all()
            or not np.isfinite(points).all()
        ):
            return candidate, 0
        visible_sorted = sorted(
            set(int(index) for index in np.asarray(visible).ravel() if 0 <= int(index) < len(previous))
        )
        if not visible_sorted or len(previous) < 2:
            return candidate, 0
        blend = float(np.clip(self.config.visible_observation_blend, 0.0, 1.0))
        radius = max(float(self.config.visible_observation_radius), 1e-4)
        min_points = max(int(self.config.visible_observation_min_points), 1)
        if blend <= 0.0:
            return candidate, 0

        # Use the finite CPD candidate as the geometric scaffold for the
        # current-frame arc projection.  It has already seen this frame's raw
        # cloud; using the stale previous chain here would reintroduce the
        # history-drift failure that this fusion mode is intended to remove.
        arc_chain = candidate
        vectors = np.diff(arc_chain, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        lengths_sq = lengths * lengths
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        starts = arc_chain[:-1]
        relative = points[:, None, :] - starts[None, :, :]
        fractions = np.divide(
            np.sum(relative * vectors[None, :, :], axis=2),
            lengths_sq[None, :],
            out=np.zeros((len(points), len(vectors)), dtype=np.float64),
            where=lengths_sq[None, :] > 1e-12,
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = starts[None, :, :] + fractions[:, :, None] * vectors[None, :, :]
        point_distances = np.linalg.norm(points[:, None, :] - projections, axis=2)
        segments = np.argmin(point_distances, axis=1)
        point_arc = cumulative[segments] + fractions[np.arange(len(points)), segments] * lengths[segments]

        raw_arc: list[float] = []
        observations: list[np.ndarray] = []
        node_indices: list[int] = []
        tangents: list[np.ndarray] = []
        weights: list[float] = []
        selected_sets: list[np.ndarray] = []
        for node_index in visible_sorted:
            if node_index == 0:
                tangent = arc_chain[1] - arc_chain[0]
            elif node_index + 1 == len(previous):
                tangent = arc_chain[-1] - arc_chain[-2]
            else:
                tangent = arc_chain[node_index + 1] - arc_chain[node_index - 1]
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1e-9:
                continue
            tangent = tangent / tangent_norm
            left_gap = cumulative[node_index] - cumulative[max(node_index - 1, 0)]
            right_gap = cumulative[min(node_index + 1, len(previous) - 1)] - cumulative[node_index]
            arc_window = max(0.022, 1.8 * max(left_gap, right_gap, 1e-6))
            arc_gate = np.abs(point_arc - cumulative[node_index]) <= arc_window
            distance_gate = np.minimum(
                np.linalg.norm(points - previous[node_index], axis=1),
                np.linalg.norm(points - candidate[node_index], axis=1),
            ) <= radius
            selected = points[arc_gate & distance_gate]
            if len(selected) < min_points:
                continue
            observation = np.median(selected, axis=0)
            residuals = np.linalg.norm(selected - observation, axis=1)
            robust_limit = max(0.006, 2.0 * float(np.median(residuals)))
            robust = selected[residuals <= robust_limit]
            if len(robust) >= min_points:
                observation = np.median(robust, axis=0)
                selected = robust
            # A current observation must measurably explain the point cloud
            # better than the CPD candidate; this prevents static RGB-D noise
            # from accumulating as an arc drift.
            candidate_residual = float(np.linalg.norm(points - candidate[node_index], axis=1).min())
            observation_residual = float(np.linalg.norm(points - observation, axis=1).min())
            if candidate_residual - observation_residual < float(
                max(0.0, self.config.visible_observation_residual_margin)
            ):
                continue
            raw_displacement = float(np.dot(observation - candidate[node_index], tangent))
            raw_arc.append(float(cumulative[node_index] + raw_displacement))
            observations.append(observation)
            node_indices.append(node_index)
            tangents.append(tangent)
            # More points and tighter spread should carry more weight in the
            # monotonic fit, while retaining a nonzero weight for every node.
            spread = float(np.median(np.linalg.norm(selected - observation, axis=1)))
            weights.append(float(max(1.0, len(selected)) / max(0.004, spread + 0.001)))
            selected_sets.append(selected)
        if not node_indices:
            return candidate, 0

        # Weighted pool-adjacent-violators algorithm.  This is a tiny 1-D
        # isotonic regression, avoiding a dependency on sklearn and preserving
        # monotonic material order through crossings.
        blocks: list[list[float | int]] = []
        for index, value in enumerate(raw_arc):
            weight = max(float(weights[index]), 1e-6)
            blocks.append([float(value), weight, index, index])
            while len(blocks) >= 2 and float(blocks[-2][0]) > float(blocks[-1][0]):
                left = blocks[-2]
                right = blocks[-1]
                total_weight = float(left[1]) + float(right[1])
                merged = [
                    (float(left[0]) * float(left[1]) + float(right[0]) * float(right[1]))
                    / max(total_weight, 1e-9),
                    total_weight,
                    int(left[2]),
                    int(right[3]),
                ]
                blocks[-2:] = [merged]
        fitted_arc = np.empty(len(raw_arc), dtype=np.float64)
        for block in blocks:
            fitted_arc[int(block[2]) : int(block[3]) + 1] = float(block[0])

        fused = candidate.copy()
        count = 0
        tangent_blend = float(np.clip(self.config.monotonic_tangent_blend, 0.0, 1.0))
        previous_steps = np.linalg.norm(np.diff(previous, axis=0), axis=1)
        max_tangent_shift = max(0.040, 3.0 * float(np.median(previous_steps)))
        for local, node_index in enumerate(node_indices):
            # Keep the complete current-frame observation (including rigid
            # translation and normal displacement).  Only the tangential
            # component is nudged toward the order-preserving isotonic arc;
            # rebuilding from ``previous`` here would accidentally discard the
            # valid CPD/global motion estimate.
            arc_adjustment = float(np.clip(
                fitted_arc[local] - raw_arc[local],
                -max_tangent_shift,
                max_tangent_shift,
            ))
            observation = observations[local] + tangent_blend * arc_adjustment * tangents[local]
            correction = observation - candidate[node_index]
            proposed = candidate[node_index] + blend * correction
            if node_index > 0:
                reference_gap = float(np.linalg.norm(candidate[node_index] - candidate[node_index - 1]))
                proposed_gap = float(np.linalg.norm(proposed - fused[node_index - 1]))
                if proposed_gap < max(0.006, 0.45 * reference_gap):
                    continue
            if node_index + 1 < len(candidate):
                reference_gap = float(np.linalg.norm(candidate[node_index + 1] - candidate[node_index]))
                proposed_gap = float(np.linalg.norm(candidate[node_index + 1] - proposed))
                if proposed_gap < max(0.006, 0.45 * reference_gap):
                    continue
            fused[node_index] = proposed
            count += 1
        return fused, count

    def _fuse_visible_observations(
        self,
        previous: np.ndarray,
        candidate: np.ndarray,
        points: np.ndarray,
        visible: list[int] | np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Fuse current-cloud observations into supported nodes only.

        Cloud samples are assigned to the previous chain by their topological
        arc coordinate.  This provides a lightweight crossing-safe gate: a
        nearby segment cannot immediately update the wrong node.  The native
        candidate remains unchanged for nodes without enough local support.
        """
        if not len(points) or not len(visible):
            return candidate, 0
        blend = float(np.clip(self.config.visible_observation_blend, 0.0, 1.0))
        radius = max(float(self.config.visible_observation_radius), 1e-4)
        min_points = max(int(self.config.visible_observation_min_points), 1)
        if blend <= 0.0:
            return candidate, 0

        def build_arc_data(topology: np.ndarray):
            starts = topology[:-1]
            vectors = np.diff(topology, axis=0)
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
            distances = np.linalg.norm(points[:, None, :] - projections, axis=2)
            segments = np.argmin(distances, axis=1)
            arc = cumulative[segments] + fractions[np.arange(len(points)), segments] * lengths[segments]
            return cumulative, lengths, arc

        topology_mode = str(self.config.visible_observation_topology)
        previous_data = build_arc_data(previous)
        candidate_data = (
            build_arc_data(candidate)
            if topology_mode in ("candidate", "adaptive", "history_guard")
            else None
        )
        if topology_mode == "history_guard":
            reference_length = max(float(self._reference_length or 0.0), 1e-6)
            previous_length = float(np.linalg.norm(np.diff(previous, axis=0), axis=1).sum())
            previous_steps = np.linalg.norm(np.diff(previous, axis=0), axis=1)
            history_bad = (
                previous_length / reference_length < 0.82
                or previous_length / reference_length > 1.22
                or (len(previous_steps) and float(np.max(previous_steps)) > 0.055)
            )
            topology_mode = "candidate" if history_bad else "previous"

        fused = candidate.copy()
        count = 0
        for node_index in sorted(set(int(i) for i in visible)):
            if node_index < 0 or node_index >= len(previous):
                continue
            data = (
                candidate_data
                if topology_mode == "candidate"
                else previous_data
            )
            if topology_mode == "adaptive":
                data = previous_data
            cumulative, lengths, arc = data
            s_node = cumulative[node_index]
            left_gap = cumulative[node_index] - cumulative[max(node_index - 1, 0)]
            right_gap = cumulative[min(node_index + 1, len(previous) - 1)] - cumulative[node_index]
            arc_window = max(
                float(self.config.visible_observation_arc_window_min),
                float(self.config.visible_observation_arc_window_scale)
                * max(left_gap, right_gap, 1e-6),
            )
            arc_gate = np.abs(arc - s_node) <= arc_window
            distance_to_reference = np.minimum(
                np.linalg.norm(points - previous[node_index], axis=1),
                np.linalg.norm(points - candidate[node_index], axis=1),
            )

            def select_radius(radius_value: float) -> tuple[np.ndarray, float]:
                selected_local = points[
                    arc_gate
                    & (distance_to_reference <= max(float(radius_value), 1e-4))
                ]
                if len(selected_local) < min_points:
                    return selected_local, float("inf")
                centre_local = np.median(selected_local, axis=0)
                spread_local = float(
                    np.median(np.linalg.norm(selected_local - centre_local, axis=1))
                )
                target = max(
                    int(self.config.visible_observation_support_target), min_points
                )
                support_penalty = 0.003 * max(
                    0.0, (target - len(selected_local)) / target
                )
                return selected_local, spread_local + support_penalty

            selected, selected_radius_score = select_radius(radius)
            selected_metric = None
            if self.config.visible_observation_adaptive_radius:
                expanded_radius = max(
                    radius,
                    float(self.config.visible_observation_fallback_radius),
                )
                expanded, expanded_score = select_radius(expanded_radius)
                if (
                    len(expanded) >= min_points
                    and expanded_score + 0.0002 < selected_radius_score
                ):
                    selected = expanded
                    selected_metric = expanded_score
            if topology_mode == "adaptive":
                candidate_cumulative, candidate_lengths, candidate_arc = candidate_data
                candidate_s_node = candidate_cumulative[node_index]
                candidate_left_gap = candidate_cumulative[node_index] - candidate_cumulative[max(node_index - 1, 0)]
                candidate_right_gap = candidate_cumulative[min(node_index + 1, len(previous) - 1)] - candidate_cumulative[node_index]
                candidate_window = max(
                    float(self.config.visible_observation_arc_window_min),
                    float(self.config.visible_observation_arc_window_scale)
                    * max(candidate_left_gap, candidate_right_gap, 1e-6),
                )
                candidate_arc_gate = np.abs(candidate_arc - candidate_s_node) <= candidate_window
                candidate_selected = points[
                    candidate_arc_gate
                    & (distance_to_reference <= radius)
                ]
                if len(candidate_selected) >= min_points:
                    candidate_observation = np.median(candidate_selected, axis=0)
                    candidate_spread = float(
                        np.median(np.linalg.norm(candidate_selected - candidate_observation, axis=1))
                    )
                    previous_spread = (
                        float(np.median(np.linalg.norm(selected - np.median(selected, axis=0), axis=1)))
                        if len(selected)
                        else float("inf")
                    )
                    # Prefer the candidate arc only when it explains the
                    # local cloud measurably better.  This keeps the crossing-
                    # safe previous topology as the tie breaker.
                    if len(selected) < min_points or candidate_spread + 0.001 < previous_spread:
                        selected = candidate_selected
                        selected_metric = candidate_spread
            if len(selected) < min_points:
                continue

            if node_index == 0:
                tangent = previous[1] - previous[0]
            elif node_index + 1 == len(previous):
                tangent = previous[-1] - previous[-2]
            else:
                tangent = previous[node_index + 1] - previous[node_index - 1]
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1e-9:
                continue
            tangent = tangent / tangent_norm
            observation = (
                self._centerline_observation(selected, previous[node_index], tangent)
                if self.config.visible_observation_centerline_fit
                else np.median(selected, axis=0)
            )
            distances_to_observation = np.linalg.norm(selected - observation, axis=1)
            robust_limit = max(0.006, 2.0 * float(np.median(distances_to_observation)))
            robust = selected[distances_to_observation <= robust_limit]
            if len(robust) >= min_points:
                observation = (
                    self._centerline_observation(robust, previous[node_index], tangent)
                    if self.config.visible_observation_centerline_fit
                    else np.median(robust, axis=0)
                )
            # A nearly stationary node is dominated by RGB-D surface noise;
            # repeatedly applying that noise would slowly stretch the chain.
            # Only use the observation when it indicates a real local motion.
            observation_motion = observation - previous[node_index]
            if self.config.visible_observation_normal_only:
                tangent_component = tangent * float(np.dot(observation_motion, tangent))
                tangent_fraction = float(
                    np.clip(self.config.visible_observation_tangent_blend, 0.0, 1.0)
                )
                observation_motion = observation_motion - (1.0 - tangent_fraction) * tangent_component
            if np.linalg.norm(observation_motion) < float(
                self.config.visible_observation_motion_threshold
            ):
                continue
            # Do not replace a CPD node merely because a local cloud exists.
            # The observation must reduce its distance to the current cloud by
            # a meaningful margin; otherwise static scenes would accumulate
            # surface/centroid bias from frame to frame.
            candidate_residual = float(
                np.linalg.norm(points - candidate[node_index], axis=1).min()
            )
            observation_residual = float(np.linalg.norm(points - observation, axis=1).min())
            if candidate_residual - observation_residual < float(
                max(0.0, self.config.visible_observation_residual_margin)
            ):
                continue
            # By default keep the CPD-provided tangential/arc-length
            # coordinate and use the cloud mainly to correct the normal
            # direction.  An explicit ablation can allow full displacement.
            correction = observation - candidate[node_index]
            if self.config.visible_observation_normal_only:
                tangent_component = tangent * float(np.dot(correction, tangent))
                tangent_fraction = float(
                    np.clip(self.config.visible_observation_tangent_blend, 0.0, 1.0)
                )
                correction = correction - (1.0 - tangent_fraction) * tangent_component
            proposed = candidate[node_index] + blend * correction
            if node_index > 0:
                reference_gap = np.linalg.norm(
                    candidate[node_index] - candidate[node_index - 1]
                )
                proposed_gap = np.linalg.norm(proposed - fused[node_index - 1])
                if proposed_gap < max(0.006, 0.45 * reference_gap):
                    continue
            if node_index + 1 < len(candidate):
                reference_gap = np.linalg.norm(
                    candidate[node_index + 1] - candidate[node_index]
                )
                proposed_gap = np.linalg.norm(candidate[node_index + 1] - proposed)
                if proposed_gap < max(0.006, 0.45 * reference_gap):
                    continue
            fused[node_index] = proposed
            count += 1
        return fused, count

    @staticmethod
    def _rigid_fit_residual(source: np.ndarray, target: np.ndarray) -> float:
        """Return RMS residual of the best 3-D rigid fit source -> target."""
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.shape != target.shape or source.ndim != 2 or source.shape[0] < 3:
            return float("inf")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            return float("inf")
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        covariance = (source - source_center).T @ (target - target_center)
        left, _, right_t = np.linalg.svd(covariance)
        rotation = right_t.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right_t[-1, :] *= -1.0
            rotation = right_t.T @ left.T
        prediction = (source - source_center) @ rotation.T + target_center
        return float(np.sqrt(np.mean(np.sum((prediction - target) ** 2, axis=1))))

    def _visible_pixel_snap_observations(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        reference: np.ndarray,
        visible: list[int] | np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Estimate visible nodes directly from this frame's mask/depth.

        The projected reference node is only a local selector.  The returned
        point is measured from the current skeleton pixel and its depth patch,
        with an optional cable-radius correction toward the centreline.  No
        previous-frame point is blended into the observation itself.  A local
        selector is deliberately used instead of a whole-path index mapping:
        at a projected crossing the candidate depth and pixel neighbourhood
        choose the branch without forcing the entire material chain to follow
        the old order.
        """
        mask = np.asarray(mask)
        depth_m = np.asarray(depth_m, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        if (
            mask.ndim != 2
            or depth_m.shape != mask.shape
            or reference.ndim != 2
            or reference.shape[1] != 3
            or not np.isfinite(reference).all()
            or not len(visible)
        ):
            return {}
        try:
            pixels = np.asarray(ordered_skeleton_pixels_simple(mask), dtype=np.int32)
        except Exception:
            return {}
        height, width = mask.shape[:2]
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < height)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < width)
        )
        pixels = pixels[valid]
        if len(pixels) < 4:
            return {}
        path_points: list[np.ndarray] = []
        path_pixels: list[tuple[float, float]] = []
        for row, col in pixels:
            value = float(depth_m[int(row), int(col)])
            if not np.isfinite(value) or value <= 0.0:
                patch = depth_m[
                    max(0, int(row) - 2) : min(height, int(row) + 3),
                    max(0, int(col) - 2) : min(width, int(col) + 3),
                ]
                finite = patch[np.isfinite(patch) & (patch > 0.0)]
                if len(finite):
                    value = float(np.median(finite))
            if not np.isfinite(value) or value <= 0.0:
                continue
            path_points.append(
                np.asarray(
                    [
                        (float(col) - self.intrinsics[0, 2]) * value / self.intrinsics[0, 0],
                        (float(row) - self.intrinsics[1, 2]) * value / self.intrinsics[1, 1],
                        value,
                    ],
                    dtype=np.float64,
                )
            )
            path_pixels.append((float(col), float(row)))
        if len(path_points) < 4:
            return {}
        path_points_array = np.asarray(path_points, dtype=np.float64)
        path_pixels_array = np.asarray(path_pixels, dtype=np.float64)
        projected = project_camera_points(reference, self.intrinsics)
        radius = max(float(self.config.visible_pixel_snap_radius_px), 2.0)
        radius_sq = radius * radius
        offset = max(float(self.config.visible_pixel_snap_center_offset_m), 0.0)
        observations: dict[int, np.ndarray] = {}
        for node_index in sorted(
            set(int(i) for i in np.asarray(visible).ravel() if 0 <= int(i) < len(reference))
        ):
            uv = projected[node_index]
            if not np.isfinite(uv).all():
                continue
            pixel_delta = path_pixels_array - uv[None, :]
            pixel_dist_sq = np.sum(pixel_delta * pixel_delta, axis=1)
            candidates = np.flatnonzero(pixel_dist_sq <= radius_sq)
            if not len(candidates):
                continue
            # Select a short ordered neighbourhood around the closest pixel;
            # this suppresses one-pixel depth spikes while avoiding a branch
            # average across a crossing.
            closest = int(candidates[np.argmin(pixel_dist_sq[candidates])])
            local_start = max(0, closest - 3)
            local_end = min(len(path_points_array), closest + 4)
            local = path_points_array[local_start:local_end]
            if len(local) < 2:
                continue
            observation = np.median(local, axis=0)
            if not np.isfinite(observation).all() or observation[2] <= 1e-6:
                continue
            if offset > 0.0:
                norm = float(np.linalg.norm(observation))
                if norm > 1e-6:
                    observation = observation + offset * observation / norm
            # A depth discontinuity at a crossing can put the selected pixel
            # on a far branch even when its 2-D distance is small.  Keep a
            # loose 3-D gate so the pixel snap remains a local correction.
            if float(np.linalg.norm(observation - reference[node_index])) > 0.04:
                continue
            observations[node_index] = observation
        return observations

    def _complete_hidden_piecewise(
        self,
        previous: np.ndarray,
        candidate: np.ndarray,
        visible: list[int] | np.ndarray,
    ) -> np.ndarray:
        """Propagate current visible-anchor displacement through hidden gaps.

        The hidden geometry is kept in the previous material coordinate only
        as a local shape template.  Displacements measured at neighbouring
        visible nodes are linearly interpolated along that template's arc,
        which is the lowest-assumption completion for a deforming cable.  A
        one-sided hidden tail uses its nearest visible anchor displacement.
        """
        previous = np.asarray(previous, dtype=np.float64)
        candidate = np.asarray(candidate, dtype=np.float64)
        if previous.shape != candidate.shape or previous.ndim != 2 or len(previous) < 2:
            return candidate
        visible_sorted = sorted(
            set(int(index) for index in np.asarray(visible).ravel() if 0 <= int(index) < len(previous))
        )
        if len(visible_sorted) < 2:
            return candidate
        blend = float(np.clip(self.config.hidden_piecewise_blend, 0.0, 1.0))
        if blend <= 0.0:
            return candidate
        if not np.isfinite(previous).all() or not np.isfinite(candidate).all():
            return candidate
        cumulative = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(previous, axis=0), axis=1)))
        )
        completed = candidate.copy()

        def fill_gap(left: int, right: int) -> None:
            if right - left <= 1 or right - left - 1 > max(
                int(self.config.hidden_piecewise_max_gap_nodes), 0
            ):
                return
            left_delta = candidate[left] - previous[left]
            right_delta = candidate[right] - previous[right]
            if not np.isfinite(left_delta).all() or not np.isfinite(right_delta).all():
                return
            if float(np.linalg.norm(left_delta - right_delta)) > max(
                float(self.config.hidden_piecewise_max_anchor_disagreement), 0.0
            ):
                return
            span = max(float(cumulative[right] - cumulative[left]), 1e-9)
            for index in range(left + 1, right):
                fraction = float((cumulative[index] - cumulative[left]) / span)
                transported = previous[index] + (1.0 - fraction) * left_delta + fraction * right_delta
                completed[index] = (
                    (1.0 - blend) * completed[index] + blend * transported
                )

        for left, right in zip(visible_sorted[:-1], visible_sorted[1:]):
            fill_gap(left, right)
        # Hidden prefixes/suffixes have one anchor.  Use its measured
        # displacement rather than extrapolating the CPD tangent.
        first = visible_sorted[0]
        if first > 0:
            delta = candidate[first] - previous[first]
            if np.isfinite(delta).all() and float(np.linalg.norm(delta)) <= max(
                float(self.config.hidden_piecewise_max_anchor_disagreement), 0.0
            ):
                completed[:first] = (1.0 - blend) * completed[:first] + blend * (
                    previous[:first] + delta
                )
        last = visible_sorted[-1]
        if last + 1 < len(previous):
            delta = candidate[last] - previous[last]
            if np.isfinite(delta).all() and float(np.linalg.norm(delta)) <= max(
                float(self.config.hidden_piecewise_max_anchor_disagreement), 0.0
            ):
                completed[last + 1 :] = (1.0 - blend) * completed[last + 1 :] + blend * (
                    previous[last + 1 :] + delta
                )
        return completed

    def _complete_hidden_current_anchor_spline(
        self,
        previous: np.ndarray,
        seed: np.ndarray,
        visible: list[int] | np.ndarray,
    ) -> np.ndarray:
        """Build a current-frame-first seed between measured visible anchors.

        ``seed`` contains current RGB-D positions at supported nodes and the
        old positions only at unsupported nodes.  This routine replaces the
        latter with cubic-Hermite interpolation in the *current anchor
        geometry*.  The previous chain contributes only the integer material
        indices and the local tangent scale; it is never copied into the
        hidden shape.  The native CPD call can subsequently refine the whole
        seed against this frame's point cloud.
        """

        previous = np.asarray(previous, dtype=np.float64)
        seed = np.asarray(seed, dtype=np.float64)
        if previous.shape != seed.shape or previous.ndim != 2 or len(previous) < 3:
            return seed
        visible_sorted = sorted(
            set(int(index) for index in np.asarray(visible).ravel() if 0 <= int(index) < len(seed))
        )
        if len(visible_sorted) < 2:
            return seed
        if not np.isfinite(seed[visible_sorted]).all():
            return seed
        completed = seed.copy()
        # Interior gaps.  Hermite tangents are estimated from the adjacent
        # current anchors; when a side has no second anchor, use the chord.
        for position, (left, right) in enumerate(zip(visible_sorted[:-1], visible_sorted[1:])):
            if right - left <= 1:
                continue
            chord = seed[right] - seed[left]
            if position > 0:
                prior = visible_sorted[position - 1]
                tangent_left = (seed[left] - seed[prior]) / max(float(left - prior), 1.0)
            else:
                tangent_left = chord / max(float(right - left), 1.0)
            if position + 2 < len(visible_sorted):
                following = visible_sorted[position + 2]
                tangent_right = (seed[following] - seed[right]) / max(
                    float(following - right), 1.0
                )
            else:
                tangent_right = chord / max(float(right - left), 1.0)
            span = float(right - left)
            m0 = tangent_left * span
            m1 = tangent_right * span
            for index in range(left + 1, right):
                t = float(index - left) / span
                h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
                h10 = t**3 - 2.0 * t**2 + t
                h01 = -2.0 * t**3 + 3.0 * t**2
                h11 = t**3 - t**2
                value = h00 * seed[left] + h10 * m0 + h01 * seed[right] + h11 * m1
                # A noisy tangent should not overshoot far outside the two
                # current anchors; clipping is a geometric safety gate, not
                # a previous-frame pose constraint.
                lo = np.minimum(seed[left], seed[right]) - 0.04
                hi = np.maximum(seed[left], seed[right]) + 0.04
                completed[index] = np.clip(value, lo, hi)
        # One-sided gaps use the nearest current anchor tangent.  The
        # endpoint extrapolation is capped to 8 cm per node to avoid a single
        # bad depth sample producing an implausible seed.
        first = visible_sorted[0]
        if first > 0:
            if len(visible_sorted) > 1:
                second = visible_sorted[1]
                tangent = (seed[second] - seed[first]) / max(float(second - first), 1.0)
            else:
                tangent = np.zeros(3, dtype=np.float64)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > 0.08:
                tangent *= 0.08 / tangent_norm
            for index in range(first - 1, -1, -1):
                completed[index] = completed[index + 1] - tangent
        last = visible_sorted[-1]
        if last + 1 < len(seed):
            if len(visible_sorted) > 1:
                prior = visible_sorted[-2]
                tangent = (seed[last] - seed[prior]) / max(float(last - prior), 1.0)
            else:
                tangent = np.zeros(3, dtype=np.float64)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > 0.08:
                tangent *= 0.08 / tangent_norm
            for index in range(last + 1, len(seed)):
                completed[index] = completed[index - 1] + tangent
        return completed

    def _complete_hidden_from_history(
        self,
        previous: np.ndarray,
        candidate: np.ndarray,
        visible: list[int] | np.ndarray,
    ) -> np.ndarray:
        """Transport the previous hidden shape between current visible anchors.

        The native CPD estimate is often under-constrained in a long occluded
        interval.  For each such interval, preserve the previous local shape
        while interpolating the endpoint displacement measured on the current
        visible cable.  This keeps node order and arc-length continuity while
        allowing the hidden portion to follow the observed motion.  End gaps
        use the displacement of their nearest visible anchor.
        """
        visible_sorted = sorted(set(int(i) for i in visible if 0 <= int(i) < len(previous)))
        if len(visible_sorted) < 2:
            return candidate
        blend = float(np.clip(self.config.hidden_history_blend, 0.0, 1.0))
        if blend <= 0.0:
            return candidate

        # Estimate whether the visible motion is approximately rigid.  This
        # is deliberately computed from the current visible CPD/fused nodes,
        # never from ground truth.  If the residual is high, the cable is
        # deforming and blindly transporting the previous hidden shape would
        # re-introduce exactly the history bias this fusion is meant to avoid.
        visible_array = np.asarray(visible_sorted, dtype=np.int64)
        source = np.asarray(previous[visible_array], dtype=np.float64)
        target = np.asarray(candidate[visible_array], dtype=np.float64)
        rigid_prediction = None
        if len(source) >= 3:
            source_center = source.mean(axis=0)
            target_center = target.mean(axis=0)
            source_zero = source - source_center
            target_zero = target - target_center
            covariance = source_zero.T @ target_zero
            try:
                left, _, right_t = np.linalg.svd(covariance)
                rotation = right_t.T @ left.T
                if np.linalg.det(rotation) < 0.0:
                    right_t[-1, :] *= -1.0
                    rotation = right_t.T @ left.T
                rigid_target = source_zero @ rotation.T + target_center
                rigid_residual = float(
                    np.sqrt(np.mean(np.sum((rigid_target - target) ** 2, axis=1)))
                )
                rigid_prediction = (
                    (np.asarray(previous, dtype=np.float64) - source_center) @ rotation.T
                    + target_center
                )
            except np.linalg.LinAlgError:
                rigid_residual = float("inf")
        else:
            # With only two visible anchors, use the disagreement between
            # their displacement vectors as a conservative deformation test.
            displacement = target - source
            rigid_residual = float(np.linalg.norm(displacement[0] - displacement[-1]))
        if rigid_residual > max(
            float(self.config.hidden_history_deformation_threshold), 0.0
        ):
            return candidate

        # For a near-rigid visible motion, apply the same SE(3)-like fit to
        # hidden nodes.  Interpolating endpoint displacements is not rotation
        # invariant and can bend a hidden tail even when the visible cable
        # simply rotated around the gripper.  With only two anchors there is
        # no stable 3-D rotation estimate, so retain the older displacement
        # interpolation below.
        if rigid_prediction is not None:
            completed = candidate.copy()
            hidden_indices = np.asarray(
                sorted(set(range(len(previous))) - set(visible_sorted)),
                dtype=np.int64,
            )
            if len(hidden_indices):
                completed[hidden_indices] = (
                    (1.0 - blend) * completed[hidden_indices]
                    + blend * rigid_prediction[hidden_indices]
                )
            return completed
        cumulative = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(previous, axis=0), axis=1)))
        )
        completed = candidate.copy()
        # Interior occlusion gaps: interpolate endpoint displacement along
        # the previous arc coordinate and preserve the previous local shape.
        for left, right in zip(visible_sorted[:-1], visible_sorted[1:]):
            if right - left <= 1:
                continue
            span = max(float(cumulative[right] - cumulative[left]), 1e-9)
            left_delta = candidate[left] - previous[left]
            right_delta = candidate[right] - previous[right]
            for index in range(left + 1, right):
                fraction = float((cumulative[index] - cumulative[left]) / span)
                transported = previous[index] + (1.0 - fraction) * left_delta + fraction * right_delta
                completed[index] = (1.0 - blend) * completed[index] + blend * transported
        # Head and tail gaps have one anchor; use a rigid translation of the
        # previous shape rather than extrapolating a potentially wrong CPD
        # tangent beyond the visible image.
        first = visible_sorted[0]
        if first > 0:
            delta = candidate[first] - previous[first]
            for index in range(first):
                transported = previous[index] + delta
                completed[index] = (1.0 - blend) * completed[index] + blend * transported
        last = visible_sorted[-1]
        if last + 1 < len(previous):
            delta = candidate[last] - previous[last]
            for index in range(last + 1, len(previous)):
                transported = previous[index] + delta
                completed[index] = (1.0 - blend) * completed[index] + blend * transported
        return completed

    def _visibility(
        self,
        points: np.ndarray,
        image_shape: tuple[int, int],
        mask: np.ndarray | None = None,
        depth_m: np.ndarray | None = None,
        nodes_override: np.ndarray | None = None,
    ):
        nodes = self._nodes if nodes_override is None else np.asarray(nodes_override, dtype=np.float64)
        if not len(points):
            return [], [], list(range(len(nodes)))
        distances = np.linalg.norm(nodes[:, None, :] - points[None, :, :], axis=2)
        shortest = distances.min(axis=1)
        neighborhood_counts = np.sum(
            distances <= max(
                self.config.visibility_neighborhood_radius,
                self.config.visibility_threshold,
            ),
            axis=1,
        )
        pixels = project_camera_points(nodes, self.intrinsics)
        height, width = image_shape
        projected_edges = np.zeros((height, width), dtype=np.uint8)
        mask_support = np.zeros(len(nodes), dtype=bool)
        if self.config.visibility_mode in ("neighborhood", "mask", "depth") and mask is not None:
            binary_mask = np.asarray(mask) > 0
            radius = max(int(self.config.visibility_mask_radius_px), 0)
            for node_index, pixel in enumerate(pixels):
                if not np.isfinite(pixel).all():
                    continue
                col, row = np.rint(pixel).astype(int)
                if not (0 <= row < height and 0 <= col < width):
                    continue
                row_min, row_max = max(0, row - radius), min(height, row + radius + 1)
                col_min, col_max = max(0, col - radius), min(width, col + radius + 1)
                mask_support[node_index] = (
                    int(binary_mask[row_min:row_max, col_min:col_max].sum())
                    >= int(self.config.visibility_min_mask_pixels)
                )
        depth_support = np.zeros(len(nodes), dtype=bool)
        if self.config.visibility_mode == "depth" and depth_m is not None:
            depth_image = np.asarray(depth_m, dtype=np.float64)
            for node_index, pixel in enumerate(pixels):
                if not np.isfinite(pixel).all():
                    continue
                col, row = np.rint(pixel).astype(int)
                if not (0 <= row < height and 0 <= col < width):
                    continue
                row_min, row_max = max(0, row - 2), min(height, row + 3)
                col_min, col_max = max(0, col - 2), min(width, col + 3)
                patch = depth_image[row_min:row_max, col_min:col_max]
                valid = patch[np.isfinite(patch) & (patch > 0.0)]
                if len(valid):
                    # The rendered cable depth is a surface depth, while a
                    # tracked node represents the cable centreline.  Allow a
                    # small radius-sized discrepancy, but reject pixels whose
                    # depth is clearly occupied by the robot/background.
                    observed_depth = float(np.median(valid))
                    node_depth = float(nodes[node_index, 2])
                    depth_support[node_index] = abs(observed_depth - node_depth) <= 0.030
        edge_indices = list(range(len(nodes) - 1))
        edge_indices.sort(key=lambda i: np.linalg.norm((nodes[i] + nodes[i + 1]) * 0.5))
        visible = set()
        not_self_occluded = set()
        for index in edge_indices:
            endpoints = []
            for node_index in (index, index + 1):
                pixel = pixels[node_index]
                in_frame = np.isfinite(pixel).all()
                if in_frame:
                    col, row = np.rint(pixel).astype(int)
                    in_frame = 0 <= row < height and 0 <= col < width
                endpoints.append((in_frame, col if in_frame else -1, row if in_frame else -1))
                if in_frame:
                    unoccluded = projected_edges[row, col] == 0
                    if unoccluded:
                        not_self_occluded.add(node_index)
                    strict_support = shortest[node_index] <= self.config.visibility_threshold
                    if self.config.visibility_mode == "neighborhood":
                        # A small cluster is more reliable than one lucky
                        # nearest point after voxel quantisation.  The weak
                        # threshold is deliberately looser than the strict
                        # 8 mm gate but remains local to the cable cloud.
                        support = (
                            shortest[node_index] <= self.config.visibility_weak_threshold
                            or neighborhood_counts[node_index]
                            >= self.config.visibility_min_neighbors
                            or mask_support[node_index]
                        )
                        # If RGB clearly contains cable at this projection but
                        # the corresponding depth is missing, allow it as a
                        # weak guide only when the projected node is not also
                        # flagged by the self-occlusion heuristic.  The mask is
                        # a depth-gap fallback, not an override for a crossing.
                        if support and unoccluded:
                            visible.add(node_index)
                    elif self.config.visibility_mode == "mask":
                        # RGB mask support is deliberately independent of the
                        # stale 3-D distance gate.  This mode is useful when
                        # the previous state lags behind a deforming cable.
                        if unoccluded and mask_support[node_index]:
                            visible.add(node_index)
                    elif self.config.visibility_mode == "mask_all":
                        if mask_support[node_index]:
                            visible.add(node_index)
                    elif self.config.visibility_mode == "depth":
                        if mask_support[node_index] and depth_support[node_index]:
                            visible.add(node_index)
                    elif unoccluded and strict_support:
                        visible.add(node_index)
            if endpoints[0][0] and endpoints[1][0]:
                cv2.line(
                    projected_edges,
                    (endpoints[0][1], endpoints[0][2]),
                    (endpoints[1][1], endpoints[1][2]),
                    255,
                    self.config.dlo_pixel_width,
                )
        visible = sorted(visible)
        extended = []
        if visible:
            for left, right in zip(visible[:-1], visible[1:]):
                extended.append(left)
                if self._geodesic[right] - self._geodesic[left] <= self.config.d_vis:
                    extended.extend(range(left + 1, right))
            extended.append(visible[-1])
        elif self.config.force_sparse_update:
            # When no node is within the 8 mm support radius, retain the
            # in-frame/topologically visible indices as weak guides.  This
            # allows the optimizer to attempt an update instead of immediately
            # returning the previous state.  If this set is empty, the C++
            # traversal still cannot run and the caller will report a failure.
            extended = sorted(not_self_occluded)
        self_occluded = sorted(set(range(len(nodes))) - not_self_occluded)
        return visible, sorted(set(extended)), self_occluded
