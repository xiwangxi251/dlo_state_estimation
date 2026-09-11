from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trackdlo_standalone.geometry import (  # noqa: E402
    backproject_mask,
    project_camera_points,
    resample_polyline,
)
from trackdlo_standalone.initialization import _ordered_crossing_trail, initialize_nodes  # noqa: E402
from trackdlo_standalone.metrics import frame_metrics  # noqa: E402


class StandaloneTrackDLOTest(unittest.TestCase):
    def setUp(self) -> None:
        self.k = np.array([[100.0, 0.0, 49.5], [0.0, 100.0, 39.5], [0.0, 0.0, 1.0]])

    def test_projection_and_backprojection_round_trip(self) -> None:
        depth = np.zeros((80, 100), dtype=np.uint16)
        mask = np.zeros_like(depth, dtype=np.uint8)
        depth[30, 60] = 1500
        mask[30, 60] = 255
        point = backproject_mask(depth.astype(np.float64) * 0.001, mask, self.k)
        np.testing.assert_allclose(project_camera_points(point, self.k)[0], [60.0, 30.0])
        self.assertAlmostEqual(point[0, 2], 1.5)

    def test_metrics_are_direction_invariant(self) -> None:
        curve = np.array([[0.0, 0.0, 1.0], [0.4, 0.2, 1.0], [1.0, 0.0, 1.0]])
        values = frame_metrics(curve[::-1], curve)
        self.assertLess(values["frame_error_m"], 1e-12)
        self.assertLess(values["ordered_error_m"], 1e-12)
        self.assertLess(values["endpoint_error_m"], 1e-12)

    def test_rgbd_initialization_returns_ordered_3d_nodes(self) -> None:
        hsv = np.zeros((80, 100, 3), dtype=np.uint8)
        hsv[:, :, 2] = 20
        cv2.line(hsv, (10, 55), (88, 25), (110, 220, 220), 7)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        depth = np.full((80, 100), 1200, dtype=np.uint16)
        nodes, mask = initialize_nodes(rgb, depth, self.k, 25, (90, 90, 80), (130, 255, 255))
        self.assertEqual(nodes.shape, (25, 3))
        self.assertGreater(np.count_nonzero(mask), 100)
        np.testing.assert_allclose(nodes[:, 2], 1.2, atol=1e-6)
        self.assertGreater(np.linalg.norm(nodes[-1] - nodes[0]), 0.7)

    def test_crossing_trail_keeps_all_graph_edges(self) -> None:
        # endpoint -> crossing -> loop -> crossing -> endpoint
        endpoint_a, crossing = (-2, 0), (0, 0)
        loop_a, loop_b, endpoint_b = (0, 2), (2, 2), (2, 0)
        edges = [
            (endpoint_a, crossing),
            (crossing, loop_a),
            (loop_a, loop_b),
            (loop_b, crossing),
            (crossing, endpoint_b),
        ]
        graph = defaultdict(list)
        for first, second in edges:
            graph[first].append((second, 1.0))
            graph[second].append((first, 1.0))
        trail = _ordered_crossing_trail(graph)
        self.assertIsNotNone(trail)
        self.assertEqual(len(trail), len(edges) + 1)
        self.assertEqual(sum(np.all(trail == crossing, axis=1)), 2)

    def test_compiled_core_smoke(self) -> None:
        try:
            from trackdlo_standalone import TrackDLOConfig
            from trackdlo_standalone._core import TrackDLOCore
        except ImportError as exc:
            self.skipTest(f"compiled extension not built: {exc}")
        config = TrackDLOConfig(num_nodes=12, max_iter=2)
        core = TrackDLOCore(
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
        nodes = np.column_stack((np.linspace(-0.2, 0.2, 12), np.zeros(12), np.ones(12)))
        core.initialize_nodes(nodes)
        core.initialize_geodesic_coord(np.linspace(0.0, 0.4, 12).tolist())
        observed = nodes + np.array([0.002, 0.0, 0.0])
        projection = np.column_stack((self.k, np.zeros(3)))
        core.tracking_step(observed, list(range(12)), list(range(12)), projection, 80, 100)
        result = np.asarray(core.get_tracking_result())
        self.assertEqual(result.shape, nodes.shape)
        self.assertTrue(np.isfinite(result).all())

    def test_resample_polyline_has_requested_count(self) -> None:
        points = np.array([[0.0, 0.0], [0.2, 0.0], [1.0, 0.0]])
        sampled = resample_polyline(points, 6)
        self.assertEqual(sampled.shape, (6, 2))
        np.testing.assert_allclose(sampled[:, 0], np.linspace(0.0, 1.0, 6))


if __name__ == "__main__":
    unittest.main()
