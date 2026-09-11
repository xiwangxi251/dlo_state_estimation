from __future__ import annotations

import unittest

import cv2
import numpy as np

from dlo_position import DLOPositionEstimator


class EstimatorTests(unittest.TestCase):
    def test_blue_curve_produces_requested_3d_points(self) -> None:
        height, width = 180, 240
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        curve = np.array(
            [[20, 100], [60, 70], [110, 90], [170, 55], [220, 80]],
            dtype=np.int32,
        )
        # RGB blue falls inside the estimator's default HSV range.
        cv2.polylines(rgb, [curve], False, (0, 0, 255), 8, cv2.LINE_AA)
        depth = np.full((height, width), 1.2, dtype=np.float32)
        intrinsics = np.array(
            [[200.0, 0.0, width / 2], [0.0, 200.0, height / 2], [0.0, 0.0, 1.0]]
        )
        estimate = DLOPositionEstimator(
            intrinsics, surface_to_center_offset_m=0.0
        ).estimate(rgb, depth)
        self.assertEqual(estimate.points_camera.shape, (14, 3))
        self.assertTrue(np.isfinite(estimate.points_camera).all())
        np.testing.assert_allclose(estimate.points_camera[:, 2], 1.2, atol=1e-6)

    def test_surface_correction_modes_produce_finite_points(self) -> None:
        height, width = 120, 160
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        curve = np.array([[15, 85], [55, 45], [105, 75], [145, 35]], dtype=np.int32)
        cv2.polylines(rgb, [curve], False, (0, 0, 255), 8, cv2.LINE_AA)
        depth = np.full((height, width), 1.0, dtype=np.float32)
        intrinsics = np.array(
            [[150.0, 0.0, width / 2], [0.0, 150.0, height / 2], [0.0, 0.0, 1.0]]
        )
        for mode in ("ray", "normal", "adaptive"):
            estimate = DLOPositionEstimator(
                intrinsics,
                surface_to_center_offset_m=0.02,
                surface_to_center_mode=mode,
            ).estimate(rgb, depth)
            self.assertEqual(estimate.points_camera.shape, (14, 3))
            self.assertTrue(np.isfinite(estimate.points_camera).all())


if __name__ == "__main__":
    unittest.main()
