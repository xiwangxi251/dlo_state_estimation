from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "legacy_offline_bridge"
sys.path.insert(0, str(SCRIPTS))

from trackdlo_offline_common import (  # noqa: E402
    camera_matrix,
    project_camera_points,
    world_from_ros_optical,
)
from visualize_trackdlo_results import centerline_metrics  # noqa: E402


class TrackDloOfflineTest(unittest.TestCase):
    def test_camera_matrix_uses_mujoco_vertical_fov(self) -> None:
        matrix = camera_matrix(width=640, height=480, fovy_degrees=90.0)
        self.assertAlmostEqual(matrix[0, 0], 240.0)
        self.assertAlmostEqual(matrix[1, 1], 240.0)
        self.assertAlmostEqual(matrix[0, 2], 319.5)
        self.assertAlmostEqual(matrix[1, 2], 239.5)

    def test_mujoco_camera_axis_conversion(self) -> None:
        transform = world_from_ros_optical(np.zeros(3), np.eye(3))
        np.testing.assert_allclose(transform[:3, :3], np.diag([1.0, -1.0, -1.0]))

    def test_projection(self) -> None:
        matrix = camera_matrix(width=640, height=480, fovy_degrees=90.0)
        pixels = project_camera_points(np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]), matrix)
        np.testing.assert_allclose(pixels[0], [319.5, 239.5])
        np.testing.assert_allclose(pixels[1], [439.5, 239.5])

    def test_centerline_metric_is_direction_invariant(self) -> None:
        points = np.array([[0.0, 0.0, 1.0], [0.5, 0.1, 1.0], [1.0, 0.0, 1.0]])
        ordered, chamfer = centerline_metrics(points[::-1], points)
        self.assertLess(ordered, 1e-12)
        self.assertLess(chamfer, 1e-12)


if __name__ == "__main__":
    unittest.main()
