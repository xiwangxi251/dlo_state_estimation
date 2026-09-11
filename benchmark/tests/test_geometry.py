from __future__ import annotations

import unittest

import numpy as np

from dlo_position.geometry import resample_polyline, reversal_invariant_errors


class GeometryTests(unittest.TestCase):
    def test_resample_polyline_is_uniform(self) -> None:
        source = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sampled = resample_polyline(source, 6)
        np.testing.assert_allclose(sampled[:, 0], np.linspace(0.0, 1.0, 6))

    def test_error_metric_accepts_endpoint_reversal(self) -> None:
        target = np.column_stack((np.linspace(0.0, 1.0, 14), np.zeros((14, 2))))
        errors, reversed_order = reversal_invariant_errors(target[::-1], target)
        self.assertTrue(reversed_order)
        self.assertLess(float(np.max(errors)), 1e-12)


if __name__ == "__main__":
    unittest.main()
