from __future__ import annotations

import unittest

import numpy as np

from dlo_position.temporal_benchmark import _order_inversion_fraction, _polyline_has_crossing


class TemporalMetricTests(unittest.TestCase):
    def test_projected_crossing_is_detected(self) -> None:
        crossing = np.array(
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=float
        )
        self.assertTrue(_polyline_has_crossing(crossing))

    def test_monotone_order_has_no_inversion(self) -> None:
        target = np.column_stack([np.linspace(0.0, 1.0, 14), np.zeros(14), np.zeros(14)])
        self.assertEqual(_order_inversion_fraction(target, target), 0.0)
        self.assertEqual(_order_inversion_fraction(target[::-1], target), 0.0)


if __name__ == "__main__":
    unittest.main()
