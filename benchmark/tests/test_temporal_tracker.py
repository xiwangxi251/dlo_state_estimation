from __future__ import annotations

import unittest

import numpy as np

from dlo_position.temporal_tracker import TemporalDLOTracker
from dlo_position.hypothesis_tracker import MultiHypothesisDLOTracker


class TemporalTrackerTests(unittest.TestCase):
    def test_partial_observation_keeps_full_state(self) -> None:
        full = np.column_stack(
            [np.linspace(0.0, 0.8, 14), np.zeros(14), np.zeros(14)]
        )
        tracker = TemporalDLOTracker(expected_length_m=0.8)
        first = tracker.update(full, observed_length_m=0.8)
        self.assertTrue(first.initialized)
        partial = full[4:11]
        result = tracker.update(partial, observed_length_m=0.8 * 6 / 13)
        self.assertTrue(result.initialized)
        self.assertEqual(result.points_world.shape, (14, 3))
        self.assertLess(result.coverage, 0.7)
        self.assertGreaterEqual(int(result.observed_mask.sum()), 5)
        self.assertGreater(float(result.points_world[-1, 0]), 0.70)

    def test_multi_hypothesis_tracker_keeps_complete_state(self) -> None:
        full = np.column_stack(
            [np.linspace(0.0, 0.8, 14), np.zeros(14), np.zeros(14)]
        )
        tracker = MultiHypothesisDLOTracker(expected_length_m=0.8, beam_width=2)
        first = tracker.update(full, observed_length_m=0.8)
        self.assertTrue(first.initialized)
        partial = full[4:11] + np.array([0.02, 0.01, 0.0])
        result = tracker.update(partial, observed_length_m=0.8 * 6 / 13)
        self.assertTrue(result.initialized)
        self.assertEqual(result.points_world.shape, (14, 3))
        self.assertGreaterEqual(result.timings_ms["hypothesis_count"], 1.0)

    def test_reversed_observation_is_aligned(self) -> None:
        full = np.column_stack(
            [np.linspace(0.0, 0.8, 14), np.zeros(14), np.zeros(14)]
        )
        tracker = TemporalDLOTracker(expected_length_m=0.8)
        tracker.update(full, observed_length_m=0.8)
        result = tracker.update(full[::-1], observed_length_m=0.8)
        np.testing.assert_allclose(result.points_world, full, atol=1e-4)

    def test_translated_partial_observation_matches_an_arc_interval(self) -> None:
        full = np.column_stack(
            [np.linspace(0.0, 0.8, 14), np.zeros(14), np.zeros(14)]
        )
        tracker = TemporalDLOTracker(expected_length_m=0.8)
        tracker.update(full, observed_length_m=0.8)
        partial = full[4:11] + np.array([0.12, -0.03, 0.02])
        result = tracker.update(partial, observed_length_m=0.8 * 6 / 13)
        self.assertGreater(result.coverage, 0.35)
        self.assertLess(result.matched_distance_m, 0.01)

    def test_visible_motion_is_propagated_to_hidden_nodes(self) -> None:
        full = np.column_stack(
            [np.linspace(0.0, 0.8, 14), np.zeros(14), np.zeros(14)]
        )
        tracker = TemporalDLOTracker(expected_length_m=0.8)
        tracker.update(full, observed_length_m=0.8)
        shift = np.array([0.0, 0.04, 0.0])
        partial = full[5:13] + shift
        result = tracker.update(partial, observed_length_m=0.8 * 7 / 13)
        self.assertGreater(float(result.points_world[0, 1]), 0.003)
        self.assertGreater(float(result.points_world[-1, 1]), 0.004)

    def test_consecutive_centerline_motion_transports_full_state(self) -> None:
        full = np.column_stack(
            [np.linspace(0.0, 0.8, 14), 0.04 * np.sin(np.linspace(0.0, 2.0, 14)), np.zeros(14)]
        )
        tracker = TemporalDLOTracker(expected_length_m=0.8)
        tracker.update(full, observed_length_m=0.8)
        shift = np.array([0.025, -0.018, 0.012])
        result = tracker.update(full + shift, observed_length_m=0.8)
        self.assertTrue(result.timings_ms["sequence_applied"])
        np.testing.assert_allclose(result.points_world, full + shift, atol=2e-2)

    def test_short_first_observation_is_not_claimed_complete(self) -> None:
        partial = np.column_stack(
            [np.linspace(0.2, 0.5, 14), np.zeros(14), np.zeros(14)]
        )
        tracker = TemporalDLOTracker(expected_length_m=0.8)
        result = tracker.update(partial, observed_length_m=0.3)
        self.assertFalse(result.initialized)
        self.assertIsNone(result.points_world)

    def test_short_first_observation_can_initialize_with_symmetric_extension(self) -> None:
        partial = np.column_stack(
            [np.linspace(0.2, 0.5, 14), np.zeros(14), np.zeros(14)]
        )
        tracker = TemporalDLOTracker(
            expected_length_m=0.8,
            allow_partial_initialization=True,
        )
        result = tracker.update(partial, observed_length_m=0.3)
        self.assertTrue(result.initialized)
        self.assertEqual(result.points_world.shape, (14, 3))
        self.assertLess(result.coverage, 0.6)
        self.assertAlmostEqual(
            float(np.linalg.norm(np.diff(result.points_world, axis=0), axis=1).sum()),
            0.8,
            places=3,
        )

    def test_disconnected_fragments_are_fused_on_one_state(self) -> None:
        full = np.column_stack(
            [np.linspace(0.0, 0.8, 14), np.zeros(14), np.zeros(14)]
        )
        tracker = TemporalDLOTracker(expected_length_m=0.8)
        tracker.update(full, observed_length_m=0.8)
        first = full[0:5]
        second = full[8:14] + np.array([0.0, 0.03, 0.0])
        result = tracker.update_fragments(
            [first, second],
            observed_lengths_m=[0.8 * 4 / 13, 0.8 * 5 / 13],
        )
        self.assertTrue(result.initialized)
        self.assertGreaterEqual(int(result.observed_mask.sum()), 7)
        self.assertEqual(result.timings_ms["fragment_count"], 2.0)

    def test_robot_occlusion_mask_removes_hidden_nodes_from_measurement(self) -> None:
        full = np.column_stack(
            [np.linspace(-0.20, 0.20, 14), np.zeros(14), np.ones(14)]
        )
        intrinsics = np.array(
            [[400.0, 0.0, 240.0], [0.0, 400.0, 180.0], [0.0, 0.0, 1.0]]
        )
        robot_mask = np.zeros((360, 480), dtype=bool)
        robot_mask[172:188, 228:252] = True
        tracker = TemporalDLOTracker(expected_length_m=0.4)
        result = tracker.update(
            full,
            observed_length_m=0.4,
            robot_occlusion_mask=robot_mask,
            camera_from_world=np.eye(4),
            intrinsics=intrinsics,
        )
        self.assertTrue(result.initialized)
        self.assertLess(int(result.observed_mask.sum()), len(full))
        self.assertGreater(result.timings_ms["robot_occlusion_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
