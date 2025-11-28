import unittest

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import MatchingMethod, MatchingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.matching import match_trajectories


class TestMatching(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(789)

        # Create a reference trajectory
        num_points = 100
        t_ref = np.linspace(0, 10, num_points)

        xyz_ref = np.column_stack([t_ref * 5, np.sin(t_ref) * 10, np.cos(t_ref) * 5])

        quat_ref = np.tile([0, 0, 0, 1], (num_points, 1))

        self.traj_ref = Trajectory(
            name="Reference",
            positions=Positions(xyz=xyz_ref, epsg=0),
            rotations=Rotations.from_quat(quat_ref),
            timestamps=t_ref,
        )

        # Create a test trajectory with slightly different timestamps
        num_test = 80
        t_test = np.linspace(0.1, 9.9, num_test)  # Slightly offset

        xyz_test = np.column_stack(
            [
                t_test * 5 + np.random.randn(num_test) * 0.1,
                np.sin(t_test) * 10 + np.random.randn(num_test) * 0.1,
                np.cos(t_test) * 5 + np.random.randn(num_test) * 0.1,
            ]
        )

        quat_test = np.tile([0, 0, 0, 1], (num_test, 1))

        self.traj_test = Trajectory(
            name="Test",
            positions=Positions(xyz=xyz_test, epsg=0),
            rotations=Rotations.from_quat(quat_test),
            timestamps=t_test,
        )

    def test_match_trajectories_nearest_temporal_basic(self):
        """Test basic temporal matching."""
        settings = MatchingSettings(method=MatchingMethod.NEAREST_TEMPORAL, max_time_diff=0.5)

        test_matched, ref_matched = match_trajectories(
            trajectory=self.traj_test, other=self.traj_ref, matching_settings=settings
        )

        # Verify both trajectories are matched
        self.assertIsNotNone(test_matched)
        self.assertIsNotNone(ref_matched)

        # Matched trajectories should have same length
        self.assertEqual(len(test_matched), len(ref_matched))

        # Length should be less than or equal to original
        self.assertLessEqual(len(test_matched), len(self.traj_test))
        self.assertLessEqual(len(ref_matched), len(self.traj_ref))

    def test_match_trajectories_nearest_temporal_tight_threshold(self):
        """Test temporal matching with tight time threshold."""
        settings = MatchingSettings(method=MatchingMethod.NEAREST_TEMPORAL, max_time_diff=0.01)

        test_matched, ref_matched = match_trajectories(
            trajectory=self.traj_test, other=self.traj_ref, matching_settings=settings
        )

        # Should still produce matches but fewer
        self.assertGreater(len(test_matched), 0)
        self.assertEqual(len(test_matched), len(ref_matched))

        # Time differences should be within threshold
        time_diffs = np.abs(test_matched.timestamps - ref_matched.timestamps)
        self.assertTrue(np.all(time_diffs <= 0.01))

    def test_match_trajectories_nearest_spatial_basic(self):
        """Test basic spatial matching."""
        settings = MatchingSettings(method=MatchingMethod.NEAREST_SPATIAL, max_distance=5.0)

        test_matched, ref_matched = match_trajectories(
            trajectory=self.traj_test, other=self.traj_ref, matching_settings=settings
        )

        # Verify matching worked
        self.assertIsNotNone(test_matched)
        self.assertIsNotNone(ref_matched)
        self.assertEqual(len(test_matched), len(ref_matched))

        # Should have found matches
        self.assertGreater(len(test_matched), 0)

    def test_match_trajectories_nearest_spatial_distance_constraint(self):
        """Test spatial matching respects distance constraint."""
        settings = MatchingSettings(method=MatchingMethod.NEAREST_SPATIAL, max_distance=1.0)

        test_matched, ref_matched = match_trajectories(
            trajectory=self.traj_test, other=self.traj_ref, matching_settings=settings
        )

        # Calculate actual distances between matched points
        distances = np.linalg.norm(test_matched.positions.xyz - ref_matched.positions.xyz, axis=1)

        # All distances should be within threshold
        self.assertTrue(np.all(distances <= 1.0))

    def test_match_trajectories_interpolation_basic(self):
        """Test interpolation-based matching."""
        settings = MatchingSettings(method=MatchingMethod.INTERPOLATION, max_gap_size=1.0)

        test_matched, ref_matched = match_trajectories(
            trajectory=self.traj_test, other=self.traj_ref, matching_settings=settings
        )

        # Verify matching worked
        self.assertIsNotNone(test_matched)
        self.assertIsNotNone(ref_matched)

        # After interpolation matching, lengths should be equal
        self.assertEqual(len(test_matched), len(ref_matched))

        # Timestamps should match
        np.testing.assert_array_almost_equal(test_matched.timestamps, ref_matched.timestamps)

    def test_match_trajectories_spatial_interpolated_basic(self):
        """Test spatial interpolation-based matching."""
        settings = MatchingSettings(method=MatchingMethod.NEAREST_SPATIAL_INTERPOLATED, max_distance=5.0, k_nearest=5)

        test_matched, ref_matched = match_trajectories(
            trajectory=self.traj_test, other=self.traj_ref, matching_settings=settings
        )

        # Verify matching worked
        self.assertIsNotNone(test_matched)
        self.assertIsNotNone(ref_matched)
        self.assertEqual(len(test_matched), len(ref_matched))
        self.assertGreater(len(test_matched), 0)

    def test_match_trajectories_preserves_rotations(self):
        """Test that matching preserves rotation data."""
        settings = MatchingSettings(method=MatchingMethod.NEAREST_TEMPORAL)

        test_matched, ref_matched = match_trajectories(
            trajectory=self.traj_test, other=self.traj_ref, matching_settings=settings
        )

        # Both should have rotations
        self.assertTrue(test_matched.has_orientation)
        self.assertTrue(ref_matched.has_orientation)
        self.assertEqual(len(test_matched.rotations), len(test_matched))
        self.assertEqual(len(ref_matched.rotations), len(ref_matched))

    def test_match_trajectories_without_rotations(self):
        """Test matching trajectories without rotations."""
        traj_test_no_rot = Trajectory(
            positions=self.traj_test.positions.copy(), timestamps=self.traj_test.timestamps.copy()
        )

        traj_ref_no_rot = Trajectory(
            positions=self.traj_ref.positions.copy(), timestamps=self.traj_ref.timestamps.copy()
        )

        settings = MatchingSettings(method=MatchingMethod.NEAREST_TEMPORAL)

        test_matched, ref_matched = match_trajectories(
            trajectory=traj_test_no_rot, other=traj_ref_no_rot, matching_settings=settings
        )

        # Should work without rotations
        self.assertIsNotNone(test_matched)
        self.assertIsNotNone(ref_matched)
        self.assertEqual(len(test_matched), len(ref_matched))

    def test_match_trajectories_no_overlap_returns_empty(self):
        """Test matching with no temporal overlap returns minimal results."""
        # Create trajectories with no overlap
        t_future = np.linspace(100, 110, 50)
        xyz_future = np.column_stack([t_future * 5, np.sin(t_future) * 10, np.cos(t_future) * 5])

        traj_future = Trajectory(positions=Positions(xyz=xyz_future, epsg=0), timestamps=t_future)

        settings = MatchingSettings(method=MatchingMethod.NEAREST_TEMPORAL, max_time_diff=0.1)

        test_matched, ref_matched = match_trajectories(
            trajectory=traj_future, other=self.traj_ref, matching_settings=settings
        )

        # Should return trajectories but with no or very few matches
        self.assertIsNotNone(test_matched)
        self.assertIsNotNone(ref_matched)
        # Both should have same length
        self.assertEqual(len(test_matched), len(ref_matched))
        # With no overlap and strict threshold, should have 0 or 1 match at most
        self.assertLessEqual(
            len(test_matched), 1, "Non-overlapping trajectories with strict threshold should produce no matches"
        )

    def test_match_trajectories_different_methods_produce_different_results(self):
        """Test that different matching methods produce different results."""
        settings_temporal = MatchingSettings(method=MatchingMethod.NEAREST_TEMPORAL)
        settings_spatial = MatchingSettings(method=MatchingMethod.NEAREST_SPATIAL)

        test_temp, ref_temp = match_trajectories(
            trajectory=self.traj_test.copy(), other=self.traj_ref.copy(), matching_settings=settings_temporal
        )

        test_spat, ref_spat = match_trajectories(
            trajectory=self.traj_test.copy(), other=self.traj_ref.copy(), matching_settings=settings_spatial
        )

        # Both should produce valid results
        self.assertGreater(len(test_temp), 0)
        self.assertGreater(len(test_spat), 0)

        # Results should be different (at least in some aspects)
        # Note: They might have same length but different point selections
        self.assertTrue(
            len(test_temp) != len(test_spat) or not np.array_equal(test_temp.positions.xyz, test_spat.positions.xyz)
        )

    def test_match_trajectories_identical_trajectories(self):
        """Test matching identical trajectories."""
        settings = MatchingSettings(method=MatchingMethod.NEAREST_TEMPORAL)

        traj_copy = self.traj_ref.copy()

        test_matched, ref_matched = match_trajectories(
            trajectory=traj_copy, other=self.traj_ref, matching_settings=settings
        )

        # Should match all points
        self.assertEqual(len(test_matched), len(self.traj_ref))
        self.assertEqual(len(ref_matched), len(self.traj_ref))

        # Timestamps should be identical
        np.testing.assert_array_equal(test_matched.timestamps, ref_matched.timestamps)


if __name__ == "__main__":
    unittest.main()
