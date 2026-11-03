import copy
import os
import tempfile
import unittest

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import MatchingMethod, MatchingSettings, ProcessingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.evaluation import ate
from trajectopy.processing.matching import match_trajectories
from trajectopy.processing.sorting import sort_spatially
from trajectopy.results.ate_result import ATEResult
from trajectopy.utils.definitions import Sorting


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

        # Index should match
        np.testing.assert_array_almost_equal(test_matched.index, ref_matched.index)

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


def _loop(noise: float, rng: np.random.Generator, name: str, laps: int = 3) -> Trajectory:
    """Several laps of an elliptic loop (60 s per lap, 10 Hz) whose height has two equal maxima."""
    t = np.arange(0, laps * 60, 0.1)
    angle = 2 * np.pi * t / 60
    xyz = np.c_[100 * np.cos(angle), 50 * np.sin(angle), 2 * np.sin(2 * angle)] + rng.normal(0, noise, (len(t), 3))
    rotations = Rotations.from_euler("xyz", np.c_[np.zeros_like(t), np.zeros_like(t), angle + np.pi / 2])
    return Trajectory(name=name, positions=Positions(xyz, epsg=0), rotations=rotations, timestamps=t + 1000.0)


class TestSpatiallySortedMatching(unittest.TestCase):
    """Two spatially sorted trajectories start their path lengths at different places and must still be matched."""

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(0)
        # noise-free laps pass exactly the same positions, which leads to duplicate path lengths
        cls.reference = sort_spatially(_loop(0.0, rng, "reference"), inplace=False)
        cls.test = sort_spatially(_loop(0.03, rng, "test"), inplace=False)

    def test_interpolation_matches_the_same_places(self):
        matched, reference = match_trajectories(self.test, self.reference, inplace=False)
        distances = np.linalg.norm(matched.positions.xyz - reference.positions.xyz, axis=1)
        self.assertGreater(len(matched), 0.95 * len(self.test))
        self.assertLess(np.median(distances), 0.1)

    def test_ate_of_sorted_trajectories(self):
        for method in (MatchingMethod.INTERPOLATION, MatchingMethod.NEAREST_SPATIAL_INTERPOLATED):
            with self.subTest(method=method):
                settings = ProcessingSettings()
                settings.matching = MatchingSettings(method=method)
                result = ate(self.test, self.reference, settings)
                self.assertLess(np.median(result.pos_dev_comb), 0.1)


class TestSortedATEResult(unittest.TestCase):
    """ATE results of spatially sorted trajectories can be ordered by time or by path length."""

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(0)
        reference = sort_spatially(_loop(0.0, rng, "reference"), inplace=False)
        test = sort_spatially(_loop(0.03, rng, "test"), inplace=False)
        cls.result = ate(test, reference)

    def _by_timestamp(self, result: ATEResult) -> dict:
        order = np.argsort(result.trajectory.timestamps)
        return {
            "t": result.trajectory.timestamps[order],
            "l": result.trajectory.path_lengths[order],
            "xyz": result.trajectory.positions.xyz[order],
            "pos_dev": result.abs_dev.pos_dev[order],
            "directed_pos_dev": result.abs_dev.directed_pos_dev[order],
            "rot_dev": result.abs_dev.rot_dev.as_quat()[order],
            "rot_dev_xyz": result.rot_dev_xyz[order],
        }

    def test_switching_keeps_deviations_with_their_epochs(self):
        result = copy.deepcopy(self.result)
        self.assertEqual(result.trajectory.sorting, Sorting.PATH_LENGTH)
        self.assertTrue(result.is_spatially_sorted)
        before = self._by_timestamp(result)  # also fills the cached rotation deviations

        result.set_sorting(Sorting.TIME)
        self.assertEqual(result.trajectory.sorting, Sorting.TIME)
        self.assertTrue(np.all(np.diff(result.trajectory.timestamps) > 0))
        self.assertTrue(result.is_spatially_sorted)  # can be switched back
        for key, values in self._by_timestamp(result).items():
            np.testing.assert_array_equal(values, before[key], err_msg=key)

        # back by path length; epochs with equal path lengths (one per lap) may change places among themselves
        result.set_sorting(Sorting.PATH_LENGTH)
        self.assertTrue(np.all(np.diff(result.trajectory.path_lengths) >= 0))
        for key, values in self._by_timestamp(result).items():
            np.testing.assert_array_equal(values, before[key], err_msg=key)

    def test_chronological_results_are_not_spatially_sorted(self):
        rng = np.random.default_rng(1)
        self.assertFalse(ate(_loop(0.03, rng, "test"), _loop(0.0, rng, "reference")).is_spatially_sorted)

    def test_file_keeps_deviations_with_their_epochs(self):
        """Reading a spatially sorted result with equal path lengths of several laps keeps the deviations with their
        positions."""
        self.assertLess(len(np.unique(self.result.trajectory.path_lengths)), len(self.result.trajectory))
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "sorted.result")
            self.result.to_file(filename, mode="w")
            loaded = ATEResult.from_file(filename)
        # a mix-up would show as differences of centimeters, not of the file's number format
        np.testing.assert_allclose(loaded.trajectory.positions.xyz, self.result.trajectory.positions.xyz, atol=1e-9)
        np.testing.assert_allclose(loaded.abs_dev.pos_dev, self.result.abs_dev.pos_dev, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
