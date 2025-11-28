import unittest

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import SortingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.sorting import divide_into_laps, sort_spatially
from trajectopy.utils.definitions import Sorting


class TestSorting(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(123)

        # Create a circular trajectory (loop)
        num_points = 200
        angles = np.linspace(0, 4 * np.pi, num_points)  # Two laps

        # Create a circular path with some noise
        radius = 100
        self.loop_xyz = np.column_stack(
            [
                radius * np.cos(angles) + np.random.randn(num_points) * 1.0,
                radius * np.sin(angles) + np.random.randn(num_points) * 1.0,
                np.ones(num_points) * 10 + np.random.randn(num_points) * 0.1,
            ]
        )

        # Shuffle the trajectory to make it unsorted
        shuffle_idx = np.random.permutation(num_points)
        shuffled_xyz = self.loop_xyz[shuffle_idx]

        self.unsorted_trajectory = Trajectory(
            positions=Positions(xyz=shuffled_xyz, epsg=0), timestamps=np.arange(num_points, dtype=float)
        )

        # Create a sorted trajectory for lap division
        self.sorted_trajectory = Trajectory(
            positions=Positions(xyz=self.loop_xyz, epsg=0), timestamps=np.arange(num_points, dtype=float)
        )

    def test_sort_spatially_basic(self):
        """Test basic spatial sorting functionality."""
        result = sort_spatially(
            trajectory=self.unsorted_trajectory.copy(), sorting_settings=SortingSettings(), inplace=False
        )

        # Verify structure
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Trajectory)
        self.assertEqual(len(result), len(self.unsorted_trajectory))

        # Verify sorting type is set
        self.assertEqual(result.sorting, Sorting.PATH_LENGTH)

        # Verify path_lengths were computed
        self.assertIsNotNone(result.path_lengths)
        self.assertEqual(len(result.path_lengths), len(result))

        # Path lengths should be non-negative
        self.assertTrue(np.all(result.path_lengths >= 0))

    def test_sort_spatially_inplace(self):
        """Test that inplace sorting modifies the original trajectory."""
        traj_copy = self.unsorted_trajectory.copy()

        result = sort_spatially(trajectory=traj_copy, sorting_settings=SortingSettings(), inplace=True)

        # Result should be the same object
        self.assertIs(result, traj_copy)

        # Path lengths and sorting should be set
        self.assertEqual(result.sorting, Sorting.PATH_LENGTH)
        self.assertIsNotNone(result.path_lengths)

    def test_sort_spatially_path_lengths_properties(self):
        """Test properties of computed path lengths after sorting."""
        result = sort_spatially(
            trajectory=self.unsorted_trajectory.copy(), sorting_settings=SortingSettings(), inplace=False
        )

        # Path lengths should be computed for all points
        self.assertEqual(len(result.path_lengths), len(result))

        # Path lengths should be valid numbers
        self.assertTrue(np.all(np.isfinite(result.path_lengths)))

        # Path lengths should have meaningful range
        self.assertGreater(np.max(result.path_lengths), 0)
        self.assertGreaterEqual(np.min(result.path_lengths), 0)

        # Test that sorting actually improves spatial coherence
        # Maximum jump distance should be smaller or equal after sorting
        sorted_dists = np.linalg.norm(np.diff(result.positions.xyz, axis=0), axis=1)
        unsorted_dists = np.linalg.norm(np.diff(self.unsorted_trajectory.positions.xyz, axis=0), axis=1)

        # Maximum distance between adjacent points should not increase after sorting
        # (sorting reduces or maintains the largest jumps)
        self.assertLessEqual(
            np.max(sorted_dists),
            np.max(unsorted_dists),
            "Sorting should not increase maximum jumps between adjacent points",
        )

    def test_sort_spatially_preserves_trajectory_properties(self):
        """Test that sorting preserves trajectory properties."""
        # Add rotations to trajectory
        traj_with_rot = self.unsorted_trajectory.copy()
        num_points = len(traj_with_rot)
        traj_with_rot.rotations = Rotations.from_quat(np.tile([0, 0, 0, 1], (num_points, 1)))

        result = sort_spatially(trajectory=traj_with_rot, sorting_settings=SortingSettings(), inplace=False)

        # Properties should be preserved
        self.assertEqual(len(result), len(traj_with_rot))
        self.assertIsNotNone(result.rotations)
        self.assertEqual(len(result.rotations), len(result))

    def test_divide_into_laps_basic(self):
        """Test basic lap division functionality."""
        # First sort the trajectory
        sorted_traj = sort_spatially(
            trajectory=self.sorted_trajectory.copy(), sorting_settings=SortingSettings(), inplace=False
        )

        laps = divide_into_laps(trajectory=sorted_traj, sorting_settings=SortingSettings(), return_lap_indices=False)

        # Should detect approximately 2 laps (we created the trajectory with 4*pi)
        self.assertIsInstance(laps, list)
        self.assertGreater(len(laps), 0)

        # Each lap should be a trajectory
        for lap in laps:
            self.assertIsInstance(lap, Trajectory)
            self.assertGreater(len(lap), 0)

    def test_divide_into_laps_with_indices(self):
        """Test lap division with indices return."""
        sorted_traj = sort_spatially(
            trajectory=self.sorted_trajectory.copy(), sorting_settings=SortingSettings(), inplace=False
        )

        laps, lap_indices = divide_into_laps(
            trajectory=sorted_traj, sorting_settings=SortingSettings(), return_lap_indices=True
        )

        # Verify return types
        self.assertIsInstance(laps, list)
        self.assertIsInstance(lap_indices, np.ndarray)

        # lap_indices should start with 0 and end with trajectory length
        self.assertEqual(lap_indices[0], 0)
        self.assertEqual(lap_indices[-1], len(sorted_traj))

        # Number of laps should match indices
        self.assertEqual(len(laps), len(lap_indices) - 1)

    def test_divide_into_laps_preserves_total_length(self):
        """Test that lap division preserves the total trajectory length."""
        sorted_traj = sort_spatially(
            trajectory=self.sorted_trajectory.copy(), sorting_settings=SortingSettings(), inplace=False
        )

        laps = divide_into_laps(trajectory=sorted_traj, sorting_settings=SortingSettings(), return_lap_indices=False)

        # Total length of all laps should not exceed original trajectory
        total_lap_length = sum(len(lap) for lap in laps)
        self.assertLessEqual(total_lap_length, len(sorted_traj))

        # Should be close to original length
        self.assertGreater(total_lap_length, len(sorted_traj) * 0.8)

    def test_divide_into_laps_already_sorted(self):
        """Test lap division on already sorted trajectory."""
        # Create a trajectory that's already sorted
        sorted_traj = self.sorted_trajectory.copy()
        sorted_traj.sorting = Sorting.PATH_LENGTH
        sorted_traj.path_lengths = np.cumsum(
            np.r_[0, np.sqrt(np.sum(np.diff(sorted_traj.positions.xyz, axis=0) ** 2, axis=1))]
        )

        laps = divide_into_laps(trajectory=sorted_traj, sorting_settings=SortingSettings(), return_lap_indices=False)

        # Should work without re-sorting
        self.assertIsInstance(laps, list)
        self.assertGreater(len(laps), 0)

    def test_sort_spatially_different_settings(self):
        """Test sorting with different voxel sizes."""
        settings_small = SortingSettings(voxel_size=5.0)
        settings_large = SortingSettings(voxel_size=20.0)

        result_small = sort_spatially(
            trajectory=self.unsorted_trajectory.copy(), sorting_settings=settings_small, inplace=False
        )

        result_large = sort_spatially(
            trajectory=self.unsorted_trajectory.copy(), sorting_settings=settings_large, inplace=False
        )

        # Both should produce valid results
        self.assertIsNotNone(result_small)
        self.assertIsNotNone(result_large)

        # Both should have path lengths
        self.assertIsNotNone(result_small.path_lengths)
        self.assertIsNotNone(result_large.path_lengths)


if __name__ == "__main__":
    unittest.main()
