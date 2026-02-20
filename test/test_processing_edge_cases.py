import unittest
from test.testdata import open_loop_trajectory

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.core.settings import MatchingMethod, MatchingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.interpolation import interpolate
from trajectopy.processing.matching import match_trajectories


class TestProcessingEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(29)

    def test_match_trajectories_interpolation_method(self):
        traj1 = open_loop_trajectory.copy()
        traj2 = open_loop_trajectory.copy()
        # Shift timestamps slightly
        traj2.timestamps += 0.5

        settings = MatchingSettings(method=MatchingMethod.INTERPOLATION)
        matched1, matched2 = match_trajectories(trajectory=traj1, other=traj2, matching_settings=settings)

        self.assertIsInstance(matched1, Trajectory)
        self.assertIsInstance(matched2, Trajectory)
        self.assertGreater(len(matched1), 0)

    def test_match_trajectories_nearest_temporal(self):
        traj1 = open_loop_trajectory.copy()
        traj2 = open_loop_trajectory.copy()

        settings = MatchingSettings(method=MatchingMethod.NEAREST_TEMPORAL, max_time_diff=1.0)
        matched1, matched2 = match_trajectories(trajectory=traj1, other=traj2, matching_settings=settings)

        self.assertEqual(len(matched1), len(matched2))

    def test_interpolate_preserves_trajectory_structure(self):
        traj = open_loop_trajectory.copy()
        original_length = len(traj)

        # Interpolate to half the timestamps
        new_timestamps = traj.timestamps[::2]
        interpolate(traj, timestamps=new_timestamps)

        self.assertEqual(len(traj), len(new_timestamps))
        self.assertIsNotNone(traj.positions)
        if original_length > 0:
            self.assertIsNotNone(traj.rotations)

    def test_interpolate_single_timestamp(self):
        traj = open_loop_trajectory.copy()
        mid_timestamp = traj.timestamps[len(traj) // 2]

        interpolate(traj, timestamps=np.array([mid_timestamp]))

        self.assertEqual(len(traj), 1)

    def test_trajectory_crop_removes_poses_outside_range(self):
        traj = open_loop_trajectory.copy()
        original_length = len(traj)

        t_start = traj.timestamps[10]
        t_end = traj.timestamps[original_length - 10]

        traj.crop(t_start=t_start, t_end=t_end)

        self.assertLess(len(traj), original_length)
        self.assertGreaterEqual(traj.timestamps[0], t_start)
        self.assertLessEqual(traj.timestamps[-1], t_end)

    def test_trajectory_intersect_with_subset(self):
        traj = open_loop_trajectory.copy()
        original_length = len(traj)

        # Create subset of timestamps
        subset_timestamps = traj.timestamps[10 : original_length - 10]

        traj.intersect(timestamps=subset_timestamps)

        self.assertLessEqual(len(traj), original_length)
        self.assertGreaterEqual(traj.timestamps[0], subset_timestamps[0])
        self.assertLessEqual(traj.timestamps[-1], subset_timestamps[-1])

    def test_trajectory_mask_with_indices(self):
        traj = open_loop_trajectory.copy()

        # Keep only every other pose
        indices = np.arange(0, len(traj), 2)
        traj.mask(indices)

        self.assertEqual(len(traj), len(indices))

    def test_trajectory_has_orientation_property(self):
        # With rotations
        traj_with_rot = open_loop_trajectory.copy()
        self.assertTrue(traj_with_rot.has_orientation)

        # Without rotations
        positions = Positions(xyz=np.random.rand(10, 3), epsg=0)
        traj_without_rot = Trajectory(positions=positions, rotations=None)
        self.assertFalse(traj_without_rot.has_orientation)

    def test_trajectory_total_length_computation(self):
        traj = open_loop_trajectory.copy()
        total_length = traj.total_length

        self.assertIsInstance(total_length, float)
        self.assertGreater(total_length, 0)

    def test_trajectory_data_rate_computation(self):
        traj = open_loop_trajectory.copy()
        data_rate = traj.data_rate

        self.assertIsInstance(data_rate, float)
        self.assertGreater(data_rate, 0)


if __name__ == "__main__":
    unittest.main()
