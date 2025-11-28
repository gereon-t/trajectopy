import unittest

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.exceptions import TrajectoryError


class TestTrajectoryValidation(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(13)

    def test_trajectory_from_arrays_with_quat(self):
        xyz = np.random.rand(10, 3)
        quat = np.tile([0, 0, 0, 1], (10, 1))
        traj = Trajectory.from_arrays(xyz=xyz, quat=quat, epsg=4326)
        self.assertEqual(len(traj), 10)
        self.assertIsNotNone(traj.rotations)
        self.assertEqual(traj.positions.epsg, 4326)

    def test_trajectory_from_arrays_with_rpy(self):
        xyz = np.random.rand(10, 3)
        rpy = np.zeros((10, 3))
        traj = Trajectory.from_arrays(xyz=xyz, rpy=rpy, epsg=0)
        self.assertEqual(len(traj), 10)
        self.assertIsNotNone(traj.rotations)

    def test_trajectory_from_arrays_quat_and_rpy_raises(self):
        xyz = np.random.rand(10, 3)
        quat = np.tile([0, 0, 0, 1], (10, 1))
        rpy = np.zeros((10, 3))
        with self.assertRaises(TrajectoryError):
            Trajectory.from_arrays(xyz=xyz, quat=quat, rpy=rpy)

    def test_trajectory_mismatched_positions_rotations_raises(self):
        positions = Positions(xyz=np.random.rand(10, 3), epsg=0)
        rotations = Rotations.from_quat(np.tile([0, 0, 0, 1], (5, 1)))
        with self.assertRaises(TrajectoryError):
            Trajectory(positions=positions, rotations=rotations)

    def test_trajectory_without_rotations(self):
        positions = Positions(xyz=np.random.rand(10, 3), epsg=0)
        traj = Trajectory(positions=positions, rotations=None)
        self.assertEqual(len(traj), 10)
        self.assertIsNone(traj.rotations)
        self.assertFalse(traj.has_orientation)

    def test_trajectory_with_timestamps(self):
        positions = Positions(xyz=np.random.rand(10, 3), epsg=0)
        timestamps = np.arange(10, dtype=float) * 0.1
        traj = Trajectory(positions=positions, timestamps=timestamps)
        np.testing.assert_array_equal(traj.timestamps, timestamps)

    def test_trajectory_without_timestamps_defaults_to_range(self):
        positions = Positions(xyz=np.random.rand(10, 3), epsg=0)
        traj = Trajectory(positions=positions)
        np.testing.assert_array_equal(traj.timestamps, np.arange(10))

    def test_trajectory_copy_is_independent(self):
        positions = Positions(xyz=np.random.rand(10, 3), epsg=0)
        traj1 = Trajectory(positions=positions)
        traj2 = traj1.copy()
        traj2.positions.xyz[0, 0] = 999.0
        self.assertNotEqual(traj1.positions.xyz[0, 0], traj2.positions.xyz[0, 0])

    def test_trajectory_equality(self):
        xyz = np.random.rand(10, 3)
        timestamps = np.arange(10, dtype=float)
        positions1 = Positions(xyz=xyz.copy(), epsg=0)
        positions2 = Positions(xyz=xyz.copy(), epsg=0)
        traj1 = Trajectory(positions=positions1, timestamps=timestamps.copy(), name="Test")
        traj2 = Trajectory(positions=positions2, timestamps=timestamps.copy(), name="Test")
        self.assertEqual(traj1, traj2)

    def test_trajectory_inequality_different_positions(self):
        positions1 = Positions(xyz=np.random.rand(10, 3), epsg=0)
        positions2 = Positions(xyz=np.random.rand(10, 3), epsg=0)
        traj1 = Trajectory(positions=positions1)
        traj2 = Trajectory(positions=positions2)
        self.assertNotEqual(traj1, traj2)

    def test_single_point_trajectory(self):
        """Test that single point trajectory is handled correctly."""
        positions = Positions(xyz=np.array([[1.0, 2.0, 3.0]]), epsg=0)
        traj = Trajectory(positions=positions)
        self.assertEqual(len(traj), 1)
        self.assertEqual(traj.total_length, 0.0)
        # Single point should have zero velocity
        self.assertEqual(traj.absolute_velocity[0], 0.0)


if __name__ == "__main__":
    unittest.main()
