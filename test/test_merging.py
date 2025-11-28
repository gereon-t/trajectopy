import unittest

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.merging import average_trajectories, merge_trajectories


class TestMerging(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(456)

        # Create base trajectory
        num_points = 50
        t = np.linspace(0, 10, num_points)

        base_xyz = np.column_stack([t * 2, np.sin(t) * 5, np.cos(t) * 2])

        base_quat = np.tile([0, 0, 0, 1], (num_points, 1))

        # Create first trajectory
        self.traj1 = Trajectory(
            name="Trajectory1",
            positions=Positions(xyz=base_xyz.copy(), epsg=4326),
            rotations=Rotations.from_quat(base_quat.copy()),
            timestamps=t.copy(),
        )

        # Create second trajectory with different timestamps
        t2 = np.linspace(10, 20, num_points)
        xyz2 = np.column_stack([t2 * 2, np.sin(t2) * 5, np.cos(t2) * 2])

        self.traj2 = Trajectory(
            name="Trajectory2",
            positions=Positions(xyz=xyz2, epsg=4326),
            rotations=Rotations.from_quat(base_quat.copy()),
            timestamps=t2,
        )

        # Create third trajectory for merging tests
        t3 = np.linspace(5, 15, num_points)
        xyz3 = np.column_stack([t3 * 2, np.sin(t3) * 5, np.cos(t3) * 2])

        self.traj3 = Trajectory(
            name="Trajectory3",
            positions=Positions(xyz=xyz3, epsg=4326),
            rotations=Rotations.from_quat(base_quat.copy()),
            timestamps=t3,
        )

    def test_merge_trajectories_basic(self):
        """Test basic trajectory merging."""
        merged = merge_trajectories([self.traj1, self.traj2])

        # Verify structure
        self.assertIsNotNone(merged)
        self.assertIsInstance(merged, Trajectory)

        # Verify length is sum of input trajectories
        expected_length = len(self.traj1) + len(self.traj2)
        self.assertEqual(len(merged), expected_length)

        # Verify name
        self.assertEqual(merged.name, "Merged")

    def test_merge_trajectories_sorted_by_timestamp(self):
        """Test that merged trajectory is sorted by timestamp."""
        merged = merge_trajectories([self.traj2, self.traj1])  # Intentionally reversed

        # Timestamps should be sorted
        self.assertTrue(np.all(np.diff(merged.timestamps) >= 0))

        # First timestamp should be from traj1, last from traj2
        self.assertAlmostEqual(merged.timestamps[0], self.traj1.timestamps[0])
        self.assertAlmostEqual(merged.timestamps[-1], self.traj2.timestamps[-1])

    def test_merge_trajectories_multiple(self):
        """Test merging more than two trajectories."""
        merged = merge_trajectories([self.traj1, self.traj2, self.traj3])

        # Verify all trajectories are included
        expected_length = len(self.traj1) + len(self.traj2) + len(self.traj3)
        self.assertEqual(len(merged), expected_length)

        # Timestamps should still be sorted
        self.assertTrue(np.all(np.diff(merged.timestamps) >= 0))

    def test_merge_trajectories_preserves_rotations(self):
        """Test that merging preserves rotation data."""
        merged = merge_trajectories([self.traj1, self.traj2])

        # Should have rotations
        self.assertTrue(merged.has_orientation)
        self.assertIsNotNone(merged.rotations)
        self.assertEqual(len(merged.rotations), len(merged))

    def test_merge_trajectories_without_rotations(self):
        """Test merging trajectories without rotations."""
        traj1_no_rot = Trajectory(positions=self.traj1.positions.copy(), timestamps=self.traj1.timestamps.copy())

        traj2_no_rot = Trajectory(positions=self.traj2.positions.copy(), timestamps=self.traj2.timestamps.copy())

        merged = merge_trajectories([traj1_no_rot, traj2_no_rot])

        # Should work without rotations (creates identity rotations)
        self.assertIsNotNone(merged)
        self.assertEqual(len(merged), len(traj1_no_rot) + len(traj2_no_rot))

    def test_merge_trajectories_preserves_epsg(self):
        """Test that EPSG code is preserved in merging."""
        merged = merge_trajectories([self.traj1, self.traj2])

        # Should preserve EPSG from input trajectories
        self.assertEqual(merged.positions.epsg, 4326)

    def test_average_trajectories_basic(self):
        """Test basic trajectory averaging."""
        # Create two trajectories with same timestamps but different positions
        traj_a = self.traj1.copy()
        traj_b = self.traj1.copy()

        # Add different noise to each
        traj_a.positions.xyz += np.random.randn(*traj_a.positions.xyz.shape) * 0.1
        traj_b.positions.xyz += np.random.randn(*traj_b.positions.xyz.shape) * 0.1

        averaged = average_trajectories([traj_a, traj_b])

        # Verify structure
        self.assertIsNotNone(averaged)
        self.assertIsInstance(averaged, Trajectory)
        self.assertEqual(len(averaged), len(traj_a))

        # Verify name
        self.assertEqual(averaged.name, "Averaged")

    def test_average_trajectories_multiple(self):
        """Test averaging multiple trajectories."""
        # Create multiple copies with same timestamps
        trajs = []
        for i in range(5):
            traj = self.traj1.copy()
            traj.positions.xyz += np.random.randn(*traj.positions.xyz.shape) * 0.05
            trajs.append(traj)

        averaged = average_trajectories(trajs)

        self.assertIsNotNone(averaged)
        self.assertEqual(len(averaged), len(self.traj1))

    def test_average_trajectories_reduces_noise(self):
        """Test that averaging actually reduces noise."""
        # Create a clean trajectory as ground truth
        clean_traj = self.traj1.copy()

        # Create noisy versions of the same trajectory
        num_versions = 10
        trajs = []
        for i in range(num_versions):
            traj = clean_traj.copy()
            traj.positions.xyz += np.random.randn(*traj.positions.xyz.shape) * 1.0
            trajs.append(traj)

        averaged = average_trajectories(trajs)

        # Averaged trajectory should be closer to clean original than individual noisy versions
        avg_distance = np.mean(np.linalg.norm(averaged.positions.xyz - clean_traj.positions.xyz, axis=1))

        # Calculate mean distance of all noisy trajectories
        noisy_distances = [np.mean(np.linalg.norm(t.positions.xyz - clean_traj.positions.xyz, axis=1)) for t in trajs]
        mean_noisy_distance = np.mean(noisy_distances)

        # Average should be significantly closer to ground truth (at least 50% better)
        self.assertLess(
            avg_distance,
            mean_noisy_distance * 0.5,
            "Averaged trajectory should be significantly closer to ground truth than noisy versions",
        )

    def test_average_trajectories_different_lengths_raises(self):
        """Test that averaging trajectories with different lengths raises error."""
        traj_short = Trajectory(
            positions=Positions(xyz=self.traj1.positions.xyz[:10], epsg=4326), timestamps=self.traj1.timestamps[:10]
        )

        with self.assertRaises(ValueError) as context:
            average_trajectories([self.traj1, traj_short])

        self.assertIn("same length", str(context.exception).lower(), "Error message should mention length mismatch")

    def test_average_trajectories_preserves_timestamps(self):
        """Test that averaging preserves timestamps from first trajectory."""
        traj_a = self.traj1.copy()
        traj_b = self.traj1.copy()
        traj_b.positions.xyz += 1.0

        averaged = average_trajectories([traj_a, traj_b])

        # Timestamps should match first trajectory
        np.testing.assert_array_equal(averaged.timestamps, traj_a.timestamps)

    def test_average_trajectories_with_rotations(self):
        """Test averaging trajectories with rotation data."""
        traj_a = self.traj1.copy()
        traj_b = self.traj1.copy()

        # Slightly modify rotations
        angles_a = traj_a.rotations.as_euler(seq="xyz")
        angles_a += np.random.randn(*angles_a.shape) * 0.01
        traj_a.rotations = Rotations.from_euler(seq="xyz", angles=angles_a, degrees=False)

        averaged = average_trajectories([traj_a, traj_b])

        # Should have rotations
        self.assertTrue(averaged.has_orientation)
        self.assertIsNotNone(averaged.rotations)

    def test_average_trajectories_without_rotations(self):
        """Test averaging trajectories without rotations."""
        traj_a = Trajectory(positions=self.traj1.positions.copy(), timestamps=self.traj1.timestamps.copy())

        traj_b = Trajectory(positions=self.traj1.positions.copy(), timestamps=self.traj1.timestamps.copy())
        traj_b.positions.xyz += 0.5

        averaged = average_trajectories([traj_a, traj_b])

        # Should work without rotations
        self.assertIsNotNone(averaged)
        self.assertEqual(len(averaged), len(traj_a))


if __name__ == "__main__":
    unittest.main()
