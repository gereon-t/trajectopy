import unittest

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import ApproximationSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.approximation import approximate_cubic


class TestApproximation(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)

        # Create a noisy trajectory for testing approximation
        num_points = 100
        t = np.linspace(0, 10, num_points)

        # Create a smooth trajectory with added noise
        self.clean_xyz = np.column_stack([np.sin(t) * 10, np.cos(t) * 10, t * 0.5])

        # Add noise
        noise = np.random.randn(num_points, 3) * 0.5
        noisy_xyz = self.clean_xyz + noise

        # Create rotations with noise
        angles = np.column_stack([t * 0.1, t * 0.05, t * 0.02])
        noisy_angles = angles + np.random.randn(num_points, 3) * 0.01

        self.trajectory = Trajectory(
            positions=Positions(xyz=noisy_xyz, epsg=0),
            rotations=Rotations.from_euler(seq="xyz", angles=noisy_angles, degrees=False),
            timestamps=t,
        )

    def test_approximate_cubic_basic(self):
        """Test basic cubic approximation functionality."""
        settings = ApproximationSettings(
            position_interval_size=0.5, position_min_observations=5, rotation_window_size=0.5
        )

        result = approximate_cubic(trajectory=self.trajectory, approximation_settings=settings, inplace=False)

        # Verify structure
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Trajectory)
        self.assertEqual(len(result), len(self.trajectory))

        # Verify approximation smoothed the data
        # Calculate noise by comparing to clean trajectory
        original_noise = np.std(self.trajectory.positions.xyz - self.clean_xyz)
        approx_noise = np.std(result.positions.xyz - self.clean_xyz)

        # Approximation should reduce noise significantly
        self.assertLess(approx_noise, original_noise * 0.9, "Approximation should reduce noise by at least 10%")

    def test_approximate_cubic_inplace(self):
        """Test that inplace parameter works correctly."""
        traj_copy = self.trajectory.copy()

        result = approximate_cubic(trajectory=traj_copy, approximation_settings=ApproximationSettings(), inplace=True)

        # Result should be the same object
        self.assertIs(result, traj_copy)

        # Original trajectory should be modified
        self.assertFalse(np.array_equal(result.positions.xyz, self.trajectory.positions.xyz))

    def test_approximate_cubic_without_rotations(self):
        """Test approximation on trajectory without rotations."""
        traj_no_rot = Trajectory(
            positions=self.trajectory.positions.copy(), timestamps=self.trajectory.timestamps.copy()
        )

        result = approximate_cubic(
            trajectory=traj_no_rot, approximation_settings=ApproximationSettings(), inplace=False
        )

        # Should work without rotations
        self.assertIsNotNone(result)
        self.assertIsNone(result.rotations)
        self.assertEqual(len(result), len(traj_no_rot))

    def test_approximate_cubic_preserves_length(self):
        """Test that approximation preserves trajectory length."""
        result = approximate_cubic(
            trajectory=self.trajectory, approximation_settings=ApproximationSettings(), inplace=False
        )

        self.assertEqual(len(result), len(self.trajectory))
        self.assertEqual(result.positions.xyz.shape, self.trajectory.positions.xyz.shape)

    def test_approximate_cubic_different_settings(self):
        """Test approximation with different settings."""
        settings_small = ApproximationSettings(position_interval_size=0.2, position_min_observations=3)

        settings_large = ApproximationSettings(position_interval_size=1.0, position_min_observations=10)

        result_small = approximate_cubic(
            trajectory=self.trajectory, approximation_settings=settings_small, inplace=False
        )

        result_large = approximate_cubic(
            trajectory=self.trajectory, approximation_settings=settings_large, inplace=False
        )

        # Both should produce valid results
        self.assertIsNotNone(result_small)
        self.assertIsNotNone(result_large)

        # Different settings should produce different results
        max_diff = np.max(np.abs(result_small.positions.xyz - result_large.positions.xyz))
        self.assertGreater(
            max_diff, 0.01, "Different approximation settings should produce meaningfully different results"
        )

    def test_approximate_cubic_rotation_smoothing(self):
        """Test that rotation approximation smooths orientation data."""
        if not self.trajectory.has_orientation:
            self.skipTest("Trajectory has no orientation")

        settings = ApproximationSettings(rotation_window_size=0.5)

        result = approximate_cubic(trajectory=self.trajectory, approximation_settings=settings, inplace=False)

        # Verify rotations exist
        self.assertIsNotNone(result.rotations)
        self.assertEqual(len(result.rotations), len(self.trajectory.rotations))

        # Check that some smoothing occurred (angles should be less noisy)
        original_angles = self.trajectory.rotations.as_euler(seq="xyz")
        approx_angles = result.rotations.as_euler(seq="xyz")

        # Compute angular velocity (differences)
        original_ang_vel = np.abs(np.diff(original_angles, axis=0))
        approx_ang_vel = np.abs(np.diff(approx_angles, axis=0))

        # Smoothed trajectory should have less or equal rapid changes
        self.assertLessEqual(
            np.max(approx_ang_vel),
            np.max(original_ang_vel),
            "Approximation should not increase maximum angular velocity changes",
        )


if __name__ == "__main__":
    unittest.main()
