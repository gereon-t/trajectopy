import unittest
from test.testdata import open_loop_trajectory
from test.util import random_number
from typing import Any

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.interpolation import interpolate
from trajectopy.processing.matching import match_timestamps


class TestTrajectory(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)

    def test_match_timestamps(self) -> None:
        trajectory_ref = open_loop_trajectory.copy()
        trajectory = open_loop_trajectory.copy()
        random_sampling = np.random.choice(
            np.arange(len(trajectory)),
            size=np.random.randint(1, len(trajectory) // 1.2),
            replace=False,
        )
        trajectory.mask(random_sampling)
        match_timestamps(trajectory_ref, trajectory.timestamps)

        self.assertListEqual(sorted(trajectory_ref.timestamps.tolist()), sorted(trajectory.timestamps.tolist()))

        self.trajectory_sanity_check(trajectory)
        self.trajectory_sanity_check(trajectory_ref)

    def test_interpolate(self) -> None:
        trajectory_ref = open_loop_trajectory.copy()
        trajectory = self.generate_altered_trajectory()

        random_sampling = np.array(
            [
                random_number(lower_bound=trajectory.timestamps[0], upper_bound=trajectory.timestamps[-1])
                for _ in range(np.random.randint(1, len(trajectory) // 1.2))
            ]
        )
        interpolate(trajectory, timestamps=random_sampling)
        interpolate(trajectory_ref, timestamps=random_sampling)

        traj_tstamps_set = set(trajectory.timestamps)
        traj_ref_tstamps_set = set(trajectory_ref.timestamps)

        if len(traj_ref_tstamps_set) > len(traj_tstamps_set):
            self.assertTrue(traj_tstamps_set.issubset(traj_ref_tstamps_set))
        else:
            self.assertTrue(traj_ref_tstamps_set.issubset(traj_tstamps_set))

        self.trajectory_sanity_check(trajectory)
        self.trajectory_sanity_check(trajectory_ref)

    def test_crop(self) -> None:
        trajectory = open_loop_trajectory.copy()

        tstamp_min = random_number(
            lower_bound=trajectory.timestamps[0], upper_bound=trajectory.timestamps[len(trajectory) // 2]
        )
        tstamp_max = random_number(lower_bound=tstamp_min, upper_bound=trajectory.timestamps[-1])

        trajectory.crop(t_start=tstamp_min, t_end=tstamp_max)

        self.assertGreaterEqual(trajectory.timestamps[0], tstamp_min, "First timestamp should be >= crop start")
        self.assertLessEqual(trajectory.timestamps[-1], tstamp_max, "Last timestamp should be <= crop end")
        self.trajectory_sanity_check(trajectory)

        trajectory.crop(t_start=tstamp_min, t_end=tstamp_min, inplace=True)
        self.assertEqual(len(trajectory), 0, "Cropping to same start and end should result in empty trajectory")
        self.trajectory_sanity_check(trajectory)

    def test_intersect(self) -> None:
        trajectory_ref = open_loop_trajectory.copy()
        trajectory = self.generate_altered_trajectory()

        trajectory.intersect(timestamps=trajectory_ref.timestamps)

        self.assertGreaterEqual(
            trajectory.timestamps[0],
            trajectory_ref.timestamps[0],
            "First timestamp should be >= reference first timestamp after intersection",
        )
        self.assertLessEqual(
            trajectory.timestamps[-1],
            trajectory_ref.timestamps[-1],
            "Last timestamp should be <= reference last timestamp after intersection",
        )

        self.trajectory_sanity_check(trajectory)
        self.trajectory_sanity_check(trajectory_ref)

    def generate_altered_trajectory(self) -> Trajectory:
        trajectory = open_loop_trajectory.copy()

        random_sampling = np.random.choice(
            np.arange(len(trajectory)),
            size=np.random.randint(
                int(len(trajectory) * 0.1),
                max(
                    int(len(trajectory) * 0.1) + 1,
                    int(len(trajectory) // (np.random.rand() * (10 - 1.1) + 1.1)),
                ),
            ),
            replace=False,
        )
        trajectory.mask(random_sampling)
        trajectory.timestamps += np.random.rand(len(trajectory)) * np.random.rand() * 10
        trajectory.mask(np.argsort(trajectory.timestamps))
        return trajectory

    def check_trajectory_attribute(self, attribute: Any, target_length: int, target_type: Any) -> None:
        self.assertIsInstance(attribute, target_type, f"Attribute should be of type {target_type}")
        if target_length > 0:
            self.assertEqual(len(attribute), target_length, f"Attribute should have length {target_length}")

    def trajectory_sanity_check(self, trajectory: Trajectory) -> None:
        target_length = len(trajectory)
        self.check_trajectory_attribute(trajectory.timestamps, target_length=target_length, target_type=np.ndarray)
        self.check_trajectory_attribute(trajectory.positions, target_length=target_length, target_type=Positions)

        if trajectory.rotations is not None:
            self.check_trajectory_attribute(trajectory.rotations, target_length=target_length, target_type=Rotations)

        self.check_trajectory_attribute(
            trajectory.absolute_velocity, target_length=target_length, target_type=np.ndarray
        )
        self.check_trajectory_attribute(trajectory.velocity_xyz, target_length=target_length, target_type=np.ndarray)
        self.check_trajectory_attribute(trajectory.path_lengths, target_length=target_length, target_type=np.ndarray)

        self.check_trajectory_attribute(trajectory.se3, target_length=target_length, target_type=list)
        self.check_trajectory_attribute(trajectory.data_rate, target_length=0, target_type=float)
        self.check_trajectory_attribute(trajectory.total_length, target_length=0, target_type=float)

    def test_empty_trajectory_operations(self) -> None:
        """Test operations on minimal trajectory after cropping."""
        trajectory = open_loop_trajectory.copy()
        t_start = trajectory.timestamps[0]
        t_end = trajectory.timestamps[0]
        trajectory.crop(t_start=t_start, t_end=t_end, inplace=True)

        # Cropping to same start and end timestamp keeps one point
        self.assertLessEqual(len(trajectory), 1, "Cropping to single timestamp should result in 0 or 1 points")

        if len(trajectory) == 0:
            self.assertEqual(trajectory.total_length, 0.0)
            # data_rate may be nan or inf for empty trajectory
            self.assertTrue(np.isnan(trajectory.data_rate) or np.isinf(trajectory.data_rate))
        else:
            # Single point has zero length
            self.assertEqual(trajectory.total_length, 0.0)
