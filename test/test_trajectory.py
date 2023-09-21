import unittest
from test.testdata import open_loop_trajectory
from test.util import random_number
from typing import Any

import numpy as np
from pointset import PointSet

from trajectopy_core.trajectory import Trajectory
from trajectopy_core.util.rotationset import RotationSet


class TestTrajectory(unittest.TestCase):
    def test_match_timestamps(self) -> None:
        trajectory_ref = open_loop_trajectory.copy()
        trajectory = open_loop_trajectory.copy()
        random_sampling = np.random.choice(
            trajectory.idx_chrono,
            size=np.random.randint(1, len(trajectory) // 1.2),
            replace=False,
        )
        trajectory.apply_index(random_sampling)
        trajectory_ref.match_timestamps(trajectory.tstamps)

        self.assertListEqual(sorted(trajectory_ref.tstamps.tolist()), sorted(trajectory.tstamps.tolist()))

        self.trajectory_sanity_check(trajectory)
        self.trajectory_sanity_check(trajectory_ref)

    def test_same_sampling(self) -> None:
        trajectory_ref = open_loop_trajectory.copy()
        trajectory = self.generate_altered_trajectory()
        trajectory_ref.same_sampling(trajectory)

        self.assertListEqual(sorted(trajectory_ref.tstamps.tolist()), sorted(trajectory.tstamps.tolist()))
        self.trajectory_sanity_check(trajectory)
        self.trajectory_sanity_check(trajectory_ref)

    def test_interpolate(self) -> None:
        trajectory_ref = open_loop_trajectory.copy()
        trajectory = self.generate_altered_trajectory()

        random_sampling = np.array(
            [
                random_number(min=trajectory.tstamps[0], max=trajectory.tstamps[-1])
                for _ in range(np.random.randint(1, len(trajectory) // 1.2))
            ]
        )
        trajectory.interpolate(tstamps=random_sampling)
        trajectory_ref.interpolate(tstamps=random_sampling)

        traj_tstamps_set = set(trajectory.tstamps)
        traj_ref_tstamps_set = set(trajectory_ref.tstamps)

        if len(traj_ref_tstamps_set) > len(traj_tstamps_set):
            self.assertTrue(traj_tstamps_set.issubset(traj_ref_tstamps_set))
        else:
            self.assertTrue(traj_ref_tstamps_set.issubset(traj_tstamps_set))

        self.trajectory_sanity_check(trajectory)
        self.trajectory_sanity_check(trajectory_ref)

    def test_crop(self) -> None:
        trajectory = open_loop_trajectory.copy()

        tstamp_min = random_number(min=trajectory.tstamps[0], max=trajectory.tstamps[len(trajectory) // 2])
        tstamp_max = random_number(min=tstamp_min, max=trajectory.tstamps[-1])

        trajectory.crop(t_start=tstamp_min, t_end=tstamp_max)

        self.assertTrue(trajectory.tstamps[0] >= tstamp_min and trajectory.tstamps[-1] <= tstamp_max)
        self.trajectory_sanity_check(trajectory)

        trajectory.crop(t_start=tstamp_min, t_end=tstamp_min, inplace=True)
        self.assertTrue(len(trajectory) == 0)
        self.trajectory_sanity_check(trajectory)

    def test_intersect(self) -> None:
        trajectory_ref = open_loop_trajectory.copy()
        trajectory = self.generate_altered_trajectory()

        trajectory.intersect(tstamps=trajectory_ref.tstamps)

        self.assertTrue(
            trajectory.tstamps[0] >= trajectory_ref.tstamps[0] and trajectory.tstamps[-1] <= trajectory_ref.tstamps[-1]
        )

        self.trajectory_sanity_check(trajectory)
        self.trajectory_sanity_check(trajectory_ref)

    def generate_altered_trajectory(self) -> Trajectory:
        trajectory = open_loop_trajectory.copy()

        random_sampling = np.random.choice(
            trajectory.idx_chrono,
            size=np.random.randint(
                int(len(trajectory) * 0.1),
                max(
                    int(len(trajectory) * 0.1) + 1,
                    int(len(trajectory) // (np.random.rand() * (10 - 1.1) + 1.1)),
                ),
            ),
            replace=False,
        )
        trajectory.apply_index(random_sampling)
        trajectory.tstamps += np.random.rand(len(trajectory)) * np.random.rand() * 10
        trajectory.apply_index(np.argsort(trajectory.tstamps))
        return trajectory

    def check_trajectory_attribute(self, attribute: Any, target_length: int, target_type: Any) -> None:
        if target_length == 0:
            self.assertTrue(isinstance(attribute, target_type))
        else:
            self.assertTrue(target_length == len(attribute) and isinstance(attribute, target_type))

    def trajectory_sanity_check(self, trajectory: Trajectory) -> None:
        target_length = len(trajectory)
        self.check_trajectory_attribute(trajectory.tstamps, target_length=target_length, target_type=np.ndarray)
        self.check_trajectory_attribute(trajectory.pos, target_length=target_length, target_type=PointSet)
        self.check_trajectory_attribute(trajectory.rot, target_length=target_length, target_type=(RotationSet, None))
        self.check_trajectory_attribute(trajectory.speed, target_length=target_length, target_type=np.ndarray)
        self.check_trajectory_attribute(trajectory.speed_3d, target_length=target_length, target_type=np.ndarray)
        self.check_trajectory_attribute(trajectory.arc_lengths, target_length=target_length, target_type=np.ndarray)
        self.check_trajectory_attribute(trajectory.sort_index, target_length=target_length, target_type=np.ndarray)
        self.check_trajectory_attribute(
            trajectory.sort_switching_index,
            target_length=target_length,
            target_type=np.ndarray,
        )
        self.check_trajectory_attribute(trajectory.idx_chrono, target_length=target_length, target_type=np.ndarray)
        self.check_trajectory_attribute(trajectory.function_of, target_length=target_length, target_type=np.ndarray)
        self.check_trajectory_attribute(trajectory.se3, target_length=target_length, target_type=list)
        self.check_trajectory_attribute(trajectory.data_rate, target_length=0, target_type=float)
        self.check_trajectory_attribute(trajectory.arc_length, target_length=0, target_type=float)
        self.check_trajectory_attribute(trajectory.all, target_length=target_length, target_type=np.ndarray)
