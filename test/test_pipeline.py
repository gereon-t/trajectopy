import unittest

from trajectopy.core.evaluation.metrics import ate, rpe
from trajectopy.core.trajectory import Trajectory


class TestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.gt = Trajectory.from_file("example_data/KITTI_gt.traj")
        self.est = Trajectory.from_file("example_data/KITTI_ORB.traj")

    def test_ate(self) -> None:
        ate(trajectory_gt=self.gt, trajectory_est=self.est)

    def test_rpe(self) -> None:
        rpe(trajectory_gt=self.gt, trajectory_est=self.est)
