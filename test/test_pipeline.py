import unittest

from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.evaluation import ate, rpe


class TestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.gt = Trajectory.from_file("example_data/KITTI_gt.traj")
        self.est = Trajectory.from_file("example_data/KITTI_ORB.traj")

    def test_ate(self) -> None:
        ate(other=self.gt, trajectory=self.est)

    def test_rpe(self) -> None:
        rpe(other=self.gt, trajectory=self.est)
