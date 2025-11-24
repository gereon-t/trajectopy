import unittest

from trajectopy.tools.evaluation import ate, rpe
from trajectopy.trajectory import Trajectory


class TestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.gt = Trajectory.from_file("example_data/KITTI_gt.traj")
        self.est = Trajectory.from_file("example_data/KITTI_ORB.traj")

    def test_ate(self) -> None:
        ate(other=self.gt, trajectory=self.est)

    def test_rpe(self) -> None:
        rpe(other=self.gt, trajectory=self.est)
