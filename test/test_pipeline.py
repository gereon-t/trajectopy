import unittest

from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.evaluation import ate, rpe


class TestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.gt = Trajectory.from_file("example_data/KITTI_gt.traj")
        self.est = Trajectory.from_file("example_data/KITTI_ORB.traj")

    def test_ate(self) -> None:
        result = ate(other=self.gt, trajectory=self.est)

        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.abs_dev)
        self.assertIsNotNone(result.abs_dev.pos_dev)

        # Verify meaningful values
        self.assertGreater(len(result.abs_dev.pos_dev), 0)
        self.assertGreater(result.pos_ate, 0)
        self.assertIsInstance(result.pos_bias_x, float)
        self.assertIsInstance(result.pos_bias_y, float)
        self.assertIsInstance(result.pos_bias_z, float)
        self.assertIsInstance(result.pos_dev_rms, float)
        self.assertGreater(result.pos_dev_rms, 0)

    def test_rpe(self) -> None:
        result = rpe(other=self.gt, trajectory=self.est)

        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.rpe_dev)
        self.assertIsNotNone(result.rpe_dev.pos_dev)

        # Verify meaningful values
        self.assertGreater(len(result), 0)  # Uses __len__ method
        self.assertGreater(len(result.rpe_dev.pos_dev), 0)
        self.assertIn(result.pair_distance_unit, ["m", "s"])
