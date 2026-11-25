import tempfile
import unittest
from pathlib import Path

import numpy as np

import trajectopy as tpy


def _trajectory(num: int = 10, name: str = "line") -> tpy.Trajectory:
    xyz = np.c_[np.arange(num, dtype=float), np.zeros(num), np.zeros(num)]
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (num, 1))
    return tpy.Trajectory(xyz, quat, timestamps=np.arange(num, dtype=float), name=name)


class TestTrajectoryApi(unittest.TestCase):
    def test_constructor_accepts_arrays(self):
        trajectory = _trajectory()
        self.assertEqual(len(trajectory), 10)
        self.assertTrue(trajectory.has_orientation)
        self.assertEqual(trajectory.positions.epsg, 0)

    def test_repr_is_one_line(self):
        text = repr(_trajectory())
        self.assertNotIn("\n", text)
        self.assertIn("10 poses", text)

    def test_integer_index_returns_pose(self):
        pose = _trajectory()[-1]
        self.assertIsInstance(pose, tpy.Pose)
        self.assertEqual(pose.position, (9.0, 0.0, 0.0))
        with self.assertRaises(IndexError):
            _trajectory()[10]

    def test_slices_and_masks_return_new_trajectories(self):
        trajectory = _trajectory()
        np.testing.assert_allclose(trajectory[2:5].timestamps, [2, 3, 4])
        np.testing.assert_allclose(trajectory[trajectory.timestamps > 7].timestamps, [8, 9])
        np.testing.assert_allclose(trajectory[[0, 9]].timestamps, [0, 9])
        self.assertEqual(len(trajectory), 10)

    def test_equals_ignores_name(self):
        self.assertTrue(_trajectory(name="a").equals(_trajectory(name="b")))
        self.assertNotEqual(_trajectory(name="a"), _trajectory(name="b"))
        self.assertEqual(_trajectory(), _trajectory())
        self.assertFalse(_trajectory().equals(_trajectory(num=9)))

    def test_path_like_and_string_round_trip(self):
        trajectory = _trajectory()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "line.traj"
            trajectory.to_file(path)
            self.assertTrue(tpy.Trajectory.from_file(path).equals(trajectory))
        self.assertTrue(tpy.Trajectory.from_string(trajectory.to_string()).equals(trajectory))


if __name__ == "__main__":
    unittest.main()
