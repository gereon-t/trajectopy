import unittest

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.exceptions import PointSetError


class TestPositionsRotations(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(17)

    def test_positions_creation(self):
        xyz = np.random.rand(10, 3)
        pos = Positions(xyz=xyz, epsg=4326)
        self.assertEqual(len(pos), 10)
        self.assertEqual(pos.epsg, 4326)
        np.testing.assert_array_equal(pos.xyz, xyz)

    def test_positions_default_epsg(self):
        xyz = np.random.rand(10, 3)
        pos = Positions(xyz=xyz)
        self.assertEqual(pos.epsg, 0)

    def test_rotations_from_quat(self):
        quat = np.tile([0, 0, 0, 1], (10, 1))
        rot = Rotations.from_quat(quat)
        self.assertEqual(len(rot), 10)

    def test_rotations_from_euler(self):
        angles = np.zeros((10, 3))
        rot = Rotations.from_euler(seq="xyz", angles=angles, degrees=False)
        self.assertEqual(len(rot), 10)

    def test_rotations_from_matrix(self):
        # Identity rotation matrices
        matrices = np.tile(np.eye(3), (10, 1, 1))
        rot = Rotations.from_matrix(matrices)
        self.assertEqual(len(rot), 10)

    def test_rotations_as_quat(self):
        quat = np.tile([0, 0, 0, 1], (10, 1))
        rot = Rotations.from_quat(quat)
        result = rot.as_quat()
        np.testing.assert_allclose(result, quat)

    def test_rotations_as_euler(self):
        angles = np.zeros((10, 3))
        rot = Rotations.from_euler(seq="xyz", angles=angles, degrees=False)
        result = rot.as_euler(seq="xyz", degrees=False)
        np.testing.assert_allclose(result, angles, atol=1e-10)

    def test_rotations_multiplication(self):
        rot1 = Rotations.from_euler(seq="xyz", angles=np.array([[0, 0, np.pi / 2]]), degrees=False)
        rot2 = Rotations.from_euler(seq="xyz", angles=np.array([[0, 0, np.pi / 2]]), degrees=False)
        result = rot1 * rot2
        # Two 90-degree rotations should give 180 degrees
        angles = result.as_euler(seq="xyz", degrees=True)
        np.testing.assert_allclose(angles[0, 2], 180.0, atol=1e-10)

    def test_rotations_subtraction(self):
        rot1 = Rotations.from_euler(seq="xyz", angles=np.array([[0, 0, np.pi / 2]]), degrees=False)
        rot2 = Rotations.from_euler(seq="xyz", angles=np.array([[0, 0, np.pi / 4]]), degrees=False)
        result = rot1 - rot2
        angles = result.as_euler(seq="xyz", degrees=True)
        np.testing.assert_allclose(angles[0, 2], 45.0, atol=1e-10)

    def test_positions_invalid_shape_raises(self):
        xyz = np.random.rand(10, 2)  # Should be (N, 3)
        with self.assertRaises((ValueError, PointSetError)):
            Positions(xyz=xyz)

    def test_rotations_invalid_quat_shape_raises(self):
        """Test that invalid quaternion shape raises error."""
        quat_invalid = np.random.rand(10, 3)  # Should be (N, 4)
        with self.assertRaises(ValueError):
            Rotations.from_quat(quat_invalid)

    def test_rotations_mismatched_multiplication_raises(self):
        """Test that multiplication of rotations with different lengths raises error."""
        rot1 = Rotations.from_quat(np.tile([0, 0, 0, 1], (10, 1)))
        rot2 = Rotations.from_quat(np.tile([0, 0, 0, 1], (5, 1)))
        with self.assertRaises((ValueError, IndexError)):
            _ = rot1 * rot2


if __name__ == "__main__":
    unittest.main()
