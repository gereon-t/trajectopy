import unittest

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.exceptions import PointSetError


class TestPositions(unittest.TestCase):
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

    def test_positions_invalid_shape_raises(self):
        xyz = np.random.rand(10, 2)  # Should be (N, 3)
        with self.assertRaises((ValueError, PointSetError)):
            Positions(xyz=xyz)


if __name__ == "__main__":
    unittest.main()
