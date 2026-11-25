import copy
import pickle
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import trajectopy as tpy


class TestRotationsApi(unittest.TestCase):
    def test_wraps_scipy_without_subclassing(self):
        rotations = tpy.Rotations.random(5, rng=0)
        self.assertNotIsInstance(rotations, Rotation)
        self.assertIsInstance(rotations.to_scipy(), Rotation)
        np.testing.assert_allclose(tpy.Rotations.from_scipy(rotations.to_scipy()).as_quat(), rotations.as_quat())

    def test_composition_and_indexing(self):
        a, b = tpy.Rotations.random(4, rng=1), tpy.Rotations.random(4, rng=2)
        np.testing.assert_allclose((a - b + b).as_matrix(), a.as_matrix(), atol=1e-12)
        self.assertEqual(len(a[1:3]), 2)
        self.assertEqual(len(a[0]), 1)
        self.assertTrue(a[0].single)
        np.testing.assert_allclose((a * a.inv()).rotangle, 0, atol=1e-12)

    def test_copy_and_pickle(self):
        rotations = tpy.Rotations.random(3, rng=3)
        for clone in (rotations.copy(), copy.deepcopy(rotations), pickle.loads(pickle.dumps(rotations))):
            np.testing.assert_allclose(clone.as_quat(), rotations.as_quat())

    def test_fast_euler_matches_scipy(self):
        angles = np.random.default_rng(4).uniform(-3, 3, size=(100, 3))
        for seq in ("xyz", "ZYX", "zx"):
            ours = tpy.Rotations.from_euler(seq, angles[:, : len(seq)]).as_matrix()
            np.testing.assert_allclose(ours, Rotation.from_euler(seq, angles[:, : len(seq)]).as_matrix(), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
