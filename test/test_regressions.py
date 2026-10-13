"""Regression tests for bugs found during the architecture cleanup."""

import unittest

import numpy as np

from test.testdata import generated_trajectory
from trajectopy.core.settings import MatchingMethod, MatchingSettings
from trajectopy.processing.matching import match_trajectories
from trajectopy.utils.definitions import Sorting, TrajectoriesSorting
from trajectopy.visualization.axes import get_sorting


class TestRegressions(unittest.TestCase):
    def test_get_sorting_with_generator(self):
        sortings = [Sorting.PATH_LENGTH, Sorting.TIME]
        self.assertEqual(get_sorting(s for s in sortings), TrajectoriesSorting.MIXED)
        self.assertEqual(get_sorting(iter([Sorting.TIME, Sorting.TIME])), TrajectoriesSorting.ALL_TIME)

    def test_spatial_interpolation_skips_poses_without_neighbours(self):
        ref = generated_trajectory.copy()
        test = generated_trajectory.copy()
        test.positions.xyz[::10] += 1000.0  # far away from the reference: no neighbours within max_distance

        settings = MatchingSettings(method=MatchingMethod.NEAREST_SPATIAL_INTERPOLATED, max_distance=1.0)
        matched_test, matched_ref = match_trajectories(test, ref, matching_settings=settings, inplace=False)

        self.assertEqual(len(matched_test), len(matched_ref))
        self.assertLess(len(matched_test), len(test))
        self.assertFalse(np.isnan(matched_ref.xyz).any())

    def test_remove_ate_above_keeps_deviations_consistent(self):
        from test.testdata import noisy_trajectory
        from trajectopy.processing.evaluation import ate

        result = ate(noisy_trajectory.copy(), generated_trajectory.copy())
        _ = result.rot_dev_x  # populate the cached rotation statistics
        result.remove_ate_above(float(np.median(result.pos_dev_comb)))

        self.assertEqual(len(result.pos_dev_comb), len(result.trajectory))
        self.assertEqual(len(result.rot_dev_x), len(result.trajectory))
        self.assertEqual(len(result.abs_dev.rot_dev), len(result.trajectory))

    def test_direct_helmert_on_planar_points_is_a_rotation(self):
        from trajectopy.core.rotations import Rotations
        from trajectopy.processing.lib.alignment.direct import direct_helmert_transformation

        xyz_from = generated_trajectory.xyz  # lies exactly in a plane
        rotation = Rotations.from_euler("z", 0.3).as_matrix()
        xyz_to = 1.4 * (rotation @ xyz_from.T).T + [5.0, -3.0, 1.0]

        helmert = direct_helmert_transformation(xyz_from, xyz_to)

        self.assertAlmostEqual(np.linalg.det(helmert.rotation_matrix), 1.0)
        np.testing.assert_allclose(helmert.rotation_matrix, rotation, atol=1e-9)
        np.testing.assert_allclose(helmert.apply_to(xyz_from), xyz_to, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
