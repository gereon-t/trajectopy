import unittest
from test.testdata import noisy_trajectory as test_trajectory

import numpy as np

from trajectopy_core.approximation.mls_approximation import mls_iterative
from trajectopy_core.settings.sorting_settings import SortingSettings
from trajectopy_core.util.spatialsorter import SpatialSorter


class TestSorting(unittest.TestCase):
    def test_sorting(self) -> None:
        trajectory = test_trajectory.copy()
        unsort_index = np.arange(len(trajectory))
        np.random.shuffle(unsort_index)

        trajectory.apply_index(unsort_index)

        settings = SortingSettings(discard_missing=True, voxel_size=0.5, k_nearest=10, movement_threshold=0.1)
        mls_approx = mls_iterative(
            xyz=trajectory.pos.xyz,
            voxel_size=settings.voxel_size,
            k_nearest=settings.k_nearest,
            movement_threshold=settings.movement_threshold,
        )
        sorter = SpatialSorter(xyz=mls_approx, discard_missing=settings.discard_missing)
        trajectory.apply_index(sorter.idx_sort)

        # the arc lengths must be monotonically increasing or decreasing. 1 wrap around is allowed.
        self.assertTrue(
            len(np.diff(trajectory.arc_lengths) >= 0) >= (len(trajectory.arc_lengths) - 1)
            or len(np.diff(trajectory.arc_lengths) <= 0) >= (len(trajectory.arc_lengths) - 1)
        )
