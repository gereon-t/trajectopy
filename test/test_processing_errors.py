import unittest
from test.testdata import open_loop_trajectory

import numpy as np

from trajectopy.core.settings import (
    AlignmentEstimationSettings,
    AlignmentSettings,
    AlignmentStochastics,
    MatchingSettings,
)
from trajectopy.processing.alignment import estimate_alignment


class TestProcessingErrors(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def test_estimate_alignment_handles_different_lengths(self):
        traj_a = open_loop_trajectory.copy()
        # Construct a shorter trajectory from arrays
        xyz_short = traj_a.positions.xyz[: max(2, len(traj_a) // 3)]
        traj_b = type(traj_a).from_arrays(xyz=xyz_short, epsg=traj_a.positions.epsg)
        result = estimate_alignment(
            trajectory=traj_a,
            other=traj_b,
            alignment_settings=AlignmentSettings(
                estimation_settings=AlignmentEstimationSettings.from_components(
                    similarity=True, time_shift=False, leverarm=False, sensor_rotation=False
                ),
                stochastics=AlignmentStochastics(),
            ),
            matching_settings=MatchingSettings(),
        )
        from trajectopy.results.alignment_result import AlignmentResult

        self.assertIsInstance(result, AlignmentResult)

    def test_estimate_alignment_invalid_settings(self):
        traj = open_loop_trajectory.copy()
        with self.assertRaises(AttributeError):
            estimate_alignment(
                trajectory=traj, other=traj, alignment_settings=None, matching_settings=MatchingSettings()
            )


if __name__ == "__main__":
    unittest.main()
