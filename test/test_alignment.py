import unittest
from test.testdata import open_loop_trajectory
from test.util import transform_randomly

import numpy as np

from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import (
    AlignmentEstimationSettings,
    AlignmentSettings,
    AlignmentStochastics,
    MatchingSettings,
)
from trajectopy.processing.alignment import estimate_alignment
from trajectopy.processing.lib.alignment.direct import align_rotations
from trajectopy.processing.lib.alignment.parameters import AlignmentParameters


class TestAlignment(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def _alignment_test(
        self, similarity_enabled: bool, time_shift_enabled: bool, lever_enabled: bool, lazy: bool = False
    ) -> None:
        transformed, groundtruth = transform_randomly(
            open_loop_trajectory,
            similarity_enabled=similarity_enabled,
            time_shift_enabled=time_shift_enabled,
            lever_enabled=lever_enabled,
        )

        alignment_result = estimate_alignment(
            trajectory=open_loop_trajectory.copy(),
            other=transformed,
            alignment_settings=AlignmentSettings(
                estimation_settings=AlignmentEstimationSettings.from_components(
                    similarity=similarity_enabled,
                    time_shift=time_shift_enabled,
                    leverarm=lever_enabled,
                    sensor_rotation=False,
                ),
                stochastics=AlignmentStochastics(),
            ),
            matching_settings=MatchingSettings(),
        )

        self._verify_alignment(target=groundtruth, estimation=alignment_result.position_parameters, lazy=lazy)

    def test_alignment_combinations(self):
        """Test various combinations of alignment parameters (similarity, time shift, lever arm)."""
        cases = [
            (True, False, False),
            (False, False, True),
            (False, True, False),
            (True, False, True),
            (True, True, False),
            (False, True, True),
            (True, True, True),
        ]
        for similarity, time_shift, lever in cases:
            with self.subTest(similarity=similarity, time_shift=time_shift, lever=lever):
                self._alignment_test(similarity_enabled=similarity, time_shift_enabled=time_shift, lever_enabled=lever)

    def test_sensor_alignment_loop(self):
        """Test that sensor rotation alignment can be applied and reversed correctly."""
        trajectory = open_loop_trajectory.copy()
        trajectory_rotated = trajectory.copy()
        random_rot = Rotations.from_euler(
            seq="xyz",
            angles=[np.random.randint(-360, 360), np.random.randint(-360, 360), np.random.randint(-360, 360)],
            degrees=True,
        )
        trajectory_rotated.rotations = random_rot * trajectory_rotated.rotations

        sensor_alignment = align_rotations(trajectory_rotated.rotations, trajectory.rotations)

        trajectory_rotated_back = trajectory_rotated.copy()

        # apply as in Trajectory.apply_alignment()
        trajectory_rotated_back.rotations = sensor_alignment.rotation_set * trajectory_rotated_back.rotations
        alignment_between_back_and_original = align_rotations(trajectory_rotated_back.rotations, trajectory.rotations)

        np.testing.assert_allclose(
            alignment_between_back_and_original.rotation_set.rotangle,
            0,
            atol=1e-8,
            rtol=1e-8,
        )

    def test_rotation_alignment(self):
        """Test alignment of rotation sets and verification of transformation accuracy."""
        rotations_1 = Rotations.from_euler(seq="xyz", angles=np.random.rand(100, 3) * 360, degrees=True)
        rotations_2 = (
            Rotations.from_euler(
                seq="xyz",
                angles=[np.random.randint(-360, 360), np.random.randint(-360, 360), np.random.randint(-360, 360)],
                degrees=True,
            )
            * rotations_1
        )

        sensor_rot_params = align_rotations(rot_from=rotations_2, rot_to=rotations_1)

        rotations_2_to_1 = sensor_rot_params.rotation_set * rotations_2
        rpy_trafo_diff = (rotations_2_to_1 - rotations_1).as_euler(seq="xyz", degrees=True)

        np.testing.assert_allclose(rpy_trafo_diff, np.zeros_like(rpy_trafo_diff), atol=1e-8, rtol=1e-8)

    def _verify_alignment(
        self, target: AlignmentParameters, estimation: AlignmentParameters, lazy: bool = False
    ) -> None:
        tolerance = 1e-1 if lazy else 1e-3
        np.testing.assert_allclose(estimation.helmert.translation, target.helmert.translation, atol=tolerance)
        np.testing.assert_allclose(target.helmert.rotation_set.rotangle, estimation.helmert.rotation_set.rotangle)
        np.testing.assert_allclose(estimation.helmert.scale.value, target.helmert.scale.value, atol=tolerance)
        np.testing.assert_allclose(estimation.time_shift.value, target.time_shift.value, atol=tolerance)
        np.testing.assert_allclose(estimation.leverarm.values, target.leverarm.values, atol=tolerance)


if __name__ == "__main__":
    unittest.main()
