import unittest
from test.testdata import open_loop_trajectory
from test.util import transform_randomly

import numpy as np

from trajectopy.alignment.direct import align_rotations
from trajectopy.alignment.parameters import AlignmentParameters
from trajectopy.rotations import Rotations
from trajectopy.settings import (
    AlignmentEstimationSettings,
    AlignmentSettings,
    AlignmentStochastics,
    MatchingSettings,
)
from trajectopy.tools.alignment import estimate_alignment


class TestAlignment(unittest.TestCase):
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

        print("Target:")
        print(groundtruth)
        print("Estimation:")
        print(alignment_result.position_parameters)
        self._verify_alignment(target=groundtruth, estimation=alignment_result.position_parameters, lazy=lazy)

    def test_similarity_alignment(self):
        self._alignment_test(similarity_enabled=True, time_shift_enabled=False, lever_enabled=False)

    def test_lever_alignment(self):
        self._alignment_test(similarity_enabled=False, time_shift_enabled=False, lever_enabled=True)

    def test_time_shift_alignment(self):
        self._alignment_test(similarity_enabled=False, time_shift_enabled=True, lever_enabled=False)

    def test_similarity_lever_alignment(self):
        self._alignment_test(similarity_enabled=True, time_shift_enabled=False, lever_enabled=True)

    def test_similarity_time_shift_alignment(self):
        self._alignment_test(similarity_enabled=True, time_shift_enabled=True, lever_enabled=False)

    def test_lever_time_shift_alignment(self):
        self._alignment_test(similarity_enabled=False, time_shift_enabled=True, lever_enabled=True)

    def test_similarity_lever_time_shift_alignment(self):
        self._alignment_test(similarity_enabled=True, time_shift_enabled=True, lever_enabled=True)

    def test_sensor_alignment_loop(self):
        trajectory = open_loop_trajectory.copy()
        trajectory_rotated = trajectory.copy()
        random_rot = Rotations.from_euler(
            seq="xyz",
            angles=[np.random.randint(-360, 360), np.random.randint(-360, 360), np.random.randint(-360, 360)],
            degrees=True,
        )
        print(f"Random rotation: {random_rot.as_euler(seq='xyz', degrees=True)}")
        trajectory_rotated.rotations = random_rot * trajectory_rotated.rotations

        sensor_alignment = align_rotations(trajectory_rotated.rotations, trajectory.rotations)

        trajectory_rotated_back = trajectory_rotated.copy()

        # apply as in Trajectory.apply_alignment()
        trajectory_rotated_back.rotations = sensor_alignment.rotation_set * trajectory_rotated_back.rotations
        alignment_between_back_and_original = align_rotations(trajectory_rotated_back.rotations, trajectory.rotations)

        print(
            "Alignment between rotated back trajectory and original trajectory:"
            f"\n{alignment_between_back_and_original}"
        )
        np.testing.assert_allclose(
            alignment_between_back_and_original.rotation_set.rotangle,
            0,
            atol=1e-8,
            rtol=1e-8,
        )

    def test_rotation_alignment(self):
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
