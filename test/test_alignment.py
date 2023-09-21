import unittest
from test.testdata import generated_trajectory
from typing import Tuple

import numpy as np

from trajectopy_core.alignment.parameters import AlignmentParameters, Parameter
from trajectopy_core.alignment.result import AlignmentResult
from trajectopy_core.alignment.rotation_alignment import align_rotations
from trajectopy_core.alignment.util import align_trajectories
from trajectopy_core.settings.alignment_settings import AlignmentEstimationSettings, AlignmentSettings, AlignmentStochastics
from trajectopy_core.settings.matching_settings import MatchingSettings
from trajectopy_core.trajectory import Trajectory
from trajectopy_core.util.definitions import Unit
from trajectopy_core.util.rotationset import RotationSet


def generate_transformation(
    similarity_enabled: bool = True,
    time_shift_enabled: bool = True,
    lever_enabled: bool = True,
):
    if similarity_enabled:
        sim_trans_x = Parameter(value=23, unit=Unit.METER)
        sim_trans_y = Parameter(value=-34, unit=Unit.METER)
        sim_trans_z = Parameter(value=12, unit=Unit.METER)
        sim_rot_x = Parameter(value=np.deg2rad(10), unit=Unit.RADIAN)
        sim_rot_y = Parameter(value=np.deg2rad(-5), unit=Unit.RADIAN)
        sim_rot_z = Parameter(value=np.deg2rad(45), unit=Unit.RADIAN)
        sim_scale = Parameter(value=1.2, unit=Unit.SCALE)
    else:
        sim_trans_x = Parameter(value=0, unit=Unit.METER)
        sim_trans_y = Parameter(value=0, unit=Unit.METER)
        sim_trans_z = Parameter(value=0, unit=Unit.METER)
        sim_rot_x = Parameter(value=0, unit=Unit.RADIAN)
        sim_rot_y = Parameter(value=0, unit=Unit.RADIAN)
        sim_rot_z = Parameter(value=0, unit=Unit.RADIAN)
        sim_scale = Parameter(value=1, unit=Unit.SCALE)

    if time_shift_enabled:
        time_shift = Parameter(value=2.23, unit=Unit.SECOND)
    else:
        time_shift = Parameter(value=0, unit=Unit.SECOND)

    if lever_enabled:
        lever_x = Parameter(value=4.5, unit=Unit.METER)
        lever_y = Parameter(value=0.1, unit=Unit.METER)
        lever_z = Parameter(value=-2.4, unit=Unit.METER)
    else:
        lever_x = Parameter(value=0, unit=Unit.METER)
        lever_y = Parameter(value=0, unit=Unit.METER)
        lever_z = Parameter(value=0, unit=Unit.METER)

    return AlignmentParameters(
        sim_trans_x=sim_trans_x,
        sim_trans_y=sim_trans_y,
        sim_trans_z=sim_trans_z,
        sim_rot_x=sim_rot_x,
        sim_rot_y=sim_rot_y,
        sim_rot_z=sim_rot_z,
        sim_scale=sim_scale,
        time_shift=time_shift,
        lever_x=lever_x,
        lever_y=lever_y,
        lever_z=lever_z,
    )


class TestAlignment(unittest.TestCase):
    def test_alignment_application(self):
        _, transformed, estimation_of, _ = self.setup_alignment_test(
            similarity_enabled=True, time_shift_enabled=True, lever_enabled=True
        )

        alignment_result = align_trajectories(
            traj_from=generated_trajectory.copy(),
            traj_to=transformed,
            alignment_settings=AlignmentSettings(estimation_of=estimation_of),
            matching_settings=MatchingSettings(),
        )
        aligned_trajectory = generated_trajectory.apply_alignment(alignment_result, inplace=False)

        np.testing.assert_allclose(transformed.pos.xyz, aligned_trajectory.pos.xyz, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            transformed.rot.rotangle,
            aligned_trajectory.rot.rotangle,
            atol=1e-3,
            rtol=1e-3,
        )

    def _alignment_test(
        self, similarity_enabled: bool, time_shift_enabled: bool, lever_enabled: bool, lazy: bool = False
    ) -> None:
        parameters, transformed, estimation_of, stochastics = self.setup_alignment_test(
            similarity_enabled, time_shift_enabled, lever_enabled
        )
        alignment_result = align_trajectories(
            traj_from=generated_trajectory.copy(),
            traj_to=transformed,
            alignment_settings=AlignmentSettings(estimation_of=estimation_of, stochastics=stochastics),
            matching_settings=MatchingSettings(),
        )

        print("Target:")
        print(parameters)
        print("Estimation:")
        print(alignment_result.position_parameters)
        self._verify_alignment(target=parameters, estimation=alignment_result.position_parameters, lazy=lazy)

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

    def test_rotation_alignment(self):
        rotations_1 = RotationSet.from_euler(seq="xyz", angles=np.random.rand(100, 3) * 360, degrees=True)
        rotations_2 = (
            RotationSet.from_euler(
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

    def setup_alignment_test(
        self, similarity_enabled, time_shift_enabled, lever_enabled
    ) -> Tuple[AlignmentParameters, Trajectory, AlignmentEstimationSettings, AlignmentStochastics]:
        parameters = generate_transformation(
            similarity_enabled=similarity_enabled,
            time_shift_enabled=time_shift_enabled,
            lever_enabled=lever_enabled,
        )
        print(parameters)

        transformed = generated_trajectory.apply_alignment(
            AlignmentResult(position_parameters=parameters), inplace=False
        )

        estimation_of = AlignmentEstimationSettings(
            helmert=similarity_enabled,
            trans_x=similarity_enabled,
            trans_y=similarity_enabled,
            trans_z=similarity_enabled,
            rot_x=similarity_enabled,
            rot_y=similarity_enabled,
            rot_z=similarity_enabled,
            scale=similarity_enabled,
            leverarm=lever_enabled,
            lever_x=lever_enabled,
            lever_y=lever_enabled,
            lever_z=lever_enabled,
            time_shift=time_shift_enabled,
            use_x_speed=time_shift_enabled,
            use_y_speed=time_shift_enabled,
            use_z_speed=time_shift_enabled,
        )

        stochastics = AlignmentStochastics()

        return parameters, transformed, estimation_of, stochastics

    def _verify_alignment(
        self, target: AlignmentParameters, estimation: AlignmentParameters, lazy: bool = False
    ) -> None:
        tolerance = 1e-1 if lazy else 1e-3
        np.testing.assert_allclose(estimation.helmert.translation, target.helmert.translation, atol=tolerance)
        np.testing.assert_allclose(
            target.helmert.rot_x.value % (2 * np.pi), estimation.helmert.rot_x.value % (2 * np.pi)
        )
        np.testing.assert_allclose(
            target.helmert.rot_y.value % (2 * np.pi), estimation.helmert.rot_y.value % (2 * np.pi)
        )
        np.testing.assert_allclose(
            target.helmert.rot_z.value % (2 * np.pi), estimation.helmert.rot_z.value % (2 * np.pi)
        )
        np.testing.assert_allclose(estimation.helmert.scale.value, target.helmert.scale.value, atol=tolerance)
        np.testing.assert_allclose(estimation.time_shift.value, target.time_shift.value, atol=tolerance)
        np.testing.assert_allclose(estimation.leverarm.values, target.leverarm.values, atol=tolerance)


if __name__ == "__main__":
    unittest.main()
