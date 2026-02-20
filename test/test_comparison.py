import unittest
from test.testdata import generated_trajectory
from test.util import random_number

import numpy as np

from trajectopy.core.settings import (
    MatchingMethod,
    MatchingSettings,
    RelativeComparisonSettings,
)
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.alignment import apply_alignment
from trajectopy.processing.evaluation import (
    _compare_trajectories_absolute,
    _compare_trajectories_relative,
    _derive_dev_directions_no_rot,
)
from trajectopy.processing.lib.alignment.parameters import (
    AlignmentParameters,
    Parameter,
)
from trajectopy.processing.matching import match_trajectories
from trajectopy.results.alignment_result import AlignmentResult
from trajectopy.results.ate_result import ATEResult
from trajectopy.results.rpe_result import RPEResult
from trajectopy.utils.definitions import Unit


def compare_trajectories_abs(traj_ref: Trajectory, traj_test: Trajectory) -> ATEResult:
    matching_settings = MatchingSettings(method=MatchingMethod.NEAREST_TEMPORAL)

    traj_test, traj_ref = match_trajectories(trajectory=traj_test, other=traj_ref, matching_settings=matching_settings)

    if len(traj_ref) != len(traj_test):
        raise ValueError("Something went wrong during matching.")

    return _compare_trajectories_absolute(trajectory=traj_test, other=traj_ref)


def compare_trajectories_rel(
    traj_ref: Trajectory, traj_test: Trajectory, settings: RelativeComparisonSettings
) -> RPEResult:
    matching_settings = MatchingSettings(method=MatchingMethod.NEAREST_TEMPORAL)

    traj_test, traj_ref = match_trajectories(trajectory=traj_test, other=traj_ref, matching_settings=matching_settings)

    if len(traj_ref) != len(traj_test):
        raise ValueError("Something went wrong during matching.")

    return _compare_trajectories_relative(trajectory=traj_test, other=traj_ref, settings=settings)


class TestComparison(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(7)

    def test_body_frame_deviations(self) -> None:
        trajectory = generated_trajectory.copy()

        parameters = AlignmentParameters(
            lever_x=Parameter(value=random_number(lower_bound=-1, upper_bound=1), unit=Unit.METER),
            lever_y=Parameter(value=random_number(lower_bound=-1, upper_bound=1), unit=Unit.METER),
            lever_z=Parameter(value=random_number(lower_bound=-1, upper_bound=1), unit=Unit.METER),
        )
        transformed = apply_alignment(
            generated_trajectory,
            alignment_result=AlignmentResult(position_parameters=parameters),
            inplace=False,
        )

        deviations = compare_trajectories_abs(traj_ref=trajectory, traj_test=transformed)

        np.testing.assert_almost_equal(deviations.pos_bias_along, parameters.lever_x.value)
        np.testing.assert_almost_equal(deviations.pos_bias_cross_h, parameters.lever_y.value)
        np.testing.assert_almost_equal(deviations.pos_bias_cross_v, parameters.lever_z.value)

    def test_xyz_deviations(self) -> None:
        trajectory = generated_trajectory.copy()

        parameters = AlignmentParameters(
            sim_trans_x=Parameter(value=random_number(lower_bound=-1, upper_bound=1), unit=Unit.METER),
            sim_trans_y=Parameter(value=random_number(lower_bound=-1, upper_bound=1), unit=Unit.METER),
            sim_trans_z=Parameter(value=random_number(lower_bound=-1, upper_bound=1), unit=Unit.METER),
        )
        transformed = apply_alignment(
            generated_trajectory,
            alignment_result=AlignmentResult(position_parameters=parameters),
            inplace=False,
        )

        deviations = compare_trajectories_abs(traj_ref=trajectory, traj_test=transformed)

        np.testing.assert_almost_equal(-deviations.pos_bias_x, parameters.sim_trans_x.value)
        np.testing.assert_almost_equal(-deviations.pos_bias_y, parameters.sim_trans_y.value)
        np.testing.assert_almost_equal(-deviations.pos_bias_z, parameters.sim_trans_z.value)

    def test_cross_track_deviations_no_rot_no_slope(self) -> None:
        # reference trajectory positions in direction 1,0 without slope
        ref_trajectory_xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
        deviations = np.array(
            [
                [0.1, 0.1, 1.0],
                [-0.1, -0.1, -1.0],
                [0.2, 0.1, 1.0],
                [0.05, -0.1, -1.0],
                [0.3, 0.1, 1.0],
            ]
        )
        test_trajectory_xyz = ref_trajectory_xyz + deviations
        dir_devs = _derive_dev_directions_no_rot(xyz_ref=ref_trajectory_xyz, xyz_test=test_trajectory_xyz)
        np.testing.assert_almost_equal(dir_devs[:, 0], deviations[:, 0])  # along-track
        np.testing.assert_almost_equal(dir_devs[:, 1], deviations[:, 1])  # cross-track horizontal
        np.testing.assert_almost_equal(dir_devs[:, 2], deviations[:, 2])  # cross-track vertical

    def test_cross_track_deviations_no_rot_slope(self) -> None:
        # reference trajectory positions in direction 1,0 without slope
        ref_trajectory_xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 2.0],
                [3.0, 0.0, 3.0],
                [4.0, 0.0, 4.0],
            ]
        )
        deviations = np.array(
            [
                [0.0, 0.1, 0],
                [0.1, 0.2, 0],
                [-0.1, 0.3, 0],
                [0.5, 0.4, 0],
                [-0.5, 1, 0],
            ]
        )
        test_trajectory_xyz = ref_trajectory_xyz + deviations
        dir_devs = _derive_dev_directions_no_rot(xyz_ref=ref_trajectory_xyz, xyz_test=test_trajectory_xyz)

        # with a slope of 45 degrees, the deviations are split equally into vertical and along-track components
        np.testing.assert_almost_equal(dir_devs[:, 0], -dir_devs[:, 2])

        # horizontal cross-track deviations remain the same
        np.testing.assert_almost_equal(dir_devs[:, 1], deviations[:, 1])

        # in total, the deviations are the same as the input deviations
        np.testing.assert_almost_equal(np.linalg.norm(dir_devs, axis=1), np.linalg.norm(deviations, axis=1))

    def test_cross_track_deviations_no_rot_rotated(self) -> None:
        # reference trajectory positions in direction 1,0 without slope
        ref_trajectory_xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 3.0, 0.0],
                [4.0, 4.0, 0.0],
            ]
        )
        deviations = np.array(
            [
                [0.1, 0.1, 0],
                [-0.1, -0.1, 0],
                [0.0, 0.0, 0],
                [0.5, 0.0, 0],
                [0.0, 0.5, 0],
            ]
        )
        test_trajectory_xyz = ref_trajectory_xyz + deviations
        dir_devs = _derive_dev_directions_no_rot(xyz_ref=ref_trajectory_xyz, xyz_test=test_trajectory_xyz)
        np.testing.assert_almost_equal(np.linalg.norm(dir_devs, axis=1), np.linalg.norm(deviations, axis=1))

    def test_cross_track_deviations_no_rot_random(self) -> None:
        # reference trajectory positions in direction 1,0 without slope
        ref_trajectory_xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 3.0, 0.0],
                [4.0, 4.0, 0.0],
            ]
        )
        deviations = np.random.uniform(low=-1.0, high=1.0, size=(5, 3))
        test_trajectory_xyz = ref_trajectory_xyz + deviations
        dir_devs = _derive_dev_directions_no_rot(xyz_ref=ref_trajectory_xyz, xyz_test=test_trajectory_xyz)
        np.testing.assert_almost_equal(np.linalg.norm(dir_devs, axis=1), np.linalg.norm(deviations, axis=1))
