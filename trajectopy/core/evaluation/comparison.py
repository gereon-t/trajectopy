"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np

from trajectopy.core.evaluation.ate_result import ATEResult
from trajectopy.core.evaluation.deviations import AbsoluteTrajectoryDeviations, RelativeTrajectoryDeviations
from trajectopy.core.evaluation.rpe_result import RPEResult
from trajectopy.core.evaluation.utils import nearest_point
from trajectopy.core.rotationset import RotationSet
from trajectopy.core.settings.comparison import PairDistanceUnit, RelativeComparisonSettings
from trajectopy.core.trajectory import Trajectory

logger = logging.getLogger("root")


def compare_trajectories_absolute(*, traj_test: Trajectory, traj_ref: Trajectory) -> ATEResult:
    """
    Compares two trajectories in absolute terms, returning the deviations between them.

    Args:
        traj_test (Trajectory): The trajectory to be tested.
        traj_ref (Trajectory): The reference trajectory.

    Returns:
        ATEResult: An object containing the absolute deviations between the two trajectories.
    """
    logger.info("Performing absolute comparison")
    pos_dev = traj_ref.pos.xyz - traj_test.pos.xyz
    directed_pos_dev = get_directed_deviations(
        xyz_ref=traj_ref.pos.xyz,
        xyz_test=traj_test.pos.xyz,
        rot=traj_ref.rot or traj_test.rot,
    )

    if traj_ref.rot is not None and traj_test.rot is not None:
        rot_dev = traj_ref.rot - traj_test.rot
    else:
        rot_dev = None

    abs_dev = AbsoluteTrajectoryDeviations(
        pos_dev=pos_dev,
        directed_pos_dev=directed_pos_dev,
        rot_dev=rot_dev,
        rotations_used=(traj_ref.rot or traj_test.rot) is not None,
    )

    return ATEResult(
        name=f"{traj_test.name} vs. {traj_ref.name}",
        trajectory=traj_test,
        abs_dev=abs_dev,
    )


def _get_pair_indices(distances: np.ndarray, settings: RelativeComparisonSettings, dist: float) -> np.ndarray:
    """Get indices of pose pairs for a given distance.

    Args:
        distances (np.ndarray): Array of distances. Either arc lengths or time stamps.
        settings (ComparisonSettings): Comparison settings.
        dist (float): Desired pose pair distance.

    Returns:
        np.ndarray: Array of pose pair indices.

    """
    if settings.use_all_pose_pairs:
        dist_steps = np.array(
            [[i, arc_length + dist] for i, arc_length in enumerate(distances) if (arc_length + dist) < distances[-1]],
            dtype=float,
        )
        if len(dist_steps) == 0:
            return np.array([])

        indices_end = np.searchsorted(distances, dist_steps[:, 1])
        return np.c_[dist_steps[:, 0].astype(int), indices_end]

    dist_steps = np.arange(distances[0], distances[-1], dist)

    if len(dist_steps) == 0:
        return np.array([])

    indices = np.searchsorted(distances, dist_steps)
    return np.c_[indices[:-1], indices[1:]]


def compare_trajectories_relative(
    *, traj_test: Trajectory, traj_ref: Trajectory, settings: RelativeComparisonSettings = RelativeComparisonSettings()
) -> RPEResult:
    """This function compares two trajectories using the relative comparison method."""
    logger.info("Performing relative comparison")
    if settings.pair_min_distance > settings.pair_max_distance:
        raise ValueError("Maximum pose distance must be larger than minimum pose distance")

    pair_dists = np.arange(
        settings.pair_min_distance,
        settings.pair_max_distance + settings.pair_distance_step,
        settings.pair_distance_step,
    )

    pos_dev: Dict[float, List[float]] = {}
    rot_dev: Dict[float, List[float]] = {}
    pair_distance: Dict[float, List[float]] = {}

    se3_ref = traj_ref.se3
    se3_test = traj_test.se3

    cnt = 0
    for pair_dist in pair_dists:
        pos_dev[pair_dist] = []
        rot_dev[pair_dist] = []
        pair_distance[pair_dist] = []

        if settings.pair_distance_unit == PairDistanceUnit.METER:
            pair_indices = _get_pair_indices(traj_ref.arc_lengths, settings, dist=pair_dist)

        elif settings.pair_distance_unit == PairDistanceUnit.SECOND:
            pair_indices = _get_pair_indices(traj_ref.tstamps, settings, dist=pair_dist)

        else:
            raise ValueError(f"Unknown unit {settings.pair_distance_unit}")

        if len(pair_indices) == 0:
            logger.warning("No pairs found for distance %.2f", pair_dist)
            continue

        for pair in pair_indices:
            if settings.pair_distance_unit == PairDistanceUnit.METER:
                distance = traj_ref.arc_lengths[pair[1]] - traj_ref.arc_lengths[pair[0]]
            else:
                distance = traj_ref.tstamps[pair[1]] - traj_ref.tstamps[pair[0]]

            if traj_ref.has_orientation and traj_test.has_orientation:
                pose_delta_gt = se3_inv(se3_ref[pair[0]]).dot(se3_ref[pair[1]])
                pose_delta_test = se3_inv(se3_test[pair[0]]).dot(se3_test[pair[1]])
                pose_error = se3_inv(pose_delta_gt).dot(pose_delta_test)

                r_err = rotation_error(pose_error)
                t_err = translation_error(pose_error)

                rot_dev[pair_dist].append(r_err / distance)
            else:
                ref_dist = np.linalg.norm(traj_ref.pos.xyz[pair[1]] - traj_ref.pos.xyz[pair[0]])
                test_dist = np.linalg.norm(traj_test.pos.xyz[pair[1]] - traj_test.pos.xyz[pair[0]])
                t_err = np.abs(ref_dist - test_dist)

            pos_dev[pair_dist].append(t_err / distance)
            pair_distance[pair_dist].append(distance)
            cnt += 1

    if cnt == 0:
        raise ValueError("No pairs found")

    rpe_dev = RelativeTrajectoryDeviations(
        pos_dev=pos_dev,
        rot_dev=rot_dev,
        pair_distance=pair_distance,
        pair_distance_unit=settings.pair_distance_unit,
    )

    return RPEResult(
        name=f"{traj_test.name} vs. {traj_ref.name}",
        rpe_dev=rpe_dev,
    )


def se3_inv(pose: np.ndarray) -> np.ndarray:
    """Invert SE3 pose"""
    rot_inv = pose[:3, :3].transpose()
    trans_inv = -rot_inv.dot(pose[:3, 3])

    se3 = np.eye(4)
    se3[:3, :3] = rot_inv
    se3[:3, 3] = trans_inv
    return se3


def rotation_error(pose_error: np.ndarray) -> float:
    """KITTI metric port"""
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))


def translation_error(pose_error: np.ndarray) -> float:
    """KITTI metric port"""
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def get_directed_deviations(
    *, xyz_ref: np.ndarray, xyz_test: np.ndarray, rot: Union[RotationSet, None] = None
) -> np.ndarray:
    if rot is None:
        return derive_dev_directions_no_rot(xyz_ref=xyz_ref, xyz_test=xyz_test)

    return derive_dev_directions_with_rot(xyz_ref=xyz_ref, xyz_test=xyz_test, rot=rot)


def derive_dev_directions_no_rot(*, xyz_ref: np.ndarray, xyz_test: np.ndarray) -> np.ndarray:
    """
    Function that computes along-track and cross-track deviations
    between two synchronized trajectories.

    By constructing a 3D line between the corresponding point in xyz_ref and
    its successor (predecessor for the last point) one can determine the
    cross- and along-track deviations for each point in xyz_test
    """
    N = len(xyz_test)

    # initialize zero arrays
    d_along = np.zeros((N, 1))
    d_cross_h = np.zeros((N, 1))
    d_cross_v = np.zeros((N, 1))

    for i in range(N):
        p_curr = xyz_ref[i, :]
        is_last = i == N - 1

        p_next = xyz_ref[i - 1, :] if is_last else xyz_ref[i + 1, :]
        line_pts = [p_next, p_curr] if is_last else [p_curr, p_next]

        d_cross_h[i], d_cross_v[i] = _cross_track_dev(p=xyz_test[i, :], line_pts=line_pts)
        d_along[i] = _along_track_dev(p=xyz_test[i, :], line_pts=line_pts, is_last=is_last)

    return np.c_[d_along, d_cross_h, d_cross_v]


def derive_dev_directions_with_rot(*, xyz_ref: np.ndarray, xyz_test: np.ndarray, rot: RotationSet) -> np.ndarray:
    """
    Function that computes the deviation between ref and single with
    respect to coordinate axes defined by rpy
    """
    devs = np.zeros((len(xyz_test), 3))

    rot_matrices = rot.as_matrix()

    for i in range(len(xyz_ref)):
        # transform to body system
        devs[i, :] = (xyz_test[i, :] - xyz_ref[i, :]) @ rot_matrices[i]

    return devs


def _along_track_dev(*, p: np.ndarray, line_pts: List, is_last: bool) -> float:
    """
    Helper function that computes the along track deviation
    """
    a = line_pts[0]
    b = line_pts[1]
    p_nearest, t = nearest_point(p=p, line_pts=line_pts)

    if not is_last:
        return np.sign(t) * np.linalg.norm(p_nearest - a)

    return float(
        -np.linalg.norm(p_nearest - b)
        if np.linalg.norm(b - a) > np.linalg.norm(p_nearest - a)
        else np.linalg.norm(p_nearest - b)
    )


def _cross_track_dev(*, p: np.ndarray, line_pts: List, z_slope_dist: bool = False) -> Tuple[float, float]:
    """
    Helper function that computes the cross track deviation
    """
    a = line_pts[0]
    b = line_pts[1]
    p_nearest, _ = nearest_point(p=p, line_pts=line_pts)

    # Determine the sign (left or right of the approximation).
    # It is important that the normal vector is always constructed
    # the same, i.e. is always aligned the same relative to the
    # 3d line. A construction with the help of the nearest point on the
    # line and the point p, is therefore out of the question.
    # The angle between the normal vector and the vector between a and p
    # is calculated indirectly. If the angle is smaller than 90°, the
    # point lies on one side of the straight line, with >90° the point
    # lies on the other side of the straight line. It is not necessary
    # to explicitly calculate the angle. The calculation of the scalar
    # product or the determination of its sign is sufficient.

    n = [b[1] - a[1], -(b[0] - a[0])]
    d = p[:2] - a[:2]
    d_sign = -np.sign(d @ n)
    diff = p_nearest - p
    d_cross_h = d_sign * np.sqrt(diff[0] ** 2 + diff[1] ** 2)
    z_diff = diff[2]
    if z_slope_dist:
        angle_z = np.arctan2(b[1] - a[1], b[0] - a[0])
        gamma = np.pi / 2 + angle_z
        rotm = np.array(
            [
                [np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1],
            ]
        )
        diff_rot = rotm @ p - rotm @ p_nearest
        d_cross_v = np.sign(z_diff) * np.sqrt(diff_rot[0] ** 2 + diff_rot[2] ** 2)
    else:
        d_cross_v = z_diff
    return d_cross_h, d_cross_v
