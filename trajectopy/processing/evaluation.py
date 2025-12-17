import logging

import numpy as np

from trajectopy.core import settings
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.alignment import apply_alignment, estimate_alignment
from trajectopy.processing.matching import match_trajectories
from trajectopy.results.alignment_result import AlignmentResult
from trajectopy.results.ate_result import AbsoluteTrajectoryDeviations, ATEResult
from trajectopy.results.rpe_result import RelativeTrajectoryDeviations, RPEResult
from trajectopy.utils.common import nearest_point

logger = logging.getLogger(__name__)


def ate(
    trajectory: Trajectory,
    other: Trajectory,
    processing_settings: settings.ProcessingSettings = settings.ProcessingSettings(),
    return_alignment: bool = False,
    align: bool = True,
) -> ATEResult | tuple[ATEResult, AlignmentResult]:
    """Computes the absolute trajectory error (ATE) between two trajectories.

    The ATE is computed by first matching the estimated trajectory to the ground truth trajectory.
    Then, the alignment between the two trajectories is estimated. The estimated trajectory is
    aligned to the ground truth trajectory using the estimated alignment. Finally, the ATE is
    computed by comparing the aligned estimated trajectory to the ground truth trajectory.

    Args:
        trajectory (Trajectory): Trajectory to be evaluated.
        other (Trajectory): Ground truth trajectory.
        processing_settings (ProcessingSettings, optional): Processing settings.
        return_alignment (bool, optional): Whether to return the alignment result.
            Defaults to False.
        align (bool, optional): Whether to perform alignment. Defaults to True.

    Returns:
        ATEResult: Result of the ATE computation. If return_alignment is True, returns a tuple
            containing (ATEResult, AlignmentResult).
    """
    trajectory, other = match_trajectories(
        trajectory=trajectory, other=other, matching_settings=processing_settings.matching, inplace=False
    )

    if align:
        alignment = estimate_alignment(
            trajectory=trajectory,
            other=other,
            alignment_settings=processing_settings.alignment,
            matching_settings=processing_settings.matching,
        )
        trajectory_est_aligned = apply_alignment(trajectory, alignment_result=alignment, inplace=False)
    else:
        alignment = AlignmentResult()
        trajectory_est_aligned = trajectory

    return (
        (
            _compare_trajectories_absolute(other=other, trajectory=trajectory_est_aligned),
            alignment,
        )
        if return_alignment
        else _compare_trajectories_absolute(other=other, trajectory=trajectory_est_aligned)
    )


def rpe(
    trajectory: Trajectory,
    other: Trajectory,
    processing_settings: settings.ProcessingSettings = settings.ProcessingSettings(),
) -> RPEResult:
    """Computes the relative pose error (RPE) between two trajectories.

    The RPE is computed by comparing the relative poses between the estimated and ground truth
    trajectories. The pose distances are either defined in meters or in seconds depending on
    the settings.

    Args:
        trajectory (Trajectory): Trajectory to be evaluated.
        other (Trajectory): Ground truth trajectory.
        processing_settings (ProcessingSettings, optional): Processing settings.

    Returns:
        RPEResult: Result of the RPE computation.
    """
    trajectory, other = match_trajectories(
        trajectory=trajectory, other=other, matching_settings=processing_settings.matching, inplace=False
    )
    return _compare_trajectories_relative(
        trajectory=trajectory,
        other=other,
        relative_comparison_settings=processing_settings.relative_comparison,
    )


def _compare_trajectories_absolute(*, trajectory: Trajectory, other: Trajectory) -> ATEResult:
    """Compares two trajectories in absolute terms, returning the deviations between them.

    Args:
        trajectory (Trajectory): The trajectory to be tested.
        other (Trajectory): The reference trajectory.

    Returns:
        ATEResult: An object containing the absolute deviations between the two trajectories.
    """
    logger.info("Performing absolute comparison")
    pos_dev = other.positions.xyz - trajectory.positions.xyz
    directed_pos_dev = _get_directed_deviations(
        xyz_ref=other.positions.xyz,
        xyz_test=trajectory.positions.xyz,
        rot=other.rotations or trajectory.rotations,
    )

    if other.rotations is not None and trajectory.rotations is not None:
        rot_dev = other.rotations - trajectory.rotations
    else:
        rot_dev = None

    abs_dev = AbsoluteTrajectoryDeviations(
        pos_dev=pos_dev,
        directed_pos_dev=directed_pos_dev,
        rot_dev=rot_dev,
        rotations_used=(other.rotations or trajectory.rotations) is not None,
    )

    return ATEResult(
        name=f"{trajectory.name} vs. {other.name}",
        trajectory=trajectory,
        abs_dev=abs_dev,
    )


def _get_pair_indices(distances: np.ndarray, settings: settings.RelativeComparisonSettings, dist: float) -> np.ndarray:
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


def _compare_trajectories_relative(
    *,
    trajectory: Trajectory,
    other: Trajectory,
    relative_comparison_settings: settings.RelativeComparisonSettings = settings.RelativeComparisonSettings(),
) -> RPEResult:
    """This function compares two trajectories using the relative comparison method."""
    logger.info("Performing relative comparison")
    if relative_comparison_settings.pair_min_distance > relative_comparison_settings.pair_max_distance:
        raise ValueError("Maximum pose distance must be larger than minimum pose distance")

    pair_dists = np.arange(
        relative_comparison_settings.pair_min_distance,
        relative_comparison_settings.pair_max_distance + relative_comparison_settings.pair_distance_step,
        relative_comparison_settings.pair_distance_step,
    )

    pos_dev: dict[float, list[float]] = {}
    rot_dev: dict[float, list[float]] = {}
    pair_distance: dict[float, list[float]] = {}

    se3_other = other.se3
    se3_test = trajectory.se3

    cnt = 0
    for pair_dist in pair_dists:
        pos_dev[pair_dist] = []
        rot_dev[pair_dist] = []
        pair_distance[pair_dist] = []

        if relative_comparison_settings.pair_distance_unit == settings.PairDistanceUnit.METER:
            pair_indices = _get_pair_indices(other.path_lengths, relative_comparison_settings, dist=pair_dist)

        elif relative_comparison_settings.pair_distance_unit == settings.PairDistanceUnit.SECOND:
            pair_indices = _get_pair_indices(other.timestamps, relative_comparison_settings, dist=pair_dist)

        else:
            raise ValueError(f"Unknown unit {relative_comparison_settings.pair_distance_unit}")

        if len(pair_indices) == 0:
            logger.warning("No pairs found for distance %.2f", pair_dist)
            continue

        for pair in pair_indices:
            if relative_comparison_settings.pair_distance_unit == settings.PairDistanceUnit.METER:
                distance = other.path_lengths[pair[1]] - other.path_lengths[pair[0]]
            else:
                distance = other.timestamps[pair[1]] - other.timestamps[pair[0]]

            if other.has_orientation and trajectory.has_orientation:
                pose_delta_gt = _se3_inv(se3_other[pair[0]]).dot(se3_other[pair[1]])
                pose_delta_test = _se3_inv(se3_test[pair[0]]).dot(se3_test[pair[1]])
                pose_error = _se3_inv(pose_delta_gt).dot(pose_delta_test)

                r_err = _rotation_error(pose_error)
                t_err = _translation_error(pose_error)

                rot_dev[pair_dist].append(r_err / distance)
            else:
                ref_dist = np.linalg.norm(other.positions.xyz[pair[1]] - other.positions.xyz[pair[0]])
                test_dist = np.linalg.norm(trajectory.positions.xyz[pair[1]] - trajectory.positions.xyz[pair[0]])
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
        pair_distance_unit=relative_comparison_settings.pair_distance_unit,
    )

    return RPEResult(
        name=f"{trajectory.name} vs. {other.name}",
        rpe_dev=rpe_dev,
    )


def _se3_inv(pose: np.ndarray) -> np.ndarray:
    """Invert SE3 pose"""
    rot_inv = pose[:3, :3].transpose()
    trans_inv = -rot_inv.dot(pose[:3, 3])

    se3 = np.eye(4)
    se3[:3, :3] = rot_inv
    se3[:3, 3] = trans_inv
    return se3


def _rotation_error(pose_error: np.ndarray) -> float:
    """KITTI metric port"""
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))


def _translation_error(pose_error: np.ndarray) -> float:
    """KITTI metric port"""
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def _get_directed_deviations(*, xyz_ref: np.ndarray, xyz_test: np.ndarray, rot: Rotations | None = None) -> np.ndarray:
    if rot is None:
        return _derive_dev_directions_no_rot(xyz_ref=xyz_ref, xyz_test=xyz_test)

    return _derive_dev_directions_with_rot(xyz_ref=xyz_ref, xyz_test=xyz_test, rot=rot)


def _derive_dev_directions_no_rot(*, xyz_ref: np.ndarray, xyz_test: np.ndarray) -> np.ndarray:
    """
    Function that computes along-track and cross-track deviations
    between two synchronized trajectories.

    By constructing a 3D line between the corresponding point in xyz_ref and
    its successor (predecessor for the last point) one can determine the
    cross- and along-track deviations for each point in xyz_test

    Returns deviations with respect to the reference trajectory.
    --> Compared to the reference, how much does the test trajectory
    deviate along-track, cross-track horizontal, cross-track vertical
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


def _derive_dev_directions_with_rot(*, xyz_ref: np.ndarray, xyz_test: np.ndarray, rot: Rotations) -> np.ndarray:
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


def _along_track_dev(*, p: np.ndarray, line_pts: list, is_last: bool) -> float:
    """
    Helper function that computes the along track deviation
    """
    a = line_pts[0]
    b = line_pts[1]
    p_nearest, t = nearest_point(p=p, line_pts=line_pts)

    if is_last:
        return float(
            -np.linalg.norm(p_nearest - b)
            if np.linalg.norm(b - a) > np.linalg.norm(p_nearest - a)
            else np.linalg.norm(p_nearest - b)
        )

    return np.sign(t) * np.linalg.norm(p_nearest - a)


def _cross_track_dev(*, p: np.ndarray, line_pts: list) -> tuple[float, float]:
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

    # vector in cross-track direction
    diff = p_nearest - p
    z_diff = diff[2]

    cross_track_direction = np.array([b[1] - a[1], -(b[0] - a[0])])
    d_cross_h = (diff[:2] @ cross_track_direction[:2]) / np.linalg.norm(cross_track_direction[:2])

    line_direction = b - a
    xy_diff_in_line_direction = (diff[:2] @ line_direction[:2]) / np.linalg.norm(line_direction[:2])
    d_cross_v = -np.sign(z_diff) * np.sqrt(z_diff**2 + xy_diff_in_line_direction**2)

    return d_cross_h, d_cross_v
