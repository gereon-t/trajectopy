"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import copy
import logging
from typing import Tuple

import numpy as np
from pointset import PointSet
from scipy.spatial import KDTree

from trajectopy.core.settings.matching import MatchingMethod, MatchingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.core.utils import Line3D

logger = logging.getLogger("root")


def match_trajectories(
    traj_from: Trajectory,
    traj_to: Trajectory,
    settings: MatchingSettings = MatchingSettings(),
    inplace: bool = True,
) -> Tuple[Trajectory, Trajectory]:
    """
    Matches two trajectories using the specified method

    Args:
        traj_from (Trajectory): Trajectory to match
        traj_to (Trajectory): Reference trajectory
        settings (MatchingSettings, optional): Matching settings. Defaults to MatchingSettings().
        inplace (bool, optional): Whether to modify the input trajectories. Defaults to True.

    Supported methods:

    - MatchingMethod.INTERPOLATION: Interpolates the test trajectory onto the reference trajectory using its timestamps
    - MatchingMethod.NEAREST_TEMPORAL: Find the nearest temporal match without interpolation
    - MatchingMethod.NEAREST_SPATIAL: Find the nearest spatial match without interpolation
    - MatchingMethod.NEAREST_SPATIAL_INTERPOLATED: Find the nearest n spatial matches and spatially interpolate using a 3d line

    Nearest Spatial

    This method matches two trajectories by finding the nearest pose in the target trajectory
    for each pose in the source trajectory. The distance between two poses is computed using
    the Euclidean distance between their positions.

    Nearest Temporal

    This method matches two trajectories using their timestamps by finding the nearest
    timestamp in the target trajectory for each timestamp in the source trajectory.

    Interpolation

    This method matches two trajectories by interpolating the timestamps of one trajectory
    to the timestamps of the other trajectory. The interpolation is linear for both positions
    and rotations (SLERP).

    Nearest Spatial Interpolated

    This method matches both trajectories spatially by requesting the nearest k positions
    from the reference trajectory for each pose in the test trajectory. Then, an interpolation
    is performed using a 3d line fit of the k nearest positions. After this operation, both
    trajectories will have the length of the test trajectory. This method does not support
    rotation matching.

    Returns:
        Tuple[Trajectory, Trajectory]: Matched trajectories

    """
    traj_from = traj_from if inplace else traj_from.copy()
    traj_to = traj_to if inplace else traj_to.copy()

    logger.info("Matching trajectories using method %s", settings.method.name)

    if settings.method == MatchingMethod.INTERPOLATION:
        return _match_trajectories_interpolation(traj_test=traj_from, traj_ref=traj_to)

    if settings.method == MatchingMethod.NEAREST_TEMPORAL:
        return _match_trajectories_temporal(traj_test=traj_from, traj_ref=traj_to, max_distance=settings.max_time_diff)

    if settings.method == MatchingMethod.NEAREST_SPATIAL:
        return _match_trajectories_spatial(traj_test=traj_from, traj_ref=traj_to, max_distance=settings.max_distance)

    if settings.method == MatchingMethod.NEAREST_SPATIAL_INTERPOLATED:
        return _match_trajectories_spatial_interpolation(
            traj_test=traj_from,
            traj_ref=traj_to,
            max_distance=settings.max_distance,
            k_nearest=settings.k_nearest,
        )

    raise ValueError(f"Matching method {settings.method} not supported!")


def _match_trajectories_interpolation(traj_test: Trajectory, traj_ref: Trajectory) -> Tuple[Trajectory, Trajectory]:
    """Ensures that both trajectories are sampled in the same way

    This method will intersect both trajectories with each other
    and then approximate the trajectory with the higher data rate
    onto the other trajectory. The sorting and the arc lengths of
    both trajectories are identical after the call of this method.

    Args:
        traj_test (Trajectory): Test trajectory
        traj_ref (Trajectory): Reference trajectory

    Returns:
        Tuple[Trajectory, Trajectory]: Both trajectories with the
                                        same sampling. The instance
                                        which called this method is
                                        the first returned trajectory.
    """
    traj_test.intersect(traj_ref.tstamps)
    traj_ref.intersect(traj_test.tstamps)

    traj_test.interpolate(traj_ref.tstamps)
    traj_test.arc_lengths = copy.deepcopy(traj_ref.arc_lengths)

    return traj_test, traj_ref


def _match_trajectories_temporal(
    traj_test: Trajectory, traj_ref: Trajectory, max_distance: float = 0.01
) -> Tuple[Trajectory, Trajectory]:
    """This method matches both trajectories temporally

    After this operation, both trajectories will have the length of the
    test trajectory. This means, that the reference trajectory may be
    modified.

    Args:
        traj_test (Trajectory): Test trajectory
        traj_ref (Trajectory): Reference trajectory
        max_distance (float, optional): Maximum distance between two timestamps.
                                        Defaults to 0.1.

    Returns:
        Tuple[Trajectory, Trajectory]: Matched trajectories
    """
    tstamps_ref_2d = np.c_[traj_ref.tstamps, np.zeros(traj_ref.tstamps.shape)]
    tstamps_test_2d = np.c_[traj_test.tstamps, np.zeros(traj_test.tstamps.shape)]
    ref_indices, test_indices = _kd_matcher(ref=tstamps_ref_2d, test=tstamps_test_2d, max_distance=max_distance)
    logger.info("Found %i temporal matches", len(ref_indices))
    return traj_test.apply_index(test_indices), traj_ref.apply_index(ref_indices)


def _match_trajectories_spatial(
    traj_test: Trajectory, traj_ref: Trajectory, max_distance: float = 0.0
) -> Tuple[Trajectory, Trajectory]:
    """This method matches both trajectories spatially

    After this operation, both trajectories will have the length of the
    test trajectory. This means, that the reference trajectory may be
    modified.

    Args:
        traj_from (Trajectory): Test trajectory
        traj_to (Trajectory): Reference trajectory
        max_distance (float, optional): Maximum distance between two poses.
                                        Defaults to None. This means all
                                        matches are accepted.

    Returns:
        Tuple[Trajectory, Trajectory]: Matched trajectories
    """
    ref_indices, test_indices = _kd_matcher(ref=traj_ref.pos.xyz, test=traj_test.pos.xyz, max_distance=max_distance)
    logger.info("Found %i spatial matches", len(ref_indices))
    return traj_test.apply_index(test_indices), traj_ref.apply_index(ref_indices)


def _match_trajectories_spatial_interpolation(
    traj_test: Trajectory, traj_ref: Trajectory, max_distance: float = 0.0, k_nearest: int = 10
) -> Tuple[Trajectory, Trajectory]:
    """This method matches both trajectories spatially by requesting
    the nearest two poses from the reference trajectory for each pose in the
    test trajectory. Then, an interpolation is performed between the two
    nearest poses.

    After this operation, both trajectories will have the length of the
    test trajectory. This means, that the reference trajectory may be
    modified.

    Args:
        traj_from (Trajectory): Test trajectory
        traj_to (Trajectory): Reference trajectory
        max_distance (float, optional): Maximum distance between two poses.
                                        Defaults to None. This means all
                                        matches are accepted.
        k_nearest (int, optional): Number of nearest poses to request from
                                   the reference trajectory. Defaults to 10.

    Returns:
        Tuple[Trajectory, Trajectory]: Matched trajectories
    """
    test_xyz = traj_test.pos.xyz
    ref_xyz = traj_ref.pos.xyz

    if max_distance == 0:
        distances, closest_indices = KDTree(ref_xyz).query(test_xyz, k=k_nearest, workers=-1)
    else:
        distances, closest_indices = KDTree(ref_xyz).query(
            test_xyz, k=k_nearest, workers=-1, distance_upper_bound=max_distance
        )

    if np.isinf(distances).all():
        raise ValueError("No matches found!")

    matched_ref_pos = []
    matched_test_pos = []
    for i, (dists, idxs) in enumerate(zip(distances, closest_indices)):
        if any(np.isinf(dists)):
            continue

        test_pos = test_xyz[i, :]

        fit_line = Line3D.from_points(ref_xyz[idxs, :])
        line_point = fit_line.evaluate_at(test_pos)

        matched_test_pos.append(test_pos)
        matched_ref_pos.append(line_point)

    traj_test = Trajectory(
        name=traj_test.name,
        pos=PointSet(np.array(matched_test_pos), local_transformer=traj_test.pos.local_transformer),
    )
    traj_ref = Trajectory(
        name=traj_ref.name,
        pos=PointSet(np.array(matched_ref_pos), local_transformer=traj_ref.pos.local_transformer),
    )

    return traj_test, traj_ref


def _kd_matcher(ref: np.ndarray, test: np.ndarray, max_distance: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """This method matches data using a KDTree

    Args:
        ref (np.ndarray): Reference data
        test (np.ndarray): Test data
        max_distance (float): Maximum distance for a match

    Returns:
        Tuple[np.ndarray, np.ndarray]: Matched indices
    """
    if max_distance == 0:
        distances, closest_indices = KDTree(ref).query(test, k=1, workers=-1)
    else:
        distances, closest_indices = KDTree(ref).query(test, k=1, workers=-1, distance_upper_bound=max_distance)

    logger.info("Mean matching distance %.3f", np.mean(distances))

    distance_index_matrix = np.array([[not np.isinf(dist), index] for index, dist in enumerate(distances)], dtype=int)
    distance_filter = distance_index_matrix[:, 0].astype(bool)

    if not distance_filter.any():
        raise ValueError("No matches found!")

    _, unique_indices = np.unique(closest_indices[distance_filter], return_index=True)

    return (
        closest_indices[distance_filter][unique_indices],
        distance_index_matrix[distance_filter, 1][unique_indices],
    )


def rough_timestamp_matching(traj_ref: Trajectory, traj_test: Trajectory, max_distance: float = 0.0) -> float:
    """This method roughly matches two trajectories temporally
    Args:
        traj_from (Trajectory): Test trajectory
        traj_to (Trajectory): Reference trajectory

    Returns:
        float: Mean time offset
    """
    traj_test, traj_ref = _match_trajectories_spatial(
        traj_test=traj_test.copy(), traj_ref=traj_ref.copy(), max_distance=max_distance
    )
    mean_time_offset = np.median(traj_ref.tstamps - traj_test.tstamps)
    logger.info("Median time offset: %.3f s", mean_time_offset)
    return mean_time_offset
