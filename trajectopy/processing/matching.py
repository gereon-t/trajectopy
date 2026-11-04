import copy
import logging

import numpy as np
from scipy.spatial import KDTree

from trajectopy.core import settings
from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.interpolation import inside_gaps, interpolate
from trajectopy.processing.lib.approximation.line_approximation import Line3D
from trajectopy.utils.definitions import Sorting

logger = logging.getLogger(__name__)

# Number of (distinct) reference positions before and after the nearest one that define the local path direction
_TANGENT_HALF_WINDOW = 5


def match_non_overlapping_stamps(trajectory: Trajectory, other: Trajectory, max_distance: float = 0.0) -> float:
    """Roughly matches two trajectories.

    Args:
        trajectory (Trajectory): Trajectory to match.
        other (Trajectory): Other trajectory to match against.
        max_distance (float, optional): Maximum distance for spatial matching. Defaults to 0.0.

    Returns:
        float: Mean offset.
    """
    other, trajectory = _match_trajectories_spatial(
        trajectory=other.copy(), other=trajectory.copy(), max_distance=max_distance
    )
    mean_offset = np.median(trajectory.index - other.index)
    logger.info("Median index offset: %.3f", mean_offset)
    return mean_offset


def match_stamps(trajectory: Trajectory, index: np.ndarray, inplace: bool = True) -> Trajectory:
    """Truncates trajectory to only those poses where the indices exactly match "index".

    Args:
        trajectory (Trajectory): Input trajectory.
        index (np.ndarray): Input indices.
        inplace (bool, optional): Perform matching in-place. Defaults to True.

    Returns:
        Trajectory: Trajectory with matched indices.
    """
    traj_self = trajectory if inplace else trajectory.copy()
    _, idx_self, _ = np.intersect1d(traj_self.index, index, return_indices=True)
    traj_self.mask(idx_self)
    return traj_self


def match_trajectories(
    trajectory: Trajectory,
    other: Trajectory,
    matching_settings: settings.MatchingSettings | None = None,
    inplace: bool = True,
) -> tuple[Trajectory, Trajectory]:
    """Matches two trajectories using the specified method.

    Supported methods:

    - **INTERPOLATION**: Interpolates the test trajectory onto the reference trajectory using
      its timestamps. The interpolation is linear for both positions and rotations (SLERP).
    - **NEAREST_TEMPORAL**: Finds the nearest temporal match without interpolation by finding
      the nearest timestamp in the target trajectory for each timestamp in the source trajectory.
    - **NEAREST_SPATIAL**: Finds the nearest spatial match without interpolation by finding
      the nearest pose in the target trajectory for each pose in the source trajectory using
      Euclidean distance.
    - **NEAREST_SPATIAL_INTERPOLATED**: Finds the nearest k spatial matches and spatially
      interpolates using a 3d line. Both trajectories will have the length of the test trajectory.
      This method does not support rotation matching.

    Args:
        trajectory (Trajectory): Trajectory to match.
        other (Trajectory): Other trajectory to match against.
        matching_settings (MatchingSettings, optional): Matching settings. Defaults to
            MatchingSettings().
        inplace (bool, optional): Whether to modify the input trajectories. Defaults to True.

    Returns:
        Tuple[Trajectory, Trajectory]: Matched trajectories.
    """
    if matching_settings is None:
        matching_settings = settings.MatchingSettings()

    trajectory = trajectory if inplace else trajectory.copy()
    other = other if inplace else other.copy()

    if (
        trajectory.sorting == Sorting.PATH_LENGTH
        and other.sorting == Sorting.PATH_LENGTH
        and matching_settings.method
        in [settings.MatchingMethod.NEAREST_TEMPORAL, settings.MatchingMethod.INTERPOLATION]
        and not np.array_equal(trajectory.index, other.index)
    ):
        _adopt_reference_path_lengths(trajectory=trajectory, other=other)

    if not trajectory.overlaps_with(other) and matching_settings.method in [
        settings.MatchingMethod.NEAREST_TEMPORAL,
        settings.MatchingMethod.INTERPOLATION,
    ]:
        logger.warning("Trajectories do not overlap! Performing rough matching first.")
        timeshift = match_non_overlapping_stamps(
            trajectory=other, other=trajectory, max_distance=matching_settings.max_distance
        )
        logger.info("Rough matching index offset: %.3f", timeshift)

        if trajectory.sorting == Sorting.TIME:
            trajectory.timestamps = trajectory.timestamps + timeshift
        else:
            trajectory.path_lengths = trajectory.path_lengths + timeshift

    logger.info("Matching trajectories using method %s", matching_settings.method.name)

    if matching_settings.method == settings.MatchingMethod.INTERPOLATION:
        return _match_trajectories_interpolation(
            trajectory=trajectory, other=other, max_gap_size=matching_settings.max_gap_size
        )

    if matching_settings.method == settings.MatchingMethod.NEAREST_TEMPORAL:
        return _match_trajectories_temporal(
            trajectory=trajectory, other=other, max_distance=matching_settings.max_time_diff
        )

    if matching_settings.method == settings.MatchingMethod.NEAREST_SPATIAL:
        return _match_trajectories_spatial(
            trajectory=trajectory, other=other, max_distance=matching_settings.max_distance
        )

    if matching_settings.method == settings.MatchingMethod.NEAREST_SPATIAL_INTERPOLATED:
        return _match_trajectories_spatial_interpolation(
            trajectory=trajectory,
            other=other,
            max_distance=matching_settings.max_distance,
            k_nearest=matching_settings.k_nearest,
        )

    raise ValueError(f"Matching method {matching_settings.method} not supported!")


def _adopt_reference_path_lengths(trajectory: Trajectory, other: Trajectory) -> None:
    """Expresses the path lengths of ``trajectory`` in the path length frame of ``other``.

    The path lengths of a spatially sorted trajectory start at an individual position and may run in the
    opposite direction, so equal path lengths of two trajectories generally do not belong to the same place.
    Therefore, each position is projected onto the path of the reference: its path length is the path length
    of the nearest reference position plus the offset along the local path direction.

    Args:
        trajectory (Trajectory): Spatially sorted trajectory whose path lengths are replaced (in-place).
        other (Trajectory): Spatially sorted reference trajectory.
    """
    # distinct path lengths only, so that the window defining the local path direction has a length
    path_lengths, first_indices = np.unique(other.path_lengths, return_index=True)
    if len(path_lengths) < 2:
        return
    ref_xyz = other.positions.xyz[first_indices]

    xyz = trajectory.positions.xyz
    _, nearest = KDTree(ref_xyz).query(xyz, workers=-1)
    tangents = (
        ref_xyz[np.minimum(nearest + _TANGENT_HALF_WINDOW, len(ref_xyz) - 1)]
        - ref_xyz[np.maximum(nearest - _TANGENT_HALF_WINDOW, 0)]
    )
    norms = np.linalg.norm(tangents, axis=1)
    offsets = np.sum((xyz - ref_xyz[nearest]) * tangents, axis=1) / np.where(norms > 0, norms, 1.0)

    trajectory.path_lengths = np.clip(path_lengths[nearest] + offsets, path_lengths[0], path_lengths[-1])
    trajectory.mask(np.argsort(trajectory.path_lengths, kind="stable"))
    logger.info("Expressed the path lengths of %s in the path length frame of %s.", trajectory.name, other.name)


def _match_trajectories_interpolation(
    trajectory: Trajectory, other: Trajectory, max_gap_size: float = 10.0
) -> tuple[Trajectory, Trajectory]:
    """Ensures that both trajectories are sampled in the same way.

    This method will intersect both trajectories with each other and then approximate
    the trajectory with the higher data rate onto the other trajectory. The sorting and
    the arc lengths of both trajectories are identical after the call of this method.

    Args:
        trajectory (Trajectory): Trajectory to match.
        other (Trajectory): Other trajectory to match against.
        max_gap_size (float, optional): Maximum gap size in seconds. Defaults to 10.0.

    Returns:
        Tuple[Trajectory, Trajectory]: Both trajectories with the same sampling. The instance
            which called this method is the first returned trajectory.
    """
    if trajectory.sorting == other.sorting and np.array_equal(trajectory.index, other.index):
        logger.info("Trajectories are already matched, no interpolation required.")
        return trajectory, other

    trajectory.intersect(other.index, max_gap_size=max_gap_size)

    if trajectory.index.shape[0] == 0:
        raise ValueError(
            "Reference trajectory has no indices after intersection! Check your matching settings, especially max_gap_size."
        )

    other.intersect(trajectory.index, max_gap_size=max_gap_size)
    # the epochs of the reference are interpolated, which is not done inside gaps of the trajectory;
    # unlike the observations of the trajectory, the reference epochs at the gap edges are not needed there
    other.mask(~inside_gaps(other.index, trajectory.index, max_gap_size))

    if other.index.shape[0] == 0:
        raise ValueError(
            "Test trajectory has no indices after intersection! Check your matching settings, especially max_gap_size."
        )
    interpolate(trajectory, other.index)
    # the poses now belong to the epochs of the reference: for spatially sorted trajectories, interpolated
    # timestamps would mix laps that pass the same place
    trajectory.timestamps = copy.deepcopy(other.timestamps)
    trajectory.path_lengths = copy.deepcopy(other.path_lengths)

    return trajectory, other


def _match_trajectories_temporal(
    trajectory: Trajectory, other: Trajectory, max_distance: float = 0.01
) -> tuple[Trajectory, Trajectory]:
    """Matches both trajectories temporally.

    After this operation, both trajectories will have the length of the test trajectory.
    This means that the reference trajectory may be modified.

    Args:
        trajectory (Trajectory): Trajectory to match.
        other (Trajectory): Other trajectory to match against.
        max_distance (float, optional): Maximum distance between two timestamps.
            Defaults to 0.01.

    Returns:
        Tuple[Trajectory, Trajectory]: Matched trajectories.
    """
    tstamps_ref_2d = np.c_[other.index, np.zeros(other.index.shape)]
    tstamps_test_2d = np.c_[trajectory.index, np.zeros(trajectory.index.shape)]
    ref_indices, test_indices = _kd_matcher(ref=tstamps_ref_2d, test=tstamps_test_2d, max_distance=max_distance)
    logger.info("Found %i temporal matches", len(ref_indices))
    return trajectory.mask(test_indices), other.mask(ref_indices)


def _match_trajectories_spatial(
    trajectory: Trajectory, other: Trajectory, max_distance: float = 0.0
) -> tuple[Trajectory, Trajectory]:
    """Matches both trajectories spatially.

    After this operation, both trajectories will have the length of the test trajectory.
    This means that the reference trajectory may be modified.

    Args:
        trajectory (Trajectory): Trajectory to match.
        other (Trajectory): Other trajectory to match against.
        max_distance (float, optional): Maximum distance between two poses. Defaults to 0.0.
            This means all matches are accepted.

    Returns:
        Tuple[Trajectory, Trajectory]: Matched trajectories.
    """
    ref_indices, test_indices = _kd_matcher(
        ref=other.positions.xyz, test=trajectory.positions.xyz, max_distance=max_distance
    )
    logger.info("Found %i spatial matches", len(ref_indices))
    return trajectory.mask(test_indices), other.mask(ref_indices)


def _determine_mean_arc_length(arc_lengths: np.ndarray, lap_length: float) -> float:
    """Determines the mean arc length from a set of arc lengths while considering the circular nature of arc lengths.

    Args:
        arc_lengths (np.ndarray): Array of arc lengths.
        lap_length (float): Arc length of one lap, at which the arc lengths start over.

    Returns:
        float: Mean arc length.
    """
    if lap_length <= 0:
        return float(np.mean(arc_lengths))
    max_arc_length = lap_length
    # convert to angles
    angles = (arc_lengths / max_arc_length) * 2 * np.pi
    xy = np.c_[np.cos(angles), np.sin(angles)]
    # average in xy space
    mean_angle = np.arctan2(np.mean(xy[:, 1]), np.mean(xy[:, 0]))

    # convert back to arc length
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return (mean_angle / (2 * np.pi)) * max_arc_length


def _match_trajectories_spatial_interpolation(
    trajectory: Trajectory, other: Trajectory, max_distance: float = 0.0, k_nearest: int = 10
) -> tuple[Trajectory, Trajectory]:
    """This method matches both trajectories spatially by requesting
    the nearest two poses from the reference trajectory for each pose in the
    test trajectory. Then, an interpolation is performed between the two
    nearest poses.

    After this operation, both trajectories will have the length of the
    test trajectory. This means, that the reference trajectory may be
    modified.

    Args:
        trajectory (Trajectory): Trajectory to match
        other (Trajectory): Other trajectory to match against
        max_distance (float, optional): Maximum distance between two poses.
                                        Defaults to None. This means all
                                        matches are accepted.
        k_nearest (int, optional): Number of nearest poses to request from
                                   the reference trajectory. Defaults to 10.

    Returns:
        Tuple[Trajectory, Trajectory]: Matched trajectories
    """
    if k_nearest < 2:
        raise ValueError(f"k_nearest must be at least 2 to fit a 3d line, got {k_nearest}.")

    test_xyz = trajectory.positions.xyz
    ref_xyz = other.positions.xyz

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

    matched_test_rot = []
    matched_ref_rot = []

    lap_length = float(np.max(other.path_lengths)) if len(other) else 0.0
    matched_arc_lengths = []
    matched_indices = []
    # once: as_quat copies all quaternions, and a row of it kept in a list would keep the whole copy alive
    ref_quat = other.rotations.as_quat() if other.rotations is not None else None
    test_quat = trajectory.rotations.as_quat() if trajectory.rotations is not None else None
    for i, (dists, idxs) in enumerate(zip(distances, closest_indices)):
        valid_mask = ~np.isinf(dists)
        idxs = idxs[valid_mask]
        if idxs.size < 2:  # a 3d line needs at least two reference points
            continue

        test_pos = test_xyz[i, :]
        fit_line = Line3D.from_points(ref_xyz[idxs, :])
        line_point = fit_line.evaluate_at(test_pos)

        if ref_quat is not None:
            ref_rots = Rotations.from_quat(ref_quat[idxs])
            matched_ref_rot.append(ref_rots.mean().as_quat())

        if test_quat is not None:
            matched_test_rot.append(test_quat[i])

        matched_test_pos.append(test_pos)
        matched_ref_pos.append(line_point)

        matched_arc_lengths.append(_determine_mean_arc_length(other.path_lengths[idxs], lap_length))
        matched_indices.append(i)

    if not matched_indices:
        raise ValueError("No matches found! Each pose needs at least two reference poses within max_distance.")

    # both keep the timestamps and velocities of the evaluated trajectory
    timestamps = trajectory.timestamps[matched_indices]
    velocities = trajectory.velocity_xyz[matched_indices]
    trajectory = Trajectory(
        name=trajectory.name,
        positions=Positions(
            np.array(matched_test_pos),
            epsg=trajectory.positions.epsg,
            local_transformer=trajectory.positions.local_transformer,
        ),
        rotations=Rotations.from_quat(np.array(matched_test_rot)) if trajectory.has_orientation else None,
        sorting=trajectory.sorting,
        path_lengths=np.array(matched_arc_lengths),
        timestamps=timestamps,
        velocity_xyz=velocities,
    )
    other = Trajectory(
        name=other.name,
        positions=Positions(
            np.array(matched_ref_pos), epsg=other.positions.epsg, local_transformer=other.positions.local_transformer
        ),
        rotations=Rotations.from_quat(np.array(matched_ref_rot)) if other.has_orientation else None,
        sorting=other.sorting,
        path_lengths=np.array(matched_arc_lengths),
        timestamps=timestamps,
        velocity_xyz=velocities,
    )

    return trajectory, other


def _kd_matcher(ref: np.ndarray, test: np.ndarray, max_distance: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Matches data using a KDTree.

    Args:
        ref (np.ndarray): Reference data points.
        test (np.ndarray): Test data points.
        max_distance (float, optional): Maximum distance for matching. Defaults to 0.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Matched indices for reference and test data.
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
