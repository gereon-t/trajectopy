import copy
import logging

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Slerp

from trajectopy.core import settings
from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.interpolation import interpolate
from trajectopy.processing.lib.approximation.line_approximation import Line3D

logger = logging.getLogger(__name__)
from trajectopy.core.trajectory import Trajectory


def match_trajectories(
    trajectories: Trajectory | list[Trajectory],
    reference: Trajectory,
    matching_settings: settings.MatchingSettings = settings.MatchingSettings(),
    inplace: bool = True,
) -> tuple[Trajectory | list[Trajectory], Trajectory]:
    """Matches one or more trajectories to a reference trajectory.

    All input trajectories are matched to the reference trajectory using the specified method.
    The reference trajectory is also adjusted to fit the common time span of all trajectories.
    After matching, all trajectories (including the reference) will have the same length and
    sampling.

    Supported methods:

    - **INTERPOLATION**: Interpolates trajectories onto the reference trajectory using
      its timestamps. The interpolation is linear for both positions and rotations (SLERP).
    - **NEAREST_TEMPORAL**: Finds the nearest temporal match without interpolation by finding
      the nearest timestamp in the reference trajectory for each timestamp in the input trajectory.
    - **NEAREST_SPATIAL**: Finds the nearest spatial match without interpolation by finding
      the nearest pose in the reference trajectory for each pose in the input trajectory using
      Euclidean distance.
    - **NEAREST_SPATIAL_INTERPOLATED**: Finds the nearest k spatial matches and spatially
      interpolates using a 3d line. This method does not support rotation matching.

    Args:
        trajectories (Trajectory | list[Trajectory]): Single trajectory or list of trajectories to match.
        reference (Trajectory): Reference trajectory that defines the target timestamps/sampling.
        matching_settings (MatchingSettings, optional): Matching settings. Defaults to MatchingSettings().
        inplace (bool, optional): Whether to modify the input trajectories. Defaults to True.

    Returns:
        tuple[Trajectory | list[Trajectory], Trajectory]: Tuple containing:
            - Matched trajectory or list of matched trajectories
            - Matched reference trajectory
            All returned trajectories have the same length and sampling.
    """
    single_input = isinstance(trajectories, Trajectory)
    traj_list = [trajectories] if single_input else trajectories

    if not inplace:
        traj_list = [traj.copy() for traj in traj_list]
        reference = reference.copy()

    # First, find the common time span across all trajectories and reference
    all_trajectories = traj_list + [reference]
    common_start, common_end = _find_common_time_span(all_trajectories)
    logger.info("Common time span across all trajectories: [%.3f, %.3f] s", common_start, common_end)

    # Crop reference to common time span first
    reference.crop(t_start=common_start, t_end=common_end)

    # Also crop all trajectories to common time span to ensure they can all be matched
    for traj in traj_list:
        traj.crop(t_start=common_start, t_end=common_end)

    # For interpolation method, we need to find timestamps that ALL trajectories can support
    # The reference timestamps are the target, but we may need to filter them based on what
    # all trajectories can provide
    if matching_settings.method == settings.MatchingMethod.INTERPOLATION:
        # Find which reference timestamps can be supported by ALL trajectories
        valid_ref_timestamps = _find_common_interpolatable_timestamps(
            traj_list, reference, matching_settings.max_gap_size
        )
        # Mask reference to only valid timestamps
        valid_mask = np.isin(reference.timestamps, valid_ref_timestamps)
        reference.mask(np.where(valid_mask)[0])

        if len(reference) == 0:
            raise ValueError(
                "No common timestamps found across all trajectories! "
                "Check your matching settings, especially max_gap_size."
            )

        # Now interpolate each trajectory to the final reference timestamps
        matched_trajectories = []
        for traj in traj_list:
            interpolate(traj, reference.timestamps)
            traj.path_lengths = copy.deepcopy(reference.path_lengths)
            matched_trajectories.append(traj)
    else:
        # For non-interpolation methods, match first trajectory to establish reference
        first_matched, matched_reference = _match_trajectory_pair(
            trajectory=traj_list[0],
            reference=reference,
            matching_settings=matching_settings,
        )
        matched_trajectories = [first_matched]
        reference = matched_reference

        # Match remaining trajectories to the SAME reference
        for traj in traj_list[1:]:
            ref_copy = reference.copy()
            matched_traj, _ = _match_trajectory_pair(
                trajectory=traj,
                reference=ref_copy,
                matching_settings=matching_settings,
            )
            matched_trajectories.append(matched_traj)

    matched_result = matched_trajectories[0] if single_input else matched_trajectories
    return matched_result, reference


def _find_common_interpolatable_timestamps(
    trajectories: list[Trajectory], reference: Trajectory, max_gap_size: float
) -> np.ndarray:
    """Finds reference timestamps that all trajectories can interpolate to.

    A reference timestamp is valid if ALL trajectories can provide interpolated data
    at that timestamp. This requires:
    1. The timestamp must be within the trajectory's time range (no extrapolation)
    2. The trajectory must have data points on BOTH sides close enough for interpolation,
       meaning the gap between the two surrounding data points must not exceed the
       allowed gap threshold (based on trajectory's average data rate)

    Data gaps detected in ANY trajectory are removed from ALL trajectories to ensure
    consistent sampling across the entire dataset.

    The gap threshold for each trajectory is calculated as:
        average_sampling_interval * max_gap_size
    This ensures the threshold adapts to each trajectory's data rate.

    Args:
        trajectories (list[Trajectory]): List of trajectories to check.
        reference (Trajectory): Reference trajectory with target timestamps.
        max_gap_size (float): Gap size multiplier. The actual allowed gap is this value
            multiplied by the trajectory's average sampling interval.

    Returns:
        np.ndarray: Array of valid reference timestamps.
    """
    # Start with all reference timestamps as potentially valid
    all_timestamps = reference.timestamps.copy()

    # Track which timestamps are valid (True = valid, False = invalid)
    global_valid_mask = np.ones(len(all_timestamps), dtype=bool)

    # First pass: identify all gaps and invalid regions from ALL trajectories
    for traj in trajectories:
        if len(traj.timestamps) == 0:
            logger.warning("Trajectory '%s' has no timestamps", traj.name)
            return np.array([])

        traj_start = traj.timestamps[0]
        traj_end = traj.timestamps[-1]

        # Calculate average sampling interval for this trajectory
        if len(traj.timestamps) > 1:
            avg_sampling_interval = np.mean(np.diff(traj.timestamps))
        else:
            avg_sampling_interval = 0.0

        # Gap threshold is based on average sampling interval
        gap_threshold = avg_sampling_interval * max_gap_size

        logger.info(
            "Trajectory '%s': %d poses, avg interval=%.4f s, gap threshold=%.4f s",
            traj.name,
            len(traj.timestamps),
            avg_sampling_interval,
            gap_threshold,
        )

        # Find all gaps in this trajectory and mark affected reference timestamps as invalid
        gaps_found = []
        if len(traj.timestamps) > 1:
            intervals = np.diff(traj.timestamps)
            for i, interval in enumerate(intervals):
                if interval > gap_threshold:
                    gap_start = traj.timestamps[i]
                    gap_end = traj.timestamps[i + 1]
                    gaps_found.append((gap_start, gap_end))

                    # Mark all reference timestamps within this gap as invalid
                    # (exclusive of endpoints since those have actual data)
                    gap_mask = (all_timestamps > gap_start) & (all_timestamps < gap_end)
                    global_valid_mask[gap_mask] = False

        if gaps_found:
            logger.info(
                "Trajectory '%s': found %d data gap(s): %s",
                traj.name,
                len(gaps_found),
                [(f"{s:.3f}-{e:.3f}") for s, e in gaps_found],
            )

        # Mark timestamps outside this trajectory's time range as invalid
        outside_range_mask = (all_timestamps < traj_start) | (all_timestamps > traj_end)
        global_valid_mask[outside_range_mask] = False

    # Apply the global mask to get valid timestamps
    valid_timestamps = all_timestamps[global_valid_mask]

    # Log summary
    n_removed = len(all_timestamps) - len(valid_timestamps)
    if n_removed > 0:
        logger.info(
            "Removed %d/%d reference timestamps due to gaps or range limits",
            n_removed,
            len(all_timestamps),
        )

    return valid_timestamps


def _find_common_time_span(trajectories: list[Trajectory]) -> tuple[float, float]:
    """Finds the common time span across all trajectories.

    The common time span is defined by the latest start time and the earliest end time
    across all trajectories.

    Args:
        trajectories (list[Trajectory]): List of trajectories to analyze.

    Returns:
        tuple[float, float]: Tuple containing (common_start, common_end) timestamps.

    Raises:
        ValueError: If no common time span exists across all trajectories.
    """
    start_times = [traj.timestamps[0] for traj in trajectories]
    end_times = [traj.timestamps[-1] for traj in trajectories]

    common_start = max(start_times)
    common_end = min(end_times)

    if common_start >= common_end:
        raise ValueError(
            f"No common time span exists across all trajectories. "
            f"Latest start time ({common_start:.3f}) >= earliest end time ({common_end:.3f})."
        )

    return common_start, common_end


def _match_trajectory_pair(
    trajectory: Trajectory,
    reference: Trajectory,
    matching_settings: settings.MatchingSettings,
) -> tuple[Trajectory, Trajectory]:
    """Matches a single trajectory to a reference trajectory.

    Args:
        trajectory (Trajectory): Trajectory to match.
        reference (Trajectory): Reference trajectory.
        matching_settings (MatchingSettings): Matching settings.

    Returns:
        tuple[Trajectory, Trajectory]: Matched trajectory and reference trajectory.
    """
    if not trajectory.overlaps_with(reference) and matching_settings.method in [
        settings.MatchingMethod.NEAREST_TEMPORAL,
        settings.MatchingMethod.INTERPOLATION,
    ]:
        logger.warning("Trajectories do not overlap! Performing rough matching first.")
        timeshift = match_non_overlapping_timestamps(
            trajectory=reference, reference=trajectory, max_distance=matching_settings.max_distance
        )
        logger.info("Rough matching time offset: %.3f s", timeshift)
        trajectory.timestamps += timeshift

    logger.info("Matching trajectory '%s' using method %s", trajectory.name, matching_settings.method.name)

    if matching_settings.method == settings.MatchingMethod.INTERPOLATION:
        return _match_trajectories_interpolation(
            trajectory=trajectory, reference=reference, max_gap_size=matching_settings.max_gap_size
        )

    if matching_settings.method == settings.MatchingMethod.NEAREST_TEMPORAL:
        return _match_trajectories_temporal(
            trajectory=trajectory, reference=reference, max_distance=matching_settings.max_time_diff
        )

    if matching_settings.method == settings.MatchingMethod.NEAREST_SPATIAL:
        return _match_trajectories_spatial(
            trajectory=trajectory, reference=reference, max_distance=matching_settings.max_distance
        )

    if matching_settings.method == settings.MatchingMethod.NEAREST_SPATIAL_INTERPOLATED:
        return _match_trajectories_spatial_interpolation(
            trajectory=trajectory,
            reference=reference,
            max_distance=matching_settings.max_distance,
            k_nearest=matching_settings.k_nearest,
        )

    raise ValueError(f"Matching method {matching_settings.method} not supported!")


# Backward compatibility alias
def match_all_trajectories_to_ref(
    trajectory: Trajectory,
    other: Trajectory,
    matching_settings: settings.MatchingSettings = settings.MatchingSettings(),
    inplace: bool = True,
) -> tuple[Trajectory, Trajectory]:
    """Matches a trajectory to a reference trajectory.

    This is an alias for backward compatibility. Use `match_trajectories` for new code.

    Args:
        trajectory (Trajectory): Trajectory to match.
        other (Trajectory): Reference trajectory.
        matching_settings (MatchingSettings, optional): Matching settings. Defaults to MatchingSettings().
        inplace (bool, optional): Whether to modify the input trajectories. Defaults to True.

    Returns:
        tuple[Trajectory, Trajectory]: Matched trajectory and reference trajectory.
    """
    if not inplace:
        trajectory = trajectory.copy()
        other = other.copy()

    matched_traj, matched_ref = _match_trajectory_pair(
        trajectory=trajectory,
        reference=other,
        matching_settings=matching_settings,
    )
    return matched_traj, matched_ref


def match_non_overlapping_timestamps(
    trajectory: Trajectory, reference: Trajectory, max_distance: float = 0.0
) -> float:
    """Roughly matches two trajectories temporally.

    Args:
        trajectory (Trajectory): Trajectory to match.
        reference (Trajectory): Reference trajectory to match against.
        max_distance (float, optional): Maximum distance for spatial matching. Defaults to 0.0.

    Returns:
        float: Median time offset.
    """
    reference_matched, trajectory_matched = _match_trajectories_spatial(
        trajectory=reference.copy(), reference=trajectory.copy(), max_distance=max_distance
    )
    mean_time_offset = np.median(trajectory_matched.timestamps - reference_matched.timestamps)
    logger.info("Median time offset: %.3f s", mean_time_offset)
    return mean_time_offset


def match_timestamps(trajectory: Trajectory, timestamps: np.ndarray, inplace: bool = True) -> Trajectory:
    """Truncates trajectory to only those poses where the timestamps exactly match "timestamps".

    Args:
        trajectory (Trajectory): Input trajectory.
        timestamps (np.ndarray): Input timestamps.
        inplace (bool, optional): Perform matching in-place. Defaults to True.

    Returns:
        Trajectory: Trajectory with matched timestamps.
    """
    traj_self = trajectory if inplace else trajectory.copy()
    _, idx_self, _ = np.intersect1d(traj_self.timestamps, timestamps, return_indices=True)
    traj_self.mask(idx_self)
    return traj_self


def _match_trajectories_interpolation(
    trajectory: Trajectory, reference: Trajectory, max_gap_size: float = 10.0
) -> tuple[Trajectory, Trajectory]:
    """Ensures that both trajectories are sampled in the same way.

    This method will intersect both trajectories with each other and then interpolate
    the trajectory onto the reference trajectory. The sorting and the arc lengths of
    both trajectories are identical after the call of this method.

    Args:
        trajectory (Trajectory): Trajectory to match.
        reference (Trajectory): Reference trajectory that defines the target timestamps.
        max_gap_size (float, optional): Maximum gap size in seconds. Defaults to 10.0.

    Returns:
        tuple[Trajectory, Trajectory]: Matched trajectory and reference trajectory.
    """
    trajectory.intersect(reference.timestamps, max_gap_size=max_gap_size)

    if trajectory.timestamps.shape[0] == 0:
        raise ValueError(
            "Trajectory has no timestamps after intersection! Check your matching settings, especially max_gap_size."
        )

    reference.intersect(trajectory.timestamps, max_gap_size=max_gap_size)

    if reference.timestamps.shape[0] == 0:
        raise ValueError(
            "Reference trajectory has no timestamps after intersection! Check your matching settings, especially max_gap_size."
        )
    interpolate(trajectory, reference.timestamps)
    trajectory.path_lengths = copy.deepcopy(reference.path_lengths)

    return trajectory, reference


def _match_trajectories_temporal(
    trajectory: Trajectory, reference: Trajectory, max_distance: float = 0.01
) -> tuple[Trajectory, Trajectory]:
    """Matches both trajectories temporally.

    Finds the nearest timestamp in the reference trajectory for each timestamp
    in the input trajectory.

    Args:
        trajectory (Trajectory): Trajectory to match.
        reference (Trajectory): Reference trajectory that defines the target timestamps.
        max_distance (float, optional): Maximum distance between two timestamps.
            Defaults to 0.01.

    Returns:
        tuple[Trajectory, Trajectory]: Matched trajectory and reference trajectory.
    """
    tstamps_ref_2d = np.c_[reference.timestamps, np.zeros(reference.timestamps.shape)]
    tstamps_test_2d = np.c_[trajectory.timestamps, np.zeros(trajectory.timestamps.shape)]
    ref_indices, test_indices = _kd_matcher(ref=tstamps_ref_2d, test=tstamps_test_2d, max_distance=max_distance)
    logger.info("Found %i temporal matches", len(ref_indices))
    return trajectory.mask(test_indices), reference.mask(ref_indices)


def _match_trajectories_spatial(
    trajectory: Trajectory, reference: Trajectory, max_distance: float = 0.0
) -> tuple[Trajectory, Trajectory]:
    """Matches both trajectories spatially.

    Finds the nearest spatial match in the reference trajectory for each pose
    in the input trajectory using Euclidean distance.

    Args:
        trajectory (Trajectory): Trajectory to match.
        reference (Trajectory): Reference trajectory that defines the target poses.
        max_distance (float, optional): Maximum distance between two poses. Defaults to 0.0.
            This means all matches are accepted.

    Returns:
        tuple[Trajectory, Trajectory]: Matched trajectory and reference trajectory.
    """
    ref_indices, test_indices = _kd_matcher(
        ref=reference.positions.xyz, test=trajectory.positions.xyz, max_distance=max_distance
    )
    logger.info("Found %i spatial matches", len(ref_indices))
    return trajectory.mask(test_indices), reference.mask(ref_indices)


def _determine_mean_arc_length(arc_lengths: np.ndarray) -> float:
    """Determines the mean arc length from a set of arc lengths while considering the circular nature of arc lengths.

    Args:
        arc_lengths (np.ndarray): Array of arc lengths.

    Returns:
        float: Mean arc length.
    """
    max_arc_length = max(arc_lengths)
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
    trajectory: Trajectory, reference: Trajectory, max_distance: float = 0.0, k_nearest: int = 10
) -> tuple[Trajectory, Trajectory]:
    """Matches trajectories spatially with interpolation.

    This method matches trajectories spatially by requesting the nearest k poses
    from the reference trajectory for each pose in the input trajectory. Then,
    a 3D line interpolation is performed between the nearest poses.

    Note: This method does not support rotation matching.

    Args:
        trajectory (Trajectory): Trajectory to match.
        reference (Trajectory): Reference trajectory that defines the target poses.
        max_distance (float, optional): Maximum distance between two poses.
            Defaults to 0.0, meaning all matches are accepted.
        k_nearest (int, optional): Number of nearest poses to request from
            the reference trajectory. Defaults to 10.

    Returns:
        tuple[Trajectory, Trajectory]: Matched trajectory and reference trajectory.
    """
    test_xyz = trajectory.positions.xyz
    ref_xyz = reference.positions.xyz

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

    matched_arc_lengths = []
    for i, (dists, idxs) in enumerate(zip(distances, closest_indices)):
        valid_mask = ~np.isinf(dists)
        idxs = idxs[valid_mask]
        if np.any(idxs) is False:
            continue

        test_pos = test_xyz[i, :]
        fit_line = Line3D.from_points(ref_xyz[idxs, :])
        line_point = fit_line.evaluate_at(test_pos)

        if reference.has_orientation:
            ref_rots = Rotations.from_quat(reference.rotations.as_quat()[idxs])
            matched_ref_rot.append(ref_rots.mean().as_quat())

        if trajectory.has_orientation:
            matched_test_rot.append(trajectory.rotations.as_quat()[i])

        matched_test_pos.append(test_pos)
        matched_ref_pos.append(line_point)

        matched_arc_lengths.append(_determine_mean_arc_length(reference.path_lengths[idxs]))

    trajectory = Trajectory(
        name=trajectory.name,
        positions=Positions(np.array(matched_test_pos), local_transformer=trajectory.positions.local_transformer),
        rotations=Rotations.from_quat(np.array(matched_test_rot)) if trajectory.has_orientation else None,
        sorting=trajectory.sorting,
        path_lengths=np.array(matched_arc_lengths),
    )
    reference = Trajectory(
        name=reference.name,
        positions=Positions(np.array(matched_ref_pos), local_transformer=reference.positions.local_transformer),
        rotations=Rotations.from_quat(np.array(matched_ref_rot)) if reference.has_orientation else None,
        sorting=reference.sorting,
        path_lengths=np.array(matched_arc_lengths),
    )

    return trajectory, reference


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
