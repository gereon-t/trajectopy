import logging

import numpy as np
from scipy.spatial.transform import Slerp

from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import InterpolationMethod
from trajectopy.core.trajectory import Trajectory
from trajectopy.utils.definitions import Sorting

logger = logging.getLogger(__name__)


def interpolate(
    trajectory: Trajectory,
    index: list | np.ndarray,
    method: InterpolationMethod = InterpolationMethod.LINEAR,
    inplace: bool = True,
    max_gap_size: float = float("inf"),
) -> "Trajectory":
    """Interpolates a trajectory to specified timestamps using the given method.

    Args:
        trajectory (Trajectory): Trajectory to interpolate
        index (list | np.ndarray): Interpolation index
        method (InterpolationMethod, optional): Interpolation method. Defaults to InterpolationMethod.LINEAR.
        inplace (bool, optional): Perform in-place interpolation. Defaults to True.
        max_gap_size (float, optional): Maximum gap in the index over which to interpolate. Defaults to infinity.

    Returns:
        Trajectory: Interpolated trajectory
    """
    if method == InterpolationMethod.LINEAR:
        return _interpolate_linear(trajectory, index, inplace, max_gap_size)
    else:
        raise ValueError(f"Interpolation method '{method}' is not supported.")


def _interpolate_linear(
    trajectory: Trajectory, index: list | np.ndarray, inplace: bool = True, max_gap_size: float = float("inf")
) -> "Trajectory":
    """Interpolates a trajectory to specified timestamps.

    This method removes timestamps from tstamps if they lie outside of the timestamp range
    of the trajectory (self). Since providing values for those timestamps would require
    an extrapolation and not an interpolation, this behaviour is consistent with the
    definition of this method.

    Args:
        trajectory (Trajectory): Trajectory to interpolate.
        index (list | np.ndarray): Interpolation index.
        inplace (bool, optional): Perform in-place interpolation. Defaults to True.

    Returns:
        Trajectory: Interpolated trajectory.

    Raises:
        ValueError: If no valid timestamps remain after cropping to trajectory range.
    """
    index = np.sort(index)
    trajectory = trajectory if inplace else trajectory.copy()

    if len(trajectory.index) == 0:
        raise ValueError("Cannot interpolate trajectory with no index")

    index_cropped = np.array([idx for idx in index if trajectory.index[0] <= idx <= trajectory.index[-1]])

    if len(index_cropped) == 0:
        raise ValueError(
            f"No valid indices for interpolation. Target indices [{index[0]:.3f}, {index[-1]:.3f}] "
            f"do not overlap with trajectory indices [{trajectory.index[0]:.3f}, {trajectory.index[-1]:.3f}]"
        )

    if max_gap_size < float("inf"):
        index_cropped = index_cropped[~inside_gaps(index_cropped, trajectory.index, max_gap_size)]

        if len(index_cropped) == 0:
            raise ValueError("No valid indices left after applying max_gap_size filter.")

    _drop_duplicate_index_values(trajectory)

    if trajectory.sorting == Sorting.TIME:
        trajectory.path_lengths = np.interp(index_cropped, trajectory.timestamps, trajectory.path_lengths)

    if trajectory.sorting == Sorting.PATH_LENGTH:
        trajectory.timestamps = np.interp(index_cropped, trajectory.path_lengths, trajectory.timestamps)

    _interpolate_positions_linear(trajectory, index_cropped)
    _interpolate_rotations_linear(trajectory, index_cropped)
    _interpolate_velocity_linear(trajectory, index_cropped)

    if trajectory.sorting == Sorting.TIME:
        trajectory.timestamps = index_cropped
    else:
        trajectory.path_lengths = index_cropped

    logger.info("Interpolated %s", trajectory.name)

    return trajectory


def inside_gaps(index: np.ndarray, observed_index: np.ndarray, max_gap_size: float) -> np.ndarray:
    """Returns which values of an index lie inside gaps of the observations that are larger than max_gap_size.

    Such values are not interpolated. Values at the edges of a gap are, since the observations on both
    sides of an edge are available.

    Args:
        index (np.ndarray): Index values to interpolate at.
        observed_index (np.ndarray): Ascending index of the observations.
        max_gap_size (float): Maximum gap in the observed index over which to interpolate.

    Returns:
        np.ndarray: Boolean mask, True for values of index that lie strictly inside a gap.
    """
    gaps = np.diff(observed_index) > max_gap_size
    in_gap = np.zeros(len(index), dtype=bool)
    for g_start, g_end in zip(observed_index[:-1][gaps], observed_index[1:][gaps]):
        in_gap |= (index > g_start) & (index < g_end)
    return in_gap


def _drop_duplicate_index_values(trajectory: Trajectory) -> None:
    """Keeps only the first pose of each index value, since interpolation requires a strictly increasing index.

    Duplicate path lengths occur for spatially sorted trajectories, e.g. at standstills or if several laps
    pass exactly the same positions.
    """
    _, first_indices = np.unique(trajectory.index, return_index=True)
    if len(first_indices) == len(trajectory):
        return
    logger.info(
        "%s: ignoring %i poses with duplicate %s for the interpolation.",
        trajectory.name,
        len(trajectory) - len(first_indices),
        "timestamps" if trajectory.sorting == Sorting.TIME else "path lengths",
    )
    trajectory.mask(first_indices)


def _interpolate_rotations_linear(
    trajectory: Trajectory, index: list | np.ndarray, inplace: bool = True
) -> "Trajectory":
    """Performs rotation interpolation of a trajectory using Spherical-Linear-Interpolation (SLERP).

    Args:
        trajectory (Trajectory): Trajectory to interpolate.
        index (list | np.ndarray): Interpolation index.
        inplace (bool, optional): Perform in-place interpolation. Defaults to True.

    Returns:
        Trajectory: Trajectory with interpolated rotations.
    """
    trajectory = trajectory if inplace else trajectory.copy()

    if not trajectory.rotations or len(index) == 0:
        return trajectory

    # the observations, not len(trajectory): the positions may already be interpolated
    if len(trajectory.index) == 1:
        # slerp needs two rotations, and a single pose is only interpolated at its own index
        trajectory.rotations = Rotations.from_quat(np.repeat(trajectory.rotations.as_quat(), len(index), axis=0))
        return trajectory

    # spherical linear orientation interpolation
    # Slerp interpolation, as geodetic curve on unit sphere
    slerp = Slerp(trajectory.index, trajectory.rotations)
    r_i = slerp(index)
    trajectory.rotations = Rotations.from_quat(r_i.as_quat())
    return trajectory


def _interpolate_positions_linear(trajectory: Trajectory, index: np.ndarray, inplace: bool = True) -> "Trajectory":
    """Performs position interpolation of a trajectory using linear interpolation.

    Args:
        trajectory (Trajectory): Trajectory to interpolate.
        index (np.ndarray): Interpolation index.
        inplace (bool, optional): Perform in-place interpolation. Defaults to True.

    Returns:
        Trajectory: Trajectory with interpolated positions.
    """
    trajectory = trajectory if inplace else trajectory.copy()

    xyz = trajectory.positions.xyz
    x_i = np.interp(index, trajectory.index, xyz[:, 0])
    y_i = np.interp(index, trajectory.index, xyz[:, 1])
    z_i = np.interp(index, trajectory.index, xyz[:, 2])
    trajectory.positions.xyz = np.c_[x_i, y_i, z_i]
    return trajectory


def _interpolate_velocity_linear(trajectory: Trajectory, index: np.ndarray, inplace: bool = True) -> "Trajectory":
    """Performs velocity interpolation of a trajectory using linear interpolation.

    Args:
        trajectory (Trajectory): Trajectory to interpolate.
        index (np.ndarray): Interpolation index.
        inplace (bool, optional): Perform in-place interpolation. Defaults to True.

    Returns:
        Trajectory: Trajectory with interpolated velocities.
    """
    trajectory = trajectory if inplace else trajectory.copy()

    x_i = np.interp(index, trajectory.index, trajectory.velocity_xyz[:, 0])
    y_i = np.interp(index, trajectory.index, trajectory.velocity_xyz[:, 1])
    z_i = np.interp(index, trajectory.index, trajectory.velocity_xyz[:, 2])
    trajectory.velocity_xyz = np.c_[x_i, y_i, z_i]
    return trajectory
