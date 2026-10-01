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
        # Find exactly where the gaps in the original trajectory are greater than max_gap_size
        gaps = np.diff(trajectory.index) > max_gap_size
        gap_starts = trajectory.index[:-1][gaps]
        gap_ends = trajectory.index[1:][gaps]

        # Filter out target indices that fall in gaps
        valid_indices = np.ones(len(index_cropped), dtype=bool)
        for g_start, g_end in zip(gap_starts, gap_ends):
            in_gap = (index_cropped > g_start) & (index_cropped < g_end)
            valid_indices[in_gap] = False

        index_cropped = index_cropped[valid_indices]

        if len(index_cropped) == 0:
            raise ValueError("No valid indices left after applying max_gap_size filter.")

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

    x_i = np.interp(index, trajectory.index, trajectory.positions.x)
    y_i = np.interp(index, trajectory.index, trajectory.positions.y)
    z_i = np.interp(index, trajectory.index, trajectory.positions.z)
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
