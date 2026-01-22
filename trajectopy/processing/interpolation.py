import logging

import numpy as np
from scipy.spatial.transform import Slerp

from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import InterpolationMethod
from trajectopy.core.trajectory import Trajectory
from trajectopy.utils.common import gradient_3d

logger = logging.getLogger(__name__)


def interpolate(
    trajectory: Trajectory,
    timestamps: list | np.ndarray,
    method: InterpolationMethod = InterpolationMethod.LINEAR,
    inplace: bool = True,
) -> "Trajectory":
    """Interpolates a trajectory to specified timestamps using the given method.

    Args:
        trajectory (Trajectory): Trajectory to interpolate
        timestamps (list | np.ndarray): Interpolation timestamps
        method (InterpolationMethod, optional): Interpolation method. Defaults to InterpolationMethod.LINEAR.
        inplace (bool, optional): Perform in-place interpolation. Defaults to True.

    Returns:
        Trajectory: Interpolated trajectory
    """
    if method == InterpolationMethod.LINEAR:
        return _interpolate_linear(trajectory, timestamps, inplace)
    else:
        raise ValueError(f"Interpolation method '{method}' is not supported.")


def _interpolate_linear(trajectory: Trajectory, timestamps: list | np.ndarray, inplace: bool = True) -> "Trajectory":
    """Interpolates a trajectory to specified timestamps.

    This method removes timestamps from tstamps if they lie outside of the timestamp range
    of the trajectory (self). Since providing values for those timestamps would require
    an extrapolation and not an interpolation, this behaviour is consistent with the
    definition of this method.

    Args:
        trajectory (Trajectory): Trajectory to interpolate.
        timestamps (list | np.ndarray): Interpolation timestamps.
        inplace (bool, optional): Perform in-place interpolation. Defaults to True.

    Returns:
        Trajectory: Interpolated trajectory.

    Raises:
        ValueError: If no valid timestamps remain after cropping to trajectory range.
    """
    timestamps = np.sort(timestamps)
    trajectory = trajectory if inplace else trajectory.copy()

    if len(trajectory.timestamps) == 0:
        raise ValueError("Cannot interpolate trajectory with no timestamps")

    tstamps_cropped = np.array(
        [tstamp for tstamp in timestamps if trajectory.timestamps[0] <= tstamp <= trajectory.timestamps[-1]]
    )

    if len(tstamps_cropped) == 0:
        raise ValueError(
            f"No valid timestamps for interpolation. Target timestamps [{timestamps[0]:.3f}, {timestamps[-1]:.3f}] "
            f"do not overlap with trajectory timestamps [{trajectory.timestamps[0]:.3f}, {trajectory.timestamps[-1]:.3f}]"
        )

    # Store original timestamps before modifying positions
    original_timestamps = trajectory.timestamps.copy()
    original_path_lengths = trajectory.path_lengths.copy()

    _interpolate_positions_linear(trajectory, tstamps_cropped)
    _interpolate_rotations_linear(trajectory, tstamps_cropped)
    trajectory.path_lengths = np.interp(tstamps_cropped, original_timestamps, original_path_lengths)
    trajectory.timestamps = tstamps_cropped
    trajectory.velocity_xyz = gradient_3d(xyz=trajectory.positions.xyz, tstamps=trajectory.timestamps)

    logger.info("Interpolated %s", trajectory.name)

    return trajectory


def _interpolate_rotations_linear(
    trajectory: Trajectory, timestamps: list | np.ndarray, inplace: bool = True
) -> "Trajectory":
    """Performs rotation interpolation of a trajectory using Spherical-Linear-Interpolation (SLERP).

    Args:
        trajectory (Trajectory): Trajectory to interpolate.
        timestamps (list | np.ndarray): Interpolation timestamps.
        inplace (bool, optional): Perform in-place interpolation. Defaults to True.

    Returns:
        Trajectory: Trajectory with interpolated rotations.
    """
    trajectory = trajectory if inplace else trajectory.copy()

    if not trajectory.rotations or len(timestamps) == 0:
        return trajectory

    # spherical linear orientation interpolation
    # Slerp interpolation, as geodetic curve on unit sphere
    slerp = Slerp(trajectory.timestamps, trajectory.rotations)
    r_i = slerp(timestamps)
    trajectory.rotations = Rotations.from_quat(r_i.as_quat())
    return trajectory


def _interpolate_positions_linear(
    trajectory: Trajectory, timestamps: np.ndarray, inplace: bool = True
) -> "Trajectory":
    """Performs position interpolation of a trajectory using linear interpolation.

    Args:
        trajectory (Trajectory): Trajectory to interpolate.
        timestamps (np.ndarray): Interpolation timestamps.
        inplace (bool, optional): Perform in-place interpolation. Defaults to True.

    Returns:
        Trajectory: Trajectory with interpolated positions.
    """
    trajectory = trajectory if inplace else trajectory.copy()

    x_i = np.interp(timestamps, trajectory.timestamps, trajectory.positions.x)
    y_i = np.interp(timestamps, trajectory.timestamps, trajectory.positions.y)
    z_i = np.interp(timestamps, trajectory.timestamps, trajectory.positions.z)
    trajectory.positions.xyz = np.c_[x_i, y_i, z_i]
    return trajectory
