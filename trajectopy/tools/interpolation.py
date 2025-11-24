import logging
from typing import Union

import numpy as np
from scipy.spatial.transform import Slerp

from trajectopy.rotations import Rotations
from trajectopy.settings import InterpolationMethod
from trajectopy.trajectory import Trajectory
from trajectopy.utils import gradient_3d

logger = logging.getLogger(__name__)


def interpolate(
    trajectory: Trajectory,
    timestamps: Union[list, np.ndarray],
    method: InterpolationMethod = InterpolationMethod.LINEAR,
    inplace: bool = True,
) -> "Trajectory":
    """Interpolates a trajectory to specified timestamps using the given method.

    Args:
        trajectory (Trajectory): Trajectory to interpolate
        timestamps (list): Interpolation timestamps
        method (InterpolationMethod, optional): Interpolation method. Defaults to "linear".
        inplace (bool, optional): Perform in-place interpolation. Defaults to True.

    Returns:
        Trajectory: Interpolated trajectory
    """
    if method == InterpolationMethod.LINEAR:
        return _interpolate_linear(trajectory, timestamps, inplace)
    else:
        raise ValueError(f"Interpolation method '{method}' is not supported.")


def _interpolate_linear(
    trajectory: Trajectory, timestamps: Union[list, np.ndarray], inplace: bool = True
) -> "Trajectory":
    """Interpolates a trajectory to specified timestamps

    This method removes timestamps from tstamps if they lie
    outside of the timestamp range of the trajectory (self).
    Since providing values for those timestamps would require
    an extrapolation and not an interpolation, this behaviour
    is consistent with the definition of this method.

    Args:
        trajectory (Trajectory): Trajectory to interpolate
        timestamps (list): Interpolation timestamps
        inplace (bool, optional): Perform in-place interpolation.
                                    Defaults to True.

    Returns:
        Trajectory: Interpolated trajectory
    """
    timestamps = np.sort(timestamps)
    trajectory = trajectory if inplace else trajectory.copy()
    tstamps_cropped = np.array(
        [tstamp for tstamp in timestamps if trajectory.timestamps[0] <= tstamp <= trajectory.timestamps[-1]]
    )

    _interpolate_positions_linear(trajectory, tstamps_cropped)
    _interpolate_rotations_linear(trajectory, tstamps_cropped)
    trajectory.velocity_xyz = gradient_3d(xyz=trajectory.positions.xyz, tstamps=tstamps_cropped)
    trajectory.path_lengths = np.interp(tstamps_cropped, trajectory.timestamps, trajectory.path_lengths)
    trajectory.timestamps = tstamps_cropped

    logger.info("Interpolated %s", trajectory.name)

    return trajectory


def _interpolate_rotations_linear(
    trajectory: Trajectory, timestamps: Union[list, np.ndarray], inplace: bool = True
) -> "Trajectory":
    """Function for rotation interpolation of a trajectory

    This method uses Spherical-Linear-Interpolation
    for rotation interpolation.

    Args:
        trajectory (Trajectory): Trajectory to interpolate
        timestamps (np.ndarray): Interpolation timestamps
        inplace (bool, optional): Perform in-place interpolation.

    Returns:
        RotationSet: Interpolated Rotationset
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
    """Function for position interpolation of a trajectory

    Args:
        timestamps (np.ndarray): Interpolation timestamps
        inplace (bool, optional): Perform in-place interpolation.

    Returns:
        np.ndarray: Interpolated positions
    """
    trajectory = trajectory if inplace else trajectory.copy()

    x_i = np.interp(timestamps, trajectory.timestamps, trajectory.positions.x)
    y_i = np.interp(timestamps, trajectory.timestamps, trajectory.positions.y)
    z_i = np.interp(timestamps, trajectory.timestamps, trajectory.positions.z)
    trajectory.positions.xyz = np.c_[x_i, y_i, z_i]
    return trajectory
