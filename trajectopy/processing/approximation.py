import logging

import numpy as np

from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import ApproximationSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.lib.approximation.cubic_approximation import (
    CubicApproximation,
)
from trajectopy.utils.common import rndodd

logger = logging.getLogger(__name__)


def approximate_cubic(
    trajectory: Trajectory,
    approximation_settings: ApproximationSettings = ApproximationSettings(),
    inplace: bool = False,
) -> Trajectory:
    """Approximate the trajectory using piecewise cubic polynomials based on the provided settings.

    Args:
        trajectory (Trajectory): The trajectory to approximate.
        approximation_settings (ApproximationSettings): Settings for the approximation.
        inplace (bool): If True, modify the trajectory in place. Defaults to False.

    Returns:
        Trajectory: The approximated trajectory.
    """
    xyz_approx = _piecewise_cubic(
        index=trajectory.index,
        values=trajectory.xyz,
        min_win_size=approximation_settings.position_interval_size,
        min_obs=approximation_settings.position_min_observations,
    )

    traj_approx = trajectory if inplace else trajectory.copy()
    traj_approx.positions.xyz = xyz_approx[trajectory.sort_switching_index, :]

    if not traj_approx.has_orientation:
        return traj_approx

    quat_approx = _average_rotations_in_window(
        index=trajectory.index,
        quat=trajectory.quat,
        win_size=approximation_settings.rotation_window_size,
    )
    traj_approx.rotations = Rotations.from_quat(quat_approx[trajectory.sort_switching_index, :])

    return traj_approx


def _average_rotations_in_window(index: np.ndarray, quat: np.ndarray, win_size: float = 0.15) -> np.ndarray:
    """Averages rotations for a given window size using quaternion averaging.

    For each rotation, all rotations within a window centered at the current rotation
    are averaged for the computation of the mean rotation. For this, the chordal L2 mean is used.
    The average rotation for the first and last rotations are computed using smaller window
    sizes (minimum half of the window size).

    Args:
        index (np.ndarray): The time / path lengths describing the "location" of the given
            rotations either in time or in trajectory length.
        quat (np.ndarray): Quaternions to average.
        win_size (float, optional): Window size used for rotation averaging in meters.
            Defaults to 0.15.

    Returns:
        np.ndarray: Averaged quaternions.
    """
    logger.info("Using window technique for rotation averaging.")

    # define window
    steps = rndodd(len(index) / len(np.unique(np.round(index / (win_size)))))
    ext = int(np.floor(steps / 2))

    if ext == 0:
        logger.warning(
            "Window size for rotation approximation is set too small. There are no rotations to approximate within the windows."
        )
        return quat

    quat_mean = np.zeros((len(index), 4))
    num_rotations = len(quat)
    for i in range(num_rotations):
        window_start_index = max(0, i - ext)
        window_end_index = min(num_rotations - 1, i + ext)
        quat_i = quat[window_start_index:window_end_index, :]
        # compute mean
        quat_mean[i, :] = Rotations.from_quat(quat_i).mean().as_quat()
    return quat_mean


def _piecewise_cubic(
    index: np.ndarray,
    values: np.ndarray,
    min_win_size: float = 0.25,
    min_obs: int = 25,
    return_approx_objects: bool = False,
) -> tuple[np.ndarray, list[CubicApproximation]] | np.ndarray:
    """Approximates a piecewise cubic function for a given set of input values.

    Args:
        index (np.ndarray): The input values to approximate the function for.
        values (np.ndarray): The output values corresponding to the input values.
        min_win_size (float, optional): The minimum window size for the approximation.
            Defaults to 0.25.
        min_obs (int, optional): The minimum number of observations required for the
            approximation. Defaults to 25.
        return_approx_objects (bool, optional): Whether to return the list of
            CubicApproximation objects along with the approximated values. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[list[CubicApproximation], np.ndarray]]: The approximated
            values. If `return_approx_objects` is True, returns a tuple containing the
            approximated values and the list of CubicApproximation objects.
    """
    # Cubic spline approximation
    # least squares
    approx_list = [CubicApproximation(index, values[:, i], min_win_size, min_obs) for i in range(values.shape[1])]

    approx_values = np.column_stack([ap.est_obs for ap in approx_list if ap.est_obs is not None])

    return (approx_values, approx_list) if return_approx_objects else approx_values
