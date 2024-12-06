"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging

import numpy as np

from trajectopy.core.rotationset import RotationSet
from trajectopy.core.utils import rndodd

# logger configuration
logger = logging.getLogger("root")


def rot_average_window(function_of: np.ndarray, quat: np.ndarray, win_size: float = 0.15) -> np.ndarray:
    """
    Function that averages rotations for a given
    window size using quaternion averaging.

    For each rotation, all rotations within a window centered
    at the current rotation are averaged for the compuation of
    the mean rotation. For this, the chordal L2 mean is used.

    The average rotation for the first and last rotations are
    computed using smaller window sizes (minimum half of the window size)

    Args:
        function_of (np.ndarray): The time / arc lengths describing the
                                  "location" of the given rotations either
                                  in time or in trajectory length.
        quat (np.ndarray): _description_
        win_size (float, optional): Window size used for rotation averaging
                                    in meters. Defaults to 0.15.

    Returns:
        np.ndarray: Averaged quaternions.
    """
    logger.info("Using window technique for rotation averaging.")

    # define window
    steps = rndodd(len(function_of) / len(np.unique(np.round(function_of / (win_size)))))
    ext = int(np.floor(steps / 2))

    if ext == 0:
        logger.warning(
            "Window size for rotation approximation is set too small. There are no rotations to approximate within the windows."
        )
        return quat

    quat_mean = np.zeros((len(function_of), 4))
    num_rotations = len(quat)
    for i in range(num_rotations):
        window_start_index = max(0, i - ext)
        window_end_index = min(num_rotations - 1, i + ext)
        quat_i = quat[window_start_index:window_end_index, :]
        # compute mean
        quat_mean[i, :] = RotationSet.from_quat(quat_i).mean().as_quat()
    return quat_mean
