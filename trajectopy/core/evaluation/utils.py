"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import Tuple, Union

import numpy as np

# logger configuration
logger = logging.getLogger("root")


def rms(x: Union[np.ndarray, float]) -> float:
    """
    Calculates the root mean square of an array.

    Args:
        x (np.ndarray): The input array.

    Returns:
        float: The root mean square of the input array.
    """
    return np.sqrt(np.mean(np.square(x)))


def nearest_point(*, p: np.ndarray, line_pts: list) -> Tuple[np.ndarray, float]:
    """
    Finds the nearest point on a 3D line to a given point.

    Args:
        p (np.ndarray): The point to find the nearest point to.
        line_pts (list): A list of two points that define the 3D line.

    Returns:
        np.ndarray: The nearest point on the 3D line to the given point.
    """
    a = line_pts[0]
    b = line_pts[1]

    # direction vector
    r_v = b - a

    r_v_norm = np.linalg.norm(r_v)

    # if both points are identical
    if r_v_norm == 0:
        p_nearest = a
        t = 0
    else:
        r_v = r_v / r_v_norm
        t = (p - a) @ r_v

        # nearest point on the 3d line
        p_nearest = a + t * r_v
    return p_nearest, t
