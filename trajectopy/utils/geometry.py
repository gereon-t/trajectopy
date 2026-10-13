"""Geometric helper functions operating on point sets and trajectories."""

import logging

import numpy as np
from scipy.sparse import spdiags

from trajectopy.utils.linalg import least_squares

logger = logging.getLogger(__name__)


def nearest_point(*, p: np.ndarray, line_pts: list) -> tuple[np.ndarray, float]:
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


def lengths_from_xyz(xyz: np.ndarray) -> np.ndarray:
    """
    Computes the cumulative distance along a path defined by a sequence of
    3D points.

    Args:
        xyz (np.ndarray): An array of shape (n, 3) containing the x, y, and z
            coordinates of the path.

    Returns:
        np.ndarray: An array of shape (n,) containing the cumulative distance
            along the path.
    """
    if not isinstance(xyz, np.ndarray):
        logger.error("Invalid data type %s", type(xyz))
        return np.array([])

    if len(xyz) == 0:
        return np.array([])

    if len(xyz) == 1:
        return np.array([0.0])

    xyz_1 = xyz[0:-1, :]
    xyz_2 = xyz[1:, :]

    diff = xyz_2 - xyz_1

    dists = np.linalg.norm(diff, axis=1)
    return np.r_[0, np.cumsum(dists)]


def gradient_3d(xyz: np.ndarray, tstamps: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of a 3D trajectory.

    Args:
        xyz (np.ndarray): Positions of the trajectory [nx3].
        tstamps (np.ndarray): Timestamps of the trajectory [nx1].

    Returns:
        np.ndarray: Gradient of the trajectory [nx3].
    """
    if len(xyz) <= 1:
        return np.zeros((len(xyz), 3))

    # gradient (use target positions for this as they are probably more precise)
    diff = xyz[1:, :] - xyz[0:-1, :]
    t_diff = tstamps[1:] - tstamps[:-1]

    # Protect against zero time differences (duplicate timestamps)
    t_diff = np.where(t_diff == 0, np.finfo(float).eps, t_diff)

    # no gradient for last position
    return np.r_[diff / t_diff[:, None], np.zeros((1, 3))]


def common_span(index1: np.ndarray, index2: np.ndarray) -> tuple[float, float] | None:
    """
    Computes the common span between two arrays of timestamps.

    Args:
        index1 (np.ndarray): First array of indices.
        index2 (np.ndarray): Second array of indices.

    Returns:
        Union[Tuple[float, float], None]: A tuple containing the start and end indices of the common span,
        or None if there is no overlap between the two arrays.
    """
    index1 = np.sort(index1)
    index2 = np.sort(index2)

    overlap = (
        index1[0] <= index2[0] <= index1[-1]
        or index1[0] <= index2[-1] <= index1[-1]
        or index2[0] <= index1[0] <= index2[-1]
        or index2[0] <= index1[-1] <= index2[-1]
    )
    if not overlap:
        return None

    # get limits (largest common span)
    index_start = max(index1[0], index2[0])
    index_end = min(index1[-1], index2[-1])

    return (index_start, index_end)


def fit_line_3d(xyz: np.ndarray) -> np.ndarray:
    """
    Fits a 3D line using least-squares

    Parameters:
    xyz (np.ndarray): A numpy array of shape (n, 3) containing the 3D points

    Returns:
    np.ndarray: A numpy array of shape (3,) containing the direction of the line
    """
    N = np.cov(xyz, rowvar=False)
    return np.linalg.eigh(N)[1][:, -1]


def fit_line_2d(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a 2D line using least-squares

    Parameters:
    x (np.ndarray): A numpy array of shape (n,) containing the x-coordinates of the 2D points
    y (np.ndarray): A numpy array of shape (n,) containing the y-coordinates of the 2D points
    weights (np.ndarray): A numpy array of shape (n,) containing the weights of the points (default is an array of ones)

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the slope, intercept, and residuals of the fitted line
    """
    # design matrix
    A = np.c_[x, np.ones((len(x), 1))]

    if weights is None or len(weights) == 0:
        weights = np.ones(len(y))

    sigma_ll = spdiags(weights, 0, len(weights), len(weights))

    # solve normal equation
    x_s, l_s, v = least_squares(design_matrix=A, observations=y, sigma_ll=sigma_ll)

    return x_s, l_s, v


def skew_symmetric_matrix(vector: np.ndarray) -> np.ndarray:
    """
    Returns the skew-symmetric matrix of a 3D vector.

    Parameters:
    vector (np.ndarray): A numpy array of shape (3,) containing the 3D vector.

    Returns:
    np.ndarray: A numpy array of shape (3, 3) containing the skew-symmetric matrix.
    """
    return np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )
