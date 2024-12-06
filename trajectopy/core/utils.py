"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csr_matrix, identity, spdiags
from scipy.sparse.linalg import spsolve

# logger configuration
logger = logging.getLogger("root")


@dataclass
class Line3D:
    """A 3D line defined by a mean point and a direction vector.

    Attributes:
        mean (np.ndarray): The mean point of the line.
        direction (np.ndarray): The direction vector of the line.
    """

    # rest of the class implementation
    mean: np.ndarray
    direction: np.ndarray

    @classmethod
    def from_points(cls, points: np.ndarray) -> "Line3D":
        """Create a 3D line from a set of points.

        This method calculates the direction vector of the line
        from the eigenvector corresponding to the largest eigenvalue
        of the covariance matrix of the input points. The mean of the
        points is used as the mean point of the line.

        Args:
            cls (Line3D): The class object.
            points (np.ndarray): The input points.

        Returns:
            Line3D: A 3D line defined by a mean point and a direction vector.
        """
        cov_matrix = np.cov(points, rowvar=False)
        direction = np.linalg.eigh(cov_matrix)[1][:, -1]

        # mean of the points
        mean = np.mean(points, axis=0)
        return cls(mean=mean, direction=direction)

    def evaluate_at(self, location: np.ndarray) -> List[float]:
        """Evaluates the 3D line at a given location.

        This method calculates the projection of the input location
        onto the 3D line and returns the resulting point.

        Args:
            location (np.ndarray): The input location.

        Returns:
            list[float]: The resulting point on the 3D line.
        """
        tr = (
            (location[0] - self.mean[0]) * self.direction[0]
            + (location[1] - self.mean[1]) * self.direction[1]
            + (location[2] - self.mean[2]) * self.direction[2]
        ) * self.direction
        return [self.mean[0] + tr[0], self.mean[1] + tr[1], self.mean[2] + tr[2]]


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
    # gradient (use target positions for this as they are probably more precise)
    diff = xyz[1:, :] - xyz[0:-1, :]
    t_diff = tstamps[1:] - tstamps[:-1]

    # no gradient for last position
    return np.r_[diff / t_diff[:, None], np.zeros((1, 3))]


def common_time_span(tstamps1: np.ndarray, tstamps2: np.ndarray) -> Union[Tuple[float, float], None]:
    """
    Computes the common time span between two arrays of timestamps.

    Args:
        tstamps1 (np.ndarray): First array of timestamps.
        tstamps2 (np.ndarray): Second array of timestamps.

    Returns:
        Union[Tuple[float, float], None]: A tuple containing the start and end times of the common time span,
        or None if there is no overlap between the two arrays.
    """
    tstamps1 = np.sort(tstamps1)
    tstamps2 = np.sort(tstamps2)

    overlap = (
        tstamps1[0] <= tstamps2[0] <= tstamps1[-1]
        or tstamps1[0] <= tstamps2[-1] <= tstamps1[-1]
        or tstamps2[0] <= tstamps1[0] <= tstamps2[-1]
        or tstamps2[0] <= tstamps1[-1] <= tstamps2[-1]
    )
    if not overlap:
        return None

    # get limits (largest common time span)
    t_start = max(tstamps1[0], tstamps2[0])
    t_end = min(tstamps1[-1], tstamps2[-1])

    return (t_start, t_end)


def rndodd(s: float) -> int:
    """
    Rounds a float to the nearest odd integer.

    Args:
        s (float): The float to round.

    Returns:
        int: The rounded odd integer.
    """
    idx = s % 2 < 1
    s = np.floor(s)

    if idx:
        s += 1
    return int(s)


def round_to_precision(
    function_of: np.ndarray, data: np.ndarray, resolution: float, filter_size: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduces the amount of deviations using smoothing and rounding

    It will first smooth the data using a convolution with a filter
    of size filter_size. Then, the data is rounded to the specified
    resolution and duplicate values that can result from this operation
    are deleted.

    Args:
        function_of (np.ndarray): nxm array that contains for example
                                  time stamps, arc lengths or positions
                                  corresponding to the data.
        data (np.ndarray): nx1 array that contains the data that should
                           be smoothed.
        precision (float): Desired resolution
        filter_size (int): Window / filter size for smoothing

    Returns:
        downsampled function_of and data
    """
    data_smoothed = np.convolve(data, [1 / filter_size] * filter_size, "same")
    data_rounded = np.round(data_smoothed / resolution) * resolution
    _, indices = np.unique(np.c_[function_of, data_rounded], return_index=True, axis=0)
    indices_sorted = np.sort(indices)

    function_of_unique = function_of[indices_sorted, :] if function_of.ndim > 1 else function_of[indices_sorted]
    data_unique = data_smoothed[indices_sorted]

    return function_of_unique, data_unique


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
    x: np.ndarray, y: np.ndarray, weights: np.ndarray = np.array([])
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    if len(weights) == 0:
        weights = np.ones(len(y))

    sigma_ll = spdiags(weights, 0, len(weights), len(weights))

    # solve normal equation
    x_s, l_s, v = least_squares(design_matrix=A, observations=y, sigma_ll=sigma_ll)

    return x_s, l_s, v


def sparse_least_squares(
    design_matrix: csr_matrix, observations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves a least squares problem with a sparse matrix A and a dense vector l.

    Parameters:
    design_matrix (csr_matrix): A sparse matrix of shape (m, n) representing the design matrix.
    l (np.ndarray): A numpy array of shape (m,) containing the observations.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the solution vector x_s, the approximated observations l_s, and the residuals v.
    """
    # solve normal equations
    x_s = spsolve(design_matrix.T @ design_matrix, design_matrix.T @ observations)

    # approximated observations
    l_s = design_matrix @ x_s

    # residuals
    v = l_s[:, None] - observations

    return x_s, l_s, v


def least_squares(
    design_matrix: np.ndarray,
    observations: np.ndarray,
    sigma_ll: np.ndarray = np.array([]),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves a least squares problem with a dense matrix A and a dense vector l.

    Parameters:
    design_matrix (np.ndarray): A numpy array of shape (m, n) representing the design matrix.
    observations (np.ndarray): A numpy array of shape (m,) containing the observations.
    sigma_ll (np.ndarray): A numpy array of shape (m, m) containing the weights of the observations (default is an identity matrix).

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the solution vector x_s, the approximated observations l_s, and the residuals v.
    """
    observations = observations.reshape(
        len(observations),
    )
    if sigma_ll.shape[0] == 0:
        sigma_ll = identity(len(observations))
    # solve normal equations
    x_s = solve(
        design_matrix.T @ sigma_ll @ design_matrix,
        design_matrix.T @ sigma_ll @ observations,
    )

    # approximated observations
    l_s = design_matrix @ x_s

    # residuals
    v = l_s - observations

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
