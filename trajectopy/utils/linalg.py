"""Least-squares solvers."""

import logging

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


def sparse_least_squares(
    design_matrix: csr_matrix, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    sigma_ll: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if sigma_ll is None or sigma_ll.shape[0] == 0:
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
