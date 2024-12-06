"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import List, Tuple, Union

import numpy as np
import numpy.matlib as npm
from scipy.sparse import csr_matrix

from trajectopy.core.utils import sparse_least_squares

# logger configuration
logger = logging.getLogger("root")


class CubicApproximation:
    """
    Class for piecewise cubic approximation
    """

    def __init__(self, function_of: np.ndarray, values: np.ndarray, int_size: float, min_obs: int) -> None:
        """
        Inititalization of a new CubicApproximation class object
        """
        self.function_of = function_of
        self.values = values
        self.int_size = int_size
        self.min_obs = min_obs

        # fit results
        self.parameters: Union[np.ndarray, None] = None
        self.est_obs: Union[np.ndarray, None] = None
        self.residuals: Union[np.ndarray, None] = None
        self.interval_steps: Union[np.ndarray, None] = None

        # compute approximation
        self._cubic_approx()

    def eval(self, locations: np.ndarray) -> np.ndarray:
        """
        Evaluate the cubic approximation at specified locations
        """
        if self.interval_steps is None or self.parameters is None:
            raise ValueError("Approximation not yet computed")

        if np.min(locations) > np.max(self.interval_steps) or np.max(locations) < np.min(self.interval_steps):
            raise ValueError("Evaluation locations are outside of the approximation interval")

        # function
        f_vals = self.parameters[::2]
        # first derivative
        f_deriv_vals = self.parameters[1::2]

        interval_indices = np.searchsorted(self.interval_steps, locations)
        int_start = self.interval_steps[interval_indices - 1]
        int_end = self.interval_steps[interval_indices]

        c0, c1, c2, c3 = self._compute_c(locations, int_start, int_end)

        return (
            f_vals[interval_indices - 1] * c0
            + f_deriv_vals[interval_indices - 1] * c1
            + f_vals[interval_indices] * c2
            + f_deriv_vals[interval_indices] * c3
        )

    def _cubic_approx(self) -> None:
        """
        Approximation using piece-wise cubic polynomials
        """
        var_red = self.function_of - self.function_of[0]

        # interval steps
        interval_steps = np.arange(0, var_red[-1], self.int_size)

        # include endpoint if necessary
        if interval_steps[-1] != var_red[-1]:
            interval_steps = np.r_[interval_steps, var_red[-1]]

        # empty list for final interval boundaries
        t_final = []
        intervals: List[list] = []
        coefficients: List[np.ndarray] = []

        int_list = []
        int_start_idx = 0

        # go through all trajectory lengths
        for i, parameters in enumerate(var_red):
            # compute the current length of the interval
            int_len = parameters - var_red[int_start_idx]
            # add current value to the interval
            int_list.append(parameters)

            # check if it's the last interval
            last_int = i == len(var_red) - 1

            # if the desired interval length is reached and there are enough observations
            if (len(int_list) >= self.min_obs and int_len >= self.int_size) or last_int:
                # if it's the last interval and there are not enough values, merge with the second to last
                last_too_small = last_int and len(int_list) < self.min_obs
                if last_too_small:
                    # extend previous interval
                    t_start = var_red[int_start_idx - 1]
                    intervals[-1].extend(int_list)
                else:
                    # new interval
                    t_start = var_red[int_start_idx]
                    t_final.append(t_start)
                    intervals.append(int_list)

                # compute coefficients
                x_a = np.repeat(t_start, len(intervals[-1]))
                x_e = np.repeat(parameters, len(intervals[-1]))
                c0, c1, c2, c3 = self._compute_c(intervals[-1], x_a, x_e)

                if last_too_small:
                    coefficients[-1] = np.c_[c0, c1, c2, c3]
                else:
                    coefficients.append(np.c_[c0, c1, c2, c3])

                # reset interval list and start index
                int_list = []

                int_start_idx = i + 1

        t_final.append(var_red[-1])

        logger.debug(
            "Average observation count per interval: %.2f",
            len(var_red) / len(intervals),
        )

        dim = [self.function_of.size, 2 * len(coefficients) + 2]
        # Design matrix (jacobian)
        a_design = self._design_matrix(coefficients, dim)

        # least squares
        logger.debug("Approximation using piece-wise cubic polynomials via least-squares method.")
        xS, lS, residuals = sparse_least_squares(csr_matrix(a_design), self.values[:, None])

        # store results
        self.parameters = xS
        self.est_obs = lS
        self.residuals = residuals
        self.interval_steps = t_final + self.function_of[0]

    @staticmethod
    @np.vectorize
    def _compute_c(location: float, interval_start: float, interval_end: float) -> Tuple[float, float, float, float]:
        """
        Helper function to compute the coefficients for cubic approximation
        """
        c0 = (
            1
            - 3 * ((location - interval_start) / (interval_end - interval_start)) ** 2
            + 2 * ((location - interval_start) / (interval_end - interval_start)) ** 3
        )

        c1 = (location - interval_start) * (
            1
            - 2 * (location - interval_start) / (interval_end - interval_start)
            + ((location - interval_start) / (interval_end - interval_start)) ** 2
        )

        c2 = (
            3 * ((location - interval_start) / (interval_end - interval_start)) ** 2
            - 2 * ((location - interval_start) / (interval_end - interval_start)) ** 3
        )

        c3 = (location - interval_start) * (
            ((location - interval_start) / (interval_end - interval_start)) ** 2
            - (location - interval_start) / (interval_end - interval_start)
        )

        return c0, c1, c2, c3

    @staticmethod
    def _design_matrix(c: List[np.ndarray], dim: List[int]) -> csr_matrix:
        """
        Helper function to create the design (jacobian) matrix for least-squares adjustment
        """
        # vectorize data
        c_mat = np.row_stack(c)
        c_vec = np.reshape(c_mat, (c_mat.size,))

        # column indices
        # [1,2,3,4;
        #  1,2,3,4;
        #    ...
        #  3,4,5,6;
        #  3,4,5,6;
        #    ...
        # Blocks for intervals

        # interval sizes
        int_sizes = [len(parameters) for parameters in c]
        # column indicex increase by 2 with each interval
        col_offs = np.r_[np.arange(0, dim[1] - 4, 2), dim[1] - 4]
        # offsets are repeated according to interval size
        col_offs_rep = np.repeat(col_offs, int_sizes).reshape((dim[0], 1))
        # add offsets to plain [0,1,2,3]
        col_idx = npm.repmat(np.array([0, 1, 2, 3]), dim[0], 1) + col_offs_rep
        # vectorize
        col_idx_vec = np.reshape(col_idx, (col_idx.size,))

        # row indices
        # [0,0,0,0; 1,1,1,1; 2,2,2,2; 3,3,3,3; ...]
        row_idx = np.repeat(np.arange(0, dim[0], 1), 4)

        # return design matrix
        return csr_matrix((c_vec, (row_idx, col_idx_vec)))


def piecewise_cubic(
    function_of: np.ndarray,
    values: np.ndarray,
    int_size: float = 0.15,
    min_obs: int = 25,
    return_approx_objects: bool = False,
) -> Union[Tuple[np.ndarray, List[CubicApproximation]], np.ndarray]:
    """
    Approximates a piecewise cubic function for a given set of input values.

    Args:
        function_of (np.ndarray): The input values to approximate the function for.
        values (np.ndarray): The output values corresponding to the input values.
        int_size (float, optional): The interval size for the approximation. Defaults to 0.15.
        min_obs (int, optional): The minimum number of observations required for the approximation. Defaults to 25.
        return_approx_objects (bool, optional): Whether to return the list of CubicApproximation objects along with the approximated values. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[list[CubicApproximation], np.ndarray]]: The approximated values. If `return_approx_objects` is True, returns a tuple containing the approximated values and the list of CubicApproximation objects.
    """
    # Cubic spline approximation
    # least squares
    approx_list = [CubicApproximation(function_of, values[:, i], int_size, min_obs) for i in range(values.shape[1])]

    approx_values = np.column_stack([ap.est_obs for ap in approx_list if ap.est_obs is not None])

    return (approx_values, approx_list) if return_approx_objects else approx_values
