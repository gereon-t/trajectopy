"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from trajectopy.utils.common import sparse_least_squares

# logger configuration
logger = logging.getLogger(__name__)


def _compute_c(relative_length: float, interval_length: float) -> tuple[float, float, float, float]:
    """
    Helper function to compute the coefficients for cubic approximation
    """
    interval_ratio = relative_length / interval_length

    c0 = 1 - 3 * interval_ratio**2 + 2 * interval_ratio**3
    c1 = relative_length * (1 - 2 * interval_ratio + interval_ratio**2)
    c2 = 3 * interval_ratio**2 - 2 * interval_ratio**3
    c3 = relative_length * (interval_ratio**2 - interval_ratio)

    return c0, c1, c2, c3


@dataclass
class Interval:
    start: float
    end: float
    values: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def coefficients(self) -> list[np.ndarray]:
        return [np.array(_compute_c(v - self.start, self.end - self.start)) for v in self.values]


class Intervals:
    def __init__(self):
        self.intervals: list[Interval] = []


class CubicApproximation:
    """
    Class for piecewise cubic approximation
    """

    def __init__(self, index: np.ndarray, values: np.ndarray, min_win_size: float = 0.25, min_obs: int = 3) -> None:
        """
        Inititalization of a new CubicApproximation class object
        """
        self.index = index
        self.values = values
        self.min_win_size = min_win_size
        self.min_obs = max(3, min_obs)

        # fit results
        self.parameters: np.ndarray | None = None
        self.est_obs: np.ndarray | None = None
        self.residuals: np.ndarray | None = None
        self.interval_steps: np.ndarray | None = None

        # compute approximation
        self._cubic_approx()

    @property
    def interval_bounds(self) -> list[float]:
        """
        Get the interval bounds of the cubic approximation
        """
        if self.interval_steps is None:
            raise ValueError("Approximation not yet computed")
        return [np.min(self.interval_steps), np.max(self.interval_steps)]

    def evaluate_at(self, locations: np.ndarray) -> np.ndarray:
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

        relative_length = locations - int_start
        interval_length = int_end - int_start

        c0, c1, c2, c3 = _compute_c(relative_length, interval_length)

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
        var_red = self.index - self.index[0]

        intervals: list[Interval] = []
        current_interval_obj = Interval(start=0.0, end=0.0)

        for i, x_value in enumerate(var_red):
            current_interval_obj.values.append(x_value)
            current_interval_obj.end = x_value

            if len(current_interval_obj) >= self.min_obs and current_interval_obj.length > self.min_win_size:
                intervals.append(current_interval_obj)
                current_interval_obj = Interval(start=x_value, end=x_value)

            if (i == len(var_red) - 1) and (
                len(current_interval_obj) < self.min_obs or current_interval_obj.length < self.min_win_size
            ):
                intervals[-1].end = current_interval_obj.end
                intervals[-1].values.extend(current_interval_obj.values)

        logger.info(
            "Average observation count per interval: %.2f",
            len(var_red) / len(intervals),
        )

        t_final = [0.0] + [interval.end for interval in intervals]

        # Design matrix (jacobian)
        a_design = self._design_matrix(intervals)

        # least squares
        logger.info("Approximation using piece-wise cubic polynomials via least-squares method.")
        xS, lS, residuals = sparse_least_squares(csr_matrix(a_design), self.values[:, None])

        # store results
        self.parameters = xS
        self.est_obs = lS
        self.residuals = residuals
        self.interval_steps = t_final + self.index[0]

    @staticmethod
    def _compute_c(relative_length: float, interval_length: float) -> tuple[float, float, float, float]:
        """
        Helper function to compute the coefficients for cubic approximation
        """
        interval_ratio = relative_length / interval_length

        c0 = 1 - 3 * interval_ratio**2 + 2 * interval_ratio**3
        c1 = relative_length * (1 - 2 * interval_ratio + interval_ratio**2)
        c2 = 3 * interval_ratio**2 - 2 * interval_ratio**3
        c3 = relative_length * (interval_ratio**2 - interval_ratio)

        return c0, c1, c2, c3

    def _design_matrix(self, intervals: list[Interval]) -> lil_matrix:
        rows = len(self.index)
        columns = 2 * len(intervals) + 2

        a_design = lil_matrix((rows, columns), dtype=float)

        row_offset = 0
        for i, interval in enumerate(intervals):
            matrix = np.array(interval.coefficients)

            column_start = i * 2
            column_end = column_start + 4

            a_design[row_offset : row_offset + len(matrix), column_start:column_end] = matrix
            row_offset += len(matrix)

        return a_design
