"""Temporal correlation of absolute trajectory deviations (see ATEResult.correlation)."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from trajectopy.utils.statistics import (
    autocorrelation,
    correlation_time,
    effective_samples,
    sampling_interval,
    windowed_std,
)

if TYPE_CHECKING:
    from trajectopy.results.ate_result import ATEResult

# longest lag of the autocorrelation [s] (at most half of the time span is used)
DEFAULT_MAX_LAG = 600.0
# length of the windows of the windowed standard deviations [s]; for short data, a tenth of the duration
DEFAULT_WINDOW = 60.0
MIN_WINDOWS = 10

POSITION_COMPONENTS = ("along", "cross_h", "cross_v", "x", "y", "z")
ROTATION_COMPONENTS = ("roll", "pitch", "yaw")
COMPONENT_LABELS = {
    "along": "Along-Track",
    "cross_h": "Horizontal Cross-Track",
    "cross_v": "Vertical Cross-Track",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "roll": "Roll",
    "pitch": "Pitch",
    "yaw": "Yaw",
}


@dataclass
class ComponentCorrelation:
    """Temporal correlation of one deviation component.

    Attributes:
        lags (np.ndarray): Time lags [s].
        autocorrelation (np.ndarray): Autocorrelation of each lag (NaN if too few pairs of samples).
        pairs (np.ndarray): Number of pairs of samples of each lag.
        correlation_time (float): Lag at which the autocorrelation drops below 1/e [s]; inf if it does not
            within the largest lag.
        effective_samples (float): Approximate number of independent deviations.
        std (float): Standard deviation of the component (around its mean).
        window_centers (np.ndarray): Centers of the windows of window_std [s].
        window_std (np.ndarray): Standard deviation in consecutive windows.
    """

    lags: np.ndarray
    autocorrelation: np.ndarray
    pairs: np.ndarray
    correlation_time: float
    effective_samples: float
    std: float
    window_centers: np.ndarray
    window_std: np.ndarray


@dataclass
class DeviationCorrelation:
    """Temporal correlation of the deviations of an ATE result.

    Deviations of trajectories (e.g. GNSS/INS) are usually correlated in time over seconds to minutes. Then,
    neighboring epochs do not provide independent information, and the number of independent deviations is much
    smaller than the number of epochs. Gaps (e.g. losses of a total station's lock) are respected: no deviation
    is correlated across a gap.

    Attributes:
        sampling_interval (float): Typical time between the deviations [s].
        num_samples (int): Number of deviations.
        components (dict[str, ComponentCorrelation]): Correlation of each component (see COMPONENT_LABELS).
        cross_correlation_directed (np.ndarray): Correlation coefficients of along-track, horizontal and vertical
            cross-track at the same epochs (3x3).
        cross_correlation_xyz (np.ndarray): Correlation coefficients of x, y and z at the same epochs (3x3).
    """

    sampling_interval: float
    num_samples: int
    components: dict[str, ComponentCorrelation] = field(default_factory=dict)
    cross_correlation_directed: np.ndarray = field(default_factory=lambda: np.full((3, 3), np.nan))
    cross_correlation_xyz: np.ndarray = field(default_factory=lambda: np.full((3, 3), np.nan))

    @classmethod
    def from_ate_result(
        cls, ate_result: "ATEResult", max_lag: float = DEFAULT_MAX_LAG, window: float = DEFAULT_WINDOW
    ) -> "DeviationCorrelation":
        """Analyzes the deviations of an ATE result in chronological order.

        Args:
            ate_result (ATEResult): Result to analyze.
            max_lag (float, optional): Largest lag of the autocorrelation [s]. Defaults to DEFAULT_MAX_LAG.
            window (float, optional): Length of the windows of the windowed standard deviations [s], at most a
                tenth of the duration (at least MIN_WINDOWS windows). Defaults to DEFAULT_WINDOW.
        """
        timestamps = ate_result.trajectory.timestamps
        if (duration := float(np.ptp(timestamps)) if len(timestamps) else 0.0) > 0:
            window = min(window, duration / MIN_WINDOWS)
        values = {
            "along": ate_result.pos_dev_along,
            "cross_h": ate_result.pos_dev_cross_h,
            "cross_v": ate_result.pos_dev_cross_v,
            "x": ate_result.pos_dev_x,
            "y": ate_result.pos_dev_y,
            "z": ate_result.pos_dev_z,
        }
        if ate_result.has_orientation:
            values |= {"roll": ate_result.rot_dev_x, "pitch": ate_result.rot_dev_y, "yaw": ate_result.rot_dev_z}

        step = sampling_interval(timestamps)
        correlation = cls(sampling_interval=step, num_samples=len(timestamps))
        # all components at once: the grid of the epochs and the pairs of each lag are the same
        lags, rho, pairs = autocorrelation(timestamps, np.column_stack(list(values.values())), max_lag=max_lag)
        for column, (name, component) in enumerate(values.items()):
            tau = correlation_time(lags, rho[:, column])
            window_centers, window_std = windowed_std(timestamps, component, window)
            correlation.components[name] = ComponentCorrelation(
                lags=lags,
                autocorrelation=rho[:, column],
                pairs=pairs,
                correlation_time=tau,
                # the time covered by the deviations, without gaps
                effective_samples=effective_samples(len(component), len(component) * step, tau),
                std=float(np.std(component)) if len(component) else float("nan"),
                window_centers=window_centers,
                window_std=window_std,
            )

        correlation.cross_correlation_directed = _correlation_matrix(ate_result.abs_dev.directed_pos_dev)
        correlation.cross_correlation_xyz = _correlation_matrix(ate_result.abs_dev.pos_dev)
        return correlation


def _correlation_matrix(values: np.ndarray) -> np.ndarray:
    """Correlation coefficients of the columns; NaN for columns without variation."""
    if len(values) < 2:
        return np.full((values.shape[1], values.shape[1]), np.nan)
    std = np.std(values, axis=0)
    centered = values - np.mean(values, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (centered.T @ centered / len(values)) / np.outer(std, std)
