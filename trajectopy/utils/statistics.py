"""Temporal correlation of irregularly sampled time series with gaps (e.g. deviations from a reference)."""

import numpy as np
from scipy import fft

# minimum number of sample pairs for an autocorrelation value
MIN_PAIRS = 10


def sampling_interval(timestamps: np.ndarray) -> float:
    """Typical sampling interval: the median of the positive time differences (0 if there are none)."""
    differences = np.diff(np.sort(timestamps))
    differences = differences[differences > 0]
    return float(np.median(differences)) if len(differences) else 0.0


def autocorrelation(
    timestamps: np.ndarray, values: np.ndarray, max_lag: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Autocorrelation of time series over the time lag, also for irregular sampling and gaps.

    The samples are placed on a regular grid with the typical sampling interval (each one in its nearest cell,
    the first one if several share a cell), where gaps stay empty. The autocorrelation of each lag is computed only
    from the pairs of samples that exist, so that nothing is correlated across a gap: for the lag k,
    rho(k) = sum(x_i * x_i+k) / pairs(k) / variance, with the mean removed. FFTs make it fast for long series.

    Args:
        timestamps (np.ndarray): Times of the samples [s], in any order.
        values (np.ndarray): Values of the samples, one series (n) or several (n x m) at the same times; samples
            with non-finite values or times are ignored.
        max_lag (float, optional): Largest lag [s]. Defaults to half of the time span.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Lags [s], autocorrelation of each lag (and series; NaN for lags
            with fewer than MIN_PAIRS pairs) and the number of pairs of each lag.
    """
    values = np.asarray(values, dtype=float)
    series = values.reshape(len(values), -1)
    valid = np.isfinite(timestamps) & np.all(np.isfinite(series), axis=1)
    timestamps, series = np.asarray(timestamps, dtype=float)[valid], series[valid]
    step = sampling_interval(timestamps)
    if len(series) < 2 or step == 0:
        rho = np.ones((1, series.shape[1]))
        return np.zeros(1), rho.reshape(1, *values.shape[1:]), np.array([len(series)])

    cells = np.round((timestamps - timestamps.min()) / step).astype(np.int64)
    first = np.unique(cells, return_index=True)[1]
    grid = np.zeros((int(cells.max()) + 1, series.shape[1]))
    grid[cells[first]] = series[first] - np.mean(series[first], axis=0)
    mask = np.zeros(len(grid))
    mask[cells[first]] = 1.0

    max_index = (len(grid) - 1) // 2 if max_lag is None else min(len(grid) - 1, int(np.floor(max_lag / step)))
    # zero padding up to the largest lag avoids the cyclic wrap-around of the FFT
    size = fft.next_fast_len(len(grid) + max_index + 1, real=True)
    pairs = np.rint(fft.irfft(np.abs(fft.rfft(mask, size, workers=-1)) ** 2, size, workers=-1)[: max_index + 1])
    products = fft.irfft(np.abs(fft.rfft(grid, size, axis=0, workers=-1)) ** 2, size, axis=0, workers=-1)
    products = products[: max_index + 1]

    variance = np.sum(grid**2, axis=0) / len(first)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = products / pairs[:, None] / np.where(variance > 0, variance, np.nan)
    rho[pairs < MIN_PAIRS] = np.nan
    rho[0] = np.where(variance > 0, 1.0, np.nan)
    return np.arange(max_index + 1) * step, rho.reshape(max_index + 1, *values.shape[1:]), pairs.astype(int)


def correlation_time(lags: np.ndarray, rho: np.ndarray) -> float:
    """Lag at which the autocorrelation first drops below 1/e, linearly interpolated.

    Returns:
        float: Correlation time [s]; inf if the autocorrelation stays above 1/e up to the largest lag.
    """
    threshold = np.exp(-1.0)
    below = np.flatnonzero(np.nan_to_num(rho, nan=np.inf) < threshold)
    if len(below) == 0:
        return float("inf")
    i = below[0]
    if i == 0:
        return 0.0
    # between the last lag above and the first below
    return float(np.interp(threshold, [rho[i], rho[i - 1]], [lags[i], lags[i - 1]]))


def effective_samples(num_samples: int, duration: float, tau: float) -> float:
    """Approximate number of independent samples of a series with the correlation time tau.

    For an exponentially decaying correlation, a series of the given duration holds about duration / (2 tau)
    independent samples (at most all samples, at least one).

    Args:
        num_samples (int): Number of samples.
        duration (float): Time covered by the samples [s] (without gaps).
        tau (float): Correlation time [s], e.g. from correlation_time.
    """
    if num_samples == 0:
        return 0.0
    if tau <= 0:
        return float(num_samples)
    return float(np.clip(duration / (2 * tau), 1.0, num_samples))


def windowed_std(
    timestamps: np.ndarray, values: np.ndarray, window: float, min_samples: int = MIN_PAIRS
) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviation in consecutive time windows, e.g. to see whether the noise changes over time.

    Returns:
        tuple[np.ndarray, np.ndarray]: Centers of the windows [s] and the standard deviations (NaN for windows
            with fewer than min_samples samples; windows without samples are left out).
    """
    timestamps, values = np.asarray(timestamps, dtype=float), np.asarray(values, dtype=float)
    if len(timestamps) == 0 or window <= 0:
        return np.array([]), np.array([])
    start = float(np.min(timestamps))
    windows = np.floor((timestamps - start) / window).astype(np.int64)
    counts = np.bincount(windows)
    # mean and variance per window from sums (centered by the overall mean for numerical accuracy)
    centered = values - np.mean(values)
    sums = np.bincount(windows, weights=centered)
    squares = np.bincount(windows, weights=centered**2)
    used = counts > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        variances = squares[used] / counts[used] - (sums[used] / counts[used]) ** 2
    stds = np.sqrt(np.maximum(variances, 0.0))
    stds[counts[used] < min_samples] = np.nan
    return start + (np.flatnonzero(used) + 0.5) * window, stds
