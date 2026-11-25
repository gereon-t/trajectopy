"""Temporal correlation of irregularly sampled time series with gaps (e.g. deviations from a reference)."""

import numpy as np
from scipy import fft

# minimum number of sample pairs for an autocorrelation value
MIN_PAIRS = 10
# the autocorrelation is computed up to this fraction of the number of samples (in sampling intervals), since the
# values of larger lags rely on too few pairs
MAX_LAG_FRACTION = 0.1


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
        max_lag (float, optional): Largest lag [s]. In any case, the lags go up to MAX_LAG_FRACTION of the number
            of samples (n / 10 sampling intervals).

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

    max_index = max(1, int(len(first) * MAX_LAG_FRACTION))
    if max_lag is not None:
        max_index = min(max_index, int(np.floor(max_lag / step)))
    max_index = min(max_index, len(grid) - 1)
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


def correlation_time(lags: np.ndarray, rho: np.ndarray, level: float = 1.0) -> float:
    """Lag at which the autocorrelation first drops below level / e, linearly interpolated.

    Args:
        lags (np.ndarray): Lags [s], starting with 0.
        rho (np.ndarray): Autocorrelation of the lags.
        level (float, optional): Autocorrelation at lag 0 of the decaying part, e.g. the correlated share of
            white_noise_split (the part above it drops at once). Defaults to 1.0 (all of it).

    Returns:
        float: Correlation time [s]; inf if the autocorrelation stays above level / e up to the largest lag.
    """
    threshold = level * np.exp(-1.0)
    rho = np.array(rho, dtype=float)
    rho[0] = level  # the decaying part starts at its level (the white noise drops between lag 0 and the next)
    below = np.flatnonzero(np.nan_to_num(rho, nan=np.inf) < threshold)
    if len(below) == 0:
        return float("inf")
    i = below[0]
    if i == 0:
        return 0.0
    # between the last lag above and the first below
    return float(np.interp(threshold, [rho[i], rho[i - 1]], [lags[i], lags[i - 1]]))


def white_noise_split(lags: np.ndarray, rho: np.ndarray, pairs: np.ndarray, max_fit_lags: int = 10) -> float:
    """Correlated share of the variance of a series of white noise and a correlated part.

    The autocorrelation of such a series drops by the white noise share between lag 0 and the next lag; the
    correlated part decays from there. Its share is extrapolated to lag 0 with an exponential decay fitted to the
    first lags (at most max_fit_lags, within the first decay by 1/e). If the first lag is not significantly
    correlated (less than two standard errors, 2 / sqrt(pairs)), the series is white noise.

    Args:
        lags (np.ndarray): Lags [s], starting with 0.
        rho (np.ndarray): Autocorrelation of the lags.
        pairs (np.ndarray): Number of pairs of samples of the lags.
        max_fit_lags (int, optional): Largest number of lags of the fit. Defaults to 10.

    Returns:
        float: Correlated share of the variance (0: white noise, 1: no white noise).
    """
    if len(rho) < 2 or not np.isfinite(rho[1]) or pairs[1] == 0 or rho[1] < 2 / np.sqrt(pairs[1]):
        return 0.0
    first = rho[1]
    fit = rho[1 : max_fit_lags + 1]
    # the lags within the first decay by 1/e, where the decay is still well determined
    used = np.cumprod(np.isfinite(fit) & (fit > first / np.e)).astype(bool)
    if np.sum(used) < 2:
        return float(min(first, 1.0))  # decays within one lag: no extrapolation
    slope, intercept = np.polyfit(lags[1 : max_fit_lags + 1][used], np.log(fit[used]), 1)
    # at least the value of the first lag (the correlated part does not grow), at most everything
    return float(np.clip(np.exp(intercept), first, 1.0)) if slope < 0 else float(min(first, 1.0))


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


DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 1000


def block_length(timestamps: np.ndarray, values: np.ndarray) -> int:
    """Block length (in samples) for a block bootstrap of a correlated series: twice its correlation time (the
    distance between independent samples, see effective_samples), at least one sample and at most a quarter of
    the series."""
    num_samples = len(values)
    step = sampling_interval(timestamps)
    if num_samples < 4 or step == 0:
        return 1
    lags, rho, pairs = autocorrelation(timestamps, values)
    correlated = white_noise_split(lags, rho, pairs)
    tau = correlation_time(lags, rho, level=correlated) if correlated > 0 else 0.0
    if not np.isfinite(tau):
        tau = float(lags[-1])
    return int(np.clip(np.ceil(2 * tau / step), 1, max(1, num_samples // 4)))


def block_bootstrap_ci(
    values: np.ndarray,
    timestamps: np.ndarray | None = None,
    statistic: str = "mean",
    confidence: float = DEFAULT_CONFIDENCE,
    num_resamples: int = DEFAULT_RESAMPLES,
    block: int | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Confidence interval of the mean or RMS of a series with a moving block bootstrap.

    Deviations of trajectories are correlated in time, so that the ordinary standard error (std / sqrt(n))
    underestimates the uncertainty of their mean by far. The moving block bootstrap resamples blocks of
    consecutive values that are about as long as the correlation, which keeps the correlation within each
    resampled series. Block sums come from cumulative sums, so that even millions of values take little time.

    Args:
        values (np.ndarray): Values in chronological order.
        timestamps (np.ndarray | None, optional): Times of the values [s], used to derive the block length from
            the correlation time. Without them (and without block), the values are treated as uncorrelated.
        statistic (str, optional): ``mean`` or ``rms``. Defaults to ``mean``.
        confidence (float, optional): Confidence level. Defaults to 0.95.
        num_resamples (int, optional): Number of bootstrap resamples. Defaults to 1000.
        block (int | None, optional): Block length in samples. Defaults to twice the correlation time
            (see block_length).
        seed (int, optional): Seed of the random numbers, so that the interval is reproducible. Defaults to 0.

    Returns:
        tuple[float, float]: Lower and upper bound (NaN for fewer than two values).
    """
    if statistic not in ("mean", "rms"):
        raise ValueError(f"Unknown statistic '{statistic}', use 'mean' or 'rms'.")
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)] if timestamps is None else values
    if timestamps is not None:
        valid = np.isfinite(values) & np.isfinite(timestamps)
        values, timestamps = values[valid], np.asarray(timestamps, dtype=float)[valid]
    num_samples = len(values)
    if num_samples < 2:
        return float("nan"), float("nan")

    if block is None:
        block = block_length(timestamps, values) if timestamps is not None else 1
    block = int(np.clip(block, 1, num_samples))

    samples = values**2 if statistic == "rms" else values
    cumulative = np.concatenate([[0.0], np.cumsum(samples)])
    num_blocks = int(np.ceil(num_samples / block))
    rng = np.random.default_rng(seed)
    statistics = np.empty(num_resamples)
    # in chunks, so that the memory stays bounded for many blocks
    chunk = max(1, 10_000_000 // num_blocks)
    for first in range(0, num_resamples, chunk):
        count = min(chunk, num_resamples - first)
        starts = rng.integers(0, num_samples - block + 1, size=(count, num_blocks))
        sums = np.sum(cumulative[starts + block] - cumulative[starts], axis=1)
        statistics[first : first + count] = sums / (num_blocks * block)
    if statistic == "rms":
        statistics = np.sqrt(statistics)

    alpha = (1.0 - confidence) / 2
    lower, upper = np.quantile(statistics, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def segment_ids(index: np.ndarray, window: float) -> np.ndarray:
    """Number of the window of length ``window`` that each index value (time or path length) falls into,
    counted from the smallest value."""
    index = np.asarray(index, dtype=float)
    if window <= 0:
        raise ValueError("The segment length must be positive.")
    if len(index) == 0:
        return np.array([], dtype=np.int64)
    return np.floor((index - np.min(index)) / window).astype(np.int64)
