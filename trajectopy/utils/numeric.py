"""Small numeric helper functions."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def rms(x: np.ndarray | float) -> float:
    """
    Calculates the root mean square of an array.

    Args:
        x (np.ndarray): The input array.

    Returns:
        float: The root mean square of the input array.
    """
    return np.sqrt(np.mean(np.square(x)))


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
    index: np.ndarray, data: np.ndarray, resolution: float, filter_size: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduces the amount of deviations using smoothing and rounding

    It will first smooth the data using a convolution with a filter
    of size filter_size. Then, the data is rounded to the specified
    resolution and duplicate values that can result from this operation
    are deleted.

    Args:
        index (np.ndarray): nxm array that contains for example
                                  time stamps, arc lengths or positions
                                  corresponding to the data.
        data (np.ndarray): nx1 array that contains the data that should
                           be smoothed.
        resolution (float): Desired resolution
        filter_size (int): Window / filter size for smoothing

    Returns:
        Tuple[np.ndarray, np.ndarray]: downsampled index and data
    """
    data_smoothed = np.convolve(data, [1 / filter_size] * filter_size, "same")
    data_rounded = np.round(data_smoothed / resolution) * resolution
    _, indices = np.unique(np.c_[index, data_rounded], return_index=True, axis=0)
    indices_sorted = np.sort(indices)

    function_of_unique = index[indices_sorted, :] if index.ndim > 1 else index[indices_sorted]
    data_unique = data_smoothed[indices_sorted]

    return function_of_unique, data_unique


def shrink_data(data: np.ndarray, max_size: int = 1000) -> np.ndarray:
    """
    Shrink the given data to the given max_size by taking
    the moving average of the data.

    Args:
        data (np.ndarray): The data to shrink
        max_size (int): The maximum size of the data

    Returns:
        np.ndarray: The shrunk data
    """
    if max_size < 0:
        return data

    if len(data) <= max_size:
        return data

    data = data.astype(float)
    array_size = (-len(data) % max_size) + len(data)
    padded_data = np.pad(data, (0, array_size - len(data)))
    padded_data[padded_data == 0] = np.nan

    reshaped_data = np.array(
        [row for row in padded_data.reshape(-1, len(padded_data) // max_size) if not np.all(np.isnan(row))]
    )

    return np.nanmean(reshaped_data, 1)
