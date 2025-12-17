import datetime
import logging
import os
import sys

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csr_matrix, identity, spdiags
from scipy.sparse.linalg import spsolve

from trajectopy.utils.definitions import Sorting, TrajectoriesSorting

logger = logging.getLogger(__name__)


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = os.path.join(sys._MEIPASS, "trajectopy")
    except Exception:
        base_path = os.path.dirname(__file__)
        base_path = os.path.dirname(base_path)

    return os.path.join(base_path, relative_path)


FULL_ICON_FILE_PATH = resource_path("gui/resources/full-icon-poppins.png")
ICON_FILE_PATH = resource_path("gui/resources/icon.png")
ICON_BG_FILE_PATH = resource_path("gui/resources/icon-bg.png")
YEAR = str(datetime.datetime.now().year)


def rms(x: np.ndarray | float) -> float:
    """
    Calculates the root mean square of an array.

    Args:
        x (np.ndarray): The input array.

    Returns:
        float: The root mean square of the input array.
    """
    return np.sqrt(np.mean(np.square(x)))


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


def dict2table(
    input_dict: dict,
    title: str,
    key_width: int = 16,
    value_width: int = 8,
    key_value_filler: str = " : ",
    decimal_places: int = 3,
) -> str:
    """Converts a dictionary to a formatted table string.

    Args:
        input_dict (dict): The dictionary to be converted to a table.
        title (str): The title of the table.
        key_width (int, optional): The width of the key column. Defaults to 16.
        value_width (int, optional): The width of the value column. Defaults to 8.
        key_value_filler (str, optional): The string used to separate the key and value columns. Defaults to " : ".
        decimal_places (int, optional): The number of decimal places to display for float values. Defaults to 3.

    Returns:
        str: The formatted table string.

    Example:
        >>> my_dict = {"key1": 123.456, "key2": "value2"}
        >>> dict2table(my_dict, "My Table")
        '\n ______________________\n| ------ My Table ------ |\n| key1            : 123.456 |  \n| key2            : value2  |  \n|________________________|\n'
    """
    if not key_value_filler:
        key_value_filler = " "

    # <|><space>item<space><|>
    left_right_width = 4

    # how much characters are the field rows
    max_value_width = max(value_width, max(len(str(value)) for value in input_dict.values()))
    max_key_width = max(key_width, max(len(str(key)) for key in input_dict))
    field_lengths = max_key_width + max_value_width + len(key_value_filler) + left_right_width

    # width of the table
    len_title = len(title) + 2 * left_right_width
    table_width = max(len_title, field_lengths)
    modulo_width = table_width % 2

    # how many dashes must be set for title padding
    title_padding = max(0, table_width - (len_title))
    right_title_padding = title_padding % 2 if title_padding != 0 else 0
    title_padding //= 2

    title_blank_padding = " " if modulo_width != 0 else ""
    # how many blank spaces must be set after the field row to match the table width
    table_width += modulo_width
    field_padding = max(0, table_width - field_lengths - 1)

    table_str = ""
    table_str += f"\n {'_'*(table_width-left_right_width)}\n"
    table_str += f"| {'-'*title_padding} {title} {'-'*(title_padding + right_title_padding)}{title_blank_padding} |\n"
    for key, value in input_dict.items():
        if isinstance(value, float):
            number_formatter = f".{decimal_places}f"
        else:
            value = str(value)
            number_formatter = ""
        table_str += f"| {key:<{max_key_width}}{key_value_filler}{value:<{max_value_width}{number_formatter}}{' '*field_padding}|\n"
    table_str += f"|{'_'*(table_width-left_right_width)}|\n"

    return table_str


def list2box(
    input_list: list,
    title: str = "",
    upper_boundary: bool = True,
    lower_boundary: bool = True,
    field_length: int = 0,
) -> str:
    """
    Converts a list to a formatted box string.

    Args:
        input_list (list): The list to be converted to a box.
        title (str, optional): The title of the box. Defaults to "".
        upper_boundary (bool, optional): Whether to include an upper boundary. Defaults to True.
        lower_boundary (bool, optional): Whether to include a lower boundary. Defaults to True.
        field_length (int, optional): The width of each field in the box. Default to 0 (auto).

    Returns:
        str: The formatted box string.

    Example:
        >>> my_list = ["item1", "item2", "item3"]
        >>> list2box(my_list, "My Box")
        '┌─────────── My Box ───────────┐\n│ item1                       │\n│ item2                       │\n│ item3                       │\n└─────────────────────────────┘\n'
    """
    if not input_list:
        return ""

    # <|><space>item<space><|>
    left_right_width = 4

    # how much characters are the field rows
    field_lengths = field_length if field_length > 0 else max(len(str(item)) for item in input_list) + left_right_width

    # width of the table
    len_title = len(title) + 2 * left_right_width
    table_width = max(len_title, field_lengths)
    modulo_width = table_width % 2

    # how many dashes must be set for title padding
    title_padding = max(0, table_width - (len_title))
    right_title_padding = title_padding % 2 if title_padding != 0 else 0
    title_padding //= 2

    title_blank_padding = " " if modulo_width != 0 else ""
    # how many blank spaces must be set after the field row to match the table width
    table_width += modulo_width

    table_str = ""
    if upper_boundary:
        table_str += f"\n {'_'*(table_width-left_right_width)}\n"

    if title:
        table_str += (
            f"| {'-'*title_padding} {title} {'-'*(title_padding + right_title_padding)}{title_blank_padding} |\n"
        )

    for item in input_list:
        table_str += f"| {item:<{field_lengths-left_right_width}}|\n"

    if lower_boundary:
        table_str += f"|{'_'*(table_width-left_right_width)}|\n"

    return table_str


def common_time_span(tstamps1: np.ndarray, tstamps2: np.ndarray) -> tuple[float, float] | None:
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

    if len(weights) == 0:
        weights = np.ones(len(y))

    sigma_ll = spdiags(weights, 0, len(weights), len(weights))

    # solve normal equation
    x_s, l_s, v = least_squares(design_matrix=A, observations=y, sigma_ll=sigma_ll)

    return x_s, l_s, v


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
    sigma_ll: np.ndarray = np.array([]),
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


def number_to_string(number: float) -> str:
    return f"{number:.3f}"


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


def get_sorting(sort_by_list: list[str]) -> TrajectoriesSorting:
    if all(sorting == Sorting.PATH_LENGTH for sorting in sort_by_list):
        return TrajectoriesSorting.ALL_SPATIAL

    if all(sorting == Sorting.TIME for sorting in sort_by_list):
        return TrajectoriesSorting.ALL_TIME

    logger.warning("Data is diffently sorted, weird things might happen.")
    return TrajectoriesSorting.MIXED


def derive_xlabel_from_sortings(trajectories_sorting: TrajectoriesSorting, all_unix: bool) -> str:
    if trajectories_sorting == TrajectoriesSorting.ALL_SPATIAL:
        return "trajectory length [m]"

    if trajectories_sorting == TrajectoriesSorting.ALL_TIME:
        return "time [hh:mm:ss]" if all_unix else "time [s]"

    logger.warning("Data is diffently sorted, weird things might happen.")
    return "time [s] / trajectory length [m]"


def get_axis_label(trajectories: list) -> tuple[str, str, str]:
    """Returns the unit of the axis"""
    if all(traj.positions.epsg == 0 for traj in trajectories):
        return "x [m]", "y [m]", "z [m]"

    unit_set = {
        traj.positions.crs.axis_info[0].unit_name if traj.positions.crs else "unknown" for traj in trajectories
    }
    unit_name = unit_set.pop().replace("metre", "m").replace("degree", "°")

    # there are multiple units
    if unit_set:
        return "x", "y", "z"

    axis_info = trajectories[0].positions.crs.axis_info
    x_axis_name = axis_info[0].name
    y_axis_name = axis_info[1].name

    if len(axis_info) > 2:
        z_axis_name = axis_info[2].name
        z_unit_name = axis_info[2].unit_name.replace("metre", "m").replace("degree", "°")
    else:
        # we assume meters as default
        z_axis_name = "z"
        z_unit_name = "m"

    return (
        f"{x_axis_name} [{unit_name}]",
        f"{y_axis_name} [{unit_name}]",
        f"{z_axis_name} [{z_unit_name}]",
    )


def set_aspect_equal_3d(ax):
    """
    https://stackoverflow.com/a/35126679
    """
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)

    plot_radius = max(
        abs(lim - mean_) for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean)) for lim in lims
    )

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
