"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from io import StringIO
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from pointset import PointSet

from trajectopy.core.definitions import GPS_LEAP_SECONDS, GPS_WEEK_ZERO, TimeFormat
from trajectopy.core.input_output.header import HeaderData
from trajectopy.core.input_output.utils import get_rot_matrix
from trajectopy.core.rotationset import RotationSet

logger = logging.getLogger("root")


def read_data(filename: str, dtype=float) -> Tuple[HeaderData, np.ndarray]:
    """Reads the header and the data from a file

    By default, the trajectory data is read using pandas. If this fails,
    numpy is used instead.

    Args:
        filename (str): File to read

    Returns:
        Tuple[HeaderData, np.ndarray]: Header data and data
    """
    header_data = HeaderData.from_file(filename)
    try:
        data = pd.read_csv(filename, comment="#", header=None, sep=header_data.delimiter).to_numpy(dtype=dtype)

        if data.shape[1] == 1:
            logger.info("Assuming whitespaces as delimiter since imported data has only one column.")
            data = pd.read_csv(filename, comment="#", header=None, delim_whitespace=True).to_numpy(dtype=dtype)
    except Exception:
        try:
            data = pd.read_csv(filename, comment="#", header=None, delim_whitespace=True).to_numpy(dtype=dtype)
        except Exception:
            logger.warning("Could not read file using pandas. Trying numpy instead.")
            data = np.loadtxt(filename, comments="#")
    return header_data, data


def read_string(input_str: str, dtype=float) -> Tuple[HeaderData, np.ndarray]:
    """Reads the header and the data from a string

    By default, the trajectory data is read using pandas. If this fails,
    numpy is used instead.

    Args:
        input_str (str): String to read

    Returns:
        Tuple[HeaderData, np.ndarray]: Header data and data
    """
    header_data = HeaderData.from_string(input_str)
    input_str = StringIO(input_str)
    try:
        data = pd.read_csv(input_str, comment="#", header=None, sep=header_data.delimiter).to_numpy(dtype=dtype)

        if data.shape[1] == 1:
            logger.info("Assuming whitespaces as delimiter since imported data has only one column.")
            data = pd.read_csv(input_str, comment="#", header=None, delim_whitespace=True).to_numpy(dtype=dtype)
    except Exception:
        try:
            data = pd.read_csv(input_str, comment="#", header=None, delim_whitespace=True).to_numpy(dtype=dtype)
        except Exception:
            logger.warning("Could not read string using pandas. Trying numpy instead.")
            data = np.loadtxt(input_str, comments="#")
    return header_data, data


def extract_trajectory_rotations(header_data: HeaderData, trajectory_data: np.ndarray) -> Union[RotationSet, None]:
    """Extracts rotations from trajectory data and returns them as RotationSet

    Loaded rotations are converted to refer to the ENU navigation frame. For this,
    the actual navigation frame must be specified in the header of the trajectory file using
    the #nframe tag. Otherwise, the default ENU frame is assumed.

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        RotationSet: Rotations read from the trajectory file
    """
    rot = None
    if all(field in header_data.fields for field in ["qx", "qy", "qz", "qw"]):
        rot = extract_quaternions(header_data, trajectory_data)

    if all(field in header_data.fields for field in ["ex", "ey", "ez"]) and rot is None:
        rot = extract_euler_angles(header_data, trajectory_data)

    if rot is None:
        return rot

    enu_rot = RotationSet.from_matrix(get_rot_matrix(header_data.nframe))
    return enu_rot * rot


def extract_quaternions(header_data: HeaderData, trajectory_data: np.ndarray) -> RotationSet:
    """Extracts quaternions from trajectory data and returns them as RotationSet

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        RotationSet: Rotations read from the trajectory file
    """
    return RotationSet.from_quat(
        trajectory_data[
            :,
            [
                header_data.fields.index("qx"),
                header_data.fields.index("qy"),
                header_data.fields.index("qz"),
                header_data.fields.index("qw"),
            ],
        ].astype(float)
    )


def extract_euler_angles(header_data: HeaderData, trajectory_data: np.ndarray) -> RotationSet:
    """Extracts euler angles from trajectory data and returns them as RotationSet

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        RotationSet: Rotations read from the trajectory file
    """
    return RotationSet.from_euler(
        seq="xyz",
        angles=trajectory_data[
            :,
            [
                header_data.fields.index("ex"),
                header_data.fields.index("ey"),
                header_data.fields.index("ez"),
            ],
        ].astype(float),
        degrees=header_data.rot_unit == "deg",
    )


def extract_trajectory_timestamps(header_data: HeaderData, trajectory_data: np.ndarray) -> np.ndarray:
    """Extracts timestamps from trajectory data and returns them as numpy array

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        np.ndarray: Timestamps read from the trajectory file
    """

    time_columns = [pos for pos, char in enumerate(header_data.fields) if char == "t"]

    if header_data.time_format == TimeFormat.UNIX and len(time_columns) == 1:
        return trajectory_data[:, header_data.fields.index("t")].astype(float) + header_data.time_offset

    if header_data.time_format == TimeFormat.DATETIME and time_columns:
        return (
            parse_datetime(
                trajectory_data=trajectory_data,
                time_columns=time_columns,
                header_data=header_data,
            )
            + header_data.time_offset
        )

    if header_data.time_format == TimeFormat.GPS_SOW and time_columns:
        return (
            parse_gps_sow(
                trajectory_data=trajectory_data,
                time_columns=time_columns,
                header_data=header_data,
            )
            + header_data.time_offset
        )

    logger.warning("To timestamps found.")
    return np.array(range(len(trajectory_data)))


def parse_datetime(trajectory_data: np.ndarray, time_columns: List[int], header_data: HeaderData) -> np.ndarray:
    """Parses datetime strings to timestamps

    Args:
        trajectory_data (np.ndarray): Holds the trajectory data
        time_columns (list[int]): Indices of the column containing the datetime strings
        header_data (HeaderData): Holds information about the header of the trajectory file

    Returns:
        np.ndarray: Timestamps read from the trajectory file
    """

    def concatenate_strings(arr, delimiter=" "):
        return delimiter.join(arr)

    datetime_strings = np.apply_along_axis(concatenate_strings, 1, trajectory_data[:, time_columns])

    ts_datetime = pd.to_datetime(datetime_strings, format=header_data.datetime_format)

    if header_data.datetime_timezone.upper() == "GPS":
        ts_datetime -= pd.Timedelta(seconds=GPS_LEAP_SECONDS)
        time_zone = "UTC"
        logger.info("Applied GPS leap seconds.")
    else:
        time_zone = header_data.datetime_timezone

    ts_datetime = pd.DatetimeIndex(ts_datetime).tz_localize(tz=time_zone)
    logger.info("Timezone: %s", time_zone)

    return np.array([dt_i.timestamp() for dt_i in ts_datetime])


def parse_gps_sow(trajectory_data: np.ndarray, time_columns: List[int], header_data: HeaderData) -> np.ndarray:
    """Parses GPS seconds of week to timestamps

    Args:
        trajectory_data (np.ndarray): Holds the trajectory data
        time_columns (list[int]): Indices of the column containing the GPS seconds of week
        header_data (HeaderData): Holds information about the header of the trajectory file

    Returns:
        np.ndarray: GPS seconds of week (SOW) read from the trajectory file
    """
    return (
        trajectory_data[:, time_columns].astype(float).flatten()
        + header_data.gps_week * 604800
        - GPS_LEAP_SECONDS
        + GPS_WEEK_ZERO.timestamp()
    )


def extract_trajectory_speed(header_data: HeaderData, trajectory_data: np.ndarray) -> Union[None, np.ndarray]:
    """Extracts speed from trajectory data and returns them as numpy array

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        np.ndarray: Speeds read from the trajectory file
    """
    return (
        None
        if any(item not in header_data.fields for item in ["vx", "vy", "vz"])
        else trajectory_data[
            :,
            [
                header_data.fields.index("vx"),
                header_data.fields.index("vy"),
                header_data.fields.index("vz"),
            ],
        ].astype(float)
    )


def extract_trajectory_arc_lengths(header_data: HeaderData, trajectory_data: np.ndarray) -> Union[None, np.ndarray]:
    """Extracts arc lengths from trajectory data and returns them as numpy array

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        np.ndarray: Arc lengths read from the trajectory file
    """
    return None if "l" not in header_data.fields else trajectory_data[:, header_data.fields.index("l")].astype(float)


def extract_trajectory_pointset(header_data: HeaderData, trajectory_data: np.ndarray) -> PointSet:
    """Extracts positions from pandas DataFrame and returns a PointSet

    The positions of 'px', 'py', 'pz' are used as indices to access
    the DataFrame.

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        PointSet: PointSet object containing the parsed positions.
    """
    return PointSet(
        xyz=trajectory_data[
            :,
            [
                header_data.fields.index("px"),
                header_data.fields.index("py"),
                header_data.fields.index("pz"),
            ],
        ].astype(float),
        epsg=header_data.epsg,
    )
