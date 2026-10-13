"""Reading of ASCII (CSV-like) trajectory files."""

import logging
from io import StringIO

import numpy as np
import pandas as pd

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.io.header import HeaderData
from trajectopy.io.utils import get_rot_matrix
from trajectopy.utils.definitions import GPS_LEAP_SECONDS, GPS_WEEK_ZERO, TimeFormat

__all__ = [
    "HeaderData",
    "read_data",
    "read_string",
    "extract_trajectory_rotations",
    "extract_trajectory_timestamps",
    "extract_trajectory_velocity_xyz",
    "extract_trajectory_path_lengths",
    "extract_trajectory_positions",
]

logger = logging.getLogger(__name__)


def read_data(filename: str, dtype=float) -> tuple[HeaderData, np.ndarray]:
    """Reads the header and the data from a file

    By default, the trajectory data is read using pandas. If this fails,
    numpy is used instead.

    Args:
        filename (str): File to read

    Returns:
        Tuple[HeaderData, np.ndarray]: Header data and data
    """
    header_data = HeaderData.from_file(filename)

    with open(filename, encoding="utf-8") as handle:
        has_data = any(line.strip() and not line.lstrip().startswith("#") for line in handle)
    if not has_data:
        logger.info("No trajectory data found in %s", filename)
        return header_data, np.empty((0, len(header_data.fields)))

    try:
        data = pd.read_csv(filename, comment="#", header=None, sep=header_data.delimiter).to_numpy(dtype=dtype)

        if data.shape[1] == 1:
            logger.info("Assuming whitespaces as delimiter since imported data has only one column.")
            data = pd.read_csv(filename, comment="#", header=None, sep="\\s+").to_numpy(dtype=dtype)
    except Exception:
        try:
            data = pd.read_csv(filename, comment="#", header=None, sep="\\s+").to_numpy(dtype=dtype)
        except Exception:
            logger.warning("Could not read file using pandas. Trying numpy instead.")
            data = np.loadtxt(filename, comments="#")
    return header_data, data


def read_string(input_str: str, dtype=float) -> tuple[HeaderData, np.ndarray]:
    """Reads the header and the data from a string

    By default, the trajectory data is read using pandas. If this fails,
    numpy is used instead.

    Args:
        input_str (str): String to read

    Returns:
        Tuple[HeaderData, np.ndarray]: Header data and data
    """
    header_data = HeaderData.from_string(input_str)

    has_data = any(line.strip() and not line.lstrip().startswith("#") for line in input_str.splitlines())
    if not has_data:
        logger.info("No trajectory samples found in provided string input")
        return header_data, np.empty((0, len(header_data.fields)))

    try:
        string_io = StringIO(input_str)
        data = pd.read_csv(string_io, comment="#", header=None, sep=header_data.delimiter).to_numpy(dtype=dtype)

        if data.shape[1] == 1:
            logger.info("Assuming whitespaces as delimiter since imported data has only one column.")
            string_io = StringIO(input_str)  # Create fresh StringIO for re-reading
            data = pd.read_csv(string_io, comment="#", header=None, sep="\\s+").to_numpy(dtype=dtype)
    except Exception:
        try:
            string_io = StringIO(input_str)
            data = pd.read_csv(string_io, comment="#", header=None, sep="\\s+").to_numpy(dtype=dtype)
        except Exception:
            logger.warning("Could not read string using pandas. Trying numpy instead.")
            string_io = StringIO(input_str)
            data = np.loadtxt(string_io, comments="#")
    return header_data, data


def extract_trajectory_rotations(header_data: HeaderData, trajectory_data: np.ndarray) -> None | Rotations:
    """Extracts rotations from trajectory data and returns them as Rotations

    Loaded rotations are converted to refer to the ENU navigation frame. For this,
    the actual navigation frame must be specified in the header of the trajectory file using
    the #nframe tag. Otherwise, the default ENU frame is assumed.

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        Rotations | None: Rotations read from the trajectory file or None if no rotations are found
    """
    rot = None
    if all(field in header_data.fields for field in ["qx", "qy", "qz", "qw"]):
        rot = _extract_rotations_from_quaternions(header_data, trajectory_data)

    if all(field in header_data.fields for field in ["ex", "ey", "ez"]) and rot is None:
        rot = _extract_rotations_from_euler_angles(header_data, trajectory_data)

    if rot is None:
        return rot

    enu_rot = Rotations.from_matrix(get_rot_matrix(header_data.nframe))
    return enu_rot * rot


def _extract_rotations_from_quaternions(header_data: HeaderData, trajectory_data: np.ndarray) -> Rotations:
    """Extracts quaternions from trajectory data and returns them as Rotations

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        Rotations: Rotations read from the trajectory file
    """
    return Rotations.from_quat(
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


def _extract_rotations_from_euler_angles(header_data: HeaderData, trajectory_data: np.ndarray) -> Rotations:
    """Extracts euler angles from trajectory data and returns them as Rotations

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        Rotations: Rotations read from the trajectory file
    """
    logger.info(
        "Extracting rotations from euler angles with sequence '%s' and unit '%s'.",
        header_data.euler_rot_seq,
        header_data.rot_unit,
    )
    return Rotations.from_euler(
        seq=header_data.euler_rot_seq,
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
            _parse_datetime(
                trajectory_data=trajectory_data,
                time_columns=time_columns,
                header_data=header_data,
            )
            + header_data.time_offset
        )

    if header_data.time_format == TimeFormat.GPS_SOW and time_columns:
        return (
            _parse_gps_sow(
                trajectory_data=trajectory_data,
                time_columns=time_columns,
                header_data=header_data,
            )
            + header_data.time_offset
        )

    logger.warning("No timestamps found.")
    return np.array(range(len(trajectory_data)))


def _parse_datetime(trajectory_data: np.ndarray, time_columns: list[int], header_data: HeaderData) -> np.ndarray:
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

    ts_datetime = pd.to_datetime(datetime_strings, format=header_data.datetime_format, utc=True)

    if header_data.datetime_timezone.upper() == "GPS":
        ts_datetime -= pd.Timedelta(seconds=GPS_LEAP_SECONDS)
        time_zone = "UTC"
        logger.info("Applied GPS leap seconds.")
    else:
        time_zone = header_data.datetime_timezone

    ts_datetime = pd.DatetimeIndex(ts_datetime).tz_convert(tz=time_zone)
    logger.info("Timezone: %s", time_zone)

    return np.array([dt_i.timestamp() for dt_i in ts_datetime])


def _parse_gps_sow(trajectory_data: np.ndarray, time_columns: list[int], header_data: HeaderData) -> np.ndarray:
    """Parses GPS seconds of week to timestamps

    Args:
        trajectory_data (np.ndarray): Holds the trajectory data
        time_columns (list[int]): Indices of the column containing the GPS seconds of week
        header_data (HeaderData): Holds information about the header of the trajectory file

    Returns:
        np.ndarray: Unix timestamps converted from GPS seconds of week
    """
    return (
        trajectory_data[:, time_columns].astype(float).flatten()
        + header_data.gps_week * 604800
        - GPS_LEAP_SECONDS
        + GPS_WEEK_ZERO.timestamp()
    )


def extract_trajectory_velocity_xyz(header_data: HeaderData, trajectory_data: np.ndarray) -> None | np.ndarray:
    """Extracts speed from trajectory data and returns them as numpy array

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        np.ndarray: Speeds read from the trajectory file
    """
    if any(item not in header_data.fields for item in ["vx", "vy", "vz"]):
        return None

    velocity = trajectory_data[
        :,
        [
            header_data.fields.index("vx"),
            header_data.fields.index("vy"),
            header_data.fields.index("vz"),
        ],
    ].astype(float)

    rot_mat = get_rot_matrix(header_data.nframe)
    velocity = (rot_mat @ velocity.T).T
    logger.info("Applied nframe '%s' to velocity (converted to ENU).", header_data.nframe)
    return velocity


def extract_trajectory_path_lengths(header_data: HeaderData, trajectory_data: np.ndarray) -> None | np.ndarray:
    """Extracts arc lengths from trajectory data and returns them as numpy array

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        np.ndarray: Arc lengths read from the trajectory file
    """
    return None if "l" not in header_data.fields else trajectory_data[:, header_data.fields.index("l")].astype(float)


def extract_trajectory_positions(header_data: HeaderData, trajectory_data: np.ndarray) -> Positions:
    """Extracts positions from pandas DataFrame and returns a Positions object

    The positions of 'px', 'py', 'pz' are used as indices to access
    the DataFrame.

    Args:
        header_data (HeaderData): Holds information about the header of the trajectory file
        trajectory_data (np.ndarray): Holds the trajectory data

    Returns:
        Positions: Positions read from the trajectory file

    Raises:
        ValueError: If required position fields (px, py, pz) are not found in header
    """
    required_fields = ["px", "py", "pz"]
    missing_fields = [f for f in required_fields if f not in header_data.fields]
    if missing_fields:
        raise ValueError(
            f"Required position fields {missing_fields} not found in header. Available fields: {header_data.fields}"
        )

    xyz = trajectory_data[
        :,
        [
            header_data.fields.index("px"),
            header_data.fields.index("py"),
            header_data.fields.index("pz"),
        ],
    ].astype(float)

    if header_data.epsg == 0:
        rot_mat = get_rot_matrix(header_data.nframe)
        xyz = (rot_mat @ xyz.T).T
        logger.info("Applied nframe '%s' to positions (EPSG=0, converted to ENU).", header_data.nframe)
    else:
        logger.info(
            "Skipped nframe '%s' for positions (EPSG=%s, CRS-defined coordinates).",
            header_data.nframe,
            header_data.epsg,
        )

    return Positions(
        xyz=xyz,
        epsg=header_data.epsg,
    )
