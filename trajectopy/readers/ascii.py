import logging
import re
from dataclasses import dataclass
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
from pyparsing import Callable

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import ComparisonMethod, PairDistanceUnit
from trajectopy.readers.utils import get_rot_matrix
from trajectopy.utils.definitions import (
    GPS_LEAP_SECONDS,
    GPS_WEEK_ZERO,
    TIME_FORMAT_DICT,
    TimeFormat,
)

logger = logging.getLogger(__name__)

HEADER_KEYS = [
    "epsg",
    "name",
    "sorting",
    "state",
    "fields",
    "delimiter",
    "nframe",
    "type",
    "comparison_method",
    "relative_dist_unit",
    "num_pairs",
    "rot_unit",
    "time_format",
    "datetime_format",
    "datetime_timezone",
    "id",
    "gps_week",
    "time_offset",
]


def default_line_handler(line: str) -> str:
    return " ".join(line.split()[1:])


def integer_line_handler(line: str) -> int:
    return int(line.split()[1])


def float_line_handler(line: str) -> float:
    return float(line.split()[1])


def delimiter_line_handler(line: str) -> str:
    """This function extracts the delimiter from the file header. All characters between the first and the last quotation mark are returned."""
    if delimiter_match := re.search(r"#delimiter ['\"](.)['\"]", line):
        delimiter = delimiter_match[1]
        logger.info("Detected delimiter %s", delimiter)
        return delimiter

    return ","


HANDLER_MAPPING: dict[str, Callable[[str], str | int | float]] = {
    "default": default_line_handler,
    "epsg": integer_line_handler,
    "delimiter": delimiter_line_handler,
    "gps_week": integer_line_handler,
    "time_offset": float_line_handler,
}


@dataclass
class HeaderData:
    """Class to store the header data of a trajectopy file."""

    data: dict[str, str | int | float]

    @property
    def id(self) -> str:
        return str(self.data.get("id", ""))

    @property
    def epsg(self) -> int:
        return int(self.data.get("epsg", 0))

    @property
    def name(self) -> str:
        return str(self.data.get("name", "Trajectory"))

    @property
    def rot_unit(self) -> str:
        return str(self.data.get("rot_unit", "rad"))

    @property
    def fields(self) -> list[str]:
        return str(self.data.get("fields", "t,px,py,pz,qx,qy,qz,qw")).split(",")

    @property
    def delimiter(self) -> str:
        return str(self.data.get("delimiter", ","))

    @property
    def nframe(self) -> str:
        return str(self.data.get("nframe", "enu")).lower()

    @property
    def sorting(self) -> str:
        return str(self.data.get("sorting", "time")).lower()

    @property
    def state(self) -> str:
        return str(self.data.get("state", "")).lower()

    @property
    def type(self) -> str:
        return str(self.data.get("type", "trajectoryentry")).lower()

    @property
    def comparison_method(self) -> ComparisonMethod:
        return ComparisonMethod.from_string(str(self.data.get("comparison_method", "unknown")))

    @property
    def relative_dist_unit(self) -> PairDistanceUnit:
        return PairDistanceUnit.from_str(str(self.data.get("relative_dist_unit", "meter")))

    @property
    def num_pairs(self) -> list[int]:
        return [int(item) for item in str(self.data.get("num_pairs", "0")).split(",")]

    @property
    def time_format(self) -> TimeFormat:
        return TIME_FORMAT_DICT[str(self.data.get("time_format", "unix")).lower()]

    @property
    def gps_week(self) -> int:
        return int(self.data.get("gps_week", np.floor((datetime.now() - GPS_WEEK_ZERO).days / 7)))

    @property
    def time_offset(self) -> float:
        return float(self.data.get("time_offset", 0.0))

    @property
    def datetime_format(self) -> str:
        return str(self.data.get("datetime_format", "%Y-%m-%d %H:%M:%S.%f"))

    @property
    def datetime_timezone(self) -> str:
        return str(self.data.get("datetime_timezone", "UTC"))

    @staticmethod
    def handle_line(metadata: dict[str, str | int | float], line: str) -> None:
        if not line.startswith("#"):
            return

        splitted_line = line.split()
        keyword = splitted_line[0][1:]

        if keyword in HEADER_KEYS:
            metadata[keyword] = HANDLER_MAPPING.get(keyword, HANDLER_MAPPING["default"])(line)

    @classmethod
    def from_file(cls, filename: str) -> "HeaderData":
        """Reads the header of a trajectory file.

        Args:
            filename (str): The path to the file.

        Returns:
            HeaderData: The header data.
        """
        metadata: dict[str, str | int | float] = {}
        with open(filename, encoding="utf-8") as file:
            for line in file:
                cls.handle_line(metadata, line)

        logger.info("Read header of %s", filename)
        return cls(metadata)

    @classmethod
    def from_string(cls, input_str: str) -> "HeaderData":
        """Reads the header of an input string.

        Args:
            input_str (str): The header string.

        Returns:
            HeaderData: The header data.
        """
        metadata: dict[str, str | int | float] = {}
        for line in input_str.splitlines():
            cls.handle_line(metadata, line)

        logger.info("Read header from string")
        return cls(metadata)


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
    input_str = StringIO(input_str)
    try:
        data = pd.read_csv(input_str, comment="#", header=None, sep=header_data.delimiter).to_numpy(dtype=dtype)

        if data.shape[1] == 1:
            logger.info("Assuming whitespaces as delimiter since imported data has only one column.")
            data = pd.read_csv(input_str, comment="#", header=None, sep="\\s+").to_numpy(dtype=dtype)
    except Exception:
        try:
            data = pd.read_csv(input_str, comment="#", header=None, sep="\\s+").to_numpy(dtype=dtype)
        except Exception:
            logger.warning("Could not read string using pandas. Trying numpy instead.")
            data = np.loadtxt(input_str, comments="#")
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
    return Rotations.from_euler(
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
        np.ndarray: GPS seconds of week (SOW) read from the trajectory file
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
    """
    return Positions(
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
