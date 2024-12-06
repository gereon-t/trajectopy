"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Union

import numpy as np

from trajectopy.core.definitions import GPS_WEEK_ZERO, TIME_FORMAT_DICT, TimeFormat
from trajectopy.core.settings.comparison import ComparisonMethod, PairDistanceUnit

logger = logging.getLogger("root")

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


HANDLER_MAPPING: Dict[str, Callable[[str], Union[str, int, float]]] = {
    "default": default_line_handler,
    "epsg": integer_line_handler,
    "delimiter": delimiter_line_handler,
    "gps_week": integer_line_handler,
    "time_offset": float_line_handler,
}


@dataclass
class HeaderData:
    """Class to store the header data of a trajectopy file."""

    data: Dict[str, Union[str, int, float]]

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
    def fields(self) -> List[str]:
        return str(self.data.get("fields", "t,px,py,pz,qx,qy,qz,qw")).split(",")

    @property
    def delimiter(self) -> str:
        return str(self.data.get("delimiter", ","))

    @property
    def nframe(self) -> str:
        return str(self.data.get("nframe", "enu")).lower()

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
    def num_pairs(self) -> List[int]:
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
    def handle_line(metadata: Dict[str, Union[str, int, float]], line: str) -> None:
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
        metadata: Dict[str, Union[str, int, float]] = {}
        with open(filename, "r", encoding="utf-8") as file:
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
        metadata: Dict[str, Union[str, int, float]] = {}
        for line in input_str.splitlines():
            cls.handle_line(metadata, line)

        logger.info("Read header from string")
        return cls(metadata)
