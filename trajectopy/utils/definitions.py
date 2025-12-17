import datetime
from enum import Enum, auto

import matplotlib.dates as mdates
import numpy as np

UNIX_TIME_THRESHOLD = 1_000_000
DATE_FORMATTER = mdates.DateFormatter("%H:%M:%S")


class Sorting(str, Enum):
    TIME = "time"
    PATH_LENGTH = "path_length"

    @classmethod
    def from_str(cls, value: str) -> "Sorting":
        if value == "time":
            return Sorting.TIME
        elif value == "path_length" or value == "arc_length":
            return Sorting.PATH_LENGTH
        else:
            raise ValueError(f"Unknown sorting method: {value}")


class TrajectoriesSorting(Enum):
    ALL_TIME = auto()
    ALL_SPATIAL = auto()
    MIXED = auto()


class Unit(Enum):
    """
    Enumeration class handling different units
    """

    DEGREE = auto()
    RADIAN = auto()
    METER = auto()
    SECOND = auto()
    SCALE = auto()
    NONE = auto()

    @classmethod
    def from_str(cls, unit_str: str) -> "Unit":
        """
        Converts a string to a unit

        Args:
            unit_str (str): String to be converted

        Returns:
            Unit: Converted unit
        """
        if unit_str.lower() in {"°", "deg", "degree"}:
            return cls.DEGREE
        if unit_str.lower() in {"rad", "radian"}:
            return cls.RADIAN
        if unit_str.lower() in {"m", "meter"}:
            return cls.METER
        if unit_str.lower() in {"s", "sec", "second"}:
            return cls.SECOND
        if unit_str.lower() in {"-", "ppm", "scale"}:
            return cls.SCALE
        if not unit_str:
            return cls.NONE

        raise ValueError(f"Unknown unit string {unit_str}")


# [multiply with , unit string, number of decimals]
UNIT_FORMAT_RULES: dict[Unit, dict[str, tuple[float, str, int]]] = {
    Unit.METER: {"normal": (1, "m", 3), "precise": (1000, "mm", 1)},
    Unit.RADIAN: {"normal": (180 / np.pi, "°", 3)},
    Unit.DEGREE: {"normal": (1, "°", 3)},
    Unit.SECOND: {"normal": (1000, "ms", 1), "precise": (1000, "ms", 1)},
    Unit.SCALE: {"normal": (1, "-", 3), "precise": (1e6, "ppm", 1)},
    Unit.NONE: {"normal": (1, "-", 3)},
}


GPS_LEAP_SECONDS = 18
GPS_WEEK_ZERO = datetime.datetime(1980, 1, 6, 0, 0, 0)


class TimeFormat(Enum):
    UNIX = auto()
    DATETIME = auto()
    GPS_SOW = auto()


TIME_FORMAT_DICT = {
    "unix": TimeFormat.UNIX,
    "datetime": TimeFormat.DATETIME,
    "gps_sow": TimeFormat.GPS_SOW,
}
