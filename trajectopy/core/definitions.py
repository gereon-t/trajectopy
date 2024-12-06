"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import datetime
from enum import Enum, auto
from typing import Dict, Tuple

import numpy as np


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
UNIT_FORMAT_RULES: Dict[Unit, Dict[str, Tuple[float, str, int]]] = {
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
