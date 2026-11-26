import datetime
from enum import Enum, auto

import numpy as np

UNIX_TIME_THRESHOLD = 1_000_000


class OptionEnum(str, Enum):
    """Base class of all enumerations that select an option, e.g. a matching method.

    Wherever an option is expected, its value can also be given as a string. The lookup ignores the case and
    accepts the aliases returned by ``aliases()`` (a mapping from alias to value), e.g.
    ``MatchingMethod("Interpolation") is MatchingMethod.INTERPOLATION``.
    """

    @classmethod
    def _missing_(cls, value: object) -> "OptionEnum | None":
        if not isinstance(value, str):
            return None
        key = value.strip().lower()
        key = cls.aliases().get(key, key)
        for member in cls:
            if member.value == key or member.name.lower() == key:
                return member
        return None

    @classmethod
    def aliases(cls) -> dict[str, str]:
        """Alternative names of the options, mapped to their values."""
        return {}

    @classmethod
    def from_str(cls, value: str):  # noqa: ANN206 - returns the subclass
        """Returns the option for a string (case-insensitive, aliases allowed). Same as ``cls(value)``."""
        return cls(value)

    def __str__(self) -> str:
        return self.value


class Sorting(OptionEnum):
    """Order of the poses of a trajectory, which also defines its index (``Trajectory.index``).

    Each option can also be given as its value, e.g. ``"time"`` (case-insensitive). The alias ``"arc_length"`` is accepted, too.

    Attributes:
        TIME: ``"time"``: chronological order; the index is the timestamps.
        PATH_LENGTH: ``"path_length"``: order along the path after spatial sorting (see
            :func:`trajectopy.sort_spatially`); the index is the path length.
    """

    TIME = "time"
    PATH_LENGTH = "path_length"

    @classmethod
    def aliases(cls) -> dict[str, str]:
        return {"arc_length": "path_length"}


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
GPS_WEEK_ZERO = datetime.datetime(1980, 1, 6, 0, 0, 0, tzinfo=datetime.timezone.utc)


class TimeFormat(Enum):
    UNIX = auto()
    DATETIME = auto()
    GPS_SOW = auto()


TIME_FORMAT_DICT = {
    "unix": TimeFormat.UNIX,
    "datetime": TimeFormat.DATETIME,
    "gps_sow": TimeFormat.GPS_SOW,
}
