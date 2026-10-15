"""Parsing of the ``#key value`` header lines used by all trajectopy file formats."""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from trajectopy.core.settings import ComparisonMethod, PairDistanceUnit
from trajectopy.utils.definitions import GPS_WEEK_ZERO, TIME_FORMAT_DICT, TimeFormat

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
    "euler_rot_seq",
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


_SCAN_CHUNK_SIZE = 1 << 24  # bytes


@dataclass(frozen=True)
class HeaderScan:
    """Header lines of a file, found without parsing the data lines in Python.

    Attributes:
        lines: All lines starting with ``#`` (in file order, decoded, ending with a newline).
        leading_lines: Number of lines before the first data line (header, comments and blank lines).
        lines_after_data: Whether any ``#`` line follows the first data line.
    """

    lines: list[str]
    leading_lines: int
    lines_after_data: bool


def scan_header(filename: str) -> HeaderScan:
    """Collects the ``#`` lines of a file. Equivalent to iterating over all lines, but the data part is
    searched with ``bytes.find`` in large chunks, which matters for files with millions of lines."""
    lines: list[bytes] = []
    leading_lines = 0
    with open(filename, "rb") as file:
        # leading header, comments and blank lines
        for line in file:
            if line.startswith(b"#"):
                lines.append(line)
            elif line.strip():
                break
            leading_lines += 1
        else:
            return HeaderScan([_decode(line) for line in lines], leading_lines, False)

        num_leading = len(lines)
        # "\n#" marks a header line after the first data line; a chunk may end in the middle of such a line
        rest = b"\n"
        while chunk := file.read(_SCAN_CHUNK_SIZE):
            rest += chunk
            position = rest.find(b"\n#")
            while position != -1:
                end = rest.find(b"\n", position + 1)
                if end == -1:
                    break
                lines.append(rest[position + 1 : end + 1])
                position = rest.find(b"\n#", end)
            # keep a possibly incomplete header line (or the last newline) for the next chunk
            keep_from = position if position != -1 else max(rest.rfind(b"\n"), 0)
            rest = rest[keep_from:]
        if (position := rest.find(b"\n#")) != -1:
            lines.append(rest[position + 1 :])

    return HeaderScan([_decode(line) for line in lines], leading_lines, len(lines) > num_leading)


def _decode(line: bytes) -> str:
    return line.decode("utf-8").replace("\r\n", "\n")


@dataclass
class HeaderData:
    """Class to store the header data of a trajectopy file."""

    data: dict[str, str | int | float]

    @property
    def id(self) -> str:
        return str(self.data.get("id", ""))

    @id.setter
    def id(self, value: str) -> None:
        try:
            UUID(value, version=4)
        except ValueError:
            logger.debug("Not setting id to '%s' since it is not a valid UUID4.", value)
            return
        self.data["id"] = value

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
    def euler_rot_seq(self) -> str:
        return str(self.data.get("euler_rot_seq", "xyz"))

    @property
    def gps_week(self) -> int:
        if "gps_week" in self.data:
            return int(self.data["gps_week"])
        return (datetime.now(tz=timezone.utc) - GPS_WEEK_ZERO).days // 7

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
        return cls.from_lines(scan_header(filename).lines, source=filename)

    @classmethod
    def from_lines(cls, lines: list[str], source: str = "") -> "HeaderData":
        """Builds the header data from the lines of a file (lines without a leading ``#`` are ignored)."""
        metadata: dict[str, str | int | float] = {}
        for line in lines:
            cls.handle_line(metadata, line)

        logger.info("Read header of %s", source)
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
