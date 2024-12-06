"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from trajectopy.core.definitions import Unit
from trajectopy.core.settings.base import Settings


class PairDistanceUnit(Enum):
    METER = "meter"
    SECOND = "second"

    @classmethod
    def from_str(cls, unit_str: str) -> "Unit":
        """
        Converts a string to a unit

        Args:
            unit_str (str): String to be converted

        Returns:
            PairDistanceUnit: Converted unit
        """
        if unit_str.lower() in {"m", "meter"}:
            return cls.METER
        if unit_str.lower() in {"s", "sec", "second"}:
            return cls.SECOND

        raise ValueError(f"Unknown unit string {unit_str}")


class ComparisonMethod(Enum):
    ABSOLUTE = auto()
    RELATIVE = auto()
    UNKNOWN = auto()

    @classmethod
    def from_string(cls, string: str):
        return comparison_method_from_string(string)


def comparison_method_from_string(string: str) -> ComparisonMethod:
    if "absolute" in string.lower():
        return ComparisonMethod.ABSOLUTE

    return ComparisonMethod.RELATIVE if "relative" in string.lower() else ComparisonMethod.UNKNOWN


@dataclass
class RelativeComparisonSettings(Settings):
    """
    A class representing the settings for relative trajectory comparison.

    Attributes:
        relative_pair_min_distance (float): The minimum distance between two poses in a relative pair.
        relative_pair_max_distance (float): The maximum distance between two poses in a relative pair.
        relative_pair_distance_step (float): The step size for the distance between two poses in a relative pair.
        relative_pair_distance_unit (PairDistanceUnit): The unit of measurement for the distance between two poses in a relative pair.
        use_all_pose_pairs (bool): Whether to use all possible pose pairs for relative comparison.
    """

    pair_min_distance: float = 100.0
    pair_max_distance: float = 800.0
    pair_distance_step: float = 100.0
    pair_distance_unit: PairDistanceUnit = PairDistanceUnit.METER
    use_all_pose_pairs: bool = True

    @staticmethod
    def encoder(name: str, value: Any) -> Any:
        if name == "pair_distance_unit":
            return value.value
        return value

    @staticmethod
    def decoder(name: str, value: Any) -> Any:
        if name == "pair_distance_unit":
            return PairDistanceUnit(value)
        return value


if __name__ == "__main__":
    settings = RelativeComparisonSettings()
    settings.to_file("rel_settings.json")
    imported_settings = RelativeComparisonSettings.from_file("rel_settings.json")

    assert imported_settings == settings
    print(imported_settings)
