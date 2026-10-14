import copy
import json
import logging
from abc import ABC
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from typing import Any, TypeVar, get_type_hints

import numpy as np

logger = logging.getLogger(__name__)

_SettingsT = TypeVar("_SettingsT", bound="Settings")

METRIC_THRESHOLD = 1e-4
TIME_THRESHOLD = 1e-4


@dataclass
class Settings(ABC):
    """Base class for all settings dataclasses.

    Provides serialization/deserialization functionality to and from dictionaries,
    JSON strings, and files.
    """

    def to_dict(self) -> dict:
        """Converts settings to a dictionary.

        Returns:
            dict: Dictionary representation of the settings
        """
        return {
            key: (value.to_dict() if issubclass(type(value), Settings) else self.encoder(key, value))
            for key, value in self.__dict__.items()
        }

    @staticmethod
    def encoder(name: str, value: Any) -> Any:
        """Encodes values for JSON serialization.

        Args:
            name: Name of the attribute being encoded
            value: Value to encode

        Returns:
            Any: Encoded value suitable for JSON serialization
        """
        return value

    @staticmethod
    def decoder(name: str, value: Any) -> Any:
        """Decodes values from JSON deserialization.

        Args:
            name: Name of the attribute being decoded
            value: Value to decode

        Returns:
            Any: Decoded value
        """
        return value

    def to_json(self) -> str:
        """Converts settings to a JSON string.

        Returns:
            str: JSON string representation of the settings
        """
        return json.dumps(self.to_dict(), indent=4)

    def to_file(self, path: str) -> None:
        """Writes settings to a JSON file.

        Args:
            path: Path to the output file
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def _field_types(cls) -> dict[str, Any]:
        """Returns the resolved types of all dataclass fields, including inherited ones."""
        hints = get_type_hints(cls)
        return {f.name: hints[f.name] for f in fields(cls) if f.init}

    @classmethod
    def _check_keys(cls, dct: dict) -> None:
        if unknown := set(dct) - set(cls._field_types()):
            logger.warning("%s: ignoring unknown settings %s", cls.__name__, sorted(unknown))

    def _decode_value(self, name: str, field_type: Any, value: Any) -> Any:
        """Decodes a value and validates it against the declared field type.

        Raises:
            TypeError: If the decoded value does not match the declared type.
        """
        decoded = self.decoder(name, value)
        if field_type is float and isinstance(decoded, int) and not isinstance(decoded, bool):
            return float(decoded)
        if field_type in (bool, int, float, str) and (
            not isinstance(decoded, field_type) or (field_type is int and isinstance(decoded, bool))
        ):
            raise TypeError(
                f"{type(self).__name__}.{name}: expected {field_type.__name__}, got {type(decoded).__name__} ({value!r})"
            )
        return decoded

    @classmethod
    def from_dict(cls: type[_SettingsT], dct: dict) -> _SettingsT:
        """Creates settings instance from a dictionary.

        Missing attributes keep their default value so that settings files written by
        older versions can still be read. Unknown attributes are ignored with a warning.

        Args:
            dct: Dictionary containing settings data

        Returns:
            Settings: New settings instance

        Raises:
            TypeError: If a value does not match the declared attribute type
        """
        settings = cls()
        settings.update_from_dict(dct)
        return settings

    @classmethod
    def from_file(cls: type[_SettingsT], path: str) -> _SettingsT:
        """Loads settings from a JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            Settings: Settings instance loaded from file
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def update_from_dict(self, dct: dict) -> None:
        """Updates settings from a dictionary.

        Attributes missing from the dictionary keep their current value. Unknown attributes
        are ignored with a warning. The update is atomic: if any value is invalid, the
        settings remain unchanged.

        Args:
            dct: Dictionary containing updated settings data

        Raises:
            TypeError: If a value does not match the declared attribute type
        """
        updates = self._decoded_updates(dct)  # validates everything before anything is changed
        for attribute_name, value in updates.items():
            if isinstance(value, Settings):
                # update nested settings in place, so that references held elsewhere stay up to date
                getattr(self, attribute_name)._assign_from(value)
            else:
                setattr(self, attribute_name, value)

    def _assign_from(self, other: "Settings") -> None:
        """Copies all values of ``other`` into this object, keeping the identity of nested settings."""
        for attribute_name in self._field_types():
            value = getattr(other, attribute_name)
            if isinstance(value, Settings):
                getattr(self, attribute_name)._assign_from(value)
            else:
                setattr(self, attribute_name, value)

    def _decoded_updates(self, dct: dict) -> dict[str, Any]:
        """Validates and decodes ``dct`` without modifying ``self``. Nested settings are returned as new objects."""
        self._check_keys(dct)
        updates: dict[str, Any] = {}
        for attribute_name, attribute_type in self._field_types().items():
            if attribute_name not in dct:
                logger.debug("%s: %s not given, keeping current value", type(self).__name__, attribute_name)
                continue

            attribute_data = dct[attribute_name]
            if isinstance(attribute_type, type) and issubclass(attribute_type, Settings):
                if not isinstance(attribute_data, dict):
                    raise TypeError(f"{type(self).__name__}.{attribute_name}: expected a dictionary")
                nested = copy.deepcopy(getattr(self, attribute_name))
                nested.update_from_dict(attribute_data)
                updates[attribute_name] = nested
            else:
                updates[attribute_name] = self._decode_value(attribute_name, attribute_type, attribute_data)
        return updates


@dataclass
class AlignmentPreprocessing(Settings):
    """Configuration for alignment preprocessing filters.

    Attributes:
        min_speed: Only poses with speed above this threshold are considered for
            alignment (meters/second). Defaults to 0.0.
        time_start: Only poses with timestamp above this threshold are considered.
            Timestamp is in seconds relative to first common timestamp. Defaults to 0.0.
        time_end: Only poses with timestamp below this threshold are considered.
            Timestamp is in seconds relative to first common timestamp. Defaults to 0.0.
        max_samples: Maximum number of matched poses used for the alignment. If more poses are
            available, a new random subset of this size is drawn on each run. Set to 0
            to use all poses. Defaults to 5000.
    """

    min_speed: float = 0.0
    time_start: float = 0.0
    time_end: float = 0.0
    max_samples: int = 5000


@dataclass
class AlignmentEstimationSettings(Settings):
    """Configuration for parameters to estimate during trajectory alignment.

    Attributes:
        translation_x: Enable x-translation of similarity transformation
        translation_y: Enable y-translation of similarity transformation
        translation_z: Enable z-translation of similarity transformation
        rotation_x: Enable rotation around X-axis of similarity transformation
        rotation_y: Enable rotation around Y-axis of similarity transformation
        rotation_z: Enable rotation around Z-axis of similarity transformation
        scale: Enable scaling of similarity transformation
        time_shift: Enable estimation of time shift between trajectories
        leverarm_x: Enable estimation of lever arm in X-direction
        leverarm_y: Enable estimation of lever arm in Y-direction
        leverarm_z: Enable estimation of lever arm in Z-direction
        sensor_rotation: Enable computation of sensor rotation offsets (independent
            of least squares, computes constant offsets between roll, pitch, yaw)
    """

    translation_x: bool = True
    translation_y: bool = True
    translation_z: bool = True
    rotation_x: bool = True
    rotation_y: bool = True
    rotation_z: bool = True
    scale: bool = False

    time_shift: bool = False

    leverarm_x: bool = False
    leverarm_y: bool = False
    leverarm_z: bool = False

    sensor_rotation: bool = False

    def __bool__(self) -> bool:
        """Returns True if any least-squares parameter is enabled.

        Returns:
            bool: True if at least one least-squares parameter is enabled
        """
        return not self.all_lq_disabled

    @classmethod
    def from_components(
        cls,
        similarity: bool = False,
        time_shift: bool = False,
        leverarm: bool = False,
        sensor_rotation: bool = False,
    ) -> "AlignmentEstimationSettings":
        """Creates settings from high-level component flags.

        Args:
            similarity: Enable all similarity transformation parameters. Defaults to False.
            time_shift: Enable time shift estimation. Defaults to False.
            leverarm: Enable all leverarm parameters. Defaults to False.
            sensor_rotation: Enable sensor rotation estimation. Defaults to False.

        Returns:
            AlignmentEstimationSettings: New settings instance
        """
        return cls(
            translation_x=similarity,
            translation_y=similarity,
            translation_z=similarity,
            rotation_x=similarity,
            rotation_y=similarity,
            rotation_z=similarity,
            scale=similarity,
            time_shift=time_shift,
            leverarm_x=leverarm,
            leverarm_y=leverarm,
            leverarm_z=leverarm,
            sensor_rotation=sensor_rotation,
        )

    @classmethod
    def all(cls, sensor_rotation: bool = True) -> "AlignmentEstimationSettings":
        """Creates settings with all parameters enabled.

        Args:
            sensor_rotation: Enable sensor rotation. Defaults to True.

        Returns:
            AlignmentEstimationSettings: Settings with all parameters enabled
        """
        return cls(
            translation_x=True,
            translation_y=True,
            translation_z=True,
            rotation_x=True,
            rotation_y=True,
            rotation_z=True,
            scale=True,
            time_shift=True,
            leverarm_x=True,
            leverarm_y=True,
            leverarm_z=True,
            sensor_rotation=sensor_rotation,
        )

    @classmethod
    def from_bool_list(cls, bool_list: list[bool]) -> "AlignmentEstimationSettings":
        """Creates settings from a list of boolean values.

        Args:
            bool_list: List of 14 booleans corresponding to parameter enable flags

        Returns:
            AlignmentEstimationSettings: New settings instance

        Raises:
            ValueError: If bool_list length is not 14
        """
        if len(bool_list) != 14:
            raise ValueError(
                f"Size mismatch: bool_list must have length 14 (Number of configurable parameters) (got {len(bool_list)})"
            )

        return AlignmentEstimationSettings(
            translation_x=bool_list[0],
            translation_y=bool_list[1],
            translation_z=bool_list[2],
            rotation_x=bool_list[3],
            rotation_y=bool_list[4],
            rotation_z=bool_list[5],
            scale=bool_list[6],
            time_shift=bool_list[7],
            leverarm_x=bool_list[8],
            leverarm_y=bool_list[9],
            leverarm_z=bool_list[10],
            sensor_rotation=any(bool_list[11:]),
        )

    @property
    def all_disabled(self) -> bool:
        """Checks if all alignment parameters are disabled.

        Returns:
            bool: True if all parameters (Helmert, time shift, leverarm, sensor rotation) are disabled
        """
        return not any(
            [
                self.helmert_enabled,
                self.time_shift_enabled,
                self.leverarm_enabled,
                self.sensor_rotation,
            ]
        )

    @property
    def all_lq_disabled(self) -> bool:
        """Checks if all least-squares parameters are disabled.

        Returns:
            bool: True if all parameters estimated with least-squares are disabled
        """
        return not any([self.helmert_enabled, self.time_shift_enabled, self.leverarm_enabled])

    @property
    def helmert_enabled(self) -> bool:
        """Checks if any Helmert transformation parameter is enabled.

        Returns:
            bool: True if any translation, rotation, or scale parameter is enabled
        """
        return any(
            [
                self.translation_x,
                self.translation_y,
                self.translation_z,
                self.rotation_x,
                self.rotation_y,
                self.rotation_z,
                self.scale,
            ]
        )

    @property
    def leverarm_enabled(self) -> bool:
        """Checks if any leverarm parameter is enabled.

        Returns:
            bool: True if any leverarm component (x, y, z) is enabled
        """
        return any([self.leverarm_x, self.leverarm_y, self.leverarm_z])

    @property
    def time_shift_enabled(self) -> bool:
        """Checks if time shift estimation is enabled.

        Returns:
            bool: True if time shift is enabled
        """
        return any(self.time_shift_filter)

    @property
    def short_mode_str(self) -> str:
        """Gets a short string describing enabled parameter groups.

        Returns:
            str: Space-separated string of enabled parameter groups
                (e.g., "Helmert Time-Shift Leverarm")
        """
        settings_str = ""

        if self.helmert_enabled:
            settings_str += "Helmert "

        if self.time_shift_enabled:
            settings_str += "Time-Shift "

        if self.leverarm_enabled:
            settings_str += "Leverarm "

        if self.sensor_rotation:
            settings_str += "Sensor-Rotation"

        return settings_str

    @property
    def time_shift_filter(self) -> list[bool]:
        """Gets time shift parameter filter.

        Returns:
            List[bool]: List of 3 bools, all True if time shift enabled, else all False
        """
        return [True] * 3 if self.time_shift else [False] * 3

    @property
    def helmert_filter(self) -> list[bool]:
        """Gets Helmert parameter filter.

        Returns:
            List[bool]: List of 7 bools indicating which Helmert parameters are enabled
        """
        if not self.helmert_enabled:
            return [False] * 7

        return [
            self.translation_x,
            self.translation_y,
            self.translation_z,
            self.rotation_x,
            self.rotation_y,
            self.rotation_z,
            self.scale,
        ]

    @property
    def leverarm_filter(self) -> list[bool]:
        """Gets leverarm parameter filter.

        Returns:
            List[bool]: List of 3 bools indicating which leverarm components are enabled
        """
        if not self.leverarm_enabled:
            return [False] * 3

        return [self.leverarm_x, self.leverarm_y, self.leverarm_z]

    @property
    def enabled_lq_parameter_filter(self) -> list[bool]:
        """Gets filter for enabled least-squares parameters only.

        Returns:
            List[bool]: List of bools for only the enabled parameter groups
        """
        full_filter = []

        if self.helmert_enabled:
            full_filter.extend(self.helmert_filter)

        if self.time_shift_enabled:
            full_filter.append(self.time_shift)

        if self.leverarm_enabled:
            full_filter.extend(self.leverarm_filter)

        return full_filter

    @property
    def lq_parameter_filter(self) -> list[bool]:
        """Gets complete least-squares parameter filter.

        Returns:
            List[bool]: List of 11 bools for all LQ parameters (7 Helmert + 1 time shift + 3 leverarm)
        """
        return self.helmert_filter + [self.time_shift_enabled] + self.leverarm_filter


@dataclass
class AlignmentStochastics(Settings):
    """Configuration for alignment stochastic model (observation uncertainties).

    Attributes:
        std_xy_from: Standard deviation of XY source position components in meters. Defaults to 1.0.
        std_z_from: Standard deviation of Z source position component in meters. Defaults to 1.0.
        std_xy_to: Standard deviation of XY target position components in meters. Defaults to 1.0.
        std_z_to: Standard deviation of Z target position component in meters. Defaults to 1.0.
        std_roll_pitch: Standard deviation of roll and pitch angles in radians. Defaults to ~0.017 rad (1°).
        std_yaw: Standard deviation of yaw angle in radians. Defaults to ~0.017 rad (1°).
        std_speed: Standard deviation of platform speed in meters per second. Defaults to 1.0.
        error_probability: Probability of error for stochastic testing. Defaults to 0.05.
        variance_estimation: Enable a-posteriori variance factor estimation. Defaults to False.
    """

    std_xy_from: float = 1.0
    std_z_from: float = 1.0
    std_xy_to: float = 1.0
    std_z_to: float = 1.0
    std_roll_pitch: float = float(np.deg2rad(1.0))
    std_yaw: float = float(np.deg2rad(1.0))
    std_speed: float = 1.0
    error_probability: float = 0.05
    variance_estimation: bool = False

    @property
    def var_xy_from(self) -> float:
        """Gets variance of XY source position components.

        Returns:
            float: Variance (squared standard deviation)
        """
        return self.std_xy_from**2

    @property
    def var_z_from(self) -> float:
        """Gets variance of Z source position component.

        Returns:
            float: Variance (squared standard deviation)
        """
        return self.std_z_from**2

    @property
    def var_xy_to(self) -> float:
        """Gets variance of XY target position components.

        Returns:
            float: Variance (squared standard deviation)
        """
        return self.std_xy_to**2

    @property
    def var_z_to(self) -> float:
        """Gets variance of Z target position component.

        Returns:
            float: Variance (squared standard deviation)
        """
        return self.std_z_to**2

    @property
    def var_roll_pitch(self) -> float:
        """Gets variance of roll and pitch angles.

        Returns:
            float: Variance (squared standard deviation)
        """
        return self.std_roll_pitch**2

    @property
    def var_yaw(self) -> float:
        """Gets variance of yaw angle.

        Returns:
            float: Variance (squared standard deviation)
        """
        return self.std_yaw**2

    @property
    def var_speed_to(self) -> float:
        """Gets variance of platform speed.

        Returns:
            float: Variance (squared standard deviation)
        """
        return self.std_speed**2


@dataclass
class AlignmentSettings(Settings):
    """Complete configuration for trajectory alignment.

    Combines preprocessing, estimation, and stochastic settings for trajectory alignment.

    Attributes:
        preprocessing: Preprocessing filter settings
        estimation_settings: Parameter estimation settings
        stochastics: Stochastic model (observation uncertainty) settings
        metric_threshold: Iteration threshold for metric parameters in least squares.
            Defaults to 1e-4.
        time_threshold: Iteration threshold for time shift parameter in least squares (seconds).
            Defaults to 1e-4.
    """

    preprocessing: AlignmentPreprocessing = field(default_factory=AlignmentPreprocessing)
    estimation_settings: AlignmentEstimationSettings = field(default_factory=AlignmentEstimationSettings)
    stochastics: AlignmentStochastics = field(default_factory=AlignmentStochastics)
    metric_threshold: float = METRIC_THRESHOLD
    time_threshold: float = TIME_THRESHOLD

    def __str__(self) -> str:
        """Returns string representation of all settings.

        Returns:
            str: Concatenated string of preprocessing, estimation, and stochastics settings
        """
        return str(self.preprocessing) + str(self.estimation_settings) + str(self.stochastics)


class RotationApproximationMethod(Enum):
    """Method for rotation approximation."""

    AVERAGE_PER_PATH_LENGTH = "average_per_path_length"
    AVERAGE_IN_WINDOW = "average_in_window"


class PositionApproximationMethod(Enum):
    """Method for position approximation."""

    CUBIC = "cubic"


@dataclass
class ApproximationSettings(Settings):
    """Configuration for trajectory approximation and smoothing.

    Attributes:
        position_interval_size: Size of position intervals in meters for cubic approximation.
            Defaults to 0.15.
        position_min_observations: Minimum number of observations required in each interval
            for cubic approximation. Defaults to 25.
        rotation_window_size: Size of rotation smoothing window in meters (not cubic).
            Defaults to 0.15.
        max_path_length_gap: Maximum gap size in path length for interpolation.
            Defaults to 0.15.
    """

    position_interval_size: float = 0.15
    position_min_observations: int = 25
    rotation_window_size: float = 0.15
    max_path_length_gap: float = 0.15
    position_approximation_method: PositionApproximationMethod = PositionApproximationMethod.CUBIC
    rotation_approximation_method: RotationApproximationMethod = RotationApproximationMethod.AVERAGE_IN_WINDOW

    @staticmethod
    def encoder(name: str, value: Any) -> Any:
        """Encodes position_approximation_method and rotation_approximation_method enums to string values.

        Args:
            name: Attribute name
            value: Value to encode

        Returns:
            Any: Encoded value
        """
        if name in ["position_approximation_method", "rotation_approximation_method"]:
            return value.value
        return value

    @staticmethod
    def decoder(name: str, value: Any) -> Any:
        """Decodes position_approximation_method and rotation_approximation_method strings to enums.
        Args:
            name: Attribute name
            value: Value to decode

        Returns:
            Any: Decoded value
        """
        if name == "position_approximation_method":
            return PositionApproximationMethod(value)
        if name == "rotation_approximation_method":
            return RotationApproximationMethod(value)
        return value


class PairDistanceUnit(Enum):
    """Unit of measurement for pose pair distances in relative comparison."""

    METER = "meter"
    SECOND = "second"

    @classmethod
    def from_str(cls, unit_str: str) -> "PairDistanceUnit":
        """Converts a string to a PairDistanceUnit.

        Args:
            unit_str: String to convert ("m", "meter", "s", "sec", "second")

        Returns:
            PairDistanceUnit: Converted unit

        Raises:
            ValueError: If unit string is not recognized
        """
        if unit_str.lower() in {"m", "meter"}:
            return cls.METER
        if unit_str.lower() in {"s", "sec", "second"}:
            return cls.SECOND

        raise ValueError(f"Unknown unit string {unit_str}")


class ComparisonMethod(Enum):
    """Method for trajectory comparison."""

    ABSOLUTE = auto()
    RELATIVE = auto()
    UNKNOWN = auto()

    @classmethod
    def from_string(cls, string: str) -> "ComparisonMethod":
        """Converts string to ComparisonMethod.

        Args:
            string: String containing "absolute" or "relative"

        Returns:
            ComparisonMethod: Parsed comparison method
        """
        return _comparison_method_from_string(string)


def _comparison_method_from_string(string: str) -> ComparisonMethod:
    """Converts string to ComparisonMethod.

    Args:
        string: String containing "absolute" or "relative"

    Returns:
        ComparisonMethod: ABSOLUTE, RELATIVE, or UNKNOWN
    """
    if "absolute" in string.lower():
        return ComparisonMethod.ABSOLUTE

    return ComparisonMethod.RELATIVE if "relative" in string.lower() else ComparisonMethod.UNKNOWN


@dataclass
class RelativeComparisonSettings(Settings):
    """Configuration for relative trajectory comparison (RPE - Relative Pose Error).

    Relative comparison involves finding pose pairs separated by specific distances or
    time intervals and computing relative translation/rotation differences.

    Attributes:
        pair_min_distance: Minimum pose pair distance to consider during RPE computation.
            Defaults to 100.0.
        pair_max_distance: Maximum pose pair distance to consider during RPE computation.
            Defaults to 800.0.
        pair_distance_step: Step size for increasing pose pair distance. Defaults to 100.0.
        pair_distance_unit: Unit of pose pair distance (METER or SECOND). Defaults to METER.
        use_all_pose_pairs: If True, use overlapping pose pairs; if False, use only
            consecutive (non-overlapping) pairs. Defaults to True.
    """

    pair_min_distance: float = 100.0
    pair_max_distance: float = 800.0
    pair_distance_step: float = 100.0
    pair_distance_unit: PairDistanceUnit = PairDistanceUnit.METER
    use_all_pose_pairs: bool = True

    @staticmethod
    def encoder(name: str, value: Any) -> Any:
        """Encodes pair_distance_unit enum to string value.

        Args:
            name: Attribute name
            value: Value to encode

        Returns:
            Any: Encoded value
        """
        return value.value if name == "pair_distance_unit" else value

    @staticmethod
    def decoder(name: str, value: Any) -> Any:
        """Decodes pair_distance_unit string to enum.

        Args:
            name: Attribute name
            value: Value to decode

        Returns:
            Any: Decoded value
        """
        return PairDistanceUnit(value) if name == "pair_distance_unit" else value


class MatchingMethod(Enum):
    """Method for matching poses between two trajectories.

    Attributes:
        NEAREST_SPATIAL: Match by nearest Euclidean distance
        NEAREST_TEMPORAL: Match by nearest timestamp
        INTERPOLATION: Match using linear interpolation (SLERP for rotations)
        NEAREST_SPATIAL_INTERPOLATED: Match spatially using k-nearest and 3D line fit
        UNKNOWN: Unknown matching method
    """

    NEAREST_SPATIAL = "nearest_spatial"
    NEAREST_TEMPORAL = "nearest_temporal"
    INTERPOLATION = "interpolation"
    NEAREST_SPATIAL_INTERPOLATED = "nearest_spatial_interpolated"
    UNKNOWN = "unknown"


class InterpolationMethod(Enum):
    """Method for interpolation."""

    LINEAR = "linear"


@dataclass
class MatchingSettings(Settings):
    """Configuration for trajectory matching.

    Attributes:
        method: Matching method to use. Defaults to INTERPOLATION.
        max_time_diff: Maximum allowed time difference in seconds for temporal matching.
            Defaults to 0.01.
        max_distance: Maximum allowed distance in meters for spatial matching.
            Defaults to 0.0.
        max_gap_size: Maximum allowed gap size in seconds within trajectories.
            Defaults to 10.0.
        k_nearest: Number of nearest neighbors for spatial interpolation matching.
            Defaults to 2.
    """

    method: MatchingMethod = MatchingMethod.INTERPOLATION
    max_time_diff: float = 0.01
    max_distance: float = 0.00
    max_gap_size: float = 10.0
    k_nearest: int = 2

    @staticmethod
    def encoder(name: str, value: Any) -> Any:
        """Encodes method enum to string value.

        Args:
            name: Attribute name
            value: Value to encode

        Returns:
            Any: Encoded value
        """
        return value.value if name == "method" else value

    @staticmethod
    def decoder(name: str, value: Any) -> Any:
        """Decodes method string to enum.

        Args:
            name: Attribute name
            value: Value to decode

        Returns:
            Any: Decoded value
        """
        return MatchingMethod(value) if name == "method" else value


@dataclass
class PlotSettings(Settings):
    """Configuration of all plots (GUI and :mod:`trajectopy.visualization`).

    The along-track direction is positive in travel direction, horizontal cross-track is
    positive to the right, and vertical cross-track is positive upwards.

    Attributes:
        ate_unit_is_mm: Show position deviations in millimeters instead of meters. Defaults to False.
        directed_ate: Show position deviations split into along-track, horizontal cross-track and
            vertical cross-track instead of x, y and z. Defaults to False.
        ate_remove_above: Exclude position deviations above this value in meters (0.0 = keep all). Defaults to 0.0.
        hist_percentile: Show histogram data up to this percentile of the absolute values. Defaults to 1.0.
        colorbar_max_std: Clip color-coded values to mean ± this many standard deviations, so that outliers
            don't dominate the color scale (0.0 = no clipping). Defaults to 3.0.
        smoothing_window: Moving-average window applied to color-coded values (1 = no smoothing). Defaults to 1.
        use_relative_timestamps: Show time axes in seconds relative to the first timestamp instead of
            absolute (UTC) time. Defaults to False.
        line_width: Line width in pixels. Defaults to 1.5.
        marker_size: Marker size in pixels of color-coded trajectory maps. Defaults to 4.
        font_size: Font size in points of axis labels and titles. Defaults to 10.
        export_format: File format for exported plots ("png", "jpg" or "svg"). Defaults to "png".
        export_width: Width in pixels of exported raster images. Defaults to 1600.
        background_map: Show a map below top views of georeferenced trajectories. Requires an internet
            connection and sends the approximate area of the trajectories to the tile server; tiles are
            cached on disk. Defaults to False.
        map_style: Map shown by ``background_map``: "osm" (OpenStreetMap) or "satellite" (Esri World
            Imagery). Defaults to "osm".
    """

    ate_unit_is_mm: bool = False
    directed_ate: bool = False
    ate_remove_above: float = 0.0
    hist_percentile: float = 1.0
    colorbar_max_std: float = 3.0
    smoothing_window: int = 1
    use_relative_timestamps: bool = False
    line_width: float = 5
    marker_size: float = 5.0
    font_size: int = 10
    export_format: str = "png"
    export_width: int = 1600
    background_map: bool = False
    map_style: str = "osm"

    @property
    def unit_multiplier(self) -> float:
        """Factor converting meters to the display unit."""
        return 1000.0 if self.ate_unit_is_mm else 1.0

    @property
    def unit(self) -> str:
        """Display unit of position deviations."""
        return "mm" if self.ate_unit_is_mm else "m"


@dataclass
class SortingSettings(Settings):
    """Configuration for trajectory sorting and downsampling using Moving Least Squares.

    Attributes:
        voxel_size: Size of voxel grid for downsampling (meters). Voxel centroids are
            used for nearest neighbor searches instead of raw points. Defaults to 0.05.
        movement_threshold: Maximum allowed point movement between MLS iterations (meters).
            Algorithm terminates when all points move less than this. Defaults to 0.005.
        k_nearest: Number of nearest voxels to consider during MLS smoothing. Defaults to 4.
    """

    voxel_size: float = 0.05
    movement_threshold: float = 0.005
    k_nearest: int = 4


@dataclass
class ProcessingSettings(Settings):
    """Complete configuration for trajectory processing.

    Combines all processing-related settings including alignment, matching,
    relative comparison, approximation, and sorting.

    Attributes:
        alignment: Alignment configuration
        matching: Matching configuration
        relative_comparison: Relative comparison (RPE) configuration
        approximation: Approximation and smoothing configuration
        sorting: Sorting and downsampling configuration
    """

    alignment: AlignmentSettings = field(default_factory=AlignmentSettings)
    matching: MatchingSettings = field(default_factory=MatchingSettings)
    relative_comparison: RelativeComparisonSettings = field(default_factory=RelativeComparisonSettings)
    approximation: ApproximationSettings = field(default_factory=ApproximationSettings)
    sorting: SortingSettings = field(default_factory=SortingSettings)
