import inspect
import json
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

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
    def from_dict(cls, dct: dict) -> "Settings":
        """Creates settings instance from a dictionary.

        Args:
            dct: Dictionary containing settings data

        Returns:
            Settings: New settings instance

        Raises:
            ValueError: If required attribute is missing from dictionary
        """
        settings = cls()
        for attribute_name, attribute_type in cls.__annotations__.items():
            if attribute_name not in dct:
                raise ValueError(f"Attribute {attribute_name} not found in input data")

            attribute_data = dct[attribute_name]
            if isinstance(attribute_data, dict) and issubclass(attribute_type, Settings):
                setattr(settings, attribute_name, attribute_type.from_dict(attribute_data))
            else:
                setattr(settings, attribute_name, settings.decoder(attribute_name, attribute_data))

        return settings

    @classmethod
    def from_file(cls, path: str) -> "Settings":
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

        Args:
            dct: Dictionary containing updated settings data

        Raises:
            ValueError: If required attribute is missing from dictionary
        """
        annotations = inspect.get_annotations(type(self))
        for attribute_name, attribute_type in annotations.items():
            if attribute_name not in dct:
                raise ValueError(f"Attribute {attribute_name} not found in input data")

            attribute_data = dct[attribute_name]
            if isinstance(attribute_data, dict) and issubclass(attribute_type, Settings):
                getattr(self, attribute_name).update_from_dict(attribute_data)
            else:
                setattr(self, attribute_name, self.decoder(attribute_name, attribute_data))


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
    """

    min_speed: float = 0.0
    time_start: float = 0.0
    time_end: float = 0.0


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


if __name__ == "__main__":
    settings = AlignmentSettings()
    settings.to_file("alignment_settings.json")
    imported_settings = AlignmentSettings.from_file("alignment_settings.json")

    assert settings == imported_settings
    print(imported_settings)


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
    """

    position_interval_size: float = 0.15
    position_min_observations: int = 25
    rotation_window_size: float = 0.15


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


if __name__ == "__main__":
    settings = RelativeComparisonSettings()
    settings.to_file("rel_settings.json")
    imported_settings = RelativeComparisonSettings.from_file("rel_settings.json")

    assert imported_settings == settings
    print(imported_settings)


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


if __name__ == "__main__":
    settings = MatchingSettings()
    settings.to_file("matching_settings.json")
    imported_settings = MatchingSettings.from_file("matching_settings.json")

    assert imported_settings == settings
    print(imported_settings)


@dataclass
class MPLPlotSettings(Settings):
    """Configuration for matplotlib plotting backend.

    Attributes:
        colorbar_show_zero_crossing: Show zero in colorbar. Defaults to True.
        colorbar_steps: Number of colorbar steps. Defaults to 4.
        colorbar_max_std: Upper colorbar limit as mean + this * std. Defaults to 3.0.
        scatter_hide_axes: Hide axes in scatter plots. Defaults to False.
        scatter_3d: Use 3D scatter plots. Defaults to False.
        scatter_smooth: Smooth scatter plot color data. Defaults to False.
        scatter_smooth_window: Window size for scatter smoothing. Defaults to 5.
        ate_unit_is_mm: Use millimeters for ATE units. Defaults to False.
        ate_remove_above: Cap ATE values above this (0.0 = no cap). Defaults to 0.0.
        hist_as_stairs: Display histogram as stairs. Defaults to False.
        hist_percentile: Show histogram data up to this percentile. Defaults to 1.0.
        directed_ate: Split ATE into along/cross-track directions. Defaults to False.
        dofs_tab: Show degrees of freedom tab. Defaults to True.
        velocity_tab: Show velocity tab. Defaults to True.
        height_tab: Show height tab. Defaults to True.
    """

    colorbar_show_zero_crossing: bool = True
    colorbar_steps: int = 4
    colorbar_max_std: float = 3.0
    scatter_hide_axes: bool = False
    scatter_3d: bool = False
    scatter_smooth: bool = False
    scatter_smooth_window: int = 5
    ate_unit_is_mm: bool = False
    ate_remove_above: float = 0.0
    hist_as_stairs: bool = False
    hist_percentile: float = 1.0
    directed_ate: bool = False
    dofs_tab: bool = True
    velocity_tab: bool = True
    height_tab: bool = True

    @property
    def unit_multiplier(self) -> float:
        """Gets unit multiplier for converting meters to display unit.

        Returns:
            float: 1000 if using millimeters, 1 otherwise
        """
        return 1000 if self.ate_unit_is_mm else 1

    @property
    def unit_str(self) -> str:
        """Gets unit string for display.

        Returns:
            str: "[mm]" if using millimeters, "[m]" otherwise
        """
        return "[mm]" if self.ate_unit_is_mm else "[m]"


class PlotBackend(Enum):
    """Plotting backend selection."""

    PLOTLY = "plotly"
    MPL = "mpl"


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


if __name__ == "__main__":
    settings = ProcessingSettings()
    settings.to_file("processing_settings.json")
    imported_settings = ProcessingSettings.from_file("processing_settings.json")

    assert imported_settings == settings
    print(imported_settings)


@dataclass
class ExportSettings(Settings):
    """Configuration for exporting plots to image files.

    Attributes:
        format: Export format ("png", "svg", "jpeg", "webp"). Defaults to "png".
        height: Export height in pixels. Defaults to 500.
        width: Export width in pixels. Defaults to 800.
        scale: Export scale factor. Defaults to 1.
    """

    format: str = "png"  # one of png, svg, jpeg, webp
    height: int = 500
    width: int = 800
    scale: int = 1

    def to_config(self) -> dict:
        """Converts export settings to plotly configuration dictionary.

        Returns:
            dict: Plotly toImageButtonOptions configuration
        """
        return {
            "toImageButtonOptions": {
                "format": self.format,
                "height": self.height,
                "width": self.width,
                "scale": self.scale,
            }
        }


@dataclass
class ReportSettings(Settings):
    """Configuration for interactive HTML report generation (plotly backend).

    Controls visualization, styling, and export settings for trajectory comparison reports.
    The along-track direction is positive in travel direction, horizontal cross-track is
    positive to the right, and vertical cross-track is positive upwards.

    Attributes:
        single_plot_height: Height of single plots in pixels. Defaults to 750.
        two_subplots_height: Height of two-subplot figures in pixels. Defaults to 750.
        three_subplots_height: Height of three-subplot figures in pixels. Defaults to 860.
        scatter_max_std: Upper colorbar limit as mean + this * std (prevents outlier
            dominance). Defaults to 4.0.
        ate_unit_is_mm: Use millimeters for ATE instead of meters. Defaults to False.
        ate_remove_above: Cap ATE values above this (0.0 = no cap). Defaults to 0.0.
        directed_ate: Split ATE into along-track, horizontal cross-track, and vertical
            cross-track directions. Defaults to False.
        histogram_opacity: Opacity of histogram bars [0-1]. Defaults to 0.7.
        histogram_bargap: Gap between histogram bars. Defaults to 0.1.
        histogram_barmode: Display mode for histogram bars. Defaults to "overlay".
        histogram_yaxis_title: Y-axis label for histograms. Defaults to "Count".
        plot_mode: Display mode for line plots. Defaults to "lines+markers".
        scatter_mode: Display mode for scatter plots. Defaults to "markers".
        scatter_colorscale: Colorscale for scatter plots. Defaults to "RdYlBu_r".
        scatter_axis_order: Axes to display in scatter plots ("xy" or "xyz"). Defaults to "xy".
        scatter_marker_size: Size of scatter plot markers. Defaults to 5.
        scatter_show_individual_dofs: Show scatter plots for each degree of freedom. Defaults to False.
        scatter_smooth: Smooth scatter plot color data. Defaults to False.
        scatter_smooth_window: Window size for scatter smoothing. Defaults to 5.
        scatter_plot_on_map: Plot trajectories on mapbox map (requires valid EPSG). Defaults to False.
        scatter_mapbox_style: Mapbox style (some require token). Defaults to "open-street-map".
        scatter_mapbox_zoom: Mapbox zoom level. Defaults to 15.
        scatter_mapbox_token: Mapbox access token (optional). Defaults to "".
        pos_x_name: Label for X position coordinate. Defaults to "x".
        pos_y_name: Label for Y position coordinate. Defaults to "y".
        pos_z_name: Label for Z position coordinate. Defaults to "z".
        pos_x_unit: Unit symbol for X position. Defaults to "m".
        pos_y_unit: Unit symbol for Y position. Defaults to "m".
        pos_z_unit: Unit symbol for Z position. Defaults to "m".
        directed_pos_dev_x_name: Label for along-track direction deviation. Defaults to "along".
        directed_pos_dev_y_name: Label for horizontal cross-track deviation. Defaults to "cross-h".
        directed_pos_dev_z_name: Label for vertical cross-track deviation. Defaults to "cross-v".
        rot_x_name: Label for roll angle. Defaults to "roll".
        rot_y_name: Label for pitch angle. Defaults to "pitch".
        rot_z_name: Label for yaw angle. Defaults to "yaw".
        rot_unit: Unit symbol for rotation angles. Defaults to "°".
        single_plot_export: Export settings for single plots.
        two_subplots_export: Export settings for two-subplot figures.
        three_subplots_export: Export settings for three-subplot figures.
    """

    single_plot_height: int = 750
    two_subplots_height: int = 750
    three_subplots_height: int = 860

    scatter_max_std: float = 4.0
    ate_unit_is_mm: bool = False
    ate_remove_above: float = 0.0
    directed_ate: bool = False

    histogram_opacity: float = 0.7
    histogram_bargap: float = 0.1
    histogram_barmode: str = "overlay"
    histogram_yaxis_title: str = "Count"

    plot_mode: str = "lines+markers"

    scatter_mode: str = "markers"
    scatter_colorscale: str = "RdYlBu_r"
    scatter_axis_order: str = "xy"
    scatter_marker_size: int = 5
    scatter_show_individual_dofs: bool = False
    scatter_smooth: bool = False
    scatter_smooth_window: int = 5

    scatter_plot_on_map: bool = False
    scatter_mapbox_style: str = "open-street-map"
    scatter_mapbox_zoom: int = 15
    scatter_mapbox_token: str = ""

    pos_x_name: str = "x"
    pos_y_name: str = "y"
    pos_z_name: str = "z"
    pos_x_unit: str = "m"
    pos_y_unit: str = "m"
    pos_z_unit: str = "m"

    directed_pos_dev_x_name: str = "along"
    directed_pos_dev_y_name: str = "cross-h"
    directed_pos_dev_z_name: str = "cross-v"

    rot_x_name: str = "roll"
    rot_y_name: str = "pitch"
    rot_z_name: str = "yaw"
    rot_unit: str = "°"

    single_plot_export: ExportSettings = field(default_factory=lambda: ExportSettings(width=800, height=540))
    two_subplots_export: ExportSettings = field(default_factory=lambda: ExportSettings(width=800, height=540))
    three_subplots_export: ExportSettings = field(default_factory=lambda: ExportSettings(width=800, height=750))


if __name__ == "__main__":
    settings = ReportSettings()
    settings.to_file("report_settings.json")
    imported_settings = ReportSettings.from_file("report_settings.json")

    assert imported_settings == settings
    print(imported_settings)
