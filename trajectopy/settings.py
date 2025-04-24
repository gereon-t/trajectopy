"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import json
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List

import numpy as np

METRIC_THRESHOLD = 1e-4
TIME_THRESHOLD = 1e-4


@dataclass
class Settings(ABC):
    """Base Class for Settings"""

    def to_dict(self) -> dict:
        output = {}
        for key, value in self.__dict__.items():
            if issubclass(type(value), Settings):
                output[key] = value.to_dict()
            else:
                output[key] = self.encoder(key, value)

        return output

    @staticmethod
    def encoder(name: str, value: Any) -> Any:
        """Encoder for json serialization of dataclasses"""
        return value

    @staticmethod
    def decoder(name: str, value: Any) -> Any:
        """Decoder for json deserialization of dataclasses"""
        return value

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, dct: dict) -> "Settings":
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
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def update_from_dict(self, dct: dict):
        for attribute_name, attribute_type in self.__annotations__.items():
            if attribute_name not in dct:
                raise ValueError(f"Attribute {attribute_name} not found in input data")

            attribute_data = dct[attribute_name]
            if isinstance(attribute_data, dict) and issubclass(attribute_type, Settings):
                getattr(self, attribute_name).update_from_dict(attribute_data)
            else:
                setattr(self, attribute_name, self.decoder(attribute_name, attribute_data))


@dataclass
class AlignmentPreprocessing(Settings):
    """Dataclass defining alignment preprocessing configuration"""

    min_speed: float = 0.0
    time_start: float = 0.0
    time_end: float = 0.0


@dataclass
class AlignmentEstimationSettings(Settings):
    """
    Dataclass defining which parameters to estimate
    during the alignment process
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
        return not self.all_lq_disabled

    @classmethod
    def from_components(
        cls,
        similarity: bool = False,
        time_shift: bool = False,
        leverarm: bool = False,
        sensor_rotation: bool = False,
    ) -> "AlignmentEstimationSettings":
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
    def from_bool_list(cls, bool_list: List[bool]) -> "AlignmentEstimationSettings":
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
        """Returns True if all parameters estimated with Least-Squares are disabled"""
        return not any([self.helmert_enabled, self.time_shift_enabled, self.leverarm_enabled])

    @property
    def helmert_enabled(self) -> bool:
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
        return any([self.leverarm_x, self.leverarm_y, self.leverarm_z])

    @property
    def time_shift_enabled(self) -> bool:
        return any(self.time_shift_filter)

    @property
    def short_mode_str(self) -> str:
        """Returns a short string describing the enabled parameters"""
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
    def time_shift_filter(self) -> List[bool]:
        if not self.time_shift:
            return [False] * 3

        return [True] * 3

    @property
    def helmert_filter(self) -> List[bool]:
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
    def leverarm_filter(self) -> List[bool]:
        if not self.leverarm_enabled:
            return [False] * 3

        return [self.leverarm_x, self.leverarm_y, self.leverarm_z]

    @property
    def enabled_lq_parameter_filter(self) -> List[bool]:
        """Returns a list of bools indicating which parameters estimated within LQ are enabled"""
        full_filter = []

        if self.helmert_enabled:
            full_filter.extend(self.helmert_filter)

        if self.time_shift_enabled:
            full_filter.append(self.time_shift)

        if self.leverarm_enabled:
            full_filter.extend(self.leverarm_filter)

        return full_filter

    @property
    def lq_parameter_filter(self) -> List[bool]:
        return self.helmert_filter + [self.time_shift_enabled] + self.leverarm_filter


@dataclass
class AlignmentStochastics(Settings):
    """Dataclass defining alignment stochastics configuration"""

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
        return self.std_xy_from**2

    @property
    def var_z_from(self) -> float:
        return self.std_z_from**2

    @property
    def var_xy_to(self) -> float:
        return self.std_xy_to**2

    @property
    def var_z_to(self) -> float:
        return self.std_z_to**2

    @property
    def var_roll_pitch(self) -> float:
        return self.std_roll_pitch**2

    @property
    def var_yaw(self) -> float:
        return self.std_yaw**2

    @property
    def var_speed_to(self) -> float:
        return self.std_speed**2


@dataclass
class AlignmentSettings(Settings):
    """Dataclass defining alignment configuration

    Args:

        - mode (str): Mode of the H(elmert)-L(everarm)-T(ime) transformation
                          Depending on the presence of the letters "h", "l", "t"
                          inside this string, the alignment will estimate the
                          corresponding parameters
        - std_xx (float): Standard deviations in their corresponding unit
                          of the supported observation_groups:
            - xy_from (source positions)
            - z_from
            - xy_to (target positions)
            - z_to
            - roll_pitch (platform orientations)
            - yaw

    """

    preprocessing: AlignmentPreprocessing = field(default_factory=AlignmentPreprocessing)
    estimation_settings: AlignmentEstimationSettings = field(default_factory=AlignmentEstimationSettings)
    stochastics: AlignmentStochastics = field(default_factory=AlignmentStochastics)
    metric_threshold: float = METRIC_THRESHOLD
    time_threshold: float = TIME_THRESHOLD

    def __str__(self) -> str:
        return str(self.preprocessing) + str(self.estimation_settings) + str(self.stochastics)


if __name__ == "__main__":
    settings = AlignmentSettings()
    settings.to_file("alignment_settings.json")
    imported_settings = AlignmentSettings.from_file("alignment_settings.json")

    assert settings == imported_settings
    print(imported_settings)


@dataclass
class ApproximationSettings(Settings):
    """Dataclass defining approximation configuration"""

    position_interval_size: float = 0.15
    position_min_observations: int = 25
    rotation_window_size: float = 0.15


class PairDistanceUnit(Enum):
    METER = "meter"
    SECOND = "second"

    @classmethod
    def from_str(cls, unit_str: str) -> "PairDistanceUnit":
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


class MatchingMethod(Enum):
    NEAREST_SPATIAL = "nearest_spatial"
    NEAREST_TEMPORAL = "nearest_temporal"
    INTERPOLATION = "interpolation"
    NEAREST_SPATIAL_INTERPOLATED = "nearest_spatial_interpolated"
    UNKNOWN = "unknown"


@dataclass
class MatchingSettings(Settings):
    method: MatchingMethod = MatchingMethod.INTERPOLATION
    max_time_diff: float = 0.01
    max_distance: float = 0.00
    k_nearest: int = 10

    @staticmethod
    def encoder(name: str, value: Any) -> Any:
        return value.value if name == "method" else value

    @staticmethod
    def decoder(name: str, value: Any) -> Any:
        return MatchingMethod(value) if name == "method" else value


if __name__ == "__main__":
    settings = MatchingSettings()
    settings.to_file("matching_settings.json")
    imported_settings = MatchingSettings.from_file("matching_settings.json")

    assert imported_settings == settings
    print(imported_settings)


@dataclass
class MPLPlotSettings(Settings):
    """Dataclass defining plot configuration"""

    colorbar_show_zero_crossing: bool = True
    colorbar_steps: int = 4
    colorbar_max_std: float = 3.0
    scatter_hide_axes: bool = False
    scatter_3d: bool = False
    ate_unit_is_mm: bool = False
    hist_as_stairs: bool = False
    directed_ate: bool = False
    dofs_tab: bool = True
    velocity_tab: bool = True
    height_tab: bool = True

    @property
    def unit_multiplier(self) -> float:
        return 1000 if self.ate_unit_is_mm else 1

    @property
    def unit_str(self) -> str:
        return "[mm]" if self.ate_unit_is_mm else "[m]"


class PlotBackend(Enum):
    PLOTLY = "plotly"
    MPL = "mpl"


@dataclass
class SortingSettings(Settings):
    """This class stores all sorting settings"""

    discard_missing: bool = True
    voxel_size: float = 0.05
    movement_threshold: float = 0.005
    k_nearest: int = 4


@dataclass
class ProcessingSettings(Settings):
    """Settings for processing the trajectory."""

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
    format: str = "png"  # one of png, svg, jpeg, webp
    height: int = 500
    width: int = 800
    scale: int = 1

    def to_config(self) -> dict:
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
    """
    ReportSettings class represents the settings for generating reports.

    - `single_plot_height` (int): The height of a single plot. Default value is 450.
    - `two_subplots_height` (int): The height of two subplots. Default value is 540.
    - `three_subplots_height` (int): The height of three subplots. Default value is 750.
    - `scatter_max_std` (float): The upper colorbar limit is set to the mean plus this value times the standard deviation of the data. This is useful to prevent outliers from dominating the colorbar. Default value is 4.0.
    - `ate_unit_is_mm` (bool): Indicates whether the unit of Absolute Trajectory Error (ATE) is millimeters. Default value is False.
    - `directed_ate` (bool): Indicates whether the ATE is split into along-, horizontal-cross- and vertical-cross-track direction. Default value is True.
    - `histogram_opacity` (float): The opacity of the histogram bars. Default value is 0.7.
    - `histogram_bargap` (float): The gap between histogram bars. Default value is 0.1.
    - `histogram_barmode` (str): The mode of displaying histogram bars. Default value is "overlay".
    - `histogram_yaxis_title` (str): The title of the y-axis in the histogram. Default value is "Count".
    - `plot_mode` (str): The mode of displaying plots. Default value is "lines+markers".
    - `scatter_mode` (str): The mode of displaying scatter plots. Default value is "markers".
    - `scatter_colorscale` (str): The colorscale for scatter plots. Default value is "RdYlBu_r".
    - `scatter_axis_order` (str): The order of the axes in scatter plots. Default value is "xy". If 3d plotting is desired, also specify "z".
    - `scatter_marker_size` (int): The size of markers in scatter plots. Default value is 5.
    - `scatter_show_individual_dofs` (bool): Indicates whether to show scatter plots for each degree of freedom. Default value is False.
    - `scatter_plot_on_map` (bool): Indicates whether to use mapbox for scatter plots. Default value is False.
    - `scatter_mapbox_style` (str): The mapbox style for scatter plots. Default value is "open-street-map".
    - `scatter_mapbox_zoom` (int): The zoom level for scatter plots. Default value is 15.
    - `pos_x_name` (str): The name of the x-coordinate in position data. Default value is "x".
    - `pos_y_name` (str): The name of the y-coordinate in position data. Default value is "y".
    - `pos_z_name` (str): The name of the z-coordinate in position data. Default value is "z".
    - `pos_x_unit` (str): The unit of the x-coordinate in position data. Default value is "m".
    - `pos_y_unit` (str): The unit of the y-coordinate in position data. Default value is "m".
    - `pos_z_unit` (str): The unit of the z-coordinate in position data. Default value is "m".
    - `directed_pos_dev_x_name` (str): The name of the along-track direction deviation in position data. Default value is "along".
    - `directed_pos_dev_y_name` (str): The name of the horizontal-cross-track direction deviation in position data. Default value is "cross-h".
    - `directed_pos_dev_z_name` (str): The name of the vertical-cross-track direction deviation in position data. Default value is "cross-v".
    - `rot_x_name` (str): The name of the roll angle in rotation data. Default value is "roll".
    - `rot_y_name` (str): The name of the pitch angle in rotation data. Default value is "pitch".
    - `rot_z_name` (str): The name of the yaw angle in rotation data. Default value is "yaw".
    - `rot_unit` (str): The unit of rotation angles. Default value is "°".
    - `single_plot_export` (ExportSettings): The export settings for single plots. Default value is an instance of ExportSettings with width=800 and height=450.
    - `two_subplots_export` (ExportSettings): The export settings for two subplots. Default value is an instance of ExportSettings with width=800 and height=540.
    - `three_subplots_export` (ExportSettings): The export settings for three subplots. Default value is an instance of ExportSettings with width=800 and height=750.

    """

    single_plot_height: int = 750
    two_subplots_height: int = 750
    three_subplots_height: int = 860

    scatter_max_std: float = 4.0
    ate_unit_is_mm: bool = False
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
