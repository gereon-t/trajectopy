"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from trajectopy.core.settings.base import Settings

METRIC_THRESHOLD = 1e-4
TIME_THRESHOLD = 1e-4


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

    trans_x: bool = True
    trans_y: bool = True
    trans_z: bool = True
    rot_x: bool = True
    rot_y: bool = True
    rot_z: bool = True
    scale: bool = False

    time_shift: bool = False
    use_x_speed: bool = True
    use_y_speed: bool = True
    use_z_speed: bool = True

    lever_x: bool = False
    lever_y: bool = False
    lever_z: bool = False

    sensor_rotation: bool = False

    auto_update: bool = False

    def __bool__(self) -> bool:
        return not self.all_lq_disabled

    @classmethod
    def from_components(
        cls,
        similarity: bool = False,
        time_shift: bool = False,
        leverarm: bool = False,
        sensor_rotation: bool = False,
        auto_update: bool = False,
    ) -> "AlignmentEstimationSettings":
        return cls(
            trans_x=similarity,
            trans_y=similarity,
            trans_z=similarity,
            rot_x=similarity,
            rot_y=similarity,
            rot_z=similarity,
            scale=similarity,
            time_shift=time_shift,
            use_x_speed=time_shift,
            use_y_speed=time_shift,
            use_z_speed=time_shift,
            lever_x=leverarm,
            lever_y=leverarm,
            lever_z=leverarm,
            sensor_rotation=sensor_rotation,
            auto_update=auto_update,
        )

    @classmethod
    def all(cls, sensor_rotation: bool = True, auto_update: bool = False) -> "AlignmentEstimationSettings":
        return cls(
            trans_x=True,
            trans_y=True,
            trans_z=True,
            rot_x=True,
            rot_y=True,
            rot_z=True,
            scale=True,
            time_shift=True,
            use_x_speed=True,
            use_y_speed=True,
            use_z_speed=True,
            lever_x=True,
            lever_y=True,
            lever_z=True,
            sensor_rotation=sensor_rotation,
            auto_update=auto_update,
        )

    @classmethod
    def from_bool_list(cls, bool_list: List[bool]) -> "AlignmentEstimationSettings":
        if len(bool_list) != 14:
            raise ValueError(
                f"Size mismatch: bool_list must have length 14 (Number of configurable parameters) (got {len(bool_list)})"
            )

        return AlignmentEstimationSettings(
            trans_x=bool_list[0],
            trans_y=bool_list[1],
            trans_z=bool_list[2],
            rot_x=bool_list[3],
            rot_y=bool_list[4],
            rot_z=bool_list[5],
            scale=bool_list[6],
            time_shift=bool_list[7],
            lever_x=bool_list[8],
            lever_y=bool_list[9],
            lever_z=bool_list[10],
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
        return any([self.trans_x, self.trans_y, self.trans_z, self.rot_x, self.rot_y, self.rot_z, self.scale])

    @property
    def leverarm_enabled(self) -> bool:
        return any([self.lever_x, self.lever_y, self.lever_z])

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

        return [self.use_x_speed, self.use_y_speed, self.use_z_speed]

    @property
    def helmert_filter(self) -> List[bool]:
        if not self.helmert_enabled:
            return [False] * 7

        return [self.trans_x, self.trans_y, self.trans_z, self.rot_x, self.rot_y, self.rot_z, self.scale]

    @property
    def leverarm_filter(self) -> List[bool]:
        if not self.leverarm_enabled:
            return [False] * 3

        return [self.lever_x, self.lever_y, self.lever_z]

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
