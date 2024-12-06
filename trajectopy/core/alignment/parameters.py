"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

from abc import ABC
from dataclasses import dataclass, field, fields
from typing import Dict, List

import numpy as np
import pandas as pd

from trajectopy.core.alignment.utils import list2box
from trajectopy.core.definitions import UNIT_FORMAT_RULES, Unit
from trajectopy.core.rotationset import RotationSet
from trajectopy.core.settings.alignment import AlignmentEstimationSettings


class AlignmentParametersError(Exception):
    pass


class Parameter:
    """
    Class representing a Parameter

    A parameter holds a stochastic value
    with some variance. It can be enabled
    and disabled.

    If a parameter is disabled, its value
    value is equal to the default value.
    The default value of a parameter
    should be chosen in a way so that the
    parameter has no effect on computations.
    For example, a default scale value of 1
    will not affect any computations this
    scale parameter is involved in. The same
    holds true for a rotation or translation
    of 0.
    The disabling of parameters is meant to
    be used to exclude parameter from being
    estimated during least-squares adjustment.
    """

    _cnt = 0

    def __init__(
        self,
        value: float,
        variance: float = 0.0,
        default: float = 0.0,
        enabled: bool = True,
        name: str = "",
        unit: Unit = Unit.NONE,
    ) -> None:
        self._value = value
        self._variance = variance
        self.default = default
        self.enabled = enabled
        self.name = name or f"Parameter {Parameter._cnt}"
        self.unit = unit

        Parameter._cnt += 1

    def __str__(self) -> str:
        formatter_value = UNIT_FORMAT_RULES.get(self.unit, {"normal": (1, "-", 4)})["normal"]
        formatter_variance = UNIT_FORMAT_RULES.get(self.unit, {"precise": (1, "-", 2)}).get("precise", formatter_value)
        return f"{self.name:<16} = {self.value*formatter_value[0]:<8.{formatter_value[2]}f}{formatter_value[1]:<4}  s-dev.: {np.sqrt(self.variance)*formatter_variance[0]:<8.{formatter_variance[2]}f}{formatter_variance[1]:<4}"

    def __sub__(self, other: "Parameter") -> "Parameter":
        return Parameter(
            value=self.value - other.value,
            variance=np.sqrt(self.variance**2 + other.variance**2),
        )

    def __add__(self, other: "Parameter") -> "Parameter":
        return Parameter(
            value=self.value + other.value,
            variance=np.sqrt(self.variance**2 + other.variance**2),
        )

    def __neg__(self) -> "Parameter":
        return Parameter(
            value=-self.value,
            variance=self.variance,
            default=-self.default,
            enabled=self.enabled,
            name=self.name,
            unit=self.unit,
        )

    @property
    def value(self) -> float:
        return self._value if self.enabled else self.default

    @value.setter
    def value(self, val: float) -> None:
        self._value = val

    @property
    def variance(self) -> float:
        return self._variance if self.enabled else 0.0

    @variance.setter
    def variance(self, val: float) -> None:
        self._variance = val

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False


@dataclass
class ParameterSet(ABC):
    """
    Abstract class representing a set of parameters

    This class groups related parameters.
    For example, 3 parameters for a 3d
    translation.
    """

    enabled: bool = True
    covariance_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))

    def __post_init__(self) -> None:
        if len(self.covariance_matrix) == len(self):
            self.validate_covariance()
            return

        self.covariance_matrix = np.zeros((len(self), len(self)), dtype=float)

    def __len__(self) -> int:
        return len(self.values)

    def __str__(self) -> str:
        return list2box(
            input_list=self.to_string_list(),
            title=self.__class__.__name__,
            upper_boundary=True,
            lower_boundary=True,
        )

    def __iter__(self):
        return iter(self.__parameter_fields)

    @property
    def any_enabled(self) -> bool:
        return self.num_enabled > 0

    @property
    def num_enabled(self) -> int:
        return len(self.values_enabled)

    def get_covariance_matrix(self, enabled_only: bool = True) -> np.ndarray:
        if enabled_only:
            return self.covariance_matrix[np.ix_(self.enabled_indices, self.enabled_indices)]

        return self.covariance_matrix

    def set_covariance_matrix(self, cov_matrix: np.ndarray) -> None:
        self.covariance_matrix = cov_matrix
        self.validate_covariance()

    def validate_covariance(self):
        """Checks dimensions of covariance matrix and sets variances accordingly"""
        if self.covariance_matrix.shape[0] != self.covariance_matrix.shape[1]:
            raise AlignmentParametersError("Covariance matrix must be square.")

        if self.covariance_matrix.shape[0] == len(self):
            self.variances = np.diag(self.covariance_matrix)
            return

        if self.covariance_matrix.shape[0] != self.num_enabled:
            raise AlignmentParametersError(
                "Covariance matrix must either be valid for all parameters or for all enabled parameters!"
            )

        covariance_matrix = np.zeros((len(self), len(self)), dtype=float)
        # for sub_row_indiex, full_row_index in enumerate(self.enabled_indices):
        #     for sub_col_index, full_col_index in enumerate(self.enabled_indices):
        #         covariance_matrix[full_row_index, full_col_index] = self.covariance_matrix[
        #             sub_row_indiex, sub_col_index
        #         ]
        covariance_matrix[np.ix_(self.enabled_indices, self.enabled_indices)] = self.covariance_matrix
        self.covariance_matrix = covariance_matrix
        self.variances_enabled = np.diag(self.get_covariance_matrix(enabled_only=True))

    def enable(self) -> None:
        for param in self.__parameter_fields:
            param.enable()
        self.enabled = True

    def disable(self) -> None:
        for param in self.__parameter_fields:
            param.disable()
        self.enabled = False

    @property
    def enabled_bool_list(self) -> List[bool]:
        return [param.enabled for param in self.__parameter_fields]

    @enabled_bool_list.setter
    def enabled_bool_list(self, enabled_list: List[bool]) -> None:
        for param, enabled in zip(self.__parameter_fields, enabled_list):
            param.enabled = enabled

    @property
    def enabled_indices(self) -> List[int]:
        return [i for i, value in enumerate(self.enabled_bool_list) if value]

    def to_string_list(self, enabled_only: bool = True) -> List[str]:
        return [str(param) for param in self.__parameter_fields if param.enabled or not enabled_only]

    def to_name_list(self, enabled_only: bool = True, lower_case: bool = True) -> List[str]:
        return [
            param.name.lower() if lower_case else param.name
            for param in self.__parameter_fields
            if param.enabled or not enabled_only
        ]

    def to_dict(self, enabled_only: bool = True) -> Dict[str, List[float]]:
        return {
            param.name: [param.value, param.variance]
            for param in self.__parameter_fields
            if param.enabled or not enabled_only
        }

    @property
    def __parameter_fields(self) -> List[Parameter]:
        """Returns a list of all Parameters of the class"""
        parameter_list = []
        for cfield in fields(self):
            attr = getattr(self, cfield.name)
            if isinstance(attr, Parameter):
                parameter_list.append(attr)

        return parameter_list

    def _get(self, key: str, enabled_only: bool = False) -> np.ndarray:
        """Returns a numpy array of all parameter values or variances of the class"""
        val_list = []
        for param in self.__parameter_fields:
            if param.enabled is False and enabled_only:
                continue

            if key == "value":
                val_list.append(param.value)
            elif key == "variance":
                val_list.append(param.variance)

        return np.array(val_list)

    def _set(self, vals: np.ndarray, key: str, enabled_only: bool = False) -> None:
        """Sets all parameter values or variances of the class"""
        if self.num_enabled != len(vals) and enabled_only:
            raise ValueError(
                f"Number of values to set must be equal to number of enabled parameters! ({self.num_enabled} != {len(vals)})"
            )

        if len(self) != len(vals) and not enabled_only:
            raise ValueError(
                f"Number of values to set must be equal to number of parameters! ({len(self)} != {len(vals)})"
            )

        val_idx = 0
        for param in self.__parameter_fields:
            if param.enabled is False and enabled_only:
                continue

            if key == "value":
                param.value = vals[val_idx]
            elif key == "variance":
                param.variance = vals[val_idx]

            val_idx += 1

    @property
    def values(self) -> np.ndarray:
        return self._get(key="value")

    @values.setter
    def values(self, vals: np.ndarray) -> None:
        self._set(vals=vals, key="value")

    @property
    def values_enabled(self) -> np.ndarray:
        return self._get(key="value", enabled_only=True)

    @values_enabled.setter
    def values_enabled(self, vals: np.ndarray) -> None:
        self._set(vals=vals, key="value", enabled_only=True)

    @property
    def variances(self) -> np.ndarray:
        return self._get(key="variance")

    @variances.setter
    def variances(self, vals: np.ndarray) -> None:
        self._set(vals=vals, key="variance")
        np.fill_diagonal(self.covariance_matrix, vals)

    @property
    def variances_enabled(self) -> np.ndarray:
        return self._get(key="variance", enabled_only=True)

    @variances_enabled.setter
    def variances_enabled(self, vals: np.ndarray) -> None:
        self._set(vals=vals, key="variance", enabled_only=True)
        self.covariance_matrix[self.enabled_indices, self.enabled_indices] = vals


@dataclass
class HelmertTransformation(ParameterSet):
    """Parameter set for a similarity transformation"""

    trans_x: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    trans_y: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    trans_z: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    rot_x: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    rot_y: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    rot_z: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    scale: Parameter = field(default_factory=lambda: Parameter(value=1.0, default=1.0))

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.trans_x.value, self.trans_y.value, self.trans_z.value])

    @translation.setter
    def translation(self, v: np.ndarray) -> None:
        self.trans_x.value = v[0]
        self.trans_y.value = v[1]
        self.trans_z.value = v[2]

    @property
    def rotation(self) -> np.ndarray:
        return np.array([self.rot_x.value, self.rot_y.value, self.rot_z.value])

    @rotation.setter
    def rotation(self, v: np.ndarray) -> None:
        self.rot_x.value = v[0]
        self.rot_y.value = v[1]
        self.rot_z.value = v[2]

    @property
    def rotation_set(self) -> RotationSet:
        return RotationSet.from_euler(seq="xyz", angles=self.rotation)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return RotationSet.from_euler(seq="xyz", angles=self.rotation).as_matrix()

    def apply_to(self, xyz: np.ndarray) -> np.ndarray:
        return self.translation + self.scale.value * (self.rotation_matrix @ xyz.T).T


@dataclass
class Leverarm(ParameterSet):
    """Parameter set for a leverarm"""

    x: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Leverarm x", enabled=False, unit=Unit.METER)
    )
    y: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Leverarm y", enabled=False, unit=Unit.METER)
    )
    z: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Leverarm z", enabled=False, unit=Unit.METER)
    )

    def apply_to(self, xyz: np.ndarray, quat_body: np.ndarray) -> np.ndarray:
        """Applies the leverarm to a set of positions using orientations

        Args:
            xyz (np.ndarray): Positions
            quat_body (np.ndarray): Orientations

        Returns:
            np.ndarray: Leverarm applied positions
        """
        if len(xyz) != len(quat_body):
            raise ValueError("Dimension mismatch between positions and orientations.")

        xyz_red = np.zeros((xyz.shape))
        rot_matrices = RotationSet.from_quat(quat_body).as_matrix()

        for i in range(len(xyz_red)):
            d_rot = rot_matrices[i] @ self.values
            xyz_red[i, :] = xyz[i, :] + d_rot

        return xyz_red


@dataclass
class AlignmentParameters(ParameterSet):
    """Parameter set for spatio-temporal alignment"""

    sim_trans_x: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Translation x", enabled=False, unit=Unit.METER)
    )
    sim_trans_y: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Translation y", enabled=False, unit=Unit.METER)
    )
    sim_trans_z: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Translation z", enabled=False, unit=Unit.METER)
    )
    sim_rot_x: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Rotation x", enabled=False, unit=Unit.RADIAN)
    )
    sim_rot_y: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Rotation y", enabled=False, unit=Unit.RADIAN)
    )
    sim_rot_z: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Rotation z", enabled=False, unit=Unit.RADIAN)
    )
    sim_scale: Parameter = field(
        default_factory=lambda: Parameter(value=1.0, default=1.0, name="Scale", enabled=False, unit=Unit.SCALE)
    )
    time_shift: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Time Shift", enabled=False, unit=Unit.SECOND)
    )
    lever_x: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Leverarm x", enabled=False, unit=Unit.METER)
    )
    lever_y: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Leverarm y", enabled=False, unit=Unit.METER)
    )
    lever_z: Parameter = field(
        default_factory=lambda: Parameter(value=0.0, default=0.0, name="Leverarm z", enabled=False, unit=Unit.METER)
    )

    def __add__(self, other: "AlignmentParameters") -> "AlignmentParameters":
        return self.values + other.values

    def __sub__(self, other: "AlignmentParameters") -> "AlignmentParameters":
        return self.values - other.values

    def params_labels(self, enabled_only: bool = True, lower_case: bool = True) -> List[str]:
        return self.to_name_list(enabled_only=enabled_only, lower_case=lower_case)

    @property
    def sim3_matrix(self) -> np.ndarray:
        rot_m = self.helmert.rotation_matrix
        trafo = self.helmert.translation
        scale = self.helmert.scale.value

        sim3_matrix = np.eye(4)
        sim3_matrix[:3, :3] = scale * rot_m
        sim3_matrix[:3, 3] = trafo
        return sim3_matrix

    @classmethod
    def from_settings(cls, settings: AlignmentEstimationSettings) -> "AlignmentParameters":
        return cls(
            sim_trans_x=Parameter(enabled=settings.trans_x, value=0.0, default=0.0),
            sim_trans_y=Parameter(enabled=settings.trans_x, value=0.0, default=0.0),
            sim_trans_z=Parameter(enabled=settings.trans_x, value=0.0, default=0.0),
            sim_rot_x=Parameter(enabled=settings.rot_x, value=0.0, default=0.0),
            sim_rot_y=Parameter(enabled=settings.rot_y, value=0.0, default=0.0),
            sim_rot_z=Parameter(enabled=settings.rot_z, value=0.0, default=0.0),
            sim_scale=Parameter(enabled=settings.scale, value=1.0, default=1.0),
            time_shift=Parameter(enabled=settings.time_shift, value=0.0, default=0.0),
            lever_x=Parameter(enabled=settings.lever_x, value=0.0, default=0.0),
            lever_y=Parameter(enabled=settings.lever_y, value=0.0, default=0.0),
            lever_z=Parameter(enabled=settings.lever_z, value=0.0, default=0.0),
        )

    def apply_settings(self, settings: AlignmentEstimationSettings) -> None:
        """Applies the estimation settings to the parameters by enabling or disabling them"""
        self.sim_trans_x.enabled = settings.trans_x and settings.helmert_enabled
        self.sim_trans_y.enabled = settings.trans_y and settings.helmert_enabled
        self.sim_trans_z.enabled = settings.trans_z and settings.helmert_enabled
        self.sim_rot_x.enabled = settings.rot_x and settings.helmert_enabled
        self.sim_rot_y.enabled = settings.rot_y and settings.helmert_enabled
        self.sim_rot_z.enabled = settings.rot_z and settings.helmert_enabled
        self.sim_scale.enabled = settings.scale and settings.helmert_enabled

        self.time_shift.enabled = settings.time_shift_enabled

        self.lever_x.enabled = settings.lever_x and settings.leverarm_enabled
        self.lever_y.enabled = settings.lever_y and settings.leverarm_enabled
        self.lever_z.enabled = settings.lever_z and settings.leverarm_enabled

    @property
    def helmert(self) -> HelmertTransformation:
        return HelmertTransformation(
            trans_x=self.sim_trans_x,
            trans_y=self.sim_trans_y,
            trans_z=self.sim_trans_z,
            scale=self.sim_scale,
            rot_x=self.sim_rot_x,
            rot_y=self.sim_rot_y,
            rot_z=self.sim_rot_z,
        )

    @property
    def leverarm(self) -> Leverarm:
        return Leverarm(x=self.lever_x, y=self.lever_y, z=self.lever_z)

    def to_file(self, filename: str) -> None:
        """Writes the alignment parameters to a file

        All parameters are written to the file no matter if they are enabled or not.
        If they are not enabled, the default value is written. Enabled parameters are
        marked with a 1, disabled with a 0. The order of the parameters is:

        - Translation x
        - Translation y
        - Translation z
        - Rotation x
        - Rotation y
        - Rotation z
        - Scale
        - Time Shift
        - Leverarm x
        - Leverarm y
        - Leverarm z

        Besides the parameters, the covariance matrix is written to the file. The covariance
        matrix is written row by row next to the parameters.

        Args:
            filename (str): Path to the file
        """
        dataframe = pd.DataFrame(np.c_[self.values, self.covariance_matrix])
        dataframe.to_csv(filename, index=False, header=False)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(np.c_[self.values, self.enabled_bool_list, self.covariance_matrix])

    @classmethod
    def from_file(cls, filename: str) -> "AlignmentParameters":
        """Reads the alignment parameters from a file

        Args:
            filename (str): Path to the file

        Returns:
            AlignmentParameters: AlignmentParameters instance
        """
        alignment_data = pd.read_csv(filename, comment="#", header=None)
        params: AlignmentParameters = cls()
        params.values = alignment_data.iloc[:11, 0].to_numpy()
        params.enabled_bool_list = [item != 0.0 for item in alignment_data.iloc[:11, 1].to_list()]
        params.set_covariance_matrix(alignment_data.iloc[:11, 2:].to_numpy())
        return params

    def to_estimation_settings(self) -> AlignmentEstimationSettings:
        return AlignmentEstimationSettings(
            trans_x=self.sim_trans_x.enabled,
            trans_y=self.sim_trans_y.enabled,
            trans_z=self.sim_trans_z.enabled,
            rot_x=self.sim_rot_x.enabled,
            rot_y=self.sim_rot_y.enabled,
            rot_z=self.sim_rot_z.enabled,
            scale=self.sim_scale.enabled,
            time_shift=self.time_shift.enabled,
            lever_x=self.lever_x.enabled,
            lever_y=self.lever_y.enabled,
            lever_z=self.lever_z.enabled,
        )


@dataclass
class SensorRotationParameters(ParameterSet):
    """Parameter set for sensor b-frame rotation"""

    sensor_rot_x: Parameter = field(
        default_factory=lambda: Parameter(
            value=0.0,
            default=0.0,
            name="Sensor Rotation x",
            enabled=False,
            unit=Unit.RADIAN,
        )
    )

    sensor_rot_y: Parameter = field(
        default_factory=lambda: Parameter(
            value=0.0,
            default=0.0,
            name="Sensor Rotation y",
            enabled=False,
            unit=Unit.RADIAN,
        )
    )

    sensor_rot_z: Parameter = field(
        default_factory=lambda: Parameter(
            value=0.0,
            default=0.0,
            name="Sensor Rotation z",
            enabled=False,
            unit=Unit.RADIAN,
        )
    )

    @property
    def rotation(self) -> np.ndarray:
        return np.array([self.sensor_rot_x.value, self.sensor_rot_y.value, self.sensor_rot_z.value])

    @property
    def rotation_set(self) -> RotationSet:
        return RotationSet.from_euler(seq="xyz", angles=self.rotation)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return RotationSet.from_euler(seq="xyz", angles=self.rotation).as_matrix()

    @classmethod
    def from_file(cls, filename: str) -> "SensorRotationParameters":
        alignment_data = pd.read_csv(filename, comment="#", header=None)

        if len(alignment_data) != 14:
            raise ValueError("Alignment file with sensor rotation information must contain 14 lines!")
        enabled_bool_list = [item != 0.0 for item in alignment_data.iloc[11:, 1].to_list()]
        params: SensorRotationParameters = cls()
        params.values = alignment_data.iloc[11:, 0].to_numpy()
        params.enabled_bool_list = enabled_bool_list
        return params

    def to_file(self, filename: str) -> None:
        with open(filename, mode="a", encoding="utf-8") as file:
            file.write(f"{self.sensor_rot_x.value},{int(self.sensor_rot_x.enabled)}\n")
            file.write(f"{self.sensor_rot_y.value},{int(self.sensor_rot_y.enabled)}\n")
            file.write(f"{self.sensor_rot_z.value},{int(self.sensor_rot_z.enabled)}\n")
