from abc import ABC
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import AlignmentEstimationSettings
from trajectopy.exceptions import AlignmentParametersError
from trajectopy.utils.common import list2box
from trajectopy.utils.definitions import UNIT_FORMAT_RULES, Unit


class Parameter:
    """Class representing a stochastic parameter with variance.

    A parameter holds a stochastic value with some variance and can be enabled
    or disabled. When disabled, the parameter's value equals its default value.
    The default value should be chosen so the parameter has no effect on
    computations (e.g., scale=1, rotation/translation=0).

    Parameter disabling is used to exclude parameters from least-squares
    estimation during trajectory alignment.

    Attributes:
        default: Default value used when parameter is disabled
        enabled: Whether the parameter is active
        name: Parameter name for identification
        unit: Unit of measurement for the parameter
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
        """Initializes a Parameter instance.

        Args:
            value: Initial parameter value
            variance: Initial parameter variance. Defaults to 0.0.
            default: Default value when disabled. Defaults to 0.0.
            enabled: Whether parameter is enabled. Defaults to True.
            name: Parameter name. Auto-generated if empty. Defaults to "".
            unit: Unit of measurement. Defaults to Unit.NONE.
        """
        self._value = value
        self._variance = variance
        self.default = default
        self.enabled = enabled
        self.name = name or f"Parameter {Parameter._cnt}"
        self.unit = unit

        Parameter._cnt += 1

    def __str__(self) -> str:
        """Returns formatted string representation of the parameter.

        Returns:
            str: Formatted string with parameter name, value, unit, and standard deviation
        """
        formatter_value = UNIT_FORMAT_RULES.get(self.unit, {"normal": (1, "-", 4)})["normal"]
        formatter_variance = UNIT_FORMAT_RULES.get(self.unit, {"precise": (1, "-", 2)}).get("precise", formatter_value)
        return f"{self.name:<16} = {self.value*formatter_value[0]:<8.{formatter_value[2]}f}{formatter_value[1]:<4}  s-dev.: {np.sqrt(self.variance)*formatter_variance[0]:<8.{formatter_variance[2]}f}{formatter_variance[1]:<4}"

    def __sub__(self, other: "Parameter") -> "Parameter":
        """Subtracts another parameter from this one.

        Args:
            other: Parameter to subtract

        Returns:
            Parameter: New parameter with subtracted value and combined variance
        """
        return Parameter(
            value=self.value - other.value,
            variance=np.sqrt(self.variance**2 + other.variance**2),
        )

    def __add__(self, other: "Parameter") -> "Parameter":
        """Adds another parameter to this one.

        Args:
            other: Parameter to add

        Returns:
            Parameter: New parameter with added value and combined variance
        """
        return Parameter(
            value=self.value + other.value,
            variance=np.sqrt(self.variance**2 + other.variance**2),
        )

    def __neg__(self) -> "Parameter":
        """Negates the parameter.

        Returns:
            Parameter: New parameter with negated value and default
        """
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
        """Gets the parameter value.

        Returns:
            float: Actual value if enabled, default value otherwise
        """
        return self._value if self.enabled else self.default

    @value.setter
    def value(self, val: float) -> None:
        """Sets the parameter value.

        Args:
            val: New parameter value
        """
        self._value = val

    @property
    def variance(self) -> float:
        """Gets the parameter variance.

        Returns:
            float: Actual variance if enabled, 0.0 otherwise
        """
        return self._variance if self.enabled else 0.0

    @variance.setter
    def variance(self, val: float) -> None:
        """Sets the parameter variance.

        Args:
            val: New parameter variance
        """
        self._variance = val

    def enable(self) -> None:
        """Enables the parameter for estimation."""
        self.enabled = True

    def disable(self) -> None:
        """Disables the parameter from estimation."""
        self.enabled = False


@dataclass
class ParameterSet(ABC):
    """Abstract base class for grouping related parameters.

    Groups related parameters (e.g., 3 parameters for 3D translation) and provides
    common functionality for managing parameter values, variances, and covariances.

    Attributes:
        enabled: Whether the entire parameter set is enabled
        covariance_matrix: Covariance matrix for the parameter set
    """

    enabled: bool = True
    covariance_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))

    def __post_init__(self) -> None:
        """Initializes covariance matrix after dataclass creation."""
        if len(self.covariance_matrix) == len(self):
            self.validate_covariance()
            return

        self.covariance_matrix = np.zeros((len(self), len(self)), dtype=float)

    def __len__(self) -> int:
        """Returns the number of parameters in the set.

        Returns:
            int: Number of parameters
        """
        return len(self.values)

    def __str__(self) -> str:
        """Returns formatted string representation of the parameter set.

        Returns:
            str: Boxed string with all parameter information
        """
        return list2box(
            input_list=self.to_string_list(),
            title=self.__class__.__name__,
            upper_boundary=True,
            lower_boundary=True,
        )

    def __iter__(self) -> iter:
        """Returns iterator over parameter fields.

        Returns:
            Iterator: Iterator over Parameter instances
        """
        return iter(self.__parameter_fields)

    @property
    def any_enabled(self) -> bool:
        """Checks if any parameter in the set is enabled.

        Returns:
            bool: True if at least one parameter is enabled
        """
        return self.num_enabled > 0

    @property
    def num_enabled(self) -> int:
        """Gets the number of enabled parameters in the set.

        Returns:
            int: Count of enabled parameters
        """
        return len(self.values_enabled)

    def get_covariance_matrix(self, enabled_only: bool = True) -> np.ndarray:
        """Returns the covariance matrix of the parameter set

        Args:
            enabled_only (bool, optional): If True, only the covariance matrix of the enabled parameters is returned. Defaults to True.

        Returns:
            np.ndarray: Covariance matrix
        """
        if enabled_only:
            return self.covariance_matrix[np.ix_(self.enabled_indices, self.enabled_indices)]

        return self.covariance_matrix

    def set_covariance_matrix(self, cov_matrix: np.ndarray) -> None:
        """Sets the covariance matrix of the parameter set

        Args:
            cov_matrix (np.ndarray): Covariance matrix
        """
        self.covariance_matrix = cov_matrix
        self.validate_covariance()

    def validate_covariance(self) -> None:
        """Validates and normalizes the covariance matrix dimensions.

        Checks that the covariance matrix is square and matches either all parameters
        or only enabled parameters. Updates parameter variances from diagonal elements.

        Raises:
            AlignmentParametersError: If covariance matrix dimensions are invalid
        """
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
        """Enables all parameters in the set."""
        for param in self.__parameter_fields:
            param.enable()
        self.enabled = True

    def disable(self) -> None:
        """Disables all parameters in the set."""
        for param in self.__parameter_fields:
            param.disable()
        self.enabled = False

    @property
    def enabled_bool_list(self) -> list[bool]:
        """Gets the enabled state of all parameters as a boolean list.

        Returns:
            List[bool]: List of booleans indicating which parameters are enabled
        """
        return [param.enabled for param in self.__parameter_fields]

    @enabled_bool_list.setter
    def enabled_bool_list(self, enabled_list: list[bool]) -> None:
        """Sets the enabled state of parameters from a boolean list.

        Args:
            enabled_list: List of booleans indicating which parameters to enable
        """
        for param, enabled in zip(self.__parameter_fields, enabled_list):
            param.enabled = enabled

    @property
    def enabled_indices(self) -> list[int]:
        """Gets indices of enabled parameters.

        Useful for extracting values or variances of only enabled parameters
        from the full parameter arrays.

        Returns:
            List[int]: List of indices where parameters are enabled
        """
        return [i for i, value in enumerate(self.enabled_bool_list) if value]

    def to_string_list(self, enabled_only: bool = True) -> list[str]:
        """Returns a list of all parameter strings of the class

        Args:
            enabled_only (bool, optional): If True, only enabled parameters are included. Defaults to True.

        Returns:
            List[str]: List of parameter strings
        """
        return [str(param) for param in self.__parameter_fields if param.enabled or not enabled_only]

    def to_name_list(self, enabled_only: bool = True, lower_case: bool = True) -> list[str]:
        """Returns a list of all parameter names of the class

        Args:
            enabled_only (bool, optional): If True, only enabled parameters are included. Defaults to True.
            lower_case (bool, optional): If True, parameter names are returned in lower case. Defaults to True.

        Returns:
            List[str]: List of parameter names
        """
        return [
            param.name.lower() if lower_case else param.name
            for param in self.__parameter_fields
            if param.enabled or not enabled_only
        ]

    def to_dict(self, enabled_only: bool = True) -> dict[str, list[float]]:
        """Returns a dictionary of all parameters of the class

        Args:
            enabled_only (bool, optional): If True, only enabled parameters are included. Defaults to True.

        Returns:
            Dict[str, List[float]]: Dictionary with parameter names as keys and [value, variance] as values
        """
        return {
            param.name: [param.value, param.variance]
            for param in self.__parameter_fields
            if param.enabled or not enabled_only
        }

    @property
    def __parameter_fields(self) -> list[Parameter]:
        """Gets all Parameter instances from dataclass fields.

        Returns:
            List[Parameter]: List of all Parameter objects in the set
        """
        parameter_list = []
        for cfield in fields(self):
            attr = getattr(self, cfield.name)
            if isinstance(attr, Parameter):
                parameter_list.append(attr)

        return parameter_list

    def _get(self, key: str, enabled_only: bool = False) -> np.ndarray:
        """Gets parameter values or variances as a numpy array.

        Args:
            key: Either 'value' or 'variance' to specify which attribute to retrieve
            enabled_only: If True, only include enabled parameters. Defaults to False.

        Returns:
            np.ndarray: Array of parameter values or variances
        """
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
        """Sets parameter values or variances from a numpy array.

        Args:
            vals: Array of values to set
            key: Either 'value' or 'variance' to specify which attribute to set
            enabled_only: If True, only set enabled parameters. Defaults to False.

        Raises:
            ValueError: If array length doesn't match parameter count
        """
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
        """Gets all parameter values as a numpy array.

        Returns:
            np.ndarray: Array of all parameter values
        """
        return self._get(key="value")

    @values.setter
    def values(self, vals: np.ndarray) -> None:
        """Sets all parameter values from a numpy array.

        Args:
            vals: Array of values to set
        """
        self._set(vals=vals, key="value")

    @property
    def values_enabled(self) -> np.ndarray:
        """Gets enabled parameter values as a numpy array.

        Returns:
            np.ndarray: Array of enabled parameter values
        """
        return self._get(key="value", enabled_only=True)

    @values_enabled.setter
    def values_enabled(self, vals: np.ndarray) -> None:
        """Sets enabled parameter values from a numpy array.

        Args:
            vals: Array of values to set
        """
        self._set(vals=vals, key="value", enabled_only=True)

    @property
    def variances(self) -> np.ndarray:
        """Gets all parameter variances as a numpy array.

        Returns:
            np.ndarray: Array of all parameter variances
        """
        return self._get(key="variance")

    @variances.setter
    def variances(self, vals: np.ndarray) -> None:
        """Sets all parameter variances from a numpy array.

        Also updates the diagonal of the covariance matrix.

        Args:
            vals: Array of variances to set
        """
        self._set(vals=vals, key="variance")
        np.fill_diagonal(self.covariance_matrix, vals)

    @property
    def variances_enabled(self) -> np.ndarray:
        """Gets enabled parameter variances as a numpy array.

        Returns:
            np.ndarray: Array of enabled parameter variances
        """
        return self._get(key="variance", enabled_only=True)

    @variances_enabled.setter
    def variances_enabled(self, vals: np.ndarray) -> None:
        """Sets enabled parameter variances from a numpy array.

        Also updates the corresponding elements of the covariance matrix.

        Args:
            vals: Array of variances to set
        """
        self._set(vals=vals, key="variance", enabled_only=True)
        self.covariance_matrix[self.enabled_indices, self.enabled_indices] = vals


@dataclass
class HelmertTransformation(ParameterSet):
    """Parameter set for a 7-parameter Helmert similarity transformation.

    Represents a 3D similarity transformation with translation, rotation, and scale.

    Attributes:
        trans_x: Translation parameter in x-direction
        trans_y: Translation parameter in y-direction
        trans_z: Translation parameter in z-direction
        rot_x: Rotation parameter around x-axis
        rot_y: Rotation parameter around y-axis
        rot_z: Rotation parameter around z-axis
        scale: Scale parameter
    """

    trans_x: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    trans_y: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    trans_z: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    rot_x: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    rot_y: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    rot_z: Parameter = field(default_factory=lambda: Parameter(value=0.0, default=0.0))
    scale: Parameter = field(default_factory=lambda: Parameter(value=1.0, default=1.0))

    @property
    def translation(self) -> np.ndarray:
        """Gets the 3D translation vector.

        Returns:
            np.ndarray: Translation vector [x, y, z]
        """
        return np.array([self.trans_x.value, self.trans_y.value, self.trans_z.value])

    @translation.setter
    def translation(self, v: np.ndarray) -> None:
        """Sets the 3D translation vector.

        Args:
            v: Translation vector [x, y, z]
        """
        self.trans_x.value = v[0]
        self.trans_y.value = v[1]
        self.trans_z.value = v[2]

    @property
    def rotation(self) -> np.ndarray:
        """Gets the Euler rotation angles.

        Returns:
            np.ndarray: Rotation angles [rx, ry, rz] in radians (XYZ order)
        """
        return np.array([self.rot_x.value, self.rot_y.value, self.rot_z.value])

    @rotation.setter
    def rotation(self, v: np.ndarray) -> None:
        """Sets the Euler rotation angles.

        Args:
            v: Rotation angles [rx, ry, rz] in radians (XYZ order)
        """
        self.rot_x.value = v[0]
        self.rot_y.value = v[1]
        self.rot_z.value = v[2]

    @property
    def rotation_set(self) -> Rotations:
        """Gets the rotation as a Rotations object.

        Returns:
            Rotations: Rotation object from Euler angles
        """
        return Rotations.from_euler(seq="xyz", angles=self.rotation)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Gets the 3x3 rotation matrix.

        Returns:
            np.ndarray: Rotation matrix (3x3)
        """
        return Rotations.from_euler(seq="xyz", angles=self.rotation).as_matrix()

    def apply_to(self, xyz: np.ndarray) -> np.ndarray:
        """Applies the Helmert transformation to 3D points.

        Args:
            xyz: Input points (Nx3 array)

        Returns:
            np.ndarray: Transformed points (Nx3 array)
        """
        return self.translation + self.scale.value * (self.rotation_matrix @ xyz.T).T


@dataclass
class Leverarm(ParameterSet):
    """Parameter set for a leverarm offset between sensor and body frame.

    Represents the 3D offset between a sensor frame and the vehicle body frame.

    Attributes:
        x: Leverarm parameter in x-direction
        y: Leverarm parameter in y-direction
        z: Leverarm parameter in z-direction
    """

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

        xyz_red = np.zeros(xyz.shape)
        rot_matrices = Rotations.from_quat(quat_body).as_matrix()

        for i in range(len(xyz_red)):
            d_rot = rot_matrices[i] @ self.values
            xyz_red[i, :] = xyz[i, :] + d_rot

        return xyz_red


@dataclass
class AlignmentParameters(ParameterSet):
    """Parameter set for spatio-temporal trajectory alignment.

    Contains all parameters for aligning two trajectories including similarity
    transformation (translation, rotation, scale), time shift, and leverarm.

    Attributes:
        sim_trans_x: Similarity transformation translation in x
        sim_trans_y: Similarity transformation translation in y
        sim_trans_z: Similarity transformation translation in z
        sim_rot_x: Similarity transformation rotation around x
        sim_rot_y: Similarity transformation rotation around y
        sim_rot_z: Similarity transformation rotation around z
        sim_scale: Similarity transformation scale factor
        time_shift: Temporal offset between trajectories
        lever_x: Leverarm offset in x
        lever_y: Leverarm offset in y
        lever_z: Leverarm offset in z
    """

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
        """Adds parameter values from another alignment parameter set.

        Args:
            other: AlignmentParameters to add

        Returns:
            AlignmentParameters: Sum of parameter values
        """
        return self.values + other.values

    def __sub__(self, other: "AlignmentParameters") -> "AlignmentParameters":
        """Subtracts parameter values from another alignment parameter set.

        Args:
            other: AlignmentParameters to subtract

        Returns:
            AlignmentParameters: Difference of parameter values
        """
        return self.values - other.values

    def params_labels(self, enabled_only: bool = True, lower_case: bool = True) -> list[str]:
        """Gets parameter labels/names.

        Args:
            enabled_only: If True, only include enabled parameters. Defaults to True.
            lower_case: If True, return lowercase labels. Defaults to True.

        Returns:
            List[str]: List of parameter labels
        """
        return self.to_name_list(enabled_only=enabled_only, lower_case=lower_case)

    @property
    def sim3_matrix(self) -> np.ndarray:
        """Gets the 4x4 similarity transformation matrix (Sim(3)).

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix
        """
        rot_m = self.helmert.rotation_matrix
        trafo = self.helmert.translation
        scale = self.helmert.scale.value

        sim3_matrix = np.eye(4)
        sim3_matrix[:3, :3] = scale * rot_m
        sim3_matrix[:3, 3] = trafo
        return sim3_matrix

    @classmethod
    def from_settings(cls, settings: AlignmentEstimationSettings) -> "AlignmentParameters":
        """Creates AlignmentParameters from estimation settings.

        Args:
            settings: Settings specifying which parameters to enable

        Returns:
            AlignmentParameters: New instance with parameters enabled per settings
        """
        return cls(
            sim_trans_x=Parameter(enabled=settings.translation_x, value=0.0, default=0.0),
            sim_trans_y=Parameter(enabled=settings.translation_x, value=0.0, default=0.0),
            sim_trans_z=Parameter(enabled=settings.translation_x, value=0.0, default=0.0),
            sim_rot_x=Parameter(enabled=settings.rotation_x, value=0.0, default=0.0),
            sim_rot_y=Parameter(enabled=settings.rotation_y, value=0.0, default=0.0),
            sim_rot_z=Parameter(enabled=settings.rotation_z, value=0.0, default=0.0),
            sim_scale=Parameter(enabled=settings.scale, value=1.0, default=1.0),
            time_shift=Parameter(enabled=settings.time_shift, value=0.0, default=0.0),
            lever_x=Parameter(enabled=settings.leverarm_x, value=0.0, default=0.0),
            lever_y=Parameter(enabled=settings.leverarm_y, value=0.0, default=0.0),
            lever_z=Parameter(enabled=settings.leverarm_z, value=0.0, default=0.0),
        )

    def apply_settings(self, settings: AlignmentEstimationSettings) -> None:
        """Applies estimation settings by enabling or disabling parameters.

        Args:
            settings: Settings specifying which parameters to enable
        """
        self.sim_trans_x.enabled = settings.translation_x and settings.helmert_enabled
        self.sim_trans_y.enabled = settings.translation_y and settings.helmert_enabled
        self.sim_trans_z.enabled = settings.translation_z and settings.helmert_enabled
        self.sim_rot_x.enabled = settings.rotation_x and settings.helmert_enabled
        self.sim_rot_y.enabled = settings.rotation_y and settings.helmert_enabled
        self.sim_rot_z.enabled = settings.rotation_z and settings.helmert_enabled
        self.sim_scale.enabled = settings.scale and settings.helmert_enabled

        self.time_shift.enabled = settings.time_shift_enabled

        self.lever_x.enabled = settings.leverarm_x and settings.leverarm_enabled
        self.lever_y.enabled = settings.leverarm_y and settings.leverarm_enabled
        self.lever_z.enabled = settings.leverarm_z and settings.leverarm_enabled

    @property
    def helmert(self) -> HelmertTransformation:
        """Gets the Helmert similarity transformation parameters.

        Returns:
            HelmertTransformation: Transformation parameters (translation, rotation, scale)
        """
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
        """Gets the leverarm offset parameters.

        Returns:
            Leverarm: Leverarm offset parameters (x, y, z)
        """
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
        """Converts alignment parameters to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with parameter values, enabled flags, and covariance matrix
        """
        variances = pd.DataFrame(self.covariance_matrix)
        parameters = pd.DataFrame({0: self.values, 1: self.enabled_bool_list})
        return pd.concat([parameters, variances], axis=1)

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
        params.enabled_bool_list = [str(item).lower() != "false" for item in alignment_data.iloc[:11, 1].to_list()]

        if alignment_data.shape[1] > 2:
            params.set_covariance_matrix(alignment_data.iloc[:11, 2:].to_numpy())

        return params

    def to_estimation_settings(self) -> AlignmentEstimationSettings:
        """Converts alignment parameters to estimation settings.

        Returns:
            AlignmentEstimationSettings: Settings object with enabled flags from parameters
        """
        return AlignmentEstimationSettings(
            translation_x=self.sim_trans_x.enabled,
            translation_y=self.sim_trans_y.enabled,
            translation_z=self.sim_trans_z.enabled,
            rotation_x=self.sim_rot_x.enabled,
            rotation_y=self.sim_rot_y.enabled,
            rotation_z=self.sim_rot_z.enabled,
            scale=self.sim_scale.enabled,
            time_shift=self.time_shift.enabled,
            leverarm_x=self.lever_x.enabled,
            leverarm_y=self.lever_y.enabled,
            leverarm_z=self.lever_z.enabled,
        )


@dataclass
class SensorRotationParameters(ParameterSet):
    """Parameter set for sensor body-frame rotation.

    Represents the rotational offset between the sensor frame and body frame.

    Attributes:
        sensor_rot_x: Sensor rotation around x-axis
        sensor_rot_y: Sensor rotation around y-axis
        sensor_rot_z: Sensor rotation around z-axis
    """

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
        """Gets the sensor rotation Euler angles.

        Returns:
            np.ndarray: Rotation angles [rx, ry, rz] in radians (XYZ order)
        """
        return np.array([self.sensor_rot_x.value, self.sensor_rot_y.value, self.sensor_rot_z.value])

    @property
    def rotation_set(self) -> Rotations:
        """Gets the sensor rotation as a Rotations object.

        Returns:
            Rotations: Rotation object from Euler angles
        """
        return Rotations.from_euler(seq="xyz", angles=self.rotation)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Gets the 3x3 sensor rotation matrix.

        Returns:
            np.ndarray: Rotation matrix (3x3)
        """
        return Rotations.from_euler(seq="xyz", angles=self.rotation).as_matrix()

    @classmethod
    def from_file(cls, filename: str) -> "SensorRotationParameters":
        """Reads sensor rotation parameters from a file.

        Args:
            filename: Path to the alignment file (must contain 14 lines)

        Returns:
            SensorRotationParameters: Loaded sensor rotation parameters

        Raises:
            ValueError: If file doesn't contain exactly 14 lines
        """
        alignment_data = pd.read_csv(filename, comment="#", header=None)

        if len(alignment_data) != 14:
            raise ValueError("Alignment file with sensor rotation information must contain 14 lines!")
        enabled_bool_list = [str(item).lower() != "false" for item in alignment_data.iloc[11:, 1].to_list()]
        params: SensorRotationParameters = cls()
        params.values = alignment_data.iloc[11:, 0].to_numpy()
        params.enabled_bool_list = enabled_bool_list
        return params

    def to_file(self, filename: str) -> None:
        """Appends sensor rotation parameters to a file.

        Args:
            filename: Path to the file to append to
        """
        with open(filename, mode="a", encoding="utf-8") as file:
            file.write(f"{self.sensor_rot_x.value},{self.sensor_rot_x.enabled}\n")
            file.write(f"{self.sensor_rot_y.value},{self.sensor_rot_y.enabled}\n")
            file.write(f"{self.sensor_rot_z.value},{self.sensor_rot_z.enabled}\n")
