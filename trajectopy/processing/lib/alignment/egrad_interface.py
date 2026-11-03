import autograd.numpy as np
from autograd import elementwise_grad as egrad

from trajectopy.core.parameters import AlignmentParameters
from trajectopy.processing.lib.alignment.data import AlignmentData
from trajectopy.processing.lib.alignment.equations import (
    functional_relationship_x,
    functional_relationship_y,
    functional_relationship_z,
)

# Arguments of functional_relationship_x and _y; "to" is the target coordinate (x_to or y_to)
# and "sim_trans" the translation along the same axis.
ARGUMENTS_XY = (
    "x_from",
    "y_from",
    "z_from",
    "to",
    "euler_x",
    "euler_y",
    "euler_z",
    "speed_x",
    "speed_y",
    "speed_z",
    "sim_trans",
    "sim_rot_x",
    "sim_rot_y",
    "sim_rot_z",
    "sim_scale",
    "time_shift",
    "lever_x",
    "lever_y",
    "lever_z",
)
# functional_relationship_z does not depend on the rotation about the z-axis
ARGUMENTS_Z = tuple(argument for argument in ARGUMENTS_XY if argument != "sim_rot_z")


class FunctionalRelationship:
    """Class for the functional relationship between the observations and the parameters."""

    def __init__(self) -> None:
        self._functions = (functional_relationship_x, functional_relationship_y, functional_relationship_z)
        self._arguments = (ARGUMENTS_XY, ARGUMENTS_XY, ARGUMENTS_Z)
        # the derivative with respect to the target coordinate is the constant -1 (see the condition matrix)
        self._differentiated = tuple(
            tuple(argument for argument in arguments if argument != "to") for arguments in self._arguments
        )
        # all partial derivatives of one coordinate from a single trace of its function
        self._partials = tuple(
            egrad(function, tuple(arguments.index(argument) for argument in differentiated))
            for function, arguments, differentiated in zip(self._functions, self._arguments, self._differentiated)
        )

    def eval(
        self, observations: AlignmentData, parameters: AlignmentParameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates the functional relationship of the x, y and z coordinates."""
        return tuple(  # type: ignore[return-value]
            function(*self._values(axis, observations, parameters)) for axis, function in enumerate(self._functions)
        )

    def partials(self, observations: AlignmentData, parameters: AlignmentParameters) -> list[dict[str, np.ndarray]]:
        """Partial derivatives of the x, y and z functional relationship with respect to all arguments
        except the target coordinate, by argument name."""
        return [
            dict(zip(self._differentiated[axis], partials(*self._values(axis, observations, parameters))))
            for axis, partials in enumerate(self._partials)
        ]

    def _values(self, axis: int, observations: AlignmentData, parameters: AlignmentParameters) -> list[np.ndarray]:
        """Argument values of the functional relationship of the given axis (0: x, 1: y, 2: z)."""
        ones = np.ones((len(observations.est_x_from),))
        values = {
            "x_from": observations.est_x_from,
            "y_from": observations.est_y_from,
            "z_from": observations.est_z_from,
            "to": (observations.est_x_to, observations.est_y_to, observations.est_z_to)[axis],
            "euler_x": observations.est_euler_x,
            "euler_y": observations.est_euler_y,
            "euler_z": observations.est_euler_z,
            "speed_x": observations.est_speed_x,
            "speed_y": observations.est_speed_y,
            "speed_z": observations.est_speed_z,
            "sim_trans": (parameters.sim_trans_x, parameters.sim_trans_y, parameters.sim_trans_z)[axis].value * ones,
            "sim_rot_x": parameters.sim_rot_x.value * ones,
            "sim_rot_y": parameters.sim_rot_y.value * ones,
            "sim_rot_z": parameters.sim_rot_z.value * ones,
            "sim_scale": parameters.sim_scale.value * ones,
            "time_shift": parameters.time_shift.value * ones,
            "lever_x": parameters.lever_x.value * ones,
            "lever_y": parameters.lever_y.value * ones,
            "lever_z": parameters.lever_z.value * ones,
        }
        return [values[argument] for argument in self._arguments[axis]]
