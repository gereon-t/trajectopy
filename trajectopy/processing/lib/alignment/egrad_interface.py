from collections.abc import Callable

import autograd.numpy as np
from autograd import elementwise_grad as egrad

from trajectopy.processing.lib.alignment.data import AlignmentData
from trajectopy.processing.lib.alignment.equations import (
    functional_relationship_x,
    functional_relationship_y,
    functional_relationship_z,
)
from trajectopy.processing.lib.alignment.parameters import AlignmentParameters


class FunctionalRelationship:
    """Class for the functional relationship between the observations and the parameters."""

    def __init__(self) -> None:
        self.x = functional_relationship_x
        self.y = functional_relationship_y
        self.z = functional_relationship_z

        self.dx_dx_from = egrad(functional_relationship_x, 0)
        self.dx_dy_from = egrad(functional_relationship_x, 1)
        self.dx_dz_from = egrad(functional_relationship_x, 2)
        self.dx_dx_to = egrad(functional_relationship_x, 3)
        self.dx_deuler_x = egrad(functional_relationship_x, 4)
        self.dx_deuler_y = egrad(functional_relationship_x, 5)
        self.dx_deuler_z = egrad(functional_relationship_x, 6)
        self.dx_dspeed_x = egrad(functional_relationship_x, 7)
        self.dx_dspeed_y = egrad(functional_relationship_x, 8)
        self.dx_dspeed_z = egrad(functional_relationship_x, 9)

        self.dx_dsim_trans_x = egrad(functional_relationship_x, 10)
        self.dx_dsim_rot_x = egrad(functional_relationship_x, 11)
        self.dx_dsim_rot_y = egrad(functional_relationship_x, 12)
        self.dx_dsim_rot_z = egrad(functional_relationship_x, 13)
        self.dx_dsim_scale = egrad(functional_relationship_x, 14)
        self.dx_dtime_shift = egrad(functional_relationship_x, 15)
        self.dx_dlever_x = egrad(functional_relationship_x, 16)
        self.dx_dlever_y = egrad(functional_relationship_x, 17)
        self.dx_dlever_z = egrad(functional_relationship_x, 18)

        self.dy_dx_from = egrad(functional_relationship_y, 0)
        self.dy_dy_from = egrad(functional_relationship_y, 1)
        self.dy_dz_from = egrad(functional_relationship_y, 2)
        self.dy_dy_to = egrad(functional_relationship_y, 3)
        self.dy_deuler_x = egrad(functional_relationship_y, 4)
        self.dy_deuler_y = egrad(functional_relationship_y, 5)
        self.dy_deuler_z = egrad(functional_relationship_y, 6)
        self.dy_dspeed_x = egrad(functional_relationship_y, 7)
        self.dy_dspeed_y = egrad(functional_relationship_y, 8)
        self.dy_dspeed_z = egrad(functional_relationship_y, 9)

        self.dy_dsim_trans_y = egrad(functional_relationship_y, 10)
        self.dy_dsim_rot_x = egrad(functional_relationship_y, 11)
        self.dy_dsim_rot_y = egrad(functional_relationship_y, 12)
        self.dy_dsim_rot_z = egrad(functional_relationship_y, 13)
        self.dy_dsim_scale = egrad(functional_relationship_y, 14)
        self.dy_dtime_shift = egrad(functional_relationship_y, 15)
        self.dy_dlever_x = egrad(functional_relationship_y, 16)
        self.dy_dlever_y = egrad(functional_relationship_y, 17)
        self.dy_dlever_z = egrad(functional_relationship_y, 18)

        self.dz_dx_from = egrad(functional_relationship_z, 0)
        self.dz_dy_from = egrad(functional_relationship_z, 1)
        self.dz_dz_from = egrad(functional_relationship_z, 2)
        self.dz_dz_to = egrad(functional_relationship_z, 3)
        self.dz_deuler_x = egrad(functional_relationship_z, 4)
        self.dz_deuler_y = egrad(functional_relationship_z, 5)
        self.dz_deuler_z = egrad(functional_relationship_z, 6)
        self.dz_dspeed_x = egrad(functional_relationship_z, 7)
        self.dz_dspeed_y = egrad(functional_relationship_z, 8)
        self.dz_dspeed_z = egrad(functional_relationship_z, 9)

        self.dz_dsim_trans_z = egrad(functional_relationship_z, 10)
        self.dz_dsim_rot_x = egrad(functional_relationship_z, 11)
        self.dz_dsim_rot_y = egrad(functional_relationship_z, 12)
        self.dz_dsim_scale = egrad(functional_relationship_z, 13)
        self.dz_dtime_shift = egrad(functional_relationship_z, 14)
        self.dz_dlever_x = egrad(functional_relationship_z, 15)
        self.dz_dlever_y = egrad(functional_relationship_z, 16)
        self.dz_dlever_z = egrad(functional_relationship_z, 17)

    def eval(
        self,
        func: Callable,
        observations: AlignmentData,
        parameters: AlignmentParameters,
    ):
        """Evaluates the functional relationship for the given observations and parameters."""
        if "functional_relationship_x" in func.__name__:
            return func(
                observations.est_x_from,
                observations.est_y_from,
                observations.est_z_from,
                observations.est_x_to,
                observations.est_euler_x,
                observations.est_euler_y,
                observations.est_euler_z,
                observations.est_speed_x,
                observations.est_speed_y,
                observations.est_speed_z,
                parameters.sim_trans_x.value * np.ones((len(observations.est_x_from),)),
                parameters.sim_rot_x.value * np.ones((len(observations.est_x_from),)),
                parameters.sim_rot_y.value * np.ones((len(observations.est_x_from),)),
                parameters.sim_rot_z.value * np.ones((len(observations.est_x_from),)),
                parameters.sim_scale.value * np.ones((len(observations.est_x_from),)),
                parameters.time_shift.value * np.ones((len(observations.est_x_from),)),
                parameters.lever_x.value * np.ones((len(observations.est_x_from),)),
                parameters.lever_y.value * np.ones((len(observations.est_x_from),)),
                parameters.lever_z.value * np.ones((len(observations.est_x_from),)),
            )

        if "functional_relationship_y" in func.__name__:
            return func(
                observations.est_x_from,
                observations.est_y_from,
                observations.est_z_from,
                observations.est_y_to,
                observations.est_euler_x,
                observations.est_euler_y,
                observations.est_euler_z,
                observations.est_speed_x,
                observations.est_speed_y,
                observations.est_speed_z,
                parameters.sim_trans_y.value * np.ones((len(observations.est_x_from),)),
                parameters.sim_rot_x.value * np.ones((len(observations.est_x_from),)),
                parameters.sim_rot_y.value * np.ones((len(observations.est_x_from),)),
                parameters.sim_rot_z.value * np.ones((len(observations.est_x_from),)),
                parameters.sim_scale.value * np.ones((len(observations.est_x_from),)),
                parameters.time_shift.value * np.ones((len(observations.est_x_from),)),
                parameters.lever_x.value * np.ones((len(observations.est_x_from),)),
                parameters.lever_y.value * np.ones((len(observations.est_x_from),)),
                parameters.lever_z.value * np.ones((len(observations.est_x_from),)),
            )

        if "functional_relationship_z" in func.__name__:
            return func(
                observations.est_x_from,
                observations.est_y_from,
                observations.est_z_from,
                observations.est_z_to,
                observations.est_euler_x,
                observations.est_euler_y,
                observations.est_euler_z,
                observations.est_speed_x,
                observations.est_speed_y,
                observations.est_speed_z,
                parameters.sim_trans_z.value * np.ones((len(observations.est_x_from),)),
                parameters.sim_rot_x.value * np.ones((len(observations.est_x_from),)),
                parameters.sim_rot_y.value * np.ones((len(observations.est_x_from),)),
                parameters.sim_scale.value * np.ones((len(observations.est_x_from),)),
                parameters.time_shift.value * np.ones((len(observations.est_x_from),)),
                parameters.lever_x.value * np.ones((len(observations.est_x_from),)),
                parameters.lever_y.value * np.ones((len(observations.est_x_from),)),
                parameters.lever_z.value * np.ones((len(observations.est_x_from),)),
            )
