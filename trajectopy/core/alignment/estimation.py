"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import Dict

import numpy as np
from numpy import matlib
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats.distributions import chi2

from trajectopy.core.alignment.data import AlignmentData
from trajectopy.core.alignment.direct import (
    align_rotations,
    direct_helmert_transformation,
    direct_leverarm,
    direct_timeshift,
)
from trajectopy.core.alignment.egrad_interface import FunctionalRelationship
from trajectopy.core.alignment.parameters import (
    AlignmentParameters,
    HelmertTransformation,
    Leverarm,
    Parameter,
    SensorRotationParameters,
)
from trajectopy.core.alignment.result import AlignmentResult
from trajectopy.core.alignment.utils import dict2table
from trajectopy.core.definitions import Unit
from trajectopy.core.settings.alignment import AlignmentSettings
from trajectopy.core.settings.matching import MatchingSettings
from trajectopy.core.trajectory import Trajectory

logger = logging.getLogger("root")


def estimate_alignment(
    traj_from: Trajectory,
    traj_to: Trajectory,
    alignment_settings: AlignmentSettings = AlignmentSettings(),
    matching_settings: MatchingSettings = MatchingSettings(),
) -> AlignmentResult:
    """Aligns two trajectories

    Performs a
    - Helmert
    - Leverarm
    - Time shift

    estimation depending on the configuration.
    After this, the estimated parameters are applied
    to the 'traj_from' trajectory.

    Args:
    - traj_from (Trajectory)
    - traj_to (Trajectory)
    - alignment_settings (AlignmentSettings, optional): Settings for the alignment process. Defaults to AlignmentSettings().
    - matching_settings (MatchingSettings, optional): Settings for the matching process. Defaults to MatchingSettings().

    Returns:
        AlignmentResult: Result of the alignment process
    """
    logger.info("Aligning trajectory positions ...")

    alignment_data = AlignmentData(
        traj_from=traj_from,
        traj_to=traj_to,
        alignment_settings=alignment_settings,
        matching_settings=matching_settings,
    )
    ghm_alignment = AlignmentEstimation(alignment_data=alignment_data)
    estimated_parameters = ghm_alignment.estimate_parameters()

    if (
        alignment_data.traj_from.rot is not None
        and alignment_data.traj_to.rot is not None
        and alignment_settings.estimation_settings.sensor_rotation
    ):
        pre_aligned_trajectory = alignment_data.traj_from.apply_transformation(
            estimated_parameters.sim3_matrix, inplace=False
        )
        logger.info("Aligning rotations ...")
        sensor_rot_params = align_rotations(rot_from=pre_aligned_trajectory.rot, rot_to=alignment_data.traj_to.rot)
        print(sensor_rot_params)
    else:
        sensor_rot_params = SensorRotationParameters(enabled=False)

    return AlignmentResult(
        name=f"{alignment_data.traj_from.name} to {alignment_data.traj_to.name}",
        position_parameters=estimated_parameters,
        rotation_parameters=sensor_rot_params,
        estimation_of=ghm_alignment.settings.estimation_settings,
        converged=ghm_alignment.has_results,
    )


class AlignmentEstimation:
    """Class representing the alignment of two trajectories

    This class will align two trajectories using a combination
    of a 3d Helmert-transformation, a leverarm estimation and a
    time-shift estimation.

    It can fully align two trajectories their separation can be
    described by:
    - a translational shift
    - a rotation of the positions
    - a rotation of the orientations (rotation of the n-frame)
    - a scale factor
    - a time shift
    - a leverarm (e.g. mounted at different locations on the platform)
    """

    def __init__(self, alignment_data: AlignmentData) -> None:
        """Constructor

        This method prepares the data and performs an trajectory alignment

        Args:
            alignment_data (AlignmentData): Stores all data required for the alignment
            mode (AlignmentMode, optional): Indicates the desired mode, i.e. whether a
                                            - helmert transformation
                                            - scale estimation
                                            - leverarm estimation
                                            - time shift estimation
                                            should be performed
            error_probability (float, optional): Used for the stochastic global test.
                                                 Defaults to 0.05.
        """
        self.funcrel = FunctionalRelationship()
        self.data = alignment_data

        self._est_params = self.init_parameters()
        self._has_results = False
        self._converged = False
        self._group_redundancies = {}

        logger.info("Initialized Alignment!")
        logger.info(self)

    def __str__(self) -> str:
        return settings_str(self)

    def init_parameters(self) -> AlignmentParameters:
        """This method computes initial parameters
        for the iterative adjustment

        For this, the helmert transformation and
        the leverarm estimation are done separatetly
        using methods that do not require inital
        parameters.

        Returns:
            AlignmentParameters: Hold the estimates parameters.
                                 14 = 7 (helmert+scale) 3 (leverarm) 1 (time) 3 (orientation)
        """
        if self.settings.estimation_settings.helmert_enabled:
            helmert_init = direct_helmert_transformation(xyz_from=self.data.xyz_from, xyz_to=self.data.xyz_to)
            xyz_init = helmert_init.apply_to(self.data.xyz_from)
        else:
            helmert_init = HelmertTransformation()
            xyz_init = self.data.xyz_from

        logger.debug("Initial Helmert: %s \n", str(helmert_init))

        if (
            self.settings.estimation_settings.time_shift_enabled
            and not self.settings.estimation_settings.leverarm_enabled
        ):
            time_shift_init, _ = direct_timeshift(xyz_from=xyz_init, xyz_to=self.data.xyz_to, speed=self.data.speed)
        else:
            time_shift_init = Parameter(value=0.0, name="Time shift", unit=Unit.SECOND)

        logger.debug("Initial time shift: %.3f", time_shift_init.value)

        if self.settings.estimation_settings.leverarm_enabled:
            leverarm_init, time_shift_init, _ = direct_leverarm(
                speed=self.data.speed if self.settings.estimation_settings.time_shift_enabled else None,
                xyz_from=xyz_init,
                xyz_to=self.data.xyz_to,
                rpy_body=self.data.rpy_from,
            )
        else:
            leverarm_init = Leverarm()

        logger.debug("Initial leverarm: %s \n", str(leverarm_init))

        alignparams = AlignmentParameters(
            sim_trans_x=helmert_init.trans_x,
            sim_trans_y=helmert_init.trans_y,
            sim_trans_z=helmert_init.trans_z,
            sim_rot_x=helmert_init.rot_x,
            sim_rot_y=helmert_init.rot_y,
            sim_rot_z=helmert_init.rot_z,
            sim_scale=helmert_init.scale,
            time_shift=time_shift_init,
            lever_x=leverarm_init.x,
            lever_y=leverarm_init.y,
            lever_z=leverarm_init.z,
        )

        alignparams.apply_settings(self.settings.estimation_settings)
        logger.debug("Applied settings: %s \n", str(self.settings.estimation_settings))
        return alignparams

    def estimate_parameters(self) -> AlignmentParameters:
        """Handles the estimation of the parameters"""

        logger.info("Performing alignment...")
        if self.settings.estimation_settings.all_lq_disabled:
            logger.warning("Nothing to estimate since all parameters are disabled")
            return AlignmentParameters()

        cnt = 0
        max_recomputations = 5
        var_fac_diff = float("inf")
        var_fac_tol = 1e-3

        while var_fac_diff > var_fac_tol and cnt < max_recomputations:
            self._estimate_parameters()
            self._global_test(variance_factor=self.variance_factor, redundancy=self.redundancy)

            if not self.data.alignment_settings.stochastics.variance_estimation:
                break

            var_fac_diff = abs(self.variance_factor - 1)

            logger.info("Adjusting variance vector by factor %.3f", self.variance_factor)
            self.data._var_vector *= self.variance_factor

            if var_fac_diff > var_fac_tol:
                logger.info("Variance component is different from 1, re-estimation required.")
            else:
                logger.info("Finished with variance estimation. Re-estimation not required.")

            cnt += 1

        if not self._converged:
            logger.info("Adjustment did not converge. Returning initial parameters.")
            return self.init_parameters()

        self._has_results = True
        print_summary(self)

        return self._est_params

    @property
    def settings(self) -> AlignmentSettings:
        return self.data.alignment_settings

    @property
    def has_results(self) -> bool:
        return self._has_results

    @property
    def est_params(self) -> AlignmentParameters:
        return self._est_params

    @property
    def group_redundancies(self) -> Dict[str, bool]:
        return self._group_redundancies

    @property
    def num_of_equations(self) -> int:
        return self.data.number_of_epochs * 3

    @property
    def redundancy(self) -> int:
        return self.num_of_equations - self._est_params.num_enabled

    @property
    def variance_factor(self) -> float:
        return (
            self.data.res_vector.T @ spsolve(csc_matrix(self.data.sigma_ll), self.data.res_vector)
        ) / self.redundancy

    def _estimate_parameters(self) -> None:
        """Helmert-Leverarm-Time Transformation using the Gauß-Helmert-Model

        The observation-equations are sorted in the following way:
        [X, Y, Z, X, Y, Z, ..., X, Y, Z]
        """
        # obs = [x_from, y_from, z_from, x_to, y_to, z_to, roll_body, pitch_body, yaw_body]

        # preparation for iterative adjustment
        delta_params = np.ones((len(self._est_params),)) * np.Inf
        self.data.res_vector = np.zeros_like(self.data.obs_vector)

        contradiction_w = self._eval_functional_relationship()

        it_counter = 0
        max_iterations = 15
        self._converged = True
        while any(abs(value) > threshold for value, threshold in zip(delta_params, self.data.thresholds)):
            if it_counter > max_iterations:
                logger.error(
                    "Adjustment did not converge after %i iterations. Maximum parameter update: %.3e",
                    it_counter,
                    np.max(np.abs(delta_params)),
                )
                self._converged = False
                break

            a_design = self._get_design_matrix()

            # filter design matrix
            a_design = a_design[:, self.settings.estimation_settings.lq_parameter_filter]
            b_cond = self._get_condition_matrix()

            bbt = b_cond @ self.data.sigma_ll @ b_cond.T

            # solve normal equations
            delta_params = self._compute_parameter_deltas(contradiction_w, a_design, bbt)
            correlates_k = -spsolve(bbt, a_design @ delta_params + contradiction_w)
            self.data.res_vector = self.data.sigma_ll @ b_cond.T @ correlates_k

            # update
            self._est_params.values_enabled += delta_params
            contradiction_w = self._eval_functional_relationship() - b_cond @ self.data.res_vector.ravel()
            it_counter += 1

        if self._converged:
            logger.info("Adjustment did converge after %i iterations", it_counter)

        self._compute_parameter_variances(a_design, bbt)

    def _compute_parameter_variances(self, a_design: csc_matrix, bbt: csc_matrix) -> None:
        sigma_xx_inv: csc_matrix = a_design.T @ spsolve(bbt, a_design)
        if sigma_xx_inv.size == 1:
            self._est_params.set_covariance_matrix(np.reciprocal(sigma_xx_inv[:, None]))
        else:
            self._est_params.set_covariance_matrix(np.linalg.pinv(sigma_xx_inv.toarray()))

    def _compute_parameter_deltas(
        self, contradiction_w: np.ndarray, a_design: csc_matrix, bbt: csc_matrix
    ) -> np.ndarray:
        if a_design.shape[1] == 1:
            return -(a_design.T @ spsolve(bbt, contradiction_w)) / (a_design.T @ spsolve(bbt, a_design))

        # quasi vermittelnd
        # spsolve(-a_design.T @ spsolve(bbt, -a_design), -a_design.T @ spsolve(bbt, contradiction_w))
        return -spsolve(
            a_design.T @ spsolve(bbt, a_design),
            a_design.T @ spsolve(bbt, contradiction_w),
        )

    def _global_test(self, variance_factor: float, redundancy: int, description: str = "global") -> bool:
        tau = variance_factor * redundancy
        quantile = chi2.ppf(1 - self.settings.stochastics.error_probability, redundancy)

        logger.info(
            "Stochastic test passed (%s): %s, quantile: %.3f, test value: %.3f, variance factor: %.3f, redundancy: %i",
            description,
            str(tau <= quantile),
            quantile,
            tau,
            variance_factor,
            redundancy,
        )
        return tau <= quantile

    def _get_design_matrix(self) -> csc_matrix:
        a_design = np.zeros((self.data.number_of_epochs * 3, 11))
        a_design[0::3, :] = self._get_design_x()
        a_design[1::3, :] = self._get_design_y()
        a_design[2::3, :] = self._get_design_z()
        return csc_matrix(a_design)

    def _get_design_z(self) -> np.ndarray:
        return np.c_[
            np.zeros((self.data.number_of_epochs, 1)),
            np.zeros((self.data.number_of_epochs, 1)),
            self.funcrel.eval(
                func=self.funcrel.dz_dsim_trans_z,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dz_dsim_rot_x,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dz_dsim_rot_y,
                parameters=self.est_params,
                observations=self.data,
            ),
            np.zeros((self.data.number_of_epochs, 1)),
            self.funcrel.eval(
                func=self.funcrel.dz_dsim_scale,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dz_dtime_shift,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dz_dlever_x,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dz_dlever_y,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dz_dlever_z,
                parameters=self.est_params,
                observations=self.data,
            ),
        ]

    def _get_design_y(self) -> np.ndarray:
        return np.c_[
            np.zeros((self.data.number_of_epochs, 1)),
            self.funcrel.eval(
                func=self.funcrel.dy_dsim_trans_y,
                parameters=self.est_params,
                observations=self.data,
            ),
            np.zeros((self.data.number_of_epochs, 1)),
            self.funcrel.eval(
                func=self.funcrel.dy_dsim_rot_x,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dy_dsim_rot_y,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dy_dsim_rot_z,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dy_dsim_scale,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dy_dtime_shift,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dy_dlever_x,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dy_dlever_y,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dy_dlever_z,
                parameters=self.est_params,
                observations=self.data,
            ),
        ]

    def _get_design_x(self) -> np.ndarray:
        return np.c_[
            self.funcrel.eval(
                func=self.funcrel.dx_dsim_trans_x,
                parameters=self.est_params,
                observations=self.data,
            ),
            np.zeros((self.data.number_of_epochs, 1)),
            np.zeros((self.data.number_of_epochs, 1)),
            self.funcrel.eval(
                func=self.funcrel.dx_dsim_rot_x,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dx_dsim_rot_y,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dx_dsim_rot_z,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dx_dsim_scale,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dx_dtime_shift,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dx_dlever_x,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dx_dlever_y,
                parameters=self.est_params,
                observations=self.data,
            ),
            self.funcrel.eval(
                func=self.funcrel.dx_dlever_z,
                parameters=self.est_params,
                observations=self.data,
            ),
        ]

    def _eval_functional_relationship(self) -> np.ndarray:
        # accounting for the time shift not by using the velocity model but by shifting the time stamps and re-interpolating
        func_xyz = np.zeros((self.data.number_of_epochs * 3,))
        func_xyz[::3] = self.funcrel.eval(func=self.funcrel.x, parameters=self.est_params, observations=self.data)
        func_xyz[1::3] = self.funcrel.eval(func=self.funcrel.y, parameters=self.est_params, observations=self.data)
        func_xyz[2::3] = self.funcrel.eval(func=self.funcrel.z, parameters=self.est_params, observations=self.data)
        return func_xyz

    def _get_condition_matrix(self) -> csc_matrix:
        """Computes the condition-matrix for the Gauß-Helmert-Model

        The matrix contains the derivatives of the
        observation equations with respect to the observations.

        Depending on whether the lever arm is to be estimated,
        additional columns are added to the condition matrix
        corresponding to the derivation of the functional
        relationship with respect to the platform orientations.

        Its dimensions are:
            [#Obs.-Equations x #Observations]

            #Obs.-Equations: 3 * #Points

        This matrix is sparse.

        Args:
            parameters (AlignmentParameters): (current) estimated parameters

        Returns:
            csc_matrix: sparse condition matrix
        """
        cond_xyz = self._get_condition_stack()

        # row indices
        # [0,0,0,0,0,0; 1,1,1,1,1,1; 2,2,2,2,2,2; 3,3,3,3,3,3; ...]
        row_idx = np.repeat(np.arange(0, self.num_of_equations, 1), self.data.num_obs_per_epoch)

        # column indices [0,1,2,3,4,5; 6,7,8,9,10,11; 12,13,14,15,16,17; ...]
        col_idx_matrix = (
            matlib.repmat(
                np.arange(0, self.data.num_obs_per_epoch),
                self.num_of_equations,
                1,
            )
            + np.repeat(
                np.arange(
                    0,
                    self.data.number_of_epochs * self.data.num_obs_per_epoch,
                    self.data.num_obs_per_epoch,
                ),
                3,
            )[:, None]
        )
        col_idx = np.reshape(col_idx_matrix, (col_idx_matrix.size,))

        return csc_matrix((np.reshape(cond_xyz, (cond_xyz.size,)), (row_idx, col_idx)))

    def _get_condition_stack(self) -> np.ndarray:
        """Helper function to get the non-zero data of the condition matrix

        Depending on which parameters are estimated, this function returns
        different data.

        Args:
            parameters (AlignmentParameters): (current) estimated parameters

        Returns:
            np.ndarray: condition matrix data
        """
        xyz_from_component = self._get_condition_xyz_from()

        rpy_body_component = (
            self._get_condition_rpy_body() if self.settings.estimation_settings.leverarm_enabled else None
        )

        speed_to_component = (
            self._get_condition_speed_to() if self.settings.estimation_settings.time_shift_enabled else None
        )

        if (
            self.settings.estimation_settings.leverarm_enabled
            and not self.settings.estimation_settings.time_shift_enabled
            and rpy_body_component is not None
        ):
            return np.column_stack(
                [
                    np.c_[
                        xyz_from_component[i],
                        self._condition_xyz_to[i],
                        rpy_body_component[i],
                    ]
                    for i in range(3)
                ]
            )

        if (
            self.settings.estimation_settings.time_shift_enabled
            and not self.settings.estimation_settings.leverarm_enabled
            and speed_to_component is not None
        ):
            return np.column_stack(
                [
                    np.c_[
                        xyz_from_component[i],
                        self._condition_xyz_to[i],
                        speed_to_component[i],
                    ]
                    for i in range(3)
                ]
            )

        if (
            self.settings.estimation_settings.leverarm_enabled
            and self.settings.estimation_settings.time_shift_enabled
            and rpy_body_component is not None
            and speed_to_component is not None
        ):
            return np.column_stack(
                [
                    np.c_[
                        xyz_from_component[i],
                        self._condition_xyz_to[i],
                        rpy_body_component[i],
                        speed_to_component[i],
                    ]
                    for i in range(3)
                ]
            )

        return np.column_stack(
            [
                np.c_[
                    xyz_from_component[i],
                    self._condition_xyz_to[i],
                ]
                for i in range(3)
            ]
        )

    def _get_condition_rpy_body(self) -> list:
        return [
            np.c_[
                self.funcrel.eval(
                    func=self.funcrel.dx_deuler_x,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dx_deuler_y,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dx_deuler_z,
                    parameters=self.est_params,
                    observations=self.data,
                ),
            ],
            np.c_[
                self.funcrel.eval(
                    func=self.funcrel.dy_deuler_x,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dy_deuler_y,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dy_deuler_z,
                    parameters=self.est_params,
                    observations=self.data,
                ),
            ],
            np.c_[
                self.funcrel.eval(
                    func=self.funcrel.dz_deuler_x,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dz_deuler_y,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dz_deuler_z,
                    parameters=self.est_params,
                    observations=self.data,
                ),
            ],
        ]

    def _get_condition_xyz_from(self) -> list:
        return [
            np.c_[
                self.funcrel.eval(
                    func=self.funcrel.dx_dx_from,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dx_dy_from,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dx_dz_from,
                    parameters=self.est_params,
                    observations=self.data,
                ),
            ],
            np.c_[
                self.funcrel.eval(
                    func=self.funcrel.dy_dx_from,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dy_dy_from,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dy_dz_from,
                    parameters=self.est_params,
                    observations=self.data,
                ),
            ],
            np.c_[
                self.funcrel.eval(
                    func=self.funcrel.dz_dx_from,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dz_dy_from,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dz_dz_from,
                    parameters=self.est_params,
                    observations=self.data,
                ),
            ],
        ]

    @property
    def _condition_xyz_to(self) -> list:
        """
        Helper function returning the constant xyz_to component of
        the condition matrix
        """
        return [
            np.c_[
                -np.ones((self.data.number_of_epochs, 1)),
                np.zeros((self.data.number_of_epochs, 1)),
                np.zeros((self.data.number_of_epochs, 1)),
            ],
            np.c_[
                np.zeros((self.data.number_of_epochs, 1)),
                -np.ones((self.data.number_of_epochs, 1)),
                np.zeros((self.data.number_of_epochs, 1)),
            ],
            np.c_[
                np.zeros((self.data.number_of_epochs, 1)),
                np.zeros((self.data.number_of_epochs, 1)),
                -np.ones((self.data.number_of_epochs, 1)),
            ],
        ]

    def _get_condition_speed_to(self) -> list:
        return [
            np.c_[
                self.funcrel.eval(
                    func=self.funcrel.dx_dspeed_x,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dx_dspeed_y,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dx_dspeed_z,
                    parameters=self.est_params,
                    observations=self.data,
                ),
            ],
            np.c_[
                self.funcrel.eval(
                    func=self.funcrel.dy_dspeed_x,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dy_dspeed_y,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dy_dspeed_z,
                    parameters=self.est_params,
                    observations=self.data,
                ),
            ],
            np.c_[
                self.funcrel.eval(
                    func=self.funcrel.dz_dspeed_x,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dz_dspeed_y,
                    parameters=self.est_params,
                    observations=self.data,
                ),
                self.funcrel.eval(
                    func=self.funcrel.dz_dspeed_z,
                    parameters=self.est_params,
                    observations=self.data,
                ),
            ],
        ]


def print_summary(alignment: AlignmentEstimation) -> None:
    logger.info(dict2table(alignment.data.group_stds, title="Group Standard Deviations"))
    logger.info(alignment.est_params)


def settings_str(alignment: AlignmentEstimation) -> str:
    return (
        f"\n _____________________________________________________________________\n"
        f"| ---------------------------- Alignment ---------------------------- |\n"
        f"| Estimation of:           {alignment.settings.estimation_settings.short_mode_str:<42} |\n"
        f"| Error probability [%]:   {alignment.settings.stochastics.error_probability*100:<42} |\n"
        f"|_____________________________________________________________________|\n"
    )
