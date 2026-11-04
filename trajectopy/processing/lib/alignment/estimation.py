import copy
import logging
import warnings

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import MatrixRankWarning, factorized, spsolve
from scipy.stats.distributions import chi2

from trajectopy.core.parameters import (
    AlignmentParameters,
    HelmertTransformation,
    Leverarm,
    Parameter,
)
from trajectopy.core.settings import AlignmentSettings
from trajectopy.processing.lib.alignment.data import AlignmentData
from trajectopy.processing.lib.alignment.direct import (
    direct_helmert_transformation,
    direct_leverarm,
    direct_timeshift,
)
from trajectopy.processing.lib.alignment.egrad_interface import FunctionalRelationship
from trajectopy.utils.definitions import UNIT_FORMAT_RULES, Unit
from trajectopy.utils.formatting import dict2table

logger = logging.getLogger(__name__)

MAX_ROBUST_ITERATIONS = 10
# Parameters in the order of the columns of the design matrix (see AlignmentParameters)
DESIGN_COLUMNS = (
    "sim_trans_x",
    "sim_trans_y",
    "sim_trans_z",
    "sim_rot_x",
    "sim_rot_y",
    "sim_rot_z",
    "sim_scale",
    "time_shift",
    "lever_x",
    "lever_y",
    "lever_z",
)
NUM_LOGGED_OUTLIERS = 5  # epochs with the largest residuals in the debug log of the robust estimation


class AlignmentEstimator:
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

    def __init__(self, alignment_data: AlignmentData, initial_parameters: AlignmentParameters | None = None) -> None:
        """Constructor

        This class holds all data required for the alignment
        of two trajectories as well as the methods to perform
        the alignment.

        Args:
            alignment_data (AlignmentData): Stores all data required for the alignment
            initial_parameters (AlignmentParameters | None): Optional prior (e.g. from a manual alignment).
                If given, it replaces the direct initialization. Parameters that are not estimated are
                kept fixed at their prior value.
        """
        self._initial_parameters = initial_parameters
        self.funcrel = FunctionalRelationship()
        self.data = alignment_data

        self._est_params = self.init_parameters()
        self._has_results = False
        self._converged = False

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
        if self._initial_parameters is not None:
            return self._init_from_prior(self._initial_parameters)

        if self.settings.estimation_settings.helmert_enabled:
            try:
                helmert_init = direct_helmert_transformation(xyz_from=self.data.xyz_from, xyz_to=self.data.xyz_to)
            except ValueError as error:
                # e.g. a single matched pose: the translation is still determined, rotation and scale are not
                logger.warning("No direct Helmert transformation (%s), starting from the translation only.", error)
                helmert_init = HelmertTransformation()
                helmert_init.translation = np.mean(self.data.xyz_to - self.data.xyz_from, axis=0)
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

    def _init_from_prior(self, prior: AlignmentParameters) -> AlignmentParameters:
        """Creates start parameters from a prior; non-estimated parameters stay fixed at their prior value."""
        params = copy.deepcopy(prior)
        for parameter in params:
            prior_value = parameter.value
            parameter.value = prior_value
            parameter.default = prior_value
            parameter.variance = 0.0

        params.apply_settings(self.settings.estimation_settings)
        logger.info("Using manual prior as initial parameters:\n%s", params)
        return params

    def estimate_parameters(self) -> AlignmentParameters:
        """Handles the estimation of the parameters"""

        logger.info("Performing alignment...")
        if self.settings.estimation_settings.all_lq_disabled:
            logger.warning("Nothing to estimate since all parameters are disabled")
            return self.init_parameters() if self._initial_parameters is not None else AlignmentParameters()

        cnt = 0
        max_recomputations = 5
        var_fac_diff = float("inf")
        var_fac_tol = 1e-3

        while var_fac_diff > var_fac_tol and cnt < max_recomputations:
            self._estimate_robustly()
            if not self._converged:
                break  # there are no residuals to test (and to estimate the variances from)
            if self.redundancy <= 0:
                logger.warning("No redundancy: the variance factor cannot be estimated and tested.")
                break
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
        parameter_filter = self.settings.estimation_settings.lq_parameter_filter
        # the updates only contain the estimated parameters, so only their thresholds apply
        thresholds = self.data.thresholds[parameter_filter]
        delta_params = np.full(len(thresholds), np.inf)
        self.data.res_vector = np.zeros_like(self.data.obs_vector)

        contradiction_w = self._eval_functional_relationship()

        it_counter = 0
        max_iterations = 15
        self._converged = True
        # "not all within the thresholds": NaN updates (a singular or diverged adjustment) must not count as converged
        while not np.all(np.abs(delta_params) <= thresholds):
            # the initial updates are infinite by definition, later ones have to be finite
            if it_counter > 0 and not np.all(np.isfinite(delta_params)):
                logger.error("Adjustment failed: the parameter updates are not finite (singular normal equations?).")
                self._converged = False
                break
            if it_counter > max_iterations:
                logger.error(
                    "Adjustment did not converge after %i iterations. Maximum parameter update: %.3e",
                    it_counter,
                    np.max(np.abs(delta_params)),
                )
                self._converged = False
                break

            partials = self.funcrel.partials(observations=self.data, parameters=self.est_params)
            a_design = self._get_design_matrix(partials)

            # filter design matrix
            a_design = a_design[:, parameter_filter]
            b_cond = self._get_condition_matrix(partials)

            try:
                with warnings.catch_warnings():
                    # a singular normal matrix (e.g. parameters that the data does not determine) is a failure
                    warnings.simplefilter("error", MatrixRankWarning)
                    # factorized once per iteration and used for all solves with it
                    bbt = _FactorizedMatrix(b_cond @ self.data.sigma_ll @ b_cond.T)
                    normal_matrix = a_design.T @ bbt.solve_sparse(a_design)

                    # solve normal equations
                    delta_params = self._compute_parameter_deltas(contradiction_w, a_design, normal_matrix, bbt)
                    correlates_k = -bbt.solve(a_design @ delta_params + contradiction_w)
            except (RuntimeError, np.linalg.LinAlgError, MatrixRankWarning) as error:
                # e.g. too few or degenerate matched poses for the enabled parameters
                logger.error("Adjustment failed: the equations are singular (%s).", error)
                self._converged = False
                return
            self.data.res_vector = self.data.sigma_ll @ b_cond.T @ correlates_k

            # update
            self._est_params.values_enabled += delta_params
            contradiction_w = self._eval_functional_relationship() - b_cond @ self.data.res_vector.ravel()
            it_counter += 1

        if self._converged:
            logger.info("Adjustment did converge after %i iterations", it_counter)

        self._compute_parameter_variances(normal_matrix)

    def _estimate_robustly(self) -> None:
        """Iteratively reweighted least squares: epochs with large residuals are down-weighted until the
        parameters do not change anymore."""
        self._estimate_parameters()
        if not self.settings.stochastics.robust_estimation:
            return
        if not self._converged:
            logger.warning("Robust estimation skipped: the adjustment did not converge.")
            return

        k = np.sqrt(chi2.ppf(1 - self.settings.stochastics.error_probability, 3))
        enabled_parameters = [parameter for parameter in self._est_params if parameter.enabled]
        unweighted_values = np.array(self._est_params.values_enabled, dtype=float)
        thresholds = self.data.thresholds[self.settings.estimation_settings.lq_parameter_filter]
        normalized = np.zeros(0)
        for iteration in range(1, MAX_ROBUST_ITERATIONS + 1):
            current = self._update_epoch_weights(k)
            if current is None:
                logger.debug("Robust estimation skipped: all residuals are zero.")
                return
            normalized = current
            previous = np.array(self._est_params.values_enabled, dtype=float)
            self._estimate_parameters()
            if not self._converged:
                logger.warning(
                    "Robust estimation stopped: the adjustment in iteration %i did not converge.", iteration
                )
                return

            change_ratio = np.abs(np.array(self._est_params.values_enabled, dtype=float) - previous) / thresholds
            logger.debug(
                "Robust iteration %i: %i epochs down-weighted, largest normalized residual %.1f (threshold %.2f), "
                "largest parameter update %.2f x convergence threshold",
                iteration,
                int(np.sum(self.data.epoch_weights < 1.0)),
                float(np.max(normalized)),
                k,
                float(np.max(change_ratio)),
            )
            if np.all(change_ratio <= 1.0):
                break
        else:
            logger.warning("Robust estimation did not converge after %i iterations.", MAX_ROBUST_ITERATIONS)

        weights = self.data.epoch_weights
        logger.info(
            "Robust estimation finished after %i iterations: %i of %i epochs (%.1f %%) down-weighted, "
            "%i of them to less than 10 %%.",
            iteration,
            int(np.sum(weights < 1.0)),
            len(weights),
            100 * np.mean(weights < 1.0),
            int(np.sum(weights < 0.1)),
        )
        largest = np.argsort(normalized)[::-1][: min(NUM_LOGGED_OUTLIERS, int(np.sum(weights < 1.0)))]
        logger.debug(
            "Largest normalized residuals: %s",
            ", ".join(f"{normalized[i]:.1f} at t = {self.data.tstamps[i]:.3f} s" for i in largest),
        )
        logger.debug(
            "Parameter changes by the robust weighting: %s",
            ", ".join(
                f"{parameter.name} {_format_change(parameter, value - unweighted)}"
                for parameter, value, unweighted in zip(
                    enabled_parameters, np.array(self._est_params.values_enabled, dtype=float), unweighted_values
                )
            ),
        )

    def _update_epoch_weights(self, k: float) -> np.ndarray | None:
        """Huber weights from the standardized residuals of each epoch.

        The sum of the squared standardized residuals of an epoch is approximately chi-square distributed
        with 3 degrees of freedom (one condition per coordinate). Its scale is estimated robustly from the
        median, so that the weights do not depend on the a-priori standard deviations.

        Args:
            k (float): Normalized residual above which epochs are down-weighted.

        Returns:
            np.ndarray | None: Normalized residual of each epoch, None if the residuals are all zero, so that
                no weights can be derived.
        """
        standardized = self.data.res_vector**2 / self.data.var_vector
        squared_norms = standardized.reshape(-1, self.data.num_obs_per_epoch).sum(axis=1)
        variance_factor = np.median(squared_norms) / chi2.ppf(0.5, 3)
        if not np.isfinite(variance_factor) or variance_factor <= 0:
            return None

        normalized = np.sqrt(squared_norms / variance_factor)
        self.data.epoch_weights = np.minimum(1.0, k / np.maximum(normalized, np.finfo(float).tiny))
        return normalized

    def _compute_parameter_variances(self, normal_matrix: csc_matrix) -> None:
        """Sets the parameter covariances from the normal matrix of the last iteration."""
        sigma_xx_inv = normal_matrix
        if sigma_xx_inv.size == 1:
            self._est_params.set_covariance_matrix(np.reciprocal(sigma_xx_inv[:, None]))
        else:
            self._est_params.set_covariance_matrix(np.linalg.pinv(sigma_xx_inv.toarray()))

    def _compute_parameter_deltas(
        self,
        contradiction_w: np.ndarray,
        a_design: csc_matrix,
        normal_matrix: csc_matrix,
        bbt: "_FactorizedMatrix",
    ) -> np.ndarray:
        if a_design.shape[1] == 1:
            return -(a_design.T @ bbt.solve(contradiction_w)) / normal_matrix

        # quasi vermittelnd
        # spsolve(-a_design.T @ spsolve(bbt, -a_design), -a_design.T @ spsolve(bbt, contradiction_w))
        return -spsolve(normal_matrix, a_design.T @ bbt.solve(contradiction_w))

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

    def _get_design_matrix(self, partials: list[dict[str, np.ndarray]]) -> csc_matrix:
        """Derivatives of the observation equations with respect to all 11 parameters.

        Args:
            partials (list[dict[str, np.ndarray]]): Partial derivatives of the x, y and z equations,
                see FunctionalRelationship.partials.
        """
        a_design = np.zeros((self.data.number_of_epochs * 3, len(DESIGN_COLUMNS)))
        for axis, axis_partials in enumerate(partials):
            # the translation of an equation is the one along its own axis, the others do not occur
            translation = f"sim_trans_{'xyz'[axis]}"
            for column, parameter in enumerate(DESIGN_COLUMNS):
                name = "sim_trans" if parameter == translation else parameter
                if name in axis_partials:
                    a_design[axis::3, column] = axis_partials[name]
        return csc_matrix(a_design)

    def _eval_functional_relationship(self) -> np.ndarray:
        # accounting for the time shift not by using the velocity model but by shifting the time stamps and re-interpolating
        func_xyz = np.zeros((self.data.number_of_epochs * 3,))
        func_x, func_y, func_z = self.funcrel.eval(observations=self.data, parameters=self.est_params)
        func_xyz[::3] = func_x
        func_xyz[1::3] = func_y
        func_xyz[2::3] = func_z
        return func_xyz

    def _get_condition_matrix(self, partials: list[dict[str, np.ndarray]]) -> csc_matrix:
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
            partials (list[dict[str, np.ndarray]]): Partial derivatives of the x, y and z equations,
                see FunctionalRelationship.partials.

        Returns:
            csc_matrix: sparse condition matrix
        """
        cond_xyz = self._get_condition_stack(partials)

        # row indices
        # [0,0,0,0,0,0; 1,1,1,1,1,1; 2,2,2,2,2,2; 3,3,3,3,3,3; ...]
        row_idx = np.repeat(np.arange(0, self.num_of_equations, 1), self.data.num_obs_per_epoch)

        # column indices [0,1,2,3,4,5; 6,7,8,9,10,11; 12,13,14,15,16,17; ...]
        col_idx_matrix = (
            np.tile(
                np.arange(0, self.data.num_obs_per_epoch),
                (self.num_of_equations, 1),
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

    def _get_condition_stack(self, partials: list[dict[str, np.ndarray]]) -> np.ndarray:
        """Helper function to get the non-zero data of the condition matrix

        For each epoch, the derivatives of the x, y and z equations with respect to the observations of the
        epoch follow each other: source positions, target positions and, depending on which parameters are
        estimated, platform orientations and speeds.

        Args:
            partials (list[dict[str, np.ndarray]]): Partial derivatives of the x, y and z equations,
                see FunctionalRelationship.partials.

        Returns:
            np.ndarray: condition matrix data
        """
        observations = ["x_from", "y_from", "z_from"]
        if self.settings.estimation_settings.leverarm_enabled:
            observations_after_target = ["euler_x", "euler_y", "euler_z"]
        else:
            observations_after_target = []
        if self.settings.estimation_settings.time_shift_enabled:
            observations_after_target += ["speed_x", "speed_y", "speed_z"]

        blocks = []
        for axis, axis_partials in enumerate(partials):
            # the derivative with respect to the own target coordinate is -1, the others do not occur
            target = [np.zeros(self.data.number_of_epochs) for _ in range(3)]
            target[axis] = -np.ones(self.data.number_of_epochs)
            blocks.append(
                np.column_stack(
                    [axis_partials[name] for name in observations]
                    + target
                    + [axis_partials[name] for name in observations_after_target]
                )
            )
        return np.column_stack(blocks)


class _FactorizedMatrix:
    """Sparse matrix with its LU factorization, so that several systems can be solved with one factorization.

    The results are the same as those of scipy's spsolve, which factorizes the matrix for every call.
    """

    def __init__(self, matrix: csc_matrix) -> None:
        self._matrix = matrix
        self._solve = factorized(matrix)

    def solve(self, b: np.ndarray) -> np.ndarray:
        return self._solve(b)

    def solve_sparse(self, b: csc_matrix) -> csc_matrix:
        """Solves for all columns of a sparse right-hand side; the solution is sparse as well (like spsolve).

        Like spsolve, a single column is treated as a vector and its solution is dense.
        """
        if b.shape[1] == 1:
            return self._solve(b.toarray().ravel())
        solutions = [self._solve(b[:, [column]].toarray().ravel()) for column in range(b.shape[1])]
        rows = [np.flatnonzero(solution) for solution in solutions]
        data = np.concatenate([solution[nonzero] for solution, nonzero in zip(solutions, rows)])
        columns = np.concatenate([np.full(len(nonzero), column) for column, nonzero in enumerate(rows)])
        return self._matrix.__class__((data, (np.concatenate(rows), columns)), shape=b.shape)


def _format_change(parameter: Parameter, change: float) -> str:
    """Formats a (small) parameter change in the precise unit of the parameter, e.g. mm instead of m."""
    rules = UNIT_FORMAT_RULES.get(parameter.unit, {"normal": (1, "-", 3)})
    factor, unit, decimals = rules.get("precise", rules["normal"])
    if "precise" not in rules:
        decimals += 3  # e.g. degrees: small changes would otherwise round to zero
    return f"{change * factor:+.{decimals}f} {unit}"


def print_summary(alignment: AlignmentEstimator) -> None:
    logger.info(dict2table(alignment.data.group_stds, title="Group Standard Deviations"))
    logger.info(alignment.est_params)


def settings_str(alignment: AlignmentEstimator) -> str:
    return (
        f"\n _____________________________________________________________________\n"
        f"| ---------------------------- Alignment ---------------------------- |\n"
        f"| Estimation of:           {alignment.settings.estimation_settings.short_mode_str:<42} |\n"
        f"| Error probability [%]:   {alignment.settings.stochastics.error_probability * 100:<42} |\n"
        f"|_____________________________________________________________________|\n"
    )
