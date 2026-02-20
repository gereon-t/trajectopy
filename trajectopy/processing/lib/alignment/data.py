import copy
import logging
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from scipy.sparse import spdiags

from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import AlignmentSettings, MatchingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.matching import match_trajectories
from trajectopy.utils.common import gradient_3d

logger = logging.getLogger(__name__)


POSITION_VARIANCE_GROUPS: list[str] = ["XY_FROM", "Z_FROM", "XY_TO", "Z_TO"]
ORIENTATION_VARIANCE_GROUPS: list[str] = ["ROLL_PITCH", "YAW"]
SPEED_VARIANCE_GROUP: list[str] = ["SPEED"]


@dataclass
class AlignmentData:
    """Class holding the observation data required for Alignment


    Attributes:
        traj_from (Trajectory): Source trajectory
        traj_to (Trajectory): Target trajectory
        alignment_settings (AlignmentSettings): Settings for the alignment
        matching_settings (MatchingSettings): Settings for the matching
    """

    traj_from: Trajectory
    traj_to: Trajectory
    alignment_settings: AlignmentSettings
    matching_settings: MatchingSettings

    def __post_init__(self) -> None:
        self.traj_from = copy.deepcopy(self.traj_from)
        self.traj_to = copy.deepcopy(self.traj_to)
        self.alignment_settings = copy.deepcopy(self.alignment_settings)

        self.setup()

    def setup(self) -> None:
        """Prepare two trajectories for alignment.

        This method will filter the trajectories
        by speed and resample both trajectories to
        the same sampling.
        """
        # speed filter
        self.traj_to.mask(self.traj_to.absolute_velocity >= self.alignment_settings.preprocessing.min_speed)
        self.traj_from.mask(self.traj_from.absolute_velocity >= self.alignment_settings.preprocessing.min_speed)

        if (
            self.alignment_settings.preprocessing.time_start != 0
            or self.alignment_settings.preprocessing.time_end != 0
        ):
            time_span = (
                self.traj_from.timestamps[0] + self.alignment_settings.preprocessing.time_start,
                self.traj_from.timestamps[0] + self.alignment_settings.preprocessing.time_end,
            )
            self.traj_to.crop(time_span[0], time_span[1])
            self.traj_from.crop(time_span[0], time_span[1])

        if len(self.traj_from) == 0 or len(self.traj_to) == 0:
            raise ValueError("At least one trajectory is empty after preprocessing!")

        match_trajectories(trajectory=self.traj_from, other=self.traj_to, matching_settings=self.matching_settings)

        logger.info(
            "Using timespan of %.3f seconds between %.3f and %.3f.",
            self.traj_from.timestamps[-1] - self.traj_from.timestamps[0],
            self.traj_from.timestamps[0],
            self.traj_from.timestamps[-1],
        )

        self.obs_vector = self.build_obs_vector(
            xyz_from=self.traj_from.positions.xyz,
            xyz_to=self.traj_to.positions.xyz,
            rot_from=self.traj_from.rotations,
            speed=self.traj_from.velocity_xyz,
        )

        logger.debug("Observation vector created.")
        self.var_vector = self.build_var_vector()
        logger.debug("Variance vector created.")
        self.res_vector = self.build_res_vector()
        logger.debug("Residual vector created.")

    @cached_property
    def observation_groups(
        self,
    ) -> dict[str, tuple[int | None, int | None]]:
        """Returns the observation groups depending on the settings"""
        speed_indices: tuple[int | None, int | None] = (None, None)

        if not self.alignment_settings.estimation_settings.time_shift_enabled:
            speed_indices = (None, None)
        elif self.alignment_settings.estimation_settings.leverarm_enabled:
            speed_indices = (9, 12)
        else:
            speed_indices = (6, 9)

        return {
            "XY_FROM": (0, 2),
            "Z_FROM": (2, 3),
            "XYZ_FROM": (0, 3),
            "XY_TO": (3, 5),
            "Z_TO": (5, 6),
            "XYZ_TO": (3, 6),
            "POSITIONS": (0, 6),
            "ROLL_PITCH": (6, 8) if self.alignment_settings.estimation_settings.leverarm_enabled else (None, None),
            "YAW": (8, 9) if self.alignment_settings.estimation_settings.leverarm_enabled else (None, None),
            "RPY": (6, 9) if self.alignment_settings.estimation_settings.leverarm_enabled else (None, None),
            "SPEED": speed_indices,
        }

    @cached_property
    def variance_groups(self) -> list[str]:
        """Returns the variance groups depending on the settings"""
        variance_groups = copy.deepcopy(POSITION_VARIANCE_GROUPS)

        if self.alignment_settings.estimation_settings.leverarm_enabled:
            variance_groups.extend(ORIENTATION_VARIANCE_GROUPS)

        if self.alignment_settings.estimation_settings.time_shift_enabled:
            variance_groups.extend(SPEED_VARIANCE_GROUP)

        return variance_groups

    @property
    def obs_vector(self) -> np.ndarray:
        return self._obs_vector

    def __len__(self) -> int:
        return len(self._obs_vector) // self.num_obs_per_epoch

    @obs_vector.setter
    def obs_vector(self, values: np.ndarray) -> None:
        if len(values) / self.num_obs_per_epoch % 1 != 0:
            raise ValueError("Observation vector is not divisible by the number of observations per epoch!")

        self._obs_vector = values

    @property
    def var_vector(self) -> np.ndarray:
        return self._var_vector

    @var_vector.setter
    def var_vector(self, values: np.ndarray) -> None:
        if len(values) / self.num_obs_per_epoch % 1 != 0:
            raise ValueError("Variance vector is not divisible by the number of observations per epoch!")

        self._var_vector = values

    @property
    def res_vector(self) -> np.ndarray:
        return self._res_vector

    @res_vector.setter
    def res_vector(self, values: np.ndarray) -> None:
        if len(values) / self.num_obs_per_epoch % 1 != 0:
            raise ValueError("Residual vector is not divisible by the number of observations per epoch!")

        self._res_vector = values

    @property
    def est_obs_vector(self) -> np.ndarray:
        return self._obs_vector + self._res_vector

    @property
    def sigma_ll(self):
        return spdiags(self.var_vector, 0, len(self.var_vector), len(self.var_vector))

    @property
    def tstamps(self) -> np.ndarray:
        return self.traj_from.timestamps

    @cached_property
    def rpy_to(self) -> np.ndarray | None:
        return self.traj_to.rotations.as_euler(seq="xyz") if self.traj_to.rotations is not None else None

    @property
    def number_of_epochs(self) -> int:
        return len(self._obs_vector) // self.num_obs_per_epoch

    @property
    def num_obs_per_epoch(self) -> int:
        """
        Returns the number of observations per epoch depending on the
        enabled estimation modes.
        """
        obs_count = 6

        if self.alignment_settings.estimation_settings.leverarm_enabled:
            obs_count += 3

        if self.alignment_settings.estimation_settings.time_shift_enabled:
            obs_count += 3

        return obs_count

    def build_obs_vector(
        self,
        xyz_from: np.ndarray,
        xyz_to: np.ndarray,
        rot_from: Rotations | None,
        speed: np.ndarray | None,
    ) -> np.ndarray:
        """
        Creates the observation vector required for the alignment adjustment.

        Args:
            xyz_from (np.ndarray): Positions to be aligned.
            xyz_to (np.ndarray): Target Positions.
            rpy_from (np.ndarray): The roll, pitch, and yaw angles of the platform.
                                   Those angles should describe the rotation of the
                                   body-fixed coordinate system with respect to the
                                   inertial coordinate system.

        Returns:
            np.ndarray: The observation vector required for the alignment adjustment.
        """
        obs_init: np.ndarray = np.c_[xyz_from, xyz_to]

        if self.alignment_settings.estimation_settings.leverarm_enabled and rot_from is not None:
            obs_init = np.c_[obs_init, rot_from.as_euler(seq="xyz")]
        elif self.alignment_settings.estimation_settings.leverarm_enabled:
            raise ValueError(
                "Failed to create observation vector: Please provide platform orientations for leverarm alignment!"
            )

        if self.alignment_settings.estimation_settings.time_shift_enabled:
            if speed is None and self.tstamps is not None:
                speed = speed or gradient_3d(xyz_from, tstamps=self.tstamps)

            if speed is None:
                raise ValueError(
                    "Failed to create observation vector: Please provide platform speed for time shift alignment!"
                )

            speed[:, not self.alignment_settings.estimation_settings.time_shift_filter] = 0
            obs_init = np.c_[obs_init, speed]

        return np.reshape(obs_init, (obs_init.size,))

    def build_var_vector(self) -> np.ndarray:
        """Sets up the variance vector

        Its size depends on whether the
        leverarm should be estimated or
        not. In this case, not only the
        source and the target positions
        are relevant but also the platform
        orientations. Also, when estimating
        the time shift, the platform speed
        is also considered.

        Returns:
            np.ndarray: variance vector
        """
        variances = np.ones_like(self._obs_vector)

        variances_xyz_from = np.ones((self.number_of_epochs, 3)) * self.alignment_settings.stochastics.var_xy_from
        variances_xyz_from[:, 2] = np.ones((self.number_of_epochs,)) * self.alignment_settings.stochastics.var_z_from

        variances_xyz_to = np.ones((self.number_of_epochs, 3)) * self.alignment_settings.stochastics.var_xy_to
        variances_xyz_to[:, 2] = np.ones((self.number_of_epochs,)) * self.alignment_settings.stochastics.var_z_to

        variances_rpy_body = np.ones((self.number_of_epochs, 3)) * self.alignment_settings.stochastics.var_roll_pitch
        variances_rpy_body[:, 2] = np.ones((self.number_of_epochs,)) * self.alignment_settings.stochastics.var_yaw

        variances_speed_to = np.ones((self.number_of_epochs, 3)) * self.alignment_settings.stochastics.var_speed_to

        self._set_vector_components(vector=variances, values=variances_xyz_from, key="XYZ_FROM")
        self._set_vector_components(vector=variances, values=variances_xyz_to, key="XYZ_TO")

        if self.alignment_settings.estimation_settings.leverarm_enabled:
            self._set_vector_components(vector=variances, values=variances_rpy_body, key="RPY")

        if self.alignment_settings.estimation_settings.time_shift_enabled:
            self._set_vector_components(vector=variances, values=variances_speed_to, key="SPEED")
        return variances

    def build_res_vector(self) -> np.ndarray:
        return np.zeros_like(self._obs_vector)

    def get_obs_group(self, key: str) -> tuple[np.ndarray, ...]:
        return self._extract_from_vector(vector=self._obs_vector, key=key)

    def get_est_obs_group(self, key: str) -> tuple[np.ndarray, ...]:
        return self._extract_from_vector(vector=self.est_obs_vector, key=key)

    def set_obs_group(self, key: str, values: np.ndarray) -> None:
        self._set_vector_components(vector=self._obs_vector, values=values, key=key)

    def get_var_group(self, key: str) -> tuple[np.ndarray, ...]:
        return self._extract_from_vector(vector=self._var_vector, key=key)

    def set_var_group(self, key: str, values: np.ndarray) -> None:
        self._set_vector_components(vector=self._var_vector, values=values, key=key)

    def get_res_group(self, key: str) -> tuple[np.ndarray, ...]:
        return self._extract_from_vector(vector=self._res_vector, key=key)

    def set_res_group(self, key: str, values: np.ndarray) -> None:
        self._set_vector_components(vector=self._res_vector, values=values, key=key)

    def _extract_from_vector(self, vector: np.ndarray, key: str) -> tuple[np.ndarray, ...]:
        group_indices = self.observation_groups[key]
        if group_indices[0] is None or group_indices[1] is None:
            raise ValueError("Tried to access observation group that is not defined")

        return tuple(
            np.array(vector[i :: self.num_obs_per_epoch]).ravel() for i in range(group_indices[0], group_indices[1])
        )

    def _set_vector_components(self, vector: np.ndarray, values: np.ndarray, key: str) -> None:
        group_indices = self.observation_groups[key]
        if group_indices[0] is None or group_indices[1] is None:
            raise ValueError("Tried to access observation group that is not defined")

        if values.shape[1] != group_indices[1] - group_indices[0]:
            raise ValueError("Input array should have 1 column for each index defined by group_indices range!")

        for col_index, i in enumerate(range(group_indices[0], group_indices[1])):
            vector[i :: self.num_obs_per_epoch] = values[:, col_index]

    @property
    def group_stds(self) -> dict[str, float]:
        """Returns the mean standard deviation for each group"""
        return {group_key: np.mean(np.sqrt(self.get_var_group(key=group_key))) for group_key in self.variance_groups}

    @property
    def xyz_from(self) -> np.ndarray:
        return np.c_[self.get_obs_group("XYZ_FROM")]

    @property
    def xyz_to(self) -> np.ndarray:
        return np.c_[self.get_obs_group("XYZ_TO")]

    @property
    def rpy_from(self) -> np.ndarray:
        if not self.alignment_settings.estimation_settings.leverarm_enabled:
            return np.zeros((self.number_of_epochs, 3))

        return np.c_[self.get_obs_group("RPY")]

    @property
    def speed(self) -> np.ndarray:
        if not self.alignment_settings.estimation_settings.time_shift_enabled:
            return np.zeros((self.number_of_epochs, 3))

        return np.c_[self.get_obs_group("SPEED")]

    @property
    def x_from(self) -> np.ndarray:
        return self.xyz_from[:, 0]

    @property
    def y_from(self) -> np.ndarray:
        return self.xyz_from[:, 1]

    @property
    def z_from(self) -> np.ndarray:
        return self.xyz_from[:, 2]

    @property
    def x_to(self) -> np.ndarray:
        return self.xyz_to[:, 0]

    @property
    def y_to(self) -> np.ndarray:
        return self.xyz_to[:, 1]

    @property
    def z_to(self) -> np.ndarray:
        return self.xyz_to[:, 2]

    @property
    def speed_x(self) -> np.ndarray:
        return self.speed[:, 0]

    @property
    def speed_y(self) -> np.ndarray:
        return self.speed[:, 1]

    @property
    def speed_z(self) -> np.ndarray:
        return self.speed[:, 2]

    @property
    def euler_x(self) -> np.ndarray:
        return self.rpy_from[:, 0]

    @property
    def euler_y(self) -> np.ndarray:
        return self.rpy_from[:, 1]

    @property
    def euler_z(self) -> np.ndarray:
        return self.rpy_from[:, 2]

    @property
    def est_xyz_from(self) -> np.ndarray:
        return np.c_[self.get_est_obs_group("XYZ_FROM")]

    @property
    def est_xyz_to(self) -> np.ndarray:
        return np.c_[self.get_est_obs_group("XYZ_TO")]

    @property
    def est_rpy_from(self) -> np.ndarray:
        if not self.alignment_settings.estimation_settings.leverarm_enabled:
            return np.zeros((self.number_of_epochs, 3))

        return np.c_[self.get_est_obs_group("RPY")]

    @property
    def est_speed(self) -> np.ndarray:
        if not self.alignment_settings.estimation_settings.time_shift_enabled:
            return np.zeros((self.number_of_epochs, 3))

        return np.c_[self.get_est_obs_group("SPEED")]

    @property
    def est_x_from(self) -> np.ndarray:
        return self.est_xyz_from[:, 0]

    @property
    def est_y_from(self) -> np.ndarray:
        return self.est_xyz_from[:, 1]

    @property
    def est_z_from(self) -> np.ndarray:
        return self.est_xyz_from[:, 2]

    @property
    def est_x_to(self) -> np.ndarray:
        return self.est_xyz_to[:, 0]

    @property
    def est_y_to(self) -> np.ndarray:
        return self.est_xyz_to[:, 1]

    @property
    def est_z_to(self) -> np.ndarray:
        return self.est_xyz_to[:, 2]

    @property
    def est_speed_x(self) -> np.ndarray:
        return self.est_speed[:, 0]

    @property
    def est_speed_y(self) -> np.ndarray:
        return self.est_speed[:, 1]

    @property
    def est_speed_z(self) -> np.ndarray:
        return self.est_speed[:, 2]

    @property
    def est_euler_x(self) -> np.ndarray:
        return self.est_rpy_from[:, 0]

    @property
    def est_euler_y(self) -> np.ndarray:
        return self.est_rpy_from[:, 1]

    @property
    def est_euler_z(self) -> np.ndarray:
        return self.est_rpy_from[:, 2]

    @cached_property
    def thresholds(self) -> np.ndarray:
        """Computes thresholds for parameter updates

        This method creates thresholds for each
        type of parameter. Based on a given metric
        threshold, this method will translate those
        to angle and scale thresholds.
        Finally it will put all thresholds together
        to a 11x1 threshold vector
        """
        max_dist_from = np.linalg.norm(np.max(self.xyz_from, axis=0) - np.min(self.xyz_from, axis=0))
        max_dist_to = np.linalg.norm(np.max(self.xyz_to, axis=0) - np.min(self.xyz_to, axis=0))
        max_dist = np.max([max_dist_from, max_dist_to])

        # scale and angle threshold should fit to metric threshold
        scale_and_angle_th = self.alignment_settings.metric_threshold / max_dist

        thresholds = np.ones(11)
        thresholds[:3] *= self.alignment_settings.metric_threshold
        thresholds[3:7] *= scale_and_angle_th
        thresholds[7] *= self.alignment_settings.time_threshold
        thresholds[8:] *= self.alignment_settings.metric_threshold
        return thresholds

    def get_variance_estimation_subset(self, num_obs: int = 200) -> "AlignmentData":
        """Returns a subset of the alignment data for variance estimation

        This method will return a subset of the alignment data
        that contains 'num_obs' observations. The subset is selected
        by finding the epoch with the highest standard deviation of
        the observations. The idea is that a higher variance in the
        observations will lead to a better estimation of the variances.

        The motivation behind this is that the variance estimation
        is memory and time consuming. Therefore, a subset of the
        observations is used for the estimation.

        Args:
            num_obs (int, optional): Subet size. Defaults to 200.

        Returns:
            AlignmentData: Cropped alignment data
        """
        obs_matrix = np.reshape(self.obs_vector, (len(self), self.num_obs_per_epoch))
        if len(obs_matrix) <= num_obs:
            return copy.deepcopy(self)

        obs_norm = np.linalg.norm(obs_matrix, axis=1)
        obs_series = pd.Series(obs_norm)
        max_idx = obs_series.rolling(window=num_obs).std().idxmax()

        traj_index = np.arange(max_idx - num_obs, max_idx)
        traj_from = self.traj_from.mask(traj_index, inplace=False)
        traj_to = self.traj_to.mask(traj_index, inplace=False)
        return AlignmentData(
            traj_from=traj_from,
            traj_to=traj_to,
            alignment_settings=self.alignment_settings,
            matching_settings=self.matching_settings,
        )
