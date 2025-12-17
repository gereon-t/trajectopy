import logging

import numpy as np
from scipy.sparse import spdiags

from trajectopy.core.rotations import Rotations
from trajectopy.processing.lib.alignment.parameters import (
    HelmertTransformation,
    Leverarm,
    Parameter,
    SensorRotationParameters,
)
from trajectopy.utils.common import least_squares
from trajectopy.utils.definitions import Unit

logger = logging.getLogger(__name__)


def direct_helmert_transformation(
    xyz_from: np.ndarray, xyz_to: np.ndarray, weights: np.ndarray = np.zeros(0)
) -> HelmertTransformation:
    """
    Args:
        xyz_from (np.ndarray): n x 3 source point set.
        xyz_to (np.ndarray): n x 3 target point set.
        weights (np.ndarray, optional): Weights for the point sets. Defaults to np.zeros(0).

    Returns:
        HelmertTransformation: Helmert transformation parameters.

    Computes helmert transformation between two point sets

    Foerstner-Wrobel (2016) Photogrammetric Computer Vision pp. 406 - 411
    """
    if len(weights) == 0:
        weights = np.ones((len(xyz_from),))

    sum_of_weights = np.sum(weights)
    weighted_centroid_from = np.sum(xyz_from * weights[:, None], axis=0) / sum_of_weights

    weighted_centroid_to = np.sum(xyz_to * weights[:, None], axis=0) / sum_of_weights

    centered_from = xyz_from - weighted_centroid_from
    centered_to = xyz_to - weighted_centroid_to

    moment_matrix = weights * centered_from.T @ centered_to

    distances_from = np.linalg.norm(centered_from, axis=1)
    weighted_sum_of_squared_distances_from = (weights * distances_from) @ distances_from

    u, _, v = np.linalg.svd(moment_matrix, full_matrices=True)

    estimated_rotation = v.T @ u.T
    estimated_scale = (
        np.sum(weights[:, None] * centered_to @ estimated_rotation * centered_from)
        / weighted_sum_of_squared_distances_from
    )
    estimated_translation = weighted_centroid_to - estimated_scale * estimated_rotation @ weighted_centroid_from

    yaw = np.arctan2(estimated_rotation[1, 0], estimated_rotation[0, 0])
    pitch = np.arctan2(
        -estimated_rotation[2, 0], np.sqrt(estimated_rotation[2, 1] ** 2 + estimated_rotation[2, 2] ** 2)
    )
    roll = np.arctan2(estimated_rotation[2, 1], estimated_rotation[2, 2])
    return HelmertTransformation(
        trans_x=Parameter(value=estimated_translation[0], name="Translation x", unit=Unit.METER),
        trans_y=Parameter(value=estimated_translation[1], name="Translation y", unit=Unit.METER),
        trans_z=Parameter(value=estimated_translation[2], name="Translation z", unit=Unit.METER),
        rot_x=Parameter(value=roll, name="Rotation x", unit=Unit.RADIAN),
        rot_y=Parameter(value=pitch, name="Rotation y", unit=Unit.RADIAN),
        rot_z=Parameter(value=yaw, name="Rotation z", unit=Unit.RADIAN),
        scale=Parameter(value=estimated_scale, default=1.0, name="Scale", unit=Unit.SCALE),
    )


def direct_leverarm(
    *,
    xyz_to: np.ndarray,
    xyz_from: np.ndarray,
    rpy_body: np.ndarray,
    weights: np.ndarray | None = None,
    speed: np.ndarray | None = None,
) -> tuple[Leverarm, Parameter, np.ndarray]:
    """Leverarm (+time) estimation

    Estimates the leverarm and the time offset
    between two trajectories using a gauß-markov model.

    The observations are the difference between the target
    and the source positions.
    The parameters are:
        - time (if gradients / speeds are provided)
        - leverarm (dx, dy, dz)

    Args:
        xyz_to (np.ndarray): Source positions
        xyz_from (np.ndarray): target positions
        rpy_body (np.ndarray): platform orientations
        weights (np.ndarray, optional): observation weights.
                                        Defaults to None.
        speed (np.ndarray, optional): speed of the platform.
                                         Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the estimated
                                       parameters and the residuals of
                                       the adjustment.
    """

    if not len(xyz_to) == len(xyz_from) == len(rpy_body):
        raise ValueError("estimate_leverarm: All arrays must have equal dimensions!")

    if weights is None:
        weights = np.ones(xyz_to.size)
        logger.debug("Using default uniform weighting")
    else:
        logger.debug("Using custom weighting")

    # x_1 - x_2 = (cos(b) * cos(g) * d_x + (sin(a) * sin(b) * cos(g) - cos(a) * sin(g)) * d_y + (cos(a) * sin(b) * cos(g) + sin(a) * sin(g)) * d_z)
    # y_1 - y_2 = (cos(b) * sin(g) * d_x + (sin(a) * sin(b) * sin(g) + cos(a) * cos(g)) * d_y + (cos(a) * sin(b) * sin(g) - sin(a) * cos(g)) * d_z)
    # z_2 - z_2 = (-sin(b) * d_x + sin(a) * cos(b) * d_y + cos(a) * cos(b) * d_z)

    # or

    # x_1 - x_2 = v_x * d_t + (cos(b) * cos(g) * d_x + (sin(a) * sin(b) * cos(g) - cos(a) * sin(g)) * d_y + (cos(a) * sin(b) * cos(g) + sin(a) * sin(g)) * d_z)
    # y_1 - y_2 = v_y * d_t + (cos(b) * sin(g) * d_x + (sin(a) * sin(b) * sin(g) + cos(a) * cos(g)) * d_y + (cos(a) * sin(b) * sin(g) - sin(a) * cos(g)) * d_z)
    # z_2 - z_2 = v_z * d_t + (-sin(b) * d_x + sin(a) * cos(b) * d_y + cos(a) * cos(b) * d_z)
    a_design = _leverarm_design_matrix(rpy_body, speed=speed)

    observations = np.r_[
        xyz_to[:, 0] - xyz_from[:, 0],
        xyz_to[:, 1] - xyz_from[:, 1],
        xyz_to[:, 2] - xyz_from[:, 2],
    ]
    sigma_ll = spdiags(weights, 0, len(weights), len(weights))

    est_params, _, residuals = least_squares(a_design, observations, sigma_ll=sigma_ll)

    if speed is not None:
        time_shift = est_params[0]
        idx_offset = 1
    else:
        time_shift = 0
        idx_offset = 0

    return (
        Leverarm(
            x=Parameter(value=est_params[idx_offset], name="Leverarm x", unit=Unit.METER),
            y=Parameter(value=est_params[idx_offset + 1], name="Leverarm y", unit=Unit.METER),
            z=Parameter(value=est_params[idx_offset + 2], name="Leverarm z", unit=Unit.METER),
        ),
        Parameter(value=time_shift, name="Time shift", unit=Unit.SECOND),
        residuals,
    )


def _leverarm_design_matrix(rpy: np.ndarray, speed: np.ndarray | None = None) -> np.ndarray:
    """Builds design matrix for leverarm (+ time) estimation

    Contains the derivation of the observation equations with respect
    to the parameters

    Args:
        rpy (np.ndarray): platform orientations
        speed (np.ndarray, optional): Gradients / speeds at the
                                         positions. Defaults to None.
                                         If no gradients are provided
                                         the adjustment will fall back
                                         to a leverarm only estimation.

    Returns:
        np.ndarray: design matrix
    """
    a = rpy[:, 0]
    b = rpy[:, 1]
    g = rpy[:, 2]

    # design matrix
    a_design_x = np.c_[
        np.cos(b) * np.cos(g),
        -np.cos(a) * np.sin(g) + np.sin(a) * np.sin(b) * np.cos(g),
        np.sin(a) * np.sin(g) + np.cos(a) * np.sin(b) * np.cos(g),
    ]

    a_design_y = np.c_[
        np.cos(b) * np.sin(g),
        np.sin(a) * np.sin(b) * np.sin(g) + np.cos(a) * np.cos(g),
        -np.sin(a) * np.cos(g) + np.cos(a) * np.sin(b) * np.sin(g),
    ]

    a_design_z = np.c_[-np.sin(b), np.sin(a) * np.cos(b), np.cos(a) * np.cos(b)]

    if speed is not None:
        return np.c_[
            speed.T.reshape(
                speed.size,
            ),
            np.r_[a_design_x, a_design_y, a_design_z],
        ]
    else:
        return np.r_[a_design_x, a_design_y, a_design_z]


def direct_timeshift(
    *,
    xyz_to: np.ndarray,
    xyz_from: np.ndarray,
    speed: np.ndarray,
    weights: np.ndarray = np.zeros(0),
) -> tuple[Parameter, np.ndarray]:
    """Time shift estimation

    Estimates the time offset between two trajectories using a gauß-markov model.

    The observations are the difference between the target
    and the source positions.
    The parameters are:
        - time

    Args:
        xyz_to (np.ndarray): Source positions
        xyz_from (np.ndarray): target positions
        speed (np.ndarray): speed of the platform.
        weights (np.ndarray, optional): observation weights.
                                        Defaults to np.zeros(0).

    Returns:
        Tuple[Parameter, np.ndarray]: Tuple containing the estimated
                                      time shift parameter and the residuals of
                                      the adjustment.
    """

    if len(xyz_to) != len(xyz_from):
        raise ValueError("estimate_leverarm: All arrays must have equal dimensions!")

    if weights is None:
        weights = np.ones(xyz_to.size)
        logger.debug("Using default uniform weighting")
    else:
        logger.debug("Using custom weighting")

    a_design = speed.T.reshape(speed.size, 1)

    observations = np.r_[
        xyz_to[:, 0] - xyz_from[:, 0],
        xyz_to[:, 1] - xyz_from[:, 1],
        xyz_to[:, 2] - xyz_from[:, 2],
    ]
    sigma_ll = spdiags(weights, 0, len(weights), len(weights))

    est_params, _, residuals = least_squares(a_design, observations, sigma_ll=sigma_ll)

    return (
        Parameter(value=est_params[0], name="Time shift", unit=Unit.SECOND),
        residuals,
    )


def align_rotations(rot_from: Rotations, rot_to: Rotations) -> SensorRotationParameters:
    """Aligns the rotations of two trajectories"""

    if rot_to is None or rot_from is None:
        return Rotations.from_quat(np.array([0, 0, 0, 1]))

    rot_difference = (rot_to - rot_from).mean()
    rpy_diff = rot_difference.as_euler(seq="xyz")
    return SensorRotationParameters(
        sensor_rot_x=Parameter(name="Rotation x", value=rpy_diff[0], enabled=True, unit=Unit.RADIAN),
        sensor_rot_y=Parameter(name="Rotation y", value=rpy_diff[1], enabled=True, unit=Unit.RADIAN),
        sensor_rot_z=Parameter(name="Rotation z", value=rpy_diff[2], enabled=True, unit=Unit.RADIAN),
    )
