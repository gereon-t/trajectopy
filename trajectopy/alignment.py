"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import logging

from trajectopy.core.alignment.data import AlignmentData
from trajectopy.core.alignment.direct import align_rotations
from trajectopy.core.alignment.estimation import AlignmentEstimation
from trajectopy.core.alignment.parameters import SensorRotationParameters
from trajectopy.core.alignment.result import AlignmentResult
from trajectopy.settings import AlignmentSettings, MatchingSettings
from trajectopy.trajectory import Trajectory

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
