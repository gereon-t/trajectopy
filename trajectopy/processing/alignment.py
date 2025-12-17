import logging

import numpy as np

from trajectopy.core import settings
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.lib.alignment.data import AlignmentData
from trajectopy.processing.lib.alignment.direct import align_rotations
from trajectopy.processing.lib.alignment.equations import leverarm_time_component
from trajectopy.processing.lib.alignment.estimation import AlignmentEstimator
from trajectopy.processing.lib.alignment.parameters import (
    AlignmentParameters,
    SensorRotationParameters,
)
from trajectopy.results.alignment_result import AlignmentResult

logger = logging.getLogger(__name__)


def estimate_alignment(
    trajectory: Trajectory,
    other: Trajectory,
    alignment_settings: settings.AlignmentSettings = settings.AlignmentSettings(),
    matching_settings: settings.MatchingSettings = settings.MatchingSettings(),
) -> AlignmentResult:
    """Estimates the alignment between two trajectories.

    Performs Helmert, Leverarm, and Time shift estimation depending on the configuration.
    This function only estimates the alignment parameters, it does not apply them to the trajectory.

    Args:
        trajectory (Trajectory): Trajectory to align.
        other (Trajectory): Reference trajectory to align to.
        alignment_settings (AlignmentSettings, optional): Settings for the alignment process. Defaults to AlignmentSettings().
        matching_settings (MatchingSettings, optional): Settings for the matching process. Defaults to MatchingSettings().

    Returns:
        AlignmentResult: Result of the alignment process.
    """
    logger.info("Aligning trajectory positions ...")

    alignment_data = AlignmentData(
        traj_from=trajectory,
        traj_to=other,
        alignment_settings=alignment_settings,
        matching_settings=matching_settings,
    )
    ghm_alignment = AlignmentEstimator(alignment_data=alignment_data)
    estimated_parameters = ghm_alignment.estimate_parameters()

    if (
        alignment_data.traj_from.rotations is not None
        and alignment_data.traj_to.rotations is not None
        and alignment_settings.estimation_settings.sensor_rotation
    ):
        pre_aligned_trajectory = alignment_data.traj_from.transform(estimated_parameters.sim3_matrix, inplace=False)
        logger.info("Aligning rotations ...")
        sensor_rot_params = align_rotations(
            rot_from=pre_aligned_trajectory.rotations, rot_to=alignment_data.traj_to.rotations
        )
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


def apply_alignment(trajectory: Trajectory, alignment_result: AlignmentResult, inplace: bool = True) -> "Trajectory":
    """Transforms trajectory using alignment parameters.

    After computing the alignment parameters needed to align two trajectories, they can be
    applied to arbitrary trajectories.

    Args:
        trajectory (Trajectory): Trajectory to apply alignment to.
        alignment_result (AlignmentResult): Alignment result containing transformation parameters.
        inplace (bool, optional): Perform in-place. Defaults to True.

    Returns:
        Trajectory: Aligned trajectory.
    """

    def _prepare_alignment_application(
        trajectory: Trajectory, alignment_parameters: AlignmentParameters
    ) -> tuple[float, ...]:
        if trajectory.rotations is not None:
            rpy = trajectory.rotations.as_euler("xyz", degrees=False)
            euler_x, euler_y, euler_z = rpy[:, 0], rpy[:, 1], rpy[:, 2]
            lever_x, lever_y, lever_z = (
                alignment_parameters.lever_x.value,
                alignment_parameters.lever_y.value,
                alignment_parameters.lever_z.value,
            )
        else:
            logger.warning("Trajectory has no orientations. Cannot apply leverarm.")
            euler_x, euler_y, euler_z = 0, 0, 0
            lever_x, lever_y, lever_z = 0, 0, 0

        return euler_x, euler_y, euler_z, lever_x, lever_y, lever_z

    trajectory = trajectory if inplace else trajectory.copy()
    has_orientations = trajectory.has_orientation

    # leverarm and time
    (
        euler_x,
        euler_y,
        euler_z,
        lever_x,
        lever_y,
        lever_z,
    ) = _prepare_alignment_application(trajectory, alignment_result.position_parameters)

    # do not use velocities for time shift application
    trafo_x, trafo_y, trafo_z = leverarm_time_component(
        euler_x=euler_x,
        euler_y=euler_y,
        euler_z=euler_z,
        lever_x=lever_x,
        lever_y=lever_y,
        lever_z=lever_z,
        time_shift=0,
        speed_x=0,
        speed_y=0,
        speed_z=0,
    )
    trajectory.positions.xyz += np.c_[trafo_x, trafo_y, trafo_z]

    # instead, apply time shift as a simple shift of the timestamps
    trajectory.timestamps -= alignment_result.position_parameters.time_shift.value
    logger.info(
        f"Applied time shift of {alignment_result.position_parameters.time_shift.value*1000:.1f} ms to timestamps."
    )

    # similiarity transformation
    trajectory.transform(alignment_result.position_parameters.sim3_matrix)

    logger.info("Applied alignment parameters to positions.")

    # sensor orientation
    if trajectory.rotations is not None:
        trajectory.rotations = alignment_result.rotation_parameters.rotation_set * trajectory.rotations
        logger.info("Applied alignment parameters to orientations.")

    if not has_orientations:
        trajectory.rotations = None

    return trajectory


def adopt_first_pose(trajectory: Trajectory, other: Trajectory, inplace: bool = True) -> Trajectory:
    """Transform trajectory so that the first pose is identical in both.

    Args:
        trajectory (Trajectory): Trajectory to transform.
        other (Trajectory): Reference trajectory.
        inplace (bool, optional): Perform transformation in place. Defaults to True.

    Returns:
        Trajectory: Transformed trajectory.
    """
    trajectory = trajectory if inplace else trajectory.copy()
    adopt_first_position(trajectory=trajectory, other=other)
    adopt_first_orientation(trajectory=trajectory, other=other)
    return trajectory


def adopt_first_position(trajectory: Trajectory, other: Trajectory, inplace: bool = True) -> Trajectory:
    """Transform trajectory so that the first position is identical in both.

    Args:
        trajectory (Trajectory): Trajectory to transform.
        other (Trajectory): Reference trajectory.
        inplace (bool, optional): Perform transformation in place. Defaults to True.

    Returns:
        Trajectory: Transformed trajectory.
    """
    trajectory = trajectory if inplace else trajectory.copy()
    position_difference = other.positions.xyz[0, :] - trajectory.positions.xyz[0, :]
    trajectory.positions.xyz += position_difference
    return other


def adopt_first_orientation(trajectory: Trajectory, other: Trajectory, inplace: bool = True) -> Trajectory:
    """Transform trajectory so that the first orientation is identical in both.

    Args:
        trajectory (Trajectory): Trajectory to transform.
        other (Trajectory): Reference trajectory.
        inplace (bool, optional): Perform transformation in place. Defaults to True.

    Returns:
        Trajectory: Transformed trajectory.
    """
    trajectory = trajectory if inplace else trajectory.copy()
    if trajectory.rotations is not None and other.rotations is not None:
        rpy = trajectory.rotations.as_euler(seq="xyz")
        rotation_difference = other.rotations.as_euler(seq="xyz")[0, :] - rpy[0, :]

        trajectory.rotations = Rotations.from_euler(seq="xyz", angles=rpy + rotation_difference)

    return other


def align(
    trajectory: Trajectory,
    other: Trajectory,
    alignment_settings: settings.AlignmentSettings = settings.AlignmentSettings(),
    matching_settings: settings.MatchingSettings = settings.MatchingSettings(),
    inplace: bool = True,
) -> Trajectory:
    """Aligns the trajectory with another trajectory.

    Args:
        trajectory (Trajectory): Trajectory to align.
        other (Trajectory): Reference trajectory.
        alignment_settings (AlignmentSettings, optional): Settings for the alignment process.
            Defaults to AlignmentSettings().
        matching_settings (MatchingSettings, optional): Settings for the matching process.
            Defaults to MatchingSettings().
        inplace (bool, optional): Perform in-place. Defaults to True.

    Returns:
        Trajectory: Aligned trajectory.
    """
    alignment = estimate_alignment(
        trajectory=trajectory,
        other=other,
        alignment_settings=alignment_settings,
        matching_settings=matching_settings,
    )
    return apply_alignment(trajectory=trajectory, alignment_result=alignment, inplace=inplace)
