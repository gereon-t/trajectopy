import copy
import logging

import numpy as np

from trajectopy.core.parameters import (
    AlignmentParameters,
    SensorRotationParameters,
)
from trajectopy.core.settings import (
    AlignmentSettings,
    MatchingSettings,
    ProcessingSettings,
    Settings,
    resolve_settings,
)
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing._lib.alignment.data import AlignmentData
from trajectopy.processing._lib.alignment.direct import align_rotations
from trajectopy.processing._lib.alignment.equations import leverarm_time_component
from trajectopy.processing._lib.alignment.estimation import AlignmentEstimator
from trajectopy.results.alignment_result import AlignmentResult

logger = logging.getLogger(__name__)


def _alignment_and_matching_settings(
    processing: Settings | None,
) -> tuple[AlignmentSettings, MatchingSettings]:
    if isinstance(processing, ProcessingSettings):
        return processing.alignment, processing.matching
    return resolve_settings(processing, AlignmentSettings), MatchingSettings()


def estimate_alignment(
    trajectory: Trajectory,
    other: Trajectory,
    settings: AlignmentSettings | ProcessingSettings | None = None,
    *,
    initial_parameters: AlignmentParameters | None = None,
) -> AlignmentResult:
    """Estimates the alignment between two trajectories.

    Performs Helmert, Leverarm, and Time shift estimation depending on the configuration.
    This function only estimates the alignment parameters, it does not apply them to the trajectory.

    Args:
        trajectory (Trajectory): Trajectory to align.
        other (Trajectory): Reference trajectory to align to.
        settings (AlignmentSettings | ProcessingSettings, optional): Alignment settings, or processing
            settings whose alignment and matching settings are used. Defaults to the default settings.
        initial_parameters (AlignmentParameters, optional): Prior (e.g. from a manual alignment) used as
            start values of the least squares adjustment instead of the direct initialization. Parameters
            that are not estimated stay fixed at their prior value. A time shift of the prior may be large
            (e.g. trajectories recorded with unrelated clocks): the timestamps are shifted by it before the
            trajectories are matched, and only the remaining time shift is estimated. Defaults to None.

    Returns:
        AlignmentResult: Result of the alignment process.

    See Also:
        [`trajectopy.apply_alignment`][trajectopy.apply_alignment]: applies the result.
        [`trajectopy.align`][trajectopy.align]: both steps at once.
        [`trajectopy.AlignmentSettings`][trajectopy.AlignmentSettings]: which parameters are estimated.

    References:
        1. G. Tombrink, A. Dreier, L. Klingbeil, H. Kuhlmann (2024): Spatio-temporal trajectory alignment for trajectory evaluation. Journal of Applied Geodesy. https://doi.org/10.1515/jag-2024-0040

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> alignment = tpy.estimate_alignment(estimate, reference)
        >>> alignment.converged
        True
        >>> print(f"{alignment.position_parameters.sim_trans_x.value:.3f} m")
        -1.363 m
        >>> aligned = tpy.apply_alignment(estimate, alignment)
    """
    alignment_settings, matching_settings = _alignment_and_matching_settings(settings)

    logger.info("Aligning trajectory positions ...")

    # The time shift is modelled by the velocity (x + v * dt), which only holds for small shifts. A prior time
    # shift is therefore applied to the timestamps beforehand and added to the estimated (remaining) one.
    prior_time_shift = 0.0 if initial_parameters is None else float(initial_parameters.time_shift.value)
    if initial_parameters is not None and prior_time_shift != 0.0:
        trajectory = trajectory.copy()
        trajectory.timestamps = trajectory.timestamps - prior_time_shift
        initial_parameters = copy.deepcopy(initial_parameters)
        initial_parameters.time_shift.value = 0.0
        initial_parameters.time_shift.default = 0.0

    alignment_data = AlignmentData(
        traj_from=trajectory,
        traj_to=other,
        alignment_settings=alignment_settings,
        matching_settings=matching_settings,
    )
    ghm_alignment = AlignmentEstimator(alignment_data=alignment_data, initial_parameters=initial_parameters)
    estimated_parameters = ghm_alignment.estimate_parameters()
    if prior_time_shift != 0.0:
        time_shift = estimated_parameters.time_shift
        time_shift.value = time_shift.value + prior_time_shift
        time_shift.default = time_shift.default + prior_time_shift

    pre_aligned_rotations = (
        alignment_data.traj_from.transform(estimated_parameters.sim3_matrix, inplace=False).rotations
        if alignment_settings.estimation_settings.sensor_rotation
        else None
    )
    if pre_aligned_rotations is not None and alignment_data.traj_to.rotations is not None:
        logger.info("Aligning rotations ...")
        sensor_rot_params = align_rotations(rot_from=pre_aligned_rotations, rot_to=alignment_data.traj_to.rotations)
        logger.info("Estimated sensor rotation:\n%s", sensor_rot_params)
    else:
        sensor_rot_params = SensorRotationParameters(enabled=False)

    return AlignmentResult(
        name=f"{alignment_data.traj_from.name} to {alignment_data.traj_to.name}",
        position_parameters=estimated_parameters,
        rotation_parameters=sensor_rot_params,
        estimation_of=ghm_alignment.settings.estimation_settings,
        # without least squares parameters there is nothing that could fail to converge
        converged=ghm_alignment.has_results or alignment_settings.estimation_settings.all_lq_disabled,
    )


def manual_alignment(trajectory: Trajectory, other: Trajectory, parameters: AlignmentParameters) -> AlignmentResult:
    """Creates an alignment result from manually chosen parameters (no estimation).

    All parameters are marked as enabled so that their values are applied by `apply_alignment`.

    Args:
        trajectory (Trajectory): Trajectory that is aligned.
        other (Trajectory): Reference trajectory.
        parameters (AlignmentParameters): Manually chosen alignment parameters.

    Returns:
        AlignmentResult: Alignment result holding the manual parameters.
    """
    position_parameters = copy.deepcopy(parameters)
    position_parameters.enable()
    return AlignmentResult(
        name=f"{trajectory.name} to {other.name} (manual)",
        position_parameters=position_parameters,
        rotation_parameters=SensorRotationParameters(enabled=False),
        estimation_of=position_parameters.to_estimation_settings(),
        converged=True,
    )


def apply_alignment(
    trajectory: Trajectory, alignment_result: AlignmentResult, *, inplace: bool = False
) -> "Trajectory":
    """Transforms trajectory using alignment parameters.

    After computing the alignment parameters needed to align two trajectories, they can be
    applied to arbitrary trajectories.

    Args:
        trajectory (Trajectory): Trajectory to apply alignment to.
        alignment_result (AlignmentResult): Alignment result containing transformation parameters.
        inplace (bool, optional): Perform in-place. Defaults to False.

    Returns:
        Trajectory: Aligned trajectory.

    See Also:
        [`trajectopy.estimate_alignment`][trajectopy.estimate_alignment]: estimates the alignment.
        [`trajectopy.align`][trajectopy.align]: both steps at once.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> alignment = tpy.estimate_alignment(estimate, reference)
        >>> aligned = tpy.apply_alignment(estimate, alignment)
        >>> len(aligned) == len(estimate)
        True
    """

    def _prepare_alignment_application(
        trajectory: Trajectory, alignment_parameters: AlignmentParameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        if trajectory.rotations is not None:
            rpy = trajectory.rotations.as_euler("xyz", degrees=False)
            euler_x, euler_y, euler_z = rpy[:, 0], rpy[:, 1], rpy[:, 2]
            lever_x, lever_y, lever_z = (
                alignment_parameters.lever_x.value,
                alignment_parameters.lever_y.value,
                alignment_parameters.lever_z.value,
            )
        else:
            if any(
                parameter.value != 0.0
                for parameter in (
                    alignment_parameters.lever_x,
                    alignment_parameters.lever_y,
                    alignment_parameters.lever_z,
                )
            ):
                logger.warning("Trajectory has no orientations. Cannot apply leverarm.")
            euler_x = euler_y = euler_z = np.zeros(len(trajectory))
            lever_x, lever_y, lever_z = 0.0, 0.0, 0.0

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
        f"Applied time shift of {alignment_result.position_parameters.time_shift.value * 1000:.1f} ms to timestamps."
    )

    # similiarity transformation
    trajectory.transform(alignment_result.position_parameters.sim3_matrix, inplace=True)

    logger.info("Applied alignment parameters to positions.")

    # sensor orientation (mounting rotation in the body frame)
    if trajectory.rotations is not None:
        trajectory.rotations = trajectory.rotations * alignment_result.rotation_parameters.rotation_set
        logger.info("Applied alignment parameters to orientations.")

    if not has_orientations:
        trajectory.rotations = None

    return trajectory


def adopt_first_pose(trajectory: Trajectory, other: Trajectory, *, inplace: bool = False) -> Trajectory:
    """Transform trajectory so that the first pose is identical in both.

    Args:
        trajectory (Trajectory): Trajectory to transform.
        other (Trajectory): Reference trajectory.
        inplace (bool, optional): Perform transformation in place. Defaults to False.

    Returns:
        Trajectory: Transformed trajectory.

    See Also:
        [`trajectopy.align`][trajectopy.align]: least squares alignment of all poses.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> matched, matched_reference = tpy.match_trajectories(estimate, reference)
        >>> adopted = tpy.adopt_first_pose(matched, matched_reference)
        >>> bool(abs(adopted.xyz[0] - matched_reference.xyz[0]).max() < 1e-6)
        True
    """
    trajectory = trajectory if inplace else trajectory.copy()
    adopt_first_position(trajectory=trajectory, other=other, inplace=True)
    adopt_first_orientation(trajectory=trajectory, other=other, inplace=True)
    return trajectory


def adopt_first_position(trajectory: Trajectory, other: Trajectory, *, inplace: bool = False) -> Trajectory:
    """Transform trajectory so that the first position is identical in both.

    Args:
        trajectory (Trajectory): Trajectory to transform.
        other (Trajectory): Reference trajectory.
        inplace (bool, optional): Perform transformation in place. Defaults to False.

    Returns:
        Trajectory: Transformed trajectory.
    """
    trajectory = trajectory if inplace else trajectory.copy()
    position_difference = other.positions.xyz[0, :] - trajectory.positions.xyz[0, :]
    trajectory.positions.xyz += position_difference
    return trajectory


def adopt_first_orientation(trajectory: Trajectory, other: Trajectory, *, inplace: bool = False) -> Trajectory:
    """Transform trajectory so that the first orientation is identical in both.

    Args:
        trajectory (Trajectory): Trajectory to transform.
        other (Trajectory): Reference trajectory.
        inplace (bool, optional): Perform transformation in place. Defaults to False.

    Returns:
        Trajectory: Transformed trajectory.
    """
    trajectory = trajectory if inplace else trajectory.copy()
    if trajectory.rotations is not None and other.rotations is not None:
        # one rotation for all poses, so that the rotations between the poses are kept
        first = trajectory.rotations[0]
        first_other = other.rotations[0]
        trajectory.rotations = first_other * first.inv() * trajectory.rotations

    return trajectory


def align(
    trajectory: Trajectory,
    other: Trajectory,
    settings: AlignmentSettings | ProcessingSettings | None = None,
    *,
    inplace: bool = False,
) -> Trajectory:
    """Aligns the trajectory with another trajectory.

    Args:
        trajectory (Trajectory): Trajectory to align.
        other (Trajectory): Reference trajectory.
        settings (AlignmentSettings | ProcessingSettings, optional): Alignment settings, or processing
            settings whose alignment and matching settings are used. Defaults to the default settings.
        inplace (bool, optional): Perform in-place. Defaults to False.

    Returns:
        Trajectory: Aligned trajectory.

    See Also:
        [`trajectopy.estimate_alignment`][trajectopy.estimate_alignment], [`trajectopy.apply_alignment`][trajectopy.apply_alignment]: the single steps.

    References:
        1. G. Tombrink, A. Dreier, L. Klingbeil, H. Kuhlmann (2024): Spatio-temporal trajectory alignment for trajectory evaluation. Journal of Applied Geodesy. https://doi.org/10.1515/jag-2024-0040

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> aligned = tpy.align(estimate, reference)
        >>> len(aligned) == len(estimate)
        True
    """
    alignment = estimate_alignment(trajectory, other, settings)
    return apply_alignment(trajectory=trajectory, alignment_result=alignment, inplace=inplace)
