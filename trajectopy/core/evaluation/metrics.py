from typing import Tuple, Union

from trajectopy.core.alignment.estimation import estimate_alignment
from trajectopy.core.alignment.result import AlignmentResult
from trajectopy.core.evaluation.ate_result import ATEResult
from trajectopy.core.evaluation.comparison import compare_trajectories_absolute, compare_trajectories_relative
from trajectopy.core.evaluation.rpe_result import RPEResult
from trajectopy.core.matching import match_trajectories
from trajectopy.core.settings.processing import ProcessingSettings
from trajectopy.core.trajectory import Trajectory


def ate(
    trajectory_gt: Trajectory,
    trajectory_est: Trajectory,
    settings: ProcessingSettings = ProcessingSettings(),
    return_alignment: bool = False,
) -> Union[ATEResult, Tuple[ATEResult, AlignmentResult]]:
    """
    Computes the absolute trajectory error (ATE) between two trajectories.

    Args:
        trajectory_gt (Trajectory): Ground truth trajectory.
        trajectory_est (Trajectory): Estimated trajectory.
        settings (ProcessingSettings, optional): Processing settings.

    Returns:
        ATEResult: Result of the ATE computation.

    """
    match_trajectories(traj_from=trajectory_est, traj_to=trajectory_gt, settings=settings.matching)
    alignment = estimate_alignment(
        traj_from=trajectory_est,
        traj_to=trajectory_gt,
        alignment_settings=settings.alignment,
        matching_settings=settings.matching,
    )
    trajectory_est_aligned = trajectory_est.apply_alignment(alignment_result=alignment, inplace=False)
    return (
        (
            compare_trajectories_absolute(traj_ref=trajectory_gt, traj_test=trajectory_est_aligned),
            alignment,
        )
        if return_alignment
        else compare_trajectories_absolute(traj_ref=trajectory_gt, traj_test=trajectory_est_aligned)
    )


def rpe(
    trajectory_gt: Trajectory, trajectory_est: Trajectory, settings: ProcessingSettings = ProcessingSettings()
) -> RPEResult:
    match_trajectories(traj_from=trajectory_est, traj_to=trajectory_gt, settings=settings.matching)
    return compare_trajectories_relative(
        traj_ref=trajectory_gt, traj_test=trajectory_est, settings=settings.relative_comparison
    )
