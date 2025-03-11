"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

from typing import Tuple, Union

from trajectopy.alignment import estimate_alignment
from trajectopy.core.alignment.result import AlignmentResult
from trajectopy.core.evaluation.ate_result import ATEResult
from trajectopy.core.evaluation.comparison import (
    compare_trajectories_absolute,
    compare_trajectories_relative,
)
from trajectopy.core.evaluation.rpe_result import RPEResult
from trajectopy.matching import match_trajectories
from trajectopy.settings import ProcessingSettings
from trajectopy.trajectory import Trajectory


def ate(
    trajectory_gt: Trajectory,
    trajectory_est: Trajectory,
    settings: ProcessingSettings = ProcessingSettings(),
    return_alignment: bool = False,
    align: bool = True,
) -> Union[ATEResult, Tuple[ATEResult, AlignmentResult]]:
    """
    Computes the absolute trajectory error (ATE) between two trajectories.

    Args:
        trajectory_gt (Trajectory): Ground truth trajectory.
        trajectory_est (Trajectory): Estimated trajectory.
        settings (ProcessingSettings, optional): Processing settings.
        return_alignment (bool, optional): Whether to return the alignment result.

    Description:

        The ATE is computed by first matching the estimated trajectory to the ground truth trajectory.
        Then, the alignment between the two trajectories is estimated.
        The estimated trajectory is aligned to the ground truth trajectory using the estimated alignment.
        Finally, the ATE is computed by comparing the aligned estimated trajectory to the ground truth trajectory.

    Returns:
        ATEResult: Result of the ATE computation.

    """
    trajectory_est, trajectory_gt = match_trajectories(
        traj_from=trajectory_est, traj_to=trajectory_gt, settings=settings.matching, inplace=False
    )

    if align:
        alignment = estimate_alignment(
            traj_from=trajectory_est,
            traj_to=trajectory_gt,
            alignment_settings=settings.alignment,
            matching_settings=settings.matching,
        )
        trajectory_est_aligned = trajectory_est.apply_alignment(alignment_result=alignment, inplace=False)
    else:
        alignment = AlignmentResult()
        trajectory_est_aligned = trajectory_est

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
    """
    Computes the relative pose error (RPE) between two trajectories.

    Args:
        trajectory_gt (Trajectory): Ground truth trajectory.
        trajectory_est (Trajectory): Estimated trajectory.
        settings (ProcessingSettings, optional): Processing settings.

    Description:

        The RPE is computed by comparing the relative poses between the estimated and ground truth trajectories.
        The pose distances are either defined in meters or in seconds depending on the settings.

    Returns:
        RPEResult: Result of the RPE computation.
    """
    trajectory_est, trajectory_gt = match_trajectories(
        traj_from=trajectory_est, traj_to=trajectory_gt, settings=settings.matching, inplace=False
    )
    return compare_trajectories_relative(
        traj_ref=trajectory_gt, traj_test=trajectory_est, settings=settings.relative_comparison
    )
