"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import List

import numpy as np
from pointset import PointSet

from trajectopy.core.rotationset import RotationSet
from trajectopy.core.trajectory import Trajectory

logger = logging.getLogger("root")


def merge_trajectories(trajectories: List[Trajectory]) -> Trajectory:
    """
    Merges a list of trajectories into one trajectory.

    This function ignores EPSG codes and merges the
    trajectories based on their timestamps. Therefore,
    all trajectories should be in the same coordinate system
    before merging.

    Args:
        list[Trajectory]: List of trajectories to merge.

    Returns:
        Trajectory: Merged trajectory.

    """
    epsg_set = {t.pos.epsg for t in trajectories}

    if len(epsg_set) > 1:
        logger.warning(
            "Merging trajectories with different EPSG codes. "
            "This may lead to unexpected results. "
            "Consider reprojecting the trajectories to the same EPSG code."
        )

    epsg = epsg_set.pop()

    merged_xyz = np.concatenate([t.pos.xyz for t in trajectories], axis=0)
    merged_quat = np.concatenate(
        [t.rot.as_quat() if t.has_orientation else RotationSet.identity(len(t)).as_quat() for t in trajectories],
        axis=0,
    )
    has_rot = [t.has_orientation for t in trajectories]
    merged_timestamps = np.concatenate([t.tstamps for t in trajectories], axis=0)

    merged = Trajectory(
        name="Merged",
        tstamps=merged_timestamps,
        pos=PointSet(xyz=merged_xyz, epsg=epsg),
        rot=RotationSet.from_quat(merged_quat) if any(has_rot) else None,
    )

    merged.apply_index(np.argsort(merged.tstamps), inplace=True)
    return merged
