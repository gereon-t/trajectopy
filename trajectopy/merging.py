"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import logging
from typing import List

import numpy as np

from trajectopy.pointset import PointSet
from trajectopy.rotationset import RotationSet
from trajectopy.trajectory import Trajectory

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


def average_trajectories(trajectories: List[Trajectory]) -> Trajectory:
    """
    Averages a list of trajectories into one trajectory.

    This function ignores EPSG codes and averages trajectories.

    Args:
        list[Trajectory]: List of trajectories to average.

    Returns:
        Trajectory: Averaged trajectory.

    """
    lengths = set(len(t) for t in trajectories)

    if len(lengths) != 1:
        raise ValueError(
            "Trajectories must have the same length to be averaged. Ensure this using trajectory matching."
        )

    xyz_stack = np.stack([t.pos.xyz for t in trajectories], axis=0)

    rots = [t.rot.as_quat() for t in trajectories if t.has_orientation]
    mean_rots = []
    for i in range(len(trajectories[0])):
        rotations = []
        for rot in rots:
            rotations.append(rot[i])

        if not rotations:
            continue

        mean_rots.append(RotationSet.from_quat(np.array(rotations)).mean().as_quat())

    avg_xyz = np.mean(xyz_stack, axis=0)

    avg = Trajectory(
        name="Averaged",
        tstamps=trajectories[0].tstamps,
        pos=PointSet(xyz=avg_xyz, epsg=trajectories[0].pos.epsg),
        rot=RotationSet.from_quat(mean_rots) if mean_rots else None,
    )

    return avg
