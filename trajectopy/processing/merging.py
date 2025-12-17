import logging

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory

logger = logging.getLogger(__name__)


def merge_trajectories(trajectories: list[Trajectory]) -> Trajectory:
    """Merges a list of trajectories into one trajectory.

    This function ignores EPSG codes and merges the trajectories based on their timestamps.
    Therefore, all trajectories should be in the same coordinate system before merging.

    Args:
        trajectories (List[Trajectory]): List of trajectories to merge.

    Returns:
        Trajectory: Merged trajectory.
    """
    epsg_set = {t.positions.epsg for t in trajectories}

    if len(epsg_set) > 1:
        logger.warning(
            "Merging trajectories with different EPSG codes. "
            "This may lead to unexpected results. "
            "Consider reprojecting the trajectories to the same EPSG code."
        )

    epsg = epsg_set.pop()

    merged_xyz = np.concatenate([t.positions.xyz for t in trajectories], axis=0)
    merged_quat = np.concatenate(
        [t.rotations.as_quat() if t.has_orientation else Rotations.identity(len(t)).as_quat() for t in trajectories],
        axis=0,
    )
    has_rot = [t.has_orientation for t in trajectories]
    merged_timestamps = np.concatenate([t.timestamps for t in trajectories], axis=0)

    merged = Trajectory(
        name="Merged",
        timestamps=merged_timestamps,
        positions=Positions(xyz=merged_xyz, epsg=epsg),
        rotations=Rotations.from_quat(merged_quat) if any(has_rot) else None,
    )

    merged.mask(np.argsort(merged.timestamps), inplace=True)
    return merged


def average_trajectories(trajectories: list[Trajectory]) -> Trajectory:
    """Averages a list of trajectories into one trajectory.

    This function ignores EPSG codes and averages trajectories.

    Args:
        trajectories (List[Trajectory]): List of trajectories to average.

    Returns:
        Trajectory: Averaged trajectory.

    Raises:
        ValueError: If trajectories have different lengths.
    """
    lengths = {len(t) for t in trajectories}

    if len(lengths) != 1:
        raise ValueError(
            "Trajectories must have the same length to be averaged. Ensure this using trajectory matching."
        )

    xyz_stack = np.stack([t.positions.xyz for t in trajectories], axis=0)

    rots = [t.rotations.as_quat() for t in trajectories if t.has_orientation]
    mean_rots = []
    for i in range(len(trajectories[0])):
        rotations = []
        rotations.extend(rot[i] for rot in rots)
        if not rotations:
            continue

        mean_rots.append(Rotations.from_quat(np.array(rotations)).mean().as_quat())

    avg_xyz = np.mean(xyz_stack, axis=0)

    return Trajectory(
        name="Averaged",
        timestamps=trajectories[0].timestamps,
        positions=Positions(xyz=avg_xyz, epsg=trajectories[0].positions.epsg),
        rotations=Rotations.from_quat(mean_rots) if mean_rots else None,
    )
