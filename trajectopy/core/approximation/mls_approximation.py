"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from functools import lru_cache
from typing import FrozenSet, Tuple, Union

import numpy as np

from trajectopy.core.utils import Line3D

from .voxelizer import Voxelizer

# logger configuration
logger = logging.getLogger("root")


def mls_iterative(
    xyz: np.ndarray,
    voxel_size: float = 0.05,
    k_nearest: int = 4,
    movement_threshold: float = 0.005,
) -> np.ndarray:
    """Performs the mls approximation iteratively

    This method approximates the neighborhood of a point
    using a 3d line. Neighborhoods are defined using voxels.
    The mls approximation is repeatetly applied to the result
    from the previous iteration until the average point movement
    falls below a user defined threshold (movement_threshold).

    Args:
        xyz (np.ndarray): Input points that should be approximated
        voxel_size (float): length of one voxel side
        k_nearest (int): number of voxels that define a neighborhood
        movement_threshold (float, optional): Threshold that defines
                                              when to stop iterating.
                                              When the average point
                                              movement is below of the
                                              pointsDefaults to 0.005.

    Returns:
        np.ndarray: Approximated positions
    """
    cnt = 1
    avg_point_movement = np.Inf
    while avg_point_movement > movement_threshold:
        logger.info("Iteration %i ... ", cnt)
        xyz, avg_point_movement = mls_single(xyz=xyz, voxel_size=voxel_size, k_nearest=k_nearest)
        cnt += 1
        logger.info("... done! Average point movement: %.4f m", avg_point_movement)
        if cnt > 10:
            logger.warning("Maximum number of iterations reached!")
            break
    return xyz


def mls_single(xyz: np.ndarray, voxel_size: float, k_nearest: int) -> Tuple[np.ndarray, float]:
    """Performs the mls approximation iteratively

    This method approximates the neighborhood of a point
    using a 3d line. Neighborhoods are defined using voxels.

    Args:
        xyz (np.ndarray): Input points that should be approximated
        voxel_size (float): length of one voxel side
        k_nearest (int): number of voxels that define a neighborhood

    Returns:
        Tuple[np.ndarray, float]: Approximated positions, average point movement
    """
    voxelizer = Voxelizer(xyz, voxel_size=voxel_size)
    neighboring_voxels = voxelizer.k_nearest_query(xyz, k_nearest=k_nearest)
    mls_approx = np.zeros(xyz.shape, dtype=float)
    for i, voxel_set in enumerate(neighboring_voxels):
        line = cached_line_approximator(voxelizer=voxelizer, voxel_set=voxel_set)
        mls_approx[i, :] = line.evaluate_at(xyz[i, :]) if line is not None else xyz[i, :]

    avg_point_movement = np.mean(np.sqrt(np.sum(np.power(xyz - mls_approx, 2), axis=1)))
    return mls_approx, avg_point_movement


@lru_cache(maxsize=None)
def cached_line_approximator(voxelizer: Voxelizer, voxel_set: FrozenSet[str]) -> Union[Line3D, None]:
    """
    Approximates a 3D line from a set of points and returns it as a Line3D object.
    If the set contains only one point, returns None.

    Args:
        voxelizer (Voxelizer): A Voxelizer object used to extract points from the voxel set.
        voxel_set (frozenset[str]): A set of voxel indices.

    Returns:
        Union[Line3D, None]: A Line3D object representing the 3D line approximated from the points in the voxel set,
        or None if the set contains only one point.
    """
    points = voxelizer.points_from_voxel_set(voxel_set)
    return Line3D.from_points(points) if len(points) > 1 else None
