"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.spatial import KDTree


@dataclass
class Voxel:
    """A class representing a voxel.

    Attributes:
        id (int): The unique identifier of the voxel.
        size (float): The size of the voxel.
        points (list): A list of points contained within the voxel.

    Properties:
        mean_point (np.ndarray): The mean point of the voxel.
        num_points (int): The number of points contained within the voxel.
        to_numpy (np.ndarray): The points contained within the voxel as a numpy array.
    """

    id: str
    size: float
    points: list

    @property
    def mean_point(self) -> np.ndarray:
        return np.mean(np.array(self.points, dtype=float), axis=0)

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def to_numpy(self) -> np.ndarray:
        return np.array(self.points, dtype=float)


class Voxelizer:
    """A class for voxelizing point clouds.

    Attributes:
        voxels (dict[str, Voxel]): A dictionary containing voxels, each with a unique id.
        kd_tree (scipy.spatial.KDTree): A KDTree containing the mean points of each voxel.

    Methods:
        ball_query(xyz: np.ndarray, r: float) -> list[frozenset[str]]: Performs a kd-ball-query within the voxels.
        index_to_id(index: int) -> int: Returns the id of the voxel at the given index.
    """

    def __init__(self, xyz: np.ndarray, voxel_size: float = 0.05) -> None:
        self.voxels = self._create_voxels(xyz, voxel_size)
        self.kd_tree = KDTree(self.mean_points)

    def _create_voxels(self, xyz: np.ndarray, voxel_size: float) -> dict[str, Voxel]:
        """Divides points into voxels of size voxel_size

        Args:
            xyz (np.ndarray): nx3 positions
            voxel_size (float): grid size

        Returns:
            dict[str, Voxel]: dictionary containing voxels, each with an unique id
        """
        grid_x = np.arange(min(xyz[:, 0]), max(xyz[:, 0]) + voxel_size, voxel_size)
        grid_y = np.arange(min(xyz[:, 1]), max(xyz[:, 1]) + voxel_size, voxel_size)
        grid_z = np.arange(min(xyz[:, 2]), max(xyz[:, 2]) + voxel_size, voxel_size)

        x_bin_ids = np.searchsorted(grid_x, xyz[:, 0])
        y_bin_ids = np.searchsorted(grid_y, xyz[:, 1])
        z_bin_ids = np.searchsorted(grid_z, xyz[:, 2])

        voxels: dict[str, Voxel] = {}
        for i, (x_id, y_id, z_id) in enumerate(zip(x_bin_ids, y_bin_ids, z_bin_ids)):
            voxel_id = f"{x_id}{y_id}{z_id}"
            if voxel_id not in voxels:
                voxels[voxel_id] = Voxel(id=voxel_id, size=voxel_size, points=[])
            voxels[voxel_id].points.append(xyz[i, :])

        return voxels

    @property
    def mean_points(self) -> np.ndarray:
        return np.array([voxel.mean_point for _, voxel in self.voxels.items()])

    @cached_property
    def voxel_ids(self) -> list[str]:
        return [voxel.id for _, voxel in self.voxels.items()]

    def index_to_id(self, index: int) -> str:
        return self.voxel_ids[index]

    def ball_query(self, xyz: np.ndarray, r: float) -> list[frozenset[str]]:
        """Performs a kd-ball-query within the voxels

        Returns a list of frozensets containing the
        voxel ids vor neighboring voxels.

        Args:
            xyz (np.ndarray): query positions
            r (float): radius in [m]

        Returns:
            list[frozenset[str]]: list of frozensets containing voxel ids.
        """
        neighbor_voxels = self.kd_tree.query_ball_point(xyz, r=r)
        return [frozenset(self.index_to_id(index) for index in voxel_list) for voxel_list in neighbor_voxels]

    def k_nearest_query(self, xyz: np.ndarray, k_nearest: int) -> list[frozenset[str]]:
        """Performs a k-nearest-query within the voxels

        Returns a list of frozensets containing the
        voxel ids vor neighboring voxels.

        Args:
            xyz (np.ndarray): query positions
            k_nearest (int): k-nearest neighbors

        Returns:
            list[frozenset[str]]: list of frozensets containing voxel ids.
        """
        _, neighbor_voxels = self.kd_tree.query(xyz, k=k_nearest)
        return [frozenset(self.index_to_id(index) for index in voxel_list) for voxel_list in neighbor_voxels]

    def points_from_voxel_set(self, voxel_set: frozenset[str]) -> np.ndarray:
        return (
            np.row_stack([self.voxels[voxel].points for voxel in voxel_set])
            if voxel_set
            else np.array([], dtype=float)
        )
