"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import logging
from enum import Enum
from typing import List, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from trajectopy.core.approximation.mls_approximation import mls_iterative
from trajectopy.core.utils import lengths_from_xyz
from trajectopy.settings import SortingSettings

logger = logging.getLogger("root")


class Sorting(str, Enum):
    TIME = "time"
    ARC_LENGTH = "arc_length"

    @classmethod
    def from_str(cls, value: str) -> "Sorting":
        if value == "time":
            return Sorting.TIME
        elif value == "arc_length":
            return Sorting.ARC_LENGTH
        else:
            raise ValueError(f"Unknown sorting method: {value}")


def sort_spatially(
    xyz_unsorted: np.ndarray, settings: SortingSettings = SortingSettings()
) -> Tuple[List[int], np.ndarray]:
    """Reconstructs the spatial sorting of the given points

    Spatially sorts the positions by constructing the
    minimum-spanning-tree of the positions.
    Finally, by performing up to 3 breadth-first-searches
    within the mst, the spatial sorting can be reconstructed

    This functionality is only useful if the positions describe a closed
    loop without intersections.

    This method can also take care of inserting missing points
    and assures that the direction of travel is kept during
    sorting.

    Args:
        xyz_unsorted (np.ndarray): unsorted positions
        settings (SortingSettings, optional): sorting settings. Defaults to SortingSettings().

    Returns:
        list: Sort index
        np.ndarray: Arc lengths along the path
    """
    mls_unsorted = mls_iterative(
        xyz=xyz_unsorted,
        voxel_size=settings.voxel_size,
        k_nearest=settings.k_nearest,
        movement_threshold=settings.movement_threshold,
    )

    sort_index = _sort(xyz_unsorted=mls_unsorted)
    mls_sorted = mls_unsorted[sort_index, :]
    return sort_index, lengths_from_xyz(mls_sorted)


def _sort(xyz_unsorted: np.ndarray) -> List[int]:
    idx_sort = _mst_sorting(xyz=xyz_unsorted)

    # Set start position of lap as the position with the maximum z-value
    idx_sort = _begin_with(idx=idx_sort, begin=int(np.argmax(xyz_unsorted[:, 2])))

    # check if direction is correct
    xyz_sorted = xyz_unsorted[idx_sort, :]
    if detect_direction(xyz_sorted) != detect_direction(xyz_unsorted):
        logger.info("Adjusted direction of travel!")
        idx_sort = np.flipud(idx_sort).tolist()

    return idx_sort


def _mst_sorting(xyz: np.ndarray) -> Tuple[list, list]:
    """Reconstruct the spatial sorting

    Given a set of points inside a numpy array,
    this method reconstruct the spatial sorting of
    those points using a minimum-spanning-tree.
    The minimum-spanning-tree is a cycle-free graph
    that connects all points while minimizing its
    edge lengths.

    Args:
        xyz (np.ndarray): Input points that should be sorted

    Returns:
        Tuple[list, list]: Index that establishes a spatial sorting
                           as well as a list containing the indices
                           of missing points. This can be the case
                           if the delaunay triangulation necessary
                           for the minimum-spanning-tree computation
                           discards some points if they are almost
                           identical. If no points are missing, the
                           list will be empty.
    """
    # create minimum spanning tree
    mst = _compute_mst(xyz)

    # mst = nx.minimum_spanning_tree(delaunay, weight='weight', algorithm='prim')
    logger.info("searching for endpoints")

    # find possible candidates for endpoints
    d1_nodes = [n for n in range(len(mst.nodes)) if mst.degree[n] == 1]
    logger.info("found %i nodes of degree 1", len(d1_nodes))

    if len(d1_nodes) == 2:
        end_nodes = d1_nodes
    else:
        # breadth-first search starting from a arbitrary d1-node
        bfs_a = _breadth_first_search(mst, d1_nodes[0])

        # the last visited node is one endpoint
        e_1 = _breadth_first_search(mst, bfs_a[-1])[-1]
        # perform breadth first search again, starting from this node
        e_2 = _breadth_first_search(mst, e_1)[-1]

        # final end nodes
        end_nodes = [e_1, e_2]

    shortest_path_lengths = nx.shortest_path_length(mst, end_nodes[0], end_nodes[1], weight="weight")
    logger.info("found minimum path length: %.3f m", shortest_path_lengths)

    # breadth-first-search through mst to reconstruct the order
    logger.info("reconstructing point order")
    return _breadth_first_search(mst, root=end_nodes[0])


def _begin_with(idx: list, begin: int) -> list:
    """Reorganizes list so that 'begin' is the first
    value in the list

    Args:
        idx (list): index that should be reorganized
        begin (int): value of the new first element

    Returns:
        list: reorganized list whose first value is now
              'begin'
    """
    idx_start = idx.index(begin)
    idx_max_sort = idx[idx_start:]
    idx_max_sort.extend(idx[:idx_start])
    return idx_max_sort


def _compute_mst(xyz: np.ndarray) -> nx.Graph:
    """
    Function that computes a Minimum-Spanning-Tree using the
    robust scipy.spatial.Delaunay implementation.

    This approach correctly includes all points.

    Args:
        xyz (np.ndarray): 2d or 3d positions used for mst computation.

    Returns:
        nx.Graph: networkx.Graph object of the mst.
    """
    num_points = len(xyz)
    if num_points < 3:
        mst = nx.Graph()
        if num_points == 2:
            cost = np.linalg.norm(xyz[1] - xyz[0])
            mst.add_weighted_edges_from([(0, 1, cost)])
        return mst

    logger.info("Building Delaunay triangulation...")

    tri = Delaunay(xyz[:, :2])

    # tri.simplices gives the indices of the points forming each triangle.
    # We extract the edges from these triangles.
    # e.g., triangle (p1, p2, p3) has edges (p1,p2), (p2,p3), (p3,p1)
    edges = np.vstack((tri.simplices[:, :2], tri.simplices[:, 1:], tri.simplices[:, [0, 2]]))

    # Sort and remove duplicate edges
    edges = np.sort(edges, axis=1)
    unique_edges = np.unique(edges, axis=0)

    # Calculate the 3D distance for each edge's weight
    e_diffs = xyz[unique_edges[:, 1]] - xyz[unique_edges[:, 0]]
    e_costs = np.linalg.norm(e_diffs, axis=1)

    logger.info("Computing minimum spanning tree...")

    delaunay_graph = nx.Graph()
    delaunay_graph.add_nodes_from(range(num_points))
    delaunay_graph.add_weighted_edges_from(np.c_[unique_edges, e_costs])

    mst = nx.minimum_spanning_tree(delaunay_graph)
    logger.info("MST computation complete. All %i points were included.", num_points)
    return mst


def _breadth_first_search(graph: nx.Graph, root: int) -> list:
    """Performs a breadth first search

    Args:
        graph (nx.Graph): networkx.Graph object
        root (int): index of starting node for breadth-first-search

    Returns:
        list: list of visited nodes
    """
    edges = nx.bfs_edges(graph, root)
    return [root] + [int(v) for _, v in edges]


def detect_direction(xyz: np.ndarray) -> int:
    """Detects the direction of travel using polar coordinates

    This only works if the points describe a somewhat circular
    trajectory which circulates around the origin of the
    coordinate frame.

    Args:
        xyz (np.ndarray): input positions

    Returns:
        int: -1 / 1 indicating direction of travel
    """
    th = np.arctan2(xyz[:, 1], xyz[:, 0])
    pSign = len(np.where(np.diff(th) > 0)[0])
    nSign = len(np.where(np.diff(th) < 0)[0])

    return np.sign(pSign - nSign)


def complete_lap_dist(xyz, dist_th: float = 0.5) -> np.bool_:
    """
    Function to determine if lap is complete
    A lap is considered as complete, if the distance between
    the starting point and the end point is below a specified
    distance
    """

    first = xyz[0, :]
    last = xyz[-1, :]

    diff = first - last
    dist = np.linalg.norm(diff)

    return dist < dist_th
