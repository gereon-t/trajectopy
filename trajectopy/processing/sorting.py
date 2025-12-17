import logging
from functools import lru_cache

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from trajectopy.core.settings import SortingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.lib.approximation.line_approximation import Line3D
from trajectopy.processing.lib.approximation.voxelizer import Voxelizer
from trajectopy.utils.common import lengths_from_xyz
from trajectopy.utils.definitions import Sorting

logger = logging.getLogger(__name__)


def sort_spatially(
    trajectory: Trajectory, sorting_settings: SortingSettings = SortingSettings(), inplace: bool = True
) -> Trajectory:
    """
    Sorts the trajectory spatially. This is only useful for trajectories
    that describe a closed loop without intersections.

    Args:
        trajectory (Trajectory): Trajectory to sort.
        sorting_settings (SortingSettings): Sorting settings.
        inplace (bool, optional): Whether to sort the trajectory in-place. Defaults to True.

    Returns:
        Trajectory: Sorted trajectory.

    """
    sort_idx, arc_lengths = _sort_xyz(xyz=trajectory.positions.xyz, settings=sorting_settings)
    arg_sort_sort_idx = np.argsort(sort_idx)
    trajectory = trajectory.mask(sorted(sort_idx), inplace=inplace)
    trajectory.path_lengths = arc_lengths[arg_sort_sort_idx]
    trajectory.sorting = Sorting.PATH_LENGTH
    return trajectory


def divide_into_laps(
    trajectory: Trajectory, sorting_settings: SortingSettings = SortingSettings(), return_lap_indices: bool = False
) -> list[Trajectory] | tuple[list[Trajectory], np.ndarray]:
    """
    Divides the trajectory into laps. This is only useful for trajectories
    that describe a closed loop without intersections.

    Args:
        trajectory (Trajectory): Trajectory to divide.
        sorting_settings (SortingSettings): Sorting settings.
        return_lap_indices (bool, optional): Whether to return lap indices. Defaults to False.

    Returns:
        List[Trajectory]: List of trajectories, each representing a lap.

    """
    if trajectory.sorting != Sorting.PATH_LENGTH:
        trajectory = sort_spatially(trajectory=trajectory, sorting_settings=sorting_settings, inplace=False)
    else:
        trajectory = trajectory.copy()

    arc_length_diffs = np.diff(trajectory.path_lengths)
    arc_length_threshold = 0.95 * np.max(trajectory.path_lengths)
    lap_indices = np.r_[0, np.where(np.abs(arc_length_diffs) > arc_length_threshold)[0], len(trajectory)]

    # divide into laps
    laps = []
    for i in range(len(lap_indices) - 1):
        lap = trajectory.mask(np.arange(lap_indices[i], lap_indices[i + 1]), inplace=False)
        laps.append(lap)

    return laps if (not return_lap_indices) else (laps, lap_indices)


def _mls_single(xyz: np.ndarray, voxel_size: float, k_nearest: int) -> tuple[np.ndarray, float]:
    """Performs the MLS approximation without iteration.

    This method approximates the neighborhood of a point using a 3d line.
    Neighborhoods are defined using voxels.

    Args:
        xyz (np.ndarray): Input points that should be approximated.
        voxel_size (float): Length of one voxel side.
        k_nearest (int): Number of voxels that define a neighborhood.

    Returns:
        Tuple[np.ndarray, float]: Approximated positions and average point movement.
    """
    voxelizer = Voxelizer(xyz, voxel_size=voxel_size)
    neighboring_voxels = voxelizer.k_nearest_query(xyz, k_nearest=k_nearest)
    mls_approx = np.zeros(xyz.shape, dtype=float)
    for i, voxel_set in enumerate(neighboring_voxels):
        line = _cached_line_approximator(voxelizer=voxelizer, voxel_set=voxel_set)
        mls_approx[i, :] = line.evaluate_at(xyz[i, :]) if line is not None else xyz[i, :]

    avg_point_movement = np.mean(np.sqrt(np.sum(np.power(xyz - mls_approx, 2), axis=1)))
    return mls_approx, avg_point_movement


@lru_cache(maxsize=None)
def _cached_line_approximator(voxelizer: Voxelizer, voxel_set: frozenset[str]) -> Line3D | None:
    """Approximates a 3D line from a set of points and returns it as a Line3D object.

    If the set contains only one point, returns None.

    Args:
        voxelizer (Voxelizer): A Voxelizer object used to extract points from the voxel set.
        voxel_set (frozenset[str]): A set of voxel indices.

    Returns:
        Union[Line3D, None]: A Line3D object representing the 3D line approximated from the
            points in the voxel set, or None if the set contains only one point.
    """
    points = voxelizer.points_from_voxel_set(voxel_set)
    return Line3D.from_points(points) if len(points) > 1 else None


def _moving_least_squares(
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
    avg_point_movement = np.inf
    while avg_point_movement > movement_threshold:
        logger.info("Iteration %i ... ", cnt)
        xyz, avg_point_movement = _mls_single(xyz=xyz, voxel_size=voxel_size, k_nearest=k_nearest)
        cnt += 1
        logger.info("... done! Average point movement: %.4f m", avg_point_movement)
        if cnt > 10:
            logger.warning("Maximum number of iterations reached!")
            break
    return xyz


def _sort_xyz(xyz: np.ndarray, settings: SortingSettings = SortingSettings()) -> tuple[list[int], np.ndarray]:
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
        xyz (np.ndarray): unsorted positions
        settings (SortingSettings, optional): sorting settings. Defaults to SortingSettings().

    Returns:
        list: Sort index
        np.ndarray: Arc lengths along the path
    """
    mls_unsorted = _moving_least_squares(
        xyz=xyz,
        voxel_size=settings.voxel_size,
        k_nearest=settings.k_nearest,
        movement_threshold=settings.movement_threshold,
    )

    sort_index = _create_sorting_index(xyz_unsorted=mls_unsorted)
    mls_sorted = mls_unsorted[sort_index, :]
    return sort_index, lengths_from_xyz(mls_sorted)


def _create_sorting_index(xyz_unsorted: np.ndarray) -> list[int]:
    idx_sort = _mst_sorting(xyz=xyz_unsorted)

    # Set start position of lap as the position with the maximum z-value
    idx_sort = _begin_with(idx=idx_sort, begin=int(np.argmax(xyz_unsorted[:, 2])))

    # check if direction is correct
    xyz_sorted = xyz_unsorted[idx_sort, :]
    if _detect_direction(xyz_sorted) != _detect_direction(xyz_unsorted):
        logger.info("Adjusted direction of travel!")
        idx_sort = np.flipud(idx_sort).tolist()

    return idx_sort


def _mst_sorting(xyz: np.ndarray) -> tuple[list, list]:
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


def _detect_direction(xyz: np.ndarray) -> int:
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
