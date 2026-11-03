import logging
from typing import Literal, overload

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import breadth_first_order, connected_components, dijkstra, minimum_spanning_tree
from scipy.spatial import KDTree

from trajectopy.core.settings import SortingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.lib.approximation.line_approximation import Line3D
from trajectopy.processing.lib.approximation.voxelizer import Voxelizer
from trajectopy.utils.definitions import Sorting
from trajectopy.utils.geometry import lengths_from_xyz

logger = logging.getLogger(__name__)

# Edge weight used instead of zero for (nearly) identical points, see _edge_matrix.
_MIN_EDGE_WEIGHT = 1e-12


def sort_spatially(
    trajectory: Trajectory, sorting_settings: SortingSettings | None = None, inplace: bool = True
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
    if sorting_settings is None:
        sorting_settings = SortingSettings()

    sort_idx, arc_lengths = _sort_xyz(xyz=trajectory.positions.xyz, settings=sorting_settings)
    trajectory = trajectory.mask(sort_idx, inplace=inplace)
    trajectory.path_lengths = arc_lengths
    trajectory.set_sorting(Sorting.PATH_LENGTH)
    return trajectory


@overload
def divide_into_laps(trajectory: Trajectory, return_lap_indices: Literal[False] = False) -> list[Trajectory]: ...


@overload
def divide_into_laps(
    trajectory: Trajectory, return_lap_indices: Literal[True]
) -> tuple[list[Trajectory], np.ndarray]: ...


def divide_into_laps(
    trajectory: Trajectory, return_lap_indices: bool = False
) -> list[Trajectory] | tuple[list[Trajectory], np.ndarray]:
    """
    Divides the trajectory into laps. This is only useful for trajectories
    that describe a closed loop without intersections.

    Args:
        trajectory (Trajectory): Trajectory to divide.
        return_lap_indices (bool, optional): Whether to return lap indices. Defaults to False.

    Returns:
        List[Trajectory]: List of trajectories, each representing a lap.

    """
    if trajectory.sorting != Sorting.PATH_LENGTH:
        logger.error("Provide a trajectory sorted by path length to divide into laps.")
        raise ValueError("Trajectory must be sorted by path length to divide into laps.")
    else:
        trajectory = trajectory.copy()

    trajectory.set_sorting(Sorting.TIME)
    arc_length_diffs = np.diff(trajectory.path_lengths)
    arc_length_threshold = 0.95 * np.max(trajectory.path_lengths)
    lap_indices = np.r_[0, np.where(np.abs(arc_length_diffs) > arc_length_threshold)[0], len(trajectory)]

    # divide into laps
    laps = []
    for i in range(len(lap_indices) - 1):
        lap = trajectory.mask(np.arange(lap_indices[i], lap_indices[i + 1]), inplace=False)
        lap.set_sorting(Sorting.PATH_LENGTH)
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
    # neighboring points often share the same voxel set; cached only for this voxelizer, i.e. this iteration
    lines: dict[frozenset[str], Line3D | None] = {}
    for i, voxel_set in enumerate(neighboring_voxels):
        if voxel_set not in lines:
            lines[voxel_set] = _line_approximator(voxelizer=voxelizer, voxel_set=voxel_set)
        line = lines[voxel_set]
        mls_approx[i, :] = line.evaluate_at(xyz[i, :]) if line is not None else xyz[i, :]

    avg_point_movement = np.mean(np.sqrt(np.sum(np.power(xyz - mls_approx, 2), axis=1)))
    return mls_approx, avg_point_movement


def _line_approximator(voxelizer: Voxelizer, voxel_set: frozenset[str]) -> Line3D | None:
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


def _sort_xyz(xyz: np.ndarray, settings: SortingSettings | None = None) -> tuple[list[int], np.ndarray]:
    """Reconstructs the spatial sorting of the given points.

    Smooths positions using Moving Least Squares, then
    reconstructs the spatial order using a minimum spanning tree.

    This functionality is only useful if the positions describe a closed
    loop without intersections.

    Args:
        xyz (np.ndarray): unsorted positions
        settings (SortingSettings, optional): sorting settings. Defaults to SortingSettings().

    Returns:
        list: Sort index
        np.ndarray: Arc lengths along the path
    """
    if settings is None:
        settings = SortingSettings()

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
    idx_sort = _mst_sort(xyz=xyz_unsorted)

    # Set start position of lap as the position with the maximum z-value
    idx_sort = _begin_with(idx=idx_sort, begin=int(np.argmax(xyz_unsorted[:, 2])))

    # check if direction is correct
    xyz_sorted = xyz_unsorted[idx_sort, :]
    if _detect_direction(xyz_sorted) != _detect_direction(xyz_unsorted):
        logger.info("Adjusted direction of travel!")
        idx_sort = list(reversed(idx_sort))

    return idx_sort


def _mst_sort(xyz: np.ndarray) -> list[int]:
    """Reconstructs the order of points along a curve using a minimum spanning tree (MST).

    The MST of densely sampled points on a curve follows the curve; noise only adds short side
    branches. The tree is traversed depth-first, starting at one end of its longest path (diameter).
    At every node, the shorter branches are visited first and the deepest one last, so that the
    traversal follows the curve and only makes short detours into side branches. A plain
    breadth-first traversal would instead interleave side branches with the curve.

    Args:
        xyz (np.ndarray): Input points that should be sorted.

    Returns:
        list[int]: Index that establishes a spatial sorting. Contains every point exactly once.
    """
    num_points = len(xyz)
    if num_points <= 2:
        return list(range(num_points))

    mst = minimum_spanning_tree(_neighbor_graph(xyz))
    mst = mst + mst.T  # symmetric, for looking up edge weights from either side

    # the farthest node from any node is an end of the tree's diameter
    root = int(np.argmax(dijkstra(mst, directed=False, indices=0)))
    bfs_order, parents = breadth_first_order(mst, root, directed=False, return_predecessors=True)
    children = bfs_order[1:]
    edge_lengths = np.asarray(mst[children, parents[children]]).ravel()

    # height of each subtree: length of the longest path from its root down to a leaf
    heights = np.zeros(num_points)
    for child, edge_length in zip(children[::-1], edge_lengths[::-1]):
        heights[parents[child]] = max(heights[parents[child]], heights[child] + edge_length)

    # children of each node, deepest first
    by_parent = children[np.lexsort((-heights[children], parents[children]))]
    split = np.flatnonzero(np.diff(parents[by_parent])) + 1
    children_of = {int(parents[group[0]]): group.tolist() for group in np.split(by_parent, split)}

    # depth-first: the deepest child is pushed first and therefore visited last
    order: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(children_of.get(node, ()))

    logger.info("MST sorting complete: %i points.", num_points)
    return order


def _neighbor_graph(xyz: np.ndarray, k_nearest: int = 10) -> csr_array:
    """Sparse, connected graph linking each point to its nearest neighbors, weighted by distance.

    Neighborhoods can miss each other, e.g. for an isolated group of points. Such separate parts are
    connected by their shortest links (Boruvka steps), since otherwise the spanning tree, and with it
    the sorting, would be incomplete.

    Args:
        xyz (np.ndarray): Points.
        k_nearest (int, optional): Number of neighbors per point. Defaults to 10.

    Returns:
        csr_array: Upper triangular adjacency matrix of the undirected graph.
    """
    num_points = len(xyz)
    _, neighbors = KDTree(xyz).query(xyz, k=min(k_nearest + 1, num_points))
    edges = np.column_stack((np.repeat(np.arange(num_points), neighbors.shape[1]), neighbors.ravel()))
    graph = _edge_matrix(xyz, edges)

    num_components, labels = connected_components(graph, directed=False)
    while num_components > 1:
        logger.info("Connecting %i separate groups of points.", num_components)
        bridges = [_shortest_link(xyz, labels, component) for component in range(num_components)]
        edges = np.vstack((edges, bridges))
        graph = _edge_matrix(xyz, edges)
        num_components, labels = connected_components(graph, directed=False)
    return graph


def _shortest_link(xyz: np.ndarray, labels: np.ndarray, component: int) -> tuple[int, int]:
    """Closest pair of points inside and outside of a component of the graph."""
    members = np.flatnonzero(labels == component)
    others = np.flatnonzero(labels != component)
    distances, nearest = KDTree(xyz[others]).query(xyz[members])
    best = int(np.argmin(distances))
    return int(members[best]), int(others[nearest[best]])


def _edge_matrix(xyz: np.ndarray, edges: np.ndarray) -> csr_array:
    """Upper triangular adjacency matrix with the edge lengths as weights (each edge once, no loops)."""
    num_points = len(xyz)
    first, second = np.minimum(edges[:, 0], edges[:, 1]), np.maximum(edges[:, 0], edges[:, 1])
    keys = np.unique(first.astype(np.int64) * num_points + second)
    first, second = np.divmod(keys, num_points)
    first, second = first[first != second], second[first != second]
    weights = np.linalg.norm(xyz[second] - xyz[first], axis=1)
    # scipy treats zero weights as missing edges, which would disconnect duplicate points
    weights = np.maximum(weights, _MIN_EDGE_WEIGHT)
    # scipy < 1.16 keeps 64-bit indices, which its csgraph routines (minimum_spanning_tree) do not accept
    first, second = first.astype(np.int32), second.astype(np.int32)
    return csr_array((weights, (first, second)), shape=(num_points, num_points))


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
