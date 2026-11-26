import copy
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from trajectopy.core.positions import Positions
from trajectopy.exceptions import TrajectoryError
from trajectopy.utils.definitions import UNIX_TIME_THRESHOLD, Sorting
from trajectopy.utils.deprecation import warn_deprecated
from trajectopy.utils.geometry import common_span, gradient_3d, lengths_from_xyz
from trajectopy.utils.rotations import rotation_from_euler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pose:
    """A single pose of a trajectory, as returned by indexing a trajectory with an integer or iterating over it.

    Attributes:
        timestamp (float): Time of the pose in seconds.
        path_length (float): Path length from the start of the trajectory in meters.
        position (tuple[float, float, float]): Position (x, y, z).
        rpy (tuple[float, float, float]): Orientation as roll, pitch and yaw angles in radians (zeros if the
            trajectory has no orientations).
        velocity (tuple[float, float, float]): Velocity along x, y and z in meters per second.
        epsg (int): EPSG code of the position; 0 for a local frame.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> pose = reference[1]
        >>> pose.timestamp, pose.epsg
        (1.0, 0)
    """

    timestamp: float
    path_length: float
    position: tuple[float, float, float]
    rpy: tuple[float, float, float]
    velocity: tuple[float, float, float]
    epsg: int = 0


class Trajectory:
    """Class representing a trajectory containing synchronized position, orientation, and time data.

    Attributes:
        positions (Positions): Container for spatial coordinates and coordinate reference system (EPSG) data.
        rotations (Rotation): Container for orientation data (quaternions), or None if not provided.
        timestamps (np.ndarray): A 1D array of timestamps corresponding to each pose.
        name (str): An identifier string for the trajectory.
        path_lengths (np.ndarray): A 1D array of cumulative path lengths starting from zero.
        sorting (Sorting): The current sorting strategy (Sorting.TIME or Sorting.PATH_LENGTH).
        position_std (np.ndarray | None): Standard deviations of the positions along x, y and z in meters,
            shape (N, 3), or None if unknown. The alignment weights the positions by them.

    See Also:
        [`trajectopy.Trajectory.from_file`][trajectopy.Trajectory.from_file]: loads a trajectory from a file.

    Examples:
        >>> import numpy as np
        >>> import trajectopy as tpy
        >>> timestamps = np.arange(0.0, 10.0, 0.1)
        >>> xyz = np.column_stack([timestamps, np.sin(timestamps), np.zeros_like(timestamps)])
        >>> trajectory = tpy.Trajectory(xyz, timestamps=timestamps, name="Curve")
        >>> trajectory
        <Trajectory 'Curve': 100 poses without orientations, ... m, EPSG 0, sorted by time>
        >>> trajectory[10].timestamp
        1.0
        >>> len(trajectory[trajectory.timestamps >= 5.0])
        50
    """

    _counter = 1

    def __init__(
        self,
        positions: Positions | ArrayLike,
        rotations: Rotation | ArrayLike | None = None,
        timestamps: ArrayLike | None = None,
        *,
        name: str = "",
        path_lengths: np.ndarray | None = None,
        velocity_xyz: np.ndarray | None = None,
        sorting: Sorting | str = Sorting.TIME,
        position_std: np.ndarray | None = None,
    ) -> None:
        """
        Initialize a Trajectory object.

        The poses are put into chronological order first; of poses with the same timestamp, only the first is
        kept (with a warning in the log). Then they are ordered according to ``sorting``.

        Args:
            positions (Positions | ArrayLike): Positions, either as a Positions container (which holds the EPSG
                code) or as an array of shape (N, 3) in a local frame (EPSG 0).
            rotations (Rotation | ArrayLike | None, optional): Orientations, either as Rotation or as an array of
                quaternions (x, y, z, w) of shape (N, 4). Defaults to None (no orientations).
            timestamps (ArrayLike | None, optional): Timestamps in seconds. If None, a range index is used.
            name (str, optional): Name of the trajectory. Defaults to generic counter name.
            path_lengths (Union[np.ndarray, None], optional): Pre-calculated path lengths. If None or of the wrong
                length, they are computed from xyz.
            velocity_xyz (Union[np.ndarray, None], optional): Pre-calculated 3D velocities. If None or of the wrong
                length, they are computed from the positions and timestamps.
            sorting (Sorting | str, optional): Order of the poses, by time or by path length. Defaults to
                Sorting.TIME.
            position_std (np.ndarray | None, optional): Standard deviations of the positions along x, y and z in
                meters, shape (N, 3). Defaults to None (unknown).

        Raises:
            TrajectoryError: If the number of positions and rotations or timestamps do not match.
        """
        if not isinstance(positions, Positions):
            positions = Positions(xyz=np.asarray(positions, dtype=float), epsg=0)
        orientations: Rotation | None = (
            None
            if rotations is None
            else rotations
            if isinstance(rotations, Rotation)
            else Rotation.from_quat(np.asarray(rotations, dtype=float))
        )
        if orientations is not None and orientations.single:
            orientations = Rotation.concatenate([orientations])  # one orientation per pose

        # check dimensions
        if orientations is not None and len(positions) != len(orientations):
            raise TrajectoryError(
                f"Number of positions ({len(positions)}) and rotations ({len(orientations)}) do not match!"
            )

        self._sorting = Sorting(sorting)

        # pose
        self.positions = positions
        self.rotations = orientations

        # float, since timestamps are shifted and interpolated
        self.timestamps = (
            np.arange(0, len(positions), dtype=float) if timestamps is None else np.asarray(timestamps, dtype=float)
        )

        num_poses = len(self.timestamps)
        if num_poses != len(positions):
            raise TrajectoryError(f"Number of positions ({len(positions)}) and timestamps ({num_poses}) do not match!")
        velocity_xyz = (
            np.asarray(velocity_xyz) if velocity_xyz is not None and len(velocity_xyz) == num_poses else None
        )
        path_lengths = (
            np.asarray(path_lengths) if path_lengths is not None and len(path_lengths) == num_poses else None
        )
        self.position_std = _valid_position_std(position_std, num_poses)

        # velocities and path lengths are derived from consecutive poses, so the data has to be in chronological
        # order with strictly increasing timestamps first
        chronological = np.argsort(self.timestamps, kind="stable")
        first_of_epoch = chronological[np.diff(self.timestamps[chronological], prepend=np.nan) != 0]

        if len(first_of_epoch) != num_poses:
            logger.warning(
                "Timestamps contain duplicates. Only the first occurrence of each unique timestamp will be kept. "
                "This may lead to loss of data if duplicates are present."
            )

        if not np.array_equal(first_of_epoch, np.arange(num_poses)):
            self.timestamps = self.timestamps[first_of_epoch]
            self.positions.xyz = self.positions.xyz[first_of_epoch, :]

            if self.rotations is not None:
                self.rotations = self.rotations[first_of_epoch]
            if velocity_xyz is not None:
                velocity_xyz = velocity_xyz[first_of_epoch]
            if path_lengths is not None:
                path_lengths = path_lengths[first_of_epoch]
            if self.position_std is not None:
                self.position_std = self.position_std[first_of_epoch]

        if velocity_xyz is not None and len(velocity_xyz) == len(self.positions):
            self._velocity_xyz = velocity_xyz
        else:
            self._velocity_xyz = gradient_3d(xyz=self.positions.xyz, tstamps=self.timestamps)
            logger.info("Speeds were not provided or had wrong dimensions. Speeds were computed instead.")

        if path_lengths is not None and len(path_lengths) == len(self.positions):
            self.path_lengths = path_lengths
        else:
            self.path_lengths = self.init_path_lengths()
            logger.info("Path lengths were not provided or had wrong dimensions. Path lengths were computed instead.")

        self.name = name or f"Trajectory {Trajectory._counter}"

        # Sort data according to initial sorting strategy
        self._apply_sorting()

        Trajectory._counter += 1

    @classmethod
    def from_file(
        cls,
        filename: str | os.PathLike,
        io_stream: bool | None = None,
        *,
        format: str = "auto",  # noqa: A002
    ) -> "Trajectory":
        """Create a trajectory instance from a file.

        The file is expected to be a CSV-like format. It handles extraction of
        timestamps, xyz positions, rotations, path lengths, and velocities via `ascii` trajectory reader.
        Files in the TUM, KITTI and EuRoC formats are recognized automatically, see
        :func:`trajectopy.io.formats.detect_format`.

        Args:
            filename (str | os.PathLike): Path to the file.
            io_stream (bool, optional): Deprecated, use :meth:`from_string` to parse the content of a file.
            format (str, optional): File format: ``auto`` (default), ``trajectopy``, ``tum``, ``kitti`` or
                ``euroc``.

        Returns:
            Trajectory: The loaded trajectory object.

        Examples:
            >>> import trajectopy as tpy
            >>> trajectory = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> trajectory.name, len(trajectory)
            ('KITTI Ground Truth', 4541)
        """
        from trajectopy.io.formats import read_trajectory

        if io_stream is not None:
            warn_deprecated("The argument 'io_stream' of Trajectory.from_file() is deprecated, use from_string().")
            if io_stream:
                return cls.from_string(str(filename))
        return read_trajectory(filename, format=format)

    @classmethod
    def from_string(cls, text: str) -> "Trajectory":
        """Creates a trajectory from the content of a file in the trajectopy format (see :meth:`to_string`).

        Args:
            text (str): Content of a trajectory file with header lines.

        Returns:
            Trajectory: The parsed trajectory.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> text = reference.to_string()
            >>> tpy.Trajectory.from_string(text).equals(reference)
            True
        """
        from trajectopy.io.trajectory_io import read_trajectory_string

        return read_trajectory_string(text)

    @classmethod
    def from_arrays(
        cls,
        xyz: np.ndarray,
        quat: np.ndarray | None = None,
        rpy: np.ndarray | None = None,
        epsg: int = 0,
        **kwargs: Any,
    ) -> "Trajectory":
        """Creates a trajectory from numpy arrays.

        Args:
            xyz (np.ndarray): Positions, shape (N, 3).
            quat (np.ndarray | None, optional): Orientations as quaternions (x, y, z, w), shape (N, 4).
            rpy (np.ndarray | None, optional): Orientations as Euler angles (roll, pitch, yaw) in radians,
                shape (N, 3). Provide either ``quat`` or ``rpy``.
            epsg (int, optional): EPSG code of the positions. 0 means a local, unknown system. Defaults to 0.
            **kwargs: Further arguments of the constructor, e.g. ``timestamps`` and ``name``.

        Returns:
            Trajectory: The new trajectory.

        Examples:
            >>> import numpy as np
            >>> import trajectopy as tpy
            >>> xyz = np.zeros((3, 3))
            >>> rpy = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])  # roll, pitch, yaw in radians
            >>> trajectory = tpy.Trajectory.from_arrays(xyz, rpy=rpy, timestamps=[0.0, 1.0, 2.0])
            >>> trajectory.has_orientation
            True
        """

        pos_obj = Positions(xyz=xyz, epsg=epsg)

        rot_obj = None
        if quat is not None and rpy is not None:
            raise TrajectoryError("Provide quat OR rpy, not both.")

        if quat is not None:
            rot_obj = Rotation.from_quat(quat)
        elif rpy is not None:
            rot_obj = rotation_from_euler(seq="xyz", angles=rpy)

        return cls(positions=pos_obj, rotations=rot_obj, **kwargs)

    def __str__(self) -> str:
        """
        Returns a formatted string summary of the trajectory, including name, length, EPSG, and data rate.
        """
        width = 24
        return (
            f"\n _______________________________________________________\n"
            f"| ------------------ Trajectory Info ------------------ |\n"
            f"| Name:                         {self.name:<{width}}|\n"
            f"| Number of poses:              {len(self):<{width}}|\n"
            f"| Orientation available:        {'yes' if self.has_orientation else 'no':<{width}}|\n"
            f"| EPSG:                         {self.positions.epsg:<{width}}|\n"
            f"| Length [m]:                   {self.total_length:<{width}.3f}|\n"
            f"| Data rate [Hz]:               {self.data_rate:<{width}.3f}|\n"
            f"| Function of:                  {self.index_label:<{width}}|\n"
            f"|_______________________________________________________|\n"
        )

    def __repr__(self) -> str:
        orientation = "with orientations" if self.has_orientation else "without orientations"
        return (
            f"<Trajectory {self.name!r}: {len(self)} poses {orientation}, {self.total_length:.3f} m, "
            f"EPSG {self.positions.epsg}, sorted by {self.sorting.value}>"
        )

    @overload
    def __getitem__(self, key: int | np.integer) -> "Pose": ...

    @overload
    def __getitem__(self, key: slice | list | np.ndarray) -> "Trajectory": ...

    def __getitem__(self, key: int | np.integer | slice | list | np.ndarray) -> "Pose | Trajectory":
        """Selects poses like a numpy array.

        An integer returns a single :class:`Pose`. A slice, an integer array or a boolean mask returns a new
        trajectory with the selected poses (the trajectory itself is not modified).
        """
        if isinstance(key, (int, np.integer)):
            index = int(key)
            if not -len(self) <= index < len(self):
                raise IndexError(f"Pose index {index} is out of range for a trajectory with {len(self)} poses.")
            return self._pose(index % len(self))
        if isinstance(key, slice):
            return self.mask(np.arange(len(self))[key])
        return self.mask(np.asarray(key))

    def _pose(self, index: int) -> "Pose":
        rpy = self.rotations[index].as_euler("xyz") if self.rotations is not None else np.zeros(3)
        position, velocity = self.xyz[index], self.velocity_xyz[index]
        return Pose(
            timestamp=float(self.timestamps[index]),
            path_length=float(self.path_lengths[index]),
            position=(float(position[0]), float(position[1]), float(position[2])),
            rpy=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
            velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
            epsg=self.positions.epsg,
        )

    def __len__(self) -> int:
        """Returns the number of poses in the trajectory."""
        return len(self.positions.xyz)

    def __iter__(self) -> Iterator[Pose]:
        """
        Iterates over the trajectory, yielding a pose for each timestamp.

        Yields:
            Pose: A Pose instance for each timestamp.
        """
        rpy = self.rpy if self.has_orientation else np.zeros((len(self), 3))
        for t, l, pos, rot, velocity in zip(self.timestamps, self.path_lengths, self.xyz, rpy, self.velocity_xyz):
            yield Pose(
                timestamp=t,
                path_length=l,
                position=(pos[0], pos[1], pos[2]),
                rpy=(rot[0], rot[1], rot[2]),
                velocity=(velocity[0], velocity[1], velocity[2]),
                epsg=self.positions.epsg,
            )

    def __eq__(self, other: object) -> bool:
        """Checks whether two trajectories have the same name and data, see :meth:`equals`."""
        if not isinstance(other, Trajectory):
            return NotImplemented
        return self.name == other.name and self.equals(other)

    __hash__ = None  # type: ignore[assignment]  # trajectories are mutable

    def equals(self, other: "Trajectory", *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks whether two trajectories hold the same data within a tolerance (the names are not compared).

        Compares positions, orientations, timestamps, path lengths and velocities with :func:`numpy.allclose`.

        Args:
            other (Trajectory): Trajectory to compare with.
            rtol (float, optional): Relative tolerance. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance. Defaults to 1e-8.

        Returns:
            bool: True if the trajectories are equal.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> copy = reference.copy()
            >>> copy.name = "Copy"
            >>> reference.equals(copy), reference == copy
            (True, False)
        """
        if len(self) != len(other) or self.has_orientation != other.has_orientation:
            return False

        def close(a: np.ndarray, b: np.ndarray) -> bool:
            return bool(np.allclose(a, b, rtol=rtol, atol=atol))

        return (
            (not self.has_orientation or close(self.quat, other.quat))
            and close(self.positions.xyz, other.positions.xyz)
            and close(self.timestamps, other.timestamps)
            and close(self.path_lengths, other.path_lengths)
            and close(self.velocity_xyz, other.velocity_xyz)
        )

    def copy(self) -> "Trajectory":
        """Returns a deep copy.

        Returns:
            Trajectory: The copy, which shares no data with this trajectory.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> copy = reference.copy()
            >>> copy.timestamps += 10.0
            >>> reference.timestamps[0], copy.timestamps[0]
            (np.float64(0.0), np.float64(10.0))
        """
        return copy.deepcopy(self)

    def init_path_lengths(self):
        """Computes cumulative path lengths based on Euclidean distances between consecutive local coordinates.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> round(float(reference.init_path_lengths()[-1]), 3)
            3724.187
        """
        if len(self.positions) == 0:
            return np.array([])  # without positions, there is no local frame either
        return lengths_from_xyz(self.positions.to_local(inplace=False).xyz)

    @property
    def has_orientation(self) -> bool:
        """Returns True if valid rotation data is available."""
        return self.rotations is not None and len(self.rotations) > 0

    @property
    def is_unix_time(self) -> bool:
        """
        Checks if the supplied trajectory has (likely) unix timestamps as seconds.
        """
        if len(self.timestamps) == 0:
            return False

        min_time = np.min(self.timestamps)

        if min_time < UNIX_TIME_THRESHOLD:
            return False

        # the latest epoch (not the first array element, e.g. of a spatially sorted trajectory) is not in the future
        return bool(self.time_end <= datetime.now().timestamp())

    @property
    def total_length(self) -> float:
        """Returns the total cumulative path length of the trajectory in meters."""
        return 0.0 if len(self.path_lengths) == 0 else self.path_lengths[-1]

    @property
    def sorting(self) -> Sorting:
        """Returns the current sorting strategy (read-only).

        To change sorting, use `set_sorting()` which physically reorders the data.
        """
        return self._sorting

    def set_sorting(self, sorting: Sorting | str) -> None:
        """Changes the sorting strategy and physically reorders all trajectory data.

        After calling this method, all data arrays (positions, rotations, timestamps,
        path_lengths, velocities) will be reordered according to the new sorting.

        Args:
            sorting (Sorting | str): The new sorting strategy (TIME or PATH_LENGTH).

        Examples:
            >>> import numpy as np
            >>> import trajectopy as tpy
            >>> angle = np.linspace(0.0, 4 * np.pi, 400)  # two laps
            >>> loop = tpy.sort_spatially(tpy.Trajectory(np.column_stack([np.cos(angle), np.sin(angle), 0 * angle])))
            >>> loop.sorting
            <Sorting.PATH_LENGTH: 'path_length'>
            >>> loop.set_sorting(tpy.Sorting.TIME)
            >>> loop.sorting
            <Sorting.TIME: 'time'>
        """
        sorting = Sorting(sorting)
        if sorting == self._sorting:
            return

        self._sorting = sorting
        self._apply_sorting()

    def sort_index(self, sorting: Sorting | str) -> np.ndarray:
        """Returns the index that orders the current data by time or path length.

        The sort is stable, so that data that is already in order (e.g. with equal path lengths of several laps)
        keeps its order and data stored alongside the trajectory, such as deviations, can be reordered alike.

        Args:
            sorting (Sorting): Sorting to order by.

        Returns:
            np.ndarray: Index into the current data.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> reference.sort_index(tpy.Sorting.TIME)[:3]
            array([0, 1, 2])
        """
        key = self.timestamps if Sorting(sorting) == Sorting.TIME else self.path_lengths
        return np.argsort(key, kind="stable")

    def _apply_sorting(self) -> None:
        """Internal method to physically reorder all data according to current sorting."""
        if len(self) == 0:
            return

        sort_index = self.sort_index(self._sorting)

        # Check if already sorted
        if np.array_equal(sort_index, np.arange(len(self))):
            return

        self.timestamps = self.timestamps[sort_index]
        self.path_lengths = self.path_lengths[sort_index]
        self.positions.xyz = self.positions.xyz[sort_index, :]

        if self.rotations is not None:
            self.rotations = self.rotations[sort_index]

        if self._velocity_xyz is not None:
            self._velocity_xyz = self._velocity_xyz[sort_index, :]

        if self.position_std is not None:
            self.position_std = self.position_std[sort_index, :]

    @property
    def index(self) -> np.ndarray:
        """
        Returns the independent variable currently parameterizing the trajectory.
        This is either the Timestamp vector or the Path Length vector, depending on `self.sorting`.
        """
        return self.timestamps if self._sorting == Sorting.TIME else self.path_lengths

    @property
    def datetimes(self) -> np.ndarray:
        """Returns the timestamps as an array of ``datetime64[ns]`` values (interpreted as seconds since epoch)."""
        return pd.to_datetime(self.timestamps, unit="s").to_numpy()

    @property
    def index_unit(self) -> str:
        """Returns the unit string of the current index ('s' for Time, 'm' for Path Length)."""
        return "s" if self.sorting == Sorting.TIME else "m"

    @property
    def index_label(self) -> str:
        """Returns the label string of the current index (e.g., 'time [s]')."""
        return "time [s]" if self.sorting == Sorting.TIME else "path length [m]"

    @property
    def data_rate(self) -> float:
        """Calculates the average data rate (frequency in Hz) based on timestamp differences."""
        if len(self.timestamps) < 2:
            return 0.0
        mean_diff = np.mean(np.diff(np.sort(self.timestamps)))
        if mean_diff == 0:
            return float("inf")
        return 1 / mean_diff

    @property
    def velocity_xyz(self) -> np.ndarray:
        """
        Returns the 3D velocity vectors.
        If not set manually, they are computed via gradient of the positions over time.
        """
        if self._velocity_xyz is not None:
            return self._velocity_xyz

        return gradient_3d(xyz=self.positions.xyz, tstamps=self.timestamps)

    @velocity_xyz.setter
    def velocity_xyz(self, speed_3d: np.ndarray) -> None:
        """Manually sets the 3D velocity vectors."""
        self._velocity_xyz = speed_3d

    @property
    def absolute_velocity(self) -> np.ndarray:
        """Returns the norm (magnitude) of the 3D velocity vectors."""
        return np.linalg.norm(self.velocity_xyz, axis=1)

    @property
    def xyz(self) -> np.ndarray:
        """
        Returns the XYZ coordinates.

        Since trajectory data is always stored in sorted order (by time or arc length),
        this is equivalent to `self.positions.xyz`.
        """
        return self.positions.xyz

    @property
    def quat(self) -> np.ndarray:
        """
        Returns the quaternions. Returns zeros if no rotations are present.
        """
        if self.rotations is None:
            return np.zeros((len(self), 4))

        return self.rotations.as_quat()

    @property
    def rpy(self) -> np.ndarray:
        """
        Returns the Roll-Pitch-Yaw angles (in radians).
        """
        return Rotation.from_quat(self.quat).as_euler(seq="xyz")

    @property
    def se3(self) -> list[np.ndarray]:
        """
        Returns a list of SE3 poses (4x4 homogeneous transformation matrices).
        """
        se3_list: list[np.ndarray] = []
        xyz = self.positions.xyz

        if len(xyz) == 0:
            return se3_list

        matrices = [np.eye(3)] * len(xyz) if self.rotations is None else self.rotations.as_matrix()
        for t, r in zip(xyz, matrices):
            se3 = np.eye(4)
            se3[:3, 3] = t
            se3[:3, :3] = r
            se3_list.append(se3)
        return se3_list

    @se3.setter
    def se3(self, se3_list: list[np.ndarray]) -> None:
        """
        Sets the position and rotation of the trajectory from a list of SE3 poses (4x4 matrices).
        """
        xyz = np.zeros((len(se3_list), 3))
        matrices = np.zeros((len(se3_list), 3, 3))

        for i, pose in enumerate(se3_list):
            xyz[i, :] = pose[:3, 3]
            matrices[i, :, :] = pose[:3, :3]

        self.positions.xyz = xyz
        self.rotations = Rotation.from_matrix(matrices)

    @property
    def time_start(self) -> float:
        """Returns the minimum timestamp of the trajectory."""
        return float(np.min(self.timestamps)) if len(self.timestamps) > 0 else 0.0

    @property
    def time_end(self) -> float:
        """Returns the maximum timestamp of the trajectory."""
        return float(np.max(self.timestamps)) if len(self.timestamps) > 0 else 0.0

    def overlaps_with(self, other: "Trajectory") -> bool:
        """Checks if the time span of this trajectory overlaps with another.

        Args:
            other (Trajectory): The trajectory to compare against.

        Returns:
            bool: True if the time ranges overlap, False otherwise.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
            >>> estimate.overlaps_with(reference)
            True
            >>> estimate.overlaps_with(reference.crop(5000.0, 6000.0))
            False
        """
        if len(self) == 0 or len(other) == 0:
            return False

        return self.time_start <= other.time_end and self.time_end >= other.time_start

    def crop(
        self, index_start: float, index_end: float, *, inverse: bool = False, inplace: bool = False
    ) -> "Trajectory":
        """Crops (or cuts) the trajectory to a window of its index.

        The index is the time for trajectories sorted by time and the path length for spatially sorted
        trajectories (see :attr:`index`).

        Args:
            index_start (float): Start of the window (seconds or meters).
            index_end (float): End of the window (seconds or meters).
            inverse (bool, optional): If True, removes data *inside* the window (cutting).
                                      If False, keeps data *inside* the window (cropping). Defaults to False.
            inplace (bool, optional): If True, modifies self. If False, returns a new instance. Defaults to False.

        Returns:
            Trajectory: The modified or new trajectory instance.

        Examples:
            >>> import trajectopy as tpy
            >>> trajectory = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> len(trajectory.crop(0.0, 99.0))
            100
            >>> len(trajectory.crop(0.0, 99.0, inverse=True))
            4441
        """
        # filter to index_start and index_end
        if inverse:
            filt = [not index_start <= idx <= index_end for idx in self.index]
        else:
            filt = [index_start <= idx <= index_end for idx in self.index]

        return self.mask(mask=filt, inplace=inplace)

    def intersect(self, index: np.ndarray, *, max_gap_size: float = 10.0, inplace: bool = False) -> "Trajectory":
        """Filters the trajectory to overlap with a reference timestamp vector.

        This method finds the common time span between self and the reference `timestamps`,
        crops self to that span, and then filters points that are either exact matches
        or exist within valid gaps defined by `max_gap_size`.

        Args:
            index (np.ndarray): The reference indices to intersect with.
            max_gap_size (float, optional): The maximum allowed time gap (in seconds or meters) between
                                            reference indices to include trajectory points. Defaults to 10.0.
            inplace (bool, optional): If True, modifies self. Defaults to False.

        Raises:
            ValueError: If the index spans do not overlap.

        Returns:
            Trajectory: The intersected trajectory.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> import numpy as np
            >>> len(reference.intersect(np.arange(0.0, 100.0, 1.0)))
            100
        """
        traj_self = self if inplace else self.copy()
        index_span = common_span(index1=index, index2=traj_self.index)

        if index_span is None:
            raise ValueError("intersect_both: Indices do not overlap!")

        traj_self.crop(index_start=index_span[0], index_end=index_span[1], inplace=True)

        index_sorted = np.sort(index)
        filter_mask = np.ones(len(traj_self), dtype=bool)

        # find where gaps are too large
        gaps = np.diff(index_sorted) > max_gap_size
        gap_starts = index_sorted[:-1][gaps]
        gap_ends = index_sorted[1:][gaps]

        # the index (time or path length) is in ascending order, see _apply_sorting
        for g_start, g_end in zip(gap_starts, gap_ends):
            idx_start = np.searchsorted(traj_self.index, g_start, side="right")
            idx_end = np.searchsorted(traj_self.index, g_end, side="left")

            # keep one point after gap start and one point before gap end for interpolation
            if idx_end - 1 > idx_start:
                filter_mask[idx_start + 1 : idx_end - 1] = False

        filter_index = np.where(filter_mask)[0]
        traj_self.mask(np.array(filter_index, dtype=int), inplace=True)

        return traj_self

    def mask(self, mask: list | np.ndarray, *, inplace: bool = False) -> "Trajectory":
        """Applies a boolean mask or index array to filter all trajectory components.

        Filtered components include: timestamps, positions, rotations, path lengths, and velocities.

        Args:
            mask (Union[list, np.ndarray]): Boolean array or list of indices to keep.
            inplace (bool, optional): If True, modifies self. Defaults to False.

        Returns:
            Trajectory: The masked trajectory.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> len(reference.mask(reference.timestamps < 10.0))
            10
            >>> len(reference.mask([0, 1, 2]))
            3
        """
        traj_self = self if inplace else self.copy()

        traj_self.timestamps = traj_self.timestamps[mask]
        traj_self.positions.xyz = traj_self.positions.xyz[mask, :]

        if traj_self.rotations is not None:
            rotations = traj_self.rotations[np.asarray(mask)]
            traj_self.rotations = rotations if len(rotations) > 0 else None

        traj_self.path_lengths = traj_self.path_lengths[mask]

        # only stored velocities are masked, derived ones follow from the masked positions
        if traj_self._velocity_xyz is not None:
            traj_self._velocity_xyz = traj_self._velocity_xyz[mask]

        if traj_self.position_std is not None:
            traj_self.position_std = traj_self.position_std[mask]

        return traj_self

    def transform(self, transformation: np.ndarray, *, inplace: bool = False) -> "Trajectory":
        """Applies a rigid body or similarity transformation to the trajectory poses.

        Velocities and position standard deviations are rotated and scaled alike, path lengths are scaled.
        A trajectory without
        orientations stays without orientations.

        Args:
            transformation (np.ndarray): A 4x4 homogeneous transformation matrix.
            inplace (bool, optional): If True, modifies self. Defaults to False.

        Returns:
            Trajectory: The transformed trajectory.

        See Also:
            [`trajectopy.apply_alignment`][trajectopy.apply_alignment]: applies an estimated alignment.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> import numpy as np
            >>> shift = np.eye(4)
            >>> shift[:3, 3] = [10.0, 0.0, 0.0]
            >>> shifted = reference.transform(shift)
            >>> float(shifted.xyz[0, 0] - reference.xyz[0, 0])
            10.0
        """
        traj_self = self if inplace else self.copy()
        linear = transformation[:3, :3]

        scale = float(np.cbrt(abs(np.linalg.det(linear))))
        traj_self.velocity_xyz = traj_self.velocity_xyz @ linear.T
        traj_self.path_lengths = traj_self.path_lengths * scale
        if traj_self.position_std is not None:
            # the variances of the rotated axes (correlations are not stored)
            rotation = linear / scale if scale > 0 else linear
            traj_self.position_std = np.sqrt(traj_self.position_std**2 @ (rotation**2).T) * scale

        has_orientation = traj_self.has_orientation
        traj_self.se3 = [np.dot(transformation, p) for p in traj_self.se3]
        if not has_orientation:
            traj_self.rotations = None
        return traj_self

    def to_dataframe(self, *, sort_by: str = "") -> pd.DataFrame:
        """Exports the trajectory to a Pandas DataFrame.

        Columns usually include: time, path_length, pos_x, pos_y, pos_z, speed_x, speed_y, speed_z,
        and rotation columns (rot_x/y/z/w) if available.

        Args:
            sort_by (str, optional): Column name to sort by. If empty, uses `self.sorting`.

        Returns:
            pd.DataFrame: A dataframe containing the trajectory data.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> reference.to_dataframe().columns.tolist()[:5]
            ['time', 'path_length', 'pos_x', 'pos_y', 'pos_z']
        """
        sort_by = sort_by or self.sorting
        if self.rotations is not None:
            dataframe = pd.DataFrame(
                np.c_[
                    self.timestamps, self.path_lengths, self.positions.xyz, self.rotations.as_quat(), self.velocity_xyz
                ],
                columns=[
                    "time",
                    "path_length",
                    "pos_x",
                    "pos_y",
                    "pos_z",
                    "rot_x",
                    "rot_y",
                    "rot_z",
                    "rot_w",
                    "speed_x",
                    "speed_y",
                    "speed_z",
                ],
            )
        else:
            dataframe = pd.DataFrame(
                np.c_[self.timestamps, self.path_lengths, self.positions.xyz, self.velocity_xyz],
                columns=["time", "path_length", "pos_x", "pos_y", "pos_z", "speed_x", "speed_y", "speed_z"],
            )

        if self.position_std is not None:
            dataframe[["std_x", "std_y", "std_z"]] = self.position_std

        return dataframe.sort_values(by=sort_by)

    def to_string(self) -> str:
        """Returns the trajectory in the trajectopy ASCII format (see :func:`trajectopy.io.trajectory_to_string`).

        Returns:
            str: Content of a trajectory file, which :meth:`from_string` reads.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> for line in reference.to_string().splitlines()[:5]:
            ...     print(line)
            #epsg 0
            #name KITTI Ground Truth
            #nframe enu
            #sorting time
            #fields t,l,px,py,pz,qx,qy,qz,qw,vx,vy,vz
        """
        from trajectopy.io.trajectory_io import trajectory_to_string

        return trajectory_to_string(self)

    def to_file(
        self,
        filename: str | os.PathLike,
        *,
        mode: str = "w",
        format: str = "auto",  # noqa: A002
    ) -> None:
        """Writes the trajectory to a file.

        Args:
            filename (str | os.PathLike): The output file path.
            mode (str, optional): ``"w"`` (default) overwrites the file, ``"a"`` appends to it.
            format (str, optional): File format: ``auto`` (default), ``trajectopy``, ``tum``, ``kitti`` or
                ``euroc``. ``auto`` picks the format from the file extension (``.tum``, ``.kitti``) and writes
                the trajectopy format (see :meth:`to_string`) otherwise.

        See Also:
            [`trajectopy.io.write_trajectory`][trajectopy.io.write_trajectory]: the function behind this method.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> reference.to_file("reference.traj")
            >>> reference.to_file("reference.tum")  # TUM format
            >>> tpy.Trajectory.from_file("reference.traj").equals(reference)
            True
        """
        from trajectopy.io.formats import write_trajectory

        write_trajectory(self, filename, format=format, mode=mode)

    def to_kml(
        self, filename: str | os.PathLike, *, precision: float = 1e-6, color: str = "#ff0000", **options: Any
    ) -> None:
        """Exports the trajectory to a Google Earth KML file, as a time-stamped track that the time slider of
        Google Earth plays back.

        Requires the trajectory to have a valid EPSG code so it can be converted to WGS84 (EPSG:4326).

        Args:
            filename (str): The output filename (e.g., "track.kml").
            precision (float, optional): Coordinate precision in degrees. Defaults to 1e-6 (about 0.1 m).
            color (str, optional): Color of the track and its points as "#RRGGBB" or "#RRGGBBAA".
                Defaults to red.
            **options: Further options of :func:`trajectopy.io.kml.write_kml`, e.g. ``field="speed"`` (color
                by a field), ``heights="ellipsoidal"`` (draw at the trajectory's heights instead of on the
                ground), ``extrude=True`` or ``line_width``.

        Raises:
            ValueError: If the trajectory does not have a known EPSG code or an option is invalid.

        See Also:
            [`trajectopy.io.write_kml`][trajectopy.io.write_kml]: all options.

        Examples:
            >>> import numpy as np
            >>> import trajectopy as tpy
            >>> xyz = np.column_stack([364000.0 + np.arange(10.0), np.full(10, 5621000.0), np.full(10, 60.0)])
            >>> track = tpy.Trajectory(tpy.Positions(xyz, epsg=25832), timestamps=np.arange(10.0))
            >>> track.to_kml("track.kml", field="speed")
        """
        from trajectopy.io.kml import write_kml

        filename = os.fspath(filename)
        write_kml(self, filename=filename, precision=precision, color=color, **options)


def _valid_position_std(position_std: np.ndarray | None, num_poses: int) -> np.ndarray | None:
    """Position standard deviations of shape (N, 3) with finite, positive values, or None."""
    if position_std is None:
        return None
    position_std = np.asarray(position_std, dtype=float)
    if position_std.shape != (num_poses, 3):
        logger.warning(
            "Ignoring position standard deviations of shape %s (expected (%i, 3)).", position_std.shape, num_poses
        )
        return None
    if not np.all(np.isfinite(position_std) & (position_std > 0)):
        logger.warning("Ignoring position standard deviations since some are not positive or not finite.")
        return None
    return position_std
