import copy
import io
import logging
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.exceptions import TrajectoryError
from trajectopy.readers import ascii
from trajectopy.utils.common import common_time_span, gradient_3d, lengths_from_xyz
from trajectopy.utils.definitions import UNIX_TIME_THRESHOLD, Sorting

logger = logging.getLogger(__name__)


class Trajectory:
    """Class representing a trajectory containing synchronized position, orientation, and time data.

    Attributes:
        positions (Positions): Container for spatial coordinates and coordinate reference system (EPSG) data.
        rotations (Rotations): Container for orientation data (quaternions), or None if not provided.
        timestamps (np.ndarray): A 1D array of timestamps corresponding to each pose.
        name (str): An identifier string for the trajectory.
        path_lengths (np.ndarray): A 1D array of cumulative path lengths starting from zero.
        sorting (Sorting): The current sorting strategy (Sorting.TIME or Sorting.ARC_LENGTH).
    """

    _counter = 1

    def __init__(
        self,
        positions: Positions,
        rotations: Rotations | None = None,
        timestamps: np.ndarray | None = None,
        name: str = "",
        path_lengths: np.ndarray | None = None,
        velocity_xyz: np.ndarray | None = None,
        sorting: Sorting = Sorting.TIME,
    ) -> None:
        """
        Initialize a Trajectory object.

        Args:
            positions (Positions): Container for spatial coordinates and coordinate reference system (EPSG) data.
            rotations (Rotations | None, optional): Container for orientation data (quaternions). Defaults to None.
            timestamps (np.ndarray | None, optional): Array of timestamps. If None, a range index is used.
            name (str, optional): Name of the trajectory. Defaults to generic counter name.
            path_lengths (Union[np.ndarray, None], optional): Pre-calculated path lengths. If None, they are computed from xyz.
            velocity_xyz (Union[np.ndarray, None], optional): Pre-calculated 3D velocities. If None, they are computed via gradient.
            sorting (Sorting, optional): Definition of the sorting logic (TIME or ARC_LENGTH). Defaults to Sorting.TIME.

        Raises:
            TrajectoryError: If the number of positions and rotations do not match.
        """
        # check dimensions
        if rotations is not None and len(positions) != len(rotations):
            raise TrajectoryError(
                f"Number of positions ({len(positions)}) and rotations ({len(rotations)}) do not match!"
            )

        self.sorting = sorting

        # pose
        self.positions = positions
        self.rotations = rotations

        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps) if timestamps is not None else None

        self.timestamps = np.arange(0, len(positions)) if timestamps is None else timestamps

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

        Trajectory._counter += 1

    @classmethod
    def from_file(cls, filename: str, io_stream: bool = False) -> "Trajectory":
        """
        Create a trajectory instance from a file.

        The file is expected to be a CSV-like format. It handles extraction of
        timestamps, xyz positions, rotations, path lengths, and velocities via `ascii` trajectory reader.

        Args:
            filename (str): Path to the file or string content if io_stream is True.
            io_stream (bool, optional): If True, `filename` is treated as the raw string content
                                        of the file/stream. Defaults to False.

        Returns:
            Trajectory: The loaded trajectory object.
        """
        if io_stream:
            header_data, trajectory_data = ascii.read_string(filename, dtype=object)
        else:
            header_data, trajectory_data = ascii.read_data(filename, dtype=object)

        tstamps = ascii.extract_trajectory_timestamps(header_data=header_data, trajectory_data=trajectory_data)
        positions = ascii.extract_trajectory_positions(header_data=header_data, trajectory_data=trajectory_data)
        path_lengths = ascii.extract_trajectory_path_lengths(header_data=header_data, trajectory_data=trajectory_data)
        velocity_xyz = ascii.extract_trajectory_velocity_xyz(header_data=header_data, trajectory_data=trajectory_data)
        rotations = ascii.extract_trajectory_rotations(header_data=header_data, trajectory_data=trajectory_data)

        return Trajectory(
            timestamps=tstamps,
            positions=positions,
            rotations=rotations,
            name=header_data.name,
            path_lengths=path_lengths,
            velocity_xyz=velocity_xyz,
            sorting=Sorting.from_str(header_data.sorting),
        )

    @classmethod
    def from_arrays(
        cls,
        xyz: np.ndarray,
        quat: np.ndarray | None = None,
        rpy: np.ndarray | None = None,
        epsg: int = 0,
        **kwargs,
    ) -> "Trajectory":
        """Factory: Handles creation from raw numpy arrays."""

        pos_obj = Positions(xyz=xyz, epsg=epsg)

        rot_obj = None
        if quat is not None and rpy is not None:
            raise TrajectoryError("Provide quat OR rpy, not both.")

        if quat is not None:
            rot_obj = Rotations.from_quat(quat)
        elif rpy is not None:
            rot_obj = Rotations.from_euler(seq="xyz", angles=rpy)

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
        return str(self)

    def __len__(self) -> int:
        """Returns the number of poses in the trajectory."""
        return len(self.positions.xyz)

    def __eq__(self, other: "Trajectory") -> bool:
        """
        Checks equality between two trajectories using `np.allclose` for numerical arrays.
        Compares positions, rotations, timestamps, path lengths, velocities, and names.
        """
        if self.rotations is not None and other.rotations is not None:
            rot_equal = np.allclose(self.rotations.as_quat(), other.rotations.as_quat())
        elif self.rotations is None and other.rotations is None:
            rot_equal = True
        else:
            rot_equal = False

        return (
            np.allclose(self.positions.xyz, other.positions.xyz)
            and rot_equal
            and np.allclose(self.timestamps, other.timestamps)
            and np.allclose(self.path_lengths, other.path_lengths)
            and np.allclose(self._velocity_xyz, other._velocity_xyz)
            and self.name == other.name
        )

    def copy(self) -> "Trajectory":
        """Returns a deep copy of the trajectory instance."""
        return copy.deepcopy(self)

    def init_path_lengths(self):
        """Computes cumulative path lengths based on Euclidean distances between consecutive local coordinates."""
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
        min_time = min(self.timestamps)

        if min_time < UNIX_TIME_THRESHOLD:
            return False

        today = datetime.now()
        try:
            converted_date = datetime.fromtimestamp(self.timestamps[0])
        except Exception:
            return False

        if converted_date > today:
            return False

        return True

    @property
    def total_length(self) -> float:
        """Returns the total cumulative path length of the trajectory in meters."""
        return 0.0 if len(self.path_lengths) == 0 else self.path_lengths[-1]

    @property
    def sort_switching_index(self) -> np.ndarray:
        """Returns an array of indices that would switch the current sorting (e.g., unsort the data)."""
        return np.argsort(self.sorting_index)

    @property
    def sorting_index(self) -> np.ndarray:
        """Returns the indices used to sort the trajectory based on the current `sorting` attribute (Time or Path Length)."""
        return np.argsort(self.timestamps) if self.sorting == Sorting.TIME else np.argsort(self.path_lengths)

    @property
    def index(self) -> np.ndarray:
        """
        Returns the independent variable currently parameterizing the trajectory.
        This is either the Timestamp vector or the Path Length vector, depending on `self.sorting`.
        """
        return (
            self.timestamps[self.sorting_index]
            if self.sorting == Sorting.TIME
            else self.path_lengths[self.sorting_index]
        )

    @property
    def datetimes(self) -> np.ndarray:
        """Returns the timestamps converted to Pandas datetime objects (unit='s')."""
        return pd.to_datetime(self.timestamps[self.sorting_index], unit="s")

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
        return 1 / np.mean(np.diff(np.sort(self.timestamps)))

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
        Returns the XYZ coordinates sorted according to the current `sorting` strategy.
        Note: This differs from `self.positions.xyz`, which retains the original order.
        """
        return self.positions.xyz[self.sorting_index]

    @property
    def quat(self) -> np.ndarray:
        """
        Returns the quaternions sorted according to the current `sorting` strategy.
        Returns zeros if no rotations are present.
        """
        if self.rotations is None:
            return np.zeros((len(self), 4))

        return self.rotations.as_quat()[self.sorting_index]

    @property
    def rpy(self) -> np.ndarray:
        """
        Returns the Roll-Pitch-Yaw angles sorted according to the current `sorting` strategy.
        """
        return Rotations.from_quat(self.quat).as_euler(seq="xyz")

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
        self.rotations = Rotations.from_matrix(matrices)

    def overlaps_with(self, other: "Trajectory") -> bool:
        """
        Checks if the time span of this trajectory overlaps with another.

        Args:
            other (Trajectory): The trajectory to compare against.

        Returns:
            bool: True if the time ranges overlap, False otherwise.
        """
        start_test = self.timestamps[0]
        end_test = self.timestamps[-1]
        start_ref = other.timestamps[0]
        end_ref = other.timestamps[-1]

        return (start_test <= end_ref and end_test >= start_ref) or (start_ref <= end_test and end_ref >= start_test)

    def crop(self, t_start: float, t_end: float, inverse: bool = False, inplace: bool = True) -> "Trajectory":
        """
        Crops (or cuts) the trajectory based on a time window.

        Args:
            t_start (float): Start timestamp of the window.
            t_end (float): End timestamp of the window.
            inverse (bool, optional): If True, removes data *inside* the window (cutting).
                                      If False, keeps data *inside* the window (cropping). Defaults to False.
            inplace (bool, optional): If True, modifies self. If False, returns a new instance. Defaults to True.

        Returns:
            Trajectory: The modified or new trajectory instance.
        """
        # filter to t_start and t_end
        if inverse:
            filt = [not t_start <= tstamps <= t_end for tstamps in self.timestamps]
        else:
            filt = [t_start <= tstamps <= t_end for tstamps in self.timestamps]

        return self.mask(mask=filt, inplace=inplace)

    def intersect(self, timestamps: np.ndarray, max_gap_size: float = 10.0, inplace: bool = True) -> "Trajectory":
        """
        Filters the trajectory to overlap with a reference timestamp vector.

        This method finds the common time span between self and the reference `timestamps`,
        crops self to that span, and then filters points that are either exact matches
        or exist within valid gaps defined by `max_gap_size`.

        Args:
            timestamps (np.ndarray): The reference timestamps to intersect with.
            max_gap_size (float, optional): The maximum allowed time gap (in seconds) between
                                            reference timestamps to include trajectory points. Defaults to 10.0.
            inplace (bool, optional): If True, modifies self. Defaults to True.

        Raises:
            ValueError: If the time spans do not overlap.

        Returns:
            Trajectory: The intersected trajectory.
        """
        traj_self = self if inplace else self.copy()
        time_span = common_time_span(tstamps1=timestamps, tstamps2=traj_self.timestamps)

        if time_span is None:
            raise ValueError("intersect_both: Timespans do not overlap!")

        traj_self.crop(t_start=time_span[0], t_end=time_span[1])

        tstamps_sorted = np.sort(timestamps)
        lower_neighbor_list = np.searchsorted(tstamps_sorted, traj_self.timestamps, side="right") - 1

        filter_index = [
            idx
            for idx, (tstamp, lower_neighbor) in enumerate(zip(traj_self.timestamps, lower_neighbor_list))
            if (
                tstamps_sorted[lower_neighbor] == tstamp
                or (tstamps_sorted[lower_neighbor + 1] - tstamps_sorted[lower_neighbor]) <= max_gap_size
            )
        ]
        traj_self.mask(np.array(filter_index, dtype=int))

        return traj_self

    def mask(self, mask: list | np.ndarray, inplace: bool = True) -> "Trajectory":
        """
        Applies a boolean mask or index array to filter all trajectory components.

        Filtered components include: timestamps, positions, rotations, path lengths, and velocities.

        Args:
            mask (Union[list, np.ndarray]): Boolean array or list of indices to keep.
            inplace (bool, optional): If True, modifies self. Defaults to True.

        Returns:
            Trajectory: The masked trajectory.
        """
        traj_self = self if inplace else self.copy()

        traj_self.timestamps = traj_self.timestamps[mask]
        traj_self.positions.xyz = traj_self.positions.xyz[mask, :]

        if traj_self.rotations:
            quat_filtered = traj_self.rotations.as_quat()[mask, :]
            traj_self.rotations = Rotations.from_quat(quat_filtered) if len(quat_filtered) > 0 else None

        traj_self.path_lengths = traj_self.path_lengths[mask]

        if traj_self.velocity_xyz is not None:
            traj_self.velocity_xyz = traj_self.velocity_xyz[mask]

        return traj_self

    def transform(self, transformation: np.ndarray, inplace: bool = True) -> "Trajectory":
        """
        Applies a rigid body transformation to the trajectory poses.

        Args:
            transformation (np.ndarray): A 4x4 homogeneous transformation matrix.
            inplace (bool, optional): If True, modifies self. Defaults to True.

        Returns:
            Trajectory: The transformed trajectory.
        """
        traj_self = self if inplace else self.copy()
        traj_self.se3 = [np.dot(transformation, p) for p in traj_self.se3]
        return traj_self

    def to_dataframe(self, sort_by: str = "") -> pd.DataFrame:
        """
        Exports the trajectory to a Pandas DataFrame.

        Columns usually include: time, path_length, pos_x, pos_y, pos_z, speed_x, speed_y, speed_z,
        and rotation columns (rot_x/y/z/w) if available.

        Args:
            sort_by (str, optional): Column name to sort by. If empty, uses `self.sorting`.

        Returns:
            pd.DataFrame: A dataframe containing the trajectory data.
        """
        sort_by = sort_by or self.sorting
        if self.rotations:
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

        return dataframe.sort_values(by=sort_by)

    def to_string(self) -> str:
        """
        Serializes the trajectory to a CSV-formatted string with metadata headers.

        Headers included: #epsg, #name, #nframe, #sorting, #fields.
        """

        def write_header() -> str:
            fields = "t,l,px,py,pz,vx,vy,vz" if self.rotations is None else "t,l,px,py,pz,qx,qy,qz,qw,vx,vy,vz"
            header = [
                f"#epsg {self.positions.epsg}",
                f"#name {self.name}",
                "#nframe enu",
                f"#sorting {self.sorting.value}",
                f"#fields {fields}",
            ]
            return "\n".join(header) + "\n"

        if self.rotations is None:
            trajectory_data = np.c_[self.timestamps, self.path_lengths, self.positions.xyz, self.velocity_xyz]
        else:
            trajectory_data = np.c_[
                self.timestamps,
                self.path_lengths,
                self.positions.xyz,
                self.rotations.as_quat(),
                self.velocity_xyz,
            ]

        output = io.StringIO()
        output.write(write_header())
        pd.DataFrame(trajectory_data).to_csv(output, header=False, index=False, float_format="%.9f")

        return output.getvalue()

    def to_file(self, filename: str, mode: str = "w") -> None:
        """
        Writes the trajectory to an ASCII file using the format defined in `to_string`.

        Args:
            filename (str): The output file path.
            mode (str, optional): File open mode. Defaults to "w".
        """
        with open(filename, mode=mode, newline="\n", encoding="utf-8") as file:
            file.write(self.to_string())

    def to_kml(self, filename: str, precision: float = 1e-6) -> str:
        """
        Exports the trajectory to a Google Earth KML file.

        Requires the trajectory to have a valid EPSG code so it can be converted to WGS84 (EPSG:4326).

        Args:
            filename (str): The output filename (e.g., "track.kml").
            precision (float, optional): Coordinate precision in degrees for rounding/simplification. Defaults to 1e-6.

        Raises:
            ValueError: If the trajectory does not have a known EPSG code.
        """
        traj = self.copy()
        if traj.positions.local_transformer is None:
            raise ValueError(
                "Trajectory must be defined in a well-known coordinate system (EPSG code) to be exported to KML. "
            )
        traj.positions.to_epsg(4326)

        traj.positions = traj.positions.round_to(precision)
        _, indices = np.unique(traj.positions.xyz[:, 0:2], return_index=True, axis=0)
        traj.mask(np.sort(indices))

        kml_file = ET.Element("kml", xmlns="http://earth.google.com/kml/2.1")
        document = ET.SubElement(kml_file, "Document")

        placemark = ET.SubElement(document, "Placemark")
        name = ET.SubElement(placemark, "name")
        name.text = traj.name

        style = ET.SubElement(placemark, "Style")
        line_style = ET.SubElement(style, "LineStyle")
        color = ET.SubElement(line_style, "color")
        color.text = "ff0000ff"
        width = ET.SubElement(line_style, "width")
        width.text = "2"

        line_string = ET.SubElement(placemark, "LineString")
        coordinates = ET.SubElement(line_string, "coordinates")

        coordinates.text = "\n".join(f"  {pos[1]:.9f},{pos[0]:.9f},{0.00:.3f}" for pos in traj.positions.xyz)

        tree = ET.ElementTree(kml_file)
        ET.indent(tree, space="", level=0)
        tree.write(filename, encoding="utf-8", xml_declaration=True)
