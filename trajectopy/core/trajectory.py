"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import copy
import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from pointset import PointSet
from scipy.spatial.transform import Slerp

import trajectopy.core.input_output.trajectory_io as trajectory_io
from trajectopy.core.alignment.equations import leverarm_time_component
from trajectopy.core.alignment.parameters import AlignmentParameters
from trajectopy.core.alignment.result import AlignmentResult
from trajectopy.core.approximation.cubic_approximation import piecewise_cubic
from trajectopy.core.approximation.rot_approximation import rot_average_window
from trajectopy.core.rotationset import RotationSet
from trajectopy.core.settings.approximation import ApproximationSettings
from trajectopy.core.settings.sorting import SortingSettings
from trajectopy.core.sorting import Sorting, sort_spatially
from trajectopy.core.utils import common_time_span, gradient_3d, lengths_from_xyz

# logger configuration
logger = logging.getLogger("root")


class TrajectoryError(Exception):
    pass


class Trajectory:
    """Class representing a trajectory, i.e. position and orientation of a plattform over time

    - Position-Computations are always done in a local frame
    - Time stamps are always in UTC time
    - Rotations are always defined in a East-North-Up frame

    Attributes:
        pos (PointSet): Position of the trajectory
        rot (RotationSet): Orientation of the trajectory
        tstamps (np.ndarray): Time stamps of the trajectory
        name (str): Name of the trajectory
        arc_lengths (np.ndarray): Arc lengths of the trajectory
        speed_3d (np.ndarray): 3D speed of the trajectory
        sorting (Sorting): Sorting of the trajectory

    Methods:

        __init__: Initialize trajectory
        __str__: Returns string describing trajectory
        __repr__: Returns string representation of trajectory
        __len__: Return number of poses
        __eq__: Check if two trajectories are equal
        init_arc_lengths: Initialize arc lengths
        copy: Deep copy of itself
        from_file: Create trajectory from file
        sort_switching_index: Returns the index that switches the sorting of the trajectory
        sorting_index: Returns the index that sorts the trajectory
        function_of: Returns the function of the trajectory
        function_of_unit: Returns the unit of the function of the trajectory
        function_of_label: Returns the label of the function of the trajectory
        xyz: Returns the xyz coordinates of the trajectory
        quat: Returns the quaternion of the trajectory
        rpy: Returns the roll, pitch, yaw of the trajectory
        to_dataframe: Returns a pandas dataframe containing tstamps, xyz, quat and speed_3d of the trajectory
        to_file: Writes trajectory to ascii file
        from_numpy: Initialize trajectory using numpy arrays
        se3: Returns SE3 pose list
        se3.setter: Sets position and rotation from se3 list
        data_rate: Returns data rate
        total_length: Return the total trajectory arc_length
        speed_3d: Returns computed speeds or custom speeds
        speed_3d.setter: Sets custom speeds
        speed: Returns trajectory speeds calculated using consecutive point distances
        crop: Crops trajectory to timespan defined by t_start and t_end
        interpolate: Interpolates a trajectory to specified timestamps
        _interpolate_rotations: Function for rotation interpolation of a trajectory
        _interpolate_positions: Function for position interpolation of a trajectory
        match_timestamps: Truncates trajectory to only those poses where the timestamps exactly match "tstamps"
        intersect: Intersects trajectory with a given timestamp vector
        apply_index: Applies index to the trajectory
        apply_transformation: Applies transformation to trajectory
    """

    _counter = 1

    def __init__(
        self,
        pos: PointSet,
        rot: Union[RotationSet, None] = None,
        tstamps: Union[np.ndarray, None] = None,
        name: str = "",
        arc_lengths: Union[np.ndarray, None] = None,
        speed_3d: Union[np.ndarray, None] = None,
        sorting: Sorting = Sorting.TIME,
    ) -> None:
        if rot and len(rot) != len(pos):
            raise ValueError("Dimension mismatch between positions and orientations.")

        self.sorting = sorting

        # pose
        self.pos = pos
        self.rot = rot
        self.tstamps = np.arange(0, len(pos)) if tstamps is None else tstamps

        if speed_3d is not None and len(speed_3d) == len(self.pos):
            self._speed_3d = speed_3d
        else:
            self._speed_3d = gradient_3d(xyz=self.pos.to_local(inplace=False).xyz, tstamps=self.tstamps)
            logger.info("Speeds were not provided or had wrong dimensions. Speeds were computed instead.")

        if arc_lengths is not None and len(arc_lengths) == len(self.pos):
            self.arc_lengths = arc_lengths
        else:
            self.arc_lengths = self.init_arc_lengths()
            logger.info("Arc lengths were not provided or had wrong dimensions. Arc lengths were computed instead.")

        self.name = name or f"Trajectory {Trajectory._counter}"

        Trajectory._counter += 1

    def __str__(self) -> str:
        """
        Returns string describing trajectory
        """
        width = 24
        return (
            f"\n _______________________________________________________\n"
            f"| ------------------ Trajectory Info ------------------ |\n"
            f"| Name:                         {self.name:<{width}}|\n"
            f"| Number of poses:              {len(self):<{width}}|\n"
            f"| Orientation available:        {'yes' if self.has_orientation else 'no':<{width}}|\n"
            f"| EPSG:                         {self.pos.epsg:<{width}}|\n"
            f"| Length [m]:                   {self.total_length:<{width}.3f}|\n"
            f"| Data rate [Hz]:               {self.data_rate:<{width}.3f}|\n"
            f"| Function of:                  {self.function_of_label:<{width}}|\n"
            f"|_______________________________________________________|\n"
        )

    @property
    def has_orientation(self) -> bool:
        """
        Returns True if orientation is available
        """
        return self.rot is not None and len(self.rot) > 0

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        """
        Return number of poses
        """
        return len(self.pos.xyz)

    def __eq__(self, other: "Trajectory") -> bool:
        if self.rot is not None and other.rot is not None:
            rot_equal = np.allclose(self.rot.as_quat(), other.rot.as_quat())
        elif self.rot is None and other.rot is None:
            rot_equal = True
        else:
            rot_equal = False

        return (
            np.allclose(self.pos.xyz, other.pos.xyz)
            and rot_equal
            and np.allclose(self.tstamps, other.tstamps)
            and np.allclose(self.arc_lengths, other.arc_lengths)
            and np.allclose(self._speed_3d, other._speed_3d)
            and self.name == other.name
        )

    def init_arc_lengths(self):
        return lengths_from_xyz(self.pos.to_local(inplace=False).xyz)

    def copy(self) -> "Trajectory":
        """
        Deep copy of itself
        """
        return copy.deepcopy(self)

    @classmethod
    def from_file(cls, filename: str, io_stream: bool = False) -> "Trajectory":
        """Create trajectory from file

        The file must be a csv file containing columns for at least
        the timestamp, x, y and z coordinates of the trajectory. Those
        fields must be named "t", "px", "py" and "pz" in the header using
        the #fields tag. However, by default a trajectory with
        "t,px,py,pz,qx,qy,qz,qw" fields is assumed. Additional fields
        include the arc length, specified by "l", and the speed, specified
        by "vx", "vy" and "vz".
        The delimiter can be specified using the #delimiter
        tag. The default delimiter is a comma.

        Args:
            filename (str): path to file
            io_stream (bool, optional): If true, the file is read from a stream.

        Returns:
            Trajectory: trajectory object
        """
        if io_stream:
            header_data, trajectory_data = trajectory_io.read_string(filename, dtype=object)
        else:
            header_data, trajectory_data = trajectory_io.read_data(filename, dtype=object)

        tstamps = trajectory_io.extract_trajectory_timestamps(header_data=header_data, trajectory_data=trajectory_data)
        pos = trajectory_io.extract_trajectory_pointset(header_data=header_data, trajectory_data=trajectory_data)
        arc_lengths = trajectory_io.extract_trajectory_arc_lengths(
            header_data=header_data, trajectory_data=trajectory_data
        )
        speed_3d = trajectory_io.extract_trajectory_speed(header_data=header_data, trajectory_data=trajectory_data)
        rot = trajectory_io.extract_trajectory_rotations(header_data=header_data, trajectory_data=trajectory_data)

        return Trajectory(
            tstamps=tstamps,
            pos=pos,
            rot=rot,
            name=header_data.name,
            arc_lengths=arc_lengths,
            speed_3d=speed_3d,
        )

    @property
    def sort_switching_index(self) -> np.ndarray:
        """
        Returns the index that switches the sorting of the trajectory
        """
        return np.argsort(self.sorting_index)

    @property
    def sorting_index(self) -> np.ndarray:
        """
        Returns the index that sorts the trajectory
        """
        return np.argsort(self.tstamps) if self.sorting == Sorting.TIME else np.argsort(self.arc_lengths)

    @property
    def function_of(self) -> np.ndarray:
        """
        Returns the function of the trajectory
        """
        return (
            self.tstamps[self.sorting_index] if self.sorting == Sorting.TIME else self.arc_lengths[self.sorting_index]
        )

    @property
    def function_of_unit(self) -> str:
        """
        Returns the unit of the function of the trajectory
        """
        return "s" if self.sorting == Sorting.TIME else "m"

    @property
    def function_of_label(self) -> str:
        """
        Returns the label of the function of the trajectory
        """
        return "time [s]" if self.sorting == Sorting.TIME else "arc length [m]"

    @property
    def xyz(self) -> np.ndarray:
        """
        Returns the xyz coordinates of the trajectory

        In contrast to the pos.xyz attribute, this method
        reflects the current sorting of the trajectory.
        """
        return self.pos.xyz[self.sorting_index]

    @property
    def quat(self) -> np.ndarray:
        """
        Returns the quaternion of the trajectory

        In contrast to the rot.as_quat() attribute, this method
        reflects the current sorting of the trajectory.
        """
        if self.rot is None:
            return np.zeros((len(self), 4))

        return self.rot.as_quat()[self.sorting_index]

    @property
    def rpy(self) -> np.ndarray:
        """
        Returns the roll, pitch, yaw of the trajectory

        In contrast to the rot.as_euler(seq="xyz") attribute, this method
        reflects the current sorting of the trajectory.
        """
        return RotationSet.from_quat(self.quat).as_euler(seq="xyz")

    def to_dataframe(self, sort_by: str = "") -> pd.DataFrame:
        """
        Returns a pandas dataframe containing tstamps, xyz, quat
        and speed_3d of the trajectory.

        The dataframe is sorted by the current sorting attribute (time or arc_length).

        Args:
            sort_by (str, optional): Column to sort by. This
                                     overrides the current sort_by
                                     attribute.

        Returns:
            pd.DataFrame: Trajectory as dataframe
        """
        sort_by = sort_by or self.sorting
        if self.rot:
            dataframe = pd.DataFrame(
                np.c_[self.tstamps, self.arc_lengths, self.pos.xyz, self.rot.as_quat(), self.speed_3d],
                columns=[
                    "time",
                    "arc_length",
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
                np.c_[self.tstamps, self.arc_lengths, self.pos.xyz, self.speed_3d],
                columns=["time", "arc_length", "pos_x", "pos_y", "pos_z", "speed_x", "speed_y", "speed_z"],
            )

        return dataframe.sort_values(by=sort_by)

    def to_file(self, filename: str, mode: str = "w") -> None:
        """Writes trajectory to ascii file

        The first line will always be the epsg information.
        After that, the trajectory data is written.

        Args:
            filename (str): Output filename
        """

        def write_header(filename: str, mode: str = "w") -> None:
            fields = "t,l,px,py,pz,vx,vy,vz" if self.rot is None else "t,l,px,py,pz,qx,qy,qz,qw,vx,vy,vz"
            with open(filename, mode=mode, newline="\n", encoding="utf-8") as file:
                file.write(f"#epsg {self.pos.epsg}\n")
                file.write(f"#name {self.name}\n")
                file.write("#nframe enu\n")
                file.write(f"#fields {fields}\n")

        if self.rot is None:
            trajectory_data = np.c_[self.tstamps, self.arc_lengths, self.pos.xyz, self.speed_3d]
        else:
            trajectory_data = np.c_[
                self.tstamps,
                self.arc_lengths,
                self.pos.xyz,
                self.rot.as_quat(),
                self.speed_3d,
            ]

        write_header(filename=filename, mode=mode)
        pd.DataFrame(trajectory_data).to_csv(filename, header=False, index=False, mode="a", float_format="%.9f")

    @classmethod
    def from_numpy(cls, xyz: np.ndarray, quat: np.ndarray, tstamps: np.ndarray, epsg: int = 0) -> "Trajectory":
        """
        Initialize trajectory using numpy arrays
        """
        pos = PointSet(xyz=xyz, epsg=epsg)
        rot = RotationSet.from_quat(quat)
        return Trajectory(pos=pos, rot=rot, tstamps=tstamps)

    @property
    def se3(self) -> List[np.ndarray]:
        """
        Returns SE3 pose list
        """
        se3_list: List[np.ndarray] = []
        xyz = self.pos.xyz

        if len(xyz) == 0:
            return se3_list

        matrices = [np.eye(3)] * len(xyz) if self.rot is None else self.rot.as_matrix()
        for t, r in zip(xyz, matrices):
            se3 = np.eye(4)
            se3[:3, 3] = t
            se3[:3, :3] = r
            se3_list.append(se3)
        return se3_list

    @se3.setter
    def se3(self, se3_list: List[np.ndarray]) -> None:
        """
        Sets position and rotation from se3 list
        """
        xyz = np.zeros((len(se3_list), 3))
        matrices = np.zeros((len(se3_list), 3, 3))

        for i, pose in enumerate(se3_list):
            xyz[i, :] = pose[:3, 3]
            matrices[i, :, :] = pose[:3, :3]

        self.pos.xyz = xyz
        self.rot = RotationSet.from_matrix(matrices)

    @property
    def data_rate(self) -> float:
        """
        Returns data rate
        """
        return 1 / np.mean(np.diff(np.sort(self.tstamps)))

    @property
    def total_length(self) -> float:
        """
        Return the total trajectory arc_length.
        """
        return 0.0 if len(self.arc_lengths) == 0 else self.arc_lengths[-1]

    @property
    def speed_3d(self) -> np.ndarray:
        """Returns computed speeds or custom speeds"""
        if self._speed_3d is not None:
            return self._speed_3d

        return gradient_3d(xyz=self.pos.to_local(inplace=False).xyz, tstamps=self.tstamps)

    @speed_3d.setter
    def speed_3d(self, speed_3d: np.ndarray) -> None:
        """Sets custom speeds"""
        self._speed_3d = speed_3d

    @property
    def speed(self) -> np.ndarray:
        """
        Returns trajectory speeds
        calculated using consecutive point distances
        """
        return np.linalg.norm(self.speed_3d, axis=1)

    def crop(self, t_start: float, t_end: float, inverse: bool = False, inplace: bool = True) -> "Trajectory":
        """Crops trajectory to timespan defined by t_start and t_end

        Args:
            t_start (float): Start timestamp of desired time span
            t_end (float): End timestamp of desired time span
            inverse (bool, optional): If true, 'crop' turns
                                      into 'cut', i.e. everthing
                                      outside of t_start and t_end
                                      will be removed.
                                      Defaults to False.
            inplace (bool, optional): Perform crop in-place.
                                      Defaults to True.

        Returns:
            Trajectory: Cropped trajectory
        """
        # filter to t_start and t_end
        if inverse:
            filt = [not t_start <= tstamps <= t_end for tstamps in self.tstamps]
        else:
            filt = [t_start <= tstamps <= t_end for tstamps in self.tstamps]

        return self.apply_index(index=filt, inplace=inplace)

    def interpolate(self, tstamps: Union[list, np.ndarray], inplace: bool = True) -> "Trajectory":
        """Interpolates a trajectory to specified timestamps

        This method removes timestamps from tstamps if they lie
        outside of the timestamp range of the trajectory (self).
        Since providing values for those timestamps would require
        an extrapolation and not an interpolation, this behaviour
        is consistent with the definition of this method.

        Args:
            tstamps (list): Interpolation timestamps
            inplace (bool, optional): Perform in-place interpolation.
                                      Defaults to True.

        Returns:
            Trajectory: Interpolated trajectory
        """
        tstamps = np.sort(tstamps)
        traj_self = self if inplace else self.copy()
        tstamps_cropped = np.array([tstamp for tstamp in tstamps if self.tstamps[0] <= tstamp <= self.tstamps[-1]])

        traj_self._interpolate_positions(tstamps_cropped)  # pylint: disable=protected-access
        traj_self._interpolate_rotations(tstamps_cropped)  # pylint: disable=protected-access
        traj_self.speed_3d = gradient_3d(xyz=traj_self.pos.xyz, tstamps=tstamps_cropped)
        traj_self.arc_lengths = np.interp(tstamps_cropped, traj_self.tstamps, traj_self.arc_lengths)
        traj_self.tstamps = tstamps_cropped

        logger.info("Interpolated %s", traj_self.name)

        return traj_self

    def _interpolate_rotations(self, tstamps: Union[list, np.ndarray], inplace: bool = True) -> "Trajectory":
        """Function for rotation interpolation of a trajectory

        This method uses Spherical-Linear-Interpolation
        for rotation interpolation.

        Args:
            tstamps (np.ndarray): Interpolation timestamps
            inplace (bool, optional): Perform in-place interpolation.

        Returns:
            RotationSet: Interpolated Rotationset
        """
        traj_self = self if inplace else self.copy()

        if not self.rot or len(tstamps) == 0:
            return traj_self

        # spherical linear orientation interpolation
        # Slerp interpolation, as geodetic curve on unit sphere
        slerp = Slerp(traj_self.tstamps, traj_self.rot)
        r_i = slerp(tstamps)
        traj_self.rot = RotationSet.from_quat(r_i.as_quat())
        return traj_self

    def _interpolate_positions(self, tstamps: np.ndarray, inplace: bool = True) -> "Trajectory":
        """Function for position interpolation of a trajectory

        Args:
            tstamps (np.ndarray): Interpolation timestamps
            inplace (bool, optional): Perform in-place interpolation.

        Returns:
            np.ndarray: Interpolated positions
        """
        traj_self = self if inplace else self.copy()

        x_i = np.interp(tstamps, traj_self.tstamps, traj_self.pos.x)
        y_i = np.interp(tstamps, traj_self.tstamps, traj_self.pos.y)
        z_i = np.interp(tstamps, traj_self.tstamps, traj_self.pos.z)
        traj_self.pos.xyz = np.c_[x_i, y_i, z_i]
        return traj_self

    def match_timestamps(self, tstamps: np.ndarray, inplace: bool = True) -> "Trajectory":
        """Truncates trajectory to only those poses where the timestamps exactly match "tstamps"

        Args:
            tstamps (np.ndarray): Input timestamps
            inplace (bool, optional): Perform matching in-place. Defaults to True.

        Returns:
            Trajectory: Trajectory with matched timestamps
        """
        traj_self = self if inplace else self.copy()
        _, idx_self, _ = np.intersect1d(traj_self.tstamps, tstamps, return_indices=True)
        traj_self.apply_index(idx_self)
        return traj_self

    def intersect(self, tstamps: np.ndarray, max_gap_size: float = 2.0, inplace: bool = True) -> "Trajectory":
        """Intersects trajectory with a given timestamp vector

        After intersection, the trajectory covers the same
        timespan as 'tstamps'. Further, gaps larger than
        'max_gap_size' are removed. If two consecutive
        timespans in tstamps have a difference of more than
        'max_gap_size' seconds, they are considered as the
        limits of a gap. All timestamps of the trajectory
        that lie within this gap will be removed.

        Args:
            tstamps (np.ndarray): Intersection timespans
            max_gap_size (float, optional): Maximum allowed gap between timespans.
                                            If Defaults to 0.5.
            inplace (bool, optional): Perform intersection in-place.
                                      Defaults to True.

        Raises:
            ValueError: If timespans do not overlap.

        Returns:
            Trajectory: Intersected trajectory
        """
        traj_self = self if inplace else self.copy()
        time_span = common_time_span(tstamps1=tstamps, tstamps2=traj_self.tstamps)

        if time_span is None:
            raise ValueError("intersect_both: Timespans do not overlap!")

        traj_self.crop(t_start=time_span[0], t_end=time_span[1])

        tstamps_sorted = np.sort(tstamps)
        lower_neighbor_list = np.searchsorted(tstamps_sorted, traj_self.tstamps, side="right") - 1

        filter_index = [
            idx
            for idx, (tstamp, lower_neighbor) in enumerate(zip(traj_self.tstamps, lower_neighbor_list))
            if (
                tstamps_sorted[lower_neighbor] == tstamp
                or (tstamps_sorted[lower_neighbor + 1] - tstamps_sorted[lower_neighbor]) <= max_gap_size
            )
        ]
        traj_self.apply_index(np.array(filter_index, dtype=int))

        return traj_self

    def apply_index(self, index: Union[list, np.ndarray], inplace: bool = True) -> "Trajectory":
        """Applies index to the trajectory

        This will be done either in-place or using a new
        instance of a trajectory. The index can be used to
        filter and / or sort the components of the trajectory.

        Those components are:
        - timestamps (tstamps)
        - positions (xyz)
        - rotations (rot)
        - arc lengths (arc_lengths)
        - sorting index (_sort_index)

        Args:
            index (Union[list, np.ndarray]): index that should be applied
            inplace (bool, optional): Perform in-place. Defaults to True.

        Returns:
            Trajectory: Trajectory with index applied.
        """
        traj_self = self if inplace else self.copy()

        traj_self.tstamps = traj_self.tstamps[index]
        traj_self.pos.xyz = traj_self.pos.xyz[index, :]

        if traj_self.rot:
            quat_filtered = traj_self.rot.as_quat()[index, :]
            traj_self.rot = RotationSet.from_quat(quat_filtered)

        traj_self.arc_lengths = traj_self.arc_lengths[index]

        if traj_self.speed_3d is not None:
            traj_self.speed_3d = traj_self.speed_3d[index]

        return traj_self

    def apply_transformation(self, transformation: np.ndarray, inplace: bool = True) -> "Trajectory":
        """Applies transformation to trajectory

        Args:
            transformation (np.ndarray): 4x4 Transformation matrix
            inplace (bool, optional): Perform in-place. Defaults to True.

        Returns:
            Trajectory: Transformed trajectory

        """
        traj_self = self if inplace else self.copy()
        traj_self.se3 = [np.dot(transformation, p) for p in traj_self.se3]
        return traj_self

    def apply_alignment(self, alignment_result: AlignmentResult, inplace: bool = True) -> "Trajectory":
        """Transforms trajectory using alignment parameters.

        After computing the alignment parameters needed to align
        two trajectories, they can be applied to arbitrary trajectories.

        Args:
            alignment_result (AlignmentResult)
            inplace (bool, optional): Perform in-place. Defaults to True.

        Returns:
            Trajectory: Aligned trajectory
        """

        def _prepare_alignment_application(
            trajectory: Trajectory, alignment_parameters: AlignmentParameters
        ) -> Tuple[float, ...]:
            if trajectory.rot is not None:
                rpy = trajectory.rot.as_euler("xyz", degrees=False)
                euler_x, euler_y, euler_z = rpy[:, 0], rpy[:, 1], rpy[:, 2]
                lever_x, lever_y, lever_z = (
                    alignment_parameters.lever_x.value,
                    alignment_parameters.lever_y.value,
                    alignment_parameters.lever_z.value,
                )
            else:
                logger.error("Trajectory has no orientations. Cannot apply leverarm.")
                euler_x, euler_y, euler_z = 0, 0, 0
                lever_x, lever_y, lever_z = 0, 0, 0

            return euler_x, euler_y, euler_z, lever_x, lever_y, lever_z

        trajectory = self if inplace else self.copy()

        # leverarm and time
        (
            euler_x,
            euler_y,
            euler_z,
            lever_x,
            lever_y,
            lever_z,
        ) = _prepare_alignment_application(trajectory, alignment_result.position_parameters)

        speed_3d = trajectory.speed_3d
        speed_x, speed_y, speed_z = speed_3d[:, 0], speed_3d[:, 1], speed_3d[:, 2]

        trafo_x, trafo_y, trafo_z = leverarm_time_component(
            euler_x=euler_x,
            euler_y=euler_y,
            euler_z=euler_z,
            lever_x=lever_x,
            lever_y=lever_y,
            lever_z=lever_z,
            time_shift=alignment_result.position_parameters.time_shift.value,
            speed_x=speed_x,
            speed_y=speed_y,
            speed_z=speed_z,
        )
        trajectory.pos.xyz += np.c_[trafo_x, trafo_y, trafo_z]

        # similiarity transformation
        trajectory.apply_transformation(alignment_result.position_parameters.sim3_matrix)

        logger.info("Applied alignment parameters to positions.")

        # sensor orientation
        if trajectory.rot is not None:
            trajectory.rot = alignment_result.rotation_parameters.rotation_set * trajectory.rot
            logger.info("Applied alignment parameters to orientations.")

        return trajectory

    def sort_spatially(
        self, sorting_settings: SortingSettings = SortingSettings(), inplace: bool = True
    ) -> "Trajectory":
        """
        Sorts the trajectory spatially.

        Args:
            sorting_settings (SortingSettings): Sorting settings.
            inplace (bool, optional): Whether to sort the trajectory in-place. Defaults to True.

        Returns:
            Trajectory: Sorted trajectory.

        """
        sort_idx, arc_lengths = sort_spatially(xyz_unsorted=self.pos.xyz, settings=sorting_settings)
        arg_sort_sort_idx = np.argsort(sort_idx)
        trajectory = self.apply_index(sorted(sort_idx), inplace=inplace)
        trajectory.arc_lengths = arc_lengths[arg_sort_sort_idx]
        trajectory.sorting = Sorting.ARC_LENGTH
        return trajectory

    def approximate(
        self, approximation_settings: ApproximationSettings = ApproximationSettings(), inplace: bool = True
    ) -> "Trajectory":
        """
        Approximates the trajectory using piecewise cubic polynomial.

        Args:
            approximation_settings (ApproximationSettings): Approximation settings.

        Returns:
            Trajectory: Approximated trajectory.

        """
        xyz_approx = piecewise_cubic(
            function_of=self.function_of,
            values=self.xyz,
            int_size=approximation_settings.fe_int_size,
            min_obs=approximation_settings.fe_min_obs,
        )

        traj_approx = self if inplace else self.copy()
        traj_approx.pos.xyz = xyz_approx[self.sort_switching_index, :]

        if not traj_approx.has_orientation:
            return traj_approx

        quat_approx = rot_average_window(
            function_of=self.function_of,
            quat=self.quat,
            win_size=approximation_settings.rot_approx_win_size,
        )
        traj_approx.rot = RotationSet.from_quat(quat_approx[self.sort_switching_index, :])

        return traj_approx

    def adopt_first_pose(self, trajectory: "Trajectory", inplace: bool = True) -> "Trajectory":
        """Transform trajectory so that the first pose is identical in both

        Args:
            trajectory (Trajectory): Target Trajectory
            inplace (bool, optional): Perform in-place. Defaults to True.

        Returns:
            Trajectory: Transformed trajectory
        """
        trajectory = self if inplace else self.copy()
        trajectory.adopt_first_position(trajectory)
        trajectory.adopt_first_orientation(trajectory)
        return trajectory

    def adopt_first_position(self, trajectory: "Trajectory", inplace: bool = True) -> "Trajectory":
        """Transform trajectory so that the first position is identical in both

        Args:
            trajectory (Trajectory): Target Trajectory
            inplace (bool, optional): Perform in-place. Defaults to True.

        Returns:
            Trajectory: Transformed trajectory
        """
        trajectory = self if inplace else self.copy()
        position_difference = trajectory.pos.xyz[0, :] - trajectory.pos.xyz[0, :]
        trajectory.pos.xyz += position_difference
        return trajectory

    def adopt_first_orientation(self, trajectory: "Trajectory", inplace: bool = True) -> "Trajectory":
        """Transform trajectory so that the first orientation is identical in both

        Args:
            trajectory (Trajectory): Target Trajectory
            inplace (bool, optional): Perform in-place. Defaults to True.

        Returns:
            Trajectory: Transformed trajectory
        """
        trajectory = self if inplace else self.copy()
        if self.rot is not None and trajectory.rot is not None:
            rpy_from = trajectory.rot.as_euler(seq="xyz")
            rotation_difference = trajectory.rot.as_euler(seq="xyz")[0, :] - rpy_from[0, :]

            trajectory.rot = RotationSet.from_euler(seq="xyz", angles=rpy_from + rotation_difference)

        return trajectory
