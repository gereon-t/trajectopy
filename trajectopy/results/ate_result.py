from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.readers.header import HeaderData
from trajectopy.utils.common import rms
from trajectopy.utils.definitions import Sorting


@dataclass
class AbsoluteTrajectoryDeviations:
    """Container holding absolute pose deviations between two trajectories.

    The arrays stored in this dataclass represent per-epoch deviations after
    an optional temporal/spatial alignment of a reference (ground truth) and
    a test trajectory. They are later aggregated by `ATEResult` to derive
    metrics such as mean/median Absolute Trajectory Error (ATE), directional
    bias, and RMS statistics.

    Attributes:
        pos_dev (np.ndarray): 3D array of absolute position deviation vectors
            in meters. Shape is (N, 3) for N deviation vectors with x, y, z
            components. These deviations are directly depending on the
            definition of the coordinate frame of the trajectories.
        directed_pos_dev (np.ndarray): 3D array of directed position deviation
            vectors (e.g. along-track, horizontal/vertical cross-track) in meters.
            Shape is (N, 3). These deviations are independent of the global
            coordinate frame, as they are derived based on the along-track
            direction of the trajectory.
        rot_dev (Rotations | None): Optional orientation deviations represented
            as a `Rotations` instance (radians). `None` if rotational evaluation
            was not performed.
        rotations_used (bool): Indicates whether trajectory rotation information
            was employed to derive the direction frame for `directed_pos_dev`.
            If False, directional decomposition has relied on positional
            differences only.
    """

    pos_dev: np.ndarray
    directed_pos_dev: np.ndarray
    rot_dev: Rotations | None = None
    rotations_used: bool = False


class ATEResult:
    """
    This class represents a set of absolute trajectory deviations

    Absolute trajectory deviations describe absolute pose deviations between
    two trajectories. The deviations are calculated by comparing pairs of
    positions and orientations in the test and reference trajectory.

    Attributes:
        trajectory (Trajectory): The trajectory the deviations are belonging to
        abs_dev (AbsoluteTrajectoryDeviations): The absolute trajectory deviations
        name (str): The name of the result
    """

    def __init__(
        self,
        trajectory: Trajectory,
        abs_dev: AbsoluteTrajectoryDeviations,
        name: str = "",
    ) -> None:
        self.name = name or trajectory.name
        self.trajectory = trajectory
        self.abs_dev = abs_dev

    @property
    def property_dict(self) -> dict[str, str]:
        return {
            "Name": self.name,
            "Type": self.__class__.__name__,
            "Number of deviations": str(len(self.abs_dev.pos_dev)),
            "Deviation directions derived using": "Rotations" if self.abs_dev.rotations_used else "Positions / Unkown",
            "Maximum position deviation [m]": f"{self.pos_dev_max:.4f}",
            "Mean position deviation [m]": f"{self.pos_ate:.4f}",
            "Median position deviation [m]": f"{self.pos_dev_median:.4f}",
            "Minimum position deviation [m]": f"{self.pos_dev_min:.4f}",
            "RMS Position [m]": f"{self.pos_dev_rms:.4f}",
            "STD Position [m]": f"{self.pos_dev_std:.4f}",
            "Bias X [m]": f"{self.pos_bias_x:.4f}",
            "Bias Y [m]": f"{self.pos_bias_y:.4f}",
            "Bias Z [m]": f"{self.pos_bias_z:.4f}",
            "RMS X [m]": f"{self.pos_rms_x:.4f}",
            "RMS Y [m]": f"{self.pos_rms_y:.4f}",
            "RMS Z [m]": f"{self.pos_rms_z:.4f}",
            "Bias Along-Track [m]": f"{self.pos_bias_along:.4f}",
            "Bias Horizontal Cross-Track [m]": f"{self.pos_bias_cross_h:.4f}",
            "Bias Vertical Cross-Track [m]": f"{self.pos_bias_cross_v:.4f}",
            "RMS Along-Track [m]": f"{self.pos_rms_along:.4f}",
            "RMS Horizontal Cross-Track [m]": f"{self.pos_rms_cross_h:.4f}",
            "RMS Vertical Cross-Track [m]": f"{self.pos_rms_cross_v:.4f}",
            "Maximum rotation deviation [°]": (
                f"{np.rad2deg(self.rot_dev_max):.4f}" if self.abs_dev.rot_dev is not None else "-"
            ),
            "Mean rotation deviation [°]": (
                f"{np.rad2deg(self.rot_ate):.4f}" if self.abs_dev.rot_dev is not None else "-"
            ),
            "Median rotation deviation [°]": (
                f"{np.rad2deg(self.rot_dev_median):.4f}" if self.abs_dev.rot_dev is not None else "-"
            ),
            "Minimum rotation deviation [°]": (
                f"{np.rad2deg(self.rot_dev_min):.4f}" if self.abs_dev.rot_dev is not None else "-"
            ),
            "RMS Rotation [°]": f"{np.rad2deg(self.rot_dev_rms):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "STD Rotation [°]": f"{np.rad2deg(self.rot_dev_std):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "RMS Roll [°]": f"{np.rad2deg(self.rot_rms_x):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "RMS Pitch [°]": f"{np.rad2deg(self.rot_rms_y):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "RMS Yaw [°]": f"{np.rad2deg(self.rot_rms_z):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "Bias Roll [°]": f"{np.rad2deg(self.rot_bias_x):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "Bias Pitch [°]": f"{np.rad2deg(self.rot_bias_y):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "Bias Yaw [°]": f"{np.rad2deg(self.rot_bias_z):.4f}" if self.abs_dev.rot_dev is not None else "-",
        }

    def __eq__(self, other) -> bool:
        for self_value, other_value in zip(self.property_dict.values(), other.property_dict.values()):
            assert self_value == other_value

        return True

    def remove_ate_above(self, threshold: float) -> None:
        """Clips the ATE at a given value.

        This is useful to remove outliers from the ATE.

        Args:
            threshold (float): Maximum ATE value to keep.
        """
        if threshold <= 0.0:
            return

        ate_below_idx = [dev < threshold for dev in self.pos_dev_comb]
        self.trajectory.mask(ate_below_idx)

        self.abs_dev.pos_dev = self.abs_dev.pos_dev[ate_below_idx]
        self.abs_dev.directed_pos_dev = self.abs_dev.directed_pos_dev[ate_below_idx]

        if self.abs_dev.rot_dev is not None:
            self.abs_dev.rot_dev = Rotations.from_euler(
                angles=self.abs_dev.rot_dev.as_euler(seq="xyz")[ate_below_idx],
                seq="xyz",
            )

    @property
    def index(self) -> np.ndarray:
        """Returns the variable that currently parametrizes the trajectory.

        Depending on the sorting of the trajectory, this is either time or arc length.

        Returns:
            np.ndarray: Current independent variable array (time or path-length).
        """
        return self.trajectory.index

    @property
    def _sorted_abs_pos_dev(self) -> np.ndarray:
        """Returns absolute deviations sorted according to the trajectory's sorting index.

        Returns:
            np.ndarray: Sorted absolute position deviations.
        """
        return self.abs_dev.pos_dev[self.trajectory.sorting_index]

    @property
    def _sorted_directed_pos_dev(self) -> np.ndarray:
        """Returns directed absolute deviations sorted according to the trajectory's sorting index.

        Returns:
            np.ndarray: Sorted directed position deviations.
        """
        return self.abs_dev.directed_pos_dev[self.trajectory.sorting_index]

    @property
    def sorted_abs_rot_dev(self) -> Rotations:
        """Returns absolute rotation deviations sorted according to the trajectory's sorting index.

        Returns:
            Rotations: Sorted absolute rotation deviations.
        """
        if self.abs_dev.rot_dev is not None:
            return Rotations.from_quat(self.abs_dev.rot_dev.as_quat()[self.trajectory.sorting_index])

    @property
    def has_orientation(self) -> bool:
        """Returns True if orientation is available.

        Returns:
            bool: True if rotation deviations are available, False otherwise.
        """
        return self.abs_dev.rot_dev is not None

    @property
    def pos_dev_x(self) -> np.ndarray:
        """Returns x deviations"""
        return self._sorted_abs_pos_dev[:, 0]

    @property
    def pos_dev_y(self) -> np.ndarray:
        """Returns y deviations"""
        return self._sorted_abs_pos_dev[:, 1]

    @property
    def pos_dev_z(self) -> np.ndarray:
        """Returns z deviations"""
        return self._sorted_abs_pos_dev[:, 2]

    @property
    def pos_bias_x(self) -> float:
        """Returns x bias"""
        return np.mean(self.pos_dev_x)

    @property
    def pos_bias_y(self) -> float:
        """Returns y bias"""
        return np.mean(self.pos_dev_y)

    @property
    def pos_bias_z(self) -> float:
        """Returns z bias"""
        return np.mean(self.pos_dev_z)

    @property
    def pos_bias_cross_h(self) -> float:
        """Returns horizontal cross track bias"""
        return np.mean(self.pos_dev_cross_h)

    @property
    def pos_bias_cross_v(self) -> float:
        """Returns vertical cross track bias"""
        return np.mean(self.pos_dev_cross_v)

    @property
    def pos_bias_along(self) -> float:
        """Returns along track bias"""
        return np.mean(self.pos_dev_along)

    @property
    def rot_dev_x(self) -> np.ndarray:
        """Returns roll deviations"""
        return self.rot_dev_xyz[:, 0]

    @property
    def rot_dev_y(self) -> np.ndarray:
        """Returns pitch deviations"""
        return self.rot_dev_xyz[:, 1]

    @property
    def rot_dev_z(self) -> np.ndarray:
        """Returns yaw deviations"""
        return self.rot_dev_xyz[:, 2]

    @cached_property
    def rot_bias_xyz(self) -> np.ndarray:
        """Returns roll, pitch and yaw bias"""
        return self.sorted_abs_rot_dev.mean().as_euler(seq="xyz") if self.abs_dev.rot_dev is not None else np.zeros(3)

    @property
    def rot_bias_x(self) -> np.ndarray:
        """Returns roll bias"""
        return self.rot_bias_xyz[0]

    @property
    def rot_bias_y(self) -> np.ndarray:
        """Returns pitch bias"""
        return self.rot_bias_xyz[1]

    @property
    def rot_bias_z(self) -> np.ndarray:
        """Returns yaw bias"""
        return self.rot_bias_xyz[2]

    @property
    def pos_dev_along(self) -> np.ndarray:
        """
        Returns deviations of along track deviations
        """
        return self._sorted_directed_pos_dev[:, 0]

    @property
    def pos_dev_cross_h(self) -> np.ndarray:
        """
        Returns deviations of horizontal cross track deviations
        """
        return self._sorted_directed_pos_dev[:, 1]

    @property
    def pos_dev_cross_v(self) -> np.ndarray:
        """
        Returns deviations of vertical cross track deviations
        """
        return self._sorted_directed_pos_dev[:, 2]

    @cached_property
    def rot_dev_xyz(self) -> np.ndarray:
        """
        Returns rpy deviations
        """
        return (
            self.sorted_abs_rot_dev.as_euler(seq="xyz")
            if self.abs_dev.rot_dev is not None
            else np.zeros_like(self.abs_dev.pos_dev)
        )

    @property
    def pos_dev_comb(self) -> np.ndarray:
        """
        Returns position deviations combined using the L2 norm
        """
        return np.linalg.norm(self._sorted_abs_pos_dev, axis=1)

    @property
    def rot_dev_comb(self) -> np.ndarray:
        """
        Returns rotation deviations as single rotation angles
        """
        return (
            self.sorted_abs_rot_dev.rotangle
            if self.abs_dev.rot_dev is not None
            else np.zeros_like(self.abs_dev.pos_dev)
        )

    @property
    def pos_dev_rms(self) -> float:
        """
        Returns RMS of 3d positions
        """
        return rms(self.pos_dev_comb)

    @property
    def pos_ate(self) -> float:
        """
        Returns mean of 3d position deviations
        """
        return np.mean(self.pos_dev_comb)

    @property
    def pos_dev_max(self) -> float:
        """
        Returns max of 3d position deviations
        """
        return np.max(self.pos_dev_comb)

    @property
    def pos_dev_min(self) -> float:
        """
        Returns min of 3d position deviations
        """
        return np.min(self.pos_dev_comb)

    @property
    def pos_dev_median(self) -> float:
        """
        Returns min of 3d position deviations
        """
        return np.median(self.pos_dev_comb)

    @property
    def pos_dev_std(self) -> float:
        """
        Returns std of 3d position deviations
        """
        return np.std(self.pos_dev_comb)

    @property
    def rot_dev_rms(self) -> float:
        """
        Returns RMS of rotations
        """
        return rms(self.rot_dev_comb) if self.abs_dev.rot_dev is not None else 0.0

    @property
    def rot_dev_std(self) -> float:
        """
        Returns STD of rotations
        """
        return float(np.std(self.rot_dev_comb))

    @property
    def rot_ate(self) -> float:
        """
        Returns mean of rotation deviations
        """
        return float(np.mean(self.rot_dev_comb))

    @property
    def rot_dev_median(self) -> float:
        """
        Returns median of rotations
        """
        return float(np.median(self.rot_dev_comb))

    @property
    def rot_dev_min(self) -> float:
        """
        Returns min of rotations
        """
        return np.min(self.rot_dev_comb)

    @property
    def rot_dev_max(self) -> float:
        """
        Returns max of rotations
        """
        return np.max(self.rot_dev_comb)

    @property
    def pos_rms_along(self) -> float:
        """
        Returns RMS of along track deviations
        """
        return rms(self.pos_dev_along)

    @property
    def pos_rms_cross_h(self) -> float:
        """
        Returns RMS of horizontal cross track deviations
        """
        return rms(self.pos_dev_cross_h)

    @property
    def pos_rms_cross_v(self) -> float:
        """
        Returns RMS of vertical cross track deviations
        """
        return rms(self.pos_dev_cross_v)

    @property
    def pos_rms_x(self) -> float:
        """
        Returns RMS of x deviations
        """
        return rms(self.pos_dev_x)

    @property
    def pos_rms_y(self) -> float:
        """
        Returns RMS of y deviations
        """
        return rms(self.pos_dev_y)

    @property
    def pos_rms_z(self) -> float:
        """
        Returns RMS of z deviations
        """
        return rms(self.pos_dev_z)

    @property
    def rot_rms_x(self) -> float:
        """
        Returns RMS of roll deviations
        """
        return rms(self.rot_dev_x)

    @property
    def rot_rms_y(self) -> float:
        """
        Returns RMS of pitch deviations
        """
        return rms(self.rot_dev_y)

    @property
    def rot_rms_z(self) -> float:
        """
        Returns RMS of yaw deviations
        """
        return rms(self.rot_dev_z)

    @property
    def columns(self) -> list[str]:
        """Returns the column names of the dataframe"""
        trajectory_columns = [
            "time",
            "arc_lengths",
            "pos_x",
            "pos_y",
            "pos_z",
        ]

        if self.trajectory.has_orientation:
            trajectory_columns.extend(["rot_x", "rot_y", "rot_z", "rot_w"])

        deviation_columns = [
            "pos_dev_x",
            "pos_dev_y",
            "pos_dev_z",
            "pos_dev_along",
            "pos_dev_cross_h",
            "pos_dev_cross_v",
        ]

        if self.abs_dev.rot_dev:
            deviation_columns.extend(["rot_dev_x", "rot_dev_y", "rot_dev_z", "rot_dev_w"])

        trajectory_columns.extend(deviation_columns)
        return trajectory_columns

    @classmethod
    def from_file(cls, filename: str):
        header_data = HeaderData.from_file(filename)
        deviation_data = pd.read_csv(filename, comment="#")

        tstamps = deviation_data["time"].to_numpy(dtype=float)
        path_lengths = deviation_data["arc_lengths"].to_numpy(dtype=float)
        positions = Positions(deviation_data[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float), epsg=header_data.epsg)

        rot_columns = ["rot_x", "rot_y", "rot_z", "rot_w"]
        if all(column in deviation_data.columns for column in rot_columns):
            rotations = Rotations.from_quat(deviation_data[rot_columns].to_numpy(dtype=float))
        else:
            rotations = None

        pos_dev = deviation_data[["pos_dev_x", "pos_dev_y", "pos_dev_z"]].to_numpy(dtype=float)
        directed_pos_dev = deviation_data[["pos_dev_along", "pos_dev_cross_h", "pos_dev_cross_v"]].to_numpy(
            dtype=float
        )

        rot_dev_columns = ["rot_dev_x", "rot_dev_y", "rot_dev_z", "rot_dev_w"]
        if all(column in deviation_data.columns for column in rot_dev_columns):
            rot_dev = Rotations.from_quat(deviation_data[rot_dev_columns].to_numpy(dtype=float))
        else:
            rot_dev = None

        trajectory = Trajectory(
            name=header_data.name,
            positions=positions,
            rotations=rotations,
            timestamps=tstamps,
            path_lengths=path_lengths,
            sorting=Sorting.from_str(header_data.sorting),
        )
        ate_result = AbsoluteTrajectoryDeviations(pos_dev=pos_dev, directed_pos_dev=directed_pos_dev, rot_dev=rot_dev)
        return ATEResult(trajectory=trajectory, abs_dev=ate_result)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Exports results as pandas dataframe
        """
        trajectory_data = np.c_[
            self.trajectory.timestamps,
            self.trajectory.path_lengths,
            self.trajectory.positions.xyz,
        ]

        if self.trajectory.has_orientation:
            trajectory_data = np.c_[trajectory_data, self.trajectory.rotations.as_quat()]

        deviation_data = np.c_[self.abs_dev.pos_dev, self.abs_dev.directed_pos_dev]

        if self.abs_dev.rot_dev:
            deviation_data = np.c_[deviation_data, self.abs_dev.rot_dev.as_quat()]

        all_data = np.c_[trajectory_data, deviation_data]

        return pd.DataFrame(all_data, columns=self.columns)

    def to_file(self, filename: str, mode: str = "a") -> None:
        """
        Exports results as csv
        """
        with open(filename, mode, newline="\n", encoding="utf-8") as file:
            file.write(f"#name {self.name}\n")
            file.write(f"#epsg {self.trajectory.positions.epsg}\n")
            file.write(f"#sorting {self.trajectory.sorting.value}\n")
        self.to_dataframe().to_csv(filename, index=False, mode="a", float_format="%.12f")
