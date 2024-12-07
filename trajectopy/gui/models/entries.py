"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

import trajectopy.api as tpy
from trajectopy.core.input_output.header import HeaderData

logger = logging.getLogger("root")


@dataclass
class TrajectoryProcessingState:
    """
    Class to store the processing state of a trajectory.
    For example, if a trajectory is interpolated, the attribute
    interpolated is set to True.
    """

    approximated: bool = False
    interpolated: bool = False
    intersected: bool = False
    aligned: bool = False
    matched: bool = False
    sorting_known: bool = False

    def __str__(self) -> str:
        return ", ".join([str(key) for key, value in self.__dict__.items() if value])

    @classmethod
    def from_string(cls, input_string: str) -> "TrajectoryProcessingState":
        return cls(
            approximated="approximated" in input_string,
            interpolated="interpolated" in input_string,
            intersected="intersected" in input_string,
            aligned="aligned" in input_string,
            matched="matched" in input_string,
            sorting_known="sorting_known" in input_string,
        )


def bool_to_str(input_bool: bool) -> str:
    return "yes" if input_bool else "no"


def generate_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Entry(ABC):
    """Abstract base class for all entries in their respective model."""

    entry_id: str = field(init=False, default_factory=generate_id)
    time: str = field(init=False, default_factory=lambda: str(datetime.now()))

    def __post_init__(self) -> None:
        logger.debug("Created new %s with id: %s", self.type, self.entry_id)

    def renew_id(self) -> None:
        self.entry_id = generate_id()

    def set_id(self, entry_id: str) -> None:
        self.entry_id = entry_id

    @property
    def type(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def property_dict(self) -> Dict[str, str]:
        pass

    def to_file(self, filename: str) -> None:
        with open(filename, "w", newline="\n", encoding="utf-8") as file:
            file.write(f"#id {self.entry_id}\n")
            file.write(f"#type {self.type}\n")


@dataclass
class TrajectoryEntry(Entry):
    """Class representing a trajectory entry in the trajectory model."""

    full_filename: str
    trajectory: tpy.Trajectory
    set_as_reference: bool = False
    settings: tpy.ProcessingSettings = field(default_factory=tpy.ProcessingSettings)
    group_id: str = field(default_factory=generate_id)
    state: TrajectoryProcessingState = field(default_factory=TrajectoryProcessingState)

    def __len__(self) -> int:
        return len(self.trajectory)

    def to_file(self, filename: str) -> None:
        super().to_file(filename)
        self.trajectory.to_file(filename=filename, mode="a")

    @classmethod
    def from_file(cls, trajectory_filename: Path, settings_filename: Path) -> "TrajectoryEntry":
        """Creates a new TrajectoryEntry from a trajectory file and a settings file."""
        header_data = HeaderData.from_file(str(trajectory_filename))
        trajectory = tpy.Trajectory.from_file(str(trajectory_filename))
        if settings_filename.is_file():
            logger.info("Using existing settings file: %s", settings_filename)
            traj_settings = tpy.ProcessingSettings.from_file(str(settings_filename))
        else:
            logger.info(
                "No settings file found. Settings can be provided by storing a yaml file with the same name in the same directory."
            )
            traj_settings = tpy.ProcessingSettings()

        if trajectory is None:
            raise ValueError(
                "This file does not seem to have correct trajectory information (Time, X, Y, Z, qx, qy, qz, qw)!"
            )

        traj_entry = TrajectoryEntry(
            full_filename=str(trajectory_filename),
            trajectory=trajectory,
            settings=traj_settings,
        )
        if header_data.id:
            traj_entry.set_id(entry_id=header_data.id)
        return traj_entry

    @property
    def name(self) -> str:
        return self.trajectory.name

    @name.setter
    def name(self, name: str) -> None:
        self.trajectory.name = name

    @property
    def column(self) -> Tuple[str, str, str, int, str, str]:
        return (
            self.name,
            bool_to_str(self.set_as_reference),
            self.trajectory.sorting.value,
            self.trajectory.pos.epsg,
            str(self.state),
            self.full_filename,
        )

    @property
    def has_orientations(self) -> str:
        return bool_to_str(self.trajectory.rot is not None)

    @property
    def filename(self) -> str:
        return os.path.basename(self.full_filename)

    @property
    def property_dict(self) -> Dict[str, str]:
        """Shows a new window with trajectory properties"""
        return {
            "Name": self.trajectory.name,
            "Date": f"{datetime.fromtimestamp(self.trajectory.tstamps[0]).strftime('%Y-%m-%d %H:%M:%S')} UTC - "
            f"{datetime.fromtimestamp(self.trajectory.tstamps[-1]).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "Duration": f"{timedelta(seconds=float(self.trajectory.tstamps[-1] - self.trajectory.tstamps[0]))}",
            "EPSG": (
                f"{self.trajectory.pos.crs}, {self.trajectory.pos.crs.name}"
                if self.trajectory.pos.crs is not None
                else "local / unknown"
            ),
            "Orientation available": "yes" if self.trajectory.rot is not None else "no",
            "Number of Poses": str(len(self.trajectory)),
            "Sort By:": self.trajectory.sorting.value,
            "Length [m]": f"{self.trajectory.total_length:.3f}",
            "Data Rate [Hz]": f"{self.trajectory.data_rate:.3f}",
            "Minimum Speed [m/s]": f"{np.min(self.trajectory.speed):.3f}",
            "Maximum Speed [m/s]": f"{np.max(self.trajectory.speed):.3f}",
            "Average Speed [m/s]": f"{np.mean(self.trajectory.speed):.3f}",
            "Sorting known": "yes" if self.state.sorting_known else "no",
            "Approximated": "yes" if self.state.approximated else "no",
            "Intersected": "yes" if self.state.intersected else "no",
            "Interpolated": "yes" if self.state.interpolated else "no",
            "Matched Timestamps": "yes" if self.state.matched else "no",
            "Filename": self.full_filename,
            "UUID": self.entry_id,
        }


@dataclass
class ResultEntry(Entry, ABC):
    """Abstract base class for result entries in the result model."""

    @abstractmethod
    def __len__(self) -> int:
        return 0

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @name.setter
    @abstractmethod
    def name(self, name: str) -> None:
        pass

    @property
    def column(self) -> Tuple[str, str, int, str]:
        return self.name, self.type, len(self), self.entry_id

    @classmethod
    def from_file(cls, filename: str) -> "ResultEntry":
        """
        Reads a result file and returns an object of the appropriate type based on the file's header data.

        Args:
            filename (str): The path to the result file to read.

        Returns:
            Union[AbsoluteDeviationEntry, RelativeDeviationEntry, AlignmentEntry]: An object of the appropriate type based on the file's header data.

        Raises:
            ValueError: If the result file type is not supported.
        """
        header_data = HeaderData.from_file(filename)

        if header_data.type == AbsoluteDeviationEntry.__name__.lower():
            logger.info("Detected Absolute Deviations file.")
            return AbsoluteDeviationEntry.from_file(filename)

        if header_data.type == RelativeDeviationEntry.__name__.lower():
            logger.info("Detected Relative Deviations file.")
            return RelativeDeviationEntry.from_file(filename)

        if header_data.type == "alignmententry":
            logger.info("Detected Alignment file.")
            return AlignmentEntry.from_file(filename)

        raise ValueError(f"No supported result type '{header_data.type}'")


@dataclass
class DeviationsEntry(ResultEntry, ABC):
    """Abstract base class for deviation entries in the result model."""

    deviations: Union[tpy.ATEResult, tpy.RPEResult]

    @property
    def name(self) -> str:
        return self.deviations.name

    @name.setter
    def name(self, name: str) -> None:
        self.deviations.name = name

    @property
    def property_dict(self) -> Dict[str, str]:
        return self.deviations.property_dict


@dataclass
class AbsoluteDeviationEntry(DeviationsEntry):
    """Class representing a absolute deviation entry in the result model."""

    deviations: tpy.ATEResult

    def __len__(self) -> int:
        return len(self.deviations.abs_dev.pos_dev)

    def to_file(self, filename: str) -> None:
        super().to_file(filename=filename)
        self.deviations.to_file(filename)

    @classmethod
    def from_file(cls, filename: str) -> "AbsoluteDeviationEntry":
        deviations = tpy.ATEResult.from_file(filename)
        abs_dev_entry = cls(deviations=deviations)
        abs_dev_entry.set_id(entry_id=HeaderData.from_file(filename).id)
        return abs_dev_entry


@dataclass
class RelativeDeviationEntry(DeviationsEntry):
    """Class representing a relative deviation entry in the result model."""

    deviations: tpy.RPEResult

    def __len__(self) -> int:
        return self.deviations.rpe_dev.num_pairs

    def to_file(self, filename: str) -> None:
        super().to_file(filename=filename)
        self.deviations.to_file(filename)

    @classmethod
    def from_file(cls, filename: str) -> "RelativeDeviationEntry":
        deviations = tpy.RPEResult.from_file(filename)
        rel_dev_entry = cls(deviations=deviations)
        rel_dev_entry.set_id(entry_id=HeaderData.from_file(filename).id)
        return rel_dev_entry


@dataclass
class AlignmentEntry(ResultEntry):
    """Entry storing alignment results."""

    alignment_result: tpy.AlignmentResult = field(default_factory=tpy.AlignmentResult)

    def __len__(self) -> int:
        return (
            self.alignment_result.position_parameters.num_enabled
            + self.alignment_result.rotation_parameters.num_enabled
        )

    @property
    def estimated_parameters(self) -> tpy.AlignmentParameters:
        return self.alignment_result.position_parameters

    @property
    def estimation_of(self) -> tpy.AlignmentEstimationSettings:
        return self.alignment_result.estimation_of

    @property
    def name(self) -> str:
        return self.alignment_result.name

    @name.setter
    def name(self, name: str) -> None:
        self.alignment_result.name = name

    @property
    def property_dict(self) -> Dict[str, str]:
        if self.alignment_result.position_parameters is None:
            return {}

        return {
            "Translation x [m]": f"{self.alignment_result.position_parameters.helmert.trans_x.value:<10.4f} s-dev.: {np.sqrt(self.alignment_result.position_parameters.helmert.trans_x.variance):<10.4f}",
            "Translation y [m]": f"{self.alignment_result.position_parameters.helmert.trans_y.value:<10.4f} s-dev.: {np.sqrt(self.alignment_result.position_parameters.helmert.trans_y.variance):<10.4f}",
            "Translation z [m]": f"{self.alignment_result.position_parameters.helmert.trans_z.value:<10.4f} s-dev.: {np.sqrt(self.alignment_result.position_parameters.helmert.trans_z.variance):<10.4f}",
            "Rotation x [°]": f"{np.rad2deg(self.alignment_result.position_parameters.helmert.rot_x.value):<10.4f} s-dev.: {np.rad2deg(np.sqrt(self.alignment_result.position_parameters.helmert.rot_x.variance)):<10.4f}",
            "Rotation y [°]": f"{np.rad2deg(self.alignment_result.position_parameters.helmert.rot_y.value):<10.4f} s-dev.: {np.rad2deg(np.sqrt(self.alignment_result.position_parameters.helmert.rot_y.variance)):<10.4f}",
            "Rotation z [°]": f"{np.rad2deg(self.alignment_result.position_parameters.helmert.rot_z.value):<10.4f} s-dev.: {np.rad2deg(np.sqrt(self.alignment_result.position_parameters.helmert.rot_z.variance)):<10.4f}",
            "Scale [-]": f"{self.alignment_result.position_parameters.helmert.scale.value:<10.4f} s-dev.: {np.sqrt(self.alignment_result.position_parameters.helmert.scale.variance)*1e6:<10.4f} ppm",
            "Time Shift [s]": f"{self.alignment_result.position_parameters.time_shift.value:<10.4f} s-dev.: {np.sqrt(self.alignment_result.position_parameters.time_shift.variance):<10.4f}",
            "Leverarm x [m]": f"{self.alignment_result.position_parameters.leverarm.x.value:<10.4f} s-dev.: {np.sqrt(self.alignment_result.position_parameters.leverarm.x.variance):<10.4f}",
            "Leverarm y [m]": f"{self.alignment_result.position_parameters.leverarm.y.value:<10.4f} s-dev.: {np.sqrt(self.alignment_result.position_parameters.leverarm.y.variance):<10.4f}",
            "Leverarm z [m]": f"{self.alignment_result.position_parameters.leverarm.z.value:<10.4f} s-dev.: {np.sqrt(self.alignment_result.position_parameters.leverarm.z.variance):<10.4f}",
            "Sensor Rotation x [°]": f"{np.rad2deg(self.alignment_result.rotation_parameters.sensor_rot_x.value):<10.4f}",
            "Sensor Rotation y [°]": f"{np.rad2deg(self.alignment_result.rotation_parameters.sensor_rot_y.value):<10.4f}",
            "Sensor Rotation z [°]": f"{np.rad2deg(self.alignment_result.rotation_parameters.sensor_rot_z.value):<10.4f}",
        }

    def to_file(self, filename: str) -> None:
        if self.alignment_result.position_parameters is None:
            raise ValueError("No estimated parameters available!")

        super().to_file(filename=filename)
        self.alignment_result.to_file(filename=filename)

    @classmethod
    def from_file(cls, filename: str) -> "AlignmentEntry":
        """Creates a new AlignmentEntry from a file."""
        alignment_result = tpy.AlignmentResult.from_file(filename)
        alignment_entry = cls(alignment_result=alignment_result)
        alignment_entry.set_id(entry_id=HeaderData.from_file(filename).id)
        return alignment_entry


@dataclass
class PropertyEntry:
    name: str
    values: Tuple[str, ...]

    @property
    def column(self) -> Tuple[str, ...]:
        return (self.name, *self.values)
