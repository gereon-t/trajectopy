"""Reading and writing of trajectories (trajectopy ASCII format). For KML, see :mod:`trajectopy.io.kml`."""

import io

import numpy as np
import pandas as pd

from trajectopy.core.trajectory import Trajectory
from trajectopy.io import ascii
from trajectopy.io.header import HeaderData
from trajectopy.utils.definitions import Sorting


def _trajectory_from_data(header_data: HeaderData, trajectory_data: np.ndarray) -> Trajectory:
    return Trajectory(
        timestamps=ascii.extract_trajectory_timestamps(header_data=header_data, trajectory_data=trajectory_data),
        positions=ascii.extract_trajectory_positions(header_data=header_data, trajectory_data=trajectory_data),
        rotations=ascii.extract_trajectory_rotations(header_data=header_data, trajectory_data=trajectory_data),
        name=header_data.name,
        path_lengths=ascii.extract_trajectory_path_lengths(header_data=header_data, trajectory_data=trajectory_data),
        velocity_xyz=ascii.extract_trajectory_velocity_xyz(header_data=header_data, trajectory_data=trajectory_data),
        sorting=Sorting.from_str(header_data.sorting),
    )


def read_trajectory(filename: str) -> Trajectory:
    """Reads a trajectory from a trajectopy ASCII file.

    Args:
        filename (str): Path to the file.

    Returns:
        Trajectory: The loaded trajectory.
    """
    return _trajectory_from_data(*ascii.read_data(filename, dtype=object))


def read_trajectory_string(content: str) -> Trajectory:
    """Reads a trajectory from the string content of a trajectopy ASCII file.

    Args:
        content (str): File content including the header.

    Returns:
        Trajectory: The loaded trajectory.
    """
    return _trajectory_from_data(*ascii.read_string(content, dtype=object))


def trajectory_to_string(trajectory: Trajectory) -> str:
    """Serializes a trajectory to a CSV-formatted string with metadata headers.

    Headers included: #epsg, #name, #nframe, #sorting, #fields.
    """
    fields = "t,l,px,py,pz,vx,vy,vz" if trajectory.rotations is None else "t,l,px,py,pz,qx,qy,qz,qw,vx,vy,vz"
    header = [
        f"#epsg {trajectory.positions.epsg}",
        f"#name {trajectory.name}",
        "#nframe enu",
        f"#sorting {trajectory.sorting.value}",
        f"#fields {fields}",
    ]

    if trajectory.rotations is None:
        trajectory_data = np.c_[
            trajectory.timestamps, trajectory.path_lengths, trajectory.positions.xyz, trajectory.velocity_xyz
        ]
    else:
        trajectory_data = np.c_[
            trajectory.timestamps,
            trajectory.path_lengths,
            trajectory.positions.xyz,
            trajectory.rotations.as_quat(),
            trajectory.velocity_xyz,
        ]

    output = io.StringIO()
    output.write("\n".join(header) + "\n")
    pd.DataFrame(trajectory_data).to_csv(output, header=False, index=False, float_format="%.9f")
    return output.getvalue()


def write_trajectory(trajectory: Trajectory, filename: str, mode: str = "w") -> None:
    """Writes a trajectory to a trajectopy ASCII file.

    Args:
        trajectory (Trajectory): Trajectory to write.
        filename (str): The output file path.
        mode (str, optional): File open mode. Defaults to "w".
    """
    with open(filename, mode=mode, newline="\n", encoding="utf-8") as file:
        file.write(trajectory_to_string(trajectory))
