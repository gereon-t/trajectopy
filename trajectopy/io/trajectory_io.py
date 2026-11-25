"""Reading and writing of trajectories (trajectopy ASCII format). For KML, see :mod:`trajectopy.io.kml`."""

import io
import os

import numpy as np
import pandas as pd

from trajectopy.core.trajectory import Trajectory
from trajectopy.io import ascii
from trajectopy.io.header import HeaderData
from trajectopy.utils.definitions import Sorting


def _trajectory_from_data(header_data: HeaderData, trajectory_data: np.ndarray) -> Trajectory:
    if trajectory_data.ndim != 2 or trajectory_data.shape[1] < len(header_data.fields):
        columns = trajectory_data.shape[1] if trajectory_data.ndim == 2 else 1
        raise ValueError(
            f"The data has {columns} columns, but the header lists {len(header_data.fields)} fields "
            f"({','.join(header_data.fields)}). Check the #fields and #delimiter lines."
        )
    timestamps = ascii.extract_trajectory_timestamps(header_data=header_data, trajectory_data=trajectory_data)
    positions = ascii.extract_trajectory_positions(header_data=header_data, trajectory_data=trajectory_data)
    _check_finite(timestamps, "timestamps")
    _check_finite(positions.xyz, "positions")
    return Trajectory(
        timestamps=timestamps,
        positions=positions,
        rotations=ascii.extract_trajectory_rotations(header_data=header_data, trajectory_data=trajectory_data),
        name=header_data.name,
        # derived quantities with invalid values are derived anew
        path_lengths=_finite_or_none(
            ascii.extract_trajectory_path_lengths(header_data=header_data, trajectory_data=trajectory_data)
        ),
        velocity_xyz=_finite_or_none(
            ascii.extract_trajectory_velocity_xyz(header_data=header_data, trajectory_data=trajectory_data)
        ),
        sorting=Sorting.from_str(header_data.sorting),
        position_std=ascii.extract_trajectory_position_std(header_data=header_data, trajectory_data=trajectory_data),
    )


def _check_finite(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        rows = np.flatnonzero(~np.isfinite(values).reshape(len(values), -1).all(axis=1))
        raise ValueError(
            f"Invalid {name} (missing, NaN or infinite) in data row(s) {', '.join(map(str, rows[:5] + 1))}."
        )


def _finite_or_none(values: np.ndarray | None) -> np.ndarray | None:
    return values if values is not None and np.all(np.isfinite(values)) else None


def read_trajectory(filename: str | os.PathLike, default_name: str = "") -> Trajectory:
    """Reads a trajectory from a trajectopy ASCII file.

    Args:
        filename (str): Path to the file.
        default_name (str, optional): Name of the trajectory if the file has no ``#name`` header line.
            Defaults to "Trajectory".

    Returns:
        Trajectory: The loaded trajectory.
    """
    filename = os.fspath(filename)
    header_data, trajectory_data = ascii.read_trajectory_data(filename)
    if default_name and "name" not in header_data.data:  # e.g. files without a header
        header_data.data["name"] = default_name
    return _trajectory_from_data(header_data, trajectory_data)


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
    if trajectory.position_std is not None:
        fields += ",sx,sy,sz"
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

    if trajectory.position_std is not None:
        trajectory_data = np.c_[trajectory_data, trajectory.position_std]

    output = io.StringIO()
    output.write("\n".join(header) + "\n")
    pd.DataFrame(trajectory_data).to_csv(output, header=False, index=False, float_format="%.9f")
    return output.getvalue()


def write_trajectory(trajectory: Trajectory, filename: str | os.PathLike, mode: str = "w") -> None:
    """Writes a trajectory to a trajectopy ASCII file.

    Args:
        trajectory (Trajectory): Trajectory to write.
        filename (str): The output file path.
        mode (str, optional): File open mode. Defaults to "w".
    """
    filename = os.fspath(filename)
    with open(filename, mode=mode, newline="\n", encoding="utf-8") as file:
        file.write(trajectory_to_string(trajectory))
