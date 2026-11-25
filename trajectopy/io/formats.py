"""Reading and writing of common third-party trajectory formats (TUM, KITTI, EuRoC).

Files in these formats have no trajectopy header, so they are either read with an explicit ``format`` or
recognized by :func:`detect_format`:

- ``tum``: ``timestamp tx ty tz qx qy qz qw`` separated by spaces (TUM RGB-D benchmark).
- ``kitti``: one row-major 3x4 pose matrix (12 values) per line, without timestamps (KITTI odometry benchmark).
  The poses are numbered unless timestamps are given (e.g. from KITTI's ``times.txt``).
- ``euroc``: ground truth CSV of the EuRoC MAV dataset: ``timestamp [ns], px, py, pz, qw, qx, qy, qz, ...``.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.utils.definitions import OptionEnum

logger = logging.getLogger(__name__)


class TrajectoryFormat(OptionEnum):
    """File formats of trajectories."""

    AUTO = "auto"
    TRAJECTOPY = "trajectopy"
    TUM = "tum"
    KITTI = "kitti"
    EUROC = "euroc"

    @classmethod
    def _missing_(cls, value: object) -> "TrajectoryFormat":
        if (member := super()._missing_(value)) is not None:
            return member  # type: ignore[return-value]
        options = ", ".join(item.value for item in cls)
        raise ValueError(f"Unknown trajectory format '{value}'. Options: {options}.")


_KITTI_COLUMNS = 12
_DETECTION_LINES = 50


def detect_format(filename: str | os.PathLike) -> TrajectoryFormat:
    """Recognizes the format of a trajectory file from its first lines.

    Files with a ``#fields`` header line are trajectopy files. Without one, a ``#timestamp`` header with comma
    separated values indicates EuRoC, and 12 values per line indicate KITTI. All other files are read as
    trajectopy files, whose default columns (``t,px,py,pz,qx,qy,qz,qw``) are those of the TUM format.

    Args:
        filename (str): Path to the file.

    Returns:
        TrajectoryFormat: The recognized format.
    """
    filename = os.fspath(filename)
    first_data_line = ""
    with open(filename, encoding="utf-8", errors="replace") as file:
        for number, line in enumerate(file):
            stripped = line.strip()
            if stripped.startswith("#fields"):
                return TrajectoryFormat.TRAJECTOPY
            if number == 0 and stripped.lower().startswith("#timestamp") and "," in stripped:
                return TrajectoryFormat.EUROC
            if stripped and not stripped.startswith("#"):
                first_data_line = stripped
                break
            if number > _DETECTION_LINES:
                break

    values = first_data_line.replace(",", " ").split()
    if len(values) == _KITTI_COLUMNS and _all_numeric(values):
        return TrajectoryFormat.KITTI
    return TrajectoryFormat.TRAJECTOPY


def _all_numeric(values: list[str]) -> bool:
    try:
        [float(value) for value in values]
    except ValueError:
        return False
    return True


def _numeric_table(filename: str, delimiter: str | None) -> np.ndarray:
    sep = r"\s+" if delimiter is None else delimiter
    data = pd.read_csv(filename, comment="#", header=None, sep=sep, dtype=float, skipinitialspace=True)
    return data.to_numpy()


def _name(filename: str) -> str:
    return Path(filename).stem


def read_tum(filename: str | os.PathLike, name: str = "") -> Trajectory:
    """Reads a trajectory in the TUM format (``timestamp tx ty tz qx qy qz qw``).

    Args:
        filename (str): Path to the file.
        name (str, optional): Name of the trajectory. Defaults to the file name.

    Returns:
        Trajectory: The trajectory, in a local coordinate system (EPSG 0).
    """
    filename = os.fspath(filename)
    data = _numeric_table(filename, delimiter=None)
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError(
            f"A TUM file has 8 columns (timestamp tx ty tz qx qy qz qw), {filename} has {data.shape[-1]}."
        )
    return Trajectory(
        positions=Positions(xyz=data[:, 1:4]),
        rotations=Rotations.from_quat(data[:, 4:8]),
        timestamps=data[:, 0],
        name=name or _name(filename),
    )


def read_kitti(filename: str | os.PathLike, timestamps: np.ndarray | str | None = None, name: str = "") -> Trajectory:
    """Reads a trajectory in the KITTI odometry format (one row-major 3x4 pose matrix per line).

    Args:
        filename (str): Path to the file.
        timestamps (np.ndarray | str | None, optional): Timestamps of the poses, or the path to a file with one
            timestamp per line (KITTI's ``times.txt``). Without them, the poses are numbered 0, 1, 2, ...
        name (str, optional): Name of the trajectory. Defaults to the file name.

    Returns:
        Trajectory: The trajectory, in a local coordinate system (EPSG 0).
    """
    filename = os.fspath(filename)
    data = _numeric_table(filename, delimiter=None)
    if data.ndim != 2 or data.shape[1] != _KITTI_COLUMNS:
        raise ValueError(f"A KITTI file has 12 values per line (3x4 pose matrix), {filename} has {data.shape[-1]}.")

    matrices = data.reshape(-1, 3, 4)
    if isinstance(timestamps, str | Path):
        timestamps = np.loadtxt(timestamps, comments="#", ndmin=1)
    if timestamps is not None and len(timestamps) != len(matrices):
        raise ValueError(f"{len(timestamps)} timestamps were given for {len(matrices)} poses.")

    return Trajectory(
        positions=Positions(xyz=matrices[:, :, 3].copy()),
        rotations=Rotations.from_matrix(matrices[:, :, :3]),
        timestamps=None if timestamps is None else np.asarray(timestamps, dtype=float),
        name=name or _name(filename),
    )


def read_euroc(filename: str | os.PathLike, name: str = "") -> Trajectory:
    """Reads a trajectory in the EuRoC ground truth format.

    The columns are ``timestamp [ns], px, py, pz, qw, qx, qy, qz``, optionally followed by velocities
    ``vx, vy, vz`` and sensor biases (ignored).

    Args:
        filename (str): Path to the file.
        name (str, optional): Name of the trajectory. Defaults to the file name.

    Returns:
        Trajectory: The trajectory, in a local coordinate system (EPSG 0), with timestamps in seconds.
    """
    filename = os.fspath(filename)
    data = _numeric_table(filename, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError(
            f"A EuRoC file has at least 8 columns (timestamp, position, quaternion), {filename} has {data.shape[-1]}."
        )
    return Trajectory(
        positions=Positions(xyz=data[:, 1:4]),
        rotations=Rotations.from_quat(data[:, [5, 6, 7, 4]]),
        timestamps=data[:, 0] * 1e-9,
        velocity_xyz=data[:, 8:11] if data.shape[1] >= 11 else None,
        name=name or _name(filename),
    )


def read_trajectory(
    filename: str | os.PathLike,
    *,
    format: "str | TrajectoryFormat" = TrajectoryFormat.AUTO,  # noqa: A002
) -> Trajectory:
    """Reads a trajectory file in any supported format.

    Args:
        filename (str | os.PathLike): Path to the file.
        format (str | TrajectoryFormat, optional): One of ``auto``, ``trajectopy``, ``tum``, ``kitti`` and
            ``euroc``. With ``auto`` (default), the format is recognized by :func:`detect_format`.

    Returns:
        Trajectory: The loaded trajectory.
    """
    from trajectopy.io import trajectory_io

    filename = os.fspath(filename)
    file_format = TrajectoryFormat(format)
    if file_format == TrajectoryFormat.AUTO:
        file_format = detect_format(filename)
        if file_format != TrajectoryFormat.TRAJECTOPY:
            logger.info("Recognized %s as a %s file.", filename, file_format.value.upper())

    if file_format == TrajectoryFormat.TRAJECTOPY:
        return trajectory_io.read_trajectory(filename, default_name=_name(filename))
    readers: dict[TrajectoryFormat, Callable[[str], Trajectory]] = {
        TrajectoryFormat.TUM: read_tum,
        TrajectoryFormat.KITTI: read_kitti,
        TrajectoryFormat.EUROC: read_euroc,
    }
    return readers[file_format](filename)


def _quat(trajectory: Trajectory) -> np.ndarray:
    return trajectory.quat if trajectory.has_orientation else np.tile([0.0, 0.0, 0.0, 1.0], (len(trajectory), 1))


def to_tum_string(trajectory: Trajectory) -> str:
    """The trajectory in the TUM format (timestamp tx ty tz qx qy qz qw); missing orientations are written as
    identity."""
    data = np.c_[trajectory.timestamps, trajectory.xyz, _quat(trajectory)]
    return "".join(" ".join(f"{value:.9f}" for value in row) + "\n" for row in data)


def to_kitti_string(trajectory: Trajectory) -> str:
    """The trajectory in the KITTI format (row-major 3x4 pose matrices); missing orientations are written as
    identity. Timestamps are not part of the format."""
    matrices = Rotations.from_quat(_quat(trajectory)).as_matrix()
    poses = np.concatenate([matrices, trajectory.xyz[:, :, None]], axis=2).reshape(-1, 12)
    return "".join(" ".join(f"{value:.9e}" for value in row) + "\n" for row in poses)


def to_euroc_string(trajectory: Trajectory) -> str:
    """The trajectory in the EuRoC ground truth format (timestamp [ns], position, quaternion w first,
    velocity)."""
    quat = _quat(trajectory)
    header = (
        "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], "
        "v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1]\n"
    )
    lines = [
        f"{int(round(t * 1e9))}," + ",".join(f"{value:.9f}" for value in values) + "\n"
        for t, values in zip(
            trajectory.timestamps, np.c_[trajectory.xyz, quat[:, [3, 0, 1, 2]], trajectory.velocity_xyz]
        )
    ]
    return header + "".join(lines)


def write_trajectory(
    trajectory: Trajectory,
    filename: str | os.PathLike,
    *,
    format: "str | TrajectoryFormat" = TrajectoryFormat.AUTO,  # noqa: A002
    mode: str = "w",
) -> None:
    """Writes a trajectory in one of the supported formats.

    Args:
        trajectory (Trajectory): Trajectory to write.
        filename (str | os.PathLike): Output path.
        format (str | TrajectoryFormat, optional): One of ``auto`` (default), ``trajectopy``, ``tum``,
            ``kitti`` and ``euroc``. ``auto`` picks the format from the file extension (``.tum``, ``.kitti``)
            and writes trajectopy files otherwise.
        mode (str, optional): ``"w"`` (default) overwrites the file, ``"a"`` appends to it.
    """
    from trajectopy.io import trajectory_io

    filename = os.fspath(filename)
    file_format = TrajectoryFormat(format)
    if file_format == TrajectoryFormat.AUTO:
        suffixes = {".tum": TrajectoryFormat.TUM, ".kitti": TrajectoryFormat.KITTI}
        file_format = suffixes.get(Path(filename).suffix.lower(), TrajectoryFormat.TRAJECTOPY)

    if file_format == TrajectoryFormat.TRAJECTOPY:
        trajectory_io.write_trajectory(trajectory, filename, mode=mode)
        return

    writers = {
        TrajectoryFormat.TUM: to_tum_string,
        TrajectoryFormat.KITTI: to_kitti_string,
        TrajectoryFormat.EUROC: to_euroc_string,
    }
    with open(filename, mode, newline="\n", encoding="utf-8") as file:
        file.write(writers[file_format](trajectory))


# names used before trajectopy 7
read_trajectory_file = read_trajectory
write_trajectory_file = write_trajectory
