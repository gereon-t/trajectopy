"""Reading and writing of evaluation results and alignment parameters."""

import csv
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from trajectopy.core.parameters import AlignmentParameters, SensorRotationParameters
from trajectopy.core.positions import Positions
from trajectopy.core.settings import AlignmentEstimationSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.exceptions import TrajectopyIOError
from trajectopy.io.header import HeaderData
from trajectopy.results.alignment_result import AlignmentResult
from trajectopy.results.ate_result import AbsoluteTrajectoryDeviations, ATEResult
from trajectopy.results.rpe_result import RelativeTrajectoryDeviations, RPEResult
from trajectopy.utils.definitions import Sorting

# sign convention of the ATE deviations that are written: test minus reference, with the horizontal cross-track
# direction positive to the right; files with another one are from earlier versions (see convert_ate_deviations)
DEVIATIONS_CONVENTION = "test-reference-cross-right"
_TEST_MINUS_REFERENCE_CROSS_LEFT = "test-reference"


def convert_ate_deviations(
    convention: str, pos_dev: np.ndarray, directed_pos_dev: np.ndarray, rot_dev: Rotation | None
) -> tuple[np.ndarray, np.ndarray, Rotation | None]:
    """Expresses ATE deviations stored with an earlier sign convention in the current one.

    Args:
        convention (str): Convention of the stored deviations; empty for the earliest one.
        pos_dev (np.ndarray): Position deviations.
        directed_pos_dev (np.ndarray): Along-track, horizontal and vertical cross-track deviations.
        rot_dev (Rotation | None): Rotation deviations.

    Returns:
        tuple: The deviations as test minus reference, with horizontal cross-track positive to the right.
    """
    if convention == DEVIATIONS_CONVENTION:
        return pos_dev, directed_pos_dev, rot_dev
    # all earlier conventions: horizontal cross-track positive to the left
    directed_pos_dev = directed_pos_dev * [1.0, -1.0, 1.0]
    if convention != _TEST_MINUS_REFERENCE_CROSS_LEFT:
        # the earliest: position and rotation deviations reference minus test
        pos_dev = -pos_dev
        rot_dev = None if rot_dev is None else rot_dev.inv()
    return pos_dev, directed_pos_dev, rot_dev


# ---------------------------------------------------------------------------
# Alignment parameters
# ---------------------------------------------------------------------------


def write_alignment_parameters(parameters: AlignmentParameters, filename: str | os.PathLike) -> None:
    """Writes alignment parameters together with their covariance matrix to a CSV file.

    All parameters are written, whether they are enabled or not, one per line in the order translation x/y/z,
    rotation x/y/z, scale, time shift and leverarm x/y/z. Each line holds the value, whether the parameter is
    enabled and the corresponding row of the covariance matrix. [`read_alignment_parameters`][trajectopy.io.read_alignment_parameters] reads the file.

    Args:
        parameters (AlignmentParameters): Parameters to write.
        filename (str | os.PathLike): Path of the file; an existing file is overwritten.

    See Also:
        [`trajectopy.io.read_alignment_parameters`][trajectopy.io.read_alignment_parameters]: reads the file.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> alignment = tpy.estimate_alignment(estimate, reference)
        >>> from trajectopy import io
        >>> io.write_alignment_parameters(alignment.position_parameters, "parameters.csv")
    """
    filename = os.fspath(filename)
    parameters.to_dataframe().to_csv(filename, index=False, header=False, float_format="%.17g")


def read_alignment_parameters(filename: str | os.PathLike) -> AlignmentParameters:
    """Reads alignment parameters written by [`write_alignment_parameters`][trajectopy.io.write_alignment_parameters].

    Args:
        filename (str | os.PathLike): Path of the file.

    Returns:
        AlignmentParameters: The parameter values, which of them are enabled and, if the file contains it,
        their covariance matrix.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> alignment = tpy.estimate_alignment(estimate, reference)
        >>> from trajectopy import io
        >>> io.write_alignment_parameters(alignment.position_parameters, "parameters.csv")
        >>> parameters = io.read_alignment_parameters("parameters.csv")
        >>> parameters.num_enabled
        6
    """
    filename = os.fspath(filename)
    alignment_data = pd.read_csv(filename, comment="#", header=None)
    params = AlignmentParameters()
    params.values = alignment_data.iloc[:11, 0].to_numpy()
    params.enabled_bool_list = [str(item).lower() != "false" for item in alignment_data.iloc[:11, 1].to_list()]

    if alignment_data.shape[1] > 2:
        params.set_covariance_matrix(alignment_data.iloc[:11, 2:].to_numpy())

    return params


def write_sensor_rotation_parameters(parameters: SensorRotationParameters, filename: str | os.PathLike) -> None:
    """Appends sensor rotation parameters to a file, e.g. to an alignment result written before.

    Three lines are appended, one per rotation angle (x, y, z): the value in radians and whether it is
    enabled.

    Args:
        parameters (SensorRotationParameters): Parameters to write.
        filename (str | os.PathLike): Path of the file.

    See Also:
        [`trajectopy.io.write_alignment_result`][trajectopy.io.write_alignment_result]: writes the complete alignment.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> alignment = tpy.estimate_alignment(estimate, reference)
        >>> from trajectopy import io
        >>> io.write_alignment_result(alignment, "alignment.result")  # also writes the sensor rotation
    """
    filename = os.fspath(filename)
    with open(filename, mode="a", encoding="utf-8") as file:
        file.write(f"{parameters.sensor_rot_x.value},{parameters.sensor_rot_x.enabled}\n")
        file.write(f"{parameters.sensor_rot_y.value},{parameters.sensor_rot_y.enabled}\n")
        file.write(f"{parameters.sensor_rot_z.value},{parameters.sensor_rot_z.enabled}\n")


def read_sensor_rotation_parameters(filename: str | os.PathLike) -> SensorRotationParameters:
    """Reads the sensor rotation parameters of an alignment result file.

    The sensor rotation follows the eleven alignment parameters, so that the file has 14 data lines.

    Args:
        filename (str | os.PathLike): Path of an alignment result file.

    Returns:
        SensorRotationParameters: The rotation angles (radians) and which of them are enabled.

    Raises:
        TrajectopyIOError: If the file does not contain exactly 14 data lines.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> alignment = tpy.estimate_alignment(estimate, reference)
        >>> from trajectopy import io
        >>> io.write_alignment_result(alignment, "alignment.result")
        >>> io.read_sensor_rotation_parameters("alignment.result").num_enabled
        0
    """
    filename = os.fspath(filename)
    alignment_data = pd.read_csv(filename, comment="#", header=None)

    if len(alignment_data) != 14:
        raise TrajectopyIOError("Alignment file with sensor rotation information must contain 14 lines!")

    params = SensorRotationParameters()
    params.values = alignment_data.iloc[11:, 0].to_numpy()
    params.enabled_bool_list = [str(item).lower() != "false" for item in alignment_data.iloc[11:, 1].to_list()]
    return params


# ---------------------------------------------------------------------------
# Alignment result
# ---------------------------------------------------------------------------


def write_alignment_result(result: AlignmentResult, filename: str | os.PathLike, *, mode: str = "w") -> None:
    """Writes an alignment result to a file that [`read_alignment_result`][trajectopy.io.read_alignment_result] and the GUI can read.

    Args:
        result (AlignmentResult): Result to write.
        filename (str | os.PathLike): Path of the file.
        mode (str, optional): ``"w"`` (default) overwrites the file, ``"a"`` appends to it.

    See Also:
        [`trajectopy.AlignmentResult.to_file`][trajectopy.AlignmentResult.to_file]: the same as a method.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> alignment = tpy.estimate_alignment(estimate, reference)
        >>> from trajectopy import io
        >>> io.write_alignment_result(alignment, "alignment.result")
        >>> io.read_alignment_result("alignment.result") == alignment
        True
    """
    filename = os.fspath(filename)
    with open(filename, mode, newline="\n", encoding="utf-8") as file:
        file.write("#type Alignment\n")  # so that the GUI recognizes the file
        file.write(f"#name {result.name}\n")
        if not result.converged:  # files without this line hold converged results
            file.write("#converged false\n")

    result.position_parameters.to_dataframe().to_csv(
        filename, header=False, index=False, mode="a", float_format="%.17g"
    )
    write_sensor_rotation_parameters(result.rotation_parameters, filename)


def read_alignment_result(filename: str | os.PathLike) -> AlignmentResult:
    """Reads an alignment result written by [`write_alignment_result`][trajectopy.io.write_alignment_result].

    Args:
        filename (str | os.PathLike): Path of the file.

    Returns:
        AlignmentResult: The alignment parameters, the sensor rotation, which of them were estimated and whether
        the estimation converged.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> alignment = tpy.estimate_alignment(estimate, reference)
        >>> from trajectopy import io
        >>> io.write_alignment_result(alignment, "alignment.result")
        >>> io.read_alignment_result("alignment.result").converged
        True
    """
    filename = os.fspath(filename)
    header_data = HeaderData.from_file(filename)
    estimated_parameters = read_alignment_parameters(filename)
    sensor_rot_parameters = read_sensor_rotation_parameters(filename)
    return AlignmentResult(
        name=str(header_data.data.get("name", "Alignment")),
        position_parameters=estimated_parameters,
        estimation_of=AlignmentEstimationSettings.from_bool_list(
            estimated_parameters.enabled_bool_list + sensor_rot_parameters.enabled_bool_list
        ),
        rotation_parameters=sensor_rot_parameters,
        converged=str(header_data.data.get("converged", "true")).lower() != "false",
    )


# ---------------------------------------------------------------------------
# ATE result
# ---------------------------------------------------------------------------


def write_ate_result(result: ATEResult, filename: str | os.PathLike, *, mode: str = "w") -> None:
    """Writes an ATE result as CSV file with a metadata header, which [`read_ate_result`][trajectopy.io.read_ate_result] and the GUI
    can read.

    The file contains the deviations of each pose together with the timestamps, path lengths and
    positions of the evaluated trajectory (see [`ATEResult.to_dataframe`][trajectopy.ATEResult.to_dataframe]).

    Args:
        result (ATEResult): Result to write.
        filename (str | os.PathLike): Path of the file.
        mode (str, optional): ``"w"`` (default) overwrites the file, ``"a"`` appends to it.

    See Also:
        [`trajectopy.ATEResult.to_file`][trajectopy.ATEResult.to_file]: the same as a method.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> ate_result = tpy.ate(estimate, reference)
        >>> from trajectopy import io
        >>> io.write_ate_result(ate_result, "ate.result")
        >>> len(io.read_ate_result("ate.result")) == len(ate_result)
        True
    """
    filename = os.fspath(filename)
    with open(filename, mode, newline="\n", encoding="utf-8") as file:
        file.write("#type ATE Result\n")  # so that the GUI recognizes the file
        file.write(f"#name {result.name}\n")
        file.write(f"#epsg {result.trajectory.positions.epsg}\n")
        file.write(f"#sorting {result.trajectory.sorting.value}\n")
        file.write(f"#deviations {DEVIATIONS_CONVENTION}\n")
        file.write(f"#directed_by {'rotations' if result.abs_dev.rotations_used else 'positions'}\n")
    result.to_dataframe().to_csv(filename, index=False, mode="a", float_format="%.12f")


def read_ate_result(filename: str | os.PathLike) -> ATEResult:
    """Reads an ATE result written by [`write_ate_result`][trajectopy.io.write_ate_result].

    Files written by trajectopy versions with an earlier sign convention of the deviations are
    converted to the current one (test minus reference).

    Args:
        filename (str | os.PathLike): Path of the file.

    Returns:
        ATEResult: The result. Its ``alignment`` is None, since result files do not store it.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> ate_result = tpy.ate(estimate, reference)
        >>> from trajectopy import io
        >>> io.write_ate_result(ate_result, "ate.result")
        >>> io.read_ate_result("ate.result").alignment is None
        True
    """
    filename = os.fspath(filename)
    header_data = HeaderData.from_file(filename)
    deviation_data = pd.read_csv(filename, comment="#")
    sorting = Sorting(header_data.sorting)

    # the rows in the order the trajectory will have, so that the deviations stay with their poses: one pose
    # per timestamp and, if sorted by path length, ties in chronological order
    tstamps = deviation_data["time"].to_numpy(dtype=float)
    chronological = np.argsort(tstamps, kind="stable")
    order = chronological[np.diff(tstamps[chronological], prepend=np.nan) != 0]
    if sorting == Sorting.PATH_LENGTH:
        path_lengths = deviation_data["arc_lengths"].to_numpy(dtype=float)
        order = order[np.argsort(path_lengths[order], kind="stable")]
    deviation_data = deviation_data.iloc[order]

    tstamps = deviation_data["time"].to_numpy(dtype=float)
    path_lengths = deviation_data["arc_lengths"].to_numpy(dtype=float)
    positions = Positions(deviation_data[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float), epsg=header_data.epsg)

    rot_columns = ["rot_x", "rot_y", "rot_z", "rot_w"]
    if all(column in deviation_data.columns for column in rot_columns):
        rotations = Rotation.from_quat(deviation_data[rot_columns].to_numpy(dtype=float))
    else:
        rotations = None

    pos_dev = deviation_data[["pos_dev_x", "pos_dev_y", "pos_dev_z"]].to_numpy(dtype=float)
    directed_pos_dev = deviation_data[["pos_dev_along", "pos_dev_cross_h", "pos_dev_cross_v"]].to_numpy(dtype=float)

    rot_dev_columns = ["rot_dev_x", "rot_dev_y", "rot_dev_z", "rot_dev_w"]
    if all(column in deviation_data.columns for column in rot_dev_columns):
        rot_dev = Rotation.from_quat(deviation_data[rot_dev_columns].to_numpy(dtype=float))
    else:
        rot_dev = None

    trajectory = Trajectory(
        name=header_data.name,
        positions=positions,
        rotations=rotations,
        timestamps=tstamps,
        path_lengths=path_lengths,
        sorting=sorting,
    )
    pos_dev, directed_pos_dev, rot_dev = convert_ate_deviations(
        header_data.deviations, pos_dev, directed_pos_dev, rot_dev
    )
    if directed_by := str(header_data.data.get("directed_by", "")):
        rotations_used = directed_by == "rotations"
    else:
        # earlier versions did not write it: rotations were used if either trajectory had some
        rotations_used = rot_dev is not None or rotations is not None
    abs_dev = AbsoluteTrajectoryDeviations(
        pos_dev=pos_dev, directed_pos_dev=directed_pos_dev, rot_dev=rot_dev, rotations_used=rotations_used
    )
    return ATEResult(trajectory=trajectory, abs_dev=abs_dev)


# ---------------------------------------------------------------------------
# RPE result
# ---------------------------------------------------------------------------


def write_rpe_result(result: RPEResult, filename: str | os.PathLike, *, mode: str = "w") -> None:
    """Writes an RPE result as CSV file with a metadata header, which [`read_rpe_result`][trajectopy.io.read_rpe_result] and the GUI
    can read.

    Args:
        result (RPEResult): Result to write.
        filename (str | os.PathLike): Path of the file.
        mode (str, optional): ``"w"`` (default) overwrites the file, ``"a"`` appends to it.

    See Also:
        [`trajectopy.RPEResult.to_file`][trajectopy.RPEResult.to_file]: the same as a method.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> rpe_result = tpy.rpe(estimate, reference)
        >>> from trajectopy import io
        >>> io.write_rpe_result(rpe_result, "rpe.result")
        >>> len(io.read_rpe_result("rpe.result")) == len(rpe_result)
        True
    """
    filename = os.fspath(filename)
    with open(filename, mode, encoding="utf-8", newline="") as file:
        file.write("#type RPE Result\n")  # so that the GUI recognizes the file
        file.write(f"#relative_dist_unit {result.rpe_dev.pair_distance_unit.name}\n")
        file.write(f"#name {result.name}\n")

        writer = csv.writer(file)
        file.write("#num_pairs ")
        writer.writerow(result.num_pairs)
    result.to_dataframe().to_csv(filename, index=False, mode="a", float_format="%.12f")


def read_rpe_result(filename: str | os.PathLike) -> RPEResult:
    """Reads an RPE result written by [`write_rpe_result`][trajectopy.io.write_rpe_result].

    Args:
        filename (str | os.PathLike): Path of the file.

    Returns:
        RPEResult: The result.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> rpe_result = tpy.rpe(estimate, reference)
        >>> from trajectopy import io
        >>> io.write_rpe_result(rpe_result, "rpe.result")
        >>> io.read_rpe_result("rpe.result").pos_drift_unit
        '%'
    """
    filename = os.fspath(filename)
    header_data = HeaderData.from_file(filename)
    deviation_data = pd.read_csv(filename, comment="#")

    pos_dev: dict[float, np.ndarray] = {}
    rot_dev: dict[float, np.ndarray] = {}
    pair_distance: dict[float, np.ndarray] = {}

    last_index = 0
    for index in header_data.num_pairs:
        dev_block = deviation_data.iloc[last_index : last_index + index, :]
        pair_distances = dev_block["pair_distance"].to_numpy(dtype=float)
        pos_devs = dev_block["pos_dev"].to_numpy(dtype=float)
        rot_devs = dev_block["rot_dev"].to_numpy(dtype=float) if "rot_dev" in deviation_data.columns else np.array([])

        mean_dist = float(np.mean(pair_distances))
        pair_distance[mean_dist] = pair_distances
        pos_dev[mean_dist] = pos_devs
        rot_dev[mean_dist] = rot_devs

        last_index += index

    rpe_dev = RelativeTrajectoryDeviations(
        pair_distance=pair_distance,
        pos_dev=pos_dev,
        rot_dev=rot_dev,
        pair_distance_unit=header_data.relative_dist_unit,
    )

    return RPEResult(rpe_dev=rpe_dev, name=header_data.name)
