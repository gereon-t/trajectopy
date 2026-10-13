"""Reading and writing of evaluation results and alignment parameters."""

import csv

import numpy as np
import pandas as pd

from trajectopy.core.parameters import AlignmentParameters, SensorRotationParameters
from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import AlignmentEstimationSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.io.header import HeaderData
from trajectopy.results.alignment_result import AlignmentResult
from trajectopy.results.ate_result import AbsoluteTrajectoryDeviations, ATEResult
from trajectopy.results.rpe_result import RelativeTrajectoryDeviations, RPEResult
from trajectopy.utils.definitions import Sorting

# ---------------------------------------------------------------------------
# Alignment parameters
# ---------------------------------------------------------------------------


def write_alignment_parameters(parameters: AlignmentParameters, filename: str) -> None:
    """Writes alignment parameters together with their covariance matrix to a CSV file.

    All parameters are written no matter if they are enabled or not. The order of the parameters is:
    translation x/y/z, rotation x/y/z, scale, time shift, leverarm x/y/z. The covariance matrix is
    written row by row next to the parameters.
    """
    dataframe = pd.DataFrame(np.c_[parameters.values, parameters.covariance_matrix])
    dataframe.to_csv(filename, index=False, header=False)


def read_alignment_parameters(filename: str) -> AlignmentParameters:
    """Reads alignment parameters (values, enabled flags and optional covariance) from a file."""
    alignment_data = pd.read_csv(filename, comment="#", header=None)
    params = AlignmentParameters()
    params.values = alignment_data.iloc[:11, 0].to_numpy()
    params.enabled_bool_list = [str(item).lower() != "false" for item in alignment_data.iloc[:11, 1].to_list()]

    if alignment_data.shape[1] > 2:
        params.set_covariance_matrix(alignment_data.iloc[:11, 2:].to_numpy())

    return params


def write_sensor_rotation_parameters(parameters: SensorRotationParameters, filename: str) -> None:
    """Appends sensor rotation parameters to a file."""
    with open(filename, mode="a", encoding="utf-8") as file:
        file.write(f"{parameters.sensor_rot_x.value},{parameters.sensor_rot_x.enabled}\n")
        file.write(f"{parameters.sensor_rot_y.value},{parameters.sensor_rot_y.enabled}\n")
        file.write(f"{parameters.sensor_rot_z.value},{parameters.sensor_rot_z.enabled}\n")


def read_sensor_rotation_parameters(filename: str) -> SensorRotationParameters:
    """Reads sensor rotation parameters from an alignment file.

    Raises:
        ValueError: If the file doesn't contain exactly 14 lines.
    """
    alignment_data = pd.read_csv(filename, comment="#", header=None)

    if len(alignment_data) != 14:
        raise ValueError("Alignment file with sensor rotation information must contain 14 lines!")

    params = SensorRotationParameters()
    params.values = alignment_data.iloc[11:, 0].to_numpy()
    params.enabled_bool_list = [str(item).lower() != "false" for item in alignment_data.iloc[11:, 1].to_list()]
    return params


# ---------------------------------------------------------------------------
# Alignment result
# ---------------------------------------------------------------------------


def write_alignment_result(result: AlignmentResult, filename: str) -> None:
    """Appends an alignment result to a file."""
    if result.position_parameters is None:
        raise ValueError("No estimated parameters available!")

    with open(filename, "a", newline="\n", encoding="utf-8") as file:
        file.write(f"#name {result.name}\n")

    result.position_parameters.to_dataframe().to_csv(
        filename, header=False, index=False, mode="a", float_format="%.15f"
    )
    write_sensor_rotation_parameters(result.rotation_parameters, filename)


def read_alignment_result(filename: str) -> AlignmentResult:
    """Reads an alignment result from a file."""
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
    )


# ---------------------------------------------------------------------------
# ATE result
# ---------------------------------------------------------------------------


def write_ate_result(result: ATEResult, filename: str, mode: str = "a") -> None:
    """Writes an ATE result as CSV with a metadata header."""
    with open(filename, mode, newline="\n", encoding="utf-8") as file:
        file.write(f"#name {result.name}\n")
        file.write(f"#epsg {result.trajectory.positions.epsg}\n")
        file.write(f"#sorting {result.trajectory.sorting.value}\n")
    result.to_dataframe().to_csv(filename, index=False, mode="a", float_format="%.12f")


def read_ate_result(filename: str) -> ATEResult:
    """Reads an ATE result from a file."""
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
    directed_pos_dev = deviation_data[["pos_dev_along", "pos_dev_cross_h", "pos_dev_cross_v"]].to_numpy(dtype=float)

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
    abs_dev = AbsoluteTrajectoryDeviations(pos_dev=pos_dev, directed_pos_dev=directed_pos_dev, rot_dev=rot_dev)
    return ATEResult(trajectory=trajectory, abs_dev=abs_dev)


# ---------------------------------------------------------------------------
# RPE result
# ---------------------------------------------------------------------------


def write_rpe_result(result: RPEResult, filename: str, mode: str = "a") -> None:
    """Writes an RPE result as CSV with a metadata header."""
    with open(filename, mode, encoding="utf-8", newline="") as file:
        file.write(f"#relative_dist_unit {result.rpe_dev.pair_distance_unit.name}\n")
        file.write(f"#name {result.name}\n")

        writer = csv.writer(file)
        file.write("#num_pairs ")
        writer.writerow(result.num_pairs)
    result.to_dataframe().to_csv(filename, index=False, mode="a", float_format="%.12f")


def read_rpe_result(filename: str) -> RPEResult:
    """Reads an RPE result from a file."""
    header_data = HeaderData.from_file(filename)
    deviation_data = pd.read_csv(filename, comment="#")

    pos_dev: dict[float, list[float]] = {}
    rot_dev: dict[float, list[float]] = {}
    pair_distance: dict[float, list[float]] = {}

    last_index = 0
    for index in header_data.num_pairs:
        dev_block = deviation_data.iloc[last_index : last_index + index, :]
        pair_distances = dev_block["pair_distance"].to_numpy(dtype=float)
        pos_devs = dev_block["pos_dev"].to_numpy(dtype=float)
        rot_devs = dev_block["rot_dev"].to_numpy(dtype=float) if "rot_dev" in deviation_data.columns else []

        mean_dist = float(np.mean(pair_distances))
        pair_distance[mean_dist] = list(pair_distances)
        pos_dev[mean_dist] = list(pos_devs)
        rot_dev[mean_dist] = list(rot_devs)

        last_index += index

    rpe_dev = RelativeTrajectoryDeviations(
        pair_distance=pair_distance,
        pos_dev=pos_dev,
        rot_dev=rot_dev,
        pair_distance_unit=header_data.relative_dist_unit,
    )

    return RPEResult(rpe_dev=rpe_dev, name=header_data.name)
