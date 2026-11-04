"""Binary encoding of the project contents (project format version 2).

A project is stored as one uncompressed NumPy archive (``.npz``): a JSON manifest with the metadata of all
entries (names, ids, settings, coordinate systems, order) and the numerical data of the trajectories and results
as plain arrays, which are written and read without any conversion. Reading never unpickles objects.

The arrays are restored exactly as they were saved, in the same order, so a saved project shows the same data
when it is opened again.
"""

from typing import Any

import numpy as np
from pyproj import Transformer

from trajectopy.core.parameters import ParameterSet
from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import AlignmentEstimationSettings, PairDistanceUnit, PlotSettings, ProcessingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.gui.models.entries import (
    AbsoluteDeviationEntry,
    AlignmentEntry,
    RelativeDeviationEntry,
    ResultEntry,
    TrajectoryEntry,
    TrajectoryProcessingState,
)
from trajectopy.io.results_io import DEVIATIONS_CONVENTION, convert_ate_deviations
from trajectopy.results.alignment_result import AlignmentResult
from trajectopy.results.ate_result import AbsoluteTrajectoryDeviations, ATEResult
from trajectopy.results.rpe_result import RelativeTrajectoryDeviations, RPEResult
from trajectopy.utils.definitions import Sorting

Arrays = dict[str, np.ndarray]


class Encoder:
    """Collects the arrays of a project; the metadata refers to them by key."""

    def __init__(self) -> None:
        self.arrays: Arrays = {}

    def put(self, array: np.ndarray) -> str:
        key = f"a{len(self.arrays)}"
        self.arrays[key] = np.ascontiguousarray(array)
        return key


class Decoder:
    """Reads the arrays that the metadata refers to."""

    def __init__(self, arrays: Any) -> None:
        self._arrays = arrays

    def get(self, key: str) -> np.ndarray:
        return self._arrays[key]


# ------------------------------------------------------------------------------------------------ trajectories


def encode_trajectory(trajectory: Trajectory, encoder: Encoder) -> dict:
    return {
        "name": trajectory.name,
        "epsg": trajectory.positions.epsg,
        "local_transformer": trajectory.positions.pipeline_str,
        "sorting": trajectory.sorting.value,
        "timestamps": encoder.put(trajectory.timestamps),
        "path_lengths": encoder.put(trajectory.path_lengths),
        "xyz": encoder.put(trajectory.positions.xyz),
        "quat": None if trajectory.rotations is None else encoder.put(trajectory.rotations.as_quat()),
        "velocity": encoder.put(trajectory.velocity_xyz),
    }


def decode_trajectory(data: dict, decoder: Decoder) -> Trajectory:
    timestamps, path_lengths = decoder.get(data["timestamps"]), decoder.get(data["path_lengths"])
    xyz, velocity = decoder.get(data["xyz"]), decoder.get(data["velocity"])
    rotations = None if data["quat"] is None else Rotations.from_quat(decoder.get(data["quat"]))
    local_transformer = Transformer.from_pipeline(data["local_transformer"]) if data["local_transformer"] else None
    trajectory = Trajectory(
        positions=Positions(xyz, epsg=data["epsg"], local_transformer=local_transformer, init_local_transformer=False),
        rotations=rotations,
        timestamps=timestamps,
        name=data["name"],
        path_lengths=path_lengths,
        velocity_xyz=velocity,
        sorting=Sorting.from_str(data["sorting"]),
    )
    # exactly as saved: the constructor orders the poses itself, which could change the order of equal values
    trajectory.timestamps, trajectory.path_lengths, trajectory.positions.xyz = timestamps, path_lengths, xyz
    trajectory.rotations, trajectory.velocity_xyz = rotations, velocity
    return trajectory


def encode_trajectory_entry(entry: TrajectoryEntry, encoder: Encoder) -> dict:
    return {
        "id": entry.entry_id,
        "source": entry.full_filename,
        "reference": entry.set_as_reference,
        "group": entry.group_id,
        "group_name": entry.group_name,
        "state": str(entry.state),
        "settings": entry.settings.to_dict(),
        "trajectory": encode_trajectory(entry.trajectory, encoder),
    }


def decode_trajectory_entry(data: dict, decoder: Decoder) -> TrajectoryEntry:
    entry = TrajectoryEntry(
        full_filename=data["source"],
        trajectory=decode_trajectory(data["trajectory"], decoder),
        set_as_reference=data["reference"],
        settings=ProcessingSettings.from_dict(data["settings"]),
        group_id=data["group"],
        group_name=data.get("group_name", ""),  # since 6.1
        state=TrajectoryProcessingState.from_string(data["state"]),
    )
    entry.set_id(data["id"])
    return entry


# ------------------------------------------------------------------------------------------------ results


def encode_result_entry(entry: ResultEntry, encoder: Encoder) -> dict:
    if isinstance(entry, AbsoluteDeviationEntry):
        data = {"type": "ate", **_encode_ate(entry.deviations, encoder)}
    elif isinstance(entry, RelativeDeviationEntry):
        data = {"type": "rpe", **_encode_rpe(entry.deviations, encoder)}
    elif isinstance(entry, AlignmentEntry):
        data = {"type": "alignment", **_encode_alignment(entry.alignment_result, encoder)}
    else:
        raise TypeError(f"Unsupported result type {type(entry).__name__}")
    return {"id": entry.entry_id, "group": entry.group_id, "group_name": entry.group_name, **data}


def decode_result_entry(data: dict, decoder: Decoder) -> ResultEntry:
    entry: ResultEntry
    if data["type"] == "ate":
        entry = AbsoluteDeviationEntry(deviations=_decode_ate(data, decoder))
    elif data["type"] == "rpe":
        entry = RelativeDeviationEntry(deviations=_decode_rpe(data, decoder))
    elif data["type"] == "alignment":
        entry = AlignmentEntry(alignment_result=_decode_alignment(data, decoder))
    else:
        raise ValueError(f"Unsupported result type {data['type']!r}")
    entry.set_id(data["id"])
    if "group" in data:  # since 6.1
        entry.group_id, entry.group_name = data["group"], data["group_name"]
    return entry


def _encode_ate(result: ATEResult, encoder: Encoder) -> dict:
    deviations = result.abs_dev
    return {
        "name": result.name,
        "trajectory": encode_trajectory(result.trajectory, encoder),
        "pos_dev": encoder.put(deviations.pos_dev),
        "directed_pos_dev": encoder.put(deviations.directed_pos_dev),
        "rot_dev": None if deviations.rot_dev is None else encoder.put(deviations.rot_dev.as_quat()),
        "rotations_used": deviations.rotations_used,
        "deviations": DEVIATIONS_CONVENTION,
    }


def _decode_ate(data: dict, decoder: Decoder) -> ATEResult:
    pos_dev, directed_pos_dev, rot_dev = convert_ate_deviations(
        data.get("deviations", ""),
        decoder.get(data["pos_dev"]),
        decoder.get(data["directed_pos_dev"]),
        None if data["rot_dev"] is None else Rotations.from_quat(decoder.get(data["rot_dev"])),
    )
    deviations = AbsoluteTrajectoryDeviations(
        pos_dev=pos_dev, directed_pos_dev=directed_pos_dev, rot_dev=rot_dev, rotations_used=data["rotations_used"]
    )
    return ATEResult(trajectory=decode_trajectory(data["trajectory"], decoder), abs_dev=deviations, name=data["name"])


def _encode_rpe(result: RPEResult, encoder: Encoder) -> dict:
    deviations = result.rpe_dev
    return {
        "name": result.name,
        "unit": deviations.pair_distance_unit.name,
        "buckets": [
            {
                "distance": float(distance),
                "pair_distance": encoder.put(np.asarray(deviations.pair_distance[distance], dtype=float)),
                "pos_dev": encoder.put(np.asarray(deviations.pos_dev.get(distance, []), dtype=float)),
                "rot_dev": encoder.put(np.asarray(deviations.rot_dev.get(distance, []), dtype=float)),
            }
            for distance in deviations.pair_distance
        ],
    }


def _decode_rpe(data: dict, decoder: Decoder) -> RPEResult:
    buckets = data["buckets"]
    deviations = RelativeTrajectoryDeviations(
        pair_distance={bucket["distance"]: decoder.get(bucket["pair_distance"]) for bucket in buckets},
        pos_dev={bucket["distance"]: decoder.get(bucket["pos_dev"]) for bucket in buckets},
        rot_dev={bucket["distance"]: decoder.get(bucket["rot_dev"]) for bucket in buckets},
        pair_distance_unit=PairDistanceUnit[data["unit"]],
    )
    return RPEResult(rpe_dev=deviations, name=data["name"])


def _encode_parameters(parameters: ParameterSet, encoder: Encoder) -> dict:
    # the stored values, not .value, which is the default of a disabled parameter
    return {
        "set_enabled": bool(parameters.enabled),
        "values": [float(parameter._value) for parameter in parameters],
        "variances": [float(parameter._variance) for parameter in parameters],
        "enabled": [bool(parameter.enabled) for parameter in parameters],
        "covariance": encoder.put(parameters.covariance_matrix),
    }


def _decode_parameters(parameters: ParameterSet, data: dict, decoder: Decoder) -> None:
    for parameter, value, variance, enabled in zip(parameters, data["values"], data["variances"], data["enabled"]):
        parameter.value, parameter.variance, parameter.enabled = value, variance, enabled
    parameters.enabled = data["set_enabled"]
    parameters.covariance_matrix = decoder.get(data["covariance"])


def _encode_alignment(result: AlignmentResult, encoder: Encoder) -> dict:
    if result.position_parameters is None:
        raise ValueError("No estimated parameters available!")
    return {
        "name": result.name,
        "converged": result.converged,
        "estimation_of": result.estimation_of.to_dict(),
        "position_parameters": _encode_parameters(result.position_parameters, encoder),
        "rotation_parameters": _encode_parameters(result.rotation_parameters, encoder),
    }


def _decode_alignment(data: dict, decoder: Decoder) -> AlignmentResult:
    result = AlignmentResult(
        name=data["name"],
        estimation_of=AlignmentEstimationSettings.from_dict(data["estimation_of"]),
        converged=data["converged"],
    )
    _decode_parameters(result.position_parameters, data["position_parameters"], decoder)
    _decode_parameters(result.rotation_parameters, data["rotation_parameters"], decoder)
    return result


# ------------------------------------------------------------------------------------------------ plot settings


def encode_plot_settings(settings: PlotSettings) -> dict:
    return settings.to_dict()


def decode_plot_settings(data: dict) -> PlotSettings:
    return PlotSettings.from_dict(data)
