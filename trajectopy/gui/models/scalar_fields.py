"""Per-pose values a trajectory can be colored by in the 3D view (like the scalar fields of a point cloud)."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from trajectopy.core.trajectory import Trajectory
from trajectopy.gui.models.entries import AbsoluteDeviationEntry, ResultEntry, TrajectoryEntry

logger = logging.getLogger(__name__)

SOLID = "solid"
ATE = "ate"  # the most recent ATE result of the trajectory
ATE_RESULT_PREFIX = "ate:"  # followed by the id of a specific ATE result


@dataclass(frozen=True)
class ScalarField:
    """``values`` has one entry per pose of the trajectory; NaN where there is no value.

    Fields with the same ``quantity`` share one color scale and colorbar in the 3D view.
    """

    key: str
    name: str
    quantity: str
    values: np.ndarray


_BUILT_IN: dict[str, tuple[str, str, Callable[[Trajectory], np.ndarray]]] = {
    "speed": ("Speed", "Speed [m/s]", lambda trajectory: trajectory.absolute_velocity),
    "height": ("Height", "Height (z) [m]", lambda trajectory: trajectory.positions.xyz[:, 2]),
    "time": ("Time", "Time [s]", lambda trajectory: trajectory.timestamps - np.min(trajectory.timestamps)),
    "path_length": ("Path length", "Path length [m]", lambda trajectory: trajectory.path_lengths),
}


def field_choices(entry: TrajectoryEntry, results: Sequence[ResultEntry]) -> list[tuple[str, str]]:
    """(key, label) of every field ``entry`` can be colored by."""
    choices = [(SOLID, "Solid color")]
    ate_results = _ate_results(entry, results)
    if ate_results:
        choices.append((ATE, "ATE"))
    if len(ate_results) > 1:
        choices.extend((ATE_RESULT_PREFIX + result.entry_id, f"ATE: {result.name}") for result in ate_results)
    choices.extend((key, label) for key, (label, _, _) in _BUILT_IN.items())
    return choices


def scalar_field(key: str, entry: TrajectoryEntry, results: Sequence[ResultEntry]) -> ScalarField | None:
    """The field ``key`` of ``entry``; None for a solid color or a field that is not available (anymore)."""
    trajectory = entry.trajectory
    if len(trajectory) == 0 or key == SOLID:
        return None

    if key in _BUILT_IN:
        name, quantity, compute = _BUILT_IN[key]
        try:
            values = np.asarray(compute(trajectory), dtype=float)
        except Exception:
            logger.exception("Could not compute the %s of '%s'", name.lower(), trajectory.name)
            return None
        return ScalarField(key=key, name=name, quantity=quantity, values=values)

    ate_results = _ate_results(entry, results)
    if key == ATE:
        result = ate_results[-1] if ate_results else None
    elif key.startswith(ATE_RESULT_PREFIX):
        result = next((r for r in ate_results if ATE_RESULT_PREFIX + r.entry_id == key), None)
    else:
        result = None
    if result is None:
        return None
    values = _match_by_time(
        trajectory.timestamps, result.deviations.trajectory.timestamps, result.deviations.pos_dev_comb
    )
    return ScalarField(key=key, name=f"ATE: {result.name}", quantity="ATE [m]", values=values)


def _ate_results(entry: TrajectoryEntry, results: Sequence[ResultEntry]) -> list[AbsoluteDeviationEntry]:
    return [r for r in results if isinstance(r, AbsoluteDeviationEntry) and r.source_id == entry.entry_id]


def _match_by_time(timestamps: np.ndarray, value_timestamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Assigns each pose the value with the nearest timestamp, if it is closer than half the sampling interval.

    The poses of a result are a subset of the trajectory's (matching, intersection), possibly shifted by an
    estimated time offset of a few milliseconds.
    """
    matched = np.full(len(timestamps), np.nan)
    if len(value_timestamps) == 0:
        return matched
    order = np.argsort(value_timestamps)
    sorted_times = value_timestamps[order]
    sorted_values = np.asarray(values, dtype=float)[order]

    right = np.clip(np.searchsorted(sorted_times, timestamps), 0, len(sorted_times) - 1)
    left = np.clip(right - 1, 0, len(sorted_times) - 1)
    nearest = np.where(
        np.abs(sorted_times[left] - timestamps) <= np.abs(sorted_times[right] - timestamps), left, right
    )
    distance = np.abs(sorted_times[nearest] - timestamps)

    time_steps = np.diff(np.sort(timestamps))
    time_steps = time_steps[time_steps > 0]
    tolerance = 0.5 * float(np.median(time_steps)) if len(time_steps) else np.inf
    within = distance <= tolerance
    matched[within] = sorted_values[nearest[within]]
    return matched
