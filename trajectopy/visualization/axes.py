"""Axis labelling helpers."""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from trajectopy.utils.definitions import Sorting, TrajectoriesSorting

if TYPE_CHECKING:
    from trajectopy.core.trajectory import Trajectory

logger = logging.getLogger(__name__)


def get_sorting(sort_by_list: Iterable[Sorting]) -> TrajectoriesSorting:
    sort_by_list = list(sort_by_list)  # may be a generator, which must not be consumed twice
    if all(sorting == Sorting.PATH_LENGTH for sorting in sort_by_list):
        return TrajectoriesSorting.ALL_SPATIAL

    if all(sorting == Sorting.TIME for sorting in sort_by_list):
        return TrajectoriesSorting.ALL_TIME

    logger.warning("Data is differently sorted, weird things might happen.")
    return TrajectoriesSorting.MIXED


@dataclass(frozen=True)
class IndexAxis:
    """Describes the x-axis used for values over time or path length.

    Attributes:
        label: Axis label.
        is_datetime: Whether timestamps are shown as UTC date/time.
        offset: Value subtracted from the timestamps (non-zero for relative time axes).
        sorting: Common sorting of the trajectories.
    """

    label: str
    is_datetime: bool
    offset: float
    sorting: TrajectoriesSorting

    def values(self, trajectory: "Trajectory") -> np.ndarray:
        """Returns the x values of ``trajectory`` on this axis."""
        if self.sorting == TrajectoriesSorting.ALL_TIME:
            return trajectory.timestamps - self.offset
        return trajectory.index


def index_axis(trajectories: Sequence["Trajectory"], use_relative_timestamps: bool = False) -> IndexAxis:
    """Determines a common x-axis for values of the given trajectories over time or path length."""
    sorting = get_sorting(trajectory.sorting for trajectory in trajectories)

    if sorting == TrajectoriesSorting.ALL_SPATIAL:
        return IndexAxis("trajectory length [m]", is_datetime=False, offset=0.0, sorting=sorting)

    if sorting == TrajectoriesSorting.MIXED:
        return IndexAxis("time [s] / trajectory length [m]", is_datetime=False, offset=0.0, sorting=sorting)

    starts = [trajectory.timestamps[0] for trajectory in trajectories if len(trajectory.timestamps) > 0]
    if use_relative_timestamps:
        return IndexAxis("relative time [s]", is_datetime=False, offset=min(starts, default=0.0), sorting=sorting)

    if trajectories and all(trajectory.is_unix_time for trajectory in trajectories):
        return IndexAxis("time (UTC)", is_datetime=True, offset=0.0, sorting=sorting)

    return IndexAxis("time [s]", is_datetime=False, offset=0.0, sorting=sorting)


def get_axis_label(trajectories: Sequence["Trajectory"]) -> tuple[str, str, str]:
    """Returns the unit of the axis"""
    if not trajectories:
        return "x [m]", "y [m]", "z [m]"

    if all(traj.positions.epsg == 0 for traj in trajectories):
        return "x [m]", "y [m]", "z [m]"

    unit_set = {
        traj.positions.crs.axis_info[0].unit_name if traj.positions.crs else "unknown" for traj in trajectories
    }

    if not unit_set:
        return "x [m]", "y [m]", "z [m]"

    unit_name = unit_set.pop().replace("metre", "m").replace("degree", "°")

    # there are multiple units
    if unit_set:
        return "x", "y", "z"

    if (crs := trajectories[0].positions.crs) is None:
        return "x [m]", "y [m]", "z [m]"
    axis_info = crs.axis_info
    x_axis_name = axis_info[0].name
    y_axis_name = axis_info[1].name

    if len(axis_info) > 2:
        z_axis_name = axis_info[2].name
        z_unit_name = axis_info[2].unit_name.replace("metre", "m").replace("degree", "°")
    else:
        # we assume meters as default
        z_axis_name = "z"
        z_unit_name = "m"

    return (
        f"{x_axis_name} [{unit_name}]",
        f"{y_axis_name} [{unit_name}]",
        f"{z_axis_name} [{z_unit_name}]",
    )
