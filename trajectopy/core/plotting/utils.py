"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import logging
from datetime import datetime
from enum import Enum, auto
from typing import List, Tuple

import numpy as np

from trajectopy.sorting import Sorting
from trajectopy.trajectory import Trajectory

logger = logging.getLogger("root")


class TrajectoriesSorting(Enum):
    ALL_TIME = auto()
    ALL_SPATIAL = auto()
    MIXED = auto()


def is_probably_unix(trajectories: List[Trajectory]) -> bool:
    """
    Checks if the supplied trajectories have (likely) unix timestamps as seconds
    and converts them to datetime objects.
    """
    min_time = min(traj.tstamps[0] for traj in trajectories)

    if min_time < 1_000_000:
        return False

    today = datetime.now()
    for traj in trajectories:
        try:
            converted_date = datetime.fromtimestamp(traj.tstamps[0])
        except Exception:
            return False

        if converted_date > today:
            return False

    return True


def get_sorting(sort_by_list: List[str]) -> TrajectoriesSorting:
    if all(sorting == Sorting.ARC_LENGTH for sorting in sort_by_list):
        return TrajectoriesSorting.ALL_SPATIAL

    if all(sorting == Sorting.TIME for sorting in sort_by_list):
        return TrajectoriesSorting.ALL_TIME

    logger.warning("Data is diffently sorted, weird things might happen.")
    return TrajectoriesSorting.MIXED


def is_all_unix(trajectories: List[Trajectory]) -> bool:
    return all(traj.is_unix_time for traj in trajectories)


def derive_xlabel_from_sortings(trajectories_sorting: TrajectoriesSorting, all_unix: bool) -> str:
    if trajectories_sorting == TrajectoriesSorting.ALL_SPATIAL:
        return "trajectory length [m]"

    if trajectories_sorting == TrajectoriesSorting.ALL_TIME:
        return "time [hh:mm:ss]" if all_unix else "time [s]"

    logger.warning("Data is diffently sorted, weird things might happen.")
    return "time [s] / trajectory length [m]"


def get_axis_label(trajectories: List) -> Tuple[str, str, str]:
    """Returns the unit of the axis"""
    if all(traj.pos.epsg == 0 for traj in trajectories):
        return "x [m]", "y [m]", "z [m]"

    unit_set = {traj.pos.crs.axis_info[0].unit_name if traj.pos.crs else "unknown" for traj in trajectories}
    unit_name = unit_set.pop().replace("metre", "m").replace("degree", "°")

    # there are multiple units
    if unit_set:
        return "x", "y", "z"

    axis_info = trajectories[0].pos.crs.axis_info
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


def set_aspect_equal_3d(ax):
    """
    https://stackoverflow.com/a/35126679
    """
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)

    plot_radius = max(
        abs(lim - mean_) for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean)) for lim in lims
    )

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def get_axis_label(trajectories: List[Trajectory]) -> Tuple[str, str, str]:
    """Returns the unit of the axis"""
    if all(traj.pos.epsg == 0 for traj in trajectories):
        return "x [m]", "y [m]", "z [m]"

    unit_set = {traj.pos.crs.axis_info[0].unit_name if traj.pos.crs else "unknown" for traj in trajectories}
    unit_name = unit_set.pop().replace("metre", "m").replace("degree", "°")

    # there are multiple units
    if unit_set:
        return "x", "y", "z"

    axis_info = trajectories[0].pos.crs.axis_info
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
