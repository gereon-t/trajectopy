"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import List, Tuple

from trajectopy.core.sorting import Sorting

logger = logging.getLogger("root")


def derive_xlabel_from_sortings(sort_by_list: List[str]) -> str:
    if all(sorting == Sorting.ARC_LENGTH for sorting in sort_by_list):
        return "trajectory length [m]"

    if all(sorting == Sorting.TIME for sorting in sort_by_list):
        return "time [s]"

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
