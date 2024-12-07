"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from trajectopy.core.plotting.utils import derive_xlabel_from_sortings
from trajectopy.core.trajectory import Trajectory


def plot_trajectories(trajectories: List[Trajectory], dim: int = 2) -> Tuple[Figure, Figure, Union[Figure, None]]:
    """
    Plots the trajectories in 2d or 3d using matplotlib.

    Args:
        trajectories (List[Trajectory]): List of trajectories to plot.
        dim (int, optional): Dimension of the plot. Defaults to 2.

    This function creates one 2D or 3D plot for the xy(z) coordinates of the trajectories,
    one subplot for the xyz coordinates and one subplot for the rpy angles.

    Returns:
        Tuple[Figure, Figure, Union[Figure, None]]: Figures for the position, xyz and rpy plots.
    """
    fig_pos = plot_pos(trajectories=trajectories, dim=dim)
    fig_xyz = plot_xyz(trajectories=trajectories)
    fig_rpy = plot_rpy(trajectories=trajectories)
    return fig_pos, fig_xyz, fig_rpy


def plot_pos(trajectories: List[Trajectory], dim: int = 2) -> Figure:
    """Plots xy(z) coordinates of trajectories as 2d or 3d plot"""
    x_label, y_label, z_label = _get_axis_label(trajectories=trajectories)

    if dim == 2:
        fig_pos, ax_pos = plt.subplots()
        ax_pos.axis("equal")
    elif dim == 3:
        fig_pos = plt.figure()
        ax_pos = fig_pos.add_subplot(111, projection="3d")
        ax_pos.set_zlabel(z_label)  # type: ignore
    else:
        raise ValueError(f"Unknown dimension: {dim}")
    ax_pos.set_xlabel(x_label)
    ax_pos.set_ylabel(y_label)

    legend_names = []
    for traj in trajectories:
        legend_names.append(traj.name)

        # pos fig
        if dim == 2:
            ax_pos.plot(traj.pos.x, traj.pos.y)
        elif dim == 3:
            ax_pos.plot(traj.pos.x, traj.pos.y, traj.pos.z)

    if dim == 3:
        _set_aspect_equal_3d(ax_pos)

    fig_pos.legend(legend_names, ncol=4, loc="upper center")
    return fig_pos


def plot_xyz(trajectories: List[Trajectory]) -> Figure:
    """Plots xyz coordinates of trajectories as subplots"""
    fig_xyz, axs_xyz = plt.subplots(3, 1, sharex=True)

    for ax, label in zip(axs_xyz, _get_axis_label(trajectories=trajectories)):
        ax.set_ylabel(label)

    x_label = derive_xlabel_from_sortings([traj.sorting.value for traj in trajectories])
    axs_xyz[-1].set_xlabel(x_label)

    legend_names = []
    for traj in trajectories:
        legend_names.append(traj.name)
        xyz = traj.xyz

        # xyz fig
        for j, ax in enumerate(axs_xyz):
            ax.plot(traj.function_of, xyz[:, j])

    fig_xyz.legend(legend_names, ncol=4, loc="upper center")
    return fig_xyz


def _get_axis_label(trajectories: List[Trajectory]) -> Tuple[str, str, str]:
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


def plot_rpy(trajectories: List[Trajectory]) -> Union[Figure, None]:
    """Plots rpy coordinates of trajectories as subplots"""
    fig_rpy, axs_rpy = plt.subplots(3, 1, sharex=True)
    x_label = derive_xlabel_from_sortings([traj.sorting.value for traj in trajectories])
    axs_rpy[-1].set_xlabel(x_label)

    not_empty = False
    legend_names = []
    for traj in trajectories:
        # rpy fig
        if traj.rot and len(traj.rot) > 0:
            legend_names.append(traj.name)
            rpy = traj.rpy
            ylabels = ["roll [°]", "pitch [°]", "yaw [°]"]
            for j, (ax, yl) in enumerate(zip(axs_rpy, ylabels)):
                ax.plot(traj.function_of, np.rad2deg(rpy[:, j]))
                ax.set_ylabel(yl)
            not_empty = True

    fig_rpy.legend(legend_names, ncol=4, loc="upper center")

    return fig_rpy if not_empty else None


def _set_aspect_equal_3d(ax):
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
