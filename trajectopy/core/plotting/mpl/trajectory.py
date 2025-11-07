"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

from typing import List, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from trajectopy.core.plotting.utils import (
    TrajectoriesSorting,
    derive_xlabel_from_sortings,
    get_axis_label,
    get_sorting,
    is_all_unix,
    set_aspect_equal_3d,
)
from trajectopy.definitions import DATE_FORMATTER
from trajectopy.trajectory import Trajectory


def plot_pos(trajectories: List[Trajectory], scatter_3d: bool = False) -> Figure:
    """Plots xy(z) coordinates of trajectories as 2d or 3d plot"""
    x_label, y_label, z_label = get_axis_label(trajectories=trajectories)

    if scatter_3d:
        fig_pos = plt.figure()
        ax_pos = fig_pos.add_subplot(111, projection="3d")
        ax_pos.set_zlabel(z_label)  # type: ignore
    else:
        fig_pos, ax_pos = plt.subplots()
        ax_pos.axis("equal")

    ax_pos.set_xlabel(x_label)
    ax_pos.set_ylabel(y_label)

    legend_names = []
    for traj in trajectories:
        xyz = traj.xyz
        legend_names.append(traj.name)

        # pos fig
        if scatter_3d:
            ax_pos.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        else:
            ax_pos.plot(xyz[:, 0], xyz[:, 1])

    if scatter_3d:
        set_aspect_equal_3d(ax_pos)

    fig_pos.legend(legend_names, ncol=4, loc="upper center")
    return fig_pos


def plot_xyz(trajectories: List[Trajectory]) -> Figure:
    """Plots xyz coordinates of trajectories as subplots"""
    fig_xyz, axs_xyz = plt.subplots(3, 1, sharex=True)

    for ax, label in zip(axs_xyz, get_axis_label(trajectories=trajectories)):
        ax.set_ylabel(label)

    trajectories_sorting = get_sorting([traj.sorting for traj in trajectories])
    all_unix = is_all_unix(trajectories)
    x_label = derive_xlabel_from_sortings(trajectories_sorting, all_unix)

    axs_xyz[-1].set_xlabel(x_label)

    if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME:
        axs_xyz[-1].xaxis.set_major_formatter(DATE_FORMATTER)

    legend_names = []
    for traj in trajectories:
        legend_names.append(traj.name)
        xyz = traj.xyz

        # xyz fig
        for j, ax in enumerate(axs_xyz):
            ax.plot(
                (
                    traj.datetimes
                    if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME
                    else traj.function_of
                ),
                xyz[:, j],
            )

    fig_xyz.legend(legend_names, ncol=4, loc="upper center")
    return fig_xyz


def plot_rpy(trajectories: List[Trajectory]) -> Union[Figure, None]:
    """Plots rpy coordinates of trajectories as subplots"""
    fig_rpy, axs_rpy = plt.subplots(3, 1, sharex=True)
    trajectories_sorting = get_sorting([traj.sorting for traj in trajectories])
    all_unix = is_all_unix(trajectories)
    x_label = derive_xlabel_from_sortings(trajectories_sorting, all_unix)

    axs_rpy[-1].set_xlabel(x_label)

    if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME:
        axs_rpy[-1].xaxis.set_major_formatter(DATE_FORMATTER)

    not_empty = False
    legend_names = []
    for traj in trajectories:
        # rpy fig
        if traj.rot and len(traj.rot) > 0:
            legend_names.append(traj.name)
            rpy = traj.rpy
            ylabels = ["roll [°]", "pitch [°]", "yaw [°]"]
            for j, (ax, yl) in enumerate(zip(axs_rpy, ylabels)):
                ax.plot(
                    (
                        traj.datetimes
                        if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME
                        else traj.function_of
                    ),
                    np.rad2deg(rpy[:, j]),
                )
                ax.set_ylabel(yl)
            not_empty = True

    fig_rpy.legend(legend_names, ncol=4, loc="upper center")

    return fig_rpy if not_empty else None
