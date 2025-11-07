"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from trajectopy.core.alignment.parameters import AlignmentParameters
from trajectopy.core.evaluation.ate_result import ATEResult
from trajectopy.core.evaluation.rpe_result import RPEResult
from trajectopy.core.plotting.mpl.alignment import _annotate_heatmap, _heatmap
from trajectopy.core.plotting.mpl.results import (
    add_rpy_legend,
    colored_scatter_plot,
    plot_position_ate_edf,
    plot_position_ate_hist,
    plot_rotation_ate_edf,
    plot_rotation_ate_hist,
    plot_rpe_pos,
    plot_rpe_rot,
)
from trajectopy.core.plotting.mpl.trajectory import plot_pos, plot_rpy, plot_xyz
from trajectopy.core.plotting.utils import (
    TrajectoriesSorting,
    derive_xlabel_from_sortings,
    get_sorting,
    is_all_unix,
)
from trajectopy.definitions import DATE_FORMATTER
from trajectopy.settings import MPLPlotSettings, PairDistanceUnit
from trajectopy.sorting import Sorting
from trajectopy.trajectory import Trajectory

logger = logging.getLogger("root")


def plot_trajectories(
    trajectories: List[Trajectory], scatter_3d: bool = False
) -> Tuple[Figure, Figure, Union[Figure, None]]:
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
    fig_pos = plot_pos(trajectories=trajectories, scatter_3d=scatter_3d)
    fig_xyz = plot_xyz(trajectories=trajectories)
    fig_rpy = plot_rpy(trajectories=trajectories)
    return fig_pos, fig_xyz, fig_rpy


def plot_correlation_heatmap(estimated_parameters: AlignmentParameters, enabled_only: bool = True) -> Figure:
    """Plots the correlation heatmap of the alignment parameters using matplotlib.

    Args:
        estimated_parameters (AlignmentParameters): Estimated parameters.
        enabled_only (bool, optional): Whether to consider only enabled parameters. Defaults to True.

    Returns:
        plt.Figure: Correlation heatmap figure.
    """
    covariance_matrix = estimated_parameters.get_covariance_matrix(enabled_only=enabled_only)
    std_devs = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
    np.fill_diagonal(correlation_matrix, np.nan)
    fig, ax = plt.subplots()
    ax.grid(False)
    im, _ = _heatmap(
        correlation_matrix,
        estimated_parameters.params_labels(enabled_only=enabled_only, lower_case=True),
        estimated_parameters.params_labels(enabled_only=enabled_only, lower_case=True),
        ax=ax,
        cmap="coolwarm",
        cbarlabel="Correlation",
        cbar_kw={"format": "%.2f"},
    )
    _annotate_heatmap(im, valfmt="{x:.2f}")
    ax.set_aspect("auto")

    plt.tight_layout()
    return fig


def plot_covariance_heatmap(estimated_parameters: AlignmentParameters, enabled_only: bool = True) -> Figure:
    """Plots the covariance heatmap of the alignment parameters using matplotlib.

    Args:
        estimated_parameters (AlignmentParameters): Estimated parameters.
        enabled_only (bool, optional): Whether to consider only enabled parameters. Defaults to True.

    Returns:
        plt.Figure: Covariance heatmap figure.
    """
    covariance_matrix = estimated_parameters.get_covariance_matrix(enabled_only=enabled_only)
    fig, ax = plt.subplots()
    ax.grid(False)
    im, _ = _heatmap(
        covariance_matrix,
        estimated_parameters.params_labels(enabled_only=enabled_only, lower_case=True),
        estimated_parameters.params_labels(enabled_only=enabled_only, lower_case=True),
        ax=ax,
        cmap="coolwarm",
        cbarlabel="Covariance",
        cbar_kw={"format": "%.2f"},
    )
    _annotate_heatmap(im, valfmt="{x:.3f}")
    ax.set_aspect("auto")
    plt.tight_layout()
    return fig


def plot_ate_3d(ate_results: List[ATEResult], plot_settings: MPLPlotSettings = MPLPlotSettings()) -> Figure:
    """
    Plots the ATE results in 2D using matplotlib.

    Args:
        ate_results (List[ATEResult]): List of ATE results.
        plot_settings (MPLPlotSettings, optional): Plot settings. Defaults to MPLPlotSettings().

    Returns:
        Figure: Figure containing the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for ate_result in ate_results:
        if len(ate_result.function_of) == 0:
            logger.warning("Skipping %s as it has no data", ate_result.name)
            continue
        ax.plot(
            (
                ate_result.pos_dev_cross_h * plot_settings.unit_multiplier
                if plot_settings.directed_ate
                else ate_result.pos_dev_x * plot_settings.unit_multiplier
            ),
            (
                ate_result.pos_dev_along * plot_settings.unit_multiplier
                if plot_settings.directed_ate
                else ate_result.pos_dev_y * plot_settings.unit_multiplier
            ),
            (
                ate_result.pos_dev_cross_v * plot_settings.unit_multiplier
                if plot_settings.directed_ate
                else ate_result.pos_dev_z * plot_settings.unit_multiplier
            ),
            ".",
            label=ate_result.name,
        )

    ax.set_xlabel(f"{'Horizontal Cross-Track' if plot_settings.directed_ate else 'X'} {plot_settings.unit_str}")
    ax.set_ylabel(f"{'Along-Track' if plot_settings.directed_ate else 'Y'} {plot_settings.unit_str}")
    ax.set_zlabel(f"{'Vertical Cross-Track' if plot_settings.directed_ate else 'Z'} {plot_settings.unit_str}")

    ax.legend()
    plt.tight_layout()
    return fig


def plot_ate_bars(
    ate_results: List[ATEResult],
    plot_settings: MPLPlotSettings = MPLPlotSettings(),
    mode: str = "positions",
) -> Figure:
    """
    Plots multiple ATE results as bars for different characteristics (min, max, mean, median, rms, std)
    using matplotlib.

    Args:
        ate_result (List[ATEResult]): List of ATE results.
        plot_settings (MPLPlotSettings, optional): Plot settings. Defaults to MPLPlotSettings().
        mode (str, optional): Mode to plot. Either 'positions' or 'rotations'. Defaults to 'positions'.

    Returns:
        Figure: Bar plot figure.

    """
    fig, ax = plt.subplots()

    bar_width = 0.9 / len(ate_results)
    characteristics = ["Min", "Max", "Mean", "Median", "RMS", "STD"]
    unit = plot_settings.unit_str if mode == "positions" else "[°]"
    spacings = np.linspace(
        -bar_width * (len(ate_results) - 1) / 2,
        bar_width * (len(ate_results) - 1) / 2,
        len(ate_results),
    )
    x_positions = np.arange(len(characteristics))
    for deviation, spacing in zip(ate_results, spacings):
        if mode == "rotations" and deviation.abs_dev.rot_dev is None:
            continue

        if mode == "positions":
            data = [
                deviation.pos_dev_min * plot_settings.unit_multiplier,
                deviation.pos_dev_max * plot_settings.unit_multiplier,
                deviation.pos_ate * plot_settings.unit_multiplier,
                deviation.pos_dev_median * plot_settings.unit_multiplier,
                deviation.pos_dev_rms * plot_settings.unit_multiplier,
                deviation.pos_dev_std * plot_settings.unit_multiplier,
            ]
        elif mode == "rotations":
            data = [
                np.rad2deg(deviation.rot_dev_min),
                np.rad2deg(deviation.rot_dev_max),
                np.rad2deg(deviation.rot_ate),
                np.rad2deg(deviation.rot_dev_median),
                np.rad2deg(deviation.rot_dev_rms),
                np.rad2deg(deviation.rot_dev_std),
            ]
        else:
            raise ValueError("Mode must be either 'positions' or 'rotations'")
        ax.bar(x_positions + spacing, data, width=bar_width, label=deviation.name)

    ax.set_xlabel("Characteristic")
    ax.set_ylabel(f"Value {unit}")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(characteristics)
    ax.legend()

    return fig


def plot_compact_ate_hist(ate_result: ATEResult, plot_settings: MPLPlotSettings = MPLPlotSettings()) -> Figure:
    """
    Plots compact ATE histograms for the given ATEResult.
    The plot contains histograms for the position deviations and, if available, the rotation deviations.

    Args:
        ate_result (ATEResult): ATE result to plot.
        plot_settings (MPLPlotSettings, optional): Plot settings. Defaults to MPLPlotSettings().

    Returns:
        Figure: Figure containing the plot.
    """
    fig = plt.figure()
    pos_ax = plt.subplot(2, 1, 1)
    plot_position_ate_hist(ate_result, plot_settings)
    pos_ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if ate_result.abs_dev.rot_dev is not None:
        rot_ax = plt.subplot(2, 1, 2)
        plot_rotation_ate_hist(ate_result, plot_settings)
        rot_ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    return fig


def plot_ate(
    ate_results: Union[ATEResult, List[ATEResult]],
    plot_settings: MPLPlotSettings = MPLPlotSettings(),
) -> Figure:
    """
    Plots ATE for the given ATEResult(s) as a line plot using matplotlib.
    If available, the plot contains the position and rotation deviations.
    The x-axis depends on the sorting of the trajectory.

    Args:
        ate_results (Union[ATEResult, List[ATEResult]]): ATE result(s) to plot.
        plot_settings (MPLPlotSettings, optional): Plot settings. Defaults to MPLPlotSettings().

    Returns:
        Figure: Figure containing the plot.
    """
    deviation_list = ate_results if isinstance(ate_results, list) else [ate_results]
    trajectories_list = [dev.trajectory for dev in deviation_list]
    trajectories_sorting = get_sorting(traj.sorting for traj in trajectories_list)
    all_unix = is_all_unix(trajectories_list)
    x_label = derive_xlabel_from_sortings(trajectories_sorting, all_unix)

    fig = plt.figure()

    ax_pos = plt.subplot(2, 1, 1)
    ax_pos.set_xlabel(x_label)
    ax_pos.set_ylabel(f"Deviation {plot_settings.unit_str}")
    if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME:
        ax_pos.xaxis.set_major_formatter(DATE_FORMATTER)

    if any(dev.abs_dev.rot_dev for dev in deviation_list):
        ax_rot = plt.subplot(2, 1, 2)
        ax_rot.set_xlabel(x_label)
        ax_rot.set_ylabel("Deviation [°]")
        if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME:
            ax_rot.xaxis.set_major_formatter(DATE_FORMATTER)
    else:
        ax_rot = None

    for dev in deviation_list:
        if len(dev.function_of) == 0:
            logger.warning("Skipping %s as it has no data", dev.name)
            continue

        function_of = (
            dev.trajectory.datetimes
            if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME
            else dev.function_of
        )

        ax_pos.plot(function_of, dev.pos_dev_comb * plot_settings.unit_multiplier)
        if ax_rot is not None:
            ax_rot.plot(function_of, np.rad2deg(dev.rot_dev_comb))

    fig.legend([dev.name for dev in deviation_list], ncol=3, loc="upper center")
    plt.tight_layout()
    return fig


def plot_ate_dof(
    ate_result: ATEResult,
    plot_settings: MPLPlotSettings = MPLPlotSettings(),
) -> Figure:
    """Plots ATE DOF (Degrees of Freedom) for the given ATEResult(s) as a line plot using matplotlib.
    The DOF plot shows the deviations in the x, y, and z directions for position and rotation.

    Args:
        ate_results (Union[ATEResult, List[ATEResult]]): ATE result(s) to plot.
        plot_settings (MPLPlotSettings, optional): Plot settings. Defaults to MPLPlotSettings().

    Returns:
        Figure: Figure containing the plot.
    """
    trajectory = ate_result.trajectory
    x_label = derive_xlabel_from_sortings(
        TrajectoriesSorting.ALL_SPATIAL if trajectory.sorting == Sorting.ARC_LENGTH else TrajectoriesSorting.ALL_TIME,
        trajectory.is_unix_time,
    )

    is_unix_time = trajectory.is_unix_time and trajectory.sorting == Sorting.TIME

    fig = plt.figure()

    ax_pos = plt.subplot(2, 1, 1)
    ax_pos.set_xlabel(x_label)
    ax_pos.set_ylabel(f"Deviation {plot_settings.unit_str}")
    if is_unix_time:
        ax_pos.xaxis.set_major_formatter(DATE_FORMATTER)

    if ate_result.has_orientation:
        ax_rot = plt.subplot(2, 1, 2)
        ax_rot.set_xlabel(x_label)
        ax_rot.set_ylabel("Deviation [°]")
        if is_unix_time:
            ax_rot.xaxis.set_major_formatter(DATE_FORMATTER)
    else:
        ax_rot = None

    if len(ate_result.function_of) == 0:
        logger.warning("Skipping %s as it has no data", ate_result.name)
        return fig

    pos_dev_x = ate_result.pos_dev_along if plot_settings.directed_ate else ate_result.pos_dev_x
    pos_dev_y = ate_result.pos_dev_cross_h if plot_settings.directed_ate else ate_result.pos_dev_y
    pos_dev_z = ate_result.pos_dev_cross_v if plot_settings.directed_ate else ate_result.pos_dev_z

    function_of = ate_result.trajectory.datetimes if is_unix_time else ate_result.function_of

    ax_pos.plot(
        function_of,
        pos_dev_x * plot_settings.unit_multiplier,
        label="Along-Track" if plot_settings.directed_ate else "X",
    )
    ax_pos.plot(
        function_of,
        pos_dev_y * plot_settings.unit_multiplier,
        label="Horizontal Cross-Track" if plot_settings.directed_ate else "Y",
    )
    ax_pos.plot(
        function_of,
        pos_dev_z * plot_settings.unit_multiplier,
        label="Vertical Cross-Track" if plot_settings.directed_ate else "Z",
    )
    ax_pos.legend()

    if ax_rot is not None:
        ax_rot.plot(function_of, np.rad2deg(ate_result.rot_dev_x), label="Roll")
        ax_rot.plot(function_of, np.rad2deg(ate_result.rot_dev_y), label="Pitch")
        ax_rot.plot(function_of, np.rad2deg(ate_result.rot_dev_z), label="Yaw")
        ax_rot.legend()

    ax_pos.set_title(f"{ate_result.name}")
    plt.tight_layout()
    return fig


def plot_ate_edf(
    ate_results: Union[ATEResult, List[ATEResult]],
    plot_settings: MPLPlotSettings = MPLPlotSettings(),
) -> Figure:
    """
    Plots ATE EDF for the given ATEResult(s) as a line plot using matplotlib.
    The EDF (Empirical Distribution Function) shows the cummulative probability of the deviations.
    Using this plot, one can easily see how many percent of the deviations are below a certain value.

    Args:
        ate_results (Union[ATEResult, List[ATEResult]]): ATE result to plot.
        plot_settings (MPLPlotSettings, optional): Plot settings. Defaults to MPLPlotSettings().

    Returns:
        Figure: Figure containing the plot.
    """
    deviation_list = ate_results if isinstance(ate_results, list) else [ate_results]

    fig = plt.figure()

    plot_position_ate_edf(deviation_list, plot_settings)
    plot_rotation_ate_edf(deviation_list)

    fig.legend([dev.name for dev in deviation_list], ncol=3, loc="upper center")
    plt.tight_layout()
    return fig


def plot_rpe(rpe_results: List[RPEResult]) -> Tuple[Figure, Figure]:
    """Plots the RPE results as a line plot with violin plots for the position and rotation deviations.

    Depending on the pair distance unit, the unit of the position deviations
    is either in meters/meters (%) or meters/seconds. The unit of the rotation
    deviations is respectively in degrees/m or degrees/second.

    Args:
        rpe_results (list[RelativeTrajectoryDeviations]): list of RelativeTrajectoryDeviations

    Returns:
        Tuple[Figure, Figure]: metric and time RPE plots

    """
    if not isinstance(rpe_results, list):
        rpe_results = [rpe_results]

    fig_metric, (fig_pos_metric, fig_rot_metric) = plt.subplots(2, 1)
    fig_time, (fig_pos_time, fig_rot_time) = plt.subplots(2, 1)

    fig_pos_metric.set_ylabel("Position RPE [%]")
    fig_pos_time.set_ylabel("Position RPE [m/s]")

    fig_rot_metric.set_ylabel("Rotation RPE [deg / 100m]")
    fig_rot_time.set_ylabel("Rotation RPE [deg/s]")

    fig_pos_metric.set_xlabel("pair distance [m]")
    fig_pos_time.set_xlabel("pair distance [s]")
    fig_rot_metric.set_xlabel("pair distance [m]")
    fig_rot_time.set_xlabel("pair distance [s]")

    figure_dict: Dict[str, Dict[PairDistanceUnit, Axes]] = {
        "pos": {
            PairDistanceUnit.METER: fig_pos_metric,
            PairDistanceUnit.SECOND: fig_pos_time,
        },
        "rot": {
            PairDistanceUnit.METER: fig_rot_metric,
            PairDistanceUnit.SECOND: fig_rot_time,
        },
    }

    plot_rpe_pos(figure_dict["pos"], rpe_results)
    plot_rpe_rot(figure_dict["rot"], rpe_results)

    add_rpy_legend(figure_dict)

    ret_sum = 1 if any(dev.rpe_dev.pair_distance_unit == PairDistanceUnit.METER for dev in rpe_results) else 0
    if any(dev.rpe_dev.pair_distance_unit == PairDistanceUnit.SECOND for dev in rpe_results):
        ret_sum += 2

    plt.close({1: fig_time, 2: fig_metric}.get(ret_sum))
    plt.tight_layout()
    return {
        0: (None, None),
        1: (fig_metric, None),
        2: (None, fig_time),
        3: (fig_metric, fig_time),
    }[ret_sum]


def scatter_ate(ate_result: ATEResult, plot_settings: MPLPlotSettings = MPLPlotSettings()) -> Tuple[Figure, Figure]:
    """
    Plots the ATE results as a scatter plot with color-coded deviations.

    Args:
        ate_result (ATEResult): ATE result to plot.
        plot_settings (MPLPlotSettings, optional): Plot settings. Defaults to MPLPlotSettings().
    """
    pos_fig = plt.figure()
    colored_scatter_plot(
        xyz=ate_result.trajectory.xyz,
        c_list=ate_result.pos_dev_comb * plot_settings.unit_multiplier,
        c_label=f"Deviation {plot_settings.unit_str}",
        plot_settings=plot_settings,
    )

    if not ate_result.has_orientation:
        return pos_fig, None

    rot_fig = plt.figure()
    colored_scatter_plot(
        xyz=ate_result.trajectory.xyz,
        c_list=ate_result.rot_dev_comb * 180 / np.pi,
        c_label="Deviation [°]",
        plot_settings=plot_settings,
    )
    return pos_fig, rot_fig
