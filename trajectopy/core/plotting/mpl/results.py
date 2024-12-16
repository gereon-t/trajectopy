"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from trajectopy.core.evaluation.ate_result import ATEResult
from trajectopy.core.evaluation.rpe_result import RPEResult
from trajectopy.core.plotting.utils import derive_xlabel_from_sortings
from trajectopy.core.settings.comparison import PairDistanceUnit
from trajectopy.core.settings.mpl_settings import MPLPlotSettings

logger = logging.getLogger("root")


def _norm_hist(*, l, mm: bool = False, alpha: float = 0.5, norm: bool = True) -> None:
    """
    Plots a histogram
    """
    l = l[np.abs(l) > 1e-6]
    hist, bin_edges = np.histogram(l, bins="auto")
    if norm:
        hist = hist / len(l)
    if mm:
        bin_edges *= 1000
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge", alpha=alpha)
    return max(hist)


def _stair_hist(*, l, mm: bool = False, linewidth: float = 1.5) -> None:
    """
    Plots a stair histogram
    """
    l = l[np.abs(l) > 1e-6]
    hist, bin_edges = np.histogram(l, bins="auto")
    n_hist = hist / len(l)
    if mm:
        bin_edges *= 1000
    plt.stairs(hist, bin_edges, linewidth=linewidth)
    return max(n_hist)


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


def plot_rotation_ate_hist(devs: ATEResult, plot_settings: MPLPlotSettings = MPLPlotSettings()) -> None:
    roll = np.rad2deg(devs.rot_dev_x)
    pitch = np.rad2deg(devs.rot_dev_y)
    yaw = np.rad2deg(devs.rot_dev_z)

    plt.xlabel("[°]")
    plt.ylabel("counts")
    if plot_settings.hist_as_stairs:
        _stair_hist(l=yaw)
        _stair_hist(l=pitch)
        _stair_hist(l=roll)
    else:
        _norm_hist(l=yaw, alpha=0.6, norm=False)
        _norm_hist(l=pitch, alpha=0.6, norm=False)
        _norm_hist(l=roll, alpha=0.4, norm=False)
    plt.legend(["yaw", "pitch", "roll"])


def plot_position_ate_hist(devs: ATEResult, plot_settings: MPLPlotSettings = MPLPlotSettings()):
    deviations_xa = (
        devs.abs_dev.directed_pos_dev[:, 0] if plot_settings.show_directed_devs else devs.abs_dev.pos_dev[:, 0]
    )
    deviations_yh = (
        devs.abs_dev.directed_pos_dev[:, 1] if plot_settings.show_directed_devs else devs.abs_dev.pos_dev[:, 1]
    )
    deviations_zv = (
        devs.abs_dev.directed_pos_dev[:, 2] if plot_settings.show_directed_devs else devs.abs_dev.pos_dev[:, 2]
    )
    labels = ["vertical", "horizontal", "along"] if plot_settings.show_directed_devs else ["x", "y", "z"]

    plt.xlabel(plot_settings.unit_str)
    plt.ylabel("counts")
    if plot_settings.hist_as_stairs:
        _stair_hist(l=deviations_zv, mm=plot_settings.ate_unit_is_mm)
        _stair_hist(l=deviations_yh, mm=plot_settings.ate_unit_is_mm)
        _stair_hist(l=deviations_xa, mm=plot_settings.ate_unit_is_mm)
    else:
        _norm_hist(l=deviations_zv, mm=plot_settings.ate_unit_is_mm, norm=False)
        _norm_hist(l=deviations_yh, mm=plot_settings.ate_unit_is_mm, norm=False)
        _norm_hist(l=deviations_xa, mm=plot_settings.ate_unit_is_mm, norm=False)

    plt.legend(labels)


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


def plot_position_ate_edf(
    deviation_list: List[ATEResult],
    plot_settings: MPLPlotSettings = MPLPlotSettings(),
) -> None:
    ax_pos = plt.subplot(2, 1, 1)
    ax_pos.set_xlabel(f"Deviation {plot_settings.unit_str}")
    ax_pos.set_ylabel("Cummulative Probability")

    for dev in deviation_list:
        sorted_comb_pos_dev = np.sort(dev.pos_dev_comb)
        pos_norm_cdf = np.arange(len(sorted_comb_pos_dev)) / float(len(sorted_comb_pos_dev))
        ax_pos.plot(sorted_comb_pos_dev * plot_settings.unit_multiplier, pos_norm_cdf)


def plot_rotation_ate_edf(deviation_list: List[ATEResult]) -> None:
    if all(dev.abs_dev.rot_dev is None for dev in deviation_list):
        return

    ax_rot = plt.subplot(2, 1, 2)

    ax_rot.set_xlabel("Deviation [°]")
    ax_rot.set_ylabel("Cummulative Probability")

    for dev in deviation_list:
        if dev.abs_dev.rot_dev is None:
            continue
        sorted_comb_rot_dev = np.sort(np.rad2deg(dev.rot_dev_comb))
        rot_norm_cdf = np.arange(len(sorted_comb_rot_dev)) / float(len(sorted_comb_rot_dev))
        ax_rot.plot(sorted_comb_rot_dev, rot_norm_cdf)


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
    x_label = derive_xlabel_from_sortings([dev.trajectory.sorting.value for dev in deviation_list])

    fig = plt.figure()

    ax_pos = plt.subplot(2, 1, 1)
    ax_pos.set_xlabel(x_label)
    ax_pos.set_ylabel(f"Deviation {plot_settings.unit_str}")

    if any(dev.abs_dev.rot_dev for dev in deviation_list):
        ax_rot = plt.subplot(2, 1, 2)
        ax_rot.set_xlabel(x_label)
        ax_rot.set_ylabel("Deviation [°]")
    else:
        ax_rot = None

    min_x = np.inf
    max_x = -np.inf
    for dev in deviation_list:
        if len(dev.trajectory.function_of) == 0:
            logger.warning("Skipping %s as it has no data", dev.name)
            continue

        arc_length_sorting = np.argsort(dev.trajectory.function_of)
        function_of_sorted = dev.trajectory.function_of[arc_length_sorting]

        if (min_val := function_of_sorted[0]) < min_x:
            min_x = min_val

        if (max_val := function_of_sorted[-1]) > max_x:
            max_x = max_val

        ax_pos.plot(
            dev.trajectory.function_of[arc_length_sorting],
            dev.pos_dev_comb[arc_length_sorting] * plot_settings.unit_multiplier,
        )
        if ax_rot is not None:
            ax_rot.plot(
                dev.trajectory.function_of[arc_length_sorting],
                np.rad2deg(dev.rot_dev_comb[arc_length_sorting]),
            )

    ax_pos.set_xlim(min_x, max_x)
    ax_rot.set_xlim(min_x, max_x)

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
        "pos": {PairDistanceUnit.METER: fig_pos_metric, PairDistanceUnit.SECOND: fig_pos_time},
        "rot": {PairDistanceUnit.METER: fig_rot_metric, PairDistanceUnit.SECOND: fig_rot_time},
    }

    _plot_rpe_pos(figure_dict["pos"], rpe_results)
    _plot_rpe_rot(figure_dict["rot"], rpe_results)

    _rpy_legend(figure_dict)

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


def _plot_rpe_pos(figure_dict: Dict[PairDistanceUnit, Axes], devs: List[RPEResult]) -> None:
    for dev in devs:
        line_plot = figure_dict[dev.rpe_dev.pair_distance_unit].plot(
            dev.mean_pair_distances, dev.pos_dev_mean, label=dev.name
        )

        if len(devs) > len({dev.rpe_dev.pair_distance_unit for dev in devs}):
            continue

        violin_plot = figure_dict[dev.rpe_dev.pair_distance_unit].violinplot(
            [
                [val * dev.drift_factor for val in pos_list]
                for pos_list in list(dev.rpe_dev.pos_dev.values())
                if pos_list
            ],
            positions=dev.mean_pair_distances,
            showmeans=True,
            widths=max(0.5, dev.step / 4),
        )
        _set_violin_color(violin_plot, line_plot[0].get_color())


def _plot_rpe_rot(figure_dict: Dict[PairDistanceUnit, Axes], devs: List[RPEResult]) -> None:
    plot_sum = 0
    for dev in devs:
        if not dev.has_rot_dev:
            continue

        plot_sum += 1
        line_plot = figure_dict[dev.rpe_dev.pair_distance_unit].plot(
            dev.mean_pair_distances, np.rad2deg(dev.rot_dev_mean), label=dev.name
        )

        if len(devs) > len({dev.rpe_dev.pair_distance_unit for dev in devs}):
            continue

        violin_plot = figure_dict[dev.rpe_dev.pair_distance_unit].violinplot(
            [
                list(np.rad2deg(rot_list) * dev.drift_factor)
                for rot_list in list(dev.rpe_dev.rot_dev.values())
                if rot_list
            ],
            positions=dev.mean_pair_distances,
            showmeans=True,
            widths=max(0.5, dev.step / 4),
        )
        _set_violin_color(violin_plot, line_plot[0].get_color())

    if plot_sum == 0:
        for ax in figure_dict.values():
            ax.axis(False)


def _set_violin_color(violin_dict: dict, color: str) -> None:
    for component in violin_dict.values():
        if isinstance(component, LineCollection):
            component.set_color(color)
            continue

        if not isinstance(component, list):
            continue

        for collection in component:
            if isinstance(collection, PolyCollection):
                collection.set_facecolor(color)
                collection.set_edgecolor(color)


def _rpy_legend(figure_dict: Dict[str, Dict[PairDistanceUnit, Axes]]):
    for d in figure_dict.values():
        for ax in d.values():
            if ax.lines:
                ax.legend()


def scatter_ate(ate_result: ATEResult, plot_settings: MPLPlotSettings = MPLPlotSettings()):
    """
    Plots the ATE results as a scatter plot with color-coded deviations.

    Args:
        ate_result (ATEResult): ATE result to plot.
        plot_settings (MPLPlotSettings, optional): Plot settings. Defaults to MPLPlotSettings().
    """
    plt.figure()
    _colored_scatter_plot(
        xyz=ate_result.trajectory.pos.xyz,
        c_list=ate_result.pos_dev_comb * plot_settings.unit_multiplier,
        c_label=f"Deviation {plot_settings.unit_str}",
    )

    if not ate_result.has_orientation:
        return

    plt.figure()
    _colored_scatter_plot(
        xyz=ate_result.trajectory.pos.xyz,
        c_list=ate_result.rot_dev_comb * 180 / np.pi,
        c_label="Deviation [°]",
    )


def _colored_scatter_plot(
    xyz: np.ndarray, c_list: list, c_label: str, plot_settings: MPLPlotSettings = MPLPlotSettings()
) -> None:
    """
    Plots 2d positions with colorcode
    """
    plt.axis("equal")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    c_list, lower_bound, upper_bound, c_bar_ticks, c_bar_ticklabels = _setup_cbar_params(c_list, plot_settings)

    sc = plt.scatter(
        xyz[:, 0],
        xyz[:, 1],
        c=c_list,
        cmap="RdYlBu_r",
        vmax=upper_bound,
        vmin=lower_bound,
    )

    cbar: Colorbar = plt.colorbar(sc, format="%.2f")
    cbar.set_label(c_label)
    cbar.set_ticks(c_bar_ticks)
    cbar.set_ticklabels(c_bar_ticklabels)

    if plot_settings.scatter_no_axis:
        plt.axis("off")


def _setup_cbar_params(c_list, plot_settings: MPLPlotSettings):
    """Configures the colorbar ticks and labels for the scatter plot"""
    if plot_settings.scatter_sigma_factor == 0:
        lower_bound = np.min(c_list)
        upper_bound = np.max(c_list)
        geq_leq_dict = {0: "", plot_settings.scatter_cbar_steps: ""}
    else:
        lower_bound = np.max([np.min(c_list), np.mean(c_list) - plot_settings.scatter_sigma_factor * np.std(c_list)])
        upper_bound = np.min([np.max(c_list), np.mean(c_list) + plot_settings.scatter_sigma_factor * np.std(c_list)])
        geq_leq_dict = {0: "$\leq$", plot_settings.scatter_cbar_steps: "$\geq$"}

    c_bar_range = np.abs(upper_bound - lower_bound)

    c_bar_ticks_and_labels = {
        lower_bound
        + i
        / plot_settings.scatter_cbar_steps
        * c_bar_range: f"{geq_leq_dict.get(i, '')}{lower_bound + i/plot_settings.scatter_cbar_steps * c_bar_range:.2f}"
        for i in range(plot_settings.scatter_cbar_steps + 1)
    }

    c_list = np.clip(c_list, lower_bound, upper_bound)

    if lower_bound < 0 and upper_bound > 0 and plot_settings.scatter_cbar_show_zero:
        c_bar_ticks_and_labels = {
            key: value
            for key, value in c_bar_ticks_and_labels.items()
            if abs(key) > 0.5 / plot_settings.scatter_cbar_steps * c_bar_range
        }
        c_bar_ticks_and_labels[0] = "0"

    c_bar_ticks = list(c_bar_ticks_and_labels.keys())
    c_bar_ticklabels = list(c_bar_ticks_and_labels.values())
    return c_list, lower_bound, upper_bound, c_bar_ticks, c_bar_ticklabels
