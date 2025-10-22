"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colorbar import Colorbar

from trajectopy.core.evaluation.ate_result import ATEResult
from trajectopy.core.evaluation.rpe_result import RPEResult
from trajectopy.settings import MPLPlotSettings, PairDistanceUnit

logger = logging.getLogger("root")


def _norm_hist(*, l, mm: bool = False, alpha: float = 0.5, norm: bool = True, percentile: float = 1.0) -> None:
    """
    Plots a histogram
    """
    l = l[np.abs(l) > 1e-6]

    percentile_value = np.percentile(np.abs(l), percentile * 100)
    l = l[np.abs(l) <= percentile_value]

    if len(l) == 0:
        return 0.0

    auto_bin_edges = np.histogram_bin_edges(l, bins="auto")
    auto_num_bins = len(auto_bin_edges) - 1
    num_bins = min(auto_num_bins, 300)

    if auto_num_bins > 300:
        logger.debug(
            f"Limited number of bins to 300, as more than 300 bins were requested ({len(auto_bin_edges) - 1})."
        )

    hist, bin_edges = np.histogram(l, bins=num_bins)

    if norm:
        hist = hist / len(l)
    if mm:
        bin_edges *= 1000

    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge", alpha=alpha)
    return max(hist) if len(hist) > 0 else 0.0


def _stair_hist(*, l, mm: bool = False, linewidth: float = 1.5, percentile: float = 1.0) -> None:
    """
    Plots a stair histogram
    """
    l = l[np.abs(l) > 1e-6]

    percentile_value = np.percentile(np.abs(l), percentile * 100)
    l = l[np.abs(l) <= percentile_value]

    auto_bin_edges = np.histogram_bin_edges(l, bins="auto")
    num_bins = min(len(auto_bin_edges) - 1, 300)
    hist, bin_edges = np.histogram(l, bins=num_bins)

    n_hist = hist / len(l)
    if mm:
        bin_edges *= 1000
    plt.stairs(hist, bin_edges, linewidth=linewidth)
    return max(n_hist)


def plot_rotation_ate_hist(devs: ATEResult, plot_settings: MPLPlotSettings = MPLPlotSettings()) -> None:
    roll = np.rad2deg(devs.rot_dev_x)
    pitch = np.rad2deg(devs.rot_dev_y)
    yaw = np.rad2deg(devs.rot_dev_z)

    plt.xlabel("[°]")
    plt.ylabel("counts")
    if plot_settings.hist_as_stairs:
        _stair_hist(l=yaw, percentile=plot_settings.hist_percentile)
        _stair_hist(l=pitch, percentile=plot_settings.hist_percentile)
        _stair_hist(l=roll, percentile=plot_settings.hist_percentile)
    else:
        _norm_hist(l=yaw, alpha=0.6, norm=False, percentile=plot_settings.hist_percentile)
        _norm_hist(l=pitch, alpha=0.6, norm=False, percentile=plot_settings.hist_percentile)
        _norm_hist(l=roll, alpha=0.4, norm=False, percentile=plot_settings.hist_percentile)
    plt.legend(["yaw", "pitch", "roll"])


def plot_position_ate_hist(devs: ATEResult, plot_settings: MPLPlotSettings = MPLPlotSettings()):
    deviations_xa = devs.abs_dev.directed_pos_dev[:, 0] if plot_settings.directed_ate else devs.abs_dev.pos_dev[:, 0]
    deviations_yh = devs.abs_dev.directed_pos_dev[:, 1] if plot_settings.directed_ate else devs.abs_dev.pos_dev[:, 1]
    deviations_zv = devs.abs_dev.directed_pos_dev[:, 2] if plot_settings.directed_ate else devs.abs_dev.pos_dev[:, 2]
    labels = ["vertical", "horizontal", "along"] if plot_settings.directed_ate else ["z", "y", "x"]

    plt.xlabel(plot_settings.unit_str)
    plt.ylabel("counts")
    if plot_settings.hist_as_stairs:
        _stair_hist(l=deviations_zv, mm=plot_settings.ate_unit_is_mm, percentile=plot_settings.hist_percentile)
        _stair_hist(l=deviations_yh, mm=plot_settings.ate_unit_is_mm, percentile=plot_settings.hist_percentile)
        _stair_hist(l=deviations_xa, mm=plot_settings.ate_unit_is_mm, percentile=plot_settings.hist_percentile)
    else:
        _norm_hist(
            l=deviations_zv, mm=plot_settings.ate_unit_is_mm, norm=False, percentile=plot_settings.hist_percentile
        )
        _norm_hist(
            l=deviations_yh, mm=plot_settings.ate_unit_is_mm, norm=False, percentile=plot_settings.hist_percentile
        )
        _norm_hist(
            l=deviations_xa, mm=plot_settings.ate_unit_is_mm, norm=False, percentile=plot_settings.hist_percentile
        )

    plt.legend(labels)


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


def plot_rpe_pos(figure_dict: Dict[PairDistanceUnit, Axes], devs: List[RPEResult]) -> None:
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


def plot_rpe_rot(figure_dict: Dict[PairDistanceUnit, Axes], devs: List[RPEResult]) -> None:
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


def add_rpy_legend(figure_dict: Dict[str, Dict[PairDistanceUnit, Axes]]):
    for d in figure_dict.values():
        for ax in d.values():
            if ax.lines:
                ax.legend()


def colored_scatter_plot(
    xyz: np.ndarray,
    c_list: list,
    c_label: str,
    x_label: str = "x [m]",
    y_label: str = "y [m]",
    plot_settings: MPLPlotSettings = MPLPlotSettings(),
) -> None:
    """
    Plots 2d positions with colorcode
    """
    plt.axis("equal")

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if plot_settings.scatter_smooth:
        c_list = np.convolve(
            c_list, np.ones(plot_settings.scatter_smooth_window) / plot_settings.scatter_smooth_window, mode="same"
        )

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

    if plot_settings.scatter_hide_axes:
        plt.axis("off")


def _setup_cbar_params(c_list, plot_settings: MPLPlotSettings):
    """Configures the colorbar ticks and labels for the scatter plot"""
    if plot_settings.colorbar_max_std <= 0:
        lower_bound = np.min(c_list)
        upper_bound = np.max(c_list)
        geq_leq_dict = {0: "", plot_settings.colorbar_steps: ""}
    else:
        lower_bound = np.max(
            [
                np.min(c_list),
                np.mean(c_list) - plot_settings.colorbar_max_std * np.std(c_list),
            ]
        )
        upper_bound = np.min(
            [
                np.max(c_list),
                np.mean(c_list) + plot_settings.colorbar_max_std * np.std(c_list),
            ]
        )
        geq_leq_dict = {0: "$\\leq$", plot_settings.colorbar_steps: "$\\geq$"}

    c_bar_range = np.abs(upper_bound - lower_bound)

    c_bar_ticks_and_labels = {
        lower_bound
        + i
        / plot_settings.colorbar_steps
        * c_bar_range: f"{geq_leq_dict.get(i, '')}{lower_bound + i/plot_settings.colorbar_steps * c_bar_range:.2f}"
        for i in range(plot_settings.colorbar_steps + 1)
    }

    c_list = np.clip(c_list, lower_bound, upper_bound)

    if lower_bound < 0 and upper_bound > 0 and plot_settings.colorbar_show_zero_crossing:
        c_bar_ticks_and_labels = {
            key: value
            for key, value in c_bar_ticks_and_labels.items()
            if abs(key) > 0.5 / plot_settings.colorbar_steps * c_bar_range
        }
        c_bar_ticks_and_labels[0] = "0"

    c_bar_ticks = list(c_bar_ticks_and_labels.keys())
    c_bar_ticklabels = list(c_bar_ticks_and_labels.values())
    return c_list, lower_bound, upper_bound, c_bar_ticks, c_bar_ticklabels
