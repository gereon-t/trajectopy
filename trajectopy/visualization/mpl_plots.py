import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from trajectopy.core.settings import MPLPlotSettings, PairDistanceUnit
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.lib.alignment.parameters import AlignmentParameters
from trajectopy.results.ate_result import ATEResult
from trajectopy.results.rpe_result import RPEResult
from trajectopy.utils.common import (
    TrajectoriesSorting,
    derive_xlabel_from_sortings,
    get_axis_label,
    get_sorting,
    set_aspect_equal_3d,
)
from trajectopy.utils.definitions import DATE_FORMATTER, Sorting

logger = logging.getLogger(__name__)


plt.rcParams["figure.max_open_warning"] = 50


def mplstyle_file_path() -> str:
    custom_path = os.path.join("./custom.mplstyle")
    if os.path.isfile(custom_path):
        logger.info("Using custom matplotlib style from %s", custom_path)
        return custom_path

    logger.info(
        "Using default settings for matplotlib style. You can use custom styles by creating a 'custom.mplstyle' file in the current directory."
    )
    return os.path.join(os.path.dirname(__file__), "default.mplstyle")


base_path = os.path.join(os.path.dirname(__file__))
MPL_STYLE_PATH = mplstyle_file_path()
plt.style.use(MPL_STYLE_PATH)
logger.info("Using matplotlib style: %s", MPL_STYLE_PATH)


# see https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def _heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, aspect="auto", **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


# see https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def _annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.ma.is_masked(val) or np.isnan(val):
                continue  # Skip masked or NaN entries
            kw.update(color=textcolors[int(im.norm(val) > threshold)])
            text = im.axes.text(j, i, valfmt(val, None), **kw)
            texts.append(text)

    return texts


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
    deviation_list: list[ATEResult],
    plot_settings: MPLPlotSettings = MPLPlotSettings(),
) -> None:
    ax_pos = plt.subplot(2, 1, 1)
    ax_pos.set_xlabel(f"Deviation {plot_settings.unit_str}")
    ax_pos.set_ylabel("Cumulative Probability")

    for dev in deviation_list:
        sorted_comb_pos_dev = np.sort(dev.pos_dev_comb)
        pos_norm_cdf = np.arange(len(sorted_comb_pos_dev)) / float(len(sorted_comb_pos_dev))
        ax_pos.plot(sorted_comb_pos_dev * plot_settings.unit_multiplier, pos_norm_cdf)


def plot_rotation_ate_edf(deviation_list: list[ATEResult]) -> None:
    if all(dev.abs_dev.rot_dev is None for dev in deviation_list):
        return

    ax_rot = plt.subplot(2, 1, 2)

    ax_rot.set_xlabel("Deviation [°]")
    ax_rot.set_ylabel("Cumulative Probability")

    for dev in deviation_list:
        if dev.abs_dev.rot_dev is None:
            continue
        sorted_comb_rot_dev = np.sort(np.rad2deg(dev.rot_dev_comb))
        rot_norm_cdf = np.arange(len(sorted_comb_rot_dev)) / float(len(sorted_comb_rot_dev))
        ax_rot.plot(sorted_comb_rot_dev, rot_norm_cdf)


def plot_rpe_pos(figure_dict: dict[PairDistanceUnit, Axes], devs: list[RPEResult]) -> None:
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


def plot_rpe_rot(figure_dict: dict[PairDistanceUnit, Axes], devs: list[RPEResult]) -> None:
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


def add_rpy_legend(figure_dict: dict[str, dict[PairDistanceUnit, Axes]]):
    for d in figure_dict.values():
        for ax in d.values():
            if ax.lines:
                ax.legend()


def _colored_scatter_plot(
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


def plot_trajectories(
    trajectories: list[Trajectory], scatter_3d: bool = False
) -> tuple[Figure, Figure, Figure | None]:
    """Plots the trajectories in 2d or 3d using matplotlib.

    This function creates one 2D or 3D plot for the xy(z) coordinates of the trajectories,
    one subplot for the xyz coordinates and one subplot for the rpy angles.

    Args:
        trajectories (List[Trajectory]): List of trajectories to plot.
        scatter_3d (bool, optional): Whether to create a 3D scatter plot. Defaults to False.

    Returns:
        Tuple[Figure, Figure, Union[Figure, None]]: Figures for the position, xyz and rpy plots.
    """
    fig_pos = plot_positions(trajectories=trajectories, scatter_3d=scatter_3d)
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


def plot_ate_3d(ate_results: list[ATEResult], plot_settings: MPLPlotSettings = MPLPlotSettings()) -> Figure:
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
        if len(ate_result.index) == 0:
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
    ate_results: list[ATEResult],
    plot_settings: MPLPlotSettings = MPLPlotSettings(),
    mode: str = "positions",
) -> Figure:
    """Plots multiple ATE results as bars for different characteristics.

    Shows min, max, mean, median, rms, std using matplotlib.

    Args:
        ate_results (List[ATEResult]): List of ATE results.
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
    ate_results: ATEResult | list[ATEResult],
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
    all_unix = all(traj.is_unix_time for traj in trajectories_list)
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
        if len(dev.index) == 0:
            logger.warning("Skipping %s as it has no data", dev.name)
            continue

        index = (
            dev.trajectory.datetimes
            if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME
            else dev.index
        )

        ax_pos.plot(index, dev.pos_dev_comb * plot_settings.unit_multiplier)
        if ax_rot is not None:
            ax_rot.plot(index, np.rad2deg(dev.rot_dev_comb))

    fig.legend([dev.name for dev in deviation_list], ncol=3, loc="upper center")
    plt.tight_layout()
    return fig


def plot_ate_dof(
    ate_result: ATEResult,
    plot_settings: MPLPlotSettings = MPLPlotSettings(),
) -> Figure:
    """Plots ATE DOF (Degrees of Freedom) for the given ATEResult as a line plot using matplotlib.

    The DOF plot shows the deviations in the x, y, and z directions for position and rotation.

    Args:
        ate_result (ATEResult): ATE result to plot.
        plot_settings (MPLPlotSettings, optional): Plot settings. Defaults to MPLPlotSettings().

    Returns:
        Figure: Figure containing the plot.
    """
    trajectory = ate_result.trajectory
    x_label = derive_xlabel_from_sortings(
        TrajectoriesSorting.ALL_SPATIAL if trajectory.sorting == Sorting.PATH_LENGTH else TrajectoriesSorting.ALL_TIME,
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

    if len(ate_result.index) == 0:
        logger.warning("Skipping %s as it has no data", ate_result.name)
        return fig

    pos_dev_x = ate_result.pos_dev_along if plot_settings.directed_ate else ate_result.pos_dev_x
    pos_dev_y = ate_result.pos_dev_cross_h if plot_settings.directed_ate else ate_result.pos_dev_y
    pos_dev_z = ate_result.pos_dev_cross_v if plot_settings.directed_ate else ate_result.pos_dev_z

    index = ate_result.trajectory.datetimes if is_unix_time else ate_result.index

    ax_pos.plot(
        index,
        pos_dev_x * plot_settings.unit_multiplier,
        label="Along-Track" if plot_settings.directed_ate else "X",
    )
    ax_pos.plot(
        index,
        pos_dev_y * plot_settings.unit_multiplier,
        label="Horizontal Cross-Track" if plot_settings.directed_ate else "Y",
    )
    ax_pos.plot(
        index,
        pos_dev_z * plot_settings.unit_multiplier,
        label="Vertical Cross-Track" if plot_settings.directed_ate else "Z",
    )
    ax_pos.legend()

    if ax_rot is not None:
        ax_rot.plot(index, np.rad2deg(ate_result.rot_dev_x), label="Roll")
        ax_rot.plot(index, np.rad2deg(ate_result.rot_dev_y), label="Pitch")
        ax_rot.plot(index, np.rad2deg(ate_result.rot_dev_z), label="Yaw")
        ax_rot.legend()

    ax_pos.set_title(f"{ate_result.name}")
    plt.tight_layout()
    return fig


def plot_ate_edf(
    ate_results: ATEResult | list[ATEResult],
    plot_settings: MPLPlotSettings = MPLPlotSettings(),
) -> Figure:
    """
    Plots ATE EDF for the given ATEResult(s) as a line plot using matplotlib.
    The EDF (Empirical Distribution Function) shows the cumulative probability of the deviations.
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


def plot_rpe(rpe_results: list[RPEResult]) -> tuple[Figure, Figure]:
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

    figure_dict: dict[str, dict[PairDistanceUnit, Axes]] = {
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


def scatter_ate(ate_result: ATEResult, plot_settings: MPLPlotSettings = MPLPlotSettings()) -> tuple[Figure, Figure]:
    """
    Plots the ATE results as a scatter plot with color-coded deviations.

    Args:
        ate_result (ATEResult): ATE result to plot.
        plot_settings (MPLPlotSettings, optional): Plot settings. Defaults to MPLPlotSettings().
    """
    pos_fig = plt.figure()
    _colored_scatter_plot(
        xyz=ate_result.trajectory.xyz,
        c_list=ate_result.pos_dev_comb * plot_settings.unit_multiplier,
        c_label=f"Deviation {plot_settings.unit_str}",
        plot_settings=plot_settings,
    )

    if not ate_result.has_orientation:
        return pos_fig, None

    rot_fig = plt.figure()
    _colored_scatter_plot(
        xyz=ate_result.trajectory.xyz,
        c_list=ate_result.rot_dev_comb * 180 / np.pi,
        c_label="Deviation [°]",
        plot_settings=plot_settings,
    )
    return pos_fig, rot_fig


def plot_positions(trajectories: list[Trajectory], scatter_3d: bool = False) -> Figure:
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


def plot_xyz(trajectories: list[Trajectory]) -> Figure:
    """Plots xyz coordinates of trajectories as subplots"""
    fig_xyz, axs_xyz = plt.subplots(3, 1, sharex=True)

    for ax, label in zip(axs_xyz, get_axis_label(trajectories=trajectories)):
        ax.set_ylabel(label)

    trajectories_sorting = get_sorting([traj.sorting for traj in trajectories])
    all_unix = all(traj.is_unix_time for traj in trajectories)
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
                (traj.datetimes if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME else traj.index),
                xyz[:, j],
            )

    fig_xyz.legend(legend_names, ncol=4, loc="upper center")
    return fig_xyz


def plot_rpy(trajectories: list[Trajectory]) -> Figure | None:
    """Plots rpy coordinates of trajectories as subplots"""
    fig_rpy, axs_rpy = plt.subplots(3, 1, sharex=True)
    trajectories_sorting = get_sorting([traj.sorting for traj in trajectories])
    all_unix = all(traj.is_unix_time for traj in trajectories)
    x_label = derive_xlabel_from_sortings(trajectories_sorting, all_unix)

    axs_rpy[-1].set_xlabel(x_label)

    if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME:
        axs_rpy[-1].xaxis.set_major_formatter(DATE_FORMATTER)

    not_empty = False
    legend_names = []
    for traj in trajectories:
        # rpy fig
        if traj.rotations and len(traj.rotations) > 0:
            legend_names.append(traj.name)
            rpy = traj.rpy
            ylabels = ["roll [°]", "pitch [°]", "yaw [°]"]
            for j, (ax, yl) in enumerate(zip(axs_rpy, ylabels)):
                ax.plot(
                    (
                        traj.datetimes
                        if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME
                        else traj.index
                    ),
                    np.rad2deg(rpy[:, j]),
                )
                ax.set_ylabel(yl)
            not_empty = True

    fig_rpy.legend(legend_names, ncol=4, loc="upper center")

    return fig_rpy if not_empty else None
