"""The figures trajectopy offers for trajectories, evaluation results and alignment parameters.

The functions in this module only describe figures (see :class:`~trajectopy.visualization.figure.Figure`);
nothing is drawn until a figure is shown or exported. The collections :func:`trajectory_figures`,
:func:`deviation_figures` and :func:`alignment_figures` are what the GUI displays.
"""

import copy
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui

from trajectopy.core.parameters import AlignmentParameters
from trajectopy.core.settings import PairDistanceUnit, PlotSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.results.ate_result import ATEResult
from trajectopy.results.rpe_result import RPEResult
from trajectopy.utils.colors import color_range
from trajectopy.utils.numeric import rms
from trajectopy.visualization import style
from trajectopy.visualization.axes import get_axis_label, index_axis
from trajectopy.visualization.figure import Figure, RenderFunction

logger = logging.getLogger(__name__)

# Top views (x/y paths) draw at most this many points; denser data is thinned out evenly.
MAX_PATH_POINTS = 200_000
# Color-coded maps draw at most this many points; denser data is thinned out evenly.
MAX_MAP_POINTS = 100_000
MAX_HISTOGRAM_BINS = 300
_COLOR_LEVELS = 256

STATISTICS = ("Min", "Max", "Mean", "Median", "RMS", "STD")


# ----------------------------------------------------------------------------------------------- helpers


def _settings(settings: PlotSettings | None) -> PlotSettings:
    return settings if settings is not None else PlotSettings()


class _TightXViewBox(pg.ViewBox):
    """View box whose x range fits the data exactly when auto-ranging; only the y range is padded."""

    def suggestPadding(self, axis: int) -> float:
        return 0.0 if axis == 0 else super().suggestPadding(axis)


def _add_plot(
    layout: pg.GraphicsLayout,
    row: int,
    settings: PlotSettings,
    *,
    col: int = 0,
    title: str | None = None,
    x_label: str = "",
    y_label: str = "",
    datetime_axis: bool = False,
    legend: bool = False,
    tight_x: bool = True,
) -> pg.PlotItem:
    axis_items = {"bottom": pg.DateAxisItem(orientation="bottom", utcOffset=0)} if datetime_axis else None
    view_box = _TightXViewBox() if tight_x else None
    plot = layout.addPlot(row=row, col=col, axisItems=axis_items, viewBox=view_box)
    if title:
        plot.setTitle(title, size=f"{settings.font_size + 1}pt", color=style.theme().foreground)
    # Passing a style replaces pyqtgraph's default label style, so the color has to be given as well.
    label_style = {"font-size": f"{settings.font_size}pt", "color": style.theme().foreground}
    plot.setLabel("bottom", x_label, **label_style)
    plot.setLabel("left", y_label, **label_style)
    tick_font = QtGui.QFont()
    tick_font.setPointSizeF(max(settings.font_size - 1, 6))
    for name in ("bottom", "left"):
        plot.getAxis(name).setStyle(tickFont=tick_font)
    plot.showGrid(x=True, y=True, alpha=style.theme().grid_alpha)
    # Keep large data sets fluid: only draw what is visible, reduced to the screen resolution.
    plot.setClipToView(True)
    plot.setDownsampling(auto=True, mode="peak")
    if legend:
        plot.addLegend(offset=(-10, 10), labelTextSize=f"{settings.font_size}pt")
    return plot


def _line(plot: pg.PlotItem, x: np.ndarray, y: np.ndarray, index: int, name: str | None, settings: PlotSettings):
    return plot.plot(x, y, pen=style.pen(index, settings.line_width), name=name, antialias=style.antialias(len(x)))


def _link_x(plots: Sequence[pg.PlotItem]) -> None:
    """Shares the x range of stacked plots; the first plot (which shows all data) determines the auto range.

    pyqtgraph's own ``setXLink`` maps ranges by screen position. Since the plots have y axes of different
    widths, the linked views keep widening each other's range and auto-ranging gets switched off, which
    leaves blank borders. Linking the values directly avoids that.
    """
    views = [plot.getViewBox() for plot in plots]
    syncing = False

    def sync(source: pg.ViewBox, x_range: Sequence[float]) -> None:
        nonlocal syncing
        if syncing:
            return
        syncing = True
        try:
            for view in views:
                if view is not source:
                    view.setXRange(*x_range, padding=0)
        finally:
            syncing = False

    for view in views[1:]:
        view.enableAutoRange(x=False)
    for view in views:
        view.sigXRangeChanged.connect(sync)
    _align_left_axes(plots)


def _align_left_axes(plots: Sequence[pg.PlotItem]) -> None:
    """Gives the y axes of stacked plots the width of the widest one, so that their x axes line up."""
    axes = [plot.getAxis("left") for plot in plots]
    natural_widths: dict[pg.AxisItem, float] = {}

    # pyqtgraph recomputes an axis' width from its tick labels in _updateWidth; widen all axes afterwards.
    # This hooks into a private method, so a pyqtgraph version without it only loses the alignment.
    if not all(callable(getattr(axis, "_updateWidth", None)) for axis in axes):
        logger.warning("pyqtgraph has no AxisItem._updateWidth anymore; stacked plots will not be aligned.")
        return

    for axis in axes:

        def update_width(axis: pg.AxisItem = axis, update: Callable[[], None] = axis._updateWidth) -> None:
            update()
            natural_widths[axis] = axis.minimumWidth()
            width = max(natural_widths.values())
            for other in axes:
                other.setMinimumWidth(width)
                other.setMaximumWidth(width)

        axis._updateWidth = update_width
        update_width()


@dataclass
class _TopView:
    """Horizontal coordinates of trajectories as drawn in a top view."""

    xy: list[np.ndarray]
    labels: tuple[str, str]


def _top_view(trajectories: Sequence[Trajectory]) -> _TopView:
    x_label, y_label, _ = get_axis_label(trajectories)
    return _TopView([trajectory.xyz[:, :2] for trajectory in trajectories], (x_label, y_label))


def _top_view_plot(layout: pg.GraphicsLayout, settings: PlotSettings, view: _TopView, **kwargs) -> pg.PlotItem:
    """Plot for x/y paths: equal axis scaling and no clipping/downsampling, which assume increasing x values."""
    # With locked aspect ratio, a tight x range would only shift the padding to the y axis.
    plot = _add_plot(layout, 0, settings, x_label=view.labels[0], y_label=view.labels[1], tight_x=False, **kwargs)
    plot.setAspectLocked(True)
    plot.setClipToView(False)
    plot.setDownsampling(auto=False)
    return plot


def _thin(array: np.ndarray, max_points: int) -> np.ndarray:
    return array[:: max(1, int(np.ceil(len(array) / max_points)))]


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="same")


_BRUSHES: list[QtGui.QBrush] = []


def _value_brushes(values: np.ndarray, lower: float, upper: float) -> list[QtGui.QBrush]:
    if not _BRUSHES:
        lut = style.VALUE_COLORMAP.getLookupTable(nPts=_COLOR_LEVELS, alpha=False)
        _BRUSHES.extend(pg.mkBrush(*rgb) for rgb in lut)
    levels = np.clip((values - lower) / (upper - lower) * (_COLOR_LEVELS - 1), 0, _COLOR_LEVELS - 1)
    levels = np.nan_to_num(levels).astype(int)
    return [_BRUSHES[level] for level in levels]


def _value_map(
    layout: pg.GraphicsLayout,
    trajectory: Trajectory,
    values: np.ndarray,
    value_label: str,
    settings: PlotSettings,
    title: str | None = None,
) -> None:
    """Draws the top view of ``trajectory`` colored by ``values`` with a colorbar."""
    values = _smooth(np.asarray(values, dtype=float), settings.smoothing_window)
    lower, upper = color_range(values, settings.colorbar_max_std)

    view = _top_view([trajectory])
    xy, values = _thin(view.xy[0], MAX_MAP_POINTS), _thin(values, MAX_MAP_POINTS)

    plot = _top_view_plot(layout, settings, view, title=title)
    plot.addItem(
        pg.ScatterPlotItem(
            x=xy[:, 0], y=xy[:, 1], size=settings.marker_size, pen=None, brush=_value_brushes(values, lower, upper)
        )
    )
    colorbar = pg.ColorBarItem(
        values=(lower, upper), colorMap=style.VALUE_COLORMAP, label=value_label, interactive=False, width=15
    )
    layout.addItem(colorbar, row=0, col=1)


def _histogram(plot: pg.PlotItem, values: np.ndarray, index: int, name: str, percentile: float) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    if percentile < 1.0:
        values = values[np.abs(values) <= np.quantile(np.abs(values), percentile)]
    edges = np.histogram_bin_edges(values, bins="auto")
    if len(edges) - 1 > MAX_HISTOGRAM_BINS:
        edges = np.histogram_bin_edges(values, bins=MAX_HISTOGRAM_BINS)
    counts, edges = np.histogram(values, bins=edges)
    plot.plot(
        edges,
        counts,
        stepMode="center",
        fillLevel=0,
        pen=style.pen(index),
        brush=style.brush(index, alpha=70),
        name=name,
        antialias=True,
    )


def _plottable_ate_results(ate_results: Sequence[ATEResult], threshold: float) -> list[ATEResult]:
    """Applies ``ate_remove_above`` (on copies) and drops results without any deviations left."""
    plottable = []
    for ate_result in ate_results:
        if threshold > 0:
            ate_result = copy.deepcopy(ate_result)
            ate_result.remove_ate_above(threshold)
        if len(ate_result.trajectory) == 0:
            logger.warning("Not plotting '%s': no deviations left (see ate_remove_above).", ate_result.name)
            continue
        plottable.append(ate_result)
    return plottable


@dataclass(frozen=True)
class _Component:
    name: str
    values: Callable[[ATEResult], np.ndarray]
    is_rotation: bool


def _position_components(settings: PlotSettings) -> tuple[_Component, ...]:
    if settings.directed_ate:
        return (
            _Component("Along-Track", lambda r: r.pos_dev_along, False),
            _Component("Horizontal Cross-Track", lambda r: r.pos_dev_cross_h, False),
            _Component("Vertical Cross-Track", lambda r: r.pos_dev_cross_v, False),
        )
    return (
        _Component("X", lambda r: r.pos_dev_x, False),
        _Component("Y", lambda r: r.pos_dev_y, False),
        _Component("Z", lambda r: r.pos_dev_z, False),
    )


ROTATION_COMPONENTS = (
    _Component("Roll", lambda r: r.rot_dev_x, True),
    _Component("Pitch", lambda r: r.rot_dev_y, True),
    _Component("Yaw", lambda r: r.rot_dev_z, True),
)


def _component_values(component: _Component, ate_result: ATEResult, settings: PlotSettings) -> np.ndarray:
    values = component.values(ate_result)
    return np.rad2deg(values) if component.is_rotation else values * settings.unit_multiplier


def _variant_name(result_name: str, quantity: str, multiple_results: bool) -> str:
    return f"{result_name}: {quantity}" if multiple_results else quantity


# ------------------------------------------------------------------------------------------ trajectories


def positions_figure(trajectories: Sequence[Trajectory], settings: PlotSettings | None = None) -> Figure:
    """Top view (x/y) of the trajectories."""
    settings = _settings(settings)

    def render(layout: pg.GraphicsLayout) -> None:
        view = _top_view(trajectories)
        plot = _top_view_plot(layout, settings, view, legend=True)
        for index, (trajectory, xy) in enumerate(zip(trajectories, view.xy)):
            xy = _thin(xy, MAX_PATH_POINTS)
            _line(plot, xy[:, 0], xy[:, 1], index, trajectory.name, settings)

    return Figure("Positions", render)


def xyz_figure(trajectories: Sequence[Trajectory], settings: PlotSettings | None = None) -> Figure:
    """Coordinates of the trajectories over time or path length."""
    settings = _settings(settings)
    axis = index_axis(trajectories, settings.use_relative_timestamps)
    labels = get_axis_label(trajectories)

    def render(layout: pg.GraphicsLayout) -> None:
        plots = []
        for dim, label in enumerate(labels):
            plot = _add_plot(
                layout,
                dim,
                settings,
                x_label=axis.label if dim == 2 else "",
                y_label=label,
                datetime_axis=axis.is_datetime,
                legend=dim == 0,
            )
            for index, trajectory in enumerate(trajectories):
                _line(plot, axis.values(trajectory), trajectory.xyz[:, dim], index, trajectory.name, settings)
            plots.append(plot)
        _link_x(plots)

    return Figure("XYZ", render)


def rpy_figure(trajectories: Sequence[Trajectory], settings: PlotSettings | None = None) -> Figure | None:
    """Roll, pitch and yaw of the trajectories over time or path length (None without orientations)."""
    settings = _settings(settings)
    oriented = [trajectory for trajectory in trajectories if trajectory.has_orientation]
    if not oriented:
        return None
    axis = index_axis(oriented, settings.use_relative_timestamps)

    def render(layout: pg.GraphicsLayout) -> None:
        rpy = {id(trajectory): np.rad2deg(trajectory.rpy) for trajectory in oriented}
        plots = []
        for dim, label in enumerate(("roll [°]", "pitch [°]", "yaw [°]")):
            plot = _add_plot(
                layout,
                dim,
                settings,
                x_label=axis.label if dim == 2 else "",
                y_label=label,
                datetime_axis=axis.is_datetime,
                legend=dim == 0,
            )
            for index, trajectory in enumerate(trajectories):
                if id(trajectory) in rpy:
                    _line(plot, axis.values(trajectory), rpy[id(trajectory)][:, dim], index, trajectory.name, settings)
            plots.append(plot)
        _link_x(plots)

    return Figure("RPY", render)


def trajectory_map_figure(trajectories: Sequence[Trajectory], settings: PlotSettings | None = None) -> Figure:
    """Top view of each trajectory colored by velocity or height."""
    settings = _settings(settings)
    z_label = get_axis_label(trajectories)[2]
    multiple = len(trajectories) > 1
    variants: dict[str, RenderFunction] = {}
    for trajectory in trajectories:
        quantities = {"Velocity": (trajectory.absolute_velocity, "velocity [m/s]")}
        quantities["Height"] = (trajectory.xyz[:, 2], z_label)
        for quantity, (values, label) in quantities.items():

            def render(layout, trajectory=trajectory, values=values, label=label) -> None:
                _value_map(layout, trajectory, values, label, settings)

            variants[_variant_name(trajectory.name, quantity, multiple)] = render
    return Figure("Map", variants=variants)


def trajectory_figures(trajectories: Sequence[Trajectory], settings: PlotSettings | None = None) -> list[Figure]:
    """All figures describing the given trajectories."""
    if not trajectories:
        return []
    figures = [
        positions_figure(trajectories, settings),
        xyz_figure(trajectories, settings),
        rpy_figure(trajectories, settings),
        trajectory_map_figure(trajectories, settings),
    ]
    return [figure for figure in figures if figure is not None]


# ---------------------------------------------------------------------------------------------- ATE


def ate_figure(ate_results: Sequence[ATEResult], settings: PlotSettings | None = None) -> Figure:
    """Position (and rotation) deviations over time or path length."""
    settings = _settings(settings)
    axis = index_axis([result.trajectory for result in ate_results], settings.use_relative_timestamps)
    with_rotation = any(result.has_orientation for result in ate_results)

    def render(layout: pg.GraphicsLayout) -> None:
        position_plot = _add_plot(
            layout,
            0,
            settings,
            x_label="" if with_rotation else axis.label,
            y_label=f"position deviation [{settings.unit}]",
            datetime_axis=axis.is_datetime,
            legend=True,
        )
        plots = [position_plot]
        if with_rotation:
            plots.append(
                _add_plot(
                    layout,
                    1,
                    settings,
                    x_label=axis.label,
                    y_label="rotation deviation [°]",
                    datetime_axis=axis.is_datetime,
                )
            )
        for index, result in enumerate(ate_results):
            x = axis.values(result.trajectory)
            _line(position_plot, x, result.pos_dev_comb * settings.unit_multiplier, index, result.name, settings)
            if with_rotation and result.has_orientation:
                _line(plots[1], x, np.rad2deg(result.rot_dev_comb), index, None, settings)
        _link_x(plots)

    return Figure("ATE", render)


def ate_components_figure(ate_results: Sequence[ATEResult], settings: PlotSettings | None = None) -> Figure:
    """Deviations of the single degrees of freedom over time or path length."""
    settings = _settings(settings)
    variants: dict[str, RenderFunction] = {}
    for result in ate_results:
        axis = index_axis([result.trajectory], settings.use_relative_timestamps)

        def render(layout: pg.GraphicsLayout, result=result, axis=axis) -> None:
            groups = [(_position_components(settings), f"position deviation [{settings.unit}]")]
            if result.has_orientation:
                groups.append((ROTATION_COMPONENTS, "rotation deviation [°]"))
            plots = []
            for row, (components, y_label) in enumerate(groups):
                plot = _add_plot(
                    layout,
                    row,
                    settings,
                    x_label=axis.label if row == len(groups) - 1 else "",
                    y_label=y_label,
                    datetime_axis=axis.is_datetime,
                    legend=True,
                )
                x = axis.values(result.trajectory)
                for dim, component in enumerate(components):
                    plot.plot(
                        x,
                        _component_values(component, result, settings),
                        pen=style.pen(style.AXIS_COLORS[dim], settings.line_width),
                        name=component.name,
                        antialias=style.antialias(len(x)),
                    )
                plots.append(plot)
            _link_x(plots)

        variants[result.name] = render
    return Figure("ATE Components", variants=variants)


def ate_histogram_figure(ate_results: Sequence[ATEResult], settings: PlotSettings | None = None) -> Figure:
    """Histograms of the deviations of the single degrees of freedom."""
    settings = _settings(settings)
    variants: dict[str, RenderFunction] = {}
    for result in ate_results:

        def render(layout: pg.GraphicsLayout, result=result) -> None:
            groups = [(_position_components(settings), f"position deviation [{settings.unit}]")]
            if result.has_orientation:
                groups.append((ROTATION_COMPONENTS, "rotation deviation [°]"))
            for row, (components, x_label) in enumerate(groups):
                plot = _add_plot(layout, row, settings, x_label=x_label, y_label="count", legend=True)
                plot.setClipToView(False)
                plot.setDownsampling(auto=False)
                for dim, component in enumerate(components):
                    values = _component_values(component, result, settings)
                    _histogram(plot, values, dim, component.name, settings.hist_percentile)

        variants[result.name] = render
    return Figure("ATE Histograms", variants=variants)


def ate_statistics_figure(ate_results: Sequence[ATEResult], settings: PlotSettings | None = None) -> Figure:
    """Min, max, mean, median, RMS and standard deviation of the deviations as bar charts."""
    settings = _settings(settings)
    with_rotation = any(result.has_orientation for result in ate_results)

    def statistics(result: ATEResult, rotation: bool) -> list[float]:
        # same definitions as the ATEResult properties, but each combined deviation is computed only once
        values = np.rad2deg(result.rot_dev_comb) if rotation else result.pos_dev_comb * settings.unit_multiplier
        return [
            float(np.min(values)),
            float(np.max(values)),
            float(np.mean(values)),
            float(np.median(values)),
            rms(values),
            float(np.std(values)),
        ]

    def render(layout: pg.GraphicsLayout) -> None:
        groups = [(False, f"position deviation [{settings.unit}]")]
        if with_rotation:
            groups.append((True, "rotation deviation [°]"))
        width = 0.8 / len(ate_results)
        for row, (rotation, y_label) in enumerate(groups):
            plot = _add_plot(layout, row, settings, y_label=y_label, legend=True)
            plot.setClipToView(False)
            plot.setDownsampling(auto=False)
            plot.getAxis("bottom").setTicks([list(enumerate(STATISTICS))])
            for index, result in enumerate(ate_results):
                if rotation and not result.has_orientation:
                    continue
                offset = (index - (len(ate_results) - 1) / 2) * width
                bars = pg.BarGraphItem(
                    x=np.arange(len(STATISTICS)) + offset,
                    height=statistics(result, rotation),
                    width=width,
                    brush=style.brush(index),
                    pen=None,
                )
                plot.addItem(bars)
                if plot.legend is not None:
                    plot.legend.addItem(bars, result.name)

    return Figure("ATE Statistics", render)


def ate_distribution_figure(ate_results: Sequence[ATEResult], settings: PlotSettings | None = None) -> Figure:
    """Empirical cumulative distribution of the position (and rotation) deviations."""
    settings = _settings(settings)
    with_rotation = any(result.has_orientation for result in ate_results)

    def edf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.sort(values[np.isfinite(values)])
        return values, np.arange(1, len(values) + 1) / max(len(values), 1)

    def render(layout: pg.GraphicsLayout) -> None:
        position_plot = _add_plot(
            layout,
            0,
            settings,
            x_label=f"position deviation [{settings.unit}]",
            y_label="cumulative probability",
            legend=True,
        )
        rotation_plot = (
            _add_plot(layout, 1, settings, x_label="rotation deviation [°]", y_label="cumulative probability")
            if with_rotation
            else None
        )
        for index, result in enumerate(ate_results):
            _line(position_plot, *edf(result.pos_dev_comb * settings.unit_multiplier), index, result.name, settings)
            if rotation_plot is not None and result.has_orientation:
                _line(rotation_plot, *edf(np.rad2deg(result.rot_dev_comb)), index, None, settings)

    return Figure("ATE Distribution", render)


def ate_map_figure(ate_results: Sequence[ATEResult], settings: PlotSettings | None = None) -> Figure:
    """Top view of the evaluated trajectories colored by their deviations."""
    settings = _settings(settings)
    multiple = len(ate_results) > 1
    variants: dict[str, RenderFunction] = {}
    for result in ate_results:
        quantities: list[tuple[str, np.ndarray, str]] = [
            ("Position", result.pos_dev_comb * settings.unit_multiplier, f"deviation [{settings.unit}]")
        ]
        if result.has_orientation:
            quantities.append(("Rotation", np.rad2deg(result.rot_dev_comb), "deviation [°]"))
        components = list(_position_components(settings))
        if result.has_orientation:
            components += ROTATION_COMPONENTS
        for component in components:
            unit = "°" if component.is_rotation else settings.unit
            values = _component_values(component, result, settings)
            quantities.append((component.name, values, f"deviation [{unit}]"))

        for quantity, values, label in quantities:

            def render(layout, result=result, values=values, label=label) -> None:
                _value_map(layout, result.trajectory, values, label, settings)

            variants[_variant_name(result.name, quantity, multiple)] = render
    return Figure("ATE Map", variants=variants)


# ---------------------------------------------------------------------------------------------- RPE


def rpe_figures(rpe_results: Sequence[RPEResult], settings: PlotSettings | None = None) -> list[Figure]:
    """Relative pose errors over the pair distance, one figure per pair distance unit."""
    settings = _settings(settings)
    figures = []
    for unit, title in ((PairDistanceUnit.METER, "RPE (per meter)"), (PairDistanceUnit.SECOND, "RPE (per second)")):
        results = [result for result in rpe_results if result.rpe_dev.pair_distance_unit == unit]
        if not results:
            continue

        def render(layout: pg.GraphicsLayout, results=results) -> None:
            first = results[0]
            with_rotation = any(result.has_rot_dev for result in results)
            position_plot = _add_plot(
                layout,
                0,
                settings,
                x_label="" if with_rotation else f"pair distance [{first.pair_distance_unit}]",
                y_label=f"position drift [{first.pos_drift_unit}]",
                legend=True,
            )
            plots = [position_plot]
            if with_rotation:
                plots.append(
                    _add_plot(
                        layout,
                        1,
                        settings,
                        x_label=f"pair distance [{first.pair_distance_unit}]",
                        y_label=f"rotation drift [{first.rot_drift_unit}]",
                    )
                )
            for index, result in enumerate(results):
                x = np.asarray(result.mean_pair_distances)
                series: list[tuple[pg.PlotItem, np.ndarray, np.ndarray, str | None]] = [
                    (position_plot, np.asarray(result.pos_dev_mean), np.asarray(result.pos_std), result.name)
                ]
                if with_rotation and result.has_rot_dev:
                    series.append((plots[1], np.rad2deg(result.rot_dev_mean), np.rad2deg(result.rot_std), None))
                for plot, mean, std, name in series:
                    plot.addItem(pg.ErrorBarItem(x=x, y=mean, height=2 * std, beam=0, pen=style.pen(index, 1.0)))
                    plot.plot(
                        x,
                        mean,
                        pen=style.pen(index, settings.line_width),
                        symbol="o",
                        symbolSize=6,
                        symbolBrush=style.brush(index),
                        symbolPen=None,
                        name=name,
                        antialias=True,
                    )
            for plot in plots:
                plot.setClipToView(False)
            _link_x(plots)

        figures.append(Figure(title, render))
    return figures


def deviation_figures(
    ate_results: Sequence[ATEResult] = (),
    rpe_results: Sequence[RPEResult] = (),
    settings: PlotSettings | None = None,
) -> list[Figure]:
    """All figures describing absolute (ATE) and relative (RPE) evaluation results.

    The same figures are used for one and for several results; figures that show one result at a time
    offer the results as variants.
    """
    settings = _settings(settings)
    ate_results = _plottable_ate_results(ate_results, settings.ate_remove_above)
    figures: list[Figure] = []
    if ate_results:
        figures += [
            ate_figure(ate_results, settings),
            ate_components_figure(ate_results, settings),
            ate_histogram_figure(ate_results, settings),
            ate_statistics_figure(ate_results, settings),
            ate_distribution_figure(ate_results, settings),
            ate_map_figure(ate_results, settings),
        ]
    figures += rpe_figures(rpe_results, settings)
    return figures


# ---------------------------------------------------------------------------------------------- alignment


def parameter_matrix_figure(
    parameters: AlignmentParameters, correlation: bool = False, settings: PlotSettings | None = None
) -> Figure:
    """Covariance or correlation matrix of estimated alignment parameters."""
    settings = _settings(settings)
    labels = parameters.params_labels(enabled_only=True, lower_case=True)
    matrix = np.array(parameters.get_covariance_matrix(enabled_only=True), dtype=float)
    if correlation:
        std = np.sqrt(np.diag(matrix))
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = matrix / np.outer(std, std)
        np.fill_diagonal(matrix, np.nan)
        limit, value_format = 1.0, "{:.2f}"
    else:
        finite = np.abs(matrix[np.isfinite(matrix)])
        limit, value_format = (float(finite.max()) if finite.size else 0.0) or 1.0, "{:.2e}"

    def render(layout: pg.GraphicsLayout) -> None:
        plot = _add_plot(layout, 0, settings, tight_x=False)
        plot.showGrid(x=False, y=False)
        plot.setClipToView(False)
        plot.setMouseEnabled(x=False, y=False)
        colormap = style.diverging_colormap()
        image = pg.ImageItem(np.nan_to_num(matrix), axisOrder="row-major")
        image.setColorMap(colormap)
        image.setLevels((-limit, limit))
        plot.addItem(image)
        plot.invertY(True)
        plot.setAspectLocked(True)
        ticks = [[(i + 0.5, label) for i, label in enumerate(labels)]]
        plot.getAxis("bottom").setTicks(ticks)
        plot.getAxis("left").setTicks(ticks)
        for (row, col), value in np.ndenumerate(matrix):
            if np.isfinite(value):
                cell_color = colormap.map((np.clip(value, -limit, limit) + limit) / (2 * limit), mode="qcolor")
                text = pg.TextItem(
                    value_format.format(value), color=style.readable_text_color(cell_color), anchor=(0.5, 0.5)
                )
                text.setPos(col + 0.5, row + 0.5)
                plot.addItem(text)
        colorbar = pg.ColorBarItem(
            values=(-limit, limit),
            colorMap=colormap,
            label="correlation" if correlation else "covariance",
            interactive=False,
            width=15,
        )
        layout.addItem(colorbar, row=0, col=1)

    return Figure("Parameter Correlation" if correlation else "Parameter Covariance", render)


def alignment_figures(parameters: AlignmentParameters, settings: PlotSettings | None = None) -> list[Figure]:
    """Covariance and correlation of estimated alignment parameters."""
    return [
        parameter_matrix_figure(parameters, correlation=False, settings=settings),
        parameter_matrix_figure(parameters, correlation=True, settings=settings),
    ]


__all__ = [
    "alignment_figures",
    "ate_components_figure",
    "ate_distribution_figure",
    "ate_figure",
    "ate_histogram_figure",
    "ate_map_figure",
    "ate_statistics_figure",
    "deviation_figures",
    "parameter_matrix_figure",
    "positions_figure",
    "rpe_figures",
    "rpy_figure",
    "trajectory_figures",
    "trajectory_map_figure",
    "xyz_figure",
]
