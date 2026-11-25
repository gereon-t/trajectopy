"""Diagnostic views of the interactive alignment.

Each view shows the deviation at equal time (trajectory minus reference) so that the signature of one kind of
error stands out, says what to look for and offers the matching correction. Hovering over a point marks the
poses at its time in the main view.

The axes keep their range while the data changes a little (e.g. while dragging), so that the plots do not
jump; they adapt when the data leaves the visible range or fills only a small part of it.
"""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from trajectopy.gui.alignment.session import AlignmentSession
from trajectopy.processing.lib.alignment import diagnostics
from trajectopy.processing.lib.alignment.diagnostics import Diagnosis, Residuals
from trajectopy.visualization import style

LINE_WIDTH = 2.0
POINT_SIZE = 4
HOVER_DISTANCE = 12.0  # px
TIME_LABEL = "time since reference start [s]"
COMPONENT_COLORS = {"along-track": "#2ca02c", "cross-track": "#9467bd", "vertical": "#8c564b"}
STANDSTILL_COLOR = "#7f7f7f"
FIT_COLOR = "#d62728"
NO_DATA = "No poses at equal time yet: place the trajectory and synchronize the time first."
SHIFT_KEYS = ("shift_x", "shift_y", "shift_z")
AXIS_WIDTH = 62  # px: fixed, so that changing tick labels do not move the plot area
RANGE_PADDING = 0.25  # headroom, so that the data can change a little before the range adapts
RANGE_SHRINK = 0.25  # the range adapts if the data fills less than this fraction of it


def _make_plot(y_label: str, x_label: str = "", view_box: pg.ViewBox | None = None) -> pg.PlotItem:
    plot = pg.PlotItem(viewBox=view_box)
    plot.setLabel("left", y_label)
    plot.setLabel("bottom", x_label)
    plot.getAxis("left").setWidth(AXIS_WIDTH)
    plot.showGrid(x=True, y=True, alpha=style.theme().grid_alpha)
    plot.setMenuEnabled(False)
    plot.addLegend(offset=(-10, 5), labelTextSize="8pt")
    plot.setTitle(" ", size="9pt")  # always a title row, so that setting a title does not move the plot
    return plot


class StableRange:
    """Keeps the visible range of a plot while its data changes a little, so that the axes do not jump.

    The range adapts only if the data leaves it or fills less than a quarter of it. Once the user zooms or pans,
    the range is left alone until :meth:`release` (a double-click in the plot).
    """

    def __init__(self, plot: pg.PlotItem, x: bool = True, y: bool = True) -> None:
        self.view_box = plot.getViewBox()
        self.view_box.disableAutoRange()
        self.axes = (x, y)
        self.user_range = False
        self._setting = False
        self.view_box.sigRangeChangedManually.connect(self._on_manual)

    def _on_manual(self, *_) -> None:
        self.user_range = True

    def release(self) -> None:
        self.user_range = False

    def update(self, x: np.ndarray | list, y: np.ndarray | list) -> None:
        """Adapts the range to the data (all points of the plot), if needed."""
        if self.user_range:
            return
        current = self.view_box.viewRange()
        for axis, (follow, values) in enumerate(zip(self.axes, (x, y), strict=True)):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if not follow or len(values) == 0:
                continue
            low, high = float(np.min(values)), float(np.max(values))
            span = max(high - low, 1e-9 * max(abs(low), abs(high), 1.0), 1e-12)
            shown_low, shown_high = current[axis]
            shown_span = shown_high - shown_low
            outside = low < shown_low or high > shown_high
            too_small = span < RANGE_SHRINK * shown_span
            if outside or too_small:
                padding = RANGE_PADDING * span
                if axis == 0:
                    self.view_box.setXRange(low - padding, high + padding, padding=0)
                else:
                    self.view_box.setYRange(low - padding, high + padding, padding=0)


def _scatter(plot: pg.PlotItem, color: str, name: str | None = None, size: int = POINT_SIZE) -> pg.ScatterPlotItem:
    item = pg.ScatterPlotItem(size=size, pen=None, brush=pg.mkBrush(color), name=name)
    plot.addItem(item)
    return item


def _colored(text: str, color: str) -> str:
    return f"<span style='color: {color}'>{text}</span>"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start and end (exclusive) of the runs of True values."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[0::2], edges[1::2], strict=True))


class DiagnosticView(QtWidgets.QWidget):
    """A header with an explanation and correction buttons above plots."""

    time_hovered = QtCore.Signal(object)  # timestamp (of the reference) under the mouse, or None

    def __init__(
        self,
        session: AlignmentSession,
        explanation: str,
        parent=None,
        signature_of: tuple[str, ...] | None = None,
        details: str = "",
    ) -> None:
        """
        Args:
            explanation (str): What to look for, in one short line.
            signature_of (tuple[str, ...], optional): Parameters whose signature the view shows. If given, the
                view can remove the effects of all other parameters (partial residuals).
            details (str, optional): Longer explanation, shown as tooltip.
        """
        super().__init__(parent)
        self.session = session
        self.residuals: Residuals | None = None
        self._actual: Residuals | None = None
        self._diagnosis: Diagnosis | None = None
        self._signature_of = signature_of
        self._t0 = float(session.preview.reference_t[0])
        self._points: dict[pg.ViewBox, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._time_axes: list[pg.ViewBox] = []
        self._ranges: list[StableRange] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 0)
        layout.setSpacing(4)
        # the explanation gets the full width and wraps (it never changes, so nothing moves while dragging);
        # the controls get a row of their own, so that neither cuts off the other in a narrow window
        self.explanation = QtWidgets.QLabel(f"{explanation} (?)" if details else explanation)
        self.explanation.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.explanation.setWordWrap(True)
        self.explanation.setToolTip(
            f"{details}<br><br>Double-click a plot to let its axes follow the data again after zooming."
            if details
            else "Double-click a plot to let its axes follow the data again after zooming."
        )
        layout.addWidget(self.explanation)
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)
        self.signature_check: QtWidgets.QCheckBox | None = None
        if signature_of is not None:
            self.signature_check = QtWidgets.QCheckBox("Hide effects of other parameters")
            self.signature_check.setChecked(True)
            self.signature_check.setToolTip(
                "Checked: the plot shows the deviation minus what all other parameters explain (as estimated "
                "jointly by the diagnosis), so that only the pattern this view is about remains, plus what no "
                "parameter explains.<br>Unchecked: the plot shows the actual deviation."
            )
            self.signature_check.toggled.connect(lambda _: self.set_data(self._actual, self._diagnosis))
            controls.addWidget(self.signature_check)
        controls.addStretch(1)
        self.buttons = QtWidgets.QHBoxLayout()
        self.buttons.setSpacing(6)
        self.action_buttons: list[QtWidgets.QPushButton] = []
        controls.addLayout(self.buttons)
        layout.addLayout(controls)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, 1)
        self.graphics.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.graphics.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def stable_range(self, plot: pg.PlotItem, x: bool = True, y: bool = True) -> StableRange:
        stable = StableRange(plot, x, y)
        self._ranges.append(stable)
        return stable

    def _on_mouse_clicked(self, event) -> None:
        if event.double():
            for stable in self._ranges:
                stable.release()
            self.refresh()

    def add_button(self, text: str, tooltip: str, slot) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setToolTip(tooltip)
        button.clicked.connect(slot)
        self.buttons.addWidget(button)
        self.action_buttons.append(button)
        return button

    def set_actions_enabled(self, enabled: bool) -> None:
        for button in self.action_buttons:
            button.setEnabled(enabled)

    def set_data(self, residuals: Residuals | None, diagnosis: Diagnosis | None) -> None:
        """The actual deviation and its diagnosis; the view shows the partial residuals if asked to."""
        self._actual, self._diagnosis = residuals, diagnosis
        self.residuals = residuals
        if (
            residuals is not None
            and diagnosis is not None
            and self._signature_of is not None
            and self.signature_check is not None
            and self.signature_check.isChecked()
        ):
            session = self.session
            self.residuals = diagnostics.partial_residuals(
                session.preview, session.state, residuals.pivot, diagnosis, self._signature_of
            )
        if self.isVisible():
            self.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        raise NotImplementedError

    # --------------------------------------------------------------------------------------------- hovering

    def _register(self, plot: pg.PlotItem, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> None:
        """Points of ``plot`` that mark their time when hovered."""
        self._points[plot.getViewBox()] = (np.asarray(x), np.asarray(y), np.asarray(t))

    def _register_time_axis(self, plot: pg.PlotItem) -> None:
        self._time_axes.append(plot.getViewBox())

    def leaveEvent(self, event) -> None:
        self.time_hovered.emit(None)
        super().leaveEvent(event)

    def _on_mouse_moved(self, scene_position) -> None:
        for view_box in self._time_axes:
            if view_box.sceneBoundingRect().contains(scene_position):
                self.time_hovered.emit(float(view_box.mapSceneToView(scene_position).x()) + self._t0)
                return
        for view_box, (x, y, t) in self._points.items():
            if len(x) == 0 or not view_box.sceneBoundingRect().contains(scene_position):
                continue
            position = view_box.mapSceneToView(scene_position)
            pixel = view_box.viewPixelSize()
            distance = np.hypot((x - position.x()) / (pixel[0] or 1.0), (y - position.y()) / (pixel[1] or 1.0))
            nearest = int(np.argmin(distance))
            self.time_hovered.emit(float(t[nearest]) if distance[nearest] <= HOVER_DISTANCE else None)
            return
        self.time_hovered.emit(None)


# ---------------------------------------------------------------------------------------------- over time


class _TimeShiftViewBox(pg.ViewBox):
    """Dragging with the left button shifts the trajectory in time instead of panning."""

    def __init__(self, session: AlignmentSession) -> None:
        super().__init__()
        self._session = session
        self._start: tuple[float, float] | None = None

    def mouseDragEvent(self, ev, axis=None) -> None:
        if axis is not None or ev.button() != QtCore.Qt.MouseButton.LeftButton or self._session.busy:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        if ev.isStart() or self._start is None:
            self._session.commit()
            self._start = (self.mapToView(ev.buttonDownPos()).x(), self._session.state.time_shift)
        start_x, start_shift = self._start
        # dragging to the right moves the trajectory to later times
        time_shift = start_shift - (self.mapToView(ev.pos()).x() - start_x)
        self._session.set_state(self._session.state.with_time_shift(time_shift), "Time shift", coalesce=True)
        if ev.isFinish():
            self._start = None
            self._session.commit()


class OverTimeView(DiagnosticView):
    """Deviation components and the speed of both trajectories over time."""

    def __init__(self, session: AlignmentSession, parent=None) -> None:
        super().__init__(
            session,
            "Deviation: "
            + " · ".join(_colored(name, color) for name, color in COMPONENT_COLORS.items())
            + f" · {_colored('standstill', STANDSTILL_COLOR)}. Drag the speed curve to shift in time.",
            parent,
            details="The deviation at equal time (trajectory minus reference at the same timestamp), split into the "
            "direction of travel of the reference (along-track), to its left (cross-track) and vertical. Gray "
            "areas are standstills. Drag left/right in the speed plot to shift the trajectory in time.",
        )
        self.deviation_plot = _make_plot("deviation [m]")
        self._component_curves = {
            name: self.deviation_plot.plot(pen=pg.mkPen(color, width=LINE_WIDTH), name=name)
            for name, color in COMPONENT_COLORS.items()
        }
        self._regions: list[pg.LinearRegionItem] = []

        self.speed_plot = _make_plot("speed [m/s]", TIME_LABEL, view_box=_TimeShiftViewBox(session))
        preview = session.preview
        self.speed_plot.plot(
            preview.reference_t - self._t0,
            preview.reference_speed,
            pen=pg.mkPen(style.color(0), width=LINE_WIDTH),
            name="reference",
        )
        self._speed_curve = self.speed_plot.plot(pen=pg.mkPen(style.color(1), width=LINE_WIDTH), name="trajectory")
        self.speed_plot.setXLink(self.deviation_plot)
        self.speed_plot.getViewBox().setCursor(QtCore.Qt.CursorShape.SizeHorCursor)

        self.graphics.addItem(self.deviation_plot, row=0, col=0)
        self.graphics.addItem(self.speed_plot, row=1, col=0)
        for plot in (self.deviation_plot, self.speed_plot):
            self._register_time_axis(plot)
        self._deviation_range = self.stable_range(self.deviation_plot, x=False)
        self._speed_range = self.stable_range(self.speed_plot, x=False)
        session.state_changed.connect(self._refresh_speed)
        self.fit_view()

    def fit_view(self, only_if_hidden: bool = False) -> None:
        """Shows the time spans of both trajectories.

        Args:
            only_if_hidden (bool, optional): Only if the trajectory is mostly out of view.
        """
        preview = self.session.preview
        t = preview.timestamps(self.session.state) - self._t0
        if only_if_hidden:
            shown_low, shown_high = self.deviation_plot.getViewBox().viewRange()[0]
            overlap = min(shown_high, t[-1]) - max(shown_low, t[0])
            if overlap > 0.5 * min(t[-1] - t[0], shown_high - shown_low):
                return
        low = min(float(t[0]), float(preview.reference_t[0] - self._t0))
        high = max(float(t[-1]), float(preview.reference_t[-1] - self._t0))
        self.deviation_plot.setXRange(low, max(high, low + 1e-6), padding=0.02)

    def _refresh_speed(self) -> None:
        if self.isVisible():
            preview, state = self.session.preview, self.session.state
            speed = state.scale * preview.speed
            self._speed_curve.setData(preview.timestamps(state) - self._t0, speed)
            self._speed_range.update([], np.r_[speed, preview.reference_speed, 0.0])

    def refresh(self) -> None:
        self._refresh_speed()
        residuals = self.residuals
        if residuals is None:
            for curve in self._component_curves.values():
                curve.setData([], [])
            self._show_standstills([])
            self.deviation_plot.setTitle(NO_DATA, size="9pt")
            return
        t = residuals.t - self._t0
        for name, values in zip(COMPONENT_COLORS, (residuals.along, residuals.cross, residuals.vertical), strict=True):
            self._component_curves[name].setData(t, values)
        self._deviation_range.update([], np.r_[residuals.along, residuals.cross, residuals.vertical, 0.0])
        self._show_standstills([(t[start], t[stop - 1]) for start, stop in _runs(residuals.standstill)])
        self.deviation_plot.setTitle(
            "RMS "
            + ", ".join(
                _colored(f"{name} {Residuals.rms_of(values):.3f} m", color)
                for (name, color), values in zip(
                    COMPONENT_COLORS.items(), (residuals.along, residuals.cross, residuals.vertical), strict=True
                )
            ),
            size="9pt",
        )

    def _show_standstills(self, spans: list[tuple[float, float]]) -> None:
        brush = style.brush(STANDSTILL_COLOR, alpha=50)
        while len(self._regions) < len(spans):
            region = pg.LinearRegionItem(movable=False, brush=brush, pen=pg.mkPen(None))
            region.setZValue(-10)
            self.deviation_plot.addItem(region, ignoreBounds=True)
            self._regions.append(region)
        for number, region in enumerate(self._regions):
            region.setVisible(number < len(spans))
            if number < len(spans):
                region.setRegion(spans[number])


# -------------------------------------------------------------------------------------------- time shift


class TimeShiftView(DiagnosticView):
    """Along-track deviation against the speed: a time shift makes them proportional."""

    def __init__(self, session: AlignmentSession, parent=None) -> None:
        super().__init__(
            session,
            "<b>Time shift</b> → along-track deviation proportional to the speed (slope = time offset)",
            parent,
            signature_of=("time_shift",),
            details="A trajectory that is ahead or behind in time deviates along the track in proportion to the "
            "speed: the slope of the line is the time offset. At standstills, a time shift has no effect. An "
            "offset at zero speed is not a time effect (e.g. a leverarm along the direction of travel).",
        )
        self.add_button(
            "Correct time shift",
            "Sets the time shift for which the deviation is smallest, determined together with all other "
            "parameters (using the epochs in motion).",
            lambda: session.correct(["time_shift"], "Correct time shift", epoch_mask="moving"),
        )
        self.plot = _make_plot("along-track deviation [m]", "speed of the reference [m/s]")
        self._moving = _scatter(self.plot, COMPONENT_COLORS["along-track"], "in motion")
        self._still = _scatter(self.plot, STANDSTILL_COLOR, "standstill")
        self._line = self.plot.plot(pen=pg.mkPen(FIT_COLOR, width=LINE_WIDTH), name="fitted line")
        self.graphics.addItem(self.plot)
        self._range = self.stable_range(self.plot)

    def refresh(self) -> None:
        residuals = self.residuals
        if residuals is None:
            for item in (self._moving, self._still):
                item.setData([], [])
            self._line.setData([], [])
            self.plot.setTitle(NO_DATA, size="9pt")
            return
        moving = ~residuals.standstill
        self._moving.setData(residuals.speed[moving], residuals.along[moving])
        self._still.setData(residuals.speed[~moving], residuals.along[~moving])
        self._register(self.plot, residuals.speed, residuals.along, residuals.t)
        self._range.update(np.r_[residuals.speed, 0.0], np.r_[residuals.along, 0.0])
        line = diagnostics.line_fit(residuals.speed[moving], residuals.along[moving])
        if line is None:
            self._line.setData([], [])
            self.plot.setTitle("The speed does not vary: the time shift cannot be seen here.", size="9pt")
            return
        slope, intercept = line
        speeds = np.array([0.0, float(np.max(residuals.speed))])
        self._line.setData(speeds, intercept + slope * speeds)
        # the trajectory is ahead (positive slope) if it reaches each position earlier than the reference
        direction = "ahead of" if slope > 0 else "behind"
        self.plot.setTitle(
            f"Slope {slope * 1000:+.1f} ms: the trajectory is about {abs(slope) * 1000:.1f} ms {direction} the "
            f"reference · offset at zero speed {intercept:+.3f} m",
            size="9pt",
        )


# ------------------------------------------------------------------------------------------- standstills


class StandstillView(DiagnosticView):
    """Deviations at standstills, where a time shift has no effect: shift and leverarm remain."""

    def __init__(self, session: AlignmentSession, parent=None) -> None:
        has_orientation = session.has_orientation
        explanation = (
            "At standstills: <b>leverarm</b> → constant in the body frame · <b>shift</b> → varies with the heading"
            if has_orientation
            else "At standstills: a <b>shift</b> is constant"
        )
        details = (
            "At standstills, a time shift has no effect. In the body frame, a leverarm error is constant (flat "
            "lines), while a shift in the world frame changes with the heading (waves). Standstills at different "
            "headings are needed to tell them apart."
            if has_orientation
            else "At standstills, a time shift has no effect; what remains is a shift (constant)."
        )
        keys = [*SHIFT_KEYS, "lever_x", "lever_y", "lever_z"] if has_orientation else list(SHIFT_KEYS)
        super().__init__(session, explanation, parent, signature_of=tuple(keys), details=details)
        self.add_button(
            "Correct shift and leverarm" if has_orientation else "Correct shift",
            "Determines the parameters from the standstills only, where a time shift has no effect.",
            lambda: session.correct(
                keys, "Correct shift and leverarm" if has_orientation else "Correct shift", epoch_mask="standstill"
            ),
        )
        self.frame = "body" if has_orientation else "world"
        x_label = "heading of the trajectory [°]" if has_orientation else TIME_LABEL
        self.plot = _make_plot(f"deviation in the {self.frame} frame [m]", x_label)
        self._points_items = [
            _scatter(self.plot, color, f"{self.frame} {axis}", size=5)
            for axis, color in zip("xyz", style.AXIS_COLORS, strict=True)
        ]
        self._means = [
            pg.InfiniteLine(angle=0, pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
            for color in style.AXIS_COLORS
        ]
        for line in self._means:
            self.plot.addItem(line, ignoreBounds=True)
        self.graphics.addItem(self.plot)
        self._range = self.stable_range(self.plot)

    def refresh(self) -> None:
        residuals = self.residuals
        still = residuals.standstill if residuals is not None else np.zeros(0, dtype=bool)
        if residuals is None or not np.any(still):
            for item in self._points_items:
                item.setData([], [])
            for line in self._means:
                line.setVisible(False)
            self.plot.setTitle(
                NO_DATA if residuals is None else "No standstills found in the time span of the trajectory.",
                size="9pt",
            )
            return
        if residuals.body_deviation is not None and residuals.body_heading is not None:
            x = np.rad2deg(residuals.body_heading[still])
            values = residuals.body_deviation[still]
        else:
            x = residuals.t[still] - self._t0
            values = residuals.deviation[still]
        for axis, (item, line) in enumerate(zip(self._points_items, self._means, strict=True)):
            item.setData(x, values[:, axis])
            line.setValue(float(np.mean(values[:, axis])))
            line.setVisible(True)
        self._register(self.plot, np.tile(x, 3), values.T.ravel(), np.tile(residuals.t[still], 3))
        self._range.update(x, np.r_[values.ravel(), 0.0])
        means = ", ".join(f"{axis} {value:+.3f}" for axis, value in zip("xyz", np.mean(values, axis=0), strict=True))
        title = f"{len(_runs(still))} standstills · mean {means} m"
        if self.frame == "body":
            title += f" · headings span {np.ptp(x):.0f}°"
        self.plot.setTitle(title, size="9pt")


# ---------------------------------------------------------------------------------------------- geometry


class GeometryView(DiagnosticView):
    """Deviations against the position relative to the pivot: rotations and scale grow with the distance."""

    def __init__(self, session: AlignmentSession, parent=None) -> None:
        super().__init__(
            session,
            "<b>Heading</b> → tangential, <b>scale</b> → radial deviation, growing with the distance from the pivot +",
            parent,
            signature_of=("rot_x", "rot_y", "rot_z", "scale"),
            details="About the pivot (+): a heading error deviates tangentially, a scale error radially, both in "
            "proportion to the distance from the pivot. A tilt (roll, pitch) makes the vertical deviation grow "
            "along x or y (right plot).",
        )
        for text, keys, tooltip in (
            ("Correct heading", ["rot_z"], "Rotation about the vertical axis"),
            ("Correct tilt", ["rot_x", "rot_y"], "Rotations about the x and y axes (roll, pitch)"),
            ("Correct scale", ["scale"], "Scale about the pivot"),
        ):
            self.add_button(
                text,
                f"{tooltip}, determined together with all other parameters.",
                lambda _=False, keys=keys, text=text: session.correct(keys, text),
            )
        self.horizontal_plot = _make_plot("deviation [m]", "horizontal distance from the pivot [m]")
        self._tangential = _scatter(self.horizontal_plot, "#1f77b4", "tangential (heading)")
        self._radial = _scatter(self.horizontal_plot, "#ff7f0e", "radial (scale)")
        self._tangential_line = self.horizontal_plot.plot(pen=pg.mkPen("#1f77b4", width=LINE_WIDTH))
        self._radial_line = self.horizontal_plot.plot(pen=pg.mkPen("#ff7f0e", width=LINE_WIDTH))

        self.vertical_plot = _make_plot("vertical deviation [m]", "offset from the pivot [m]")
        self._along_x = _scatter(self.vertical_plot, style.AXIS_COLORS[0], "along x (pitch)")
        self._along_y = _scatter(self.vertical_plot, style.AXIS_COLORS[1], "along y (roll)")
        self._x_line = self.vertical_plot.plot(pen=pg.mkPen(style.AXIS_COLORS[0], width=LINE_WIDTH))
        self._y_line = self.vertical_plot.plot(pen=pg.mkPen(style.AXIS_COLORS[1], width=LINE_WIDTH))

        self.graphics.addItem(self.horizontal_plot, row=0, col=0)
        self.graphics.addItem(self.vertical_plot, row=0, col=1)
        self._horizontal_range = self.stable_range(self.horizontal_plot)
        self._vertical_range = self.stable_range(self.vertical_plot)

    @staticmethod
    def _show_line(item: pg.PlotDataItem, x: np.ndarray, y: np.ndarray) -> float | None:
        line = diagnostics.line_fit(x, y)
        if line is None:
            item.setData([], [])
            return None
        slope, intercept = line
        ends = np.array([float(np.min(x)), float(np.max(x))])
        item.setData(ends, intercept + slope * ends)
        return slope

    def refresh(self) -> None:
        residuals = self.residuals
        items = (self._tangential, self._radial, self._along_x, self._along_y)
        lines = (self._tangential_line, self._radial_line, self._x_line, self._y_line)
        if residuals is None:
            for item in (*items, *lines):
                item.setData([], [])
            for plot in (self.horizontal_plot, self.vertical_plot):
                plot.setTitle(NO_DATA if plot is self.horizontal_plot else " ", size="9pt")
            return
        offset = residuals.reference_xyz - residuals.pivot
        self._tangential.setData(residuals.radius, residuals.tangential)
        self._radial.setData(residuals.radius, residuals.radial)
        self._along_x.setData(offset[:, 0], residuals.vertical)
        self._along_y.setData(offset[:, 1], residuals.vertical)
        self._register(
            self.horizontal_plot,
            np.r_[residuals.radius, residuals.radius],
            np.r_[residuals.tangential, residuals.radial],
            np.r_[residuals.t, residuals.t],
        )
        self._register(
            self.vertical_plot,
            np.r_[offset[:, 0], offset[:, 1]],
            np.r_[residuals.vertical, residuals.vertical],
            np.r_[residuals.t, residuals.t],
        )
        self._horizontal_range.update(np.r_[residuals.radius, 0.0], np.r_[residuals.tangential, residuals.radial, 0.0])
        self._vertical_range.update(np.r_[offset[:, 0], offset[:, 1]], np.r_[residuals.vertical, 0.0])

        heading = self._show_line(self._tangential_line, residuals.radius, residuals.tangential)
        scale = self._show_line(self._radial_line, residuals.radius, residuals.radial)
        slope_x = self._show_line(self._x_line, offset[:, 0], residuals.vertical)
        slope_y = self._show_line(self._y_line, offset[:, 1], residuals.vertical)
        findings = []
        if heading is not None:
            findings.append(f"rotated by {np.rad2deg(heading):+.4f}° (counter-clockwise)")
        if scale is not None:
            findings.append(f"scale {scale * 1e6:+.0f} ppm")
        self.horizontal_plot.setTitle(" · ".join(findings), size="9pt")
        tilts = [
            f"{name} {np.rad2deg(np.arctan(slope)):+.4f}°"
            for name, slope in (("slope along x", slope_x), ("along y", slope_y))
            if slope is not None
        ]
        self.vertical_plot.setTitle(" · ".join(tilts), size="9pt")


# ------------------------------------------------------------------------------------------- coordinates


class CoordinatesView(DiagnosticView):
    """The coordinates of both trajectories over time."""

    def __init__(self, session: AlignmentSession, parent=None) -> None:
        super().__init__(session, "Coordinates of both trajectories over time", parent)
        preview = session.preview
        self._curves = []
        self._coordinate_ranges = []
        first_plot = None
        for i, label in enumerate(("x [m]", "y [m]", "z [m]")):
            plot = _make_plot(label, TIME_LABEL if i == 2 else "")
            if first_plot is None:
                first_plot = plot
            else:
                plot.setXLink(first_plot)
                plot.legend.setVisible(False)
            plot.plot(
                preview.reference_t - self._t0,
                preview.reference_xyz[:, i],
                pen=pg.mkPen(style.color(0), width=LINE_WIDTH),
                name="reference",
            )
            self._curves.append(plot.plot(pen=pg.mkPen(style.color(1), width=LINE_WIDTH), name="trajectory"))
            self.graphics.addItem(plot, row=i, col=0)
            self._register_time_axis(plot)
            self._coordinate_ranges.append(self.stable_range(plot, x=i == 0))
        session.state_changed.connect(self.refresh)

    def refresh(self) -> None:
        if not self.isVisible():
            return
        preview, state = self.session.preview, self.session.state
        t = preview.timestamps(state) - self._t0
        xyz = preview.positions(state)
        for i, (curve, stable) in enumerate(zip(self._curves, self._coordinate_ranges, strict=True)):
            curve.setData(t, xyz[:, i])
            stable.update(np.r_[t, preview.reference_t - self._t0], np.r_[xyz[:, i], preview.reference_xyz[:, i]])
