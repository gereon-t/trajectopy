"""Main view of the interactive alignment: both trajectories in one projection, manipulated directly.

- drag the trajectory to move it, Ctrl+drag to rotate and Shift+drag to scale it about the pivot
- double-click to place the pivot
- with the point pair tool: click a pose of the trajectory and the reference position it belongs to
"""

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from trajectopy.gui.alignment.session import AlignmentSession
from trajectopy.processing.lib.alignment.interactive import AlignmentState, axis_rotation
from trajectopy.visualization import style

LINE_WIDTH = 2.5
GRAB_DISTANCE = 12.0  # px: how close to the trajectory a drag must start to move it
SNAP_DISTANCE = 25.0  # px: how close to a curve a click must be to pick a point of it
MAX_DEVIATION_LINES = 400
AUTO_MAGNIFICATION = -1.0
AUTO_ARROW_FRACTION = 0.04  # of the visible area: length of the typical deviation arrow
AUTO_ARROW_BAND = (0.3, 3.0)  # the automatic magnification is kept while the typical arrow stays within this
HINT_DETAILS = (
    "Drag the trajectory to move it; Ctrl+drag rotates and Shift+drag scales it about the pivot (+). Rotations turn "
    "about the viewing direction. Double-click places the pivot. Drag elsewhere to pan, use the wheel to zoom. "
    "Keys (after clicking into the view): arrows move, Q/E rotate; Shift: 10× larger, Ctrl: 10× finer steps."
)
PIVOT_COLOR = "#e377c2"
PAIR_COLOR = "#2ca02c"
DEVIATION_COLOR = "#d62728"
ROTATION_STEP = np.deg2rad(0.5)
MIN_SCALE_RADIUS = 1e-9

TOOL_MOVE = "move"
TOOL_PAIRS = "pairs"


@dataclass(frozen=True)
class Projection:
    """A view of the scene along a coordinate axis."""

    name: str
    axes: tuple[int, int]  # coordinates shown on the horizontal and the vertical axis

    @property
    def normal(self) -> np.ndarray:
        """Axis that counter-clockwise rotations in this view turn about."""
        unit = np.eye(3)
        return np.cross(unit[self.axes[0]], unit[self.axes[1]])

    def project(self, xyz: np.ndarray) -> np.ndarray:
        return np.asarray(xyz)[..., list(self.axes)]

    def lift(self, delta: np.ndarray) -> np.ndarray:
        """A displacement in the view as a 3D vector."""
        vector = np.zeros(3)
        vector[list(self.axes)] = delta
        return vector


PROJECTIONS = (
    Projection("Top (x, y)", (0, 1)),
    Projection("Side (x, z)", (0, 2)),
    Projection("Front (y, z)", (1, 2)),
)
AXIS_LABELS = ("x [m]", "y [m]", "z [m]")


@dataclass(frozen=True)
class _Pick:
    """A picked point: a pose of the trajectory or a position of the reference (index into the preview)."""

    on_reference: bool
    index: int

    @property
    def name(self) -> str:
        return "reference" if self.on_reference else "trajectory"


def polyline_distance(points: np.ndarray, position: np.ndarray) -> float:
    """Distance of ``position`` to the polyline through ``points`` (n x 2)."""
    if len(points) == 0:
        return float("inf")
    if len(points) == 1:
        return float(np.linalg.norm(points[0] - position))
    start, segment = points[:-1], np.diff(points, axis=0)
    length = np.sum(segment**2, axis=1)
    along = np.sum((position - start) * segment, axis=1) / np.where(length > 0, length, 1.0)
    nearest = start + np.clip(along, 0.0, 1.0)[:, None] * segment
    return float(np.min(np.linalg.norm(nearest - position, axis=1)))


class _ViewBox(pg.ViewBox):
    """Forwards drags, clicks and hovering to the canvas; what the canvas does not use pans/zooms as usual."""

    def __init__(self, canvas: "AlignmentCanvas") -> None:
        super().__init__()
        self._canvas = canvas
        self.setMenuEnabled(False)

    def mouseDragEvent(self, ev, axis=None) -> None:
        if axis is None and ev.button() == QtCore.Qt.MouseButton.LeftButton and self._canvas.drag(ev):
            ev.accept()
            return
        super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self._canvas.click(ev):
            ev.accept()
            return
        super().mouseClickEvent(ev)

    def hoverEvent(self, ev) -> None:
        if not ev.isExit():
            self._canvas.hover(ev.pos())


class AlignmentCanvas(pg.PlotWidget):
    hint_changed = QtCore.Signal(str)
    magnification_changed = QtCore.Signal(float)  # the automatic magnification of the deviation arrows

    def __init__(self, session: AlignmentSession, parent=None) -> None:
        view_box = _ViewBox(self)
        super().__init__(parent=parent, viewBox=view_box)
        self._view_box = view_box
        view_box.disableAutoRange()
        self.session = session
        self.projection = PROJECTIONS[0]
        self.tool = TOOL_MOVE
        self.arrow_magnification = AUTO_MAGNIFICATION
        self._auto_magnification: float | None = None
        self._positions = session.preview.positions(session.state)
        self._drag: tuple[str, AlignmentState, np.ndarray, np.ndarray] | None = None
        self._pending: _Pick | None = None  # first click of a point pair
        self._cursor_time: float | None = None

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=style.theme().grid_alpha)
        self._setup_items()
        session.state_changed.connect(self.refresh)
        self._view_box.sigRangeChanged.connect(self._on_range_changed)
        self.set_projection(0)

    # ---------------------------------------------------------------------------------------------- items

    def _setup_items(self) -> None:
        reference_color, aligned_color = style.color(0), style.color(1)
        foreground = style.theme().foreground
        legend = self.addLegend(offset=(10, 10))
        self._reference_curve = self.plot(pen=pg.mkPen(reference_color, width=LINE_WIDTH), name="reference")
        self._aligned_curve = self.plot(pen=pg.mkPen(aligned_color, width=LINE_WIDTH), name="trajectory")
        self._deviation_lines = pg.PlotDataItem(pen=pg.mkPen(DEVIATION_COLOR, width=1), connect="pairs")
        self.addItem(self._deviation_lines, ignoreBounds=True)
        self._pair_lines = self.plot(pen=pg.mkPen(PAIR_COLOR, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self._pair_lines.opts["connect"] = "pairs"

        def markers(color: str, symbol: str, size: int = 11, filled: bool = True) -> pg.ScatterPlotItem:
            item = pg.ScatterPlotItem(
                symbol=symbol,
                size=size,
                pen=pg.mkPen(color, width=2),
                brush=pg.mkBrush(color) if filled else pg.mkBrush(None),
            )
            self.addItem(item, ignoreBounds=True)
            return item

        self._reference_ends = (markers(reference_color, "o"), markers(reference_color, "s"))
        self._aligned_ends = (markers(aligned_color, "o"), markers(aligned_color, "s"))
        self._pair_sources = markers(PAIR_COLOR, "o", 12, filled=False)
        self._pair_targets = markers(PAIR_COLOR, "x", 12)
        self._pending_marker = markers(PAIR_COLOR, "o", 16, filled=False)
        self._deviation_tips = markers(DEVIATION_COLOR, "o", 4)
        self._pivot_marker = markers(PIVOT_COLOR, "+", 20)
        self._pivot_ring = markers(PIVOT_COLOR, "o", 14, filled=False)
        self._cursor_markers = (markers(foreground, "o", 15, filled=False), markers(foreground, "o", 15, filled=False))
        self._pair_labels: list[pg.TextItem] = []
        for item in (self._pivot_marker, self._pivot_ring, *self._cursor_markers):
            item.setZValue(100)

        for name, symbol in (("start", "o"), ("end", "s")):
            legend.addItem(pg.ScatterPlotItem(symbol=symbol, size=9, pen=foreground, brush=foreground), name)
        legend.addItem(pg.ScatterPlotItem(symbol="+", size=14, pen=pg.mkPen(PIVOT_COLOR, width=2)), "pivot")

    def set_projection(self, index: int) -> None:
        self.projection = PROJECTIONS[index]
        horizontal, vertical = self.projection.axes
        self.setLabel("bottom", AXIS_LABELS[horizontal])
        self.setLabel("left", AXIS_LABELS[vertical])
        reference = self.projection.project(self.session.preview.reference_xyz)
        self._reference_curve.setData(reference[:, 0], reference[:, 1])
        self._reference_ends[0].setData(pos=reference[:1])
        self._reference_ends[1].setData(pos=reference[-1:])
        self.refresh()
        self.fit_view()

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self._pending = None
        self.refresh()
        self._emit_hint()

    def set_arrow_magnification(self, magnification: float) -> None:
        """Draws the deviation at equal time as arrows from the reference, ``magnification`` times longer than
        they are (0: no arrows, negative: automatic). Their pattern shows the cause: parallel arrows a shift,
        arrows along the path a time shift, a swirl a rotation, arrows away from (or towards) the pivot a scale."""
        self.arrow_magnification = magnification
        self._auto_magnification = None
        self._refresh_deviations()

    def _on_range_changed(self, *_) -> None:
        if self.arrow_magnification < 0:  # automatic lengths depend on the zoom
            self._refresh_deviations()

    def set_cursor_time(self, time: float | None) -> None:
        """Marks the poses of both trajectories at ``time`` (of the reference); None hides the marks."""
        self._cursor_time = time
        self._refresh_cursor()

    def refresh(self) -> None:
        session, project = self.session, self.projection.project
        self._positions = session.preview.positions(session.state)
        aligned = project(self._positions)
        self._aligned_curve.setData(aligned[:, 0], aligned[:, 1])
        self._aligned_ends[0].setData(pos=aligned[:1])
        self._aligned_ends[1].setData(pos=aligned[-1:])
        pivot = project(session.pivot)[None, :]
        self._pivot_marker.setData(pos=pivot)
        self._pivot_ring.setData(pos=pivot)
        self._refresh_deviations()
        self._refresh_pairs()
        self._refresh_cursor()

    def _refresh_deviations(self) -> None:
        matches = self.session.preview.time_matches(self.session.state) if self.arrow_magnification else None
        if matches is None:
            self._deviation_lines.setData([], [])
            self._deviation_tips.setData(pos=np.empty((0, 2)))
            return
        _, xyz, reference_xyz = matches
        step = max(len(xyz) // MAX_DEVIATION_LINES, 1)
        start = self.projection.project(reference_xyz[::step])
        arrows = self.projection.project(xyz[::step]) - start
        magnification = self.arrow_magnification
        if magnification < 0:
            magnification = self._automatic_magnification(arrows)
        tip = start + magnification * arrows
        ends = np.empty((2 * len(start), 2))
        ends[0::2], ends[1::2] = start, tip
        self._deviation_lines.setData(ends[:, 0], ends[:, 1])
        self._deviation_tips.setData(pos=tip)

    def _automatic_magnification(self, arrows: np.ndarray) -> float:
        """Magnification for which the typical arrow is a few percent of the visible area. It is kept while the
        arrows change a little (e.g. while dragging), so that they grow and shrink visibly instead of jumping."""
        typical = float(np.percentile(np.linalg.norm(arrows, axis=1), 90))
        (left, right), (bottom, top) = self._view_box.viewRange()
        target = AUTO_ARROW_FRACTION * max(right - left, top - bottom)
        current = self._auto_magnification
        if typical <= 0:
            return current or 1.0
        if current is None or not AUTO_ARROW_BAND[0] * target <= current * typical <= AUTO_ARROW_BAND[1] * target:
            # a round factor (1, 2, 5, 10, 20, ...) that is easy to read
            exact = target / typical
            decade = 10 ** np.floor(np.log10(exact))
            current = float(decade * min((1, 2, 5, 10), key=lambda factor: abs(np.log(factor * decade / exact))))
            self._auto_magnification = current
            self.magnification_changed.emit(current)
        return current

    def _refresh_pairs(self) -> None:
        project, pairs = self.projection.project, self.session.pairs
        sources = project(self._positions[[pair.index for pair in pairs]]) if pairs else np.empty((0, 2))
        targets = project(np.array([pair.target for pair in pairs])) if pairs else np.empty((0, 2))
        self._pair_sources.setData(pos=sources)
        self._pair_targets.setData(pos=targets)
        ends = np.empty((2 * len(pairs), 2))
        ends[0::2], ends[1::2] = sources, targets
        self._pair_lines.setData(ends[:, 0], ends[:, 1])

        while len(self._pair_labels) < len(pairs):
            label = pg.TextItem(color=PAIR_COLOR, anchor=(0, 1))
            self.addItem(label, ignoreBounds=True)
            self._pair_labels.append(label)
        for number, label in enumerate(self._pair_labels):
            label.setVisible(number < len(pairs))
            if number < len(pairs):
                label.setText(str(number + 1))
                label.setPos(*targets[number])

        if self._pending is None:
            self._pending_marker.setData(pos=np.empty((0, 2)))
        else:
            points = self.session.preview.reference_xyz if self._pending.on_reference else self._positions
            self._pending_marker.setData(pos=project(points[self._pending.index])[None, :])

    def _refresh_cursor(self) -> None:
        preview, time = self.session.preview, self._cursor_time
        points: list[np.ndarray | None] = [None, None]
        if time is not None:
            curves = (
                (preview.timestamps(self.session.state), self._positions),
                (preview.reference_t, preview.reference_xyz),
            )
            for number, (t, xyz) in enumerate(curves):
                if t[0] <= time <= t[-1]:
                    points[number] = np.array([np.interp(time, t, xyz[:, i]) for i in range(3)])
        for marker, point in zip(self._cursor_markers, points, strict=True):
            marker.setData(pos=np.empty((0, 2)) if point is None else self.projection.project(point)[None, :])

    def fit_view_if_hidden(self) -> None:
        """Fits the view only if most of the trajectory is out of view, e.g. after a large change; otherwise the
        user's zoom is kept."""
        (left, right), (bottom, top) = self._view_box.viewRange()
        points = self.projection.project(self._positions)
        inside = (points[:, 0] >= left) & (points[:, 0] <= right) & (points[:, 1] >= bottom) & (points[:, 1] <= top)
        if np.mean(inside) < 0.5:
            self.fit_view()

    def fit_view(self) -> None:
        points = np.vstack(
            (self.projection.project(self.session.preview.reference_xyz), self.projection.project(self._positions))
        )
        low, high = np.min(points, axis=0), np.max(points, axis=0)
        margin = 0.05 * np.maximum(np.max(high - low), 1.0)
        self.setRange(
            xRange=(low[0] - margin, high[0] + margin), yRange=(low[1] - margin, high[1] + margin), padding=0
        )

    # ------------------------------------------------------------------------------------------- geometry

    def _pixel_size(self) -> np.ndarray:
        size = self._view_box.viewPixelSize()
        return np.array([size[0] or 1.0, size[1] or 1.0])

    def _view_position(self, item_position) -> np.ndarray:
        point = self._view_box.mapToView(item_position)
        return np.array([point.x(), point.y()])

    def _distance_to_trajectory(self, position: np.ndarray) -> float:
        """Distance (px) of a position in the view to the trajectory."""
        pixel = self._pixel_size()
        return polyline_distance(self.projection.project(self._positions) / pixel, position / pixel)

    def _nearest(self, xyz: np.ndarray, position: np.ndarray) -> tuple[int, float]:
        """Index of the point of ``xyz`` nearest to a position in the view, and its distance (px)."""
        distances = np.linalg.norm((self.projection.project(xyz) - position) / self._pixel_size(), axis=1)
        index = int(np.argmin(distances))
        return index, float(distances[index])

    # ------------------------------------------------------------------------------------------ mouse

    def drag(self, ev) -> bool:
        """Handles a drag with the left button. Returns False if it is not a manipulation (the view pans)."""
        if self.session.busy:
            return False
        position = self._view_position(ev.pos())
        if ev.isStart():
            self._drag = None
            start = self._view_position(ev.buttonDownPos())
            modifiers = ev.modifiers()
            if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                mode = "Rotate"
            elif modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
                mode = "Scale"
            elif self.tool == TOOL_MOVE and self._distance_to_trajectory(start) <= GRAB_DISTANCE:
                mode = "Move"
            else:
                return False
            self.session.commit()
            self._drag = (mode, self.session.state, start, self.session.pivot)
        if self._drag is None:
            return False

        mode, state, start, pivot = self._drag
        center = self.projection.project(pivot)
        if mode == "Move":
            new_state = state.moved(self.projection.lift(position - start))
        elif mode == "Rotate":
            before, after = start - center, position - center
            angle = np.arctan2(after[1], after[0]) - np.arctan2(before[1], before[0])
            new_state = state.rotated_about(pivot, axis_rotation(self.projection.normal, float(angle)))
        else:
            radius = np.linalg.norm(start - center)
            factor = np.linalg.norm(position - center) / radius if radius > MIN_SCALE_RADIUS else 1.0
            new_state = state.scaled_about(pivot, max(float(factor), 1e-6))
        self.session.set_state(new_state, mode, coalesce=True)
        if ev.isFinish():
            self._drag = None
            self.session.commit()
        return True

    def click(self, ev) -> bool:
        if self.session.busy:
            return False
        self.setFocus()
        position = self._view_position(ev.pos())
        if self.tool == TOOL_PAIRS and not ev.double():
            self._pick(position)
            return True
        if ev.double():
            self._place_pivot(position)
            return True
        return False

    def _place_pivot(self, position: np.ndarray) -> None:
        index, distance = self._nearest(self._positions, position)
        if distance <= SNAP_DISTANCE:
            point = self._positions[index]
        else:  # anywhere else: there, at the depth of the current pivot
            point = self.session.pivot.copy()
            point[list(self.projection.axes)] = position
        self.session.set_pivot(point)

    def _pick(self, position: np.ndarray) -> None:
        """Picks a pose of the trajectory or a reference position; two of them (one each) form a point pair."""
        preview = self.session.preview
        index, distance = self._nearest(self._positions, position)
        reference_index, reference_distance = self._nearest(preview.reference_xyz, position)
        if self._pending is None:
            on_reference = reference_distance < distance
        else:  # the other one of the pair
            on_reference = not self._pending.on_reference
        picked = _Pick(on_reference, reference_index if on_reference else index)
        if (reference_distance if on_reference else distance) > SNAP_DISTANCE:
            expected = picked.name if self._pending is not None else "trajectory or the reference"
            self._emit_hint(f"Click closer to the {expected}.")
            return
        if self._pending is None:
            self._pending = picked
            self._refresh_pairs()
            self._emit_hint()
            return
        on_trajectory, on_reference_pick = (picked, self._pending) if on_reference else (self._pending, picked)
        self._pending = None
        self.session.add_pair(on_trajectory.index, preview.reference_xyz[on_reference_pick.index])
        self._emit_hint()

    def cancel_pick(self) -> None:
        if self._pending is not None:
            self._pending = None
            self._refresh_pairs()
            self._emit_hint()

    def hover(self, item_position) -> None:
        if self.tool == TOOL_PAIRS:
            shape = QtCore.Qt.CursorShape.CrossCursor
        elif self._distance_to_trajectory(self._view_position(item_position)) <= GRAB_DISTANCE:
            shape = QtCore.Qt.CursorShape.SizeAllCursor
        else:
            shape = QtCore.Qt.CursorShape.ArrowCursor
        self.setCursor(shape)

    def _emit_hint(self, problem: str = "") -> None:
        self.hint_changed.emit(f"{problem} {self.hint()}".strip())

    def hint(self) -> str:
        if self.tool == TOOL_PAIRS:
            if self._pending is None:
                return "Click a point on the trajectory, then the reference position it belongs to (or vice versa)."
            missing = "trajectory" if self._pending.on_reference else "reference"
            return f"Now click the corresponding point on the {missing}. Esc cancels."
        return "Drag: move · Ctrl+drag: rotate · Shift+drag: scale · Double-click: pivot · Wheel: zoom"

    # ------------------------------------------------------------------------------------------- keyboard

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not self.handle_key(event):
            super().keyPressEvent(event)

    def handle_key(self, event: QtGui.QKeyEvent) -> bool:
        """Keyboard nudging. Returns True if the key was handled."""
        key = QtCore.Qt.Key(event.key())
        if key == QtCore.Qt.Key.Key_Escape and self._pending is not None:
            self.cancel_pick()
            return True
        moves = {
            QtCore.Qt.Key.Key_Left: (-1.0, 0.0),
            QtCore.Qt.Key.Key_Right: (1.0, 0.0),
            QtCore.Qt.Key.Key_Down: (0.0, -1.0),
            QtCore.Qt.Key.Key_Up: (0.0, 1.0),
        }
        turns = {QtCore.Qt.Key.Key_Q: 1.0, QtCore.Qt.Key.Key_E: -1.0}
        if self.session.busy or (key not in moves and key not in turns):
            return False
        factor = 1.0
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            factor *= 10
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            factor /= 10
        state = self.session.state
        if key in moves:
            # a step is ten pixels, so zooming in makes the steps finer
            delta = 10.0 * factor * np.array(moves[key]) * self._pixel_size()
            self.session.set_state(state.moved(self.projection.lift(delta)), "Move", coalesce=True)
        else:
            rotation = axis_rotation(self.projection.normal, turns[key] * factor * ROTATION_STEP)
            self.session.set_state(state.rotated_about(self.session.pivot, rotation), "Rotate", coalesce=True)
        return True
