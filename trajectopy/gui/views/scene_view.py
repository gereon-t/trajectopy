"""Central 3D view of the visible trajectories (OpenGL via pyqtgraph)."""

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.core.trajectory import Trajectory
from trajectopy.gui.models.scalar_fields import ScalarField
from trajectopy.visualization import style
from trajectopy.visualization.plots import _color_range

logger = logging.getLogger(__name__)

MAX_DISPLAY_POINTS = 50_000  # per trajectory; long trajectories are thinned out so that slow systems stay responsive
FRAMES_PER_TRAJECTORY = 150
FRAME_SIZE = 0.015  # length of the orientation axes relative to the size of the scene
COLORBAR_MAX_STD = 3.0  # same default as PlotSettings.colorbar_max_std
NO_VALUE_ALPHA = 0.35  # poses without a value of the scalar field are drawn in a faint foreground color
FIT_MARGIN = 1.1

# sizes in logical pixels; OpenGL works in device pixels, so they are scaled on high-DPI screens
LINE_WIDTH = 2.0
SELECTED_LINE_WIDTH = 4.5
FRAME_LINE_WIDTH = 1.5

# camera directions (elevation, azimuth) in degrees; azimuth -90 looks along +y with x pointing right
VIEW_DIRECTIONS = {
    "top": (90.0, -90.0),
    "front": (0.0, -90.0),
    "side": (0.0, 0.0),
    "iso": (30.0, -135.0),
}


@dataclass(frozen=True)
class SceneTrajectory:
    """A trajectory to draw; ``key`` identifies it across updates (the entry id).

    It is drawn in ``color``, or colored by ``field`` if given.
    """

    key: str
    trajectory: Trajectory
    color: str
    field: ScalarField | None = None


@dataclass
class _SceneItem:
    fingerprint: tuple
    item: gl.GLLinePlotItem
    indices: np.ndarray  # drawn poses
    xyz: np.ndarray  # scene coordinates of the drawn poses


def _fingerprint(trajectory: Trajectory) -> tuple:
    """Cheap identity of the geometry; trajectories may be modified in place."""
    xyz = trajectory.positions.xyz
    if len(xyz) == 0:
        return (id(trajectory), 0)
    return (id(trajectory), len(xyz), trajectory.positions.epsg, xyz[0].tobytes(), xyz[-1].tobytes())


def _sample_indices(num_points: int, max_points: int = MAX_DISPLAY_POINTS) -> np.ndarray:
    if num_points <= max_points:
        return np.arange(num_points)
    return np.unique(np.linspace(0, num_points - 1, max_points).round().astype(int))


def _rgba(color: str) -> tuple[float, float, float, float]:
    return pg.mkColor(color).getRgbF()


def _nice_step(extent: float) -> float:
    """Grid spacing of 1, 2 or 5 times a power of ten with roughly ten cells across ``extent``."""
    raw = max(extent / 10.0, 1e-6)
    magnitude = 10 ** np.floor(np.log10(raw))
    return float(next(factor * magnitude for factor in (1, 2, 5, 10) if factor * magnitude >= raw))


class ColorBarOverlay(QtWidgets.QWidget):
    """Vertical colorbars, one per quantity, drawn side by side on top of the 3D view."""

    BAR_WIDTH = 14
    BAR_HEIGHT = 200
    SPACING = 16
    MARGIN = 4

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._ranges: list[tuple[str, float, float]] = []

    def set_ranges(self, ranges: list[tuple[str, float, float]]) -> None:
        """(label, lower, upper) of each colorbar."""
        self._ranges = ranges
        metrics = self.fontMetrics()
        widths = [self._column_width(metrics, *colorbar) for colorbar in ranges]
        self.resize(sum(widths) + self.SPACING * max(len(widths) - 1, 0) + 2 * self.MARGIN, self._height(metrics))
        self.setVisible(bool(ranges))
        self.update()

    def _ticks(self, lower: float, upper: float) -> list[tuple[float, str]]:
        return [(fraction, f"{lower + fraction * (upper - lower):.3g}") for fraction in np.linspace(0.0, 1.0, 5)]

    def _column_width(self, metrics: QtGui.QFontMetrics, label: str, lower: float, upper: float) -> int:
        tick_width = max(metrics.horizontalAdvance(text) for _, text in self._ticks(lower, upper))
        return max(metrics.horizontalAdvance(label), self.BAR_WIDTH + 7 + tick_width)

    def _height(self, metrics: QtGui.QFontMetrics) -> int:
        return self.MARGIN + metrics.height() + 6 + self.BAR_HEIGHT + metrics.height() // 2 + self.MARGIN

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QColor(style.theme().foreground))
        metrics = painter.fontMetrics()
        stops = list(zip(*style.VALUE_COLORMAP.getStops(mode="qcolor")))

        left = self.MARGIN
        for label, lower, upper in self._ranges:
            bar = QtCore.QRectF(left, self.MARGIN + metrics.height() + 6, self.BAR_WIDTH, self.BAR_HEIGHT)
            gradient = QtGui.QLinearGradient(bar.bottomLeft(), bar.topLeft())
            for position, color in stops:
                gradient.setColorAt(float(position), color)
            painter.fillRect(bar, gradient)
            painter.drawRect(bar)
            painter.drawText(QtCore.QPointF(left, self.MARGIN + metrics.ascent()), label)
            for fraction, text in self._ticks(lower, upper):
                y = bar.bottom() - fraction * bar.height()
                painter.drawLine(QtCore.QPointF(bar.right(), y), QtCore.QPointF(bar.right() + 4, y))
                painter.drawText(QtCore.QPointF(bar.right() + 7, y + metrics.ascent() / 2 - 1), text)
            left += self._column_width(metrics, label, lower, upper) + self.SPACING
        painter.end()


class SceneView(gl.GLViewWidget):
    """Draws trajectories as 3D lines in one shared local frame, each in its color or colored by a scalar field.

    Coordinates are shifted by the center of the first data shown, because OpenGL works with single
    precision and would otherwise lose the decimetres of projected or geocentric coordinates.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setMinimumSize(200, 200)
        self._lines: dict[str, _SceneItem] = {}
        self._selected: frozenset[str] = frozenset()
        self._local_transformer = None
        self._origin: np.ndarray | None = None
        self._fitted = False
        self._auto_fit = False  # refit on resize until the user moves the camera (the first fit may precede layout)
        self._pixel_ratio = 1.0
        self._show_frames = False
        self._frames_source: tuple = ()
        self._trajectories: Sequence[SceneTrajectory] = ()

        self._grid = gl.GLGridItem(glOptions="translucent")
        self._grid.setVisible(False)
        self.addItem(self._grid)
        self._frames = gl.GLLinePlotItem(mode="lines", glOptions="translucent", antialias=True)
        self._frames.setVisible(False)
        self.addItem(self._frames)

        self._colorbars = ColorBarOverlay(self)
        self._colorbars.hide()

        self.apply_theme()
        self.setCameraPosition(distance=100.0, elevation=VIEW_DIRECTIONS["top"][0], azimuth=VIEW_DIRECTIONS["top"][1])

    def apply_theme(self) -> None:
        theme = style.theme()
        self.setBackgroundColor(theme.background)
        grid_color = pg.mkColor(theme.foreground)
        grid_color.setAlphaF(0.18)
        self._grid.setColor(grid_color)
        self._colorbars.update()
        if self._trajectories:
            self._color_lines()  # poses without a value use the foreground color

    # ------------------------------------------------------------------------------------------------- content

    def set_content(self, trajectories: Sequence[SceneTrajectory], selected: Collection[str] = ()) -> None:
        """Shows exactly ``trajectories``; unchanged geometry stays on the GPU."""
        self._selected = frozenset(selected)
        self._trajectories = trajectories
        self._lines = self._sync(self._lines, trajectories)
        self._color_lines()
        self._apply_pixel_sizes()
        self._update_grid()
        self._update_frames()
        if not self._fitted and self._lines:
            self.fit()
            self._fitted = True
            self._auto_fit = True

    def clear_data(self) -> None:
        """Forgets the frame and the camera, so that the next data set gets its own; call when no data is loaded."""
        self.set_content([])
        self._local_transformer = None
        self._origin = None
        self._fitted = False
        self._auto_fit = False

    def set_frames_visible(self, visible: bool) -> None:
        self._show_frames = visible
        self._update_frames()

    def _sync(self, current: dict[str, _SceneItem], wanted: Sequence[SceneTrajectory]) -> dict[str, _SceneItem]:
        items: dict[str, _SceneItem] = {}
        current = dict(current)
        for data in wanted:
            cached = current.pop(data.key, None)
            fingerprint = _fingerprint(data.trajectory)
            if cached is not None and cached.fingerprint == fingerprint:
                items[data.key] = cached
                continue
            if cached is not None:
                self.removeItem(cached.item)
            indices = _sample_indices(len(data.trajectory.positions.xyz))
            if len(indices) == 0 or (xyz := self._scene_xyz(data.trajectory, indices)) is None:
                continue
            item = gl.GLLinePlotItem(pos=xyz, antialias=True, glOptions="translucent")
            self.addItem(item)
            items[data.key] = _SceneItem(fingerprint=fingerprint, item=item, indices=indices, xyz=xyz)
        for stale in current.values():
            self.removeItem(stale.item)
        return items

    def _color_lines(self) -> None:
        """Solid colors, or the colors of the scalar fields; fields of the same quantity share one scale."""
        drawn = [(data, self._lines[data.key]) for data in self._trajectories if data.key in self._lines]
        values: dict[str, list[np.ndarray]] = {}
        for data, item in drawn:
            if data.field is not None:
                values.setdefault(data.field.quantity, []).append(data.field.values[item.indices])
        ranges = {
            quantity: _color_range(np.concatenate(arrays), COLORBAR_MAX_STD) for quantity, arrays in values.items()
        }

        no_value = np.array(_rgba(style.theme().foreground), dtype=np.float32)
        no_value[3] = NO_VALUE_ALPHA
        for data, item in drawn:
            if data.field is None:
                item.item.setData(color=_rgba(data.color))
                continue
            lower, upper = ranges[data.field.quantity]
            field_values = data.field.values[item.indices]
            levels = np.clip((np.nan_to_num(field_values) - lower) / (upper - lower), 0.0, 1.0)
            colors = style.VALUE_COLORMAP.map(levels, mode="float").astype(np.float32)
            colors[~np.isfinite(field_values)] = no_value
            item.item.setData(color=colors)

        self._colorbars.set_ranges([(quantity, *bounds) for quantity, bounds in ranges.items()])
        self._place_colorbars()

    def _scene_xyz(self, trajectory: Trajectory, indices: np.ndarray) -> np.ndarray | None:
        positions = trajectory.positions
        if positions.epsg == 0:
            xyz = positions.xyz[indices]
        else:
            if self._local_transformer is None:
                self._local_transformer = positions.local_transformer
            sampled = positions.copy()
            sampled.xyz = sampled.xyz[indices]
            try:
                sampled.local_transformer = self._local_transformer
                xyz = sampled.to_local(inplace=False).xyz
            except Exception:
                logger.exception("3D view: could not transform '%s' to the local frame", trajectory.name)
                xyz = sampled.xyz

        if self._origin is None:
            self._origin = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        return np.ascontiguousarray(xyz - self._origin, dtype=np.float32)

    def _apply_pixel_sizes(self) -> None:
        for key, item in self._lines.items():
            width = SELECTED_LINE_WIDTH if key in self._selected else LINE_WIDTH
            item.item.setData(width=width * self._pixel_ratio)
        self._frames.setData(width=FRAME_LINE_WIDTH * self._pixel_ratio)

    def paintGL(self) -> None:
        if (ratio := self.devicePixelRatioF()) != self._pixel_ratio:
            self._pixel_ratio = ratio  # first paint or moved to a screen with a different scaling
            self._apply_pixel_sizes()
        super().paintGL()

    def _bounds(self, keys: Collection[str] | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        items = [item for key, item in self._lines.items() if keys is None or key in keys]
        if not items:
            return None
        lower = np.min([item.xyz.min(axis=0) for item in items], axis=0)
        upper = np.max([item.xyz.max(axis=0) for item in items], axis=0)
        return lower, upper

    def _update_grid(self) -> None:
        if (bounds := self._bounds()) is None:
            self._grid.setVisible(False)
            return
        lower, upper = bounds
        step = _nice_step(float(np.max(upper[:2] - lower[:2])))
        # whole cells, aligned to multiples of the spacing and covering the data with one cell of margin
        start = np.floor(lower[:2] / step) * step - step
        end = np.ceil(upper[:2] / step) * step + step
        size = end - start
        self._grid.setSpacing(step, step)
        self._grid.setSize(float(size[0]), float(size[1]))
        self._grid.resetTransform()
        self._grid.translate(float(start[0] + size[0] / 2), float(start[1] + size[1] / 2), float(lower[2]))
        self._grid.setVisible(True)

    def _update_frames(self) -> None:
        """Body frame axes (x red, y green, z blue) at evenly spaced poses of the visible trajectories."""
        with_rotations = [
            (data, self._lines[data.key])
            for data in self._trajectories
            if data.key in self._lines and data.trajectory.rotations is not None
        ]
        bounds = self._bounds()
        if not self._show_frames or not with_rotations or bounds is None:
            self._frames.setVisible(False)
            return

        source = (tuple(item.fingerprint for _, item in with_rotations), tuple(map(tuple, bounds)))
        if source != self._frames_source:
            self._frames_source = source
            length = FRAME_SIZE * max(float(np.linalg.norm(bounds[1] - bounds[0])), 1e-3)
            axis_colors = np.array([_rgba(color) for color in style.AXIS_COLORS], dtype=np.float32)
            segments, colors = [], []
            for data, item in with_rotations:
                picks = _sample_indices(len(item.indices), FRAMES_PER_TRAJECTORY)
                origins = item.xyz[picks]
                matrices = np.asarray(data.trajectory.rotations[item.indices[picks]].as_matrix(), dtype=np.float32)
                for axis in range(3):
                    tips = origins + length * matrices[:, :, axis]
                    segments.append(np.stack([origins, tips], axis=1).reshape(-1, 3))
                    colors.append(np.repeat(axis_colors[axis][np.newaxis], 2 * len(origins), axis=0))
            self._frames.setData(pos=np.concatenate(segments), color=np.concatenate(colors))
        self._frames.setVisible(True)

    # -------------------------------------------------------------------------------------------------- camera

    def fit(self, keys: Collection[str] | None = None) -> None:
        """Centers the camera on everything shown (or on the items ``keys``), keeping the viewing direction."""
        if (bounds := self._bounds(keys)) is None:
            return
        lower, upper = bounds
        center = (lower + upper) / 2
        radius = max(float(np.linalg.norm(upper - lower)) / 2, 1e-3)
        # pyqtgraph's field of view is the horizontal one; the narrower of both directions limits the fit
        half_horizontal = np.deg2rad(self.opts["fov"]) / 2
        aspect = self.height() / max(self.width(), 1)
        half_angle = min(half_horizontal, np.arctan(np.tan(half_horizontal) * aspect))
        distance = FIT_MARGIN * radius / np.sin(half_angle)
        self.setCameraPosition(pos=QtGui.QVector3D(*map(float, center)), distance=distance)

    def view_from(self, direction: str) -> None:
        elevation, azimuth = VIEW_DIRECTIONS[direction]
        self.setCameraPosition(elevation=elevation, azimuth=azimuth)
        self.fit()

    # ---------------------------------------------------------------------------------------------- overlays

    def _place_colorbars(self) -> None:
        self._colorbars.move(self.width() - self._colorbars.width() - 8, 8)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_colorbars()
        if self._auto_fit:
            self.fit()

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self._auto_fit = False
        super().mousePressEvent(ev)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        self._auto_fit = False
        super().wheelEvent(ev)
