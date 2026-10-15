"""3D preview of the selected trajectories and ATE results in the details panel (OpenGL, like the playback)."""

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtGui, QtWidgets

from trajectopy.core.trajectory import Trajectory
from trajectopy.gui.views.preview_plot import (
    MARKER_SIZE,
    ate_preview_values,
    fit_colorbar_axis,
    local_xyz,
)
from trajectopy.results.ate_result import ATEResult
from trajectopy.visualization import style

LINE_WIDTH = 3.0  # pixels, trajectories
GRID_ALPHA = 60  # 0-255
GRID_LINES = 20  # per side
ELEVATION = 35  # degrees, initial camera
AZIMUTH = -90  # degrees, initial camera (looking north, as the top view)


class PreviewPlot3D(QtWidgets.QWidget):
    """Shows the positions of trajectories in a common local frame in an orbitable 3D view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self._view = gl.GLViewWidget()
        self._view.setToolTip("Drag to rotate, wheel to zoom, middle-drag or Ctrl+drag to pan")
        # legend below the view: child widgets on top of the OpenGL view are not composed reliably on all platforms
        self._legend = QtWidgets.QLabel()
        self._legend.setContentsMargins(6, 2, 6, 2)
        self._legend.setWordWrap(True)
        self._colorbar = pg.ColorBarItem(colorMap=style.VALUE_COLORMAP, interactive=False, width=12)
        self._colorbar.axis.setLabel("ATE [m]")
        self._colorbar_widget = pg.GraphicsLayoutWidget()
        self._colorbar_widget.ci.layout.setContentsMargins(0, 10, 4, 4)
        self._colorbar_widget.addItem(self._colorbar)

        view_layout = QtWidgets.QHBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)
        view_layout.addWidget(self._view, stretch=1)
        view_layout.addWidget(self._colorbar_widget)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(view_layout, stretch=1)
        layout.addWidget(self._legend)

        self._grid: gl.GLGridItem | None = None
        self.apply_theme()
        self.clear_preview()

    def apply_theme(self) -> None:
        theme = style.theme()
        self._view.setBackgroundColor(theme.background)
        self._colorbar_widget.setBackground(theme.background)
        self._colorbar.axis.setPen(theme.foreground)
        self._colorbar.axis.setTextPen(theme.foreground)
        self._legend.setStyleSheet(f"background: {theme.background}; color: {theme.foreground};")
        if self._grid is not None:
            self._grid.setColor(self._grid_color())

    def dispose(self) -> None:
        """Nothing to detach; same interface as the 2D preview."""

    def clear_preview(self) -> None:
        self._clear_items()
        self._colorbar_widget.hide()
        self._legend.hide()

    def show_trajectories(self, trajectories: Sequence[Trajectory]) -> None:
        self._clear_items()
        self._colorbar_widget.hide()
        local_transformer = trajectories[0].positions.local_transformer if trajectories else None
        all_xyz = []
        for index, trajectory in enumerate(trajectories):
            xyz = local_xyz(trajectory, local_transformer)
            all_xyz.append(xyz)
            color = pg.mkColor(style.color(index)).getRgbF()
            self._view.addItem(gl.GLLinePlotItem(pos=xyz, color=color, width=LINE_WIDTH, antialias=True))
        self._show_legend([(style.color(index), t.name) for index, t in enumerate(trajectories)])
        self._frame(all_xyz)

    def show_ate_results(self, ate_results: Sequence[ATEResult]) -> None:
        """Shows the trajectories of the results colored by their position deviations, on a shared color scale."""
        self._clear_items()
        self._legend.hide()
        ate_results = [ate_result for ate_result in ate_results if len(ate_result.abs_dev.pos_dev)]
        if not ate_results:
            self._colorbar_widget.hide()
            return

        samples, values, lower, upper = ate_preview_values(ate_results)
        local_transformer = ate_results[0].trajectory.positions.local_transformer
        all_xyz = []
        for ate_result, indices, ate in zip(ate_results, samples, values):
            xyz = local_xyz(ate_result.trajectory, local_transformer, indices)
            all_xyz.append(xyz)
            levels = np.nan_to_num(np.clip((ate - lower) / (upper - lower), 0.0, 1.0))
            colors = style.VALUE_COLORMAP.map(levels, mode=pg.ColorMap.FLOAT)
            self._view.addItem(gl.GLScatterPlotItem(pos=xyz, color=colors, size=MARKER_SIZE, pxMode=True))

        self._colorbar.setLevels(values=(lower, upper))
        fit_colorbar_axis(self._colorbar, lower, upper)
        self._colorbar_widget.setFixedWidth(int(self._colorbar.axis.width() + self._colorbar.width() + 8))
        self._colorbar_widget.show()
        self._frame(all_xyz)

    def _clear_items(self) -> None:
        for item in list(self._view.items):
            self._view.removeItem(item)
        self._grid = None

    def _show_legend(self, entries: list[tuple[str, str]]) -> None:
        if len(entries) < 2:
            self._legend.hide()
            return
        self._legend.setText(
            "&nbsp;&nbsp; ".join(f"<span style='color:{color}'>&#9632;</span>&nbsp;{name}" for color, name in entries)
        )
        self._legend.show()

    def _grid_color(self) -> QtGui.QColor:
        color = pg.mkColor(style.theme().foreground)
        color.setAlpha(GRID_ALPHA)
        return color

    def _frame(self, all_xyz: list[np.ndarray]) -> None:
        """Adds a ground grid below the trajectories and points the camera at their center."""
        all_xyz = [xyz for xyz in all_xyz if len(xyz)]
        if not all_xyz:
            return
        stacked = np.vstack(all_xyz)
        lower, upper = stacked.min(axis=0), stacked.max(axis=0)
        center = (lower + upper) / 2
        extent = float(np.linalg.norm(upper - lower)) or 1.0

        size = _nice_step(extent * 1.2 / GRID_LINES) * GRID_LINES
        self._grid = gl.GLGridItem(color=self._grid_color())
        self._grid.setSize(size, size)
        self._grid.setSpacing(size / GRID_LINES, size / GRID_LINES)
        self._grid.translate(float(center[0]), float(center[1]), float(lower[2]))
        self._view.addItem(self._grid)

        self._view.setCameraPosition(
            pos=QtGui.QVector3D(*center.tolist()), distance=extent * 1.2, elevation=ELEVATION, azimuth=AZIMUTH
        )


def _nice_step(value: float) -> float:
    """Smallest of 1, 2 and 5 times a power of ten that is at least value."""
    magnitude = 10 ** np.floor(np.log10(value))
    return float(next(step * magnitude for step in (1, 2, 5, 10) if step * magnitude >= value))
