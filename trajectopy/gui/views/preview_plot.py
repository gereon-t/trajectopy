"""Top-view preview of the selected trajectories and ATE results in the details panel."""

import logging
from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg

from trajectopy.core.trajectory import Trajectory
from trajectopy.results.ate_result import ATEResult
from trajectopy.visualization import style
from trajectopy.visualization.plots import _color_range, _value_brushes

logger = logging.getLogger(__name__)

MAX_PREVIEW_POINTS = 5000
COLORBAR_MAX_STD = 3.0  # same default as PlotSettings.colorbar_max_std


class PreviewPlot(pg.PlotWidget):
    """Shows the x/y positions of trajectories in a common local frame."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.plotItem.setAspectLocked(True)
        self.plotItem.setLabel("bottom", "x [m]")
        self.plotItem.setLabel("left", "y [m]")
        self.plotItem.setMenuEnabled(False)
        # room for the outermost tick labels, which extend beyond the plot area (left, top, right, bottom)
        self.plotItem.layout.setContentsMargins(4, 10, 18, 4)
        self._legend = self.plotItem.addLegend(offset=(-5, 5), labelTextSize="8pt")
        # created once and only shown/hidden: re-adding it to the layout makes the plot flicker while switching results
        self._colorbar = pg.ColorBarItem(colorMap=style.VALUE_COLORMAP, label="ATE [m]", interactive=False, width=12)
        self.plotItem.layout.addItem(self._colorbar, 2, 3)
        self._colorbar.hide()
        self.apply_theme()
        self.clear_preview()

    def apply_theme(self) -> None:
        theme = style.theme()
        self.setBackground(theme.background)
        for axis in (self.plotItem.getAxis("bottom"), self.plotItem.getAxis("left"), self._colorbar.axis):
            axis.setPen(theme.foreground)
            axis.setTextPen(theme.foreground)
        self.plotItem.showGrid(x=True, y=True, alpha=theme.grid_alpha)

    def dispose(self) -> None:
        """Detaches the legend; its anchor otherwise reacts to the view box after Qt deleted it on shutdown."""
        if (scene := self._legend.scene()) is not None:
            scene.removeItem(self._legend)
        self.plotItem.legend = None

    def clear_preview(self) -> None:
        self.plotItem.clear()
        self._colorbar.hide()
        self.plotItem.hide()

    def show_trajectories(self, trajectories: Sequence[Trajectory]) -> None:
        self.plotItem.clear()
        self._colorbar.hide()
        self.plotItem.show()
        local_transformer = None
        for index, trajectory in enumerate(trajectories):
            if local_transformer is None:
                local_transformer = trajectory.positions.local_transformer
            xyz = self._local_xyz(trajectory, local_transformer)
            self.plotItem.plot(
                xyz[:, 0],
                xyz[:, 1],
                pen=style.pen(index, 2.5),
                name=trajectory.name if len(trajectories) > 1 else None,
                antialias=True,  # at most MAX_PREVIEW_POINTS points
            )
        self.plotItem.autoRange()

    def show_ate_results(self, ate_results: Sequence[ATEResult]) -> None:
        """Shows the trajectories of the results colored by their position deviations, on a shared color scale."""
        self.plotItem.clear()
        self.plotItem.show()
        ate_results = [ate_result for ate_result in ate_results if len(ate_result.abs_dev.pos_dev)]
        if not ate_results:
            self._colorbar.hide()
            return

        samples = [_sample_indices(len(ate_result.trajectory)) for ate_result in ate_results]
        values = [ate_result.pos_dev_comb[indices] for ate_result, indices in zip(ate_results, samples)]
        lower, upper = _color_range(np.concatenate(values), COLORBAR_MAX_STD)

        local_transformer = ate_results[0].trajectory.positions.local_transformer
        for ate_result, indices, ate in zip(ate_results, samples, values):
            xyz = self._local_xyz(ate_result.trajectory, local_transformer, indices)
            self.plotItem.addItem(
                pg.ScatterPlotItem(x=xyz[:, 0], y=xyz[:, 1], size=4, pen=None, brush=_value_brushes(ate, lower, upper))
            )

        self._colorbar.setLevels(values=(lower, upper))
        self._colorbar.show()
        self.plotItem.autoRange()

    @staticmethod
    def _local_xyz(trajectory: Trajectory, local_transformer, indices: np.ndarray | None = None) -> np.ndarray:
        """Positions (thinned out to the preview resolution) in the local frame shared by all trajectories."""
        positions = trajectory.positions.copy()
        positions.xyz = positions.xyz[_sample_indices(len(positions.xyz)) if indices is None else indices]
        try:
            positions.local_transformer = local_transformer
            return positions.to_local(inplace=False).xyz
        except Exception:
            logger.exception("Preview: could not transform '%s' to a local frame", trajectory.name)
            return positions.xyz


def _sample_indices(num_points: int) -> np.ndarray:
    """Indices of at most MAX_PREVIEW_POINTS evenly spread points."""
    if num_points <= MAX_PREVIEW_POINTS:
        return np.arange(num_points)
    return np.linspace(0, num_points - 1, MAX_PREVIEW_POINTS, dtype=int)
