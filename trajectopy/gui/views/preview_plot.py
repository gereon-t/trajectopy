"""Top-view preview of the selected trajectories in the details panel."""

import logging
from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg

from trajectopy.core.trajectory import Trajectory
from trajectopy.visualization import style

logger = logging.getLogger(__name__)

MAX_PREVIEW_POINTS = 5000


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
        self.apply_theme()
        self.clear_preview()

    def apply_theme(self) -> None:
        theme = style.theme()
        self.setBackground(theme.background)
        for name in ("bottom", "left"):
            axis = self.plotItem.getAxis(name)
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
        self.plotItem.hide()

    def show_trajectories(self, trajectories: Sequence[Trajectory]) -> None:
        self.plotItem.clear()
        self.plotItem.show()
        local_transformer = None
        for index, trajectory in enumerate(trajectories):
            if local_transformer is None:
                local_transformer = trajectory.positions.local_transformer
            xyz = self._local_xyz(trajectory, local_transformer)
            self.plotItem.plot(
                xyz[:, 0],
                xyz[:, 1],
                pen=style.pen(index, 1.5),
                name=trajectory.name if len(trajectories) > 1 else None,
                antialias=True,  # at most MAX_PREVIEW_POINTS points
            )
        self.plotItem.autoRange()

    @staticmethod
    def _local_xyz(trajectory: Trajectory, local_transformer) -> np.ndarray:
        """Positions (thinned out to the preview resolution) in the local frame shared by all trajectories."""
        positions = trajectory.positions.copy()
        if len(positions.xyz) > MAX_PREVIEW_POINTS:
            positions.xyz = positions.xyz[np.linspace(0, len(positions.xyz) - 1, MAX_PREVIEW_POINTS, dtype=int)]
        try:
            positions.local_transformer = local_transformer
            return positions.to_local(inplace=False).xyz
        except Exception:
            logger.exception("Preview: could not transform '%s' to a local frame", trajectory.name)
            return positions.xyz
