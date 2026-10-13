"""Trajectory playback animation window - OpenGL 3D view (pyqtgraph) with body frame axes."""

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from trajectopy.core.trajectory import Trajectory
from trajectopy.visualization import style

SPEED_OPTIONS = [
    ("0.1x", 0.1),
    ("0.25x", 0.25),
    ("0.5x", 0.5),
    ("1x", 1.0),
    ("2x", 2.0),
    ("5x", 5.0),
    ("10x", 10.0),
]
DEFAULT_SPEED_IDX = 3  # 1x
TIMER_INTERVAL_MS = 33  # ~30 fps
SLIDER_STEPS = 1000
MAX_DISPLAY_POINTS = 20000  # subsample visual paths with more points

PATH_COLOR = (0.35, 0.35, 0.35, 1.0)
MARKER_COLOR = (1.0, 1.0, 1.0, 1.0)
BACKGROUND = "#1e1e1e"


def _reset_gl_program_caches() -> None:
    """Makes pyqtgraph recompile its shader programs for a new OpenGL context.

    pyqtgraph caches the compiled program of each item class, regardless of the context it was compiled
    in. Unless all contexts are shared (``Qt.AA_ShareOpenGLContexts``, set by the GUI), the cached programs
    become invalid once the window that compiled them is closed.
    """
    if QtCore.QCoreApplication.testAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts):
        return
    for item_class in (gl.GLLinePlotItem, gl.GLScatterPlotItem):
        item_class._shaderProgram = None


def _rgba(hex_color: str) -> tuple[float, float, float, float]:
    return pg.mkColor(hex_color).getRgbF()


class _TrajData:
    """Pre-computed data and the OpenGL items of one trajectory."""

    def __init__(self, name: str, color: str, xyz: np.ndarray, t_rel: np.ndarray, rot_mats: np.ndarray | None):
        self.name = name
        self.color = color
        self.xyz = xyz  # (N, 3) full-resolution local coordinates
        self.t_rel = t_rel
        self.rot_mats = rot_mats  # (N, 3, 3) or None
        self.duration = float(t_rel[-1]) if len(t_rel) else 0.0
        self.n = len(t_rel)

        if self.n > MAX_DISPLAY_POINTS:
            indices = np.unique(np.linspace(0, self.n - 1, MAX_DISPLAY_POINTS).round().astype(int))
        else:
            indices = np.arange(self.n, dtype=int)
        self.display_idx_map = indices
        self.display_xyz = self.xyz[indices]

        self.traversed: gl.GLLinePlotItem | None = None
        self.marker: gl.GLScatterPlotItem | None = None
        self.body_axes: gl.GLLinePlotItem | None = None
        self.last_display_idx = -1

    def display_idx_for_full_idx(self, full_idx: int) -> int:
        """Map a full-resolution index to the nearest display index."""
        pos = int(np.searchsorted(self.display_idx_map, full_idx, side="right"))
        return min(pos, len(self.display_idx_map) - 1)

    def idx_for_time(self, sim_time: float) -> int:
        t = min(sim_time, self.duration)
        idx = int(np.searchsorted(self.t_rel, t, side="right")) - 1
        return max(0, min(idx, self.n - 1))


class PlaybackWindow(QtWidgets.QDialog):
    """Non-modal dialog that plays back one or more trajectories in 3D (GPU-rendered via OpenGL),
    with body frame axes at the current pose."""

    def __init__(self, trajectories: list[Trajectory], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Playback - " + ", ".join(t.name for t in trajectories))
        self.resize(900, 680)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._trajs = self._build_traj_data(trajectories)
        self._max_duration = max((td.duration for td in self._trajs), default=0.0) or 1.0
        self._axes_scale = self._compute_axes_scale()

        _reset_gl_program_caches()
        self._setup_ui()
        self._setup_scene()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)
        self._elapsed = QtCore.QElapsedTimer()

        self._sim_time: float = 0.0
        self._playing: bool = False
        self._slider_dragging: bool = False
        self._update_frame(0.0)

    # ------------------------------------------------------------------ data

    @staticmethod
    def _build_traj_data(trajectories: list[Trajectory]) -> list[_TrajData]:
        result = []
        for i, traj in enumerate(trajectories):
            local_xyz = traj.positions.to_local(inplace=False).xyz
            t_rel = traj.timestamps - traj.timestamps[0]
            rot_mats: np.ndarray | None = None
            if traj.rotations is not None:
                rot_mats = np.asarray(traj.rotations.as_matrix())
                if rot_mats.ndim == 2:
                    rot_mats = rot_mats[np.newaxis]
            result.append(_TrajData(traj.name, style.color(i), local_xyz, t_rel, rot_mats))
        return result

    def _compute_axes_scale(self) -> float:
        all_xyz = np.vstack([td.xyz for td in self._trajs])
        diag = float(np.linalg.norm(all_xyz.max(axis=0) - all_xyz.min(axis=0)))
        return max(diag * 0.05, 1.0)

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor(BACKGROUND)
        layout.addWidget(self._view, stretch=1)

        bottom = QtWidgets.QWidget()
        bottom.setContentsMargins(8, 4, 8, 8)
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setSpacing(4)

        self._time_label = QtWidgets.QLabel(f"0.00 s / {self._max_duration:.2f} s")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(self._time_label)

        self._slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, SLIDER_STEPS)
        self._slider.setValue(0)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.valueChanged.connect(self._on_slider_moved)
        bl.addWidget(self._slider)

        ctrl = QtWidgets.QHBoxLayout()
        self._play_btn = QtWidgets.QPushButton("Play")
        self._play_btn.setFixedWidth(100)
        self._play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self._play_btn)
        ctrl.addStretch()
        for td in self._trajs:
            ctrl.addWidget(QtWidgets.QLabel(f"<span style='color:{td.color}'>&#9632;</span> {td.name}"))
        ctrl.addStretch()
        ctrl.addWidget(QtWidgets.QLabel("Speed:"))
        self._speed_combo = QtWidgets.QComboBox()
        for label, _ in SPEED_OPTIONS:
            self._speed_combo.addItem(label)
        self._speed_combo.setCurrentIndex(DEFAULT_SPEED_IDX)
        ctrl.addWidget(self._speed_combo)
        bl.addLayout(ctrl)

        layout.addWidget(bottom)

    # ------------------------------------------------------------------ scene

    def _setup_scene(self) -> None:
        all_xyz = np.vstack([td.xyz for td in self._trajs])
        center = all_xyz.mean(axis=0)
        extent = float(np.linalg.norm(all_xyz.max(axis=0) - all_xyz.min(axis=0))) or 1.0

        grid = gl.GLGridItem(color=(255, 255, 255, 25))
        grid_size = float(np.ceil(extent * 1.2))
        grid.setSize(grid_size, grid_size)
        grid.setSpacing(max(grid_size / 20, 1e-3), max(grid_size / 20, 1e-3))
        grid.translate(center[0], center[1], float(all_xyz[:, 2].min()))
        self._view.addItem(grid)

        for td in self._trajs:
            self._view.addItem(gl.GLLinePlotItem(pos=td.display_xyz, color=PATH_COLOR, width=1.0, antialias=True))

            td.traversed = gl.GLLinePlotItem(pos=td.display_xyz[:1], color=_rgba(td.color), width=3.0, antialias=True)
            self._view.addItem(td.traversed)

            td.marker = gl.GLScatterPlotItem(pos=td.xyz[:1], color=MARKER_COLOR, size=12.0, pxMode=True)
            self._view.addItem(td.marker)

            if td.rot_mats is not None:
                axis_colors = np.repeat([_rgba(c) for c in style.AXIS_COLORS], 2, axis=0)
                td.body_axes = gl.GLLinePlotItem(
                    pos=np.zeros((6, 3)), color=axis_colors, width=3.0, mode="lines", antialias=True
                )
                self._view.addItem(td.body_axes)

        self._view.setCameraPosition(
            pos=QtGui.QVector3D(*center.tolist()), distance=extent * 1.2, elevation=35, azimuth=-90
        )

    # ------------------------------------------------------------------ helpers

    @property
    def _speed(self) -> float:
        return SPEED_OPTIONS[self._speed_combo.currentIndex()][1]

    def _update_frame(self, sim_time: float) -> None:
        for td in self._trajs:
            if td.traversed is None or td.marker is None:
                continue
            idx = td.idx_for_time(sim_time)
            pos = td.xyz[idx]

            display_idx = td.display_idx_for_full_idx(idx)
            if display_idx != td.last_display_idx:
                td.traversed.setData(pos=td.display_xyz[: display_idx + 1])
                td.last_display_idx = display_idx

            td.marker.setData(pos=pos[np.newaxis])

            if td.body_axes is not None and td.rot_mats is not None:
                # the columns of the rotation matrix are the body axes in the local frame
                tips = pos + td.rot_mats[idx].T * self._axes_scale
                segments = np.empty((6, 3))
                segments[0::2] = pos
                segments[1::2] = tips
                td.body_axes.setData(pos=segments)

    def _update_ui(self, sim_time: float) -> None:
        self._time_label.setText(f"{sim_time:.2f} s / {self._max_duration:.2f} s")
        self._slider.blockSignals(True)
        self._slider.setValue(int(sim_time / self._max_duration * SLIDER_STEPS))
        self._slider.blockSignals(False)

    # ------------------------------------------------------------------ slots

    def _on_tick(self) -> None:
        # advance by the actually elapsed time, so the playback speed does not depend on the frame rate
        elapsed = self._elapsed.restart() / 1000.0
        self._sim_time = min(self._sim_time + elapsed * self._speed, self._max_duration)
        self._update_frame(self._sim_time)
        self._update_ui(self._sim_time)
        if self._sim_time >= self._max_duration:
            self._pause()

    def _toggle_play(self) -> None:
        if self._playing:
            self._pause()
        else:
            if self._sim_time >= self._max_duration:
                self._sim_time = 0.0
            self._play()

    def _play(self) -> None:
        self._playing = True
        self._play_btn.setText("Pause")
        self._elapsed.start()
        self._timer.start()

    def _pause(self) -> None:
        self._playing = False
        self._play_btn.setText("Play")
        self._timer.stop()

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True
        self._pause()

    def _on_slider_released(self) -> None:
        self._slider_dragging = False

    def _on_slider_moved(self, value: int) -> None:
        if not self._slider_dragging:
            return
        self._sim_time = value / SLIDER_STEPS * self._max_duration
        self._update_frame(self._sim_time)
        self._time_label.setText(f"{self._sim_time:.2f} s / {self._max_duration:.2f} s")

    def closeEvent(self, event) -> None:
        self._pause()
        super().closeEvent(event)
