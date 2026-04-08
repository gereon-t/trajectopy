"""Trajectory playback animation window — 3D with body frame axes."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3D projection
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from trajectopy.core.trajectory import Trajectory

SPEED_OPTIONS = [
    ("0.1×", 0.1),
    ("0.25×", 0.25),
    ("0.5×", 0.5),
    ("1×", 1.0),
    ("2×", 2.0),
    ("5×", 5.0),
    ("10×", 10.0),
]
DEFAULT_SPEED_IDX = 3  # 1×
TIMER_INTERVAL_MS = 50  # ~20 fps
SLIDER_STEPS = 1000

# Body frame axis colors: X = red, Y = green, Z = blue
_AXIS_COLORS = ("#ff4444", "#44cc44", "#5599ff")
_AXIS_LABELS = ("X", "Y", "Z")

# Per-trajectory path colors
_TRAJ_COLORS = (
    "#4fc3f7",
    "#81c784",
    "#ffb74d",
    "#f06292",
    "#ba68c8",
    "#4dd0e1",
    "#fff176",
)


class _TrajData:
    """Pre-computed trajectory data and live plot handles for one trajectory."""

    __slots__ = (
        "name",
        "color",
        "x",
        "y",
        "z",
        "t_rel",
        "rot_mats",
        "duration",
        "n",
        "traversed_line",
        "marker",
        "body_lines",
    )

    def __init__(
        self,
        name: str,
        color: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t_rel: np.ndarray,
        rot_mats: np.ndarray | None,
    ) -> None:
        self.name = name
        self.color = color
        self.x = x
        self.y = y
        self.z = z
        self.t_rel = t_rel
        self.rot_mats = rot_mats  # shape (N, 3, 3) or None
        self.duration = float(t_rel[-1])
        self.n = len(t_rel)
        # Assigned during plot setup
        self.traversed_line = None
        self.marker = None
        self.body_lines: list = []  # one Line3D per body axis

    def idx_for_time(self, sim_time: float) -> int:
        t = min(sim_time, self.duration)
        idx = int(np.searchsorted(self.t_rel, t, side="right")) - 1
        return max(0, min(idx, self.n - 1))


class PlaybackWindow(QtWidgets.QDialog):
    """Non-modal window that plays back one or more trajectories in 3D
    with body frame axes (X/Y/Z) shown at the current pose."""

    def __init__(self, trajectories: list[Trajectory], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Playback — " + ", ".join(t.name for t in trajectories))
        self.resize(800, 620)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._trajs = self._build_traj_data(trajectories)
        self._max_duration = max(td.duration for td in self._trajs)
        self._arrow_scale = self._compute_arrow_scale()

        self._setup_ui()
        self._setup_plot()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

        self._sim_time: float = 0.0
        self._playing: bool = False
        self._slider_dragging: bool = False

    # ------------------------------------------------------------------ data

    @staticmethod
    def _build_traj_data(trajectories: list[Trajectory]) -> list[_TrajData]:
        result = []
        for i, traj in enumerate(trajectories):
            color = _TRAJ_COLORS[i % len(_TRAJ_COLORS)]
            local_xyz = traj.positions.to_local(inplace=False).xyz
            x, y, z = local_xyz[:, 0], local_xyz[:, 1], local_xyz[:, 2]
            ts = traj.timestamps
            t_rel = ts - ts[0]
            rot_mats: np.ndarray | None = None
            if traj.has_orientation:
                rot_mats = np.asarray(traj.rotations.as_matrix())  # (N, 3, 3)
                if rot_mats.ndim == 2:  # single pose edge case
                    rot_mats = rot_mats[np.newaxis]
            result.append(_TrajData(traj.name, color, x, y, z, t_rel, rot_mats))
        return result

    def _compute_arrow_scale(self) -> float:
        all_x = np.concatenate([td.x for td in self._trajs])
        all_y = np.concatenate([td.y for td in self._trajs])
        all_z = np.concatenate([td.z for td in self._trajs])
        diag = float(
            np.sqrt(
                (all_x.max() - all_x.min()) ** 2 + (all_y.max() - all_y.min()) ** 2 + (all_z.max() - all_z.min()) ** 2
            )
        )
        return max(diag * 0.05, 1.0)

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._fig = Figure(figsize=(7, 5), tight_layout=True)
        self._fig.patch.set_facecolor("#1e1e1e")
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas, stretch=1)

        self._time_label = QtWidgets.QLabel(f"0.00 s / {self._max_duration:.2f} s")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._time_label)

        self._slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, SLIDER_STEPS)
        self._slider.setValue(0)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.valueChanged.connect(self._on_slider_moved)
        layout.addWidget(self._slider)

        ctrl = QtWidgets.QHBoxLayout()
        self._play_btn = QtWidgets.QPushButton("▶  Play")
        self._play_btn.setFixedWidth(100)
        self._play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self._play_btn)
        ctrl.addStretch()
        ctrl.addWidget(QtWidgets.QLabel("Speed:"))
        self._speed_combo = QtWidgets.QComboBox()
        for label, _ in SPEED_OPTIONS:
            self._speed_combo.addItem(label)
        self._speed_combo.setCurrentIndex(DEFAULT_SPEED_IDX)
        ctrl.addWidget(self._speed_combo)
        layout.addLayout(ctrl)

    # ------------------------------------------------------------------ plot

    def _setup_plot(self) -> None:
        self._ax: Axes3D = self._fig.add_subplot(111, projection="3d")
        self._ax.set_facecolor("#2d2d2d")
        self._ax.tick_params(colors="#aaaaaa", labelsize=7)
        self._ax.xaxis.label.set_color("#aaaaaa")
        self._ax.yaxis.label.set_color("#aaaaaa")
        self._ax.zaxis.label.set_color("#aaaaaa")
        self._ax.set_xlabel("East [m]", fontsize=8)
        self._ax.set_ylabel("North [m]", fontsize=8)
        self._ax.set_zlabel("Up [m]", fontsize=8)

        for td in self._trajs:
            # Full path (faded background)
            self._ax.plot3D(td.x, td.y, td.z, color="#444444", linewidth=0.8)

            # Traversed portion (updated each frame)
            (line,) = self._ax.plot3D([], [], [], color=td.color, linewidth=1.5, label=td.name)
            td.traversed_line = line

            # Current position marker
            (marker,) = self._ax.plot3D([td.x[0]], [td.y[0]], [td.z[0]], "o", color=td.color, markersize=7)
            td.marker = marker

            # Body frame axes — three Line3D segments, one per axis
            if td.rot_mats is not None:
                p = np.array([td.x[0], td.y[0], td.z[0]])
                R = td.rot_mats[0]
                for i, ax_color in enumerate(_AXIS_COLORS):
                    end = p + R[:, i] * self._arrow_scale
                    (bline,) = self._ax.plot3D(
                        [p[0], end[0]],
                        [p[1], end[1]],
                        [p[2], end[2]],
                        color=ax_color,
                        linewidth=2,
                    )
                    td.body_lines.append(bline)

        self._set_equal_axes()

        if len(self._trajs) > 1:
            self._ax.legend(fontsize=7, facecolor="#2d2d2d", labelcolor="#aaaaaa", loc="upper left")

        self._canvas.draw()

    def _set_equal_axes(self) -> None:
        all_x = np.concatenate([td.x for td in self._trajs])
        all_y = np.concatenate([td.y for td in self._trajs])
        all_z = np.concatenate([td.z for td in self._trajs])
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min(), 1.0) / 2.0
        mid_x = (all_x.max() + all_x.min()) / 2
        mid_y = (all_y.max() + all_y.min()) / 2
        mid_z = (all_z.max() + all_z.min()) / 2
        self._ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self._ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self._ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # ------------------------------------------------------------------ helpers

    @property
    def _speed(self) -> float:
        return SPEED_OPTIONS[self._speed_combo.currentIndex()][1]

    def _update_frame(self, sim_time: float) -> None:
        for td in self._trajs:
            idx = td.idx_for_time(sim_time)
            p = np.array([td.x[idx], td.y[idx], td.z[idx]])

            td.traversed_line.set_data_3d(td.x[: idx + 1], td.y[: idx + 1], td.z[: idx + 1])
            td.marker.set_data_3d([p[0]], [p[1]], [p[2]])

            if td.rot_mats is not None and td.body_lines:
                R = td.rot_mats[idx]
                for i, bline in enumerate(td.body_lines):
                    end = p + R[:, i] * self._arrow_scale
                    bline.set_data_3d([p[0], end[0]], [p[1], end[1]], [p[2], end[2]])

        self._canvas.draw_idle()

    def _update_ui(self, sim_time: float) -> None:
        self._time_label.setText(f"{sim_time:.2f} s / {self._max_duration:.2f} s")
        self._slider.blockSignals(True)
        self._slider.setValue(int(sim_time / self._max_duration * SLIDER_STEPS))
        self._slider.blockSignals(False)

    # ------------------------------------------------------------------ slots

    def _on_tick(self) -> None:
        self._sim_time = min(self._sim_time + TIMER_INTERVAL_MS / 1000.0 * self._speed, self._max_duration)
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
        self._play_btn.setText("⏸  Pause")
        self._timer.start()

    def _pause(self) -> None:
        self._playing = False
        self._play_btn.setText("▶  Play")
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
