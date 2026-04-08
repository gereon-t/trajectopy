"""Trajectory playback animation window - PyVista 3D with body frame axes."""

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from pyvistaqt import QtInteractor

from trajectopy.core.trajectory import Trajectory

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
TIMER_INTERVAL_MS = 50  # ~20 fps
SLIDER_STEPS = 1000
MAX_DISPLAY_POINTS = 10000  # subsample visual paths with more points

# Per-trajectory path colors (PyVista accepts hex strings)
_TRAJ_COLORS = (
    "#4fc3f7",
    "#81c784",
    "#ffb74d",
    "#f06292",
    "#ba68c8",
    "#4dd0e1",
    "#fff176",
)

# Body frame axis colors: X = red, Y = green, Z = blue
_AXIS_COLORS = ("#ff4444", "#44cc44", "#5599ff")


class _TrajData:
    """Pre-computed data and live PyVista actor handles for one trajectory."""

    __slots__ = (
        "name",
        "color",
        "xyz",
        "t_rel",
        "rot_mats",
        "duration",
        "n",
        "display_xyz",
        "display_idx_map",
        "full_path_actor",
        "traversed_mesh",
        "traversed_actor",
        "marker_mesh",
        "marker_actor",
        "arrow_meshes",
        "arrow_actors",
        "_last_trav_idx",
    )

    def __init__(
        self,
        name: str,
        color: str,
        xyz: np.ndarray,
        t_rel: np.ndarray,
        rot_mats: np.ndarray | None,
    ) -> None:
        self.name = name
        self.color = color
        self.xyz = xyz  # (N, 3) full-resolution coords
        self.t_rel = t_rel
        self.rot_mats = rot_mats  # (N, 3, 3) or None
        self.duration = float(t_rel[-1])
        self.n = len(t_rel)
        self.full_path_actor = None
        self.traversed_mesh: pv.PolyData | None = None
        self.traversed_actor = None
        self.marker_mesh: pv.PolyData | None = None
        self.marker_actor = None
        self.arrow_meshes: list[pv.PolyData] = []
        self.arrow_actors: list = []
        self._last_trav_idx: int = -1

        # Subsample for display if too many points
        if self.n > MAX_DISPLAY_POINTS:
            step = self.n / MAX_DISPLAY_POINTS
            indices = np.unique(np.round(np.arange(0, self.n, step)).astype(int))
            if indices[-1] != self.n - 1:
                indices = np.append(indices, self.n - 1)
            self.display_xyz = self.xyz[indices]
            self.display_idx_map = indices
        else:
            self.display_xyz = self.xyz
            self.display_idx_map = np.arange(self.n, dtype=int)

    def display_idx_for_full_idx(self, full_idx: int) -> int:
        """Map a full-resolution index to the nearest display index."""
        pos = int(np.searchsorted(self.display_idx_map, full_idx, side="right"))
        return min(pos, len(self.display_idx_map) - 1)

    def idx_for_time(self, sim_time: float) -> int:
        t = min(sim_time, self.duration)
        idx = int(np.searchsorted(self.t_rel, t, side="right")) - 1
        return max(0, min(idx, self.n - 1))


class PlaybackWindow(QtWidgets.QDialog):
    """Non-modal dialog that plays back one or more trajectories in 3D
    using PyVista (GPU-rendered), with body frame axes at the current pose."""

    def __init__(self, trajectories: list[Trajectory], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Playback - " + ", ".join(t.name for t in trajectories))
        self.resize(900, 680)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._trajs = self._build_traj_data(trajectories)
        self._max_duration = max(td.duration for td in self._trajs)
        self._axes_scale = self._compute_axes_scale()
        self._arrow_templates = self._build_arrow_templates()

        self._setup_ui()
        self._setup_scene()

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
            ts = traj.timestamps
            t_rel = ts - ts[0]
            rot_mats: np.ndarray | None = None
            if traj.has_orientation:
                rot_mats = np.asarray(traj.rotations.as_matrix())
                if rot_mats.ndim == 2:
                    rot_mats = rot_mats[np.newaxis]
            result.append(_TrajData(traj.name, color, local_xyz, t_rel, rot_mats))
        return result

    def _compute_axes_scale(self) -> float:
        all_xyz = np.vstack([td.xyz for td in self._trajs])
        ranges = all_xyz.max(axis=0) - all_xyz.min(axis=0)
        diag = float(np.linalg.norm(ranges))
        return max(diag * 0.05, 1.0)

    def _build_arrow_templates(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Pre-compute arrow mesh point arrays and faces for each body axis (at origin)."""
        templates = []
        for i in range(3):
            direction = [0.0, 0.0, 0.0]
            direction[i] = 1.0
            arrow = pv.Arrow(
                start=(0, 0, 0),
                direction=direction,
                scale=self._axes_scale,
                tip_length=0.2,
                tip_radius=0.07,
                shaft_radius=0.03,
            )
            templates.append((arrow.points.copy(), arrow.faces.copy()))
        return templates

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # PyVista viewport embedded as a Qt widget
        self._plotter = QtInteractor(self, auto_update=False)
        layout.addWidget(self._plotter.interactor, stretch=1)

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
        pl = self._plotter
        pl.set_background("#1e1e1e")

        all_xyz = np.vstack([td.xyz for td in self._trajs])
        center = all_xyz.mean(axis=0)

        for td in self._trajs:
            # Full path - simple polyline (no expensive spline interpolation)
            n_disp = len(td.display_xyz)
            if n_disp > 1:
                path = pv.PolyData(td.display_xyz)
                cells = np.empty(n_disp + 1, dtype=np.int64)
                cells[0] = n_disp
                cells[1:] = np.arange(n_disp, dtype=np.int64)
                path.lines = cells
                td.full_path_actor = pl.add_mesh(
                    path,
                    color="#444444",
                    line_width=1,
                    render_lines_as_tubes=False,
                )

            # Traversed path - pre-allocated polyline, lines updated in-place
            traversed = pv.PolyData(td.display_xyz.copy())
            traversed.lines = np.array([1, 0], dtype=np.int64)  # degenerate single-point
            td.traversed_mesh = traversed
            td.traversed_actor = pl.add_mesh(
                traversed,
                color=td.color,
                line_width=3,
                render_lines_as_tubes=False,
            )

            # Current position marker (screen-space point, zoom-independent)
            marker_mesh = pv.PolyData(td.xyz[0:1].copy())
            td.marker_mesh = marker_mesh
            td.marker_actor = pl.add_points(
                marker_mesh,
                color=td.color,
                point_size=12,
                render_points_as_spheres=True,
            )

            # Body frame arrows - create meshes once, updated in-place
            if td.rot_mats is not None:
                pos = td.xyz[0]
                R = td.rot_mats[0]
                for i, ax_color in enumerate(_AXIS_COLORS):
                    tmpl_pts, tmpl_faces = self._arrow_templates[i]
                    pts = (R @ tmpl_pts.T).T + pos
                    mesh = pv.PolyData(pts, faces=tmpl_faces.copy())
                    actor = pl.add_mesh(mesh, color=ax_color, render=False)
                    td.arrow_meshes.append(mesh)
                    td.arrow_actors.append(actor)

        # Camera
        pl.camera.focal_point = center.tolist()
        ranges = all_xyz.max(axis=0) - all_xyz.min(axis=0)
        pl.camera.position = (
            center + np.array([0, -np.linalg.norm(ranges) * 1.5, np.linalg.norm(ranges) * 0.5])
        ).tolist()
        pl.camera.up = (0, 0, 1)

        # Axis widget in corner
        pl.add_axes(interactive=False)

        pl.render()

    # ------------------------------------------------------------------ helpers

    @property
    def _speed(self) -> float:
        return SPEED_OPTIONS[self._speed_combo.currentIndex()][1]

    def _update_frame(self, sim_time: float) -> None:
        needs_render = False

        for td in self._trajs:
            idx = td.idx_for_time(sim_time)
            pos = td.xyz[idx]

            # Update traversed path line connectivity (no new objects)
            disp_idx = td.display_idx_for_full_idx(idx)
            if disp_idx > 0 and disp_idx != td._last_trav_idx:
                n = disp_idx + 1
                cells = np.empty(n + 1, dtype=np.int64)
                cells[0] = n
                cells[1:] = np.arange(n, dtype=np.int64)
                td.traversed_mesh.lines = cells
                td.traversed_mesh.Modified()
                td._last_trav_idx = disp_idx
                needs_render = True

            # Move the position marker in-place
            td.marker_mesh.points[0] = pos
            td.marker_mesh.Modified()
            needs_render = True

            # Update body frame arrows in-place via template rotation
            if td.rot_mats is not None and td.arrow_meshes:
                R = td.rot_mats[idx]
                for i, mesh in enumerate(td.arrow_meshes):
                    tmpl_pts = self._arrow_templates[i][0]
                    mesh.points[:] = (R @ tmpl_pts.T).T + pos
                    mesh.Modified()
                needs_render = True

        if needs_render:
            self._plotter.render()

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
        self._play_btn.setText("Pause")
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
        self._plotter.close()
        super().closeEvent(event)
