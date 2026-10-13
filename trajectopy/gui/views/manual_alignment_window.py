"""Interactive manual alignment of a trajectory to a reference.

The workflow is "rough, then fine":

1. "Auto align" runs the spatio-temporal least squares alignment, estimating the parameters checked with
   "estimate". Its start values are chosen automatically: the current values, the first pose matched,
   a fit on poses matched in time, or the centers matched - whatever is closest to the reference.
2. If needed, the start can be improved manually: "Match first pose" (independent of the timestamps),
   dragging the trajectory in the three views (top, side, front), jog sliders, typed values, keyboard.
3. "Apply" applies and saves what is shown.

Translation, rotation and scale are presented relative to the trajectory's center (the *pivot*): the user
edits the offset of the center and rotates/scales *about* it. Internally, these are converted to the
similarity transformation ``x' = t + s R x`` of :class:`~trajectopy.core.parameters.AlignmentParameters`
(rotation about the coordinate origin), which would otherwise make rotations extremely sensitive for
georeferenced coordinates.
"""

import copy
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.spatial import KDTree

from trajectopy.core.parameters import AlignmentParameters
from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import AlignmentEstimationSettings, AlignmentSettings, MatchingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.gui.utils import center_window
from trajectopy.processing import alignment, matching
from trajectopy.processing.lib.alignment.direct import direct_helmert_transformation
from trajectopy.results.alignment_result import AlignmentResult
from trajectopy.visualization import style

logger = logging.getLogger(__name__)

PREVIEW_POINTS = 5000  # poses used for the interactive preview; the final alignment uses all poses
UPDATE_DELAY_MS = 30

JOG_TICK_MS = 30
JOG_RANGE = 100
JOG_DECADES = 3.0  # full deflection changes the value 10**JOG_DECADES - 1 steps per second
DEG = float(np.rad2deg(1.0))

_alignment_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TrajectopyAlignment")


@dataclass(frozen=True)
class ParameterSpec:
    """Describes how one parameter of the manual alignment is presented to the user."""

    key: str
    label: str
    unit: str
    decimals: int
    step: float  # spin box step and jog unit, in display units
    to_display: float = 1.0  # display value = internal value * to_display
    default: float = 0.0  # internal unit
    tooltip: str = ""
    estimation: str = ""  # corresponding field of AlignmentEstimationSettings


POSITION_SPECS = (
    ParameterSpec(
        "east",
        "East (x)",
        "m",
        3,
        0.1,
        tooltip="Offset of the trajectory's center along x",
        estimation="translation_x",
    ),
    ParameterSpec(
        "north",
        "North (y)",
        "m",
        3,
        0.1,
        tooltip="Offset of the trajectory's center along y",
        estimation="translation_y",
    ),
    ParameterSpec(
        "up", "Up (z)", "m", 3, 0.1, tooltip="Offset of the trajectory's center along z", estimation="translation_z"
    ),
)
ROTATION_SPECS = (
    ParameterSpec(
        "roll",
        "Roll (x)",
        "°",
        3,
        0.1,
        DEG,
        tooltip="Rotation about the x axis through the center",
        estimation="rotation_x",
    ),
    ParameterSpec(
        "pitch",
        "Pitch (y)",
        "°",
        3,
        0.1,
        DEG,
        tooltip="Rotation about the y axis through the center",
        estimation="rotation_y",
    ),
    ParameterSpec(
        "yaw",
        "Yaw (z)",
        "°",
        3,
        0.1,
        DEG,
        tooltip="Heading: rotation about the z axis through the center",
        estimation="rotation_z",
    ),
)
SCALE_TIME_SPECS = (
    ParameterSpec("scale", "Scale", "", 6, 0.001, default=1.0, tooltip="Scale about the center", estimation="scale"),
    ParameterSpec(
        "time_shift",
        "Time shift",
        "s",
        4,
        0.01,
        tooltip="Shift of the trajectory's timestamps",
        estimation="time_shift",
    ),
)
LEVERARM_SPECS = (
    ParameterSpec("lever_x", "Leverarm x", "m", 3, 0.01, estimation="leverarm_x"),
    ParameterSpec("lever_y", "Leverarm y", "m", 3, 0.01, estimation="leverarm_y"),
    ParameterSpec("lever_z", "Leverarm z", "m", 3, 0.01, estimation="leverarm_z"),
)
ALL_SPECS = POSITION_SPECS + ROTATION_SPECS + SCALE_TIME_SPECS + LEVERARM_SPECS
LEVERARM_KEYS = tuple(spec.key for spec in LEVERARM_SPECS)

TIME_SERIES_LABELS = ("x [m]", "y [m]", "z [m]", "roll [°]", "pitch [°]", "yaw [°]")
LINE_WIDTH = 2.5
ERROR_COLOR = "#d62728"
HANDLE_COLOR = "#e377c2"


@dataclass(frozen=True)
class _View:
    """One of the three projections: plot axes (indices into xyz) and the rotation its rotate handle controls."""

    title: str
    axes: tuple[int, int]
    rotation_key: str
    # rotation angle -> direction of the rotate handle in plot coordinates (and back)
    direction_sign: float


VIEWS = (
    _View("Top view (drag ◆ for yaw)", (0, 1), "yaw", 1.0),
    _View("Side view (drag ◆ for pitch)", (0, 2), "pitch", -1.0),
    _View("Front view (drag ◆ for roll)", (1, 2), "roll", 1.0),
)
AXIS_NAMES = ("x [m]", "y [m]", "z [m]")
OFFSET_KEYS = ("east", "north", "up")


@contextmanager
def _quiet_alignment_logs():
    """Silences the info logs of the alignment module while the preview is recomputed repeatedly."""
    alignment_logger = logging.getLogger(alignment.__name__)
    previous_level = alignment_logger.level
    alignment_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        alignment_logger.setLevel(previous_level)


def _downsample(trajectory: Trajectory, max_points: int) -> Trajectory:
    copied = trajectory.copy()
    if len(copied) > max_points:
        copied.mask(np.arange(0, len(copied), int(np.ceil(len(copied) / max_points))))
    return copied


def _extent(xyz: np.ndarray) -> float:
    return float(np.max(np.ptp(xyz, axis=0))) if len(xyz) else 0.0


def _heading(xyz: np.ndarray, distance: float) -> float | None:
    """Direction of travel (rad, from the x axis) over the first ``distance`` meters."""
    travelled = np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1)
    beyond = np.flatnonzero(travelled >= distance)
    if len(beyond) == 0:
        return None
    delta = xyz[beyond[0], :2] - xyz[0, :2]
    return float(np.arctan2(delta[1], delta[0]))


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return np.asarray(Rotations.from_euler("xyz", np.array([roll, pitch, yaw])).as_matrix())


def _pens() -> tuple[QtGui.QPen, QtGui.QPen, QtGui.QPen]:
    return (
        pg.mkPen(style.color(0), width=LINE_WIDTH),
        pg.mkPen(style.color(1), width=LINE_WIDTH),
        pg.mkPen(ERROR_COLOR, width=LINE_WIDTH),
    )


def _make_plot(title: str | None = None, x_label: str = "", y_label: str = "") -> pg.PlotItem:
    plot = pg.PlotItem()
    if title:
        plot.setTitle(title)
    plot.setLabel("bottom", x_label)
    plot.setLabel("left", y_label)
    plot.showGrid(x=True, y=True, alpha=style.theme().grid_alpha)
    plot.setMenuEnabled(False)
    return plot


def _dofs(trajectory: Trajectory, include_orientation: bool) -> np.ndarray:
    """Returns an (n, 3) or (n, 6) array with position and optionally roll/pitch/yaw in degrees."""
    xyz = trajectory.positions.xyz
    if not include_orientation:
        return xyz
    return np.column_stack((xyz, np.rad2deg(trajectory.rpy)))


class TimeSeriesWindow(QtWidgets.QMainWindow):
    """Shows the time series of all degrees of freedom of the reference and the aligned trajectory."""

    def __init__(self, reference: Trajectory, include_orientation: bool, parent=None) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle("Manual Alignment: Time Series")
        self._include_orientation = include_orientation
        self._t0 = float(reference.timestamps[0])
        count = 6 if include_orientation else 3
        reference_pen, aligned_pen, _ = _pens()

        self._graphics = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self._graphics)
        self.resize(1100, 850)

        self._ref_curves = []
        self._curves = []
        first_plot = None
        for i in range(count):
            plot = _make_plot(
                y_label=TIME_SERIES_LABELS[i], x_label="time since reference start [s]" if i == count - 1 else ""
            )
            if i == 0:
                plot.addLegend(offset=(-10, 10))
                first_plot = plot
            else:
                plot.setXLink(first_plot)
            self._ref_curves.append(plot.plot(pen=reference_pen, antialias=True, name="reference"))
            self._curves.append(plot.plot(pen=aligned_pen, antialias=True, name="aligned"))
            self._graphics.addItem(plot, row=i, col=0)
        self.set_reference(reference)

    def set_reference(self, reference: Trajectory) -> None:
        ref_t = reference.timestamps - self._t0
        ref_dofs = _dofs(reference, self._include_orientation)
        for i, curve in enumerate(self._ref_curves):
            curve.setData(ref_t, ref_dofs[:, i])

    def update_aligned(self, aligned: Trajectory) -> None:
        t = aligned.timestamps - self._t0
        dofs = _dofs(aligned, self._include_orientation)
        for i, curve in enumerate(self._curves):
            curve.setData(t, dofs[:, i])


class JogSlider(QtWidgets.QSlider):
    """Spring-loaded slider: the deflection sets the rate of change; it returns to the center when released.

    The response is exponential: a small deflection changes the value by a fraction of a step per second,
    full deflection by about a thousand steps per second. This covers fine and coarse adjustments without
    a separate step size setting.
    """

    jogged = QtCore.Signal(float)  # change in steps

    def __init__(self, parent=None) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.setRange(-JOG_RANGE, JOG_RANGE)
        self.setValue(0)
        self.setPageStep(0)  # clicking the groove must not jump
        self.setMinimumWidth(110)
        self.setToolTip("Drag left/right and hold: the further, the faster. Snaps back when released.")
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(JOG_TICK_MS)
        self._timer.timeout.connect(self._tick)
        self.sliderPressed.connect(self._start)
        self.sliderReleased.connect(self._stop)

    @staticmethod
    def steps_per_second(deflection: float) -> float:
        return float(np.sign(deflection) * (10 ** (JOG_DECADES * abs(deflection)) - 1.0))

    def _start(self) -> None:
        self._timer.start()

    def _stop(self) -> None:
        self._timer.stop()
        self.setValue(0)

    def _tick(self) -> None:
        if deflection := self.value() / JOG_RANGE:
            self.jogged.emit(self.steps_per_second(deflection) * JOG_TICK_MS / 1000.0)


class ParameterRow(QtWidgets.QWidget):
    """Label, jog slider and exact value of one parameter."""

    edited = QtCore.Signal(str, float)  # key, new value in internal units

    def __init__(self, spec: ParameterSpec, parent=None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._value = spec.default

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.label = QtWidgets.QLabel(spec.label)
        self.label.setMinimumWidth(80)
        self.label.setToolTip(spec.tooltip)
        self.jog = JogSlider()
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(spec.decimals)
        self.spin.setRange(-1e9, 1e9)
        self.spin.setSingleStep(spec.step)
        self.spin.setSuffix(f" {spec.unit}" if spec.unit else "")
        self.spin.setMinimumWidth(120)
        self.spin.setKeyboardTracking(False)
        self.spin.setToolTip("Type a value, or use the arrows / mouse wheel (Ctrl: 10× steps).")

        layout.addWidget(self.label)
        layout.addWidget(self.jog, 1)
        layout.addWidget(self.spin)

        self.estimate = QtWidgets.QCheckBox("estimate")
        self.estimate.setToolTip(
            "Estimate this parameter with “Auto align”. Parameters that are not estimated keep their value."
        )
        layout.addWidget(self.estimate)

        self.set_value(spec.default)
        self.jog.jogged.connect(self.nudge)
        self.spin.valueChanged.connect(self._on_spin)

    @property
    def value(self) -> float:
        """Exact value in the internal unit; the spin box only displays it rounded."""
        return self._value

    def set_value(self, value: float) -> None:
        """Shows ``value`` (internal unit) without emitting ``edited``."""
        self._value = float(value)
        self.spin.blockSignals(True)
        self.spin.setValue(self._value * self.spec.to_display)
        self.spin.blockSignals(False)

    def nudge(self, steps: float) -> None:
        # accumulate on the exact value: small jog increments would otherwise be lost to the display rounding
        self.set_value(self._value + steps * self.spec.step / self.spec.to_display)
        self.edited.emit(self.spec.key, self._value)

    def _on_spin(self, display_value: float) -> None:
        """The user typed a value or used the arrows/mouse wheel of the spin box."""
        self._value = display_value / self.spec.to_display
        self.edited.emit(self.spec.key, self._value)


class _PlotArea(pg.GraphicsLayoutWidget):
    """Plot widget that forwards keyboard nudging to the window."""

    def __init__(self, on_key: Callable[[QtGui.QKeyEvent], bool]) -> None:
        super().__init__()
        self._on_key = on_key
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not self._on_key(event):
            super().keyPressEvent(event)


@dataclass
class _ViewItems:
    """Curves and handles of one projection."""

    view: _View
    plot: pg.PlotItem
    reference_curve: pg.PlotDataItem
    aligned_curve: pg.PlotDataItem
    arm: pg.PlotDataItem
    move_handle: pg.TargetItem
    rotate_handle: pg.TargetItem


class _AlignmentSignals(QtCore.QObject):
    finished = QtCore.Signal(object)  # AlignmentResult or the exception that occurred


class ManualAlignmentWindow(QtWidgets.QMainWindow):
    """Manually align a trajectory to a reference (similarity transformation, leverarm and time shift).

    "Auto align" estimates the checked parameters with the least squares alignment; "Apply" applies the result.
    """

    alignment_accepted = QtCore.Signal(object)  # AlignmentResult

    def __init__(
        self,
        trajectory: Trajectory,
        reference: Trajectory,
        matching_settings: MatchingSettings | None = None,
        alignment_settings: AlignmentSettings | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle(f"Manual Alignment: {trajectory.name} to {reference.name}")
        self.matching_settings = matching_settings or MatchingSettings()
        self.alignment_settings = alignment_settings or AlignmentSettings()
        self._has_orientation = trajectory.has_orientation

        self._full_trajectory = trajectory
        self._full_reference = reference
        # rotations and scale act about this point (the trajectory's center)
        self.pivot = np.mean(trajectory.positions.xyz, axis=0)
        self._extent = max(_extent(trajectory.positions.xyz), _extent(reference.positions.xyz), 1.0)
        self._moving = _downsample(trajectory, PREVIEW_POINTS)
        self._reference = _downsample(reference, PREVIEW_POINTS)
        self._reference_tree = KDTree(self._reference.positions.xyz)
        self._series_window: TimeSeriesWindow | None = None

        self._values = {spec.key: spec.default for spec in ALL_SPECS}
        self._syncing_handles = False
        self._status = ""
        # values shown after the last least squares alignment and its result (kept if applied unchanged)
        self._estimated: tuple[dict[str, float], AlignmentResult] | None = None
        self._alignment_signals: _AlignmentSignals | None = None

        self._matched_reference_t: np.ndarray | None = None
        self._matched_reference_xyz: np.ndarray | None = None
        self._error_ylim = 1.0
        self._match_once()

        self._rows: dict[str, ParameterRow] = {}
        self._views: list[_ViewItems] = []
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(UPDATE_DELAY_MS)
        self._update_timer.timeout.connect(self._update_preview)

        self._setup_ui()
        self._init_estimation()
        self.resize(1500, 820)
        center_window(self)
        self._update_preview()
        self._fit_view()

    # ------------------------------------------------------------------------------------------ parameters

    @property
    def values(self) -> dict[str, float]:
        """Current pivot-relative values (m, rad, s, scale)."""
        return dict(self._values)

    def _similarity(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Translation, rotation matrix and scale of ``x' = t + s R x`` for the current values."""
        v = self._values
        rotation = _rotation_matrix(v["roll"], v["pitch"], v["yaw"])
        offset = np.array([v["east"], v["north"], v["up"]])
        translation = self.pivot + offset - v["scale"] * rotation @ self.pivot
        return translation, rotation, v["scale"]

    def _set_similarity(self, translation: np.ndarray, rotation: np.ndarray, scale: float) -> None:
        """Sets the pivot-relative values that correspond to ``x' = t + s R x``."""
        roll, pitch, yaw = Rotations.from_matrix(rotation).as_euler("xyz")
        offset = translation + scale * rotation @ self.pivot - self.pivot
        self._apply_values(
            {"east": offset[0], "north": offset[1], "up": offset[2], "roll": roll, "pitch": pitch, "yaw": yaw}
            | {"scale": scale}
        )

    @property
    def parameters(self) -> AlignmentParameters:
        translation, _, scale = self._similarity()
        # the rotation is built from exactly these angles (no ambiguous conversion back from the matrix)
        roll, pitch, yaw = self._values["roll"], self._values["pitch"], self._values["yaw"]
        params = AlignmentParameters()
        values = {
            "sim_trans_x": translation[0],
            "sim_trans_y": translation[1],
            "sim_trans_z": translation[2],
            "sim_rot_x": roll,
            "sim_rot_y": pitch,
            "sim_rot_z": yaw,
            "sim_scale": scale,
            "time_shift": self._values["time_shift"],
        }
        for key in LEVERARM_KEYS:
            values[key] = self._values[key] if self._has_orientation else 0.0
        for attribute, value in values.items():
            parameter = getattr(params, attribute)
            parameter.value = float(value)
            parameter.enabled = True
        return params

    # ---------------------------------------------------------------------------------------------- state

    def _apply_values(self, values: dict[str, float]) -> None:
        self._values.update({key: float(value) for key, value in values.items()})
        for key, value in values.items():
            if key in self._rows:
                self._rows[key].set_value(value)
        self._schedule_update()

    def _on_row_edited(self, key: str, value: float) -> None:
        self._values[key] = value
        self._status = ""
        self._schedule_update()

    def reset_all(self) -> None:
        self._status = ""
        self._apply_values({spec.key: spec.default for spec in ALL_SPECS})

    # ------------------------------------------------------------------------------------- auto alignment

    def _compose(self, delta_translation: np.ndarray, delta_rotation: np.ndarray, delta_scale: float = 1.0) -> None:
        """Applies an additional similarity transformation on top of the current one."""
        translation, rotation, scale = self._similarity()
        self._set_similarity(
            delta_translation + delta_scale * delta_rotation @ translation,
            delta_rotation @ rotation,
            delta_scale * scale,
        )

    def center_on_reference(self) -> None:
        """Moves the trajectory so that its center coincides with the reference's center."""
        aligned = self._aligned_preview()
        delta = np.mean(self._reference.positions.xyz, axis=0) - np.mean(aligned.positions.xyz, axis=0)
        self._compose(delta, np.eye(3))

    def match_first_pose(self) -> None:
        """Moves the first position onto the reference's first position and aligns the initial headings.

        Useful if the timestamps cannot be relied on (unknown time offset), as no time matching is needed.
        """
        aligned_xyz = self._aligned_preview().positions.xyz
        reference_xyz = self._reference.positions.xyz
        distance = max(0.02 * self._extent, 2.0)
        heading, reference_heading = _heading(aligned_xyz, distance), _heading(reference_xyz, distance)
        delta_yaw = 0.0 if heading is None or reference_heading is None else reference_heading - heading
        delta_rotation = _rotation_matrix(0.0, 0.0, delta_yaw)
        self._compose(reference_xyz[0] - delta_rotation @ aligned_xyz[0], delta_rotation)

    def best_fit(self, with_scale: bool = False) -> bool:
        """Closed-form fit of the current alignment to the reference using poses matched in time.

        Returns:
            bool: False if the trajectories could not be matched.
        """
        try:
            aligned, matched_reference = matching.match_trajectories(
                trajectory=self._aligned_preview(),
                other=self._reference,
                matching_settings=self.matching_settings,
                inplace=False,
            )
            if len(aligned) < 3:
                raise ValueError("fewer than three matched poses")
            helmert = direct_helmert_transformation(aligned.positions.xyz, matched_reference.positions.xyz)
        except Exception as error:
            logger.debug("Best fit not possible: %s", error)
            return False

        delta_rotation = helmert.rotation_matrix
        delta_scale = helmert.scale.value if with_scale else 1.0
        centroid_from = np.mean(aligned.positions.xyz, axis=0)
        centroid_to = np.mean(matched_reference.positions.xyz, axis=0)
        self._compose(centroid_to - delta_scale * delta_rotation @ centroid_from, delta_rotation, delta_scale)
        return True

    def spatial_distance(self) -> float:
        """RMS distance of the aligned trajectory to the nearest reference positions.

        Unlike the deviation, this does not depend on the timestamps, so it also judges alignments of
        trajectories whose time offset is still unknown.
        """
        distances, _ = self._reference_tree.query(self._aligned_preview().positions.xyz)
        return float(np.sqrt(np.mean(distances**2)))

    def _then_fit(self, first: Callable[[], None], with_scale: bool) -> None:
        first()
        self.best_fit(with_scale)

    def choose_start(self) -> str:
        """Sets the start values for the least squares alignment: tries several placements and keeps the one
        closest to the reference (by distance to the nearest reference position, independent of the timestamps).

        Returns:
            str: Description of the chosen start.
        """
        with_scale = self._rows["scale"].estimate.isChecked()
        strategies: dict[str, Callable[[], object]] = {
            "current values": lambda: None,
            "first pose": self.match_first_pose,
            "fit on matched poses": lambda: self.best_fit(with_scale),
            "first pose, then fit": lambda: self._then_fit(self.match_first_pose, with_scale),
            "centers, then fit": lambda: self._then_fit(self.center_on_reference, with_scale),
        }
        initial = dict(self._values)
        candidates: list[tuple[float, str, dict[str, float]]] = []
        for name, strategy in strategies.items():
            self._apply_values(initial)
            strategy()
            candidates.append((self.spatial_distance(), name, dict(self._values)))
        # among (practically) equally good starts, prefer the simplest one (they are ordered so)
        tolerance = max(1e-9, 1e-6 * self._extent)
        best = min(candidate[0] for candidate in candidates)
        _, name, values = next(candidate for candidate in candidates if candidate[0] <= best + tolerance)
        self._apply_values(values)
        return name

    def auto_align(self) -> None:
        """Spatio-temporal least squares alignment of the checked parameters from automatically chosen start
        values. Runs in the background; the result is shown when it is available."""
        start = self.choose_start()
        if not any(self.estimation_settings.to_dict().values()):
            self._status = f"Placed roughly (start: {start}). Check “estimate” for the parameters to estimate."
            self._update_preview()
            return
        self.run_least_squares(start)

    def run_least_squares(self, start: str = "current values") -> None:
        """Runs the least squares alignment in the background, starting from the current values."""
        self._set_busy(True)
        self._status = "Aligning …"
        self.metrics_label.setText(self._status)
        signals = _AlignmentSignals()
        signals.finished.connect(lambda outcome: self._on_aligned(outcome, start))
        self._alignment_signals = signals  # keep alive until finished
        # read the widgets here, on the GUI thread, not in the worker
        parameters, estimation = self.parameters, self.estimation_settings
        _alignment_executor.submit(lambda: signals.finished.emit(self.estimate(parameters, estimation)))

    @property
    def estimation_settings(self) -> AlignmentEstimationSettings:
        """The parameters the least squares alignment estimates (the checked "estimate" boxes)."""
        # unavailable parameters (leverarm without orientations) are unchecked and disabled
        selected = {row.spec.estimation: row.estimate.isChecked() for row in self._rows.values()}
        return AlignmentEstimationSettings(**selected, sensor_rotation=False)

    def _init_estimation(self) -> None:
        """Preselects the parameters estimated according to the trajectory's alignment settings."""
        configured = self.alignment_settings.estimation_settings
        for row in self._rows.values():
            row.estimate.setChecked(bool(getattr(configured, row.spec.estimation)))
            row.estimate.toggled.connect(self._on_estimation_changed)
            if row.spec.key in LEVERARM_KEYS and not self._has_orientation:
                row.estimate.setChecked(False)
                row.estimate.setEnabled(False)
        self._on_estimation_changed()

    def _on_estimation_changed(self) -> None:
        if any(self._rows[key].estimate.isChecked() for key in LEVERARM_KEYS):
            self._leverarm_box.setChecked(True)  # make estimated parameters visible

    def estimate(
        self, initial_parameters: AlignmentParameters, estimation: AlignmentEstimationSettings | None = None
    ) -> AlignmentResult | Exception:
        """Least squares alignment of the full trajectories. Parameters that are not estimated stay at their
        initial values."""
        settings = copy.deepcopy(self.alignment_settings)
        settings.estimation_settings = estimation if estimation is not None else self.estimation_settings
        try:
            return alignment.estimate_alignment(
                trajectory=self._full_trajectory,
                other=self._full_reference,
                alignment_settings=settings,
                matching_settings=self.matching_settings,
                initial_parameters=initial_parameters,
            )
        except Exception as error:
            logger.exception("Least squares alignment failed")
            return error

    def _on_aligned(self, outcome: AlignmentResult | Exception, start: str = "current values") -> None:
        try:
            self._set_busy(False)
        except RuntimeError:  # the window was closed in the meantime
            return
        if isinstance(outcome, Exception):
            self._status = f"Least squares alignment failed ({outcome}); showing the start values ({start})."
        elif not outcome.converged:
            self._status = f"Least squares alignment did not converge; showing the start values ({start})."
        else:
            self._show_parameters(outcome.position_parameters)
            if any(self._values[key] != 0.0 for key in LEVERARM_KEYS):
                self._leverarm_box.setChecked(True)
            self._estimated = (dict(self._values), outcome)
            self._status = f"Aligned (start: {start}). Press “Apply” to keep the result."
        self._update_preview()

    def _show_parameters(self, parameters: AlignmentParameters) -> None:
        """Shows estimated parameters in the controls (and plots)."""
        helmert = parameters.helmert
        self._set_similarity(helmert.translation, helmert.rotation_matrix, helmert.scale.value)
        values = {"time_shift": parameters.time_shift.value}
        if self._has_orientation:
            values |= {key: getattr(parameters, key).value for key in LEVERARM_KEYS}
        self._apply_values(values)

    def _set_busy(self, busy: bool) -> None:
        self._control_panel.setEnabled(not busy)
        self._toolbar.setEnabled(not busy)
        if busy:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()

    @property
    def alignment_result(self) -> AlignmentResult:
        """What "Apply" applies: the least squares result if the values were not changed since the
        alignment (keeping its statistics), otherwise the manually chosen values."""
        if self._estimated is not None and self._estimated[0] == self._values:
            return self._estimated[1]
        return alignment.manual_alignment(self._full_trajectory, self._full_reference, self.parameters)

    # ------------------------------------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        control_panel = QtWidgets.QWidget()
        self._control_panel = control_panel
        control_panel.setMinimumWidth(510)
        control_panel.setMaximumWidth(600)
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)

        control_layout.addLayout(self._auto_controls())
        control_layout.addWidget(self._parameter_box("Position of the center", POSITION_SPECS))
        control_layout.addWidget(self._parameter_box("Rotation about the center", ROTATION_SPECS))
        control_layout.addWidget(self._parameter_box("Scale and time", SCALE_TIME_SPECS))
        leverarm_box = self._parameter_box("Leverarm", LEVERARM_SPECS)
        leverarm_box.setCheckable(True)
        leverarm_box.setChecked(False)
        leverarm_box.setFlat(True)
        leverarm_box.setToolTip("Check to show the leverarm parameters")
        self._leverarm_box = leverarm_box
        leverarm_box.toggled.connect(self._show_leverarm)
        for row in self._leverarm_rows():
            row.setVisible(False)
        if not self._has_orientation:
            leverarm_box.setEnabled(False)
            leverarm_box.setToolTip("The trajectory has no orientations, a leverarm cannot be applied.")
        control_layout.addWidget(leverarm_box)

        self.metrics_label = QtWidgets.QLabel()
        self.metrics_label.setWordWrap(True)
        control_layout.addWidget(self.metrics_label)
        control_layout.addStretch()
        control_layout.addLayout(self._accept_buttons())

        main_layout.addWidget(control_panel)
        main_layout.addWidget(self._plot_panel(), 1)
        self._setup_toolbar()

    def _auto_controls(self) -> QtWidgets.QHBoxLayout:
        layout = QtWidgets.QHBoxLayout()
        auto_button = QtWidgets.QPushButton("Auto align")
        auto_button.setToolTip(
            "Spatio-temporal least squares alignment of the parameters checked with “estimate”. "
            "Starts from the current values, the first pose, a fit on matched poses or the centers - "
            "whatever is closest to the reference."
        )
        auto_button.setDefault(True)
        first_pose_button = QtWidgets.QPushButton("Match first pose")
        first_pose_button.setToolTip(
            "Puts the first position onto the reference's first position and aligns the initial direction "
            "of travel. Does not use the timestamps, so it also works with an unknown time offset."
        )
        auto_button.clicked.connect(self.auto_align)
        first_pose_button.clicked.connect(self._on_match_first_pose)
        layout.addWidget(auto_button, 1)
        layout.addWidget(first_pose_button, 1)
        return layout

    def _on_match_first_pose(self) -> None:
        self.match_first_pose()
        self._status = "Matched the first pose."

    def _parameter_box(self, title: str, specs: tuple[ParameterSpec, ...]) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(4)
        for spec in specs:
            row = ParameterRow(spec)
            row.edited.connect(self._on_row_edited)
            self._rows[spec.key] = row
            layout.addWidget(row)
        return box

    def _show_leverarm(self, shown: bool) -> None:
        for row in self._leverarm_rows():
            row.setVisible(shown)
        self._leverarm_box.setFlat(not shown)  # no empty frame while collapsed

    def _leverarm_rows(self) -> list[ParameterRow]:
        return [self._rows[key] for key in LEVERARM_KEYS]

    def _accept_buttons(self) -> QtWidgets.QHBoxLayout:
        layout = QtWidgets.QHBoxLayout()
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.setToolTip("Aligns the trajectory with the values shown and saves the alignment.")
        cancel_button = QtWidgets.QPushButton("Cancel")
        apply_button.clicked.connect(self._accept)
        cancel_button.clicked.connect(self.close)
        layout.addWidget(apply_button)
        layout.addWidget(cancel_button)
        return layout

    def _setup_toolbar(self) -> None:
        toolbar = self.addToolBar("Manual alignment")
        toolbar.setMovable(False)
        toolbar.addAction("Reset", self.reset_all).setToolTip("Reset all parameters")
        toolbar.addAction("Fit view", self._fit_view)
        toolbar.addAction("Time series…", self._show_time_series).setToolTip(
            "Open a window with the time series of all DOFs of both trajectories."
        )
        self._toolbar = toolbar

    def _plot_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self._graphics = _PlotArea(self._handle_key)
        layout.addWidget(self._graphics, 1)
        hint = QtWidgets.QLabel(
            "Drag ● to move and ◆ to rotate. After clicking into the plots: arrows and PgUp/PgDn move, "
            "Q/E turn (Shift: 10× faster, Ctrl: 10× finer)."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._setup_plots()
        return panel

    # ---------------------------------------------------------------------------------------------- plots

    def _setup_plots(self) -> None:
        reference_pen, aligned_pen, error_pen = _pens()
        ref_xyz = self._reference.positions.xyz
        first_plot = None
        for column, view in enumerate(VIEWS):
            x_axis, y_axis = view.axes
            plot = _make_plot(view.title, AXIS_NAMES[x_axis], AXIS_NAMES[y_axis])
            plot.setAspectLocked(True)
            if first_plot is None:
                plot.addLegend(offset=(10, 10))
                first_plot = plot
            reference_curve = plot.plot(
                ref_xyz[:, x_axis], ref_xyz[:, y_axis], pen=reference_pen, antialias=True, name="reference"
            )
            aligned_curve = plot.plot(pen=aligned_pen, antialias=True, name="aligned")
            arm, move_handle, rotate_handle = self._make_handles(plot)
            self._views.append(_ViewItems(view, plot, reference_curve, aligned_curve, arm, move_handle, rotate_handle))
            move_handle.sigPositionChanged.connect(lambda _, items=self._views[-1]: self._on_move_handle(items))
            rotate_handle.sigPositionChanged.connect(lambda _, items=self._views[-1]: self._on_rotate_handle(items))
            rotate_handle.sigPositionChangeFinished.connect(self._sync_handles)
            self._graphics.addItem(plot, row=0, col=column)

        self._plot_error = _make_plot("Position deviation to reference", "time since reference start [s]", "[m]")
        self._curve_error = self._plot_error.plot(pen=error_pen, antialias=True)
        self._plot_error.setYRange(0, self._error_ylim, padding=0)
        self._graphics.addItem(self._plot_error, row=1, col=0, colspan=len(VIEWS))
        self._graphics.ci.layout.setRowStretchFactor(0, 3)
        self._graphics.ci.layout.setRowStretchFactor(1, 2)

    def _make_handles(self, plot: pg.PlotItem) -> tuple[pg.PlotDataItem, pg.TargetItem, pg.TargetItem]:
        handle_pen = pg.mkPen(HANDLE_COLOR, width=2)
        options = {
            "pen": handle_pen,
            "brush": pg.mkBrush(HANDLE_COLOR),
            "hoverBrush": pg.mkBrush("#ffffff"),
            "hoverPen": handle_pen,
            "size": 16,
        }
        arm = pg.PlotDataItem(pen=pg.mkPen(HANDLE_COLOR, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        move_handle = pg.TargetItem(symbol="o", **options)
        rotate_handle = pg.TargetItem(symbol="d", **options)
        move_handle.setToolTip("Drag to move the trajectory")
        rotate_handle.setToolTip("Drag around ● to rotate the trajectory")
        plot.addItem(arm, ignoreBounds=True)
        for handle in (move_handle, rotate_handle):
            plot.addItem(handle, ignoreBounds=True)
            handle.setZValue(100)
        return arm, move_handle, rotate_handle

    @property
    def _rotation_radius(self) -> float:
        return 0.2 * self._extent

    @property
    def _pivot_position(self) -> np.ndarray:
        """Where the pivot is drawn: it only moves with the offset (rotation and scale act about it)."""
        return self.pivot + np.array([self._values[key] for key in OFFSET_KEYS])

    def _sync_handles(self) -> None:
        self._syncing_handles = True
        try:
            pivot = self._pivot_position
            for items in self._views:
                x_axis, y_axis = items.view.axes
                angle = items.view.direction_sign * self._values[items.view.rotation_key]
                center = np.array([pivot[x_axis], pivot[y_axis]])
                tip = center + self._rotation_radius * np.array([np.cos(angle), np.sin(angle)])
                items.move_handle.setPos(*center)
                items.rotate_handle.setPos(*tip)
                items.arm.setData([center[0], tip[0]], [center[1], tip[1]])
        finally:
            self._syncing_handles = False

    def _on_move_handle(self, items: _ViewItems) -> None:
        if self._syncing_handles:
            return
        position = items.move_handle.pos()
        x_axis, y_axis = items.view.axes
        self._status = ""
        self._apply_values(
            {
                OFFSET_KEYS[x_axis]: position.x() - self.pivot[x_axis],
                OFFSET_KEYS[y_axis]: position.y() - self.pivot[y_axis],
            }
        )

    def _on_rotate_handle(self, items: _ViewItems) -> None:
        if self._syncing_handles:
            return
        position = items.rotate_handle.pos()
        x_axis, y_axis = items.view.axes
        pivot = self._pivot_position
        angle = np.arctan2(position.y() - pivot[y_axis], position.x() - pivot[x_axis])
        self._status = ""
        self._apply_values({items.view.rotation_key: float(items.view.direction_sign * angle)})

    def _handle_key(self, event: QtGui.QKeyEvent) -> bool:
        """Keyboard nudging in the plots. Returns True if the key was handled."""
        keys = {
            QtCore.Qt.Key.Key_Left: ("east", -1),
            QtCore.Qt.Key.Key_Right: ("east", 1),
            QtCore.Qt.Key.Key_Down: ("north", -1),
            QtCore.Qt.Key.Key_Up: ("north", 1),
            QtCore.Qt.Key.Key_PageDown: ("up", -1),
            QtCore.Qt.Key.Key_PageUp: ("up", 1),
            QtCore.Qt.Key.Key_Q: ("yaw", 1),
            QtCore.Qt.Key.Key_E: ("yaw", -1),
        }
        if (action := keys.get(QtCore.Qt.Key(event.key()))) is None:
            return False
        key, direction = action
        factor = 1.0
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            factor *= 10
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            factor /= 10
        spec = self._rows[key].spec
        self._status = ""
        self._apply_values({key: self._values[key] + direction * spec.step * factor / spec.to_display})
        return True

    # -------------------------------------------------------------------------------------------- preview

    def _schedule_update(self) -> None:
        self._sync_handles()
        self._update_timer.start()

    def _aligned_preview(self) -> Trajectory:
        result = alignment.manual_alignment(self._moving, self._reference, self.parameters)
        with _quiet_alignment_logs():
            return alignment.apply_alignment(self._moving, result, inplace=False)

    def _update_preview(self) -> None:
        aligned = self._aligned_preview()
        xyz = aligned.positions.xyz
        for items in self._views:
            x_axis, y_axis = items.view.axes
            items.aligned_curve.setData(xyz[:, x_axis], xyz[:, y_axis])
        self._sync_handles()
        self._update_deviation(aligned)
        if self._series_window is not None and self._series_window.isVisible():
            self._series_window.update_aligned(aligned)

    def _show_time_series(self) -> None:
        if self._series_window is None:
            self._series_window = TimeSeriesWindow(
                self._reference, include_orientation=self._has_orientation and self._reference.has_orientation
            )
        self._series_window.show()
        self._series_window.raise_()
        self._series_window.update_aligned(self._aligned_preview())

    def closeEvent(self, event) -> None:
        if self._series_window is not None:
            self._series_window.close()
        super().closeEvent(event)

    def _match_once(self) -> None:
        """Matches the unaligned trajectories a single time to select the reference poses used for the deviation."""
        try:
            _, matched_ref = matching.match_trajectories(
                trajectory=self._moving,
                other=self._reference,
                matching_settings=self.matching_settings,
                inplace=False,
            )
            if len(matched_ref) == 0:
                raise ValueError("no matched poses")
            self._matched_reference_t = matched_ref.timestamps.copy()
            self._matched_reference_xyz = matched_ref.positions.xyz.copy()
        except Exception as error:
            logger.debug("Could not match trajectories for preview: %s", error)
            self._matched_reference_t = None
            self._matched_reference_xyz = None

    def _set_metrics(self, text: str, aligned: Trajectory) -> None:
        distance = np.linalg.norm(
            np.mean(aligned.positions.xyz, axis=0) - np.mean(self._reference.positions.xyz, axis=0)
        )
        hint = "  The trajectories are far apart: try “Auto align”." if distance > self._extent else ""
        status = f"\n{self._status}" if self._status else ""
        self.metrics_label.setText(text + hint + status)

    def _update_deviation(self, aligned: Trajectory) -> None:
        if self._matched_reference_t is None or self._matched_reference_xyz is None:
            self._curve_error.setData([], [])
            self._set_metrics("Deviation: not available (trajectories could not be matched).", aligned)
            return

        # the aligned trajectory carries the time shift in its timestamps, so it is sampled at the
        # fixed reference epochs; this makes the time shift visible in the deviation
        t = aligned.timestamps
        valid = (self._matched_reference_t >= t[0]) & (self._matched_reference_t <= t[-1])
        if not np.any(valid):
            self._curve_error.setData([], [])
            self._set_metrics("Deviation: not available (no temporal overlap).", aligned)
            return

        ref_t = self._matched_reference_t[valid]
        aligned_xyz = np.column_stack([np.interp(ref_t, t, aligned.positions.xyz[:, i]) for i in range(3)])
        deviations = np.linalg.norm(aligned_xyz - self._matched_reference_xyz[valid], axis=1)

        t0 = self._matched_reference_t[0]
        self._curve_error.setData(ref_t - t0, deviations)
        self._plot_error.setXRange(0, max(float(self._matched_reference_t[-1] - t0), 1e-6), padding=0)
        peak = float(np.max(deviations)) * 1.05
        if peak > self._error_ylim or peak < 0.3 * self._error_ylim:
            self._error_ylim = max(peak * 1.2, 1e-6)
            self._plot_error.setYRange(0, self._error_ylim, padding=0)
        rmse = float(np.sqrt(np.mean(deviations**2)))
        self._set_metrics(
            f"Matched poses: {len(deviations)}    RMSE: {rmse:.4f} m    Max: {float(np.max(deviations)):.4f} m",
            aligned,
        )

    def _fit_view(self) -> None:
        aligned_xyz = self._aligned_preview().positions.xyz
        all_xyz = np.vstack((self._reference.positions.xyz, aligned_xyz))
        low, high = np.min(all_xyz, axis=0), np.max(all_xyz, axis=0)
        margin = 0.05 * np.maximum(high - low, 1.0)
        low, high = low - margin, high + margin
        for items in self._views:
            x_axis, y_axis = items.view.axes
            items.plot.setRange(xRange=(low[x_axis], high[x_axis]), yRange=(low[y_axis], high[y_axis]), padding=0)

    def _accept(self) -> None:
        self.alignment_accepted.emit(self.alignment_result)
        self.close()
