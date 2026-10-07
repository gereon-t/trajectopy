"""Interactive manual alignment: sliders for all 11 spatio-temporal parameters with a live preview."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from trajectopy.core.settings import MatchingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.gui.utils import center_window
from trajectopy.processing import alignment, matching
from trajectopy.processing.lib.alignment.parameters import AlignmentParameters

logger = logging.getLogger(__name__)

SLIDER_STEPS = 2000
MIN_PREVIEW_POINTS = 200
MAX_PREVIEW_POINTS = 20000
DEFAULT_PREVIEW_POINTS = 2000
PREVIEW_SLIDER_STEPS = 100
UPDATE_DELAY_MS = 30
RESAMPLE_DELAY_MS = 250


@dataclass(frozen=True)
class ParameterSpec:
    """Describes how one alignment parameter is presented to the user."""

    attribute: str
    label: str
    unit: str
    decimals: int
    step: float
    to_display: float = 1.0  # display value = internal value * to_display


@dataclass
class ParameterGroup:
    title: str
    specs: tuple[ParameterSpec, ...]


PARAMETER_GROUPS: tuple[ParameterGroup, ...] = (
    ParameterGroup(
        "Similarity - Translation",
        (
            ParameterSpec("sim_trans_x", "Translation x", "m", 3, 0.01),
            ParameterSpec("sim_trans_y", "Translation y", "m", 3, 0.01),
            ParameterSpec("sim_trans_z", "Translation z", "m", 3, 0.01),
        ),
    ),
    ParameterGroup(
        "Similarity - Rotation",
        (
            ParameterSpec("sim_rot_x", "Rotation x", "°", 4, 0.01, float(np.rad2deg(1.0))),
            ParameterSpec("sim_rot_y", "Rotation y", "°", 4, 0.01, float(np.rad2deg(1.0))),
            ParameterSpec("sim_rot_z", "Rotation z", "°", 4, 0.01, float(np.rad2deg(1.0))),
        ),
    ),
    ParameterGroup("Similarity - Scale", (ParameterSpec("sim_scale", "Scale", "", 6, 0.0001),)),
    ParameterGroup(
        "Leverarm",
        (
            ParameterSpec("lever_x", "Leverarm x", "m", 3, 0.01),
            ParameterSpec("lever_y", "Leverarm y", "m", 3, 0.01),
            ParameterSpec("lever_z", "Leverarm z", "m", 3, 0.01),
        ),
    ),
    ParameterGroup("Temporal", (ParameterSpec("time_shift", "Time shift", "s", 4, 0.001),)),
)

DEFAULT_VALUES = {
    "sim_scale": 1.0,
}


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
    return float(np.max(np.ptp(xyz, axis=0)))


TIME_SERIES_LABELS = ("x [m]", "y [m]", "z [m]", "roll [?]", "pitch [?]", "yaw [?]")
LINE_WIDTH = 3
REFERENCE_PEN = pg.mkPen("#1f77b4", width=LINE_WIDTH)
ALIGNED_PEN = pg.mkPen("#ff7f0e", width=LINE_WIDTH)
ERROR_PEN = pg.mkPen("#d62728", width=LINE_WIDTH)


def _make_plot(title: str | None = None, x_label: str = "", y_label: str = "") -> pg.PlotItem:
    plot = pg.PlotItem()
    if title:
        plot.setTitle(title)
    plot.setLabel("bottom", x_label)
    plot.setLabel("left", y_label)
    plot.showGrid(x=True, y=True, alpha=0.3)
    plot.setClipToView(True)
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
            self._ref_curves.append(plot.plot(pen=REFERENCE_PEN, antialias=True, name="reference"))
            self._curves.append(plot.plot(pen=ALIGNED_PEN, antialias=True, name="aligned"))
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


class ParameterRow(QtWidgets.QWidget):
    """Slider + spin box + reset button for one parameter. The spin box holds the exact value."""

    value_changed = QtCore.Signal()

    def __init__(self, spec: ParameterSpec, half_range: float, parent=None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.default = DEFAULT_VALUES.get(spec.attribute, 0.0) * spec.to_display
        self._half_range = half_range * spec.to_display

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QtWidgets.QLabel(spec.label)
        self.label.setMinimumWidth(85)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(spec.decimals)
        self.spin.setRange(-1e12, 1e12)
        self.spin.setSingleStep(spec.step)
        self.spin.setSuffix(f" {spec.unit}" if spec.unit else "")
        self.spin.setMinimumWidth(110)
        self.spin.setKeyboardTracking(False)
        self.reset_button = QtWidgets.QToolButton()
        self.reset_button.setText("↺")
        self.reset_button.setToolTip("Reset to default")

        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        layout.addWidget(self.reset_button)

        self.set_display_value(self.default)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.reset_button.clicked.connect(lambda: self.set_display_value(self.default, notify=True))

    @property
    def display_value(self) -> float:
        return self.spin.value()

    @property
    def value(self) -> float:
        """Value in the internal unit (m, rad, s, scale)."""
        return self.spin.value() / self.spec.to_display

    def _slider_to_display(self, position: int) -> float:
        return self.default + (position / SLIDER_STEPS * 2.0 - 1.0) * self._half_range

    def _display_to_slider(self, value: float) -> int:
        position = (value - self.default) / self._half_range + 1.0
        return int(round(min(max(position, 0.0), 2.0) / 2.0 * SLIDER_STEPS))

    def set_display_value(self, value: float, notify: bool = False) -> None:
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(value)
        self.slider.setValue(self._display_to_slider(value))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)
        if notify:
            self.value_changed.emit()

    def _on_slider(self, position: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(self._slider_to_display(position))
        self.spin.blockSignals(False)
        self.value_changed.emit()

    def _on_spin(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._display_to_slider(value))
        self.slider.blockSignals(False)
        self.value_changed.emit()


class ManualAlignmentWindow(QtWidgets.QMainWindow):
    """Manually align a trajectory to a reference using sliders for the 7 similarity, 3 leverarm and 1 time parameter.

    The result can be used directly or as prior (start values) for the least squares alignment.
    """

    alignment_accepted = QtCore.Signal(object, bool)  # AlignmentParameters, refine with least squares

    def __init__(
        self,
        trajectory: Trajectory,
        reference: Trajectory,
        matching_settings: MatchingSettings | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle(f"Manual Alignment: {trajectory.name} to {reference.name}")
        self.matching_settings = matching_settings or MatchingSettings()
        self._has_orientation = trajectory.has_orientation

        self._full_trajectory = trajectory
        self._full_reference = reference
        self._moving = _downsample(trajectory, DEFAULT_PREVIEW_POINTS)
        self._reference = _downsample(reference, DEFAULT_PREVIEW_POINTS)
        self._reference_plot_xyz = self._reference.positions.xyz
        self._series_window: TimeSeriesWindow | None = None

        self._matched_reference_t: np.ndarray | None = None
        self._matched_reference_xyz: np.ndarray | None = None
        self._error_ylim = 1.0
        self._match_once()

        self._rows: dict[str, ParameterRow] = {}
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(UPDATE_DELAY_MS)
        self._update_timer.timeout.connect(self._update_preview)
        self._resample_timer = QtCore.QTimer(self)
        self._resample_timer.setSingleShot(True)
        self._resample_timer.setInterval(RESAMPLE_DELAY_MS)
        self._resample_timer.timeout.connect(self._resample)

        self._setup_ui(self._slider_ranges(trajectory, reference))
        self.resize(1300, 760)
        center_window(self)
        self._update_preview()
        self._fit_view()

    @staticmethod
    def _slider_ranges(trajectory: Trajectory, reference: Trajectory) -> dict[str, float]:
        xyz, ref_xyz = trajectory.positions.xyz, reference.positions.xyz
        extent = max(_extent(xyz), _extent(ref_xyz), 1.0)
        offset = float(np.linalg.norm(np.mean(ref_xyz, axis=0) - np.mean(xyz, axis=0)))
        duration = max(float(trajectory.timestamps[-1] - trajectory.timestamps[0]), 1.0)

        trans_range = offset + extent
        ranges = {
            "sim_rot_x": np.pi,
            "sim_rot_y": np.pi,
            "sim_rot_z": np.pi,
            "sim_scale": 0.5,
            "time_shift": max(10.0, 0.25 * duration),
        }
        ranges.update({f"sim_trans_{axis}": trans_range for axis in "xyz"})
        ranges.update({f"lever_{axis}": max(5.0, 0.1 * extent) for axis in "xyz"})
        return ranges

    def _setup_ui(self, ranges: dict[str, float]) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        control_panel = QtWidgets.QWidget()
        control_panel.setMinimumWidth(480)
        control_panel.setMaximumWidth(560)
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)

        for group in PARAMETER_GROUPS:
            box = QtWidgets.QGroupBox(group.title)
            box_layout = QtWidgets.QVBoxLayout(box)
            for spec in group.specs:
                row = ParameterRow(spec, half_range=ranges[spec.attribute])
                row.value_changed.connect(self._schedule_update)
                if spec.attribute.startswith("lever_") and not self._has_orientation:
                    row.setEnabled(False)
                    row.setToolTip("The trajectory has no orientations, the leverarm cannot be applied.")
                self._rows[spec.attribute] = row
                box_layout.addWidget(row)
            control_layout.addWidget(box)

        self.metrics_label = QtWidgets.QLabel()
        self.metrics_label.setWordWrap(True)
        control_layout.addWidget(self.metrics_label)

        resolution_layout = QtWidgets.QHBoxLayout()
        resolution_layout.addWidget(QtWidgets.QLabel("Preview points"))
        self._resolution_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._resolution_slider.setRange(0, PREVIEW_SLIDER_STEPS)
        self._resolution_slider.setValue(self._points_to_slider(DEFAULT_PREVIEW_POINTS))
        self._resolution_slider.setToolTip(
            "Number of poses used for the interactive preview (logarithmic). "
            "Fewer points make the preview faster; the final alignment always uses all poses."
        )
        self._resolution_label = QtWidgets.QLabel()
        self._resolution_label.setMinimumWidth(50)
        self._resolution_slider.valueChanged.connect(self._on_resolution_slider)
        resolution_layout.addWidget(self._resolution_slider, 1)
        resolution_layout.addWidget(self._resolution_label)
        control_layout.addLayout(resolution_layout)
        self._on_resolution_slider(self._resolution_slider.value(), schedule=False)
        control_layout.addStretch()

        reset_button = QtWidgets.QPushButton("Reset all")
        fit_button = QtWidgets.QPushButton("Fit view")
        reset_button.clicked.connect(self.reset_all)
        fit_button.clicked.connect(self._fit_view)
        series_button = QtWidgets.QPushButton("Time series...")
        series_button.setToolTip("Open a window with the time series of all DOFs of both trajectories.")
        series_button.clicked.connect(self._show_time_series)
        helper_layout = QtWidgets.QHBoxLayout()
        helper_layout.addWidget(reset_button)
        helper_layout.addWidget(fit_button)
        helper_layout.addWidget(series_button)
        control_layout.addLayout(helper_layout)

        apply_button = QtWidgets.QPushButton("Apply directly")
        apply_button.setToolTip("Align the trajectory with exactly these parameters.")
        prior_button = QtWidgets.QPushButton("Use as prior for least squares")
        prior_button.setToolTip(
            "Start the least squares alignment from these parameters. "
            "Parameters that are not estimated (see trajectory settings) stay fixed at the chosen values."
        )
        cancel_button = QtWidgets.QPushButton("Cancel")
        apply_button.clicked.connect(lambda: self._accept(refine=False))
        prior_button.clicked.connect(lambda: self._accept(refine=True))
        cancel_button.clicked.connect(self.close)
        control_layout.addWidget(apply_button)
        control_layout.addWidget(prior_button)
        control_layout.addWidget(cancel_button)

        plot_panel = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._graphics = pg.GraphicsLayoutWidget()
        plot_layout.addWidget(self._graphics, 1)
        self._setup_plot()

        main_layout.addWidget(control_panel)
        main_layout.addWidget(plot_panel, 1)

    @staticmethod
    def _points_to_slider(points: int) -> int:
        fraction = np.log(points / MIN_PREVIEW_POINTS) / np.log(MAX_PREVIEW_POINTS / MIN_PREVIEW_POINTS)
        return int(round(fraction * PREVIEW_SLIDER_STEPS))

    @staticmethod
    def _slider_to_points(position: int) -> int:
        fraction = position / PREVIEW_SLIDER_STEPS
        return int(round(MIN_PREVIEW_POINTS * (MAX_PREVIEW_POINTS / MIN_PREVIEW_POINTS) ** fraction))

    def _on_resolution_slider(self, position: int, schedule: bool = True) -> None:
        self._resolution_label.setText(str(self._slider_to_points(position)))
        if schedule:
            self._resample_timer.start()

    def _resample(self) -> None:
        points = self._slider_to_points(self._resolution_slider.value())
        self._moving = _downsample(self._full_trajectory, points)
        self._reference = _downsample(self._full_reference, points)
        self._reference_plot_xyz = self._reference.positions.xyz
        self._ref_curve_top.setData(self._reference_plot_xyz[:, 0], self._reference_plot_xyz[:, 1])
        self._ref_curve_side.setData(self._reference_plot_xyz[:, 0], self._reference_plot_xyz[:, 2])
        self._match_once()
        if self._series_window is not None:
            self._series_window.set_reference(self._reference)
        self._update_preview()

    def _setup_plot(self) -> None:
        self._plot_top = _make_plot("Top view", "x [m]", "y [m]")
        self._plot_side = _make_plot("Side view", "x [m]", "z [m]")
        self._plot_error = _make_plot("Position deviation to reference", "matched pose", "3D distance [m]")
        self._plot_top.setAspectLocked(True)
        self._plot_side.setAspectLocked(True)
        self._plot_top.addLegend(offset=(10, 10))

        ref_xyz = self._reference_plot_xyz
        self._ref_curve_top = self._plot_top.plot(
            ref_xyz[:, 0], ref_xyz[:, 1], pen=REFERENCE_PEN, antialias=True, name="reference"
        )
        self._ref_curve_side = self._plot_side.plot(ref_xyz[:, 0], ref_xyz[:, 2], pen=REFERENCE_PEN, antialias=True)
        self._curve_top = self._plot_top.plot(pen=ALIGNED_PEN, antialias=True, name="aligned")
        self._curve_side = self._plot_side.plot(pen=ALIGNED_PEN, antialias=True)
        self._curve_error = self._plot_error.plot(pen=ERROR_PEN, antialias=True)
        self._plot_error.setYRange(0, self._error_ylim, padding=0)

        self._graphics.addItem(self._plot_top, row=0, col=0)
        self._graphics.addItem(self._plot_side, row=0, col=1)
        self._graphics.addItem(self._plot_error, row=1, col=0, colspan=2)
        self._graphics.ci.layout.setRowStretchFactor(0, 3)
        self._graphics.ci.layout.setRowStretchFactor(1, 2)

    @property
    def parameters(self) -> AlignmentParameters:
        params = AlignmentParameters()
        for attribute, row in self._rows.items():
            parameter = getattr(params, attribute)
            parameter.value = row.value
            parameter.enabled = True
        if not self._has_orientation:
            for attribute in ("lever_x", "lever_y", "lever_z"):
                getattr(params, attribute).value = 0.0
        return params

    def reset_all(self) -> None:
        for row in self._rows.values():
            row.set_display_value(row.default)
        self._schedule_update()

    def _schedule_update(self) -> None:
        self._update_timer.start()

    def _aligned_preview(self) -> Trajectory:
        result = alignment.manual_alignment(self._moving, self._reference, self.parameters)
        with _quiet_alignment_logs():
            return alignment.apply_alignment(self._moving, result, inplace=False)

    def _update_preview(self) -> None:
        aligned = self._aligned_preview()
        xyz = aligned.positions.xyz
        self._curve_top.setData(xyz[:, 0], xyz[:, 1])
        self._curve_side.setData(xyz[:, 0], xyz[:, 2])
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

    def _update_deviation(self, aligned: Trajectory) -> None:
        if self._matched_reference_t is None:
            self._curve_error.setData([], [])
            self.metrics_label.setText("Deviation: not available (trajectories could not be matched).")
            return

        # the aligned trajectory carries the time shift in its timestamps, so it is sampled at the
        # fixed reference epochs; this makes the time shift visible in the deviation
        t = aligned.timestamps
        valid = (self._matched_reference_t >= t[0]) & (self._matched_reference_t <= t[-1])
        if not np.any(valid):
            self._curve_error.setData([], [])
            self.metrics_label.setText("Deviation: not available (no temporal overlap).")
            return

        ref_t = self._matched_reference_t[valid]
        aligned_xyz = np.column_stack([np.interp(ref_t, t, aligned.positions.xyz[:, i]) for i in range(3)])
        deviations = np.linalg.norm(aligned_xyz - self._matched_reference_xyz[valid], axis=1)

        self._curve_error.setData(np.flatnonzero(valid), deviations)
        self._plot_error.setXRange(0, max(len(self._matched_reference_t) - 1, 1), padding=0)
        peak = float(np.max(deviations)) * 1.05
        if peak > self._error_ylim or peak < 0.3 * self._error_ylim:
            self._error_ylim = max(peak * 1.2, 1e-6)
            self._plot_error.setYRange(0, self._error_ylim, padding=0)
        rmse = float(np.sqrt(np.mean(deviations**2)))
        self.metrics_label.setText(
            f"Matched poses: {len(deviations)}    RMSE: {rmse:.4f} m    Max: {float(np.max(deviations)):.4f} m"
        )

    def _fit_view(self) -> None:
        aligned_x, aligned_y = self._curve_top.getData()
        _, aligned_z = self._curve_side.getData()
        ref = self._reference_plot_xyz
        if aligned_x is not None and len(aligned_x):
            all_xyz = np.vstack((ref, np.column_stack((aligned_x, aligned_y, aligned_z))))
        else:
            all_xyz = ref
        low, high = np.min(all_xyz, axis=0), np.max(all_xyz, axis=0)
        margin = 0.05 * np.maximum(high - low, 1.0)
        low, high = low - margin, high + margin
        self._plot_top.setRange(xRange=(low[0], high[0]), yRange=(low[1], high[1]), padding=0)
        self._plot_side.setRange(xRange=(low[0], high[0]), yRange=(low[2], high[2]), padding=0)

    def _accept(self, refine: bool) -> None:
        self.alignment_accepted.emit(self.parameters, refine)
        self.close()
