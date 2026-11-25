"""Exact values of the placement and the selection of the parameters estimated by least squares."""

from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtWidgets

from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import AlignmentEstimationSettings
from trajectopy.gui.alignment.session import AlignmentSession
from trajectopy.processing._lib.alignment.interactive import AlignmentState

DEG = float(np.rad2deg(1.0))


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    unit: str
    decimals: int
    step: float  # in display units
    estimation: str  # field of AlignmentEstimationSettings
    tooltip: str
    to_display: float = 1.0  # display value = internal value * to_display
    minimum: float = -1e12


SHIFT_KEYS = ("shift_x", "shift_y", "shift_z")
ANGLE_KEYS = ("roll", "pitch", "yaw")
LEVERARM_KEYS = ("lever_x", "lever_y", "lever_z")

GROUPS: tuple[tuple[str, tuple[ParameterSpec, ...]], ...] = (
    (
        "Shift of the pivot",
        tuple(
            ParameterSpec(key, axis, "m", 3, 0.1, f"translation_{axis}", f"How far the pivot is moved along {axis}")
            for key, axis in zip(SHIFT_KEYS, "xyz", strict=True)
        ),
    ),
    (
        "Rotation about the pivot",
        tuple(
            ParameterSpec(key, f"{key} ({axis})", "°", 3, 0.1, f"rotation_{axis}", tooltip, DEG)
            for key, axis, tooltip in (
                ("roll", "x", "Rotation about the x axis"),
                ("pitch", "y", "Rotation about the y axis"),
                ("yaw", "z", "Heading: rotation about the vertical (z) axis"),
            )
        ),
    ),
    (
        "Scale and time",
        (
            ParameterSpec("scale", "scale", "", 6, 0.001, "scale", "Scale about the pivot", minimum=1e-6),
            ParameterSpec(
                "time_shift", "time shift", "s", 4, 0.01, "time_shift", "Subtracted from the trajectory's timestamps"
            ),
        ),
    ),
    (
        "Leverarm (body frame)",
        tuple(
            ParameterSpec(key, axis, "m", 3, 0.01, f"leverarm_{axis}", f"Leverarm along the body {axis} axis")
            for key, axis in zip(LEVERARM_KEYS, "xyz", strict=True)
        ),
    ),
)
SPECS = {spec.key: spec for _, specs in GROUPS for spec in specs}
# keys of the panel -> keys of diagnostics.PARAMETERS
DIAGNOSTIC_KEYS = {"roll": "rot_x", "pitch": "rot_y", "yaw": "rot_z"}


class ParameterPanel(QtWidgets.QWidget):
    """One row per parameter: its value (editable) and whether the least squares alignment estimates it."""

    def __init__(self, session: AlignmentSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.estimate: dict[str, QtWidgets.QCheckBox] = {}

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setVerticalSpacing(3)
        layout.setColumnStretch(1, 1)
        row = 0
        for title, specs in GROUPS:
            header = QtWidgets.QLabel(f"<b>{title}</b>")
            layout.addWidget(header, row, 0, 1, 2)
            if row == 0:
                estimate_header = QtWidgets.QLabel("estimate")
                estimate_header.setToolTip(
                    "Parameters estimated by “Auto align” and “Refine”; the others keep their value."
                )
                layout.addWidget(estimate_header, row, 2, QtCore.Qt.AlignmentFlag.AlignHCenter)
            row += 1
            for spec in specs:
                self._add_row(layout, row, spec)
                row += 1

        self._init_estimation()
        for check in self.estimate.values():
            check.toggled.connect(self._publish_estimated)
        self._publish_estimated()
        session.state_changed.connect(self.refresh)
        self.refresh()

    def _add_row(self, layout: QtWidgets.QGridLayout, row: int, spec: ParameterSpec) -> None:
        label = QtWidgets.QLabel(spec.label)
        label.setToolTip(spec.tooltip)
        label.setIndent(10)
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(spec.decimals)
        spin.setRange(spec.minimum, 1e12)
        spin.setSingleStep(spec.step)
        spin.setSuffix(f" {spec.unit}" if spec.unit else "")
        spin.setKeyboardTracking(False)
        spin.setToolTip(f"{spec.tooltip}. Type a value, or use the arrows / mouse wheel (Ctrl: 10× steps).")
        spin.valueChanged.connect(lambda value, key=spec.key: self._on_edited(key, value))
        check = QtWidgets.QCheckBox()
        check.setToolTip(f"Estimate the {spec.label.split(' (')[0]} by least squares")
        layout.addWidget(label, row, 0)
        layout.addWidget(spin, row, 1)
        layout.addWidget(check, row, 2, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.spins[spec.key] = spin
        self.estimate[spec.key] = check

    def _init_estimation(self) -> None:
        """Preselects the parameters to estimate from the alignment settings of the trajectory."""
        configured = self.session.alignment_settings.estimation_settings
        for key, check in self.estimate.items():
            check.setChecked(bool(getattr(configured, SPECS[key].estimation)))
        if not self.session.has_orientation:
            for key in LEVERARM_KEYS:
                reason = "The trajectory has no orientations: a leverarm cannot be applied."
                self.estimate[key].setChecked(False)
                for widget in (self.estimate[key], self.spins[key]):
                    widget.setEnabled(False)
                    widget.setToolTip(reason)

    def estimated_parameters(self) -> list[str]:
        """The checked parameters, as keys of ``diagnostics.PARAMETERS``."""
        return [DIAGNOSTIC_KEYS.get(key, key) for key, check in self.estimate.items() if check.isChecked()]

    def _publish_estimated(self, *_) -> None:
        self.session.set_estimated_parameters(self.estimated_parameters())

    @property
    def estimation_settings(self) -> AlignmentEstimationSettings:
        """The parameters the least squares alignment estimates (the checked boxes)."""
        selected = {SPECS[key].estimation: check.isChecked() for key, check in self.estimate.items()}
        return AlignmentEstimationSettings(**selected, sensor_rotation=False)

    def values(self) -> dict[str, float]:
        """Exact values in internal units (m, rad, s)."""
        state = self.session.state
        return (
            dict(zip(SHIFT_KEYS, self.session.pivot_shift, strict=True))
            | dict(zip(ANGLE_KEYS, state.euler, strict=True))
            | {"scale": state.scale, "time_shift": state.time_shift}
            | dict(zip(LEVERARM_KEYS, state.leverarm, strict=True))
        )

    def refresh(self) -> None:
        for key, value in self.values().items():
            spin = self.spins[key]
            spin.blockSignals(True)
            spin.setValue(value * SPECS[key].to_display)
            spin.blockSignals(False)

    def set_value(self, key: str, value: float) -> None:
        """Sets a parameter to ``value`` (internal unit) as if the user had typed it."""
        self._on_edited(key, value * SPECS[key].to_display)

    def _on_edited(self, key: str, display_value: float) -> None:
        # only the edited value changes; the others keep their exact (not the displayed, rounded) values
        value = display_value / SPECS[key].to_display
        session, state = self.session, self.session.state
        new_state: AlignmentState
        if key in SHIFT_KEYS:
            delta = np.zeros(3)
            index = SHIFT_KEYS.index(key)
            delta[index] = value - session.pivot_shift[index]
            new_state = state.moved(delta)
        elif key in ANGLE_KEYS:
            angles = state.euler.copy()
            angles[ANGLE_KEYS.index(key)] = value
            new_state = state.with_rotation(Rotations.from_euler("xyz", angles).as_matrix(), session.pivot)
        elif key == "scale":
            new_state = state.with_scale(value, session.pivot)
        elif key == "time_shift":
            new_state = state.with_time_shift(value)
        else:
            leverarm = state.leverarm.copy()
            leverarm[LEVERARM_KEYS.index(key)] = value
            new_state = state.with_leverarm(leverarm)
        session.set_state(new_state, f"Set {SPECS[key].label.split(' (')[0]}")
