from collections.abc import Sequence
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.gui.utils import app_settings
from trajectopy.io.kml import DEFAULT_COLOR_CLASSES, KmlField

HEIGHT_MODES = {
    "ground": "On the ground (terrain)",
    "ellipsoidal": "Trajectory heights: ellipsoidal (e.g. GNSS)",
    "sea_level": "Trajectory heights: above sea level",
}
COLORMAPS = {
    "rdylbu": "Blue - yellow - red (like the plots)",
    "viridis": "Viridis (colorblind-safe)",
}
ONE_COLOR = ""  # "color by" entry of trajectories drawn in one color

_KEY = "google_earth/"


class KmlExportDialog(QtWidgets.QDialog):
    """Options of the Google Earth export: one color or a field that colors the line (trajectories), or the
    deviation field (ATE results), the color scale, the vertical placement and the line. All choices are
    remembered; fields separately for trajectories and ATE results."""

    def __init__(
        self, parent: QtWidgets.QWidget | None = None, fields: Sequence[KmlField] = (), ate: bool = False
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export to Google Earth")
        self._settings = app_settings()
        self._field_key = _KEY + ("ate_field" if ate else "trajectory_field")
        form = QtWidgets.QFormLayout()

        self.field_box = QtWidgets.QComboBox()
        if not ate:
            self.field_box.addItem("One color", ONE_COLOR)
        for field in fields:
            self.field_box.addItem(f"{field.label} [{field.unit}]", field.key)
        default_field = "position" if ate else ONE_COLOR
        self.field_box.setCurrentIndex(max(0, self.field_box.findData(self._value(self._field_key, default_field))))
        self.field_box.currentIndexChanged.connect(self._update_enabled)
        form.addRow("Color by:", self.field_box)

        self.color = QtGui.QColor(str(self._value("color", "#ff0000")))
        self.color_button = QtWidgets.QPushButton()
        self.color_button.clicked.connect(self._choose_color)
        self._show_color()
        if not ate:
            form.addRow("Color:", self.color_button)

        self.colormap_box = QtWidgets.QComboBox()
        for key, label in COLORMAPS.items():
            self.colormap_box.addItem(label, key)
        self.colormap_box.setCurrentIndex(max(0, self.colormap_box.findData(self._value("colormap", "rdylbu"))))
        form.addRow("Color scale:", self.colormap_box)

        self.max_std_spin = QtWidgets.QDoubleSpinBox()
        self.max_std_spin.setRange(0.0, 100.0)
        self.max_std_spin.setSingleStep(0.5)
        self.max_std_spin.setSpecialValueText("off (minimum to maximum)")
        self.max_std_spin.setValue(self._value("max_std", 3.0, float))
        self.max_std_spin.setToolTip("Clips the color scale to mean ± this many standard deviations.")
        form.addRow("Clip at ± σ:", self.max_std_spin)

        self.fixed_range_check = QtWidgets.QCheckBox("Fixed range")
        self.fixed_range_check.setChecked(self._value("fixed_range", False, bool))
        self.fixed_range_check.setToolTip("The same limits for several exports make their colors comparable.")
        self.fixed_range_check.toggled.connect(self._update_enabled)
        self.range_min_spin, self.range_max_spin = (self._range_spin(), self._range_spin())
        self.range_min_spin.setValue(self._value("range_min", 0.0, float))
        self.range_max_spin.setValue(self._value("range_max", 1.0, float))
        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(self.fixed_range_check)
        range_row.addWidget(self.range_min_spin)
        range_row.addWidget(QtWidgets.QLabel("to"))
        range_row.addWidget(self.range_max_spin)
        form.addRow("Range:", range_row)

        self.classes_spin = QtWidgets.QSpinBox()
        self.classes_spin.setRange(1, 50)
        self.classes_spin.setValue(self._value("color_classes", DEFAULT_COLOR_CLASSES, int))
        self.classes_spin.setToolTip("Number of colors; Google Earth lists each with its value range (legend).")
        form.addRow("Colors:", self.classes_spin)

        self.heights_box = QtWidgets.QComboBox()
        for key, label in HEIGHT_MODES.items():
            self.heights_box.addItem(label, key)
        self.heights_box.setCurrentIndex(max(0, self.heights_box.findData(self._value("heights", "ground"))))
        self.heights_box.setToolTip(
            "Where the trajectory is drawn vertically. Google Earth uses heights above sea level; ellipsoidal "
            "heights are converted with the EGM96 geoid model, which is downloaded (once) from cdn.proj.org."
        )
        self.heights_box.currentIndexChanged.connect(self._update_enabled)
        form.addRow("Heights:", self.heights_box)

        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(-10000.0, 10000.0)
        self.offset_spin.setDecimals(2)
        self.offset_spin.setSuffix(" m")
        self.offset_spin.setValue(self._value("height_offset", 0.0, float))
        self.offset_spin.setToolTip("Added to the heights; on the ground, the trajectory floats this far above it.")
        self.offset_spin.valueChanged.connect(self._update_enabled)
        form.addRow("Height offset:", self.offset_spin)

        self.extrude_check = QtWidgets.QCheckBox("Connect to the ground (vertical curtain)")
        self.extrude_check.setChecked(self._value("extrude", False, bool))
        form.addRow("", self.extrude_check)

        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 20.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setSuffix(" px")
        self.line_width_spin.setValue(self._value("line_width", 4.0 if ate else 3.0, float))
        form.addRow("Line width:", self.line_width_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Export…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setMinimumWidth(460)
        self._update_enabled()

    @property
    def field(self) -> str | None:
        """Key of the field that colors the line, or None for one color."""
        return str(self.field_box.currentData()) or None

    def options(self) -> dict[str, Any]:
        """Keyword arguments of :func:`trajectopy.io.kml.write_kml` / :func:`~trajectopy.io.kml.write_ate_kml`."""
        options: dict[str, Any] = {
            "heights": str(self.heights_box.currentData()),
            "height_offset": self.offset_spin.value(),
            "extrude": self.extrude_check.isEnabled() and self.extrude_check.isChecked(),
            "line_width": self.line_width_spin.value(),
        }
        if self.field is None:
            options["color"] = self.color.name()
        else:
            options |= {
                "field": self.field,
                "colormap": str(self.colormap_box.currentData()),
                "max_std": self.max_std_spin.value(),
                "value_range": (
                    (self.range_min_spin.value(), self.range_max_spin.value())
                    if self.fixed_range_check.isChecked()
                    else None
                ),
                "color_classes": self.classes_spin.value(),
            }
        return options

    def accept(self) -> None:
        if self.field is not None and self.fixed_range_check.isChecked():
            if self.range_min_spin.value() >= self.range_max_spin.value():
                QtWidgets.QMessageBox.warning(self, "Invalid Range", "The lower limit must be below the upper one.")
                return
        values = {
            "color": self.color.name(),
            "colormap": self.colormap_box.currentData(),
            "max_std": self.max_std_spin.value(),
            "fixed_range": self.fixed_range_check.isChecked(),
            "range_min": self.range_min_spin.value(),
            "range_max": self.range_max_spin.value(),
            "color_classes": self.classes_spin.value(),
            "heights": self.heights_box.currentData(),
            "height_offset": self.offset_spin.value(),
            "extrude": self.extrude_check.isChecked(),
            "line_width": self.line_width_spin.value(),
        }
        for key, value in values.items():
            self._settings.setValue(_KEY + key, value)
        self._settings.setValue(self._field_key, self.field_box.currentData())
        super().accept()

    def _value(self, key: str, default: Any, value_type: type = str) -> Any:
        key = key if key.startswith(_KEY) else _KEY + key
        try:
            return self._settings.value(key, default, type=value_type)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _range_spin() -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1e9, 1e9)
        spin.setDecimals(3)
        return spin

    def _update_enabled(self) -> None:
        by_field = self.field is not None
        self.color_button.setEnabled(not by_field)
        for widget in (self.colormap_box, self.fixed_range_check, self.classes_spin):
            widget.setEnabled(by_field)
        fixed = self.fixed_range_check.isChecked()
        self.max_std_spin.setEnabled(by_field and not fixed)
        self.range_min_spin.setEnabled(by_field and fixed)
        self.range_max_spin.setEnabled(by_field and fixed)
        on_ground = self.heights_box.currentData() == "ground" and self.offset_spin.value() == 0
        self.extrude_check.setEnabled(not on_ground)

    def _choose_color(self) -> None:
        color = QtWidgets.QColorDialog.getColor(self.color, self, "Trajectory Color")
        if color.isValid():
            self.color = color
            self._show_color()

    def _show_color(self) -> None:
        self.color_button.setText(self.color.name())
        icon = QtGui.QPixmap(QtCore.QSize(14, 14))
        icon.fill(self.color)
        self.color_button.setIcon(QtGui.QIcon(icon))
