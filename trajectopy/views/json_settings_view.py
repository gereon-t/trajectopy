import json
import logging
from typing import Any, Dict, List

from PyQt6 import QtWidgets
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QFont
from trajectopy_core.settings.base import Settings

from trajectopy.util import save_file_dialog, show_msg_box

logger = logging.getLogger("root")

EXPORT_FORMATS = ["png", "svg", "webp", "jpeg"]
PLOT_MODES = ["lines+markers", "lines", "markers"]


class JSONViewer(QtWidgets.QMainWindow):
    def __init__(self, parent, settings: Settings, name: str = "Settings Viewer"):
        super().__init__(parent=parent)
        self.form_item_cnt = 0
        self.settings = settings
        self.setWindowTitle(name)
        self.setup_ui()
        self.update_view()

    def setup_ui(self):
        self.resize(400, 400)

        self.scroll_area = QtWidgets.QScrollArea()
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.vlayout = QtWidgets.QVBoxLayout(self.central_widget)
        self.hlayout = QtWidgets.QHBoxLayout()

        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.central_widget)

        self.setCentralWidget(self.scroll_area)

        self.form_layout = QtWidgets.QFormLayout()
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.form_layout.setFormAlignment(Qt.AlignmentFlag.AlignRight)
        self.form_layout.setObjectName("formLayout")

        self.vlayout.addLayout(self.form_layout)

        vspacer = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.vlayout.addItem(vspacer)
        self.vlayout.addLayout(self.hlayout)

        self.import_button = QtWidgets.QPushButton("Import", self)
        self.import_button.clicked.connect(self.open_file_dialog)
        self.export_button = QtWidgets.QPushButton("Export", self)
        self.export_button.clicked.connect(self.export_json)

        self.hlayout.addWidget(self.import_button)
        self.hlayout.addWidget(self.export_button)

        hspacer = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.hlayout.addItem(hspacer)
        self.apply_button = QtWidgets.QPushButton("Apply", self)
        self.apply_button.clicked.connect(lambda: self.apply_changes(close_after=False))
        self.apply_and_close_button = QtWidgets.QPushButton("Apply and Close", self)
        self.apply_and_close_button.clicked.connect(lambda: self.apply_changes(close_after=True))

        self.hlayout.addWidget(self.apply_button)
        self.hlayout.addWidget(self.apply_and_close_button)

    def open_file_dialog(self):
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setNameFilter("JSON Files (*.json)")
        filename, _ = file_dialog.getOpenFileName()
        if filename:
            self.load_json_file(filename)
            logger.info("Loaded processing settings file %s", filename)

    def remove_items(self):
        for item in self.central_widget.children():
            if isinstance(
                item,
                (
                    QtWidgets.QPushButton,
                    QtWidgets.QSpacerItem,
                    QtWidgets.QScrollArea,
                    QtWidgets.QVBoxLayout,
                    QtWidgets.QHBoxLayout,
                ),
            ):
                continue
            item.deleteLater()

        self.form_item_cnt = 0

    def load_json_file(self, filename):
        self.remove_items()
        try:
            with open(filename, "r", encoding="utf-8") as file:
                json_content = file.read()
                self.populate_form("root", json.loads(json_content))
        except Exception as e:
            show_msg_box(f"Error loading JSON file: {e}")

    def export_json(self):
        if not (file := save_file_dialog(parent=self, file_filter="Settings File (*.json)")):
            return

        try:
            json_content = self.populate_json()
            with open(file, "w", encoding="utf-8") as file:
                file.write(json_content)

            logger.info("Saved processing settings file %s", file.name)
        except Exception as e:
            show_msg_box(f"Error saving JSON file: {e}")

    def apply_changes(self, close_after: bool = False):
        try:
            json_content = self.populate_json()
            parsed_json = json.loads(json_content)

            self.settings.update_from_dict(parsed_json)

            logger.info("Updated settings")
            if close_after:
                self.close()
        except json.JSONDecodeError as e:
            show_msg_box(f"Error decoding JSON: {e}")

    def update_view(self):
        self.remove_items()
        self.populate_form("root", self.settings.to_dict())

    def populate_json(self) -> str:
        """Populate the JSON with the values from the form."""

        def init_dict(json_dict: Dict[str, Any], keys: List[str]) -> None:
            local_reference = json_dict
            for key in keys:
                local_reference = local_reference.setdefault(key, {})

        def setval(json_dict: Dict[str, Any], keys: List[str], val: Any) -> None:
            local_reference = json_dict

            lastkey = keys[-1]

            for k in keys[:-1]:
                local_reference = local_reference[k]
            local_reference[lastkey] = val

        json_content: dict = {}
        for item in self.central_widget.children():
            if isinstance(item, QtWidgets.QLabel):
                continue
            keys = item.objectName().split("-")[2:]

            if not keys:
                continue

            init_dict(json_content, keys)

            if isinstance(item, QtWidgets.QCheckBox):
                item_value = item.isChecked()
            elif isinstance(item, QtWidgets.QSpinBox):
                item_value = item.value()
            elif isinstance(item, QtWidgets.QDoubleSpinBox):
                item_value = item.value()
            elif isinstance(item, QtWidgets.QComboBox):
                item_value = item.currentText()
            elif isinstance(item, QtWidgets.QLineEdit):
                item_value = item.text()
            else:
                item_value = None

            setval(json_content, keys, item_value)

        return json.dumps(json_content, indent=4)

    def populate_form(self, parent_name: str, json_content: Dict[str, Any]):
        def pretty_type(type_string: str) -> str:
            return type_string.split("'")[1]

        def add_settings_field(key: str, value: Any) -> None:
            settings_field = construct_settings_field(self.central_widget, value)
            settings_field.setObjectName(f"field-{parent_name}-{key}")
            self.form_layout.setWidget(self.form_item_cnt, QtWidgets.QFormLayout.ItemRole.FieldRole, settings_field)

        _translate = QCoreApplication.translate
        for key, value in json_content.items():
            if isinstance(value, dict):
                group_label = QtWidgets.QLabel(self.central_widget)
                font = QFont()
                font.setBold(True)

                group_label.setFont(font)
                group_label.setObjectName(f"label-{parent_name}-{key}")
                group_label.setText(_translate("MainWindow", key.replace("_", " ").title()))
                self.form_layout.setWidget(self.form_item_cnt, QtWidgets.QFormLayout.ItemRole.LabelRole, group_label)
                self.form_item_cnt += 1
                self.populate_form(f"{parent_name}-{key}", value)
            else:
                settings_label = QtWidgets.QLabel(self.central_widget)
                settings_label.setObjectName(f"label-{parent_name}-{key}")
                self.form_layout.setWidget(
                    self.form_item_cnt, QtWidgets.QFormLayout.ItemRole.LabelRole, settings_label
                )
                settings_label.setText(
                    _translate("MainWindow", f"{key.replace('_', ' ').title()} ({pretty_type(str(type(value)))})")
                )

                add_settings_field(key, value)
                self.form_item_cnt += 1


def construct_settings_field(central_widget: QtWidgets.QWidget, value: Any) -> QtWidgets.QWidget:
    if isinstance(value, bool):
        settings_field = QtWidgets.QCheckBox(central_widget)
        settings_field.setChecked(value)
        return settings_field

    if isinstance(value, int):
        settings_field = QtWidgets.QSpinBox(central_widget)
        settings_field.setMaximum(10000)
        settings_field.setValue(value)
        return settings_field

    if isinstance(value, float):
        settings_field = QtWidgets.QDoubleSpinBox(central_widget)
        settings_field.setDecimals(4)
        settings_field.setSingleStep(0.0001)
        settings_field.setMaximum(10000)
        settings_field.setValue(value)
        return settings_field

    if value in EXPORT_FORMATS:
        settings_field = QtWidgets.QComboBox(central_widget)
        settings_field.addItems(EXPORT_FORMATS)
        settings_field.setCurrentIndex(EXPORT_FORMATS.index(value))
        return settings_field

    if value in PLOT_MODES:
        settings_field = QtWidgets.QComboBox(central_widget)
        settings_field.addItems(PLOT_MODES)
        settings_field.setCurrentIndex(PLOT_MODES.index(value))
        return settings_field

    settings_field = QtWidgets.QLineEdit(central_widget)
    settings_field.setText(str(value))
    return settings_field
