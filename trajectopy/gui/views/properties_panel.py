from collections.abc import Sequence

from PySide6 import QtCore, QtWidgets

from trajectopy.gui.managers.requests import PropertyModelRequest, PropertyModelRequestType
from trajectopy.gui.models.entries import PropertyEntry, ResultEntry, TrajectoryEntry
from trajectopy.gui.models.property_model import PropertyTableModel
from trajectopy.gui.utils import save_file_dialog


def merge_property_dicts(dicts: Sequence[dict[str, str]]) -> dict[str, list[str]]:
    """One row per property, one column per entry ("-" where an entry lacks the property)."""
    merged: dict[str, list[str]] = {}
    for i, properties in enumerate(dicts):
        for key, value in properties.items():
            merged.setdefault(key, ["-"] * len(dicts))[i] = value
    return merged


class PropertiesPanel(QtWidgets.QWidget):
    """All properties of the selected trajectories or results, side by side."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = PropertyTableModel()
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)

        self.placeholder = QtWidgets.QLabel("Select trajectories or results to see their properties.")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setEnabled(False)

        self.export_button = QtWidgets.QPushButton("Export…")
        self.export_button.setToolTip("Saves the properties shown as a CSV file.")
        self.export_button.clicked.connect(self._export)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.export_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.placeholder, 1)
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)
        self.show_entries([])

    def show_entries(self, entries: Sequence[TrajectoryEntry | ResultEntry]) -> None:
        merged = merge_property_dicts([entry.property_dict for entry in entries])
        self.model.beginResetModel()
        self.model.set_headers(["Property", *(entry.name for entry in entries)])
        self.model.items = [PropertyEntry(name=key, values=tuple(values)) for key, values in merged.items()]
        self.model.endResetModel()
        has_entries = bool(entries)
        self.table.setVisible(has_entries)
        self.placeholder.setVisible(not has_entries)
        self.export_button.setEnabled(has_entries)
        if has_entries:
            self.table.resizeColumnsToContents()

    def _export(self) -> None:
        if selected_file := save_file_dialog(None, file_filter="CSV file (*.csv);;All Files (*)"):
            self.model.handle_request(
                PropertyModelRequest(type=PropertyModelRequestType.EXPORT, file_path=selected_file)
            )
