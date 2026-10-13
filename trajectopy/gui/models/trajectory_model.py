import copy
import logging
import os

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QInputDialog

from trajectopy.gui.managers.requests import (
    FileRequest,
    FileRequestType,
    TrajectoryModelRequest,
    TrajectoryModelRequestType,
)
from trajectopy.gui.models.entries import TrajectoryEntry
from trajectopy.gui.models.selection import TrajectorySelection
from trajectopy.gui.models.table_model import _ROOT_INDEX, RequestTableModel
from trajectopy.visualization import style

logger = logging.getLogger(__name__)


def _row_color(row: int) -> str:
    return style.PALETTE[row % len(style.PALETTE)]


class TrajectoryTableModel(RequestTableModel):
    """Class representing the trajectory table model."""

    file_request = Signal(FileRequest)
    items: list[TrajectoryEntry]

    def __init__(self):
        REQUEST_MAPPING = {
            TrajectoryModelRequestType.ADD: self.add_entries,
            TrajectoryModelRequestType.UPDATE: self.update_selected_entries,
            TrajectoryModelRequestType.RENAME: self.rename_trajectories,
            TrajectoryModelRequestType.COPY: self.copy_trajectories,
            TrajectoryModelRequestType.REMOVE: self.remove_selected_entries,
            TrajectoryModelRequestType.REMOVE_RELATED: self.remove_related_trajectories,
            TrajectoryModelRequestType.SET_REFERENCE: self.set_as_reference,
            TrajectoryModelRequestType.UNSET_REFERENCE: self.unset_as_reference,
            TrajectoryModelRequestType.RESET: self.reset,
            TrajectoryModelRequestType.EXPORT_ALL: self.export_all,
            TrajectoryModelRequestType.SORT: self.sort_items,
        }

        super().__init__(
            headers=[
                "Name",
                "Reference",
                "# Poses",
                "EPSG",
                "Sorted by",
                "Filename",
            ],
            REQUEST_MAPPING=REQUEST_MAPPING,
        )
        self.badge_column = len(self._headers)
        self.badges: dict[str, str] = {}  # entry id -> short summary of its evaluation, e.g. the ATE RMS

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX) -> int:
        return len(self._headers) + 1

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if section == self.badge_column and orientation == Qt.Orientation.Horizontal:
            return "ATE RMS" if role == Qt.ItemDataRole.DisplayRole else None
        return super().headerData(section, orientation, role)

    def set_badges(self, badges: dict[str, str]) -> None:
        if badges == self.badges:
            return
        self.badges = badges
        if self.items:
            self.dataChanged.emit(self.index(0, self.badge_column), self.index(len(self.items) - 1, self.badge_column))

    def is_drawable(self, entry: TrajectoryEntry) -> bool:
        return True

    def color(self, entry: TrajectoryEntry) -> str:
        """Color of the trajectory in all views, also shown as a swatch next to its name."""
        return _row_color(self.items.index(entry))

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if index.row() >= len(self.items):
            return None
        if index.column() == self.badge_column:
            if role == Qt.ItemDataRole.DisplayRole:
                return self.badges.get(self.items[index.row()].entry_id, "")
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            if role == Qt.ItemDataRole.FontRole:
                return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            return None
        if role == Qt.ItemDataRole.DecorationRole and index.column() == 0:
            return QColor(_row_color(index.row()))
        if role == Qt.ItemDataRole.FontRole and index.column() == 0 and self.items[index.row()].set_as_reference:
            font = QFont()
            font.setBold(True)
            return font
        return super().data(index, role)

    def remove_group(self, group_id: str) -> None:
        self.items = [item for item in self.items if item.group_id not in group_id]
        self.layoutChanged.emit()

    def export_all(self, request: TrajectoryModelRequest) -> None:
        for item in self.items:
            trajectory_filename = os.path.join(request.file_path, f"{item.trajectory.name}_{item.entry_id}.traj")
            settings_filename = os.path.join(request.file_path, f"{item.trajectory.name}_{item.entry_id}.json")
            self.file_request.emit(
                FileRequest(
                    type=FileRequestType.WRITE_TRAJ,
                    trajectory_selection=TrajectorySelection(entries=[item]),
                    file_list=[trajectory_filename],
                )
            )
            item.settings.to_file(settings_filename)

            if item.trajectory.positions.local_transformer is not None:
                item.trajectory.positions.export_local_transformer(
                    os.path.join(request.file_path, f"{item.trajectory.name}_{item.entry_id}_local_transformer.txt")
                )

        self.write_trajectory_order(request.file_path)

    def write_trajectory_order(self, directory: str) -> None:
        trajectory_order_filename = os.path.join(directory, "trajectory_order.txt")
        self.file_request.emit(
            FileRequest(
                type=FileRequestType.WRITE_LIST,
                id_list=[item.entry_id for item in self.items],
                file_list=[trajectory_order_filename],
            )
        )

    @property
    def reference_entry(self) -> TrajectoryEntry | None:
        return next((item for item in self.items if item.set_as_reference), None)

    def rename_trajectories(self, request: TrajectoryModelRequest) -> None:
        if not isinstance((selection := request.selection), TrajectorySelection):
            raise ValueError("No trajectory selection provided")

        input_name, ok = QInputDialog.getText(None, "Please enter a name", "Name:", text=selection.entries[0].name)
        if ok and input_name is not None:
            selection.entries[0].trajectory.name = input_name
        self.layoutChanged.emit()

    def remove_related_trajectories(self, request: TrajectoryModelRequest) -> None:
        """Remove all trajectories that are related"""
        if not isinstance((selection := request.selection), TrajectorySelection):
            raise ValueError("No trajectory selection provided")

        for entry in selection.entries:
            self.remove_group(entry.group_id)

    def copy_trajectories(self, request: TrajectoryModelRequest) -> None:
        """Deep copy trajectory"""
        if not isinstance((selection := request.selection), TrajectorySelection):
            raise ValueError("No trajectory selection provided")

        for entry in selection.entries:
            entry_copy = copy.deepcopy(entry)
            entry_copy.renew_id()
            entry_copy.set_as_reference = False
            self.add(entry_copy)

    def set_as_reference(self, request: TrajectoryModelRequest) -> None:
        """Marks selected trajectory as reference"""
        if not isinstance((selection := request.selection), TrajectorySelection):
            raise ValueError("No trajectory selection provided")

        selected_id = selection.entries[0].entry_id
        for entry in self.items:
            entry.set_as_reference = entry.entry_id == selected_id
        self.layoutChanged.emit()

    def unset_as_reference(self, request: TrajectoryModelRequest) -> None:
        """Removes the reference status from the selected trajectory"""
        for entry in self.items:
            entry.set_as_reference = False
        self.layoutChanged.emit()
