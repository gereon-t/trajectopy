import logging
import os

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QInputDialog

from trajectopy.gui import themes
from trajectopy.gui.managers.requests import (
    FileRequest,
    FileRequestType,
    ResultModelRequest,
    ResultModelRequestType,
    UIRequest,
    UIRequestType,
)
from trajectopy.gui.models.entries import (
    AbsoluteDeviationEntry,
    AlignmentEntry,
    RelativeDeviationEntry,
    ResultEntry,
)
from trajectopy.gui.models.selection import ResultSelection
from trajectopy.gui.models.table_model import _ROOT_INDEX, RequestTableModel

logger = logging.getLogger(__name__)


class ResultTableModel(RequestTableModel):
    """Class representing the result table model."""

    file_request = Signal(FileRequest)
    items: list[ResultEntry]

    def __init__(self):
        REQUEST_MAPPING = {
            ResultModelRequestType.ADD: self.add_entries,
            ResultModelRequestType.UPDATE: self.update_selected_entries,
            ResultModelRequestType.RENAME: self.rename_result,
            ResultModelRequestType.REMOVE: self.remove_selected_entries,
            ResultModelRequestType.PASS_ALIGNMENTS_TO_UI: self.pass_alignments_to_ui,
            ResultModelRequestType.RESET: self.reset,
            ResultModelRequestType.EXPORT_ALL: self.export_all,
            ResultModelRequestType.EDIT_ALIGNMENT: self.edit_alignment,
            ResultModelRequestType.COPY: self.copy_selected_entries,
            ResultModelRequestType.SORT: self.sort_items,
        }
        super().__init__(
            headers=["Name", "Type", "# Values", "ID"],
            REQUEST_MAPPING=REQUEST_MAPPING,
        )
        self.value_column = len(self._headers)
        self.in_use: set[str] | None = None  # ids of the results the views use; None: not known (all look alike)
        self.short_names: dict[str, str] = {}  # names without the trajectory they belong to, e.g. "vs. Reference"

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX) -> int:
        return len(self._headers) + 1

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if section == self.value_column and orientation == Qt.Orientation.Horizontal:
            return "Value" if role == Qt.ItemDataRole.DisplayRole else None
        return super().headerData(section, orientation, role)

    def set_in_use(self, ids: set[str], short_names: dict[str, str]) -> None:
        if ids == self.in_use and short_names == self.short_names:
            return
        self.in_use = ids
        self.short_names = short_names
        if self.items:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.items) - 1, self.value_column))

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.items):
            return None
        entry = self.items[index.row()]
        unused = self.in_use is not None and entry.entry_id not in self.in_use
        if role == Qt.ItemDataRole.DecorationRole and index.column() == 0:
            label, color = themes.RESULT_BADGES.get(entry.type, (entry.type[:3].upper(), "#57606a"))
            return themes.badge_icon(label, color)
        if role == Qt.ItemDataRole.ForegroundRole and unused:
            return QColor(themes.color("disabled"))
        if role == Qt.ItemDataRole.ToolTipRole:
            if unused:
                return (
                    f"{entry.name}\nNot shown in the views: they use the newest result of each type "
                    "of the loaded trajectories."
                )
            return entry.name
        if role == Qt.ItemDataRole.DisplayRole and index.column() == 0 and entry.entry_id in self.short_names:
            return self.short_names[entry.entry_id]
        if index.column() == self.value_column:
            if role == Qt.ItemDataRole.DisplayRole:
                return _summary(entry)
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return None
        return super().data(index, role)

    def export_all(self, request: ResultModelRequest) -> None:
        """Exports all results to files named after their entry id."""
        for item in self.items:
            filename = os.path.join(request.file_path, f"{item.name}_{item.entry_id}.result")
            self.file_request.emit(
                FileRequest(
                    type=FileRequestType.WRITE_RES,
                    result_selection=ResultSelection(entries=[item]),
                    file_list=[filename],
                )
            )
        self.write_result_order(request.file_path)

    def write_result_order(self, directory: str) -> None:
        result_order_filename = os.path.join(directory, "result_order.txt")
        self.file_request.emit(
            FileRequest(
                type=FileRequestType.WRITE_LIST,
                id_list=[item.entry_id for item in self.items],
                file_list=[result_order_filename],
            )
        )

    def rename_result(self, request: ResultModelRequest) -> None:
        selected_entry = request.selection.entries[0]
        input_name, ok = QInputDialog.getText(None, "Please enter a name", "Name:", text=selected_entry.name)
        if ok and input_name is not None:
            selected_entry.name = input_name

        self.layoutChanged.emit()

    @property
    def alignment_entries(self) -> list[AlignmentEntry]:
        return [entry for entry in self.items if isinstance(entry, AlignmentEntry)]

    def pass_alignments_to_ui(self, request: ResultModelRequest) -> None:
        if alignment_entries := self.alignment_entries:
            self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.ALIGNMENT_SELECTION,
                    result_selection=ResultSelection(entries=list[ResultEntry](alignment_entries)),
                    trajectory_selection=request.trajectory_selection,
                )
            )
        else:
            self.ui_request.emit(UIRequest(type=UIRequestType.MESSAGE, message="No Alignments available!"))

    def edit_alignment(self, request: ResultModelRequest) -> None:
        self.ui_request.emit(
            UIRequest(
                type=UIRequestType.EDIT_ALIGNMENT,
                alignment_entry=request.alignment,
            )
        )


def _summary(entry: ResultEntry) -> str:
    """The key number of a result, shown next to its name."""
    try:
        if isinstance(entry, AbsoluteDeviationEntry):
            return f"{entry.deviations.pos_dev_rms:.3f} m"
        if isinstance(entry, RelativeDeviationEntry):
            return f"{entry.deviations.pos_rpe:.3f} {entry.deviations.pos_drift_unit}"
        if isinstance(entry, AlignmentEntry):
            return f"{len(entry)} params"
    except Exception:
        logger.debug("No summary for result '%s'", entry.name, exc_info=True)
    return ""
