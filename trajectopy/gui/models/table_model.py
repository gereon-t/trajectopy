import copy
import logging
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, Signal, Slot

from trajectopy.gui.managers.requests import (
    Request,
    ResultModelRequest,
    TrajectoryModelRequest,
    UIRequest,
    generic_request_handler,
)
from trajectopy.gui.models.entries import Entry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Entry)

_ROOT_INDEX = QModelIndex()


class BaseTableModel(QAbstractTableModel, Generic[T]):
    """Base class for all table models.

    Entries for which ``is_drawable`` is true get a checkbox in the first column that controls whether
    they are shown in the 3D view.
    """

    visibility_changed = Signal()

    def __init__(self, headers: list[str] | None = None):
        super().__init__()
        self.items: list[T] = []
        self._headers = [""] if headers is None else headers
        self._hidden_ids: set[str] = set()

    def is_drawable(self, entry: T) -> bool:
        return False

    def is_visible(self, entry: T) -> bool:
        return self.is_drawable(entry) and entry.entry_id not in self._hidden_ids

    def set_visible(self, entries: list[T], visible: bool) -> None:
        for entry in entries:
            if visible:
                self._hidden_ids.discard(entry.entry_id)
            else:
                self._hidden_ids.add(entry.entry_id)
        if self.items:
            # whole rows: views may also show hidden entries differently (e.g. muted)
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.items) - 1, self.columnCount() - 1))
        self.visibility_changed.emit()

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX) -> int:
        return len(self.items)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX) -> int:
        return len(self._headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        flags = super().flags(index)
        if index.column() == 0 and index.row() < len(self.items) and self.is_drawable(self.items[index.row()]):
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not self.items or index.row() >= len(self.items):
            return None

        entry = self.items[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return entry.column[index.column()]

        if role == Qt.ItemDataRole.CheckStateRole and index.column() == 0 and self.is_drawable(entry):
            return Qt.CheckState.Checked if self.is_visible(entry) else Qt.CheckState.Unchecked

        return None

    def setData(
        self, index: QModelIndex | QPersistentModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole
    ) -> bool:
        if role != Qt.ItemDataRole.CheckStateRole or index.column() != 0 or index.row() >= len(self.items):
            return False
        self.set_visible([self.items[index.row()]], Qt.CheckState(value) == Qt.CheckState.Checked)
        return True

    def get(self, item_id: str) -> None | T:
        return next((item for item in self.items if item.entry_id == item_id), None)

    def set(self, item_id: str, entry: T) -> None:
        for i, item in enumerate(self.items):
            if item.entry_id == item_id:
                self.items[i] = entry
                self.layoutChanged.emit()
                return

    def add(self, entry: T) -> None:
        self.items.append(entry)
        self.layoutChanged.emit()

    def remove(self, ids: list[str]) -> None:
        self.items = [item for item in self.items if item.entry_id not in ids]
        self.layoutChanged.emit()


class RequestTableModel(BaseTableModel):
    """Base class for all table models that handle requests."""

    operation_finished = Signal()
    ui_request = Signal(UIRequest)

    def __init__(
        self,
        headers: list[str],
        REQUEST_MAPPING: Mapping[Any, Callable[[Any], None]] | None = None,
    ):
        if REQUEST_MAPPING is None:
            REQUEST_MAPPING = {}
        super().__init__(headers)
        self.REQUEST_MAPPING = REQUEST_MAPPING

    @Slot(Request)
    def handle_request(self, request: TrajectoryModelRequest | ResultModelRequest) -> None:
        generic_request_handler(self, request)

    def add_entries(self, request) -> None:
        for entry in request.selection.entries:
            self.add(entry)

    def update_selected_entries(self, request) -> None:
        for entry in request.selection.entries:
            self.set(entry.entry_id, entry)

    def remove_selected_entries(self, request) -> None:
        """Remove selected entries"""
        self.remove([entry.entry_id for entry in request.selection.entries])

    def copy_selected_entries(self, request) -> None:
        """Deep copy entries"""
        for entry in request.selection.entries:
            entry_copy = copy.deepcopy(entry)
            entry_copy.renew_id()
            self.add(entry_copy)

    def reset(self, request=None) -> None:
        self.items = []
        self.layoutChanged.emit()

    def sort_items(self, request) -> None:
        """Sorts the items so that their id order matches the desired order"""
        desired_order = request.index_list
        present_ids = [item.entry_id for item in self.items]
        if any(id not in present_ids for id in desired_order):
            logger.error("Desired order contains ids that are not present in the model. Sorting aborted.")
            return

        self.items = [entry for id in desired_order if (entry := self.get(id)) is not None]
        self.layoutChanged.emit()
