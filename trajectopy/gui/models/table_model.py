import copy
import logging
from collections.abc import Callable, Collection, Mapping
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
    """Base class for all table models."""

    def __init__(self, headers: list[str] | None = None):
        super().__init__()
        self.items: list[T] = []
        self._headers = [""] if headers is None else headers

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX) -> int:
        return len(self.items)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX) -> int:
        return len(self._headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not self.items:
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return self.items[index.row()].column[index.column()]

        return None

    def get(self, item_id: str) -> None | T:
        return next((item for item in self.items if item.entry_id == item_id), None)

    def set(self, item_id: str, entry: T) -> None:
        for row, item in enumerate(self.items):
            if item.entry_id == item_id:
                self.items[row] = entry
                self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount() - 1))
                return

    def add(self, entry: T) -> None:
        row = len(self.items)
        self.beginInsertRows(_ROOT_INDEX, row, row)
        self.items.append(entry)
        self.endInsertRows()

    def remove(self, ids: Collection[str]) -> None:
        # one row at a time from the bottom, so that the views keep their selection on the remaining rows
        for row in reversed(range(len(self.items))):
            if self.items[row].entry_id in ids:
                self.beginRemoveRows(_ROOT_INDEX, row, row)
                del self.items[row]
                self.endRemoveRows()

    def clear(self) -> None:
        self.beginResetModel()
        self.items = []
        self.endResetModel()

    def refresh(self) -> None:
        """Notifies the views that the shown values of (any of) the entries changed."""
        if self.items:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.items) - 1, self.columnCount() - 1))

    def reorder(self, items: list[T]) -> None:
        """Replaces the items by the same items in a new order; selections move along with their entries."""
        self.layoutAboutToBeChanged.emit()
        old_rows = {id(item): row for row, item in enumerate(self.items)}
        new_rows = {old_rows[id(item)]: row for row, item in enumerate(items)}
        self.items = items
        old_indexes = self.persistentIndexList()
        new_indexes = [self.index(new_rows[index.row()], index.column()) for index in old_indexes]
        self.changePersistentIndexList(old_indexes, new_indexes)
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
        self.clear()

    def sort_items(self, request) -> None:
        """Sorts the items so that their id order matches the desired order"""
        desired_order = request.index_list
        present_ids = [item.entry_id for item in self.items]
        if any(id not in present_ids for id in desired_order):
            logger.error("Desired order contains ids that are not present in the model. Sorting aborted.")
            return

        ordered = [entry for id in desired_order if (entry := self.get(id)) is not None]
        # entries missing in the order file stay at the end
        self.reorder(ordered + [item for item in self.items if item.entry_id not in desired_order])
