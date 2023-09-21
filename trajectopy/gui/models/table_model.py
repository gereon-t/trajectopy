"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import copy
import logging
from enum import Enum
from typing import Callable, Dict, Generic, List, TypeVar, Union

from PyQt6.QtCore import QAbstractTableModel, Qt, QVariant, pyqtSignal, pyqtSlot
from trajectopy_core.util.entries import Entry

from trajectopy.gui.managers.requests import (
    Request,
    ResultModelRequest,
    TrajectoryModelRequest,
    UIRequest,
    generic_request_handler,
)

logger = logging.getLogger("root")

T = TypeVar("T", bound=Entry)


class BaseTableModel(QAbstractTableModel, Generic[T]):
    """Base class for all table models."""

    def __init__(self, headers: Union[List[str], None] = None):
        super().__init__()
        self.items: List[T] = []
        self._headers = [""] if headers is None else headers

    def rowCount(self, _):
        return len(self.items)

    def columnCount(self, _):
        return len(self._headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]

    def data(self, index, role):
        if not self.items:
            return

        if role == Qt.ItemDataRole.DisplayRole:
            return self.items[index.row()].column[index.column()]

        return QVariant()

    def get(self, item_id: str) -> Union[None, T]:
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

    def remove(self, ids: List[str]) -> None:
        self.items = [item for item in self.items if item.entry_id not in ids]
        self.layoutChanged.emit()


class RequestTableModel(BaseTableModel):
    """Base class for all table models that handle requests."""

    operation_finished = pyqtSignal()
    ui_request = pyqtSignal(UIRequest)

    def __init__(
        self,
        headers: List[str],
        REQUEST_MAPPING: Union[None, Dict[Enum, Callable]] = None,
    ):
        if REQUEST_MAPPING is None:
            REQUEST_MAPPING = {}
        super().__init__(headers)
        self.REQUEST_MAPPING = REQUEST_MAPPING
        self.request: Union[TrajectoryModelRequest, ResultModelRequest]

    @pyqtSlot(Request)
    def handle_request(self, request: Union[TrajectoryModelRequest, ResultModelRequest]) -> None:
        self.request = request
        generic_request_handler(self, request, passthrough_request=False)

    def add_entries(self) -> None:
        for entry in self.request.selection.entries:
            self.add(entry)

    def update_selected_entries(self) -> None:
        for entry in self.request.selection.entries:
            self.set(entry.entry_id, entry)

    def remove_selected_entries(self) -> None:
        """Remove selected entries"""
        self.remove([entry.entry_id for entry in self.request.selection.entries])

    def copy_selected_entries(self) -> None:
        """Deep copy entries"""
        for entry in self.request.selection.entries:
            entry_copy = copy.deepcopy(entry)
            entry_copy.renew_id()
            self.add(entry_copy)

    def reset(self) -> None:
        self.items = []
        self.layoutChanged.emit()

    def sort_items(self) -> None:
        """Sorts the items so that their id order matches the desired order"""
        desired_order = self.request.index_list
        present_ids = [item.entry_id for item in self.items]
        if any(id not in present_ids for id in desired_order):
            raise ValueError("Desired order contains ids that are not present in the model.")

        self.items = [entry for id in desired_order if (entry := self.get(id)) is not None]
        self.layoutChanged.emit()
