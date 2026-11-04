import copy
import logging
from collections.abc import Callable, Collection, Mapping
from typing import Any, Generic, TypeVar

from PySide6.QtCore import QAbstractItemModel, QMimeData, QModelIndex, QPersistentModelIndex, Qt, Signal, Slot
from PySide6.QtGui import QFont

from trajectopy.gui.managers.requests import (
    Request,
    ResultModelRequest,
    TrajectoryModelRequest,
    UIRequest,
    generic_request_handler,
)
from trajectopy.gui.models.entries import Entry, generate_id

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Entry)

_ROOT_INDEX = QModelIndex()


class BaseTableModel(QAbstractItemModel, Generic[T]):
    """Base class for all table models: one row per item."""

    def __init__(self, headers: list[str] | None = None):
        super().__init__()
        self.items: list[T] = []
        self._headers = [""] if headers is None else headers

    def index(self, row: int, column: int, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX) -> QModelIndex:
        if parent.isValid() or not 0 <= row < len(self.items) or not 0 <= column < self.columnCount():
            return QModelIndex()
        return self.createIndex(row, column, 0)

    def parent(self, *args: Any) -> Any:
        if not args:  # QObject.parent()
            return super().parent()
        return QModelIndex()

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX) -> int:
        return 0 if parent.isValid() else len(self.items)

    def entry_at(self, index: QModelIndex | QPersistentModelIndex) -> T | None:
        """The item shown in the row of the index (None for rows without one, e.g. group rows)."""
        return self.items[index.row()] if index.isValid() and 0 <= index.row() < len(self.items) else None

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX) -> int:
        return len(self._headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and (item := self.entry_at(index)) is not None:
            return item.column[index.column()]

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
        # entries are addressed by their id (update, removal), so it must be unique, e.g. if a file is opened twice
        if self.get(entry.entry_id) is not None:
            entry.renew_id()
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


ENTRY_IDS_MIME_TYPE = "application/x-trajectopy-entry-ids"


class GroupedTableModel(RequestTableModel):
    """Table of entries in groups of one level: each group with more than one entry (or with a name) is shown as
    a row whose children are its entries; the other entries are shown at the top level.

    The entries are still kept as a flat list (``items``, in the order shown); their ``group_id`` and
    ``group_name`` define the groups. Rows of the top level have the internal id 0, entries of a group the row of
    their group + 1.
    """

    def __init__(
        self,
        headers: list[str],
        REQUEST_MAPPING: Mapping[Any, Callable[[Any], None]] | None = None,
    ):
        self._nodes: list[tuple[str | None, list]] = []  # (group id, or None for a single entry; entries)
        super().__init__(headers, REQUEST_MAPPING)

    # ------------------------------------------------------------------------------------------------- structure

    def _rebuild(self) -> None:
        sizes: dict[str, int] = {}
        named: set[str] = set()
        for item in self.items:
            sizes[item.group_id] = sizes.get(item.group_id, 0) + 1
            if item.group_name:
                named.add(item.group_id)
        nodes: list[tuple[str | None, list]] = []
        positions: dict[str, int] = {}
        for item in self.items:
            group_id = item.group_id
            if sizes[group_id] < 2 and group_id not in named:
                nodes.append((None, [item]))
            elif group_id in positions:
                nodes[positions[group_id]][1].append(item)
            else:
                positions[group_id] = len(nodes)
                nodes.append((group_id, [item]))
        self._nodes = nodes

    def _changing_structure(self, change: Callable[[], None]) -> None:
        """Applies a change of the entries or groups; the views restore their selection and collapsed groups."""
        self.beginResetModel()
        change()
        self._rebuild()
        self.endResetModel()

    def index(self, row: int, column: int, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX) -> QModelIndex:
        if not 0 <= column < self.columnCount():
            return QModelIndex()
        if not parent.isValid():
            return self.createIndex(row, column, 0) if 0 <= row < len(self._nodes) else QModelIndex()
        if parent.internalId() != 0 or not 0 <= parent.row() < len(self._nodes):
            return QModelIndex()
        group_id, entries = self._nodes[parent.row()]
        if group_id is None or not 0 <= row < len(entries):
            return QModelIndex()
        return self.createIndex(row, column, parent.row() + 1)

    def parent(self, *args: Any) -> Any:
        if not args:  # QObject.parent()
            return super().parent()
        child = args[0]
        if not child.isValid() or child.internalId() == 0:
            return QModelIndex()
        return self.createIndex(int(child.internalId()) - 1, 0, 0)

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT_INDEX) -> int:
        if not parent.isValid():
            return len(self._nodes)
        if parent.internalId() != 0 or parent.column() != 0 or not 0 <= parent.row() < len(self._nodes):
            return 0
        group_id, entries = self._nodes[parent.row()]
        return len(entries) if group_id is not None else 0

    def entry_at(self, index: QModelIndex | QPersistentModelIndex) -> Any:
        if not index.isValid():
            return None
        if index.internalId() == 0:
            if not 0 <= index.row() < len(self._nodes):
                return None
            group_id, entries = self._nodes[index.row()]
            return entries[0] if group_id is None else None
        node = int(index.internalId()) - 1
        if not 0 <= node < len(self._nodes) or not 0 <= index.row() < len(self._nodes[node][1]):
            return None
        return self._nodes[node][1][index.row()]

    def group_at(self, index: QModelIndex | QPersistentModelIndex) -> str | None:
        """The group id of a group row (None for rows of entries)."""
        if not index.isValid() or index.internalId() != 0 or not 0 <= index.row() < len(self._nodes):
            return None
        return self._nodes[index.row()][0]

    def entries_of(self, index: QModelIndex | QPersistentModelIndex) -> list:
        """The entries of a row: all entries of a group row, otherwise the entry of the row."""
        if self.group_at(index) is not None:
            return list(self._nodes[index.row()][1])
        entry = self.entry_at(index)
        return [entry] if entry is not None else []

    def group_index(self, group_id: str) -> QModelIndex:
        row = next((row for row, (node_group, _) in enumerate(self._nodes) if node_group == group_id), None)
        return self.createIndex(row, 0, 0) if row is not None else QModelIndex()

    def entry_index(self, entry_id: str) -> QModelIndex:
        for row, (group_id, entries) in enumerate(self._nodes):
            for child_row, entry in enumerate(entries):
                if entry.entry_id == entry_id:
                    if group_id is None:
                        return self.createIndex(row, 0, 0)
                    return self.createIndex(child_row, 0, row + 1)
        return QModelIndex()

    def groups(self) -> dict[str, str]:
        """The shown groups: their names by group id."""
        return {group_id: self.group_name(entries) for group_id, entries in self._nodes if group_id is not None}

    def group_name(self, entries: list) -> str:
        """Name of the group of the entries: the name it was given, otherwise the default name."""
        given = next((entry.group_name for entry in entries if entry.group_name), "")
        return given or self.default_group_name(entries)

    def default_group_name(self, entries: list) -> str:
        return "Group"

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if self.group_at(index) is None:
            return super().data(index, role)
        entries = self._nodes[index.row()][1]
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{self.group_name(entries)} ({len(entries)})" if index.column() == 0 else None
        if role == Qt.ItemDataRole.FontRole:
            font = QFont()
            font.setBold(True)
            return font
        return None

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.ItemIsDropEnabled  # dropped between the rows of the top level: out of the groups
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDragEnabled
            | Qt.ItemFlag.ItemIsDropEnabled
        )

    # ---------------------------------------------------------------------------------------- changes of entries

    def set(self, item_id: str, entry: T) -> None:
        row = next((row for row, item in enumerate(self.items) if item.entry_id == item_id), None)
        if row is None:
            return
        old: Any = self.items[row]
        new: Any = entry
        if (old.group_id, old.group_name) != (new.group_id, new.group_name):
            self._changing_structure(lambda: self.items.__setitem__(row, entry))
            return
        self.items[row] = entry
        self._rebuild()
        self.refresh()

    def add(self, entry: T) -> None:
        # entries are addressed by their id (update, removal), so it must be unique, e.g. if a file is opened twice
        if self.get(entry.entry_id) is not None:
            entry.renew_id()
        # after the last entry of its group (e.g. an aligned copy of a trajectory), so that the order of the
        # entries is the shown one
        group_id = getattr(entry, "group_id", None)
        position = next(
            (
                row + 1
                for row in range(len(self.items) - 1, -1, -1)
                if getattr(self.items[row], "group_id", None) == group_id
            ),
            len(self.items),
        )
        self._changing_structure(lambda: self.items.insert(position, entry))

    def remove(self, ids: Collection[str]) -> None:
        if any(item.entry_id in ids for item in self.items):
            self._changing_structure(
                lambda: self._set_items([item for item in self.items if item.entry_id not in ids])
            )

    def clear(self) -> None:
        self._changing_structure(lambda: self._set_items([]))

    def reorder(self, items: list[T]) -> None:
        self._changing_structure(lambda: self._set_items(items))

    def _set_items(self, items: list[T]) -> None:
        self.items = items

    def refresh(self) -> None:
        """Notifies the views that the shown values of (any of) the entries or groups changed."""
        if not self._nodes:
            return
        last_column = self.columnCount() - 1
        self.dataChanged.emit(self.index(0, 0), self.index(len(self._nodes) - 1, last_column))
        for row, (group_id, entries) in enumerate(self._nodes):
            if group_id is not None:
                parent = self.index(row, 0)
                self.dataChanged.emit(self.index(0, 0, parent), self.index(len(entries) - 1, last_column, parent))

    # --------------------------------------------------------------------------------------------------- groups

    def group_entries(self, ids: Collection[str], name: str) -> None:
        """Puts the entries into a new group with the given name, at the position of the first of them."""
        group_id = generate_id()

        def change() -> None:
            moved = [item for item in self.items if item.entry_id in ids]
            for item in moved:
                item.group_id, item.group_name = group_id, name
            if moved:
                self.items = _moved_after(self.items, moved[1:], moved[0])

        self._changing_structure(change)

    def ungroup_entries(self, ids: Collection[str]) -> None:
        """Takes the entries out of their groups (each one then forms a group of its own)."""

        def change() -> None:
            for item in self.items:
                if item.entry_id in ids:
                    item.group_id, item.group_name = generate_id(), ""

        self._changing_structure(change)

    def rename_groups(self, group_ids: Collection[str], name: str) -> None:
        def change() -> None:
            for item in self.items:
                if item.group_id in group_ids:
                    item.group_name = name

        self._changing_structure(change)

    def move_to_group(self, ids: Collection[str], group_id: str) -> None:
        """Moves the entries into an existing group (after its last entry)."""
        members = [item for item in self.items if item.group_id == group_id and item.entry_id not in ids]
        if not members:
            return
        name = next((item.group_name for item in members if item.group_name), "")

        def change() -> None:
            moved = [item for item in self.items if item.entry_id in ids]
            for item in moved:
                item.group_id, item.group_name = group_id, name
            self.items = _moved_after(self.items, moved, members[-1])

        self._changing_structure(change)

    def group_selected_entries(self, request) -> None:
        self.group_entries({entry.entry_id for entry in request.selection.entries}, request.name)

    def ungroup_selected_entries(self, request) -> None:
        self.ungroup_entries({entry.entry_id for entry in request.selection.entries})

    def rename_selected_groups(self, request) -> None:
        if request.name:
            self.rename_groups({entry.group_id for entry in request.selection.entries}, request.name)

    def move_selected_entries_to_group(self, request) -> None:
        self.move_to_group({entry.entry_id for entry in request.selection.entries}, request.group_id)

    # ------------------------------------------------------------------------------------------- drag and drop

    def supportedDropActions(self) -> Qt.DropAction:
        return Qt.DropAction.MoveAction

    def mimeTypes(self) -> list[str]:
        return [ENTRY_IDS_MIME_TYPE]

    def mimeData(self, indexes: Any) -> QMimeData:
        ids: list[str] = []
        for index in indexes:
            for entry in self.entries_of(index):
                if entry.entry_id not in ids:
                    ids.append(entry.entry_id)
        data = QMimeData()
        data.setData(ENTRY_IDS_MIME_TYPE, "\n".join(ids).encode())
        return data

    def dropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        row: int,
        column: int,
        parent: QModelIndex | QPersistentModelIndex,
    ) -> bool:
        """Dropped onto a group or one of its entries: into that group; onto an entry outside of groups: into a
        new group with that entry; between the rows of the top level: out of their groups."""
        if not data.hasFormat(ENTRY_IDS_MIME_TYPE):
            return False
        text = bytes(data.data(ENTRY_IDS_MIME_TYPE).data()).decode()
        ids = {entry_id for entry_id in text.split("\n") if entry_id}
        if not ids:
            return False
        if not parent.isValid():
            self.ungroup_entries(ids)
            return True
        group_id = self.group_at(parent)
        if group_id is None:
            target = self.entry_at(parent)
            if target is None or target.entry_id in ids:
                return False
            group_id = target.group_id
        # the entries are moved here; the view does not remove them, since removeRows is not supported
        self.move_to_group(ids, group_id)
        return True


def _moved_after(items: list, moved: list, anchor: Any) -> list:
    """The items with the moved ones placed right after the anchor (keeping their order)."""
    moved_ids = {id(item) for item in moved if item is not anchor}
    rest = [item for item in items if id(item) not in moved_ids]
    position = rest.index(anchor) + 1
    return rest[:position] + [item for item in items if id(item) in moved_ids] + rest[position:]
