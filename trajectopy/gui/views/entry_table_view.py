from typing import Any

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt, Signal

from trajectopy.gui.managers.requests import FileRequest, FileRequestType
from trajectopy.gui.models.table_model import ENTRY_IDS_MIME_TYPE
from trajectopy.gui.project import PROJECT_SUFFIX
from trajectopy.gui.views.menu_helpers import SelectionMenu


class EntryTableView(QtWidgets.QTreeView):
    """Shared behaviour of the trajectory and the result table: entries in collapsible groups, row selection (a
    selected group stands for all of its entries), file drops, moving entries between groups by drag and drop, the
    context menu (whose entries also handle the keyboard shortcuts) and a hint while the table is empty."""

    file_request = Signal(FileRequest)
    project_dropped = Signal(str)

    def __init__(
        self,
        model: Any,
        context_menu: SelectionMenu,
        read_request: FileRequestType,
        empty_hint: str,
        parent: QtWidgets.QWidget | None,
    ) -> None:
        super().__init__(parent)
        self.table_model = model
        self.setModel(model)
        self._read_request = read_request
        self._empty_hint = empty_hint
        # restored after the model changed its rows (it is reset for every change of the entries or groups)
        self._selected_ids: set[str] = set()
        self._selected_groups: set[str] = set()
        self._collapsed_groups: set[str] = set()
        model.modelAboutToBeReset.connect(self._save_state)
        model.modelReset.connect(self._restore_state)
        self.collapsed.connect(lambda index: self._track_expansion(index, collapsed=True))
        self.expanded.connect(lambda index: self._track_expansion(index, collapsed=False))

        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setAllColumnsShowFocus(True)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setSortingEnabled(False)
        # the name takes the free space; the other (short) columns fit their content
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)

        # the menu is a child of the view: its status tips reach the main window's status bar
        context_menu.setParent(self, context_menu.windowFlags())
        self._context_menu = context_menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda: context_menu.show_context_menu(self.selection))

    @property
    def selection(self) -> Any:
        raise NotImplementedError

    @property
    def selected_entries(self) -> list:
        """The selected entries (all entries of selected groups), in the order of the table."""
        if (selection_model := self.selectionModel()) is None:
            return []
        ids = {
            entry.entry_id for index in selection_model.selectedRows() for entry in self.table_model.entries_of(index)
        }
        return [entry for entry in self.table_model.items if entry.entry_id in ids]

    def select_entries(self, entries: list) -> None:
        """Selects the rows of the entries (only them)."""
        self._select({entry.entry_id for entry in entries}, set())

    # ------------------------------------------------------------------------------------- selection and groups

    def _track_expansion(self, index: Any, collapsed: bool) -> None:
        if (group_id := self.table_model.group_at(index)) is not None:
            (self._collapsed_groups.add if collapsed else self._collapsed_groups.discard)(group_id)

    def _save_state(self) -> None:
        self._selected_ids, self._selected_groups = set(), set()
        if (selection_model := self.selectionModel()) is None:
            return
        for index in selection_model.selectedRows():
            if (group_id := self.table_model.group_at(index)) is not None:
                self._selected_groups.add(group_id)
            elif (entry := self.table_model.entry_at(index)) is not None:
                self._selected_ids.add(entry.entry_id)

    def _restore_state(self) -> None:
        model = self.table_model
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            if (group_id := model.group_at(index)) is not None:
                # new groups are expanded, so that new entries are visible
                self.setExpanded(index, group_id not in self._collapsed_groups)
        self._select(self._selected_ids, self._selected_groups)

    def _select(self, entry_ids: set[str], group_ids: set[str]) -> None:
        if (selection_model := self.selectionModel()) is None:
            return
        model = self.table_model
        selection = QItemSelection()
        indexes = [model.group_index(group_id) for group_id in group_ids if group_id in model.groups()]
        indexes += [model.entry_index(entry_id) for entry_id in entry_ids]
        last_column = model.columnCount() - 1
        for index in indexes:
            if index.isValid():
                selection.select(index, model.index(index.row(), last_column, index.parent()))
        selection_model.select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    # ------------------------------------------------------------------------------------------- drag and drop

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if self._is_entry_move(event):
            super().dragEnterEvent(event)
        else:
            self._accept_files(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if self._is_entry_move(event):
            super().dragMoveEvent(event)
        else:
            self._accept_files(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if self._is_entry_move(event):
            super().dropEvent(event)
            return
        if self._accept_files(event):
            file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
            if projects := [path for path in file_paths if path.lower().endswith(PROJECT_SUFFIX)]:
                self.project_dropped.emit(projects[0])
            if files := [path for path in file_paths if path not in projects]:
                self.file_request.emit(FileRequest(type=self._read_request, file_list=files))

    def _is_entry_move(self, event: QtGui.QDropEvent) -> bool:
        """Entries dragged within this table (to move them between groups)."""
        mime_data = event.mimeData()
        return event.source() is self and mime_data is not None and mime_data.hasFormat(ENTRY_IDS_MIME_TYPE)

    @staticmethod
    def _accept_files(event: QtGui.QDropEvent) -> bool:
        if (mime_data := event.mimeData()) is None or not mime_data.hasUrls():
            event.ignore()
            return False
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        return True

    # --------------------------------------------------------------------------------------------------- other

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        """Shortcuts trigger the entries of the context menu, which shows them."""
        if not self._context_menu.handle_key_press(self.selection, e):
            super().keyPressEvent(e)

    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        super().paintEvent(e)
        if self.model().rowCount() > 0:
            return
        viewport = self.viewport()
        painter = QtGui.QPainter(viewport)
        color = self.palette().color(QtGui.QPalette.ColorRole.Text)
        color.setAlphaF(0.5)
        painter.setPen(color)
        flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap
        painter.drawText(viewport.rect().adjusted(12, 12, -12, -12), flags, self._empty_hint)
        painter.end()
