from typing import Any

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from trajectopy.gui.managers.requests import FileRequest, FileRequestType
from trajectopy.gui.views.menu_helpers import SelectionMenu


class EntryTableView(QtWidgets.QTableView):
    """Shared behaviour of the trajectory and the result table: row selection, file drops, the context menu
    (whose entries also handle the keyboard shortcuts) and a hint while the table is empty."""

    file_request = Signal(FileRequest)

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

        self.setAlternatingRowColors(True)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setSortingEnabled(False)
        self.verticalHeader().setVisible(False)

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
        if (selection_model := self.selectionModel()) is None:
            return []
        return [self.table_model.items[index.row()] for index in selection_model.selectedRows()]

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        self._accept_files(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        self._accept_files(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if self._accept_files(event):
            file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
            self.file_request.emit(FileRequest(type=self._read_request, file_list=file_paths))

    @staticmethod
    def _accept_files(event: QtGui.QDropEvent) -> bool:
        if (mime_data := event.mimeData()) is None or not mime_data.hasUrls():
            event.ignore()
            return False
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        return True

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
