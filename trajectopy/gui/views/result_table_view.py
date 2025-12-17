import logging

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

from trajectopy.gui.managers.requests import (
    FileRequest,
    FileRequestType,
    PlotRequest,
    ResultModelRequest,
    ResultModelRequestType,
    UIRequest,
    UIRequestType,
)
from trajectopy.gui.models.entries import ResultEntry
from trajectopy.gui.models.result_model import ResultTableModel
from trajectopy.gui.models.selection import ResultSelection
from trajectopy.gui.views.result_context_menu import ResultContextMenu

logger = logging.getLogger(__name__)


class ResultTableView(QtWidgets.QTableView):
    """Class representing the result table

    This table displays the loaded and computed
    results. It is possible to perform
    actions on these results using
    the right click.

    """

    result_model_request = pyqtSignal(ResultModelRequest)
    ui_request = pyqtSignal(UIRequest)
    file_request = pyqtSignal(FileRequest)
    plot_request = pyqtSignal(PlotRequest)

    def __init__(
        self,
        result_table_model: ResultTableModel,
        parent: QtWidgets.QWidget | None,
    ) -> None:
        QtWidgets.QTableView.__init__(self, parent)

        self.result_table_model = result_table_model
        self.setModel(self.result_table_model)

        self.setAlternatingRowColors(False)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setSortingEnabled(False)
        self.setObjectName("resultTableView")

        if (header := self.horizontalHeader()) is not None:
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.result_context_menu = ResultContextMenu(parent=self)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda: self.result_context_menu.show_context_menu(self.selection))

        # forward signals
        self.result_context_menu.result_model_request.connect(self.result_model_request.emit)
        self.result_context_menu.ui_request.connect(self.ui_request.emit)
        self.result_context_menu.file_request.connect(self.file_request.emit)
        self.result_context_menu.plot_request.connect(self.plot_request.emit)

    @property
    def selection(self) -> ResultSelection:
        return ResultSelection(entries=self.selected_entries)

    @property
    def selected_entries(self) -> list[ResultEntry]:
        if (selection_model := self.selectionModel()) is None:
            return []

        return [
            self.result_table_model.items[row_index]
            for row_index in [row.row() for row in selection_model.selectedRows()]
        ]

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent | None):
        """Drag enter event handler."""
        if event is None:
            return

        if (mime_data := event.mimeData()) is None:
            return

        if mime_data.hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent | None):
        """Drag move event handler."""
        if event is None:
            return

        if (mime_data := event.mimeData()) is None:
            return

        if mime_data.hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent | None):
        """Drop event handler."""
        if event is None:
            return

        if (mime_data := event.mimeData()) is None:
            return

        if mime_data.hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            file_paths = [url.toLocalFile() for url in mime_data.urls()]
            self.file_request.emit(FileRequest(type=FileRequestType.READ_RES, file_list=file_paths))
        else:
            event.ignore()

    def keyPressEvent(self, e: QtGui.QKeyEvent | None) -> None:
        """Key press event handler."""
        if e is None:
            return

        if not self.selection.entries:
            return super().keyPressEvent(e)

        if e.key() == Qt.Key.Key_Delete:
            self.result_model_request.emit(
                ResultModelRequest(type=ResultModelRequestType.REMOVE, selection=self.selection)
            )

        if e.key() == Qt.Key.Key_C and e.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.result_model_request.emit(
                ResultModelRequest(type=ResultModelRequestType.COPY, selection=self.selection)
            )

        if e.key() == Qt.Key.Key_P:
            self.ui_request.emit(UIRequest(type=UIRequestType.RES_PROPERTIES, result_selection=self.selection))

        if len(self.selected_entries) > 1:
            return

        if e.key() == Qt.Key.Key_U:
            self.result_model_request.emit(
                ResultModelRequest(type=ResultModelRequestType.RENAME, selection=self.selection)
            )

        if e.key() == Qt.Key.Key_E:
            self.ui_request.emit(UIRequest(type=UIRequestType.EXPORT_RES, result_selection=self.selection))
