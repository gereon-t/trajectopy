from PySide6 import QtWidgets
from PySide6.QtCore import Signal

from trajectopy.gui.managers.requests import FileRequestType, PlotRequest, ResultModelRequest, UIRequest
from trajectopy.gui.models.entries import ResultEntry
from trajectopy.gui.models.result_model import ResultTableModel
from trajectopy.gui.models.selection import ResultSelection
from trajectopy.gui.views.entry_table_view import EntryTableView
from trajectopy.gui.views.result_context_menu import ResultContextMenu


class ResultTableView(EntryTableView):
    """Table of the loaded and computed results. All operations are in the context menu."""

    result_model_request = Signal(ResultModelRequest)
    ui_request = Signal(UIRequest)
    plot_request = Signal(PlotRequest)

    def __init__(self, result_table_model: ResultTableModel, parent: QtWidgets.QWidget | None) -> None:
        self.result_context_menu = ResultContextMenu()
        super().__init__(
            model=result_table_model,
            context_menu=self.result_context_menu,
            read_request=FileRequestType.READ_RES,
            empty_hint="No results yet.\nRight-click a result for all actions.",
            parent=parent,
        )
        self.result_table_model = result_table_model
        self.setObjectName("resultTableView")
        if (header := self.horizontalHeader()) is not None:
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
            header.setStretchLastSection(True)

        menu = self.result_context_menu
        menu.result_model_request.connect(self.result_model_request.emit)
        menu.ui_request.connect(self.ui_request.emit)
        menu.file_request.connect(self.file_request.emit)
        menu.plot_request.connect(self.plot_request.emit)

    @property
    def selection(self) -> ResultSelection:
        return ResultSelection(entries=self.selected_entries)

    @property
    def selected_entries(self) -> list[ResultEntry]:
        return super().selected_entries
