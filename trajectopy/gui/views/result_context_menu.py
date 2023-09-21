"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QCursor
from trajectopy_core.util.entries import AbsoluteDeviationEntry, AlignmentEntry, RelativeDeviationEntry

from trajectopy.gui.managers.requests import (
    FileRequest,
    PlotRequest,
    PlotRequestType,
    ResultModelRequest,
    ResultModelRequestType,
    UIRequest,
    UIRequestType,
)
from trajectopy.gui.models.selection import ResultSelection


class ResultContextMenu(QtWidgets.QMenu):
    result_model_request = pyqtSignal(ResultModelRequest)
    ui_request = pyqtSignal(UIRequest)
    file_request = pyqtSignal(FileRequest)
    plot_request = pyqtSignal(PlotRequest)

    def __init__(self, parent) -> None:
        super().__init__(parent=parent)
        self.edit_context_menu = QtWidgets.QMenu("Edit")
        self._selection: ResultSelection

    def clear(self) -> None:
        super().clear()
        self.edit_context_menu.clear()

    def set_selection(self, selection: ResultSelection) -> None:
        self._selection = selection

    def get_selection(self) -> ResultSelection:
        return self._selection

    @pyqtSlot(ResultSelection)
    def show_context_menu(self, selection: ResultSelection):
        if not selection:
            return

        self.clear()
        self.set_selection(selection)

        self.property_context()
        self.edit_context()
        self.plot_context()
        self.exec(QCursor.pos())

    def property_context(self) -> None:
        property_action = QAction("Properties", self)
        property_action.triggered.connect(
            lambda: self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.RES_PROPERTIES,
                    result_selection=self.get_selection(),
                )
            )
        )
        self.addAction(property_action)

    def plot_context(self):
        single_selection = len(self.get_selection().entries) == 1

        if single_selection:
            return self.plot_single()

        # only useful case to plot multiple results is when they are all deviations
        if all(isinstance(entry, AbsoluteDeviationEntry) for entry in self.get_selection().entries):
            plot_action = QAction("Plot", self)
            plot_action.triggered.connect(
                lambda: self.plot_request.emit(
                    PlotRequest(
                        type=PlotRequestType.MULTI_ABS_DEVIATIONS,
                        result_selection=self.get_selection(),
                    )
                )
            )
            self.addAction(plot_action)

        if all(isinstance(entry, RelativeDeviationEntry) for entry in self.get_selection().entries):
            plot_action = QAction("Plot", self)
            plot_action.triggered.connect(
                lambda: self.plot_request.emit(
                    PlotRequest(
                        type=PlotRequestType.REL_DEVIATIONS,
                        result_selection=self.get_selection(),
                    )
                )
            )
            self.addAction(plot_action)

    def plot_single(self):
        plot_action = QAction("Plot", self)
        if isinstance(self.get_selection().entries[0], AbsoluteDeviationEntry):
            plot_action.triggered.connect(
                lambda: self.plot_request.emit(
                    PlotRequest(
                        type=PlotRequestType.SINGLE_ABS_DEVIATIONS,
                        result_selection=self.get_selection(),
                    )
                )
            )
            self.addAction(plot_action)

            plot_laps_action = QAction("Plot Laps", self)
            plot_laps_action.triggered.connect(
                lambda: self.plot_request.emit(
                    PlotRequest(
                        type=PlotRequestType.DEVIATION_LAPS,
                        result_selection=self.get_selection(),
                    )
                )
            )
            if self.get_selection().entries[0].deviations.trajectory.state.sorting_known:
                self.addAction(plot_laps_action)

        if isinstance(self.get_selection().entries[0], RelativeDeviationEntry):
            plot_action.triggered.connect(
                lambda: self.plot_request.emit(
                    PlotRequest(
                        type=PlotRequestType.REL_DEVIATIONS,
                        result_selection=self.get_selection(),
                    )
                )
            )
            self.addAction(plot_action)

        if isinstance(self.get_selection().entries[0], AlignmentEntry):
            plot_action.triggered.connect(
                lambda: self.plot_request.emit(
                    PlotRequest(
                        type=PlotRequestType.CORRELATION,
                        result_selection=self.get_selection(),
                    )
                )
            )
            self.addAction(plot_action)

    def edit_context(self) -> None:
        """Edit sub-context menu"""
        single_selection = len(self.get_selection()) == 1
        self.addMenu(self.edit_context_menu)

        self.edit_multi()

        if single_selection:
            self.edit_single()

    def edit_multi(self):
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(
            lambda: self.result_model_request.emit(
                ResultModelRequest(type=ResultModelRequestType.COPY, selection=self.get_selection())
            )
        )
        self.edit_context_menu.addAction(copy_action)

        remove_action = QAction("Remove", self)
        remove_action.triggered.connect(
            lambda: self.result_model_request.emit(
                ResultModelRequest(type=ResultModelRequestType.REMOVE, selection=self.get_selection())
            )
        )
        self.edit_context_menu.addAction(remove_action)

    def edit_single(self):
        if isinstance(alignment_entry := self.get_selection().entries[0], AlignmentEntry):
            edit_alignment_action = QAction("Edit Enabled Parameters", self)
            edit_alignment_action.triggered.connect(
                lambda: self.result_model_request.emit(
                    ResultModelRequest(
                        type=ResultModelRequestType.EDIT_ALIGNMENT,
                        alignment=alignment_entry,
                    )
                )
            )
            self.edit_context_menu.addAction(edit_alignment_action)

        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(
            lambda: self.result_model_request.emit(
                ResultModelRequest(type=ResultModelRequestType.RENAME, selection=self.get_selection())
            )
        )
        self.edit_context_menu.addAction(rename_action)

        export_action = QAction("Export", self)
        export_action.triggered.connect(
            lambda: self.ui_request.emit(
                UIRequest(type=UIRequestType.EXPORT_RES, result_selection=self.get_selection())
            )
        )
        self.edit_context_menu.addAction(export_action)
