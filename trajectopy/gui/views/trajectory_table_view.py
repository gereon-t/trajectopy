"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging
from typing import List, Optional, Union

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal
from trajectopy_core.util.entries import TrajectoryEntry

from trajectopy.gui.managers.requests import (
    FileRequest,
    FileRequestType,
    PlotRequest,
    PlotRequestType,
    ResultModelRequest,
    TrajectoryManagerRequest,
    TrajectoryManagerRequestType,
    TrajectoryModelRequest,
    TrajectoryModelRequestType,
    UIRequest,
    UIRequestType,
)
from trajectopy.gui.models.selection import TrajectorySelection
from trajectopy.gui.models.trajectory_model import TrajectoryTableModel
from trajectopy.gui.views.trajectory_context_menu import TrajectoryContextMenu

logger = logging.getLogger("root")


class TrajectoryTableView(QtWidgets.QTableView):
    """Class representing the trajectory table

    This table displays the loaded and computed
    trajectories. It is possible to perform
    computations on these trajectories using
    the right click.
    """

    trajectory_model_request = pyqtSignal(TrajectoryModelRequest)
    trajectory_manager_request = pyqtSignal(TrajectoryManagerRequest)
    result_model_request = pyqtSignal(ResultModelRequest)
    ui_request = pyqtSignal(UIRequest)
    file_request = pyqtSignal(FileRequest)
    plot_request = pyqtSignal(PlotRequest)

    def __init__(
        self,
        trajectory_table_model: TrajectoryTableModel,
        parent: Optional[QtWidgets.QWidget],
    ) -> None:
        QtWidgets.QTableView.__init__(self, parent)

        self.trajectory_table_model = trajectory_table_model
        self.setModel(self.trajectory_table_model)

        self.setAlternatingRowColors(True)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setSortingEnabled(False)
        self.setObjectName("trajectoryTableView")

        self.trajectory_context_menu = TrajectoryContextMenu(parent=self)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda: self.trajectory_context_menu.show_context_menu(self.selection))

        # forward signals
        self.trajectory_context_menu.trajectory_model_request.connect(self.trajectory_model_request.emit)
        self.trajectory_context_menu.trajectory_manager_request.connect(self.trajectory_manager_request.emit)
        self.trajectory_context_menu.ui_request.connect(self.ui_request.emit)
        self.trajectory_context_menu.plot_request.connect(self.plot_request.emit)
        self.trajectory_context_menu.result_model_request.connect(self.result_model_request.emit)

    @property
    def selection(self) -> TrajectorySelection:
        return TrajectorySelection(
            entries=self.selected_entries,
            reference_entry=self.trajectory_table_model.reference_entry,
        )

    @property
    def selected_entries(self) -> List[TrajectoryEntry]:
        if (selection_model := self.selectionModel()) is None:
            return []

        return [
            self.trajectory_table_model.items[row_index]
            for row_index in [row.row() for row in selection_model.selectedRows()]
        ]

    def dragEnterEvent(self, event: Union[QtGui.QDragEnterEvent, None]):
        """Drag enter event handler."""
        if event is None:
            return

        if (mime_data := event.mimeData()) is None:
            return

        if mime_data.hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event: Union[QtGui.QDragMoveEvent, None]):
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

    def dropEvent(self, event: Union[QtGui.QDropEvent, None]):
        """Drop event handler."""
        if event is None:
            return

        if (mime_data := event.mimeData()) is None:
            return

        if mime_data.hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            file_paths = [url.toLocalFile() for url in mime_data.urls()]
            self.file_request.emit(FileRequest(type=FileRequestType.READ_TRAJ, file_list=file_paths))
        else:
            event.ignore()

    def keyPressEvent(self, e: Union[QtGui.QKeyEvent, None]) -> None:
        """Key press event handler."""
        if e is None:
            return

        if not self.selection.entries:
            return super().keyPressEvent(e)

        if e.key() == Qt.Key.Key_Delete:
            self.trajectory_model_request.emit(
                TrajectoryModelRequest(type=TrajectoryModelRequestType.REMOVE, selection=self.selection)
            )

        if e.key() == Qt.Key.Key_T:
            self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.EPSG_SELECTION,
                    trajectory_selection=self.selection,
                )
            )

        if e.key() == Qt.Key.Key_C and e.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.trajectory_model_request.emit(
                TrajectoryModelRequest(type=TrajectoryModelRequestType.COPY, selection=self.selection)
            )

        if e.key() == Qt.Key.Key_P:
            self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.TRAJ_PROPERTIES,
                    trajectory_selection=self.selection,
                )
            )

        if e.key() == Qt.Key.Key_V:
            self.plot_request.emit(
                PlotRequest(
                    type=PlotRequestType.TRAJECTORIES,
                    trajectory_selection=self.selection,
                    dimension=2,
                )
            )

        if e.key() == Qt.Key.Key_M:
            self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(type=TrajectoryManagerRequestType.MERGE, selection=self.selection)
            )

        if len(self.selected_entries) > 1:
            return

        if e.key() == Qt.Key.Key_R:
            self.trajectory_model_request.emit(
                TrajectoryModelRequest(
                    type=TrajectoryModelRequestType.SET_REFERENCE,
                    selection=self.selection,
                )
            )

        if e.key() == Qt.Key.Key_R and e.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            self.trajectory_model_request.emit(TrajectoryModelRequest(type=TrajectoryModelRequestType.UNSET_REFERENCE))

        if e.key() == Qt.Key.Key_S:
            self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.TRAJ_SETTINGS,
                    trajectory_selection=self.selection,
                )
            )

        if e.key() == Qt.Key.Key_U:
            self.trajectory_model_request.emit(
                TrajectoryModelRequest(type=TrajectoryModelRequestType.RENAME, selection=self.selection)
            )

        if e.key() == Qt.Key.Key_E:
            self.ui_request.emit(UIRequest(type=UIRequestType.EXPORT_TRAJ, trajectory_selection=self.selection))
