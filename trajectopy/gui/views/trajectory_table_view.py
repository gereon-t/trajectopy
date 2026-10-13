from PySide6 import QtWidgets
from PySide6.QtCore import Signal

from trajectopy.gui.managers.requests import (
    FileRequestType,
    PlotRequest,
    ResultModelRequest,
    TrajectoryManagerRequest,
    TrajectoryModelRequest,
    UIRequest,
)
from trajectopy.gui.models.entries import TrajectoryEntry
from trajectopy.gui.models.selection import TrajectorySelection
from trajectopy.gui.models.trajectory_model import TrajectoryTableModel
from trajectopy.gui.views.entry_table_view import EntryTableView
from trajectopy.gui.views.trajectory_context_menu import TrajectoryContextMenu


class TrajectoryTableView(EntryTableView):
    """Table of the loaded and computed trajectories. All operations are in the context menu."""

    trajectory_model_request = Signal(TrajectoryModelRequest)
    trajectory_manager_request = Signal(TrajectoryManagerRequest)
    result_model_request = Signal(ResultModelRequest)
    ui_request = Signal(UIRequest)
    plot_request = Signal(PlotRequest)
    show_timeline = Signal(object)  # emits list[TrajectoryEntry]

    def __init__(self, trajectory_table_model: TrajectoryTableModel, parent: QtWidgets.QWidget | None) -> None:
        self.trajectory_context_menu = TrajectoryContextMenu()
        super().__init__(
            model=trajectory_table_model,
            context_menu=self.trajectory_context_menu,
            read_request=FileRequestType.READ_TRAJ,
            empty_hint="No trajectories yet",
            parent=parent,
        )
        self.trajectory_table_model = trajectory_table_model
        self.setObjectName("trajectoryTableView")

        menu = self.trajectory_context_menu
        menu.trajectory_model_request.connect(self.trajectory_model_request.emit)
        menu.trajectory_manager_request.connect(self.trajectory_manager_request.emit)
        menu.ui_request.connect(self.ui_request.emit)
        menu.plot_request.connect(self.plot_request.emit)
        menu.result_model_request.connect(self.result_model_request.emit)
        menu.show_timeline.connect(self.show_timeline.emit)

    @property
    def selection(self) -> TrajectorySelection:
        return TrajectorySelection(
            entries=self.selected_entries,
            reference_entry=self.trajectory_table_model.reference_entry,
        )

    @property
    def selected_entries(self) -> list[TrajectoryEntry]:
        return super().selected_entries
