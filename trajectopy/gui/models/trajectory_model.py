"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import copy
import logging
import os
from typing import List, Union

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QInputDialog
from trajectopy_core.util.entries import TrajectoryEntry

from trajectopy.gui.managers.requests import (
    FileRequest,
    FileRequestType,
    TrajectoryModelRequest,
    TrajectoryModelRequestType,
)
from trajectopy.gui.models.selection import TrajectorySelection
from trajectopy.gui.models.table_model import RequestTableModel

logger = logging.getLogger("root")


class TrajectoryTableModel(RequestTableModel):
    """Class representing the trajectory table model."""

    file_request = pyqtSignal(FileRequest)
    items: List[TrajectoryEntry]
    request: TrajectoryModelRequest

    def __init__(self):
        REQUEST_MAPPING = {
            TrajectoryModelRequestType.ADD: self.add_entries,
            TrajectoryModelRequestType.UPDATE: self.update_selected_entries,
            TrajectoryModelRequestType.RENAME: self.rename_trajectories,
            TrajectoryModelRequestType.COPY: self.copy_trajectories,
            TrajectoryModelRequestType.REMOVE: self.remove_selected_entries,
            TrajectoryModelRequestType.REMOVE_RELATED: self.remove_related_trajectories,
            TrajectoryModelRequestType.SET_REFERENCE: self.set_as_reference,
            TrajectoryModelRequestType.UNSET_REFERENCE: self.unset_as_reference,
            TrajectoryModelRequestType.RESET: self.reset,
            TrajectoryModelRequestType.EXPORT_ALL: self.export_all,
            TrajectoryModelRequestType.SORT: self.sort_items,
        }

        super().__init__(
            headers=[
                "Name",
                "Reference",
                "Sorting",
                "EPSG",
                "States",
                "Filename",
            ],
            REQUEST_MAPPING=REQUEST_MAPPING,
        )

    def remove_group(self, group_id: str) -> None:
        self.items = [item for item in self.items if item.group_id not in group_id]
        self.layoutChanged.emit()

    def export_all(self) -> None:
        for item in self.items:
            trajectory_filename = os.path.join(self.request.file_path, f"{item.entry_id}.traj")
            settings_filename = os.path.join(self.request.file_path, f"{item.entry_id}.yaml")
            self.file_request.emit(
                FileRequest(
                    type=FileRequestType.WRITE_TRAJ,
                    trajectory_selection=TrajectorySelection(entries=[item]),
                    file_list=[trajectory_filename],
                )
            )
            item.settings.to_file(settings_filename)

        self.write_trajectory_order()

    def write_trajectory_order(self):
        trajectory_order_filename = os.path.join(self.request.file_path, "trajectory_order.txt")
        self.file_request.emit(
            FileRequest(
                type=FileRequestType.WRITE_LIST,
                id_list=[item.entry_id for item in self.items],
                file_list=[trajectory_order_filename],
            )
        )

    @property
    def reference_entry(self) -> Union[TrajectoryEntry, None]:
        return next((item for item in self.items if item.set_as_reference), None)

    def rename_trajectories(self) -> None:
        if not isinstance((selection := self.request.selection), TrajectorySelection):
            raise ValueError("No trajectory selection provided")

        input_name, ok = QInputDialog.getText(None, "Please enter a name", "Name:", text=selection.entries[0].name)
        if ok and input_name is not None:
            selection.entries[0].trajectory.name = input_name
        self.layoutChanged.emit()

    def remove_related_trajectories(self) -> None:
        """Remove all trajectories that are related"""
        if not isinstance((selection := self.request.selection), TrajectorySelection):
            raise ValueError("No trajectory selection provided")

        for entry in selection.entries:
            self.remove_group(entry.group_id)

    def copy_trajectories(self) -> None:
        """Deep copy trajectory"""
        if not isinstance((selection := self.request.selection), TrajectorySelection):
            raise ValueError("No trajectory selection provided")

        for entry in selection.entries:
            entry_copy = copy.deepcopy(entry)
            entry_copy.renew_id()
            entry_copy.set_as_reference = False
            self.add(entry_copy)

    def set_as_reference(self) -> None:
        """Marks selected trajectory as reference"""
        if not isinstance((selection := self.request.selection), TrajectorySelection):
            raise ValueError("No trajectory selection provided")

        selected_id = selection.entries[0].entry_id
        for entry in self.items:
            entry.set_as_reference = entry.entry_id == selected_id
        self.layoutChanged.emit()

    def unset_as_reference(self) -> None:
        """Removes the reference status from the selected trajectory"""
        for entry in self.items:
            entry.set_as_reference = False
        self.layoutChanged.emit()
