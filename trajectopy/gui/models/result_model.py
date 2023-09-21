"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging
import os
from typing import List

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QInputDialog
from trajectopy_core.util.entries import AlignmentEntry, ResultEntry

from trajectopy.gui.managers.requests import (
    FileRequest,
    FileRequestType,
    ResultModelRequest,
    ResultModelRequestType,
    UIRequest,
    UIRequestType,
)
from trajectopy.gui.models.selection import ResultSelection
from trajectopy.gui.models.table_model import RequestTableModel

logger = logging.getLogger("root")


class ResultTableModel(RequestTableModel):
    """Class representing the result table model."""

    file_request = pyqtSignal(FileRequest)
    items: List[ResultEntry]
    request: ResultModelRequest

    def __init__(self):
        REQUEST_MAPPING = {
            ResultModelRequestType.ADD: self.add_entries,
            ResultModelRequestType.UPDATE: self.update_selected_entries,
            ResultModelRequestType.RENAME: self.rename_result,
            ResultModelRequestType.REMOVE: self.remove_selected_entries,
            ResultModelRequestType.PASS_ALIGNMENTS_TO_UI: self.pass_alignments_to_ui,
            ResultModelRequestType.RESET: self.reset,
            ResultModelRequestType.EXPORT_ALL: self.export_all,
            ResultModelRequestType.EDIT_ALIGNMENT: self.edit_alignment,
            ResultModelRequestType.COPY: self.copy_selected_entries,
            ResultModelRequestType.SORT: self.sort_items,
        }
        super().__init__(
            headers=["Name", "Type", "Process Time", "ID"],
            REQUEST_MAPPING=REQUEST_MAPPING,
        )

    def export_all(self) -> None:
        """Exports all results to files named after their entry id."""
        for item in self.items:
            filename = os.path.join(self.request.file_path, f"{item.entry_id}.result")
            self.file_request.emit(
                FileRequest(
                    type=FileRequestType.WRITE_RES,
                    result_selection=ResultSelection(entries=[item]),
                    file_list=[filename],
                )
            )
        self.write_result_order()

    def write_result_order(self):
        result_order_filename = os.path.join(self.request.file_path, "result_order.txt")
        self.file_request.emit(
            FileRequest(
                type=FileRequestType.WRITE_LIST,
                id_list=[item.entry_id for item in self.items],
                file_list=[result_order_filename],
            )
        )

    def rename_result(self) -> None:
        selected_entry = self.request.selection.entries[0]
        input_name, ok = QInputDialog.getText(None, "Please enter a name", "Name:", text=selected_entry.name)
        if ok and input_name is not None:
            selected_entry.name = input_name

        self.layoutChanged.emit()

    @property
    def alignment_entries(self) -> List[AlignmentEntry]:
        return [entry for entry in self.items if isinstance(entry, AlignmentEntry)]

    def pass_alignments_to_ui(self) -> None:
        if alignment_entries := self.alignment_entries:
            self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.ALIGNMENT_SELECTION,
                    result_selection=ResultSelection(entries=alignment_entries),
                    trajectory_selection=self.request.trajectory_selection,
                )
            )
        else:
            self.ui_request.emit(UIRequest(type=UIRequestType.MESSAGE, message="No Alignments available!"))

    def edit_alignment(self) -> None:
        self.ui_request.emit(
            UIRequest(
                type=UIRequestType.EDIT_ALIGNMENT,
                alignment_entry=self.request.alignment,
            )
        )
