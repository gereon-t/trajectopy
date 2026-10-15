import logging

from trajectopy.gui.managers.requests import (
    ResultModelRequest,
    ResultModelRequestType,
    UIRequest,
    UIRequestType,
)
from trajectopy.gui.models.entries import AlignmentEntry, ResultEntry
from trajectopy.gui.models.selection import ResultSelection
from trajectopy.gui.models.table_model import RequestTableModel

logger = logging.getLogger(__name__)


class ResultTableModel(RequestTableModel):
    """Class representing the result table model."""

    items: list[ResultEntry]

    def __init__(self):
        REQUEST_MAPPING = {
            ResultModelRequestType.ADD: self.add_entries,
            ResultModelRequestType.UPDATE: self.update_selected_entries,
            ResultModelRequestType.RENAME: self.rename_result,
            ResultModelRequestType.REMOVE: self.remove_selected_entries,
            ResultModelRequestType.PASS_ALIGNMENTS_TO_UI: self.pass_alignments_to_ui,
            ResultModelRequestType.RESET: self.reset,
            ResultModelRequestType.EDIT_ALIGNMENT: self.edit_alignment,
            ResultModelRequestType.COPY: self.copy_selected_entries,
            ResultModelRequestType.SORT: self.sort_items,
        }
        super().__init__(
            headers=["Name", "Result Type"],
            REQUEST_MAPPING=REQUEST_MAPPING,
        )

    def rename_result(self, request: ResultModelRequest) -> None:
        """Renames the (first) selected result to ``request.name``."""
        if request.name and request.selection and (entry := self.get(request.selection.entries[0].entry_id)):
            entry.name = request.name
            self.refresh()

    @property
    def alignment_entries(self) -> list[AlignmentEntry]:
        return [entry for entry in self.items if isinstance(entry, AlignmentEntry)]

    def pass_alignments_to_ui(self, request: ResultModelRequest) -> None:
        if alignment_entries := self.alignment_entries:
            self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.ALIGNMENT_SELECTION,
                    result_selection=ResultSelection(entries=list[ResultEntry](alignment_entries)),
                    trajectory_selection=request.trajectory_selection,
                )
            )
        else:
            self.ui_request.emit(UIRequest(type=UIRequestType.MESSAGE, message="No Alignments available!"))

    def edit_alignment(self, request: ResultModelRequest) -> None:
        self.ui_request.emit(
            UIRequest(
                type=UIRequestType.EDIT_ALIGNMENT,
                alignment_entry=request.alignment,
            )
        )
