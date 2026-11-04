import copy
import logging

from trajectopy.gui.managers.requests import (
    TrajectoryModelRequest,
    TrajectoryModelRequestType,
)
from trajectopy.gui.models.entries import TrajectoryEntry
from trajectopy.gui.models.selection import TrajectorySelection
from trajectopy.gui.models.table_model import GroupedTableModel

logger = logging.getLogger(__name__)


class TrajectoryTableModel(GroupedTableModel):
    """Class representing the trajectory table model."""

    items: list[TrajectoryEntry]

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
            TrajectoryModelRequestType.SORT: self.sort_items,
            TrajectoryModelRequestType.GROUP: self.group_selected_entries,
            TrajectoryModelRequestType.UNGROUP: self.ungroup_selected_entries,
            TrajectoryModelRequestType.RENAME_GROUP: self.rename_selected_groups,
            TrajectoryModelRequestType.MOVE_TO_GROUP: self.move_selected_entries_to_group,
        }

        super().__init__(
            headers=["Name", "EPSG", "Reference"],
            REQUEST_MAPPING=REQUEST_MAPPING,
        )

    def update_selected_entries(self, request) -> None:
        # the results of operations on (copies of) the entries; which entry is the reference is decided in the
        # table, possibly while the operation ran
        for entry in request.selection.entries:
            if (current := self.get(entry.entry_id)) is not None:
                entry.set_as_reference = current.set_as_reference
        super().update_selected_entries(request)

    def default_group_name(self, entries: list) -> str:
        # trajectories derived from another one share its group: named after the first one (usually the source)
        return entries[0].name

    def remove_group(self, group_id: str) -> None:
        self.remove([item.entry_id for item in self.items if item.group_id == group_id])

    @property
    def reference_entry(self) -> TrajectoryEntry | None:
        return next((item for item in self.items if item.set_as_reference), None)

    def rename_trajectories(self, request: TrajectoryModelRequest) -> None:
        """Renames the (first) selected trajectory to ``request.name``."""
        if request.name and request.selection and (entry := self.get(request.selection.entries[0].entry_id)):
            entry.trajectory.name = request.name
            self.refresh()

    def remove_related_trajectories(self, request: TrajectoryModelRequest) -> None:
        """Remove all trajectories that are related"""
        if not isinstance((selection := request.selection), TrajectorySelection):
            raise ValueError("No trajectory selection provided")

        group_ids = {entry.group_id for entry in selection.entries}
        self.remove([item.entry_id for item in self.items if item.group_id in group_ids])

    def copy_trajectories(self, request: TrajectoryModelRequest) -> None:
        """Deep copy trajectory"""
        if not isinstance((selection := request.selection), TrajectorySelection):
            raise ValueError("No trajectory selection provided")

        for entry in selection.entries:
            entry_copy = copy.deepcopy(entry)
            entry_copy.renew_id()
            entry_copy.set_as_reference = False
            self.add(entry_copy)

    def set_as_reference(self, request: TrajectoryModelRequest) -> None:
        """Marks selected trajectory as reference"""
        if not isinstance((selection := request.selection), TrajectorySelection):
            raise ValueError("No trajectory selection provided")

        selected_id = selection.entries[0].entry_id
        for entry in self.items:
            entry.set_as_reference = entry.entry_id == selected_id
        self.refresh()

    def unset_as_reference(self, request: TrajectoryModelRequest) -> None:
        """Removes the reference status from the selected trajectory"""
        for entry in self.items:
            entry.set_as_reference = False
        self.refresh()
