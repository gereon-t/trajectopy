import copy
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from trajectopy.core.rotations import Rotations
from trajectopy.gui.managers.requests import (
    ResultModelRequest,
    ResultModelRequestType,
    TrajectoryManagerRequest,
    TrajectoryManagerRequestType,
    TrajectoryModelRequest,
    TrajectoryModelRequestType,
    UIRequest,
    generic_request_handler,
)
from trajectopy.gui.models.entries import (
    AbsoluteDeviationEntry,
    AlignmentEntry,
    RelativeDeviationEntry,
    ResultEntry,
    TrajectoryEntry,
)
from trajectopy.gui.models.selection import ResultSelection, TrajectorySelection
from trajectopy.gui.utils import show_progress
from trajectopy.processing import (
    alignment,
    approximation,
    evaluation,
    matching,
    merging,
    sorting,
)
from trajectopy.utils.definitions import Sorting

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryEntryPair:
    entry: TrajectoryEntry
    request: TrajectoryManagerRequest
    reference_entry: TrajectoryEntry | None = None

    def __post_init__(self) -> None:
        if self.request.type in [
            TrajectoryManagerRequestType.ATE,
            TrajectoryManagerRequestType.RPE,
        ]:  # pipeline requests use matching setting from trajectory
            return

        self.entry.settings.matching.method = self.request.matching_method


class TrajectoryManager(QObject):
    """
    A class that manages trajectories and provides methods for various operations such as alignment, approximation, and comparison.

    Attributes:
        trajectory_model_request (pyqtSignal): A signal emitted when a request for the trajectory model is made.
        result_model_request (pyqtSignal): A signal emitted when a request for the result model is made.
        update_view (pyqtSignal): A signal emitted when the view needs to be updated.

    Possible Requests:
        TrajectoryManagerRequestType.EDIT_EPSG: Edits the EPSG code of the selected trajectory without transformation (only metadata).
        TrajectoryManagerRequestType.CHANGE_ESPG: Changes the EPSG code of the selected trajectory to the specified EPSG code.
        TrajectoryManagerRequestType.EPSG_TO_REF: Changes the EPSG code of the selected trajectory to the EPSG code of the reference trajectory.
        TrajectoryManagerRequestType.ALIGN: Aligns the selected trajectory to the reference trajectory.
        TrajectoryManagerRequestType.ADAPT_FIRST_POSE: Adopts the position and orientation of the first pose of the current trajectory to the reference trajectory.
        TrajectoryManagerRequestType.APPLY_ALIGNMENT: Applies the selected alignment to the trajectory of the given entry pair.
        TrajectoryManagerRequestType.APPROXIMATE: Approximates the selected trajectory using the settings specified in the entry.
        TrajectoryManagerRequestType.COMPARE: Compares the selected trajectory to the reference trajectory.
        TrajectoryManagerRequestType.MERGE: Merges all selected trajectories into one trajectory.
        TrajectoryManagerRequestType.MATCH_TIMESTAMPS: Matches the timestamps of the two trajectories in the given `TrajectoryEntryPair`. After this, both trajectories will have the same number of poses at the same points in time. This may result in cropping the reference trajectory.
        TrajectoryManagerRequestType.SORT: Sorts the selected trajectory using the settings specified in the entry.
        TrajectoryManagerRequestType.SWITCH_SORTING: Changes the sorting of the trajectory.
        TrajectoryManagerRequestType.ATE: Computes the absolute trajectory error between the selected trajectory and the reference trajectory.
        TrajectoryManagerRequestType.RPE: Computes the relative pose error between the selected trajectory and the reference trajectory.


    Methods:
        __init__(): Initializes the TrajectoryManager object.
    """

    trajectory_model_request = pyqtSignal(TrajectoryModelRequest)
    result_model_request = pyqtSignal(ResultModelRequest)
    update_view = pyqtSignal()
    operation_started = pyqtSignal()
    operation_finished = pyqtSignal()
    ui_request = pyqtSignal(UIRequest)

    def __init__(self) -> None:
        """
        Initializes the TrajectoryManager object.
        """
        super().__init__()
        self.request: TrajectoryManagerRequest
        self.REQUEST_MAPPING: dict[TrajectoryManagerRequestType, Any] = {
            TrajectoryManagerRequestType.EDIT_EPSG: lambda: self.handle_trajectory_operation(
                operation=self.operation_epsg_edit,
                inplace=True,
                apply_to_reference=True,
            ),
            TrajectoryManagerRequestType.CHANGE_ESPG: lambda: self.handle_trajectory_operation(
                operation=self.operation_epsg_change,
                inplace=True,
                apply_to_reference=True,
            ),
            TrajectoryManagerRequestType.REARANGE_DOF: lambda: self.handle_trajectory_operation(
                operation=self.operation_rearange_dof,
                inplace=True,
                apply_to_reference=True,
            ),
            TrajectoryManagerRequestType.EPSG_TO_REF: lambda: self.handle_trajectory_operation(
                operation=self.operation_ref_epsg,
                inplace=True,
                apply_to_reference=False,
            ),
            TrajectoryManagerRequestType.ALIGN: lambda: self.handle_trajectory_operation(
                operation=self.operation_align, inplace=False, apply_to_reference=False
            ),
            TrajectoryManagerRequestType.ADAPT_FIRST_POSE: lambda: self.handle_trajectory_operation(
                operation=self.operation_adopt_first_pose,
                inplace=False,
                apply_to_reference=False,
            ),
            TrajectoryManagerRequestType.APPLY_ALIGNMENT: lambda: self.handle_trajectory_operation(
                operation=self.operation_apply_alignment,
                inplace=False,
                apply_to_reference=True,
            ),
            TrajectoryManagerRequestType.APPROXIMATE: lambda: self.handle_trajectory_operation(
                operation=self.operation_approximate,
                inplace=False,
                apply_to_reference=True,
            ),
            TrajectoryManagerRequestType.AVERAGE: self.operation_average_trajectories,
            TrajectoryManagerRequestType.COMPARE_ABS: lambda: self.handle_trajectory_operation(
                operation=self.operation_compare_abs,
                inplace=False,
                apply_to_reference=False,
            ),
            TrajectoryManagerRequestType.COMPARE_REL: lambda: self.handle_trajectory_operation(
                operation=self.operation_compare_rel,
                inplace=False,
                apply_to_reference=False,
            ),
            TrajectoryManagerRequestType.MERGE: self.operation_merge_trajectories,
            TrajectoryManagerRequestType.SORT: lambda: self.handle_trajectory_operation(
                operation=self.operation_sort, inplace=False, apply_to_reference=True
            ),
            TrajectoryManagerRequestType.DIVIDE_INTO_LAPS: self.operation_divide_into_laps,
            TrajectoryManagerRequestType.SWITCH_SORTING: lambda: self.handle_trajectory_operation(
                operation=self.operation_switch_sorting,
                inplace=True,
                apply_to_reference=True,
            ),
            TrajectoryManagerRequestType.MATCH: lambda: self.handle_trajectory_operation(
                operation=self.operation_match, inplace=False, apply_to_reference=False
            ),
            TrajectoryManagerRequestType.ATE: lambda: self.handle_trajectory_operation(
                operation=self.operation_ate, inplace=False, apply_to_reference=False
            ),
            TrajectoryManagerRequestType.RPE: lambda: self.handle_trajectory_operation(
                operation=self.operation_rpe, inplace=False, apply_to_reference=False
            ),
        }

    def selected_trajectory_entries(self, return_reference: bool = True) -> list[TrajectoryEntry]:
        """
        Returns a list of selected trajectory entries.

        Args:
            return_reference: If True, returns the reference trajectory entry as well.

        Returns:
            A list of selected trajectory entries.
        """
        return [entry for entry in self.request.selection.entries if return_reference or not entry.set_as_reference]

    @property
    def reference_entry(self) -> TrajectoryEntry | None:
        """
        Returns the reference trajectory entry of the current trajectory manager request.

        Returns:
            The reference trajectory entry of the current trajectory manager request.
        """
        return self.request.selection.reference_entry

    @show_progress
    @pyqtSlot(TrajectoryManagerRequest)
    def handle_request(self, request: TrajectoryManagerRequest) -> None:
        """
        Handles a trajectory manager request and emits a signal to update the view.

        Args:
            trajectory_manager_request: The trajectory manager request to handle.

        Returns:
            None.
        """
        self.request = request
        request_thread = threading.Thread(target=generic_request_handler, args=(self, request, False))
        request_thread.start()
        request_thread.join()
        self.update_view.emit()

    def emit_add_trajectory_signal(self, new_trajectory_entry: TrajectoryEntry):
        """
        Emits a signal to add a new trajectory entry to the trajectory model.

        Args:
            new_trajectory_entry: The new trajectory entry to add.

        Returns:
            None.
        """
        self.trajectory_model_request.emit(
            TrajectoryModelRequest(
                type=TrajectoryModelRequestType.ADD,
                selection=TrajectorySelection(entries=[new_trajectory_entry]),
            )
        )

    def emit_add_result_signal(self, new_result_entry: ResultEntry):
        """
        Emits a signal to add a new result entry to the result model.

        Args:
            new_result_entry: The new result entry to add.

        Returns:
            None.
        """
        self.result_model_request.emit(
            ResultModelRequest(
                type=ResultModelRequestType.ADD,
                selection=ResultSelection(entries=[new_result_entry]),
            )
        )

    def emit_update_trajectory_signal(self, trajectory_entry: TrajectoryEntry) -> None:
        """
        Emits a signal to update a trajectory entry in the trajectory model.

        Args:
            trajectory_entry: The updated trajectory entry

        Returns:
            None.
        """
        self.trajectory_model_request.emit(
            TrajectoryModelRequest(
                type=TrajectoryModelRequestType.UPDATE,
                selection=TrajectorySelection(entries=[trajectory_entry]),
            )
        )

    def handle_trajectory_operation(
        self,
        operation: Callable[[TrajectoryEntryPair], tuple | None],
        inplace: bool = False,
        apply_to_reference: bool = True,
    ) -> None:
        """
        Executes a given operation on each selected trajectory entry and emits the resulting trajectory and/or result
        entries.

        Args:
            operation: A callable that takes a TrajectoryEntryPair as input and returns either tuple of TrajectoryEntry,
                ResultEntry, or None.

        Returns:
            None.
        """
        selected_entries = self.selected_trajectory_entries(return_reference=apply_to_reference)

        if selected_entries is None:
            return

        for selected_entry in selected_entries:
            entry_pair = TrajectoryEntryPair(
                entry=selected_entry,
                reference_entry=self.reference_entry,
                request=self.request,
            )

            output_entries = operation(entry_pair)

            if output_entries is None:
                continue

            for entry in output_entries:
                if isinstance(entry, TrajectoryEntry):
                    if inplace:
                        self.emit_update_trajectory_signal(entry)
                    else:
                        self.emit_add_trajectory_signal(entry)
                elif isinstance(entry, ResultEntry):
                    self.emit_add_result_signal(entry)

    def operation_merge_trajectories(self) -> None:
        """
        Merges all selected trajectories into one trajectory. The first selected trajectory is used as reference for
        the local coordinate system and the EPSG code. The resulting trajectory is added to the trajectory model.

        Args:
            None.

        Returns:
            None.
        """
        if (selected_entries := self.selected_trajectory_entries()) is None:
            return

        merged_trajectory = merging.merge_trajectories([entry.trajectory for entry in selected_entries])

        new_trajectory_entry = TrajectoryEntry(
            full_filename="",
            trajectory=merged_trajectory,
            group_id=selected_entries[0].group_id,
        )
        self.emit_add_trajectory_signal(new_trajectory_entry)

    @staticmethod
    def operation_switch_sorting(
        entry_pair: TrajectoryEntryPair,
    ) -> tuple[TrajectoryEntry]:
        """
        Changes the sorting of the trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The trajectory entry pair containing the trajectory to be sorted.

        Returns:
            None.
        """
        entry_pair.entry.trajectory.sorting = (
            Sorting.TIME if entry_pair.entry.trajectory.sorting == Sorting.PATH_LENGTH else Sorting.PATH_LENGTH
        )
        return (entry_pair.entry,)

    @staticmethod
    def operation_sort(entry_pair: TrajectoryEntryPair) -> tuple[TrajectoryEntry]:
        """
        Sorts the selected trajectory using the settings specified in the entry.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to sort.

        Returns:
            TrajectoryEntry: The sorted trajectory.
        """
        logger.info("Sorting trajectory ...")
        sorting.sort_spatially(
            trajectory=entry_pair.entry.trajectory, sorting_settings=entry_pair.entry.settings.sorting
        )
        entry_pair.entry.state.sorting_known = True
        entry_pair.entry.trajectory.name += " (sorted)"
        return (
            TrajectoryEntry(
                full_filename=entry_pair.entry.full_filename,
                trajectory=entry_pair.entry.trajectory,
                settings=entry_pair.entry.settings,
                group_id=entry_pair.entry.group_id,
                state=entry_pair.entry.state,
            ),
        )

    def operation_divide_into_laps(self) -> tuple[TrajectoryEntry]:
        """
        Divides the trajectory into laps based on spatial sorting

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to divide into laps.

        Returns:
            TrajectoryEntry: The

        """
        if (selected_entries := self.selected_trajectory_entries()) is None:
            return

        for selected_entry in selected_entries:
            logger.info(f"Dividing trajectory {selected_entry.trajectory.name} into laps ...")
            laps = sorting.divide_into_laps(
                trajectory=selected_entry.trajectory, sorting_settings=selected_entry.settings.sorting
            )

            for i, lap in enumerate(laps):
                lap.name += f" Lap {i+1}"
                state = copy.deepcopy(selected_entry.state)
                state.sorting_known = True
                new_trajectory_entry = TrajectoryEntry(
                    full_filename="",
                    trajectory=lap,
                    group_id=selected_entry.group_id,
                    state=state,
                )
                self.emit_add_trajectory_signal(new_trajectory_entry)

    @staticmethod
    def operation_approximate(
        entry_pair: TrajectoryEntryPair,
    ) -> tuple[TrajectoryEntry]:
        """
        Approximates the selected trajectory using the settings specified in the entry.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to approximate.

        Returns:
            TrajectoryEntry: The approximated trajectory.
        """
        approximation.approximate_cubic(
            trajectory=entry_pair.entry.trajectory,
            approximation_settings=entry_pair.entry.settings.approximation,
            inplace=True,
        )
        entry_pair.entry.state.approximated = True
        entry_pair.entry.trajectory.name += " (approximated)"
        return (
            TrajectoryEntry(
                full_filename=entry_pair.entry.full_filename,
                trajectory=entry_pair.entry.trajectory,
                settings=entry_pair.entry.settings,
                group_id=entry_pair.entry.group_id,
                state=entry_pair.entry.state,
            ),
        )

    def operation_average_trajectories(self) -> None:
        """
        Averages all selected trajectories into one trajectory.

        Args:
            None.

        Returns:
            None.
        """
        if (selected_entries := self.selected_trajectory_entries()) is None:
            return

        averaged_trajectory = merging.average_trajectories([entry.trajectory for entry in selected_entries])

        new_trajectory_entry = TrajectoryEntry(
            full_filename="",
            trajectory=averaged_trajectory,
            group_id=selected_entries[0].group_id,
        )

        self.emit_add_trajectory_signal(new_trajectory_entry)

    @staticmethod
    def operation_compare_abs(entry_pair: TrajectoryEntryPair) -> tuple[ResultEntry]:
        """
        Compares the selected trajectory to the reference trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to compare.

        Returns:
            ResultEntry: The result of the comparison.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        traj_test, traj_ref = matching.match_trajectories(
            trajectory=entry_pair.entry.trajectory,
            other=reference_entry.trajectory,
            matching_settings=entry_pair.entry.settings.matching,
        )

        if len(traj_ref) != len(traj_test):
            raise ValueError("Something went wrong during matching.")

        comparison_result = evaluation._compare_trajectories_absolute(trajectory=traj_test, other=traj_ref)

        return (AbsoluteDeviationEntry(deviations=comparison_result),)

    @staticmethod
    def operation_compare_rel(entry_pair: TrajectoryEntryPair) -> tuple[ResultEntry]:
        """
        Compares the selected trajectory to the reference trajectory using relative comparison.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to compare.

        Returns:
            ResultEntry: The result of the comparison.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        traj_test, traj_ref = matching.match_trajectories(
            trajectory=entry_pair.entry.trajectory,
            other=reference_entry.trajectory,
            matching_settings=entry_pair.entry.settings.matching,
        )

        if len(traj_ref) != len(traj_test):
            raise ValueError("Something went wrong during matching.")

        comparison_result = evaluation._compare_trajectories_relative(
            trajectory=traj_test,
            other=traj_ref,
            relative_comparison_settings=entry_pair.entry.settings.relative_comparison,
        )

        return (RelativeDeviationEntry(deviations=comparison_result),)

    @staticmethod
    def operation_epsg_edit(
        entry_pair: TrajectoryEntryPair,
    ) -> tuple[TrajectoryEntry]:
        """
        Edits the EPSG of the selected trajectory to the specified EPSG code.
        No transformation is applied, only the metadata is changed.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to edit the EPSG code for.

        Returns:
            None
        """
        entry_pair.entry.trajectory.positions.epsg = entry_pair.request.target_epsg
        entry_pair.entry.trajectory.positions.build_local_transformer()
        return (entry_pair.entry,)

    @staticmethod
    def operation_epsg_change(
        entry_pair: TrajectoryEntryPair,
    ) -> tuple[TrajectoryEntry]:
        """
        Changes the datum of the selected trajectory to the specified EPSG code.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to change the EPSG code for.

        Returns:
            None
        """
        entry_pair.entry.trajectory.positions.to_epsg(entry_pair.request.target_epsg)
        return (entry_pair.entry,)

    @staticmethod
    def operation_rearange_dof(
        entry_pair: TrajectoryEntryPair,
    ) -> tuple[TrajectoryEntry]:
        """
        Rearranges the degrees of freedom (DOF) of the selected trajectory according to the specified request.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to rearrange DOF for.

        Returns:
            TrajectoryEntry: The trajectory entry with rearranged DOF.
        """
        dof_mapping = entry_pair.request.dof_mapping

        # rearrange the DOF of the trajectory
        index_mapping = {
            "X": 0,
            "Y": 1,
            "Z": 2,
            "Roll": 3,
            "Pitch": 4,
            "Yaw": 5,
            "Time": 6,
        }

        xyz = entry_pair.entry.trajectory.positions.xyz

        if entry_pair.entry.trajectory.has_orientation:
            rpy = np.rad2deg(entry_pair.entry.trajectory.rpy)
        else:
            rpy = np.zeros(xyz.shape)

        xyz_rpy_t = np.hstack((xyz, rpy, entry_pair.entry.trajectory.timestamps[:, None]))
        new_xyz_rpy_t = np.zeros_like(xyz_rpy_t)
        for i, mapping in enumerate(dof_mapping.values()):
            new_column = (xyz_rpy_t[:, index_mapping[mapping["target"]]] + mapping["bias"]) * (
                1.0 if mapping["sign"] == "+" else -1.0
            )
            new_xyz_rpy_t[:, i] = new_column

        entry_pair.entry.trajectory.timestamps = new_xyz_rpy_t[:, 6]
        entry_pair.entry.trajectory.positions.xyz = new_xyz_rpy_t[:, :3]
        if entry_pair.entry.trajectory.has_orientation:
            entry_pair.entry.trajectory.rotations = Rotations.from_euler(
                seq="xyz", angles=new_xyz_rpy_t[:, 3:6], degrees=True
            )

        return (entry_pair.entry,)

    @staticmethod
    def operation_ref_epsg(entry_pair: TrajectoryEntryPair) -> tuple[TrajectoryEntry]:
        """
        Adapt the datum of the reference trajectory to the selected trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to align.

        Returns:
            None
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        reference_trajectory = reference_entry.trajectory
        reference_epsg = reference_trajectory.positions.epsg
        reference_local_transformer = reference_trajectory.positions.local_transformer

        if reference_local_transformer is None:
            raise ValueError("Reference trajectory has an unknown EPSG code.")

        entry_pair.entry.trajectory.positions.local_transformer = reference_local_transformer
        entry_pair.entry.trajectory.positions.to_epsg(reference_epsg)
        return (entry_pair.entry,)

    @staticmethod
    def operation_align(
        entry_pair: TrajectoryEntryPair,
    ) -> tuple[TrajectoryEntry, AlignmentEntry]:
        """
        Aligns the selected trajectory to a reference trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to align.

        Returns:
            list[TrajectoryEntry, AlignmentEntry]: A list containing the aligned trajectory and the alignment information.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        alignment_result = alignment.estimate_alignment(
            trajectory=entry_pair.entry.trajectory,
            other=reference_entry.trajectory,
            alignment_settings=entry_pair.entry.settings.alignment,
            matching_settings=entry_pair.entry.settings.matching,
        )

        traj_aligned = alignment.apply_alignment(
            trajectory=entry_pair.entry.trajectory, alignment_result=alignment_result, inplace=False
        )
        entry_pair.entry.state.aligned = True
        traj_aligned.name += " (aligned)"

        return (
            TrajectoryEntry(
                full_filename=entry_pair.entry.full_filename,
                trajectory=traj_aligned,
                settings=entry_pair.entry.settings,
                group_id=entry_pair.entry.group_id,
                state=entry_pair.entry.state,
            ),
            AlignmentEntry(alignment_result=alignment_result),
        )

    @staticmethod
    def operation_apply_alignment(
        entry_pair: TrajectoryEntryPair,
    ) -> tuple[TrajectoryEntry]:
        """
        Applies the selected alignment to the trajectory of the given entry pair.

        Args:
            entry_pair (TrajectoryEntryPair): The entry pair containing the trajectory to be aligned and the reference
                trajectory.

        Returns:
            TrajectoryEntry: A new trajectory entry with the aligned trajectory.
        """
        aligned_trajectory = alignment.apply_alignment(
            trajectory=entry_pair.entry.trajectory,
            alignment_result=entry_pair.request.alignment.alignment_result,
            inplace=False,
        )
        entry_pair.entry.state.aligned = True
        aligned_trajectory.name += " (aligned)"
        new_entry = TrajectoryEntry(
            full_filename=entry_pair.entry.full_filename,
            trajectory=aligned_trajectory,
            settings=entry_pair.entry.settings,
            group_id=entry_pair.entry.group_id,
            state=entry_pair.entry.state,
        )
        logger.info("Applied alignment to trajectory %s", entry_pair.entry.name)
        return (new_entry,)

    @staticmethod
    def operation_adopt_first_pose(
        entry_pair: TrajectoryEntryPair,
    ) -> tuple[TrajectoryEntry, ...]:
        """
        Adopts the position and orientation of the first pose of the current trajectory to the reference trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The entry pair containing the current trajectory and the reference
                trajectory.

        Returns:
            TrajectoryEntry: A new trajectory entry with the adopted position and orientation.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        traj_test, traj_ref = matching.match_trajectories(
            trajectory=entry_pair.entry.trajectory,
            other=reference_entry.trajectory,
            matching_settings=entry_pair.entry.settings.matching,
        )

        traj_aligned = alignment.adopt_first_pose(trajectory=traj_test, other=traj_ref)
        entry_pair.entry.state.aligned = True
        traj_aligned.name += " (aligned)"

        return TrajectoryEntry(
            full_filename=entry_pair.entry.full_filename,
            trajectory=traj_aligned,
            settings=entry_pair.entry.settings,
            group_id=entry_pair.entry.group_id,
            state=entry_pair.entry.state,
        ), TrajectoryEntry(
            full_filename=reference_entry.full_filename,
            trajectory=traj_ref,
            settings=reference_entry.settings,
            group_id=reference_entry.group_id,
        )

    @staticmethod
    def operation_match(entry_pair: TrajectoryEntryPair) -> tuple[TrajectoryEntry, ...]:
        """
        Matches the current trajectory to the reference trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The entry pair containing the current trajectory and the reference
                trajectory.

        Returns:
            TrajectoryEntry: A new trajectory entry with the matched trajectory.
        """

        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        traj_test, traj_ref = matching.match_trajectories(
            trajectory=entry_pair.entry.trajectory,
            other=reference_entry.trajectory,
            matching_settings=entry_pair.entry.settings.matching,
        )
        reference_entry.state.matched = True
        entry_pair.entry.state.matched = True

        traj_test.name += " (matched)"
        traj_ref.name += " (matched)"

        return TrajectoryEntry(
            full_filename=entry_pair.entry.full_filename,
            trajectory=traj_test,
            settings=entry_pair.entry.settings,
            group_id=entry_pair.entry.group_id,
            state=entry_pair.entry.state,
        ), TrajectoryEntry(
            full_filename=reference_entry.full_filename,
            trajectory=traj_ref,
            settings=reference_entry.settings,
            group_id=reference_entry.group_id,
            state=reference_entry.state,
        )

    @staticmethod
    def operation_ate(entry_pair: TrajectoryEntryPair) -> tuple[ResultEntry]:
        """
        Computes the absolute trajectory error (ATE) by aligning the selected
        trajectory to the reference trajectory and computing the pose differences.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to compare.

        Returns:
            ResultEntry: The result of the comparison.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        ate_result, alignment_result = evaluation.ate(
            trajectory=entry_pair.entry.trajectory,
            other=reference_entry.trajectory,
            processing_settings=entry_pair.entry.settings,
            return_alignment=True,
        )

        return (AbsoluteDeviationEntry(deviations=ate_result), AlignmentEntry(alignment_result=alignment_result))

    @staticmethod
    def operation_rpe(entry_pair: TrajectoryEntryPair) -> tuple[ResultEntry]:
        """
        Computes the relative pose error (RPE)

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to compare.

        Returns:
            ResultEntry: The result of the comparison.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        rpe_result = evaluation.rpe(
            trajectory=entry_pair.entry.trajectory,
            other=reference_entry.trajectory,
            processing_settings=entry_pair.entry.settings,
        )

        return (RelativeDeviationEntry(deviations=rpe_result),)
