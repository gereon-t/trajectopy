import copy
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

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
from trajectopy.gui.worker import BackgroundWorker
from trajectopy.processing import (
    alignment,
    approximation,
    evaluation,
    matching,
    merging,
    sorting,
)
from trajectopy.processing.interpolation import interpolate
from trajectopy.utils.definitions import Sorting

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PerEntryOperation:
    """An operation applied to every selected trajectory entry.

    Attributes:
        function: Receives the entry (and reference) and returns the new or updated entries.
        inplace: Whether returned trajectory entries replace the input entries or are added as new entries.
        apply_to_reference: Whether the operation is also applied to the reference trajectory if it is selected.
    """

    function: Callable[["TrajectoryEntryPair"], tuple | None]
    inplace: bool
    apply_to_reference: bool


@dataclass
class TrajectoryEntryPair:
    entry: TrajectoryEntry
    request: TrajectoryManagerRequest
    reference_entry: TrajectoryEntry | None = None


class TrajectoryManager(QObject):
    """
    A class that manages trajectories and provides methods for various operations such as alignment, approximation, and comparison.

    Attributes:
        trajectory_model_request (Signal): A signal emitted when a request for the trajectory model is made.
        result_model_request (Signal): A signal emitted when a request for the result model is made.
        update_view (Signal): A signal emitted when the view needs to be updated.

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

    trajectory_model_request = Signal(TrajectoryModelRequest)
    result_model_request = Signal(ResultModelRequest)
    update_view = Signal()
    operation_started = Signal()
    operation_finished = Signal()
    ui_request = Signal(UIRequest)

    def __init__(self, worker: BackgroundWorker | None = None) -> None:
        """
        Initializes the TrajectoryManager object.

        Args:
            worker: Worker that executes the (potentially long-running) requests. Requests are processed
                synchronously on the calling thread if no worker is given.
        """
        super().__init__()
        self._worker = worker or BackgroundWorker(inline=True)

        per_entry_operations: dict[TrajectoryManagerRequestType, _PerEntryOperation] = {
            TrajectoryManagerRequestType.EDIT_EPSG: _PerEntryOperation(
                self.operation_epsg_edit, inplace=True, apply_to_reference=True
            ),
            TrajectoryManagerRequestType.CHANGE_ESPG: _PerEntryOperation(
                self.operation_epsg_change, inplace=True, apply_to_reference=True
            ),
            TrajectoryManagerRequestType.REARANGE_DOF: _PerEntryOperation(
                self.operation_rearange_dof, inplace=True, apply_to_reference=True
            ),
            TrajectoryManagerRequestType.EPSG_TO_REF: _PerEntryOperation(
                self.operation_ref_epsg, inplace=True, apply_to_reference=False
            ),
            TrajectoryManagerRequestType.ALIGN: _PerEntryOperation(
                self.operation_align, inplace=False, apply_to_reference=False
            ),
            TrajectoryManagerRequestType.ADAPT_FIRST_POSE: _PerEntryOperation(
                self.operation_adopt_first_pose, inplace=False, apply_to_reference=False
            ),
            TrajectoryManagerRequestType.APPLY_ALIGNMENT: _PerEntryOperation(
                self.operation_apply_alignment, inplace=False, apply_to_reference=True
            ),
            TrajectoryManagerRequestType.APPROXIMATE: _PerEntryOperation(
                self.operation_approximate, inplace=False, apply_to_reference=True
            ),
            TrajectoryManagerRequestType.SORT: _PerEntryOperation(
                self.operation_sort, inplace=False, apply_to_reference=True
            ),
            TrajectoryManagerRequestType.SWITCH_SORTING: _PerEntryOperation(
                self.operation_switch_sorting, inplace=True, apply_to_reference=True
            ),
            TrajectoryManagerRequestType.MATCH: _PerEntryOperation(
                self.operation_match, inplace=False, apply_to_reference=False
            ),
            TrajectoryManagerRequestType.ATE_WITH_ALIGNMENT: _PerEntryOperation(
                self.operation_ate_with_alignment, inplace=False, apply_to_reference=False
            ),
            TrajectoryManagerRequestType.ATE_NO_ALIGNMENT: _PerEntryOperation(
                self.operation_ate_without_alignment, inplace=False, apply_to_reference=False
            ),
            TrajectoryManagerRequestType.RPE: _PerEntryOperation(
                self.operation_rpe, inplace=False, apply_to_reference=False
            ),
            TrajectoryManagerRequestType.INTERPOLATE_TO_GRID: _PerEntryOperation(
                self.operation_interpolate_to_grid, inplace=False, apply_to_reference=True
            ),
        }
        self.REQUEST_MAPPING: dict[TrajectoryManagerRequestType, Callable[[TrajectoryManagerRequest], None]] = {
            request_type: functools.partial(self.handle_trajectory_operation, operation=operation)
            for request_type, operation in per_entry_operations.items()
        }
        self.REQUEST_MAPPING.update(
            {
                TrajectoryManagerRequestType.AVERAGE: self.operation_average_trajectories,
                TrajectoryManagerRequestType.MERGE: self.operation_merge_trajectories,
                TrajectoryManagerRequestType.DIVIDE_INTO_LAPS: self.operation_divide_into_laps,
            }
        )

    @staticmethod
    def selected_trajectory_entries(
        request: TrajectoryManagerRequest, return_reference: bool = True
    ) -> list[TrajectoryEntry]:
        """
        Returns a list of selected trajectory entries.

        Args:
            request: The request containing the selection.
            return_reference: If True, returns the reference trajectory entry as well.

        Returns:
            A list of selected trajectory entries.
        """
        return [entry for entry in request.selection.entries if return_reference or not entry.set_as_reference]

    @Slot(TrajectoryManagerRequest)
    def handle_request(self, request: TrajectoryManagerRequest) -> None:
        """
        Schedules a trajectory manager request on the worker.

        ``operation_started`` is emitted immediately; ``operation_finished`` and ``update_view`` are
        emitted once the request has been processed.

        Args:
            request: The trajectory manager request to handle.
        """
        self.operation_started.emit()
        self._worker.submit(self._process_request, request)

    def _process_request(self, request: TrajectoryManagerRequest) -> None:
        try:
            generic_request_handler(self, request)
        finally:
            self.operation_finished.emit()
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

    def handle_trajectory_operation(self, request: TrajectoryManagerRequest, operation: "_PerEntryOperation") -> None:
        """
        Executes a given operation on each selected trajectory entry and emits the resulting trajectory and/or result
        entries.

        Args:
            request: The request containing the selected entries and operation parameters.
            operation: The operation to apply to each selected entry.
        """
        for selected_entry in self.selected_trajectory_entries(request, return_reference=operation.apply_to_reference):
            entry_pair = TrajectoryEntryPair(
                entry=selected_entry,
                reference_entry=copy.deepcopy(request.selection.reference_entry),
                request=request,
            )

            output_entries = operation.function(entry_pair)

            if output_entries is None:
                continue

            for entry in output_entries:
                if isinstance(entry, TrajectoryEntry):
                    if operation.inplace:
                        self.emit_update_trajectory_signal(entry)
                    else:
                        self.emit_add_trajectory_signal(entry)
                elif isinstance(entry, ResultEntry):
                    self.emit_add_result_signal(entry)

    def operation_merge_trajectories(self, request: TrajectoryManagerRequest) -> None:
        """
        Merges all selected trajectories into one trajectory. The first selected trajectory is used as reference for
        the local coordinate system and the EPSG code. The resulting trajectory is added to the trajectory model.

        Args:
            None.

        Returns:
            None.
        """
        if not (selected_entries := self.selected_trajectory_entries(request)):
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
        new_sorting = (
            Sorting.TIME if entry_pair.entry.trajectory.sorting == Sorting.PATH_LENGTH else Sorting.PATH_LENGTH
        )
        entry_pair.entry.trajectory.set_sorting(new_sorting)
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
        return (entry_pair.entry.derive(entry_pair.entry.trajectory),)

    def operation_divide_into_laps(self, request: TrajectoryManagerRequest) -> None:
        """
        Divides the trajectory into laps based on spatial sorting

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to divide into laps.

        Returns:
            TrajectoryEntry: The

        """
        if not (selected_entries := self.selected_trajectory_entries(request)):
            return

        for selected_entry in selected_entries:
            logger.info("Dividing trajectory %s into laps ...", selected_entry.trajectory.name)
            laps = sorting.divide_into_laps(trajectory=selected_entry.trajectory)

            for i, lap in enumerate(laps):
                lap.name += f" Lap {i + 1}"
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
        approximation.approximate(
            trajectory=entry_pair.entry.trajectory,
            approximation_settings=entry_pair.entry.settings.approximation,
            inplace=True,
        )
        entry_pair.entry.state.approximated = True
        entry_pair.entry.trajectory.name += " (approximated)"
        return (entry_pair.entry.derive(entry_pair.entry.trajectory),)

    def operation_average_trajectories(self, request: TrajectoryManagerRequest) -> None:
        """
        Averages all selected trajectories into one trajectory.

        Args:
            None.

        Returns:
            None.
        """
        if not (selected_entries := self.selected_trajectory_entries(request)):
            return

        averaged_trajectory = merging.average_trajectories([entry.trajectory for entry in selected_entries])

        new_trajectory_entry = TrajectoryEntry(
            full_filename="",
            trajectory=averaged_trajectory,
            group_id=selected_entries[0].group_id,
        )

        self.emit_add_trajectory_signal(new_trajectory_entry)

    @staticmethod
    def operation_interpolate_to_grid(
        entry_pair: TrajectoryEntryPair,
    ) -> tuple[TrajectoryEntry]:
        """Resamples the trajectory to a uniform time grid with the given step size."""
        t = entry_pair.entry.trajectory.timestamps
        step = entry_pair.request.grid
        new_timestamps = np.arange(t[0], t[-1] + step, step)
        interpolate(entry_pair.entry.trajectory, index=new_timestamps, inplace=True)
        entry_pair.entry.trajectory.name += f" (grid {step:.4g}s)"
        return (entry_pair.entry.derive(entry_pair.entry.trajectory),)

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
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "time": 6,
        }

        xyz = entry_pair.entry.trajectory.positions.xyz

        if entry_pair.entry.trajectory.has_orientation:
            rpy = np.rad2deg(entry_pair.entry.trajectory.rpy)
        else:
            rpy = np.zeros(xyz.shape)

        xyz_rpy_t = np.hstack((xyz, rpy, entry_pair.entry.trajectory.timestamps[:, None]))
        new_xyz_rpy_t = np.zeros_like(xyz_rpy_t)
        for key, mapping in dof_mapping.items():
            source_col_index = index_mapping[key.lower()]
            target_col_index = index_mapping[mapping["target"].lower()]
            new_column = (xyz_rpy_t[:, source_col_index] + mapping["bias"]) * (1.0 if mapping["sign"] == "+" else -1.0)
            new_xyz_rpy_t[:, target_col_index] = new_column

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

        if (alignment_result := entry_pair.request.manual_alignment) is None:
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
            entry_pair.entry.derive(traj_aligned),
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
        new_entry = entry_pair.entry.derive(aligned_trajectory)
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

        return entry_pair.entry.derive(traj_aligned), reference_entry.derive(traj_ref, keep_state=False)

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

        return entry_pair.entry.derive(traj_test), reference_entry.derive(traj_ref)

    @staticmethod
    def operation_ate_with_alignment(entry_pair: TrajectoryEntryPair) -> tuple[ResultEntry, ...]:
        """
        Computes the absolute trajectory error (ATE) by aligning the selected
        trajectory to the reference trajectory and computing the pose differences.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to compare.

        Returns:
            ResultEntry: The result of the comparison and the alignment information.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        ate_result, alignment_result = evaluation.ate(
            trajectory=entry_pair.entry.trajectory,
            other=reference_entry.trajectory,
            processing_settings=entry_pair.entry.settings,
            return_alignment=True,
            align=True,
        )

        return (AbsoluteDeviationEntry(deviations=ate_result), AlignmentEntry(alignment_result=alignment_result))

    @staticmethod
    def operation_ate_without_alignment(entry_pair: TrajectoryEntryPair) -> tuple[ResultEntry, ...]:
        """
        Computes the absolute trajectory error (ATE) by aligning the selected
        trajectory to the reference trajectory and computing the pose differences.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to compare.

        Returns:
            ResultEntry: The result of the comparison and the alignment information.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        ate_result = evaluation.ate(
            trajectory=entry_pair.entry.trajectory,
            other=reference_entry.trajectory,
            processing_settings=entry_pair.entry.settings,
            return_alignment=False,
            align=False,
        )

        return (AbsoluteDeviationEntry(deviations=ate_result),)

    @staticmethod
    def operation_rpe(entry_pair: TrajectoryEntryPair) -> tuple[ResultEntry, ...]:
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
