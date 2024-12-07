"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

import trajectopy.api as tpy
from trajectopy.core.evaluation.comparison import compare_trajectories_absolute, compare_trajectories_relative
from trajectopy.core.matching import rough_timestamp_matching
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
from trajectopy.util import show_progress

logger = logging.getLogger("root")


@dataclass
class TrajectoryEntryPair:
    entry: TrajectoryEntry
    request: TrajectoryManagerRequest
    reference_entry: Union[TrajectoryEntry, None] = None

    def __post_init__(self) -> None:
        self.entry.settings.matching.method = self.request.matching_method


class TrajectoryManager(QObject):
    """
    A class that manages trajectories and provides methods for various operations such as alignment, approximation, and comparison.

    Attributes:
        trajectory_model_request (pyqtSignal): A signal emitted when a request for the trajectory model is made.
        result_model_request (pyqtSignal): A signal emitted when a request for the result model is made.
        update_view (pyqtSignal): A signal emitted when the view needs to be updated.

    Possible Requests:
        TrajectoryManagerRequestType.CHANGE_ESPG: Changes the EPSG code of the selected trajectory to the specified EPSG code.
        TrajectoryManagerRequestType.EPSG_TO_REF: Changes the EPSG code of the selected trajectory to the EPSG code of the reference trajectory.
        TrajectoryManagerRequestType.ALIGN: Aligns the selected trajectory to the reference trajectory.
        TrajectoryManagerRequestType.ADAPT_FIRST_POSE: Adopts the position and orientation of the first pose of the current trajectory to the reference trajectory.
        TrajectoryManagerRequestType.ADAPT_FIRST_POSITION: Adopts the position of the first pose of the current trajectory to the reference trajectory.
        TrajectoryManagerRequestType.ADAPT_FIRST_ORIENTATION: Adopts the orientation of the first pose of the current trajectory to the reference trajectory.
        TrajectoryManagerRequestType.APPLY_ALIGNMENT: Applies the selected alignment to the trajectory of the given entry pair.
        TrajectoryManagerRequestType.APPROXIMATE: Approximates the selected trajectory using the settings specified in the entry.
        TrajectoryManagerRequestType.COMPARE: Compares the selected trajectory to the reference trajectory.
        TrajectoryManagerRequestType.INTERPOLATE: Interpolates the trajectory to match the timestamps of the reference trajectory.
        TrajectoryManagerRequestType.MERGE: Merges all selected trajectories into one trajectory.
        TrajectoryManagerRequestType.INTERSECT: Intersects the trajectory with the reference trajectory, keeping only the poses that have a corresponding pose in the reference trajectory.
        TrajectoryManagerRequestType.ADAPT_SAMPLING: Combination of intersection and interpolation. After this, the trajectory will be present in the same time intervals as the reference and will have the same sampling. However, this does not mean that they will have exactely the same number of poses.
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
        self.REQUEST_MAPPING: Dict[TrajectoryManagerRequestType, Any] = {
            TrajectoryManagerRequestType.CHANGE_ESPG: lambda: self.handle_trajectory_operation(
                operation=self.operation_epsg_change,
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
            TrajectoryManagerRequestType.ADAPT_FIRST_POSITION: lambda: self.handle_trajectory_operation(
                operation=self.operation_adopt_first_position,
                inplace=False,
                apply_to_reference=False,
            ),
            TrajectoryManagerRequestType.ADAPT_FIRST_ORIENTATION: lambda: self.handle_trajectory_operation(
                operation=self.operation_adopt_first_orientation,
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
            TrajectoryManagerRequestType.INTERPOLATE: lambda: self.handle_trajectory_operation(
                operation=self.operation_interpolate,
                inplace=False,
                apply_to_reference=False,
            ),
            TrajectoryManagerRequestType.MERGE: self.operation_merge_trajectories,
            TrajectoryManagerRequestType.INTERSECT: lambda: self.handle_trajectory_operation(
                operation=self.operation_intersect,
                inplace=False,
                apply_to_reference=False,
            ),
            TrajectoryManagerRequestType.ADAPT_SAMPLING: lambda: self.handle_trajectory_operation(
                operation=self.operation_adapt_ref_sampling,
                inplace=False,
                apply_to_reference=False,
            ),
            TrajectoryManagerRequestType.MATCH_TIMESTAMPS: lambda: self.handle_trajectory_operation(
                operation=self.operation_match_timestamps,
                inplace=False,
                apply_to_reference=False,
            ),
            TrajectoryManagerRequestType.SORT: lambda: self.handle_trajectory_operation(
                operation=self.operation_sort, inplace=False, apply_to_reference=True
            ),
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

    def selected_trajectory_entries(self, return_reference: bool = True) -> List[TrajectoryEntry]:
        """
        Returns a list of selected trajectory entries.

        Args:
            return_reference: If True, returns the reference trajectory entry as well.

        Returns:
            A list of selected trajectory entries.
        """
        return [entry for entry in self.request.selection.entries if return_reference or not entry.set_as_reference]

    @property
    def reference_entry(self) -> Union[TrajectoryEntry, None]:
        """
        Returns the reference trajectory entry of the current trajectory manager request.

        Returns:
            The reference trajectory entry of the current trajectory manager request.
        """
        return self.request.selection.reference_entry

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
        generic_request_handler(self, request, passthrough_request=False)
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

    @show_progress
    def handle_trajectory_operation(
        self,
        operation: Callable[[TrajectoryEntryPair], Union[tuple, None]],
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

        merged_trajectory = tpy.merge_trajectories([entry.trajectory for entry in selected_entries])

        new_trajectory_entry = TrajectoryEntry(
            full_filename="",
            trajectory=merged_trajectory,
            group_id=selected_entries[0].group_id,
        )
        self.emit_add_trajectory_signal(new_trajectory_entry)

    @staticmethod
    def operation_switch_sorting(
        entry_pair: TrajectoryEntryPair,
    ) -> Tuple[TrajectoryEntry]:
        """
        Changes the sorting of the trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The trajectory entry pair containing the trajectory to be sorted.

        Returns:
            None.
        """
        entry_pair.entry.trajectory.sorting = (
            tpy.Sorting.TIME
            if entry_pair.entry.trajectory.sorting == tpy.Sorting.ARC_LENGTH
            else tpy.Sorting.ARC_LENGTH
        )
        return (entry_pair.entry,)

    @staticmethod
    def operation_intersect(entry_pair: TrajectoryEntryPair) -> Tuple[TrajectoryEntry]:
        """
        Intersects the trajectory with the reference trajectory, keeping only the poses that have a corresponding pose
        in the reference trajectory. The resulting trajectory will not have any poses at timespans where the reference
        trajectory does not have any poses.

        Args:
            entry_pair (TrajectoryEntryPair): The trajectory entry pair containing the trajectory to be intersected.

        Returns:
            TrajectoryEntry: The intersected trajectory entry.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        return (
            TrajectoryEntry(
                full_filename=entry_pair.entry.full_filename,
                trajectory=entry_pair.entry.trajectory.intersect(reference_entry.trajectory.tstamps),
                settings=entry_pair.entry.settings,
                group_id=entry_pair.entry.group_id,
            ),
        )

    @staticmethod
    def operation_interpolate(
        entry_pair: TrajectoryEntryPair,
    ) -> Tuple[TrajectoryEntry]:
        """
        Interpolates the trajectory to match the timestamps of the reference trajectory.
        The resulting trajectory will have the same number of poses, interpolated at the timestamps
        of the reference trajectory. However, this is only possible if the reference timestamps do not exceed the
        timestamps of the trajectory to be interpolated.

        Args:
            entry_pair (TrajectoryEntryPair): The trajectory entry pair containing the trajectory to be interpolated.

        Returns:
            TrajectoryEntry: The interpolated trajectory entry.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        return (
            TrajectoryEntry(
                full_filename=entry_pair.entry.full_filename,
                trajectory=entry_pair.entry.trajectory.interpolate(reference_entry.trajectory.tstamps),
                settings=entry_pair.entry.settings,
                group_id=entry_pair.entry.group_id,
            ),
        )

    @staticmethod
    def operation_adapt_ref_sampling(
        entry_pair: TrajectoryEntryPair,
    ) -> Tuple[TrajectoryEntry]:
        """Combination of intersection and interpolation.
        After this, the trajectory will be present in the
        same time intervals as the reference and will have
        the same sampling. However, this does not mean that
        they will have exactely the same number of poses."""

        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        resampled_trajectory, _ = entry_pair.entry.trajectory.same_sampling(reference_entry.trajectory)
        return (
            TrajectoryEntry(
                full_filename=entry_pair.entry.full_filename,
                trajectory=resampled_trajectory,
                settings=entry_pair.entry.settings,
                group_id=entry_pair.entry.group_id,
            ),
        )

    @staticmethod
    def operation_match_timestamps(
        entry_pair: TrajectoryEntryPair,
    ) -> Tuple[TrajectoryEntry, ...]:
        """
        Matches the timestamps of the two trajectories in the given `TrajectoryEntryPair`.
        After this, both trajectories will have the same number of poses at the same
        points in time. This may result in cropping the reference trajectory.

        Most intrusive time related function which
        - intersects
        - interpolates
        - matches
        the timestamps

        Args:
            entry_pair (TrajectoryEntryPair): The trajectory entry pair containing the trajectories to be matched.

        Returns:
            Tuple[TrajectoryEntry]: A tuple containing the updated trajectory entries for the original and reference trajectories.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        reference_trajectory = reference_entry.trajectory
        current_trajectory = entry_pair.entry.trajectory

        current_trajectory.same_sampling(reference_trajectory)
        current_trajectory.match_timestamps(reference_trajectory.tstamps)
        reference_trajectory.match_timestamps(current_trajectory.tstamps)

        new_ref_trajectory_entry = TrajectoryEntry(
            full_filename=reference_entry.full_filename,
            trajectory=reference_trajectory,
            settings=reference_entry.settings,
            group_id=reference_entry.group_id,
        )

        new_trajectory_entry = TrajectoryEntry(
            full_filename=entry_pair.entry.full_filename,
            trajectory=current_trajectory,
            settings=entry_pair.entry.settings,
            group_id=entry_pair.entry.group_id,
        )
        return new_trajectory_entry, new_ref_trajectory_entry

    @staticmethod
    def operation_sort(entry_pair: TrajectoryEntryPair) -> Tuple[TrajectoryEntry]:
        """
        Sorts the selected trajectory using the settings specified in the entry.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to sort.

        Returns:
            TrajectoryEntry: The sorted trajectory.
        """
        logger.info("Sorting trajectory ...")
        entry_pair.entry.trajectory.sort_spatially(sorting_settings=entry_pair.entry.settings.sorting)
        entry_pair.entry.state.sorting_known = True
        return (
            TrajectoryEntry(
                full_filename=entry_pair.entry.full_filename,
                trajectory=entry_pair.entry.trajectory,
                settings=entry_pair.entry.settings,
                group_id=entry_pair.entry.group_id,
                state=entry_pair.entry.state,
            ),
        )

    @staticmethod
    def operation_approximate(
        entry_pair: TrajectoryEntryPair,
    ) -> Tuple[TrajectoryEntry]:
        """
        Approximates the selected trajectory using the settings specified in the entry.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to approximate.

        Returns:
            TrajectoryEntry: The approximated trajectory.
        """
        entry_pair.entry.trajectory.approximate(approximation_settings=entry_pair.entry.settings.approximation)
        entry_pair.entry.state.approximated = True
        return (
            TrajectoryEntry(
                full_filename=entry_pair.entry.full_filename,
                trajectory=entry_pair.entry.trajectory,
                settings=entry_pair.entry.settings,
                group_id=entry_pair.entry.group_id,
                state=entry_pair.entry.state,
            ),
        )

    @staticmethod
    def operation_compare_abs(entry_pair: TrajectoryEntryPair) -> Tuple[ResultEntry]:
        """
        Compares the selected trajectory to the reference trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to compare.

        Returns:
            ResultEntry: The result of the comparison.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        traj_test, traj_ref = tpy.match_trajectories(
            traj_from=entry_pair.entry.trajectory,
            traj_to=reference_entry.trajectory,
            settings=entry_pair.entry.settings.matching,
        )

        if len(traj_ref) != len(traj_test):
            raise ValueError("Something went wrong during matching.")

        comparison_result = compare_trajectories_absolute(traj_test=traj_test, traj_ref=traj_ref)

        return (AbsoluteDeviationEntry(deviations=comparison_result),)

    @staticmethod
    def operation_compare_rel(entry_pair: TrajectoryEntryPair) -> Tuple[ResultEntry]:
        """
        Compares the selected trajectory to the reference trajectory using relative comparison.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to compare.

        Returns:
            ResultEntry: The result of the comparison.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        traj_test, traj_ref = tpy.match_trajectories(
            traj_from=entry_pair.entry.trajectory,
            traj_to=reference_entry.trajectory,
            settings=entry_pair.entry.settings.matching,
        )

        if len(traj_ref) != len(traj_test):
            raise ValueError("Something went wrong during matching.")

        comparison_result = compare_trajectories_relative(
            traj_test=traj_test,
            traj_ref=traj_ref,
            settings=entry_pair.entry.settings.relative_comparison,
        )

        return (RelativeDeviationEntry(deviations=comparison_result),)

    @staticmethod
    def operation_epsg_change(
        entry_pair: TrajectoryEntryPair,
    ) -> Tuple[TrajectoryEntry]:
        """
        Changes the datum of the selected trajectory to the specified EPSG code.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to change the EPSG code for.

        Returns:
            None
        """
        entry_pair.entry.trajectory.pos.to_epsg(entry_pair.request.target_epsg)
        return (entry_pair.entry,)

    @staticmethod
    def operation_ref_epsg(entry_pair: TrajectoryEntryPair) -> Tuple[TrajectoryEntry]:
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
        reference_epsg = reference_trajectory.pos.epsg
        reference_local_transformer = reference_trajectory.pos.local_transformer

        if reference_local_transformer is None:
            raise ValueError("Reference trajectory has an unknown EPSG code.")

        entry_pair.entry.trajectory.pos.local_transformer = reference_local_transformer
        entry_pair.entry.trajectory.pos.to_epsg(reference_epsg)
        return (entry_pair.entry,)

    @staticmethod
    def operation_align(
        entry_pair: TrajectoryEntryPair,
    ) -> Tuple[TrajectoryEntry, AlignmentEntry]:
        """
        Aligns the selected trajectory to a reference trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to align.

        Returns:
            list[TrajectoryEntry, AlignmentEntry]: A list containing the aligned trajectory and the alignment information.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        alignment_result = tpy.estimate_alignment(
            traj_from=entry_pair.entry.trajectory,
            traj_to=reference_entry.trajectory,
            alignment_settings=entry_pair.entry.settings.alignment,
            matching_settings=entry_pair.entry.settings.matching,
        )

        traj_aligned = entry_pair.entry.trajectory.apply_alignment(alignment_result=alignment_result, inplace=False)
        entry_pair.entry.state.aligned = True

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
    ) -> Tuple[TrajectoryEntry]:
        """
        Applies the selected alignment to the trajectory of the given entry pair.

        Args:
            entry_pair (TrajectoryEntryPair): The entry pair containing the trajectory to be aligned and the reference
                trajectory.

        Returns:
            TrajectoryEntry: A new trajectory entry with the aligned trajectory.
        """
        aligned_trajectory = entry_pair.entry.trajectory.apply_alignment(
            alignment_result=entry_pair.request.alignment.alignment_result, inplace=False
        )
        entry_pair.entry.state.aligned = True
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
    ) -> Tuple[TrajectoryEntry, ...]:
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

        traj_test, traj_ref = tpy.match_trajectories(
            traj_from=entry_pair.entry.trajectory,
            traj_to=reference_entry.trajectory,
            settings=entry_pair.entry.settings.matching,
        )

        traj_aligned = traj_test.adopt_first_pose(traj_to=traj_ref)
        entry_pair.entry.state.aligned = True

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
    def operation_adopt_first_position(
        entry_pair: TrajectoryEntryPair,
    ) -> Tuple[TrajectoryEntry]:
        """
        Adopts the position of the first pose of the current trajectory to the reference trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The entry pair containing the current trajectory and the reference
                trajectory.

        Returns:
            TrajectoryEntry: A new trajectory entry with the adopted position.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        traj_aligned = entry_pair.entry.trajectory.adopt_first_position(traj_to=reference_entry.trajectory)
        entry_pair.entry.state.aligned = True

        return (
            TrajectoryEntry(
                full_filename=entry_pair.entry.full_filename,
                trajectory=traj_aligned,
                settings=entry_pair.entry.settings,
                group_id=entry_pair.entry.group_id,
                state=entry_pair.entry.state,
            ),
        )

    @staticmethod
    def operation_adopt_first_orientation(
        entry_pair: TrajectoryEntryPair,
    ) -> Tuple[TrajectoryEntry]:
        """
        Adopts the orientation of the first pose of the current trajectory to the reference trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The entry pair containing the current trajectory and the reference
                trajectory.

        Returns:
            TrajectoryEntry: A new trajectory entry with the adopted orientation.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        traj_aligned = entry_pair.entry.trajectory.adopt_first_orientation(traj_to=reference_entry.trajectory)
        entry_pair.entry.state.aligned = True

        return (
            TrajectoryEntry(
                full_filename=entry_pair.entry.full_filename,
                trajectory=traj_aligned,
                settings=entry_pair.entry.settings,
                group_id=entry_pair.entry.group_id,
                state=entry_pair.entry.state,
            ),
        )

    @staticmethod
    def operation_match_timestamps_roughly(
        entry_pair: TrajectoryEntryPair,
    ) -> Tuple[TrajectoryEntry]:
        """
        Matches the timestamps of the current trajectory roughly to the reference trajectory.

        Args:
            entry_pair (TrajectoryEntryPair): The entry pair containing the current trajectory and the reference
                trajectory.
        Returns:
            TrajectoryEntry: A new trajectory entry with the roughly matched timestamps
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        time_delay = rough_timestamp_matching(
            traj_test=entry_pair.entry.trajectory, traj_ref=reference_entry.trajectory
        )
        entry_pair.entry.trajectory.tstamps += time_delay
        entry_pair.entry.state.matched = True

        return (
            TrajectoryEntry(
                full_filename=entry_pair.entry.full_filename,
                trajectory=entry_pair.entry.trajectory,
                settings=entry_pair.entry.settings,
                group_id=entry_pair.entry.group_id,
                state=entry_pair.entry.state,
            ),
        )

    @staticmethod
    def operation_match(entry_pair: TrajectoryEntryPair) -> Tuple[TrajectoryEntry, ...]:
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

        traj_test, traj_ref = tpy.match_trajectories(
            traj_from=entry_pair.entry.trajectory,
            traj_to=reference_entry.trajectory,
            settings=entry_pair.entry.settings.matching,
        )
        reference_entry.state.matched = True
        entry_pair.entry.state.matched = True

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
    def operation_ate(entry_pair: TrajectoryEntryPair) -> Tuple[ResultEntry]:
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

        ate_result, alignment_result = tpy.ate(
            trajectory_est=entry_pair.entry.trajectory,
            trajectory_gt=reference_entry.trajectory,
            settings=entry_pair.entry.settings,
            return_alignment=True,
        )

        return (AbsoluteDeviationEntry(deviations=ate_result), AlignmentEntry(alignment_result=alignment_result))

    @staticmethod
    def operation_rpe(entry_pair: TrajectoryEntryPair) -> Tuple[ResultEntry]:
        """
        Computes the relative pose error (RPE)

        Args:
            entry_pair (TrajectoryEntryPair): The pair of trajectories to compare.

        Returns:
            ResultEntry: The result of the comparison.
        """
        if (reference_entry := entry_pair.reference_entry) is None:
            raise ValueError("No reference trajectory selected.")

        rpe_result = tpy.rpe(
            trajectory_est=entry_pair.entry.trajectory,
            trajectory_gt=reference_entry.trajectory,
            settings=entry_pair.entry.settings,
        )

        return (RelativeDeviationEntry(deviations=rpe_result),)
