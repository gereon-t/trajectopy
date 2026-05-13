import logging

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QAction, QCursor

from trajectopy.gui.managers.requests import (
    PlotRequest,
    PlotRequestType,
    ResultModelRequest,
    ResultModelRequestType,
    TrajectoryManagerRequest,
    TrajectoryManagerRequestType,
    TrajectoryModelRequest,
    TrajectoryModelRequestType,
    UIRequest,
    UIRequestType,
)
from trajectopy.gui.models.selection import TrajectorySelection

logger = logging.getLogger(__name__)


class TrajectoryContextMenu(QtWidgets.QMenu):
    trajectory_manager_request = QtCore.Signal(TrajectoryManagerRequest)
    trajectory_model_request = QtCore.Signal(TrajectoryModelRequest)
    result_model_request = QtCore.Signal(ResultModelRequest)
    ui_request = QtCore.Signal(UIRequest)
    plot_request = QtCore.Signal(PlotRequest)
    show_timeline = QtCore.Signal(object)  # emits list[TrajectoryEntry]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.evaluate_context_menu = QtWidgets.QMenu("Evaluate")
        self.comparison_context_menu = QtWidgets.QMenu("Compare With Reference")
        self.other_context_menu = QtWidgets.QMenu("Utilities")
        self.view_context_menu = QtWidgets.QMenu("View")
        self.edit_context_menu = QtWidgets.QMenu("Edit")
        self.process_context_menu = QtWidgets.QMenu("Process")
        self.align_with_reference_sub_menu = QtWidgets.QMenu("Align With Reference", self)
        self._selection: TrajectorySelection

    def set_selection(self, selection: TrajectorySelection) -> None:
        self._selection = selection

    def get_selection(self) -> TrajectorySelection:
        return self._selection

    @QtCore.Slot(TrajectorySelection)
    def show_context_menu(self, selection: TrajectorySelection):
        if not selection:
            return

        self.clear()
        self.set_selection(selection)

        if len(selection) == 1:
            self.single_only_context()

        self.multi_context()
        self.exec(QCursor.pos())

    def clear(self) -> None:
        super().clear()
        self.evaluate_context_menu.clear()
        self.comparison_context_menu.clear()
        self.other_context_menu.clear()
        self.view_context_menu.clear()
        self.edit_context_menu.clear()
        self.process_context_menu.clear()
        self.align_with_reference_sub_menu.clear()

    def single_only_context(self) -> None:
        """These options are only shown for single trajectories"""
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(
            lambda: self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.TRAJ_SETTINGS,
                    trajectory_selection=self.get_selection(),
                )
            )
        )

        if self.get_selection().entries[0].set_as_reference:
            reference_action = QAction("Unset As Reference", self)
            reference_action.triggered.connect(
                lambda: self.trajectory_model_request.emit(
                    TrajectoryModelRequest(type=TrajectoryModelRequestType.UNSET_REFERENCE)
                )
            )
        else:
            reference_action = QAction("Set As Reference", self)
            reference_action.triggered.connect(
                lambda: self.trajectory_model_request.emit(
                    TrajectoryModelRequest(
                        type=TrajectoryModelRequestType.SET_REFERENCE,
                        selection=self.get_selection(),
                    )
                )
            )

        self.addAction(reference_action)
        self.addAction(settings_action)

    def multi_context(self) -> None:
        """These options are shown for single and multiple trajectories"""
        self.view_context()

        change_datum_action = QAction("Transform to EPSG", self)
        change_datum_action.triggered.connect(
            lambda: self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.EPSG_TRANSFORMATION,
                    trajectory_selection=self.get_selection(),
                )
            )
        )
        self.addAction(change_datum_action)

        if self.get_selection().reference_is_set:
            self.addSeparator()
            self.evaluate_context()
            self.addAction(self.match_action())
            self.addMenu(self.align_context())

        self.addSeparator()
        self.edit_context()
        self.process_context()

    def edit_context(self) -> None:
        """Edit sub-context menu"""
        single_selection = len(self.get_selection()) == 1
        self.addMenu(self.edit_context_menu)

        edit_epsg_action = QAction("EPSG", self)
        edit_epsg_action.triggered.connect(
            lambda: self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.EPSG_EDIT,
                    trajectory_selection=self.get_selection(),
                )
            )
        )
        self.edit_context_menu.addAction(edit_epsg_action)

        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(
            lambda: self.trajectory_model_request.emit(
                TrajectoryModelRequest(type=TrajectoryModelRequestType.COPY, selection=self.get_selection())
            )
        )
        self.edit_context_menu.addAction(copy_action)

        remove_action = QAction("Remove", self)
        remove_action.triggered.connect(
            lambda: self.trajectory_model_request.emit(
                TrajectoryModelRequest(
                    type=TrajectoryModelRequestType.REMOVE,
                    selection=self.get_selection(),
                )
            )
        )
        self.edit_context_menu.addAction(remove_action)

        remove_related_action = QAction("Remove All Related", self)
        remove_related_action.triggered.connect(
            lambda: self.trajectory_model_request.emit(
                TrajectoryModelRequest(
                    type=TrajectoryModelRequestType.REMOVE_RELATED,
                    selection=self.get_selection(),
                )
            )
        )
        self.edit_context_menu.addAction(remove_related_action)

        merge_action = QAction("Merge", self)
        merge_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.MERGE,
                    selection=self.get_selection(),
                )
            )
        )
        self.edit_context_menu.addAction(merge_action)

        if self.get_selection().entries[0].state.sorting_known:
            switch_sorting_action = QAction("Switch Sorting", self)
            switch_sorting_action.triggered.connect(
                lambda: self.trajectory_manager_request.emit(
                    TrajectoryManagerRequest(
                        type=TrajectoryManagerRequestType.SWITCH_SORTING,
                        selection=self.get_selection(),
                    )
                )
            )
            self.edit_context_menu.addAction(switch_sorting_action)

        dof_action = QAction("Reorganize DOFs", self)
        dof_action.triggered.connect(
            lambda: self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.DOF_ORGANIZER,
                    trajectory_selection=self.get_selection(),
                )
            )
        )
        self.edit_context_menu.addAction(dof_action)

        if single_selection:
            self.edit_single()

    def edit_single(self):
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(
            lambda: self.trajectory_model_request.emit(
                TrajectoryModelRequest(
                    type=TrajectoryModelRequestType.RENAME,
                    selection=self.get_selection(),
                )
            )
        )
        self.edit_context_menu.addAction(rename_action)

        export_action = QAction("Export", self)
        export_action.triggered.connect(
            lambda: self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.EXPORT_TRAJ,
                    trajectory_selection=self.get_selection(),
                )
            )
        )
        self.edit_context_menu.addAction(export_action)

    def view_context(self) -> None:
        """View Sub-Context Menu"""
        self.addMenu(self.view_context_menu)

        property_action = QAction("Properties", self)
        property_action.triggered.connect(
            lambda: self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.TRAJ_PROPERTIES,
                    trajectory_selection=self.get_selection(),
                )
            )
        )
        self.view_context_menu.addAction(property_action)

        timeline_action = QAction("Timeline", self)
        timeline_action.triggered.connect(lambda: self.show_timeline.emit(self.get_selection().entries))
        self.view_context_menu.addAction(timeline_action)

        plot_2d_action = QAction("Plot", self)
        plot_2d_action.triggered.connect(
            lambda: self.plot_request.emit(
                PlotRequest(
                    type=PlotRequestType.TRAJECTORIES,
                    trajectory_selection=self.get_selection(),
                    dimension=2,
                )
            )
        )
        self.view_context_menu.addAction(plot_2d_action)

        playback_action = QAction("Playback", self)
        playback_action.triggered.connect(
            lambda: self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.PLAYBACK,
                    trajectory_selection=self.get_selection(),
                )
            )
        )
        self.view_context_menu.addAction(playback_action)

    def process_context(self) -> None:
        self.addMenu(self.process_context_menu)

        approximate_action = QAction("Approximate", self)
        approximate_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.APPROXIMATE,
                    selection=self.get_selection(),
                )
            )
        )
        self.process_context_menu.addAction(approximate_action)

        interpolate_to_grid_action = QAction("Interpolate to Grid", self)
        interpolate_to_grid_action.triggered.connect(
            lambda: self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.GRID_SELECTION,
                    trajectory_selection=self.get_selection(),
                )
            )
        )
        self.process_context_menu.addAction(interpolate_to_grid_action)

        apply_alignment_action = QAction("Apply Alignment", self)
        # -> ResultModelRequest(selection=selected_trajectories) -> ResultModel
        # -> UIRequest(result_selection=Alignments, trajectory_selection=selected_trajectories)
        # -> TrajectoryManagerRequest(selection=selected_trajectories, alignment_entry=selected_alignment)
        # -> TrajectoryModelRequest(ADD, selection=aligned_trajectory)
        apply_alignment_action.triggered.connect(
            lambda: self.result_model_request.emit(
                ResultModelRequest(
                    type=ResultModelRequestType.PASS_ALIGNMENTS_TO_UI,
                    trajectory_selection=self.get_selection(),
                )
            )
        )
        self.process_context_menu.addAction(apply_alignment_action)

        self.process_context_menu.addMenu(self.other_context())

        if not self.get_selection().reference_is_set:
            return

        if self.get_selection().reference_entry.trajectory.positions.local_transformer is None:
            return

        epsg_to_ref_action = QAction("Adapt EPSG from Reference", self)
        epsg_to_ref_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.EPSG_TO_REF,
                    selection=self.get_selection(),
                )
            )
        )
        self.process_context_menu.addAction(epsg_to_ref_action)

    def other_context(self) -> QtWidgets.QMenu:
        sort_action = QAction("Sort Spatially", self)
        sort_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.SORT,
                    selection=self.get_selection(),
                )
            )
        )
        self.other_context_menu.addAction(sort_action)

        divide_into_laps_action = QAction("Divide into Laps", self)
        divide_into_laps_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.DIVIDE_INTO_LAPS,
                    selection=self.get_selection(),
                )
            )
        )
        self.other_context_menu.addAction(divide_into_laps_action)

        average_action = QAction("Average", self)
        average_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.AVERAGE,
                    selection=self.get_selection(),
                )
            )
        )
        self.other_context_menu.addAction(average_action)
        return self.other_context_menu

    def align_context(self) -> QtWidgets.QMenu:
        self.align_with_reference_sub_menu.setEnabled(self.get_selection().reference_is_set)

        align_all_poses_action = QAction("All Poses", self)
        align_all_poses_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.ALIGN,
                    selection=self.get_selection(),
                )
            ),
        )

        align_first_pose_action = QAction("First Pose", self)
        align_first_pose_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.ADAPT_FIRST_POSE,
                    selection=self.get_selection(),
                )
            ),
        )

        self.align_with_reference_sub_menu.addAction(align_all_poses_action)
        self.align_with_reference_sub_menu.addAction(align_first_pose_action)

        return self.align_with_reference_sub_menu

    def match_action(self) -> QAction:
        match_action = QAction("Match With Reference", self)
        match_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.MATCH,
                    selection=self.get_selection(),
                )
            ),
        )

        return match_action

    def evaluate_context(self) -> None:
        self.addMenu(self.evaluate_context_menu)
        self.evaluate_context_menu.setEnabled(self.get_selection().reference_is_set)

        ate = QAction("ATE", self)
        ate.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.ATE,
                    selection=self.get_selection(),
                )
            ),
        )

        rpe_action = QAction("RPE", self)
        rpe_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.RPE,
                    selection=self.get_selection(),
                )
            ),
        )

        self.evaluate_context_menu.addAction(ate)
        self.evaluate_context_menu.addAction(rpe_action)
