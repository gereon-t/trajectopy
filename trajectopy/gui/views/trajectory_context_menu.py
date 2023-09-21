"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QAction, QCursor
from trajectopy_core.settings.matching_settings import MatchingMethod
from trajectopy_core.trajectory import Sorting

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

logger = logging.getLogger("root")


class TrajectoryContextMenu(QtWidgets.QMenu):
    trajectory_manager_request = QtCore.pyqtSignal(TrajectoryManagerRequest)
    trajectory_model_request = QtCore.pyqtSignal(TrajectoryModelRequest)
    result_model_request = QtCore.pyqtSignal(ResultModelRequest)
    ui_request = QtCore.pyqtSignal(UIRequest)
    plot_request = QtCore.pyqtSignal(PlotRequest)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.comparison_context_menu = QtWidgets.QMenu("Compare With Reference")
        self.match_context_menu = QtWidgets.QMenu("Match With Reference")
        self.view_context_menu = QtWidgets.QMenu("View")
        self.edit_context_menu = QtWidgets.QMenu("Edit")
        self.action_context_menu = QtWidgets.QMenu("Action")
        self.absolute_menu = QtWidgets.QMenu("Absolute", self)
        self.relative_menu = QtWidgets.QMenu("Relative", self)
        self.least_squares_menu = QtWidgets.QMenu("Least Squares", self)
        self.align_first_pose_menu = QtWidgets.QMenu("Align First Pose", self)
        self.align_with_reference_sub_menu = QtWidgets.QMenu("Align With Reference", self)
        self._selection: TrajectorySelection

    def set_selection(self, selection: TrajectorySelection) -> None:
        self._selection = selection

    def get_selection(self) -> TrajectorySelection:
        return self._selection

    @QtCore.pyqtSlot(TrajectorySelection)
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
        self.comparison_context_menu.clear()
        self.match_context_menu.clear()
        self.view_context_menu.clear()
        self.edit_context_menu.clear()
        self.action_context_menu.clear()
        self.absolute_menu.clear()
        self.relative_menu.clear()
        self.least_squares_menu.clear()
        self.align_first_pose_menu.clear()
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
        self.edit_context()
        self.action_context()

    def edit_context(self) -> None:
        """Actions sub-context menu"""
        single_selection = len(self.get_selection()) == 1
        self.addMenu(self.edit_context_menu)

        change_datum_action = QAction("EPSG", self)
        change_datum_action.triggered.connect(
            lambda: self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.EPSG_SELECTION,
                    trajectory_selection=self.get_selection(),
                )
            )
        )
        self.edit_context_menu.addAction(change_datum_action)

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

        if not single_selection:
            return

        self.edit_single()

    def edit_single(self):
        selected_trajectory = self.get_selection().entries[0].trajectory

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

        if selected_trajectory.state.sorting_known:
            target_sorting = Sorting.SPATIAL if selected_trajectory.sorting == Sorting.CHRONO else Sorting.CHRONO
            sort_description = (
                "Switch Sorting To Spatial" if target_sorting == Sorting.SPATIAL else "Switch Sorting To Chronological"
            )
            sort_action = QAction(sort_description, self)
            sort_action.triggered.connect(
                lambda: self.trajectory_manager_request.emit(
                    TrajectoryManagerRequest(
                        type=TrajectoryManagerRequestType.SWITCH_SORTING,
                        selection=self.get_selection(),
                        sorting=target_sorting,
                    )
                )
            )
            self.edit_context_menu.addAction(sort_action)

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

        plot_2d_action = QAction("Trajectory 2D", self)
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

        plot_3d_action = QAction("Trajectory 3D", self)
        plot_3d_action.triggered.connect(
            lambda: self.plot_request.emit(
                PlotRequest(
                    type=PlotRequestType.TRAJECTORIES,
                    trajectory_selection=self.get_selection(),
                    dimension=3,
                )
            )
        )
        self.view_context_menu.addAction(plot_3d_action)

        plot_laps_action = QAction("Trajectory Laps", self)
        plot_laps_action.triggered.connect(
            lambda: self.plot_request.emit(
                PlotRequest(
                    type=PlotRequestType.TRAJECTORY_LAPS,
                    trajectory_selection=self.get_selection(),
                )
            )
        )
        plot_laps_enabled = (
            len(self.get_selection()) == 1 and self.get_selection().entries[0].trajectory.state.sorting_known
        )
        plot_laps_action.setEnabled(plot_laps_enabled)
        if plot_laps_enabled:
            self.view_context_menu.addAction(plot_laps_action)

    def action_context(self) -> None:
        self.addMenu(self.action_context_menu)

        self.action_context_menu.addMenu(self.align_context())
        self.action_context_menu.addMenu(self.compare_context())
        self.action_context_menu.addMenu(self.match_context())

        sort_action = QAction("Sort Spatially", self)
        sort_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.SORT,
                    selection=self.get_selection(),
                )
            )
        )
        self.action_context_menu.addAction(sort_action)

        approximate_action = QAction("Approximate", self)
        approximate_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.APPROXIMATE,
                    selection=self.get_selection(),
                )
            )
        )
        self.action_context_menu.addAction(approximate_action)

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
        self.action_context_menu.addAction(apply_alignment_action)

    def align_context(self) -> QtWidgets.QMenu:
        self.align_with_reference_sub_menu.setEnabled(self.get_selection().reference_is_set)

        align_interpolate_action = QAction("Match by Interpolation", self)
        align_interpolate_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.ALIGN,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.INTERPOLATION,
                )
            ),
        )

        align_nearest_temporal_action = QAction("Match Nearest Temporal", self)
        align_nearest_temporal_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.ALIGN,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.NEAREST_TEMPORAL,
                )
            ),
        )

        align_nearest_spatial_action = QAction("Match Nearest Spatial", self)
        align_nearest_spatial_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.ALIGN,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.NEAREST_SPATIAL,
                )
            ),
        )

        align_first_pose_interpolation_action = QAction("Match by Interpolation", self)
        align_first_pose_interpolation_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.ADAPT_FIRST_POSE,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.INTERPOLATION,
                )
            ),
        )

        align_first_pose_nearest_temporal_action = QAction("Match Nearest Temporal", self)
        align_first_pose_nearest_temporal_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.ADAPT_FIRST_POSE,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.NEAREST_TEMPORAL,
                )
            ),
        )

        align_first_pose_nearest_spatial_action = QAction("Match Nearest Spatial", self)
        align_first_pose_nearest_spatial_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.ADAPT_FIRST_POSE,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.NEAREST_SPATIAL,
                )
            ),
        )

        self.least_squares_menu.addAction(align_interpolate_action)
        self.least_squares_menu.addAction(align_nearest_temporal_action)
        self.least_squares_menu.addAction(align_nearest_spatial_action)
        self.align_with_reference_sub_menu.addMenu(self.least_squares_menu)

        self.align_first_pose_menu.addAction(align_first_pose_interpolation_action)
        self.align_first_pose_menu.addAction(align_first_pose_nearest_temporal_action)
        self.align_first_pose_menu.addAction(align_first_pose_nearest_spatial_action)
        self.align_with_reference_sub_menu.addMenu(self.align_first_pose_menu)

        return self.align_with_reference_sub_menu

    def compare_context(self) -> QtWidgets.QMenu:
        """Compare-Sub-Context menu"""
        self.comparison_context_menu.setEnabled(self.get_selection().reference_is_set)

        abs_nearest_spatial_action = QAction("Match Nearest Spatial", self)
        abs_nearest_spatial_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.COMPARE_ABS,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.NEAREST_SPATIAL,
                )
            ),
        )

        abs_nearest_temporal_action = QAction("Match Nearest Temporal", self)
        abs_nearest_temporal_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.COMPARE_ABS,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.NEAREST_TEMPORAL,
                )
            ),
        )

        abs_interpolation_action = QAction("Match by Interpolation", self)
        abs_interpolation_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.COMPARE_ABS,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.INTERPOLATION,
                )
            ),
        )

        rel_nearest_spatial_action = QAction("Match Nearest Spatial", self)
        rel_nearest_spatial_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.COMPARE_REL,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.NEAREST_SPATIAL,
                )
            ),
        )

        rel_nearest_temporal_action = QAction("Match Nearest Temporal", self)
        rel_nearest_temporal_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.COMPARE_REL,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.NEAREST_TEMPORAL,
                )
            ),
        )

        rel_interpolation_action = QAction("Match by Interpolation", self)
        rel_interpolation_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.COMPARE_REL,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.INTERPOLATION,
                )
            ),
        )

        self.absolute_menu.addAction(abs_interpolation_action)
        self.absolute_menu.addAction(abs_nearest_temporal_action)
        self.absolute_menu.addAction(abs_nearest_spatial_action)

        self.relative_menu.addAction(rel_interpolation_action)
        self.relative_menu.addAction(rel_nearest_temporal_action)
        self.relative_menu.addAction(rel_nearest_spatial_action)

        self.comparison_context_menu.addMenu(self.absolute_menu)
        self.comparison_context_menu.addMenu(self.relative_menu)
        return self.comparison_context_menu

    def match_context(self) -> QtWidgets.QMenu:
        self.match_context_menu.setEnabled(self.get_selection().reference_is_set)

        match_interpolation_action = QAction("Interpolation", self)
        match_interpolation_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.MATCH,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.INTERPOLATION,
                )
            ),
        )

        match_nearest_temporal_action = QAction("Nearest Temporal", self)
        match_nearest_temporal_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.MATCH,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.NEAREST_TEMPORAL,
                )
            ),
        )

        match_nearest_spatial_action = QAction("Nearest Spatial", self)
        match_nearest_spatial_action.triggered.connect(
            lambda: self.trajectory_manager_request.emit(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.MATCH,
                    selection=self.get_selection(),
                    matching_method=MatchingMethod.NEAREST_SPATIAL,
                )
            ),
        )

        self.match_context_menu.addAction(match_interpolation_action)
        self.match_context_menu.addAction(match_nearest_temporal_action)
        self.match_context_menu.addAction(match_nearest_spatial_action)

        return self.match_context_menu
