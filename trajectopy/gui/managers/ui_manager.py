import functools
import logging

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QObject, Signal, Slot

from trajectopy.gui.managers.requests import (
    FileRequest,
    FileRequestType,
    ResultModelRequest,
    ResultModelRequestType,
    SessionManagerRequest,
    SessionManagerRequestType,
    TrajectoryManagerRequest,
    TrajectoryManagerRequestType,
    UIRequest,
    UIRequestType,
    generic_request_handler,
)
from trajectopy.gui.models.entries import AbsoluteDeviationEntry, AlignmentEntry
from trajectopy.gui.models.selection import TrajectorySelection
from trajectopy.gui.utils import (
    browse_dir_dialog,
    read_file_dialog,
    save_file_dialog,
    show_msg_box,
)
from trajectopy.gui.views.alignment_edit_window import AlignmentEditWindow
from trajectopy.gui.views.dof_organizer import DOFOrganizer
from trajectopy.gui.views.json_settings_view import JSONViewer
from trajectopy.gui.views.manual_alignment_window import ManualAlignmentWindow
from trajectopy.gui.views.playback_window import PlaybackWindow
from trajectopy.gui.views.result_selection_window import AlignmentSelector
from trajectopy.io.kml import ate_kml_fields
from trajectopy.results.alignment_result import AlignmentResult

logger = logging.getLogger(__name__)


class UIManager(QObject):
    """
    Manager for handling UI requests.

    Possible requests:
    - EPSG selection
    - Alignment selection
    - Import trajectory
    - Import result
    - Export trajectory
    - Export result
    - Message box
    - Show trajectory properties
    - Show result properties
    - Show trajectory settings
    - Show reset question
    - Export session
    - Import session
    - Edit alignment

    """

    trajectory_manager_request = Signal(TrajectoryManagerRequest)
    result_model_request = Signal(ResultModelRequest)
    file_request = Signal(FileRequest)
    session_manager_request = Signal(SessionManagerRequest)
    operation_finished = Signal()
    ui_request = Signal(UIRequest)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sub_windows: list[QtWidgets.QWidget] = []
        self._kml_color = QtGui.QColor("#ff0000")  # last color chosen for Google Earth exports
        self._ate_kml_field = "position"  # last field chosen to color ATE results in Google Earth
        self.REQUEST_MAPPING = {
            UIRequestType.EPSG_TRANSFORMATION: self.epsg_input,
            UIRequestType.EPSG_EDIT: self.epsg_input_edit,
            UIRequestType.ALIGNMENT_SELECTION: self.alignment_selection,
            UIRequestType.IMPORT_TRAJ: self.trajectory_import_dialog,
            UIRequestType.IMPORT_RES: self.result_import_dialog,
            UIRequestType.EXPORT_TRAJ: self.trajectory_export_dialog,
            UIRequestType.EXPORT_KML: self.kml_export_dialog,
            UIRequestType.EXPORT_RES: self.result_export_dialog,
            UIRequestType.MESSAGE: self.message_box,
            UIRequestType.DOF_ORGANIZER: self.show_dof_organizer,
            UIRequestType.TRAJ_SETTINGS: self.show_trajectory_settings,
            UIRequestType.CONFIRM_RESET: self.show_reset_question,
            UIRequestType.EXPORT_SESSION: self.session_export_dialog,
            UIRequestType.IMPORT_SESSION: self.session_import_dialog,
            UIRequestType.EDIT_ALIGNMENT: self.edit_alignment,
            UIRequestType.MANUAL_ALIGNMENT: self.manual_alignment,
            UIRequestType.GRID_SELECTION: self.grid_selection,
            UIRequestType.PLAYBACK: self.trajectory_playback,
        }

    @Slot(UIRequest)
    def handle_request(self, request: UIRequest) -> None:
        generic_request_handler(self, request)

    def message_box(self, request: UIRequest) -> None:
        show_msg_box(request.message)

    def _track_window(self, window: QtWidgets.QWidget) -> None:
        """Close any previously tracked sub-windows and track the new one."""
        for w in self._sub_windows:
            w.close()
        self._sub_windows.clear()
        self._sub_windows.append(window)

    def show_dof_organizer(self, request: UIRequest) -> None:
        dof_organizer = DOFOrganizer(parent=self.parent(), selection=request.trajectory_selection)
        dof_organizer.selection_made.connect(
            functools.partial(self.handle_dof_selection, selection=request.trajectory_selection)
        )
        self._track_window(dof_organizer)
        dof_organizer.show()

    def show_trajectory_settings(self, request: UIRequest) -> None:
        settings_window = JSONViewer(
            parent=self.parent(),
            settings=request.trajectory_selection.entries[0].settings,
        )
        self._track_window(settings_window)
        settings_window.show()

    def trajectory_playback(self, request: UIRequest) -> None:
        trajectories = [entry.trajectory for entry in request.trajectory_selection.entries]
        playback_window = PlaybackWindow(trajectories=trajectories, parent=self.parent())
        playback_window.show()

    def manual_alignment(self, request: UIRequest) -> None:
        selection = request.trajectory_selection
        if (reference_entry := selection.reference_entry) is None:
            show_msg_box("No reference trajectory selected.")
            return

        entry = next((e for e in selection.entries if not e.set_as_reference), None)
        if entry is None:
            show_msg_box("Select a trajectory that is not the reference.")
            return

        manual_window = ManualAlignmentWindow(
            trajectory=entry.trajectory,
            reference=reference_entry.trajectory,
            matching_settings=entry.settings.matching,
            alignment_settings=entry.settings.alignment,
            parent=self.parent(),
        )
        manual_window.alignment_accepted.connect(
            functools.partial(self.handle_manual_alignment, selection=request.trajectory_selection)
        )
        self._track_window(manual_window)
        manual_window.show()

    def handle_manual_alignment(self, alignment_result: AlignmentResult, *, selection: TrajectorySelection) -> None:
        self.trajectory_manager_request.emit(
            TrajectoryManagerRequest(
                type=TrajectoryManagerRequestType.ALIGN,
                selection=selection,
                manual_alignment=alignment_result,
            )
        )

    def alignment_selection(self, request: UIRequest) -> None:
        alignment_entries = [entry for entry in request.result_selection.entries if isinstance(entry, AlignmentEntry)]
        result_selection = AlignmentSelector(parent=self.parent(), alignments=alignment_entries)
        result_selection.selection_made.connect(
            functools.partial(self.handle_alignment_selection, selection=request.trajectory_selection)
        )
        self._track_window(result_selection)
        result_selection.show()

    def handle_alignment_selection(
        self, selected_alignment: AlignmentEntry, *, selection: TrajectorySelection
    ) -> None:
        self.trajectory_manager_request.emit(
            TrajectoryManagerRequest(
                type=TrajectoryManagerRequestType.APPLY_ALIGNMENT,
                selection=selection,
                alignment=selected_alignment,
            )
        )

    def handle_dof_selection(self, dof_mapping: dict, *, selection: TrajectorySelection) -> None:
        self.trajectory_manager_request.emit(
            TrajectoryManagerRequest(
                type=TrajectoryManagerRequestType.REARANGE_DOF,
                selection=selection,
                dof_mapping=dof_mapping,
            )
        )

    def trajectory_import_dialog(self, _: UIRequest) -> None:
        if selected_files := read_file_dialog(None, file_filter="Trajectory Files (*.traj);;All Files (*.*)"):
            self.file_request.emit(FileRequest(type=FileRequestType.READ_TRAJ, file_list=selected_files))
        else:
            return

    def trajectory_export_dialog(self, request: UIRequest) -> None:
        if selected_file := save_file_dialog(None, file_filter="Trajectory File (*.traj)"):
            self.file_request.emit(
                FileRequest(
                    type=FileRequestType.WRITE_TRAJ,
                    file_list=[selected_file],
                    trajectory_selection=request.trajectory_selection,
                )
            )
        else:
            return

    def kml_export_dialog(self, request: UIRequest) -> None:
        """Google Earth export: of a trajectory in a chosen color, or of an ATE result colored by a chosen field."""
        if request.result_selection.entries:
            self._ate_kml_export_dialog(request)
            return
        trajectory = request.trajectory_selection.entries[0].trajectory
        color = QtWidgets.QColorDialog.getColor(self._kml_color, self._dialog_parent(), "Track Color in Google Earth")
        if not color.isValid():  # canceled
            return
        self._kml_color = color
        if selected_file := save_file_dialog(None, file_filter="Google Earth (*.kml)"):
            try:
                trajectory.to_kml(selected_file, color=color.name())
            except ValueError as error:
                show_msg_box(str(error))

    def _ate_kml_export_dialog(self, request: UIRequest) -> None:
        entry = request.result_selection.entries[0]
        if not isinstance(entry, AbsoluteDeviationEntry):
            return
        fields = ate_kml_fields(entry.deviations)
        labels = [f"{field.label} [{field.unit}]" for field in fields]
        keys = [field.key for field in fields]
        current = keys.index(self._ate_kml_field) if self._ate_kml_field in keys else 0
        label, accepted = QtWidgets.QInputDialog.getItem(
            self._dialog_parent(), "Google Earth Export", "Color the trajectory by:", labels, current, False
        )
        if not accepted:
            return
        self._ate_kml_field = keys[labels.index(label)]
        if selected_file := save_file_dialog(None, file_filter="Google Earth (*.kml)"):
            try:
                entry.deviations.to_kml(selected_file, field=self._ate_kml_field)
            except ValueError as error:
                show_msg_box(str(error))

    def _dialog_parent(self) -> QtWidgets.QWidget | None:
        parent = self.parent()
        return parent if isinstance(parent, QtWidgets.QWidget) else None

    def result_export_dialog(self, request: UIRequest) -> None:
        if selected_file := save_file_dialog(None, file_filter="Result File (*.result)"):
            self.file_request.emit(
                FileRequest(
                    type=FileRequestType.WRITE_RES,
                    file_list=[selected_file],
                    result_selection=request.result_selection,
                )
            )
        else:
            return

    def result_import_dialog(self, _: UIRequest) -> None:
        if selected_files := read_file_dialog(None, file_filter="Result Files (*.result);;All Files (*.*)"):
            self.file_request.emit(FileRequest(type=FileRequestType.READ_RES, file_list=selected_files))
        else:
            return

    def show_reset_question(self, _: UIRequest) -> None:
        reply = QtWidgets.QMessageBox.question(
            None,
            "Confirmation",
            "The current Session will be lost. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.session_manager_request.emit(SessionManagerRequest(type=SessionManagerRequestType.NEW))

    def session_export_dialog(self, _: UIRequest) -> None:
        if selected_file := browse_dir_dialog(None):
            self.session_manager_request.emit(
                SessionManagerRequest(type=SessionManagerRequestType.EXPORT, file_path=selected_file)
            )
        else:
            return

    def session_import_dialog(self, _: UIRequest) -> None:
        if selected_file := browse_dir_dialog(None):
            self.session_manager_request.emit(
                SessionManagerRequest(type=SessionManagerRequestType.NEW, file_path=selected_file)
            )
            self.session_manager_request.emit(
                SessionManagerRequest(type=SessionManagerRequestType.IMPORT, file_path=selected_file)
            )
        else:
            return

    def epsg_input(self, request: UIRequest) -> None:
        epsg, ok = QtWidgets.QInputDialog.getInt(
            None, "EPSG Dialog", "Please enter the EPSG code to which the trajectory should be transformed:"
        )

        if not ok or epsg is None:
            return

        self.trajectory_manager_request.emit(
            TrajectoryManagerRequest(
                type=TrajectoryManagerRequestType.CHANGE_ESPG,
                selection=request.trajectory_selection,
                target_epsg=epsg,
            )
        )

    def grid_selection(self, request: UIRequest) -> None:
        step, ok = QtWidgets.QInputDialog.getDouble(
            None,
            "Interpolate to Grid",
            "Enter the grid step size (seconds):",
            0.1,
            1e-6,
            3600.0,
            6,
        )

        if not ok or step <= 0:
            return

        self.trajectory_manager_request.emit(
            TrajectoryManagerRequest(
                type=TrajectoryManagerRequestType.INTERPOLATE_TO_GRID,
                selection=request.trajectory_selection,
                grid=step,
            )
        )

    def epsg_input_edit(self, request: UIRequest) -> None:
        epsg, ok = QtWidgets.QInputDialog.getInt(
            None,
            "EPSG Dialog",
            "Please enter the new EPSG code. The trajectory WILL NOT be transformed (use Action -> Transform to EPSG instead):",
        )

        if not ok or epsg is None:
            return

        self.trajectory_manager_request.emit(
            TrajectoryManagerRequest(
                type=TrajectoryManagerRequestType.EDIT_EPSG,
                selection=request.trajectory_selection,
                target_epsg=epsg,
            )
        )

    def edit_alignment(self, request: UIRequest) -> None:
        alignment_window = AlignmentEditWindow(parent=self.parent(), alignment_entry=request.alignment_entry)
        alignment_window.update_signal.connect(
            lambda: self.result_model_request.emit(
                ResultModelRequest(
                    type=ResultModelRequestType.UPDATE,
                    selection=request.result_selection,
                )
            )
        )
        self._track_window(alignment_window)
        alignment_window.show()
