"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging

from PyQt6 import QtWidgets
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from trajectopy_core.util.datahandling import merge_dicts
from trajectopy_core.util.entries import AlignmentEntry

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
from trajectopy.gui.util import browse_dir_dialog, read_file_dialog, save_file_dialog, show_msg_box
from trajectopy.gui.views.alignment_edit_window import AlignmentEditWindow
from trajectopy.gui.views.properties_window import PropertiesGUI
from trajectopy.gui.views.result_selection_window import AlignmentSelector
from trajectopy.gui.views.settings_window import SettingsGUI

logger = logging.getLogger("root")


class UIManager(QObject):
    """
    Manager for handling UI requests.

    Possible requests:
    - EPSG selection
    - Grid selection
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

    trajectory_manager_request = pyqtSignal(TrajectoryManagerRequest)
    result_model_request = pyqtSignal(ResultModelRequest)
    file_request = pyqtSignal(FileRequest)
    session_manager_request = pyqtSignal(SessionManagerRequest)
    operation_finished = pyqtSignal()
    ui_request = pyqtSignal(UIRequest)
    request: UIRequest

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.REQUEST_MAPPING = {
            UIRequestType.EPSG_SELECTION: self.epsg_input,
            UIRequestType.GRID_SELECTION: self.grid_input,
            UIRequestType.ALIGNMENT_SELECTION: self.alignment_selection,
            UIRequestType.IMPORT_TRAJ: self.trajectory_import_dialog,
            UIRequestType.IMPORT_RES: self.result_import_dialog,
            UIRequestType.EXPORT_TRAJ: self.trajectory_export_dialog,
            UIRequestType.EXPORT_RES: self.result_export_dialog,
            UIRequestType.MESSAGE: self.message_box,
            UIRequestType.TRAJ_PROPERTIES: self.show_trajectory_properties,
            UIRequestType.RES_PROPERTIES: self.show_result_properties,
            UIRequestType.TRAJ_SETTINGS: self.show_trajectory_settings,
            UIRequestType.CONFIRM_RESET: self.show_reset_question,
            UIRequestType.EXPORT_SESSION: self.session_export_dialog,
            UIRequestType.IMPORT_SESSION: self.session_import_dialog,
            UIRequestType.EDIT_ALIGNMENT: self.edit_alignment,
        }

    @pyqtSlot(UIRequest)
    def handle_request(self, request: UIRequest) -> None:
        self.request = request
        generic_request_handler(self, request, passthrough_request=True)

    def message_box(self, request: UIRequest) -> None:
        show_msg_box(request.message)

    def show_trajectory_properties(self, request: UIRequest) -> None:
        property_window = PropertiesGUI(parent=self.parent(), num_cols=len(request.trajectory_selection.entries) + 1)
        property_window.reset()
        merged_properties = merge_dicts(tuple(entry.property_dict for entry in request.trajectory_selection.entries))
        property_window.add_from_dict(merged_properties)
        property_window.show()

    def show_result_properties(self, request: UIRequest) -> None:
        property_window = PropertiesGUI(parent=self.parent(), num_cols=len(request.result_selection.entries) + 1)
        property_window.reset()
        merged_properties = merge_dicts(tuple(entry.property_dict for entry in request.result_selection.entries))
        property_window.add_from_dict(merged_properties)
        property_window.show()

    def show_trajectory_settings(self, request: UIRequest) -> None:
        settings_window = SettingsGUI(
            parent=self.parent(),
            trajectory_entry=request.trajectory_selection.entries[0],
        )
        settings_window.show()

    def alignment_selection(self, request: UIRequest) -> None:
        alignment_entries = [entry for entry in request.result_selection.entries if isinstance(entry, AlignmentEntry)]
        result_selection = AlignmentSelector(parent=self.parent(), alignments=alignment_entries)
        result_selection.selection_made.connect(self.handle_alignment_selection)
        result_selection.show()

    @pyqtSlot(AlignmentEntry)
    def handle_alignment_selection(self, selected_alignment: AlignmentEntry) -> None:
        self.trajectory_manager_request.emit(
            TrajectoryManagerRequest(
                type=TrajectoryManagerRequestType.APPLY_ALIGNMENT,
                selection=self.request.trajectory_selection,
                alignment=selected_alignment,
            )
        )

    def trajectory_import_dialog(self, _: UIRequest) -> None:
        if selected_files := read_file_dialog(None, file_filter="Trajectory Files (*.traj);;All Files (*.*)"):
            self.file_request.emit(FileRequest(type=FileRequestType.READ_TRAJ, file_list=selected_files))
        else:
            return

    def trajectory_export_dialog(self, _: UIRequest) -> None:
        if selected_file := save_file_dialog(None, file_filter="Trajectory File (*.traj)"):
            self.file_request.emit(
                FileRequest(
                    type=FileRequestType.WRITE_TRAJ,
                    file_list=[selected_file],
                    trajectory_selection=self.request.trajectory_selection,
                )
            )
        else:
            return

    def result_export_dialog(self, _: UIRequest) -> None:
        if selected_file := save_file_dialog(None, file_filter="Result File (*.result)"):
            self.file_request.emit(
                FileRequest(
                    type=FileRequestType.WRITE_RES,
                    file_list=[selected_file],
                    result_selection=self.request.result_selection,
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
        epsg, ok = QtWidgets.QInputDialog.getInt(None, "Please enter an EPSG code", "EPSG:")

        if not ok or epsg is None:
            return

        self.trajectory_manager_request.emit(
            TrajectoryManagerRequest(
                type=TrajectoryManagerRequestType.CHANGE_ESPG,
                selection=request.trajectory_selection,
                target_epsg=epsg,
            )
        )

    def grid_input(self, request: UIRequest) -> None:
        grid, ok = QtWidgets.QInputDialog.getDouble(
            None,
            "Please enter a grid size in seconds",
            "Grid size [s]:",
            min=0.0001,
            value=0.01,
            decimals=4,
        )

        if not ok or grid is None:
            return

        self.trajectory_manager_request.emit(
            TrajectoryManagerRequest(
                type=TrajectoryManagerRequestType.INTERPOLATE_GRID,
                selection=request.trajectory_selection,
                grid=grid,
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
        alignment_window.show()
