"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import glob
import logging
import os
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from trajectopy.gui.managers.requests import (
    FileRequest,
    FileRequestType,
    PlotSettingsRequest,
    PlotSettingsRequestType,
    ResultModelRequest,
    ResultModelRequestType,
    SessionManagerRequest,
    SessionManagerRequestType,
    TrajectoryModelRequest,
    TrajectoryModelRequestType,
    UIRequest,
    generic_request_handler,
)

logger = logging.getLogger("root")


class SessionManager(QObject):
    """
    Manager for handling session requests.

    Possible requests:
    - New session
    - Import session
    - Export session

    """

    trajectory_model_request = pyqtSignal(TrajectoryModelRequest)
    result_model_request = pyqtSignal(ResultModelRequest)
    plot_settings_request = pyqtSignal(PlotSettingsRequest)
    ui_request = pyqtSignal(UIRequest)
    file_request = pyqtSignal(FileRequest)
    operation_finished = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.REQUEST_MAPPING: dict[SessionManagerRequestType, Callable[[SessionManagerRequest], None]] = {
            SessionManagerRequestType.NEW: self.new_session,
            SessionManagerRequestType.EXPORT: self.export_session,
            SessionManagerRequestType.IMPORT: self.import_session,
        }

    @pyqtSlot(SessionManagerRequest)
    def handle_request(self, request: SessionManagerRequest) -> None:
        generic_request_handler(self, request, passthrough_request=True)

    def new_session(self, _: SessionManagerRequest) -> None:
        self.trajectory_model_request.emit(TrajectoryModelRequest(type=TrajectoryModelRequestType.RESET))
        self.result_model_request.emit(ResultModelRequest(type=ResultModelRequestType.RESET))
        self.plot_settings_request.emit(PlotSettingsRequest(type=PlotSettingsRequestType.RESET))
        logger.info("Cleared application and started a new session.")

    def import_session(self, request: SessionManagerRequest) -> None:
        traj_file_list = glob.glob(os.path.join(request.file_path, "*.traj"))
        result_file_list = glob.glob(os.path.join(request.file_path, "*.result"))

        self.file_request.emit(FileRequest(type=FileRequestType.READ_TRAJ, file_list=traj_file_list))
        self.file_request.emit(FileRequest(type=FileRequestType.READ_RES, file_list=result_file_list))

        self.file_request.emit(
            FileRequest(
                type=FileRequestType.READ_TRAJ_ORDER,
                file_list=[os.path.join(request.file_path, "trajectory_order.txt")],
            )
        )

        self.file_request.emit(
            FileRequest(
                type=FileRequestType.READ_RES_ORDER,
                file_list=[os.path.join(request.file_path, "result_order.txt")],
            )
        )

        self.plot_settings_request.emit(
            PlotSettingsRequest(
                type=PlotSettingsRequestType.IMPORT_FROM_SESSION,
                file_path=request.file_path,
            )
        )

    def export_session(self, request: SessionManagerRequest) -> None:
        os.makedirs(request.file_path, exist_ok=True)
        self.trajectory_model_request.emit(
            TrajectoryModelRequest(type=TrajectoryModelRequestType.EXPORT_ALL, file_path=request.file_path)
        )
        self.result_model_request.emit(
            ResultModelRequest(type=ResultModelRequestType.EXPORT_ALL, file_path=request.file_path)
        )
        self.plot_settings_request.emit(
            PlotSettingsRequest(
                type=PlotSettingsRequestType.EXPORT_TO_SESSION,
                file_path=request.file_path,
            )
        )
