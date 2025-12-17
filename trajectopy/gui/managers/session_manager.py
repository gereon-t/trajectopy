import glob
import logging
import os
import threading
from collections.abc import Callable

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

logger = logging.getLogger(__name__)


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
    ui_request = pyqtSignal(UIRequest)
    file_request = pyqtSignal(FileRequest)
    operation_finished = pyqtSignal()
    report_settings_request = pyqtSignal(PlotSettingsRequest)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.REQUEST_MAPPING: dict[SessionManagerRequestType, Callable[[SessionManagerRequest], None]] = {
            SessionManagerRequestType.NEW: self.new_session,
            SessionManagerRequestType.EXPORT: self.export_session,
            SessionManagerRequestType.IMPORT: self.import_session,
        }

    @pyqtSlot(SessionManagerRequest)
    def handle_request(self, request: SessionManagerRequest) -> None:
        request_thread = threading.Thread(target=generic_request_handler, args=(self, request, True))
        request_thread.start()
        request_thread.join()

    def new_session(self, _: SessionManagerRequest) -> None:
        self.trajectory_model_request.emit(TrajectoryModelRequest(type=TrajectoryModelRequestType.RESET))
        self.result_model_request.emit(ResultModelRequest(type=ResultModelRequestType.RESET))
        self.report_settings_request.emit(PlotSettingsRequest(type=PlotSettingsRequestType.RESET))
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
        self.report_settings_request.emit(
            PlotSettingsRequest(type=PlotSettingsRequestType.IMPORT, file_path=request.file_path)
        )

    def export_session(self, request: SessionManagerRequest) -> None:
        os.makedirs(request.file_path, exist_ok=True)
        self.trajectory_model_request.emit(
            TrajectoryModelRequest(type=TrajectoryModelRequestType.EXPORT_ALL, file_path=request.file_path)
        )
        self.result_model_request.emit(
            ResultModelRequest(type=ResultModelRequestType.EXPORT_ALL, file_path=request.file_path)
        )
        self.report_settings_request.emit(
            PlotSettingsRequest(type=PlotSettingsRequestType.EXPORT, file_path=request.file_path)
        )
