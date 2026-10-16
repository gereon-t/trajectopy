import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from trajectopy.gui.managers.requests import (
    FileRequest,
    FileRequestType,
    ResultModelRequest,
    ResultModelRequestType,
    TrajectoryModelRequest,
    TrajectoryModelRequestType,
    UIRequest,
    generic_request_handler,
)
from trajectopy.gui.models.entries import ResultEntry, TrajectoryEntry
from trajectopy.gui.models.selection import ResultSelection, TrajectorySelection
from trajectopy.gui.worker import BackgroundWorker
from trajectopy.io.rosbag import trajectories_from_rosbag

logger = logging.getLogger(__name__)


class FileManager(QObject):
    """
    Manager for handling file requests.

    Possible requests:
    - Read trajectory files
    - Write trajectory files
    - Read result files
    - Write result files
    - Write KML files (trajectory or ATE result)

    """

    ui_request = Signal(UIRequest)
    trajectory_model_request = Signal(TrajectoryModelRequest)
    result_model_request = Signal(ResultModelRequest)
    operation_started = Signal()
    operation_finished = Signal()

    def __init__(self, worker: BackgroundWorker | None = None, parent=None) -> None:
        super().__init__(parent)
        self._worker = worker or BackgroundWorker(inline=True)
        self.REQUEST_MAPPING: dict[FileRequestType, Callable[[FileRequest], None]] = {
            FileRequestType.READ_TRAJ: self.read_trajectory_files,
            FileRequestType.WRITE_TRAJ: self.write_trajectory,
            FileRequestType.READ_RES: self.read_result_files,
            FileRequestType.WRITE_RES: self.write_result,
            FileRequestType.WRITE_KML: self.write_kml,
        }

    @Slot(FileRequest)
    def handle_request(self, request: FileRequest) -> None:
        """Schedules a file request on the worker."""
        self.operation_started.emit()
        self._worker.submit(self._process_request, request)

    def _process_request(self, request: FileRequest) -> None:
        try:
            generic_request_handler(self, request)
        finally:
            self.operation_finished.emit()

    def read_trajectory_files(self, request: FileRequest) -> None:
        for file in request.file_list:
            if Path(file).suffix == ".bag":
                logger.info("Reading ROS bag file: %s", file)
                self.handle_ros_bag(file)
            else:
                self.handle_ascii(file)
                logger.info("Reading ASCII file: %s", file)

    def handle_ascii(self, file: str) -> None:
        """Reads an ASCII trajectory file and emits a request to add it to the model."""
        traj_entry = TrajectoryEntry.from_trajectory_file(file)

        self.trajectory_model_request.emit(
            TrajectoryModelRequest(
                type=TrajectoryModelRequestType.ADD,
                selection=TrajectorySelection(entries=[traj_entry]),
            )
        )

    def handle_ros_bag(self, file: str) -> None:
        """Reads a ROS bag file and emits a request to add it to the model."""
        trajectories = trajectories_from_rosbag(file)
        traj_entries = [TrajectoryEntry(full_filename=file, trajectory=traj) for traj in trajectories]
        self.trajectory_model_request.emit(
            TrajectoryModelRequest(
                type=TrajectoryModelRequestType.ADD,
                selection=TrajectorySelection(entries=traj_entries),
            )
        )

    def write_trajectory(self, request: FileRequest) -> None:
        trajectory_entry = request.trajectory_selection.entries[0]
        trajectory_entry.to_file(request.file_list[0])

    def read_result_files(self, request: FileRequest) -> None:
        for file in request.file_list:
            logger.info("Reading file: %s", file)

            result_entry = ResultEntry.from_file(filename=file)

            self.result_model_request.emit(
                ResultModelRequest(
                    type=ResultModelRequestType.ADD,
                    selection=ResultSelection(entries=[result_entry]),
                )
            )

    def write_result(self, request: FileRequest) -> None:
        result_entry = request.result_selection.entries[0]
        result_entry.to_file(request.file_list[0])

    def write_kml(self, request: FileRequest) -> None:
        """Google Earth export of the selected ATE result, or of the selected trajectory if no result is selected."""
        if request.result_selection.entries:
            request.result_selection.entries[0].deviations.to_kml(request.file_list[0], **request.kml_options)
        else:
            request.trajectory_selection.entries[0].trajectory.to_kml(request.file_list[0], **request.kml_options)
