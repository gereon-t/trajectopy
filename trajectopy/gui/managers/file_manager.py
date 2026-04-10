import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
from trajectopy.gui.utils import show_progress
from trajectopy.readers.rosbag import trajectories_from_rosbag

logger = logging.getLogger(__name__)


class FileManager(QObject):
    """
    Manager for handling file requests.

    Possible requests:
    - Read trajectory files
    - Write trajectory files
    - Read result files
    - Write result files
    - Read result order
    - Read trajectory order
    - Write list

    """

    ui_request = Signal(UIRequest)
    trajectory_model_request = Signal(TrajectoryModelRequest)
    result_model_request = Signal(ResultModelRequest)
    operation_started = Signal()
    operation_finished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Python 3.14 workaround: execute request handlers in a real Python thread.
        # Keep a persistent single worker to avoid per-request thread creation overhead.
        self._request_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TrajectopyFileMgr")
        self.REQUEST_MAPPING: dict[FileRequestType, Callable[[FileRequest], None]] = {
            FileRequestType.READ_TRAJ: self.read_trajectory_files,
            FileRequestType.WRITE_TRAJ: self.write_trajectory,
            FileRequestType.READ_RES: self.read_result_files,
            FileRequestType.WRITE_RES: self.write_result,
            FileRequestType.READ_RES_ORDER: self.read_res_order,
            FileRequestType.READ_TRAJ_ORDER: self.read_traj_order,
            FileRequestType.WRITE_LIST: self.write_list,
        }

    @show_progress
    @Slot(FileRequest)
    def handle_request(self, request: FileRequest) -> None:
        """Logic for handling a request."""
        self._request_executor.submit(generic_request_handler, self, request, True).result()

    def shutdown_executor(self) -> None:
        """Best-effort cleanup for the persistent Python worker thread."""
        if hasattr(self, "_request_executor") and self._request_executor is not None:
            self._request_executor.shutdown(wait=False, cancel_futures=True)
            self._request_executor = None

    def __del__(self) -> None:
        self.shutdown_executor()

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
        trajectory_file, settings_file = self._get_traj_filenames(file)

        traj_entry = TrajectoryEntry.from_file(trajectory_filename=trajectory_file, settings_filename=settings_file)

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

    @staticmethod
    def _get_traj_filenames(file: str) -> tuple[Path, Path]:
        file_path = Path(file)
        file_name = file_path.stem
        file_directory = file_path.parent
        settings_file = file_directory / f"{file_name}.json"
        return Path(file), Path(settings_file)

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

    def write_list(self, request: FileRequest) -> None:
        if not request.id_list:
            return

        with open(request.file_list[0], "w", encoding="utf-8") as f:
            f.write("\n".join(request.id_list))
            f.write("\n")

    def read_list(self, request: FileRequest) -> None | list[str]:
        if not Path(request.file_list[0]).is_file():
            logger.warning("No order file found.")
            return None

        with open(request.file_list[0], encoding="utf-8") as f:
            file_list = [line.strip() for line in f.readlines()]
        return file_list

    def read_res_order(self, request: FileRequest) -> None:
        if id_list := self.read_list(request=request):
            self.result_model_request.emit(ResultModelRequest(type=ResultModelRequestType.SORT, index_list=id_list))

    def read_traj_order(self, request: FileRequest) -> None:
        if id_list := self.read_list(request=request):
            self.trajectory_model_request.emit(
                TrajectoryModelRequest(type=TrajectoryModelRequestType.SORT, index_list=id_list)
            )
