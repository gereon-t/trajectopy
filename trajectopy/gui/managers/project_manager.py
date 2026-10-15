import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal, Slot

from trajectopy.gui.managers.requests import (
    ProjectRequest,
    ProjectRequestType,
    ResultModelRequest,
    ResultModelRequestType,
    TrajectoryModelRequest,
    TrajectoryModelRequestType,
    UIRequest,
    generic_request_handler,
)
from trajectopy.gui.models.selection import ResultSelection, TrajectorySelection
from trajectopy.gui.project import Project, load_project, load_session_folder, save_project
from trajectopy.gui.worker import BackgroundWorker

logger = logging.getLogger(__name__)


class ProjectManager(QObject):
    """
    Manager for handling project requests.

    Possible requests:
    - New project
    - Open project
    - Save project
    - Import session folder (saved by earlier versions)

    Requests run on the worker, one after another with the file and trajectory operations. Opening replaces the
    contents of the tables only once the project has been read completely, so a damaged file changes nothing.
    """

    trajectory_model_request = Signal(TrajectoryModelRequest)
    result_model_request = Signal(ResultModelRequest)
    ui_request = Signal(UIRequest)
    plot_settings_loaded = Signal(object)  # PlotSettings
    # emitted after the tables have been filled; the path is empty for a new project and an imported session folder
    project_loaded = Signal(str)
    project_saved = Signal(str)
    save_failed = Signal()
    operation_started = Signal()
    operation_finished = Signal()

    def __init__(self, worker: BackgroundWorker | None = None, parent=None) -> None:
        super().__init__(parent)
        self._worker = worker or BackgroundWorker(inline=True)
        self.REQUEST_MAPPING: dict[ProjectRequestType, Callable[[ProjectRequest], None]] = {
            ProjectRequestType.NEW: self.new_project,
            ProjectRequestType.OPEN: self.open_project,
            ProjectRequestType.SAVE: self.save_project,
            ProjectRequestType.IMPORT_SESSION_FOLDER: self.import_session_folder,
        }

    @Slot(ProjectRequest)
    def handle_request(self, request: ProjectRequest) -> None:
        """Schedules a project request on the worker."""
        self.operation_started.emit()
        self._worker.submit(self._process_request, request)

    def _process_request(self, request: ProjectRequest) -> None:
        try:
            generic_request_handler(self, request)
        finally:
            self.operation_finished.emit()

    def new_project(self, _: ProjectRequest) -> None:
        self._show(Project(), path="")
        logger.info("Started a new project.")

    def open_project(self, request: ProjectRequest) -> None:
        self._show(load_project(request.file_path), path=request.file_path)

    def import_session_folder(self, request: ProjectRequest) -> None:
        self._show(load_session_folder(request.file_path), path="")
        logger.info("Imported session folder %s", request.file_path)

    def save_project(self, request: ProjectRequest) -> None:
        try:
            save_project(request.project, request.file_path)
        except Exception:
            self.save_failed.emit()
            raise  # reported to the user by the request handler
        self.project_saved.emit(request.file_path)

    def _show(self, project: Project, path: str) -> None:
        """Replaces the contents of the tables and the plot settings by those of ``project``."""
        self.trajectory_model_request.emit(TrajectoryModelRequest(type=TrajectoryModelRequestType.RESET))
        self.result_model_request.emit(ResultModelRequest(type=ResultModelRequestType.RESET))
        if project.trajectories:
            self.trajectory_model_request.emit(
                TrajectoryModelRequest(
                    type=TrajectoryModelRequestType.ADD, selection=TrajectorySelection(entries=project.trajectories)
                )
            )
        if project.results:
            self.result_model_request.emit(
                ResultModelRequest(type=ResultModelRequestType.ADD, selection=ResultSelection(entries=project.results))
            )
        self.plot_settings_loaded.emit(project.plot_settings)
        self.project_loaded.emit(path)
