import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol

from PySide6.QtCore import SignalInstance

from trajectopy.core.settings import PlotSettings
from trajectopy.gui.models.entries import AlignmentEntry
from trajectopy.gui.models.selection import ResultSelection, TrajectorySelection
from trajectopy.results.alignment_result import AlignmentResult

logger = logging.getLogger(__name__)


class FileRequestType(Enum):
    READ_TRAJ = auto()
    WRITE_TRAJ = auto()
    READ_RES = auto()
    WRITE_RES = auto()
    WRITE_LIST = auto()
    READ_TRAJ_ORDER = auto()
    READ_RES_ORDER = auto()


class PlotRequestType(Enum):
    DEVIATIONS = auto()
    TRAJECTORIES = auto()
    ALIGNMENT = auto()


class PropertyModelRequestType(Enum):
    EXPORT = auto()


class ResultModelRequestType(Enum):
    ADD = auto()
    UPDATE = auto()
    RENAME = auto()
    REMOVE = auto()
    PASS_ALIGNMENTS_TO_UI = auto()
    RESET = auto()
    EXPORT_ALL = auto()
    EDIT_ALIGNMENT = auto()
    COPY = auto()
    SORT = auto()


class SessionManagerRequestType(Enum):
    IMPORT = auto()
    EXPORT = auto()
    NEW = auto()


class TrajectoryManagerRequestType(Enum):
    SWITCH_SORTING = auto()
    ALIGN = auto()
    ADAPT_FIRST_POSE = auto()
    APPLY_ALIGNMENT = auto()
    APPROXIMATE = auto()
    AVERAGE = auto()
    SORT = auto()
    DIVIDE_INTO_LAPS = auto()
    EDIT_EPSG = auto()
    CHANGE_ESPG = auto()
    REARANGE_DOF = auto()
    EPSG_TO_REF = auto()
    MERGE = auto()
    MATCH = auto()
    ATE_WITH_ALIGNMENT = auto()
    ATE_NO_ALIGNMENT = auto()
    RPE = auto()
    INTERPOLATE_TO_GRID = auto()


class TrajectoryModelRequestType(Enum):
    ADD = auto()
    UPDATE = auto()
    RENAME = auto()
    SET_REFERENCE = auto()
    UNSET_REFERENCE = auto()
    COPY = auto()
    REMOVE = auto()
    REMOVE_RELATED = auto()
    RESET = auto()
    EXPORT_ALL = auto()
    SORT = auto()


class UIRequestType(Enum):
    DOF_ORGANIZER = auto()
    TRAJ_SETTINGS = auto()
    EPSG_EDIT = auto()
    EPSG_TRANSFORMATION = auto()
    GRID_SELECTION = auto()
    ALIGNMENT_SELECTION = auto()
    EXPORT_TRAJ = auto()
    EXPORT_KML = auto()
    IMPORT_TRAJ = auto()
    EXPORT_RES = auto()
    IMPORT_RES = auto()
    MESSAGE = auto()
    CONFIRM_RESET = auto()
    EXPORT_SESSION = auto()
    IMPORT_SESSION = auto()
    EDIT_ALIGNMENT = auto()
    MANUAL_ALIGNMENT = auto()
    PLAYBACK = auto()
    RENAME_TRAJ = auto()
    RENAME_RES = auto()


class PlotSettingsRequestType(Enum):
    IMPORT = auto()
    EXPORT = auto()
    RESET = auto()


@dataclass
class DeepCopyRequest:
    """Base class for requests that are processed on the background worker.

    The payload is deep-copied when the request is created on the UI thread, so the worker never shares
    mutable entries (and their trajectories) with the models displayed by the UI.
    """

    def __post_init__(self):
        self.__dict__ = {k: copy.deepcopy(v) for k, v in self.__dict__.items()}


@dataclass
class FileRequest(DeepCopyRequest):
    type: FileRequestType
    file_list: list[str] = field(default_factory=list)
    trajectory_selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    result_selection: ResultSelection = field(default_factory=ResultSelection)
    id_list: list[str] = field(default_factory=list)


@dataclass
class PlotRequest(DeepCopyRequest):
    type: PlotRequestType
    result_selection: ResultSelection = field(default_factory=ResultSelection)
    trajectory_selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    plot_settings: PlotSettings = field(default_factory=PlotSettings)


@dataclass
class PropertyModelRequest:
    type: PropertyModelRequestType
    file_path: str = ""


@dataclass
class ResultModelRequest:
    type: ResultModelRequestType
    selection: ResultSelection = field(default_factory=ResultSelection)
    trajectory_selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    alignment: AlignmentEntry = field(default_factory=AlignmentEntry)
    index_list: list[str] = field(default_factory=list)
    file_path: str = ""
    name: str = ""  # new name (RENAME)


@dataclass
class SessionManagerRequest:
    type: SessionManagerRequestType
    file_path: str = ""


@dataclass
class PlotSettingsRequest:
    type: PlotSettingsRequestType
    file_path: str = ""


@dataclass
class TrajectoryModelRequest:
    type: TrajectoryModelRequestType
    selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    index_list: list[str] = field(default_factory=list)
    file_path: str = ""
    name: str = ""  # new name (RENAME)


@dataclass
class TrajectoryManagerRequest(DeepCopyRequest):
    type: TrajectoryManagerRequestType
    selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    target_epsg: int = 0
    grid: float = 0.0
    alignment: AlignmentEntry = field(default_factory=AlignmentEntry)
    dof_mapping: dict[str, dict] = field(default_factory=dict)
    # alignment chosen in the manual alignment window; applied as is instead of being estimated
    manual_alignment: AlignmentResult | None = None


@dataclass
class UIRequest:
    type: UIRequestType
    trajectory_selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    result_selection: ResultSelection = field(default_factory=ResultSelection)
    alignment_entry: AlignmentEntry = field(default_factory=AlignmentEntry)
    message: str = ""
    error: bool = False  # the message reports a failure
    epsg: int = 0
    grid: int = 0
    file_path: str = ""


class Request(Protocol):
    """Interface for requests."""

    @property
    def type(self) -> Enum: ...


class RequestHandler(Protocol):
    """Interface for request handlers: a mapping from request type to a handler receiving the request."""

    @property
    def REQUEST_MAPPING(self) -> Mapping[Any, Callable[[Any], None]]: ...

    @property
    def ui_request(self) -> SignalInstance: ...


def generic_request_handler(manager: RequestHandler, request: Request) -> None:
    """Dispatches ``request`` to the handler registered for its type and reports failures to the user."""
    func = manager.REQUEST_MAPPING.get(request.type)
    if func is None:
        logger.error(
            "%s: Unable to handle request of type %s",
            manager.__class__.__name__,
            request.type,
        )
        return
    try:
        func(request)
    except Exception as e:
        logger.exception(
            "%s: Error processing request of type %s: %s",
            manager.__class__.__name__,
            request.type,
            str(e),
        )
        manager.ui_request.emit(
            UIRequest(
                type=UIRequestType.MESSAGE,
                message=f"{request.type.name.replace('_', ' ').capitalize()} failed:\n\n{e}",
                error=True,
            )
        )
    finally:
        logger.debug("%s: Handled request of type %s", manager.__class__.__name__, request.type)
