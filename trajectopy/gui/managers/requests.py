import copy
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol

from PyQt6.QtCore import pyqtSignal

from trajectopy.core.settings import MatchingMethod, MPLPlotSettings, ReportSettings
from trajectopy.gui.models.entries import AlignmentEntry
from trajectopy.gui.models.selection import ResultSelection, TrajectorySelection

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
    SINGLE_DEVIATIONS = auto()
    MULTI_DEVIATIONS = auto()
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
    COMPARE_ABS = auto()
    COMPARE_REL = auto()
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
    ATE = auto()
    RPE = auto()


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
    TRAJ_PROPERTIES = auto()
    DOF_ORGANIZER = auto()
    RES_PROPERTIES = auto()
    TRAJ_SETTINGS = auto()
    SETTINGS = auto()
    EPSG_EDIT = auto()
    EPSG_TRANSFORMATION = auto()
    GRID_SELECTION = auto()
    ALIGNMENT_SELECTION = auto()
    EXPORT_TRAJ = auto()
    IMPORT_TRAJ = auto()
    EXPORT_RES = auto()
    IMPORT_RES = auto()
    MESSAGE = auto()
    CONFIRM_RESET = auto()
    EXPORT_SESSION = auto()
    IMPORT_SESSION = auto()
    EDIT_ALIGNMENT = auto()
    EXPORT_DEV_SUMMARY = auto()
    EXPORT_REPORT = auto()


class PlotSettingsRequestType(Enum):
    SHOW = auto()
    IMPORT = auto()
    EXPORT = auto()
    RESET = auto()


@dataclass
class DeepCopyRequest:
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
    report_settings: ReportSettings = field(default_factory=ReportSettings)
    result_selection: ResultSelection = field(default_factory=ResultSelection)
    trajectory_selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    mpl_plot_settings: MPLPlotSettings = field(default_factory=MPLPlotSettings)
    dimension: int = 2


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


@dataclass
class TrajectoryManagerRequest(DeepCopyRequest):
    type: TrajectoryManagerRequestType
    selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    target_epsg: int = 0
    grid: float = 0.0
    alignment: AlignmentEntry = field(default_factory=AlignmentEntry)
    matching_method: MatchingMethod = MatchingMethod.NEAREST_TEMPORAL
    dof_mapping: dict[str, dict] = field(default_factory=dict)


@dataclass
class UIRequest:
    type: UIRequestType
    trajectory_selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    result_selection: ResultSelection = field(default_factory=ResultSelection)
    alignment_entry: AlignmentEntry = field(default_factory=AlignmentEntry)
    report_settings: ReportSettings = field(default_factory=ReportSettings)
    message: str = ""
    confirm_reset: bool = False
    epsg: int = 0
    grid: int = 0
    file_path: str = ""


class Request(Protocol):
    """Interface for requests."""

    type: Any


class RequestHandler(Protocol):
    """Interface for request handlers."""

    operation_finished: pyqtSignal
    ui_request: pyqtSignal
    REQUEST_MAPPING: dict


def generic_request_handler(manager: RequestHandler, request: Request, passthrough_request: bool = True) -> None:
    """Function to handle requests."""
    func = manager.REQUEST_MAPPING.get(request.type)
    if func is None:
        logger.error(
            "%s: Unable to handle request of type %s",
            manager.__class__.__name__,
            request.type,
        )
        return
    try:
        func(request) if passthrough_request else func()
    except Exception as e:
        manager.ui_request.emit(
            UIRequest(
                type=UIRequestType.MESSAGE,
                message=f"{manager.__class__.__name__}: Error processing request: {request.type.name} ({e})",
            )
        )
    finally:
        manager.operation_finished.emit()
        logger.debug("%s: Handled request of type %s", manager.__class__.__name__, request.type)
