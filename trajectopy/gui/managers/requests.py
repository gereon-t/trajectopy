"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Protocol

from PyQt6.QtCore import pyqtSignal

from trajectopy.api.settings import MatchingMethod, ReportSettings
from trajectopy.gui.models.entries import AlignmentEntry
from trajectopy.gui.models.selection import ResultSelection, TrajectorySelection

logger = logging.getLogger("root")


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
    ADAPT_FIRST_POSITION = auto()
    ADAPT_FIRST_ORIENTATION = auto()
    APPLY_ALIGNMENT = auto()
    APPROXIMATE = auto()
    SORT = auto()
    CHANGE_ESPG = auto()
    EPSG_TO_REF = auto()
    INTERSECT = auto()
    INTERPOLATE = auto()
    INTERPOLATE_GRID = auto()
    MERGE = auto()
    ADAPT_SAMPLING = auto()
    MATCH_TIMESTAMPS = auto()
    ADAPT_ORIENTATIONS = auto()
    ROUGH_TIMESTAMPS_MATCHING = auto()
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
    RES_PROPERTIES = auto()
    TRAJ_SETTINGS = auto()
    SETTINGS = auto()
    EPSG_SELECTION = auto()
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


class ReportSettingsRequestType(Enum):
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
    file_list: List[str] = field(default_factory=list)
    trajectory_selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    result_selection: ResultSelection = field(default_factory=ResultSelection)
    id_list: List[str] = field(default_factory=list)


@dataclass
class PlotRequest(DeepCopyRequest):
    type: PlotRequestType
    report_settings: ReportSettings = field(default_factory=ReportSettings)
    result_selection: ResultSelection = field(default_factory=ResultSelection)
    trajectory_selection: TrajectorySelection = field(default_factory=TrajectorySelection)
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
    index_list: List[str] = field(default_factory=list)
    file_path: str = ""


@dataclass
class SessionManagerRequest:
    type: SessionManagerRequestType
    file_path: str = ""


@dataclass
class ReportSettingsRequest:
    type: ReportSettingsRequestType
    file_path: str = ""


@dataclass
class TrajectoryModelRequest:
    type: TrajectoryModelRequestType
    selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    index_list: List[str] = field(default_factory=list)
    file_path: str = ""


@dataclass
class TrajectoryManagerRequest(DeepCopyRequest):
    type: TrajectoryManagerRequestType
    selection: TrajectorySelection = field(default_factory=TrajectorySelection)
    target_epsg: int = 0
    grid: float = 0.0
    alignment: AlignmentEntry = field(default_factory=AlignmentEntry)
    matching_method: MatchingMethod = MatchingMethod.NEAREST_TEMPORAL


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
