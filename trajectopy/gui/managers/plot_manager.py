import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal, Slot

from trajectopy.gui.managers.requests import (
    PlotRequest,
    PlotRequestType,
    UIRequest,
    UIRequestType,
    generic_request_handler,
)
from trajectopy.gui.models.entries import (
    AbsoluteDeviationEntry,
    AlignmentEntry,
    RelativeDeviationEntry,
    ResultEntry,
)
from trajectopy.gui.utils import show_progress
from trajectopy.results.ate_result import ATEResult
from trajectopy.results.rpe_result import RPEResult
from trajectopy.visualization import (
    Figure,
    PlotWindow,
    alignment_figures,
    deviation_figures,
    trajectory_figures,
)

logger = logging.getLogger(__name__)


class PlotManager(QObject):
    """
    Manager for handling plot requests: builds the figures for the selection and opens a plot window.

    Possible requests:
    - Plot trajectories
    - Plot deviations (one or several ATE/RPE results)
    - Plot alignment
    """

    ui_request = Signal(UIRequest)
    operation_started = Signal()
    operation_finished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.REQUEST_MAPPING: dict[PlotRequestType, Callable[[PlotRequest], None]] = {
            PlotRequestType.TRAJECTORIES: self.plot_trajectories,
            PlotRequestType.DEVIATIONS: self.plot_deviations,
            PlotRequestType.ALIGNMENT: self.plot_alignment,
        }
        self._windows: list[PlotWindow] = []

    @show_progress
    @Slot(PlotRequest)
    def handle_request(self, request: PlotRequest) -> None:
        """Logic for handling a request."""
        generic_request_handler(self, request)

    def _open_window(self, figures: list[Figure], title: str, request: PlotRequest) -> None:
        if not figures:
            self.ui_request.emit(UIRequest(type=UIRequestType.MESSAGE, message="Nothing to plot."))
            return
        window = PlotWindow(
            figures,
            title=title,
            export_format=request.plot_settings.export_format,
            export_width=request.plot_settings.export_width,
            parent=self.parent(),  # type: ignore[arg-type]
        )
        window.destroyed.connect(lambda _=None, w=window: self._windows.remove(w) if w in self._windows else None)
        self._windows.append(window)
        window.show()

    def plot_trajectories(self, request: PlotRequest) -> None:
        trajectories = [entry.trajectory for entry in request.trajectory_selection.entries]
        title = trajectories[0].name if len(trajectories) == 1 else "Trajectories"
        self._open_window(trajectory_figures(trajectories, request.plot_settings), title, request)

    def plot_deviations(self, request: PlotRequest) -> None:
        ate_results = get_ate_results(request.result_selection.entries)
        rpe_results = get_rpe_results(request.result_selection.entries)
        results: list[ATEResult | RPEResult] = [*ate_results, *rpe_results]
        title = results[0].name if len(results) == 1 else "Trajectory Comparison"
        self._open_window(deviation_figures(ate_results, rpe_results, request.plot_settings), title, request)

    def plot_alignment(self, request: PlotRequest) -> None:
        if not isinstance((entry := request.result_selection.entries[0]), AlignmentEntry):
            raise TypeError("Entry must be of type AlignmentEntry!")
        self._open_window(alignment_figures(entry.estimated_parameters, request.plot_settings), entry.name, request)


def get_ate_results(entries: list[ResultEntry]) -> list[ATEResult]:
    """Return a list of ate results."""
    return [entry.deviations for entry in entries if isinstance(entry, AbsoluteDeviationEntry)]


def get_rpe_results(entries: list[ResultEntry]) -> list[RPEResult]:
    """Return a list of rpe results."""
    return [entry.deviations for entry in entries if isinstance(entry, RelativeDeviationEntry)]
