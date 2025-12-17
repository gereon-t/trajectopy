import logging
import os
from collections.abc import Callable

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from trajectopy.core.settings import PlotBackend
from trajectopy.gui.managers.requests import (
    PlotRequest,
    PlotRequestType,
    UIRequest,
    generic_request_handler,
)
from trajectopy.gui.models.entries import (
    AbsoluteDeviationEntry,
    AlignmentEntry,
    RelativeDeviationEntry,
    ResultEntry,
)
from trajectopy.gui.utils import show_progress
from trajectopy.gui.views.mpl_plot_tabs import PlotTabs
from trajectopy.results.ate_result import ATEResult
from trajectopy.results.rpe_result import RPEResult
from trajectopy.visualization.plotly_reports import (
    create_alignment_report,
    create_deviation_report,
    create_trajectory_report,
    show_report,
)

logger = logging.getLogger(__name__)


class PlotManager(QObject):
    """
    Manager for handling plot requests.

    Possible requests:
    - Plot trajectories
    - Plot single deviations
    - Plot multi deviations
    - Plot alignment

    """

    ui_request = pyqtSignal(UIRequest)
    operation_started = pyqtSignal()
    operation_finished = pyqtSignal()

    def __init__(
        self,
        report_dir: str,
        plot_backend: PlotBackend = PlotBackend.PLOTLY,
        parent=None,
    ) -> None:
        self.cnt = 0
        self.plot_backend = plot_backend
        super().__init__(parent)
        self.REQUEST_MAPPING: dict[PlotRequestType, Callable[[PlotRequest], None]] = {
            PlotRequestType.TRAJECTORIES: self.plot_selected_trajectories,
            PlotRequestType.SINGLE_DEVIATIONS: self.plot_single_deviations,
            PlotRequestType.MULTI_DEVIATIONS: self.plot_multi_deviations,
            PlotRequestType.ALIGNMENT: self.plot_alignment,
        }
        self.report_dir = os.path.abspath(report_dir)

    def set_plot_backend(self, plot_backend: PlotBackend) -> None:
        """Set the plot backend."""
        logger.info(f"Setting plot backend to {plot_backend}")
        self.plot_backend = plot_backend

    @show_progress
    @pyqtSlot(PlotRequest)
    def handle_request(self, request: PlotRequest) -> None:
        """Logic for handling a request."""
        generic_request_handler(self, request, True)

    def report_path(self, prefix: str = "report") -> str:
        """Return the path to the report file."""
        prefix = prefix.replace(" ", "_").replace(".", "")
        path = os.path.join(self.report_dir, f"{prefix}_{self.cnt:03}.html")
        self.cnt += 1
        return path

    def plot_selected_trajectories(self, request: PlotRequest) -> None:
        trajectory_list = [entry.trajectory for entry in request.trajectory_selection.entries]

        if self.plot_backend == PlotBackend.PLOTLY:
            traj_report = create_trajectory_report(
                trajectories=trajectory_list, report_settings=request.report_settings
            )
            show_report(traj_report, filepath=self.report_path(prefix="trajectories"))
        elif self.plot_backend == PlotBackend.MPL:
            name = "Trajectopy Viewer" if len(trajectory_list) > 1 else trajectory_list[0].name
            plot_tabs = PlotTabs(parent=self.parent(), window_title=name)
            plot_tabs.show_trajectories(trajectory_list, mpl_plot_settings=request.mpl_plot_settings)

    def plot_single_deviations(self, request: PlotRequest) -> None:
        ate_results = get_ate_results(request.result_selection.entries)
        rpe_results = get_rpe_results(request.result_selection.entries)

        if not ate_results and not rpe_results:
            logger.error("No deviations selected!")
            return

        if len(ate_results) > 1 or len(rpe_results) > 1:
            logger.error("Too many deviations selected for single report!")
            return

        ate_result = ate_results[0] if ate_results else None
        rpe_result = rpe_results[0] if rpe_results else None

        name = ate_result.name if ate_result else rpe_result.name

        if ate_result:
            ate_result.remove_ate_above(
                request.mpl_plot_settings.ate_remove_above
                if self.plot_backend == PlotBackend.MPL
                else request.report_settings.ate_remove_above
            )

        if self.plot_backend == PlotBackend.PLOTLY:
            report = create_deviation_report(
                ate_result=ate_result,
                rpe_result=rpe_result,
                report_settings=request.report_settings,
            )
            show_report(
                report_text=report,
                filepath=self.report_path(prefix=ate_result.name if ate_result else rpe_result.name),
            )
        elif self.plot_backend == PlotBackend.MPL:
            plot_tabs = PlotTabs(parent=self.parent(), window_title=name)
            plot_tabs.show_single_deviations(
                ate_result=ate_result, rpe_result=rpe_result, mpl_plot_settings=request.mpl_plot_settings
            )

    def plot_multi_deviations(self, request: PlotRequest) -> None:
        """Plot multiple absolute deviations."""
        ate_results = get_ate_results(request.result_selection.entries)
        rpe_results = get_rpe_results(request.result_selection.entries)

        if not ate_results and not rpe_results:
            logger.error("No deviations selected!")
            return

        for ate in ate_results:
            ate.remove_ate_above(
                request.mpl_plot_settings.ate_remove_above
                if self.plot_backend == PlotBackend.MPL
                else request.report_settings.ate_remove_above
            )

        if self.plot_backend == PlotBackend.PLOTLY:
            multi_report = create_deviation_report(
                ate_result=ate_results or None,
                rpe_result=rpe_results or None,
                report_settings=request.report_settings,
            )
            show_report(
                report_text=multi_report,
                filepath=self.report_path(prefix="multi_deviations"),
            )
        elif self.plot_backend == PlotBackend.MPL:
            plot_tabs = PlotTabs(parent=self.parent())
            plot_tabs.show_multiple_deviations(
                ate_results=ate_results, rpe_results=rpe_results, mpl_plot_settings=request.mpl_plot_settings
            )

    def plot_alignment(self, request: PlotRequest) -> None:
        if not isinstance((entry := request.result_selection.entries[0]), AlignmentEntry):
            raise TypeError("Entry must be of type AlignmentEntry!")

        if self.plot_backend == PlotBackend.PLOTLY:
            report = create_alignment_report(
                alignment_parameters=entry.estimated_parameters,
                name=entry.name,
                report_settings=request.report_settings,
            )

            show_report(report_text=report, filepath=self.report_path(prefix=entry.name))
        elif self.plot_backend == PlotBackend.MPL:
            plot_tabs = PlotTabs(parent=self.parent())
            plot_tabs.show_alignment_parameters(entry.estimated_parameters)


def get_ate_results(entries: list[ResultEntry]) -> list[ATEResult]:
    """Return a list of ate results."""
    return [entry.deviations for entry in entries if isinstance(entry, AbsoluteDeviationEntry)]


def get_rpe_results(entries: list[ResultEntry]) -> list[RPEResult]:
    """Return a list of rpe results."""
    return [entry.deviations for entry in entries if isinstance(entry, RelativeDeviationEntry)]
