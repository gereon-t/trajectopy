"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging
import os
from typing import Callable, Dict

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from trajectopy_core.evaluation.ate_result import ATEResult
from trajectopy_core.evaluation.rpe_result import RPEResult
from trajectopy_core.report.alignment import render_heatmaps
from trajectopy_core.report.multi import render_multi_report
from trajectopy_core.report.single import render_single_report
from trajectopy_core.report.trajectory import render_trajectories
from trajectopy_core.report.utils import show_report

from trajectopy.managers.requests import (
    PlotRequest,
    PlotRequestType,
    UIRequest,
    generic_request_handler,
)
from trajectopy.models.entries import AbsoluteDeviationEntry, AlignmentEntry, RelativeDeviationEntry, ResultEntry
from trajectopy.path import REPORT_PATH

logger = logging.getLogger("root")


class PlotManager(QObject):
    """
    Manager for handling plot requests.

    Possible requests:
    - Plot selected trajectories
    - Plot selected deviations
    - Plot trajectory laps
    - Plot multi deviations
    - Plot deviation laps
    - Update settings
    - Plot correlation

    """

    ui_request = pyqtSignal(UIRequest)
    operation_started = pyqtSignal()
    operation_finished = pyqtSignal()

    def __init__(self, parent) -> None:
        self.cnt = 0
        super().__init__(parent=parent)
        self.REQUEST_MAPPING: Dict[PlotRequestType, Callable[[PlotRequest], None]] = {
            PlotRequestType.TRAJECTORIES: self.plot_selected_trajectories,
            PlotRequestType.SINGLE_DEVIATIONS: self.plot_single_deviations,
            PlotRequestType.MULTI_DEVIATIONS: self.plot_multi_deviations,
            PlotRequestType.ALIGNMENT: self.plot_correlation,
        }

    @pyqtSlot(PlotRequest)
    def handle_request(self, request: PlotRequest) -> None:
        """Logic for handling a request."""
        generic_request_handler(self, request, passthrough_request=True)

    @property
    def report_path(self) -> str:
        """Return the path to the report file."""
        path = os.path.join(REPORT_PATH, f"report_{self.cnt:03}.html")
        self.cnt += 1
        return path

    def plot_selected_trajectories(self, request: PlotRequest) -> None:
        trajectory_list = [entry.trajectory for entry in request.trajectory_selection.entries]
        traj_report = render_trajectories(trajectories=trajectory_list, report_settings=request.report_settings)
        show_report(traj_report, filepath=self.report_path)

    def plot_single_deviations(self, request: PlotRequest) -> None:
        ate_results = get_ate_results(request.result_selection.entries)
        rpe_results = get_rpe_results(request.result_selection.entries)

        if not ate_results and not rpe_results:
            logger.error("No deviations selected!")
            return

        if len(ate_results) > 1 or len(rpe_results) > 1:
            logger.error("Too many deviations selected for single report!")
            return

        report = render_single_report(
            ate_result=ate_results[0], rpe_result=rpe_results[0], report_settings=request.report_settings
        )
        show_report(report_text=report, filepath=self.report_path)

    def plot_multi_deviations(self, request: PlotRequest) -> None:
        """Plot multiple absolute deviations."""
        ate_results = get_ate_results(request.result_selection.entries)
        rpe_results = get_rpe_results(request.result_selection.entries)

        if not ate_results and not rpe_results:
            logger.error("No deviations selected!")
            return

        multi_report = render_multi_report(
            ate_results=ate_results, rpe_results=rpe_results, report_settings=request.report_settings
        )
        show_report(report_text=multi_report, filepath=self.report_path)

    def plot_correlation(self, request: PlotRequest) -> None:
        if not isinstance((entry := request.result_selection.entries[0]), AlignmentEntry):
            raise TypeError("Entry must be of type AlignmentEntry!")

        report = render_heatmaps(
            alignment_parameters=entry.estimated_parameters, name=entry.name, report_settings=request.report_settings
        )

        show_report(report_text=report, filepath=self.report_path)


def get_ate_results(entries: list[ResultEntry]) -> list[ATEResult]:
    """Return a list of ate results."""
    return [entry.deviations for entry in entries if isinstance(entry, AbsoluteDeviationEntry)]


def get_rpe_results(entries: list[ResultEntry]) -> list[RPEResult]:
    """Return a list of rpe results."""
    return [entry.deviations for entry in entries if isinstance(entry, RelativeDeviationEntry)]
