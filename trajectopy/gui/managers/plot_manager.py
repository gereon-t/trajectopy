"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
import os
from typing import Callable, Dict, List

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

import trajectopy.api as tpy
from trajectopy.gui.managers.requests import PlotRequest, PlotRequestType, UIRequest, generic_request_handler
from trajectopy.gui.models.entries import AbsoluteDeviationEntry, AlignmentEntry, RelativeDeviationEntry, ResultEntry
from trajectopy.util import show_progress

logger = logging.getLogger("root")


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

    def __init__(self, report_dir: str, parent=None) -> None:
        self.cnt = 0
        super().__init__(parent)
        self.REQUEST_MAPPING: Dict[PlotRequestType, Callable[[PlotRequest], None]] = {
            PlotRequestType.TRAJECTORIES: self.plot_selected_trajectories,
            PlotRequestType.SINGLE_DEVIATIONS: self.plot_single_deviations,
            PlotRequestType.MULTI_DEVIATIONS: self.plot_multi_deviations,
            PlotRequestType.ALIGNMENT: self.plot_alignment,
        }
        self.report_dir = os.path.abspath(report_dir)

    @show_progress
    @pyqtSlot(PlotRequest)
    def handle_request(self, request: PlotRequest) -> None:
        """Logic for handling a request."""
        generic_request_handler(self, request, passthrough_request=True)

    def report_path(self, prefix: str = "report") -> str:
        """Return the path to the report file."""
        prefix = prefix.replace(" ", "_").replace(".", "")
        path = os.path.join(self.report_dir, f"{prefix}_{self.cnt:03}.html")
        self.cnt += 1
        return path

    def plot_selected_trajectories(self, request: PlotRequest) -> None:
        trajectory_list = [entry.trajectory for entry in request.trajectory_selection.entries]
        traj_report = tpy.create_trajectory_report(
            trajectories=trajectory_list, report_settings=request.report_settings
        )
        tpy.show_report(traj_report, filepath=self.report_path(prefix="trajectories"))

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

        report = tpy.create_deviation_report(
            ate_result=ate_result, rpe_result=rpe_result, report_settings=request.report_settings
        )
        tpy.show_report(
            report_text=report, filepath=self.report_path(prefix=ate_result.name if ate_result else rpe_result.name)
        )

    def plot_multi_deviations(self, request: PlotRequest) -> None:
        """Plot multiple absolute deviations."""
        ate_results = get_ate_results(request.result_selection.entries)
        rpe_results = get_rpe_results(request.result_selection.entries)

        if not ate_results and not rpe_results:
            logger.error("No deviations selected!")
            return

        multi_report = tpy.create_deviation_report(
            ate_results=ate_results or None,
            rpe_results=rpe_results or None,
            report_settings=request.report_settings,
        )
        tpy.show_report(report_text=multi_report, filepath=self.report_path(prefix="multi_deviations"))

    def plot_alignment(self, request: PlotRequest) -> None:
        if not isinstance((entry := request.result_selection.entries[0]), AlignmentEntry):
            raise TypeError("Entry must be of type AlignmentEntry!")

        report = tpy.create_alignment_report(
            alignment_parameters=entry.estimated_parameters, name=entry.name, report_settings=request.report_settings
        )

        tpy.show_report(report_text=report, filepath=self.report_path(prefix=entry.name))


def get_ate_results(entries: List[ResultEntry]) -> List[tpy.ATEResult]:
    """Return a list of ate results."""
    return [entry.deviations for entry in entries if isinstance(entry, AbsoluteDeviationEntry)]


def get_rpe_results(entries: List[ResultEntry]) -> List[tpy.RPEResult]:
    """Return a list of rpe results."""
    return [entry.deviations for entry in entries if isinstance(entry, RelativeDeviationEntry)]
