"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging
from typing import Callable, Dict

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from trajectopy_core.evaluation.ate_result import ATEResult
from trajectopy_core.evaluation.rpe_result import RPEResult
from trajectopy_core.report.alignment import render_heatmaps
from trajectopy_core.report.multi import render_multi_report
from trajectopy_core.report.single import render_single_report
from trajectopy_core.report.trajectory import render_trajectories
from trajectopy_core.report.utils import show_report
from trajectopy_core.settings.report import ReportSettings

from trajectopy.managers.requests import (
    PlotRequest,
    PlotRequestType,
    UIRequest,
    UIRequestType,
    generic_request_handler,
)
from trajectopy.models.entries import AbsoluteDeviationEntry, AlignmentEntry, RelativeDeviationEntry

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
        super().__init__(parent=parent)
        self.report_settings = ReportSettings()
        self.REQUEST_MAPPING: Dict[PlotRequestType, Callable[[PlotRequest], None]] = {
            PlotRequestType.TRAJECTORIES: self.plot_selected_trajectories,
            PlotRequestType.SINGLE_ABS_DEVIATIONS: self.plot_selected_abs_deviations,
            PlotRequestType.REL_DEVIATIONS: self.plot_multi_rel_deviations,
            PlotRequestType.TRAJECTORY_LAPS: self.plot_trajectory_laps,
            PlotRequestType.MULTI_ABS_DEVIATIONS: self.plot_multi_abs_deviations,
            PlotRequestType.DEVIATION_LAPS: self.plot_deviation_laps,
            PlotRequestType.UPDATE_SETTINGS: self.update_settings,
            PlotRequestType.CORRELATION: self.plot_correlation,
        }

    @pyqtSlot(PlotRequest)
    def handle_request(self, request: PlotRequest) -> None:
        """Logic for handling a request."""
        generic_request_handler(self, request, passthrough_request=True)

    def update_settings(self, request: PlotRequest) -> None:
        self.report_settings = request.plot_settings

    def plot_selected_trajectories(self, request: PlotRequest) -> None:
        trajectory_list = [entry.trajectory for entry in request.trajectory_selection.entries]
        traj_report = render_trajectories(trajectories=trajectory_list, report_settings=self.report_settings)
        show_report(traj_report, filepath="trajectopy_report.html")

    def plot_selected_abs_deviations(self, request: PlotRequest) -> None:
        if not isinstance((entry := request.result_selection.entries[0]), AbsoluteDeviationEntry):
            raise TypeError("Entry must be of type AbsoluteDeviationEntry!")

        if not isinstance((deviations := entry.deviations), ATEResult):
            raise TypeError("Deviations must be of type ATEResult!")

        report = render_single_report(ate_result=deviations, report_settings=self.report_settings)
        show_report(report_text=report, filepath="trajectopy_report.html")

    def plot_trajectory_laps(self, request: PlotRequest) -> None:
        """
        Plot laps of a trajectory.

        Only possible for spatially sorted trajectories.
        """
        trajectory_entry = request.trajectory_selection.entries[0]
        traj_list = trajectory_entry.trajectory.divide_into_laps()
        if traj_list is None:
            self.ui_request.emit(
                UIRequest(
                    type=UIRequestType.MESSAGE,
                    message="Please sort Trajectory first!",
                )
            )
            return

        traj_report = render_trajectories(trajectories=traj_list, report_settings=self.report_settings)
        show_report(traj_report, filepath="trajectopy_report.html")

    def plot_multi_abs_deviations(self, request: PlotRequest) -> None:
        """Plot multiple absolute deviations."""
        deviation_list = [
            entry.deviations
            for entry in request.result_selection.entries
            if (isinstance(entry, AbsoluteDeviationEntry) and isinstance(entry.deviations, ATEResult))
        ]
        multi_report = render_multi_report(ate_results=deviation_list)
        show_report(report_text=multi_report, filepath="trajectopy_report.html")

    def plot_multi_rel_deviations(self, request: PlotRequest) -> None:
        """Plot multiple relative deviations."""
        deviation_list = [
            entry.deviations
            for entry in request.result_selection.entries
            if (isinstance(entry, RelativeDeviationEntry) and isinstance(entry.deviations, RPEResult))
        ]
        multi_report = render_multi_report(rpe_results=deviation_list)
        show_report(report_text=multi_report, filepath="trajectopy_report.html")

    def plot_deviation_laps(self, request: PlotRequest) -> None:
        """
        Plot laps of absolute deviations.

        Only possible for spatially sorted deviations.
        """
        if not isinstance((entry := request.result_selection.entries[0]), AbsoluteDeviationEntry):
            raise TypeError("Entry must be of type AbsoluteDeviationEntry!")

        if not isinstance((deviations := entry.deviations), ATEResult):
            raise TypeError("Deviations must be of type ATEResult!")

        deviations_list = deviations.divide_into_laps()

        if deviations_list is None:
            raise ValueError("Failed to divide deviations into laps!")

        multi_report = render_multi_report(ate_results=deviations_list)
        show_report(report_text=multi_report, filepath="trajectopy_report.html")

    def plot_correlation(self, request: PlotRequest) -> None:
        if not isinstance((entry := request.result_selection.entries[0]), AlignmentEntry):
            raise TypeError("Entry must be of type AlignmentEntry!")

        report = render_heatmaps(alignment_parameters=entry.estimated_parameters, name=entry.name)

        show_report(report_text=report, filepath="trajectopy_report.html")
