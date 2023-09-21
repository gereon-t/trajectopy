"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging
from typing import Callable, Dict

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from trajectopy_core.evaluation.abs_traj_dev import AbsoluteTrajectoryDeviations
from trajectopy_core.evaluation.rel_traj_dev import RelativeTrajectoryDeviations
from trajectopy_core.settings.plot_settings import PlotSettings
from trajectopy_core.util.entries import AbsoluteDeviationEntry, AlignmentEntry, RelativeDeviationEntry

from trajectopy.gui.managers.requests import (
    PlotRequest,
    PlotRequestType,
    UIRequest,
    UIRequestType,
    generic_request_handler,
)
from trajectopy.gui.views.plot_tabs import PlotTabs

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
        self.plot_settings = PlotSettings()
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
        self.plot_settings = request.plot_settings

    def plot_selected_trajectories(self, request: PlotRequest) -> None:
        trajectory_list = [entry.trajectory for entry in request.trajectory_selection.entries]
        plot_tabs = PlotTabs(parent=self.parent())
        plot_tabs.show_trajectories(trajectory_list, dim=request.dimension)

    def plot_selected_abs_deviations(self, request: PlotRequest) -> None:
        if not isinstance((entry := request.result_selection.entries[0]), AbsoluteDeviationEntry):
            raise TypeError("Entry must be of type AbsoluteDeviationEntry!")

        if not isinstance((deviations := entry.deviations), AbsoluteTrajectoryDeviations):
            raise TypeError("Deviations must be of type AbsoluteTrajectoryDeviations!")

        plot_tabs = PlotTabs(parent=self.parent())
        plot_tabs.show_single_abs_deviations(devs=deviations, plot_settings=self.plot_settings)

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

        plot_tabs = PlotTabs(parent=self.parent())
        plot_tabs.show_trajectories(traj_list)

    def plot_multi_abs_deviations(self, request: PlotRequest) -> None:
        """Plot multiple absolute deviations."""
        plot_tabs = PlotTabs(parent=self.parent())
        deviation_list = [
            entry.deviations
            for entry in request.result_selection.entries
            if (
                isinstance(entry, AbsoluteDeviationEntry)
                and isinstance(entry.deviations, AbsoluteTrajectoryDeviations)
            )
        ]
        plot_tabs.show_multi_abs_deviations(deviation_list=deviation_list, plot_settings=self.plot_settings)

    def plot_multi_rel_deviations(self, request: PlotRequest) -> None:
        """Plot multiple relative deviations."""
        plot_tabs = PlotTabs(parent=self.parent())
        deviation_list = [
            entry.deviations
            for entry in request.result_selection.entries
            if (
                isinstance(entry, RelativeDeviationEntry)
                and isinstance(entry.deviations, RelativeTrajectoryDeviations)
            )
        ]
        plot_tabs.show_multi_rel_deviations(devs=deviation_list)

    def plot_deviation_laps(self, request: PlotRequest) -> None:
        """
        Plot laps of absolute deviations.

        Only possible for spatially sorted deviations.
        """
        if not isinstance((entry := request.result_selection.entries[0]), AbsoluteDeviationEntry):
            raise TypeError("Entry must be of type AbsoluteDeviationEntry!")

        if not isinstance((deviations := entry.deviations), AbsoluteTrajectoryDeviations):
            raise TypeError("Deviations must be of type AbsoluteTrajectoryDeviations!")

        plot_tabs = PlotTabs(parent=self.parent())
        deviations_list = deviations.divide_into_laps()

        if deviations_list is None:
            raise ValueError("Failed to divide deviations into laps!")

        plot_tabs.show_multi_abs_deviations(
            deviation_list=deviations_list,
            plot_settings=self.plot_settings,
        )

    def plot_correlation(self, request: PlotRequest) -> None:
        if not isinstance((entry := request.result_selection.entries[0]), AlignmentEntry):
            raise TypeError("Entry must be of type AlignmentEntry!")

        estimated_parameters = entry.estimated_parameters
        plot_tabs = PlotTabs(parent=self.parent())
        plot_tabs.show_estimation(estimated_parameters=estimated_parameters)
