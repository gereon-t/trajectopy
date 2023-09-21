"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging
from typing import List

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtGui, QtWidgets
from trajectopy_core.alignment.parameters import AlignmentParameters
from trajectopy_core.evaluation.abs_traj_dev import AbsoluteTrajectoryDeviations, DeviationCollection
from trajectopy_core.evaluation.rel_traj_dev import RelativeTrajectoryDeviations
from trajectopy_core.plotting.alignment_plot import plot_correlation_heatmap, plot_covariance_heatmap
from trajectopy_core.plotting.deviation_plot import (
    plot_bars,
    plot_bias_heatmap,
    plot_combined_devs,
    plot_compact_deviations,
    plot_compact_hist,
    plot_dof_dev,
    plot_edf,
    plot_multiple_comb_deviations,
    plot_multiple_deviations,
    plot_raw_position_devs,
    plot_raw_rotation_devs,
    plot_rms_heatmap,
    plot_rpe,
)
from trajectopy_core.plotting.trajectory_plot import plot_trajectories
from trajectopy_core.settings.plot_settings import PlotSettings
from trajectopy_core.trajectory import Trajectory

from trajectopy.gui.path import mplstyle_file_path

logger = logging.getLogger("root")


class PlotTabs(QtWidgets.QMainWindow):
    """Frame for plotting multiple figures in tabs"""

    def __init__(self, parent, window_title: str = "Trajectopy - Viewer"):
        super().__init__(parent=parent)
        plt.style.use(mplstyle_file_path())
        self.tabs = QtWidgets.QTabWidget()
        self.setWindowTitle(window_title)
        self.setCentralWidget(self.tabs)

        if (primary_screen := QtGui.QGuiApplication.primaryScreen()) is not None:
            screen_geometry = primary_screen.availableGeometry()
            self.resize(screen_geometry.width(), screen_geometry.height() - 50)
        else:
            logger.warning("Could not determine screen size. Using default size.")

        desired_pos = QtCore.QPoint(screen_geometry.left(), screen_geometry.top())
        self.move(desired_pos)

        self.canvases: list = []
        self.figure_handles: list = []
        self.toolbar_handles: list = []
        self.tab_handles: list = []
        self.current_window = -1

    def addPlot(self, title: str, figure: Figure):
        """Adds a new tab with a plot"""
        new_tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        new_tab.setLayout(layout)

        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        self.tabs.addTab(new_tab, title)

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)

    def show_trajectories(self, trajectories: List[Trajectory], dim: int = 2) -> None:
        """
        Plots Trajectories
        """
        if not trajectories:
            return

        self.tabs.clear()
        fig_pos, fig_xyz, fig_rpy = plot_trajectories(trajectories, dim=dim)
        # create tab group
        self.addPlot("Trajectory", fig_pos)
        self.addPlot("XYZ", fig_xyz)
        if fig_rpy is not None:
            self.addPlot("RPY", fig_rpy)
        self.show()

    def show_single_abs_deviations(
        self,
        devs: AbsoluteTrajectoryDeviations,
        plot_settings: PlotSettings = PlotSettings(),
        title: str = "",
    ) -> None:
        """Plots single trajectory deviations"""
        # compact deviations
        if len(devs.ate_result.pos_dev) < 1:
            logger.error("Please compute deviations first!")
            return

        self.tabs.clear()

        if not title:
            title = devs.name

        # compact histograms
        # devs somehow change?!
        fig_hist = plot_compact_hist(devs=devs, plot_settings=plot_settings)
        self.addPlot("Combined Histograms", fig_hist)

        fig_cdf = plot_edf(devs, plot_settings=plot_settings)
        self.addPlot("Empirical Distribution Function", fig_cdf)

        # compact deviations
        fig_rms = plot_compact_deviations(devs=devs, plot_settings=plot_settings)
        dev_comb_lengths_fig = plot_combined_devs(devs, plot_settings=plot_settings)
        self.addPlot("Combined Deviations (trajectory length)", dev_comb_lengths_fig)

        if fig_rms is not None:
            self.addPlot("Combined RMS (xy-view)", fig_rms)

        # deviations for each DOF
        # devs somehow change?!
        dof_devs_fig = plot_dof_dev(devs=devs, plot_settings=plot_settings)

        self.addPlot("Deviations for each DOF", dof_devs_fig)

        dev_pos_lengths_fig = plot_raw_position_devs(devs, plot_settings=plot_settings)
        self.addPlot("Raw Position Deviations", dev_pos_lengths_fig)

        if devs.ate_result.rot_dev is not None:
            # devs somehow change?!
            dev_rot_lengths_fig = plot_raw_rotation_devs(devs, plot_settings=plot_settings)
            self.addPlot("Raw Orientation Deviations", dev_rot_lengths_fig)

        self.show()

    def show_multi_rel_deviations(self, devs: List[RelativeTrajectoryDeviations], title: str = "") -> None:
        self.tabs.clear()

        if not title and len(devs) == 1:
            title = devs[0].name

        fig_rpe_metric, fig_rpe_time = plot_rpe(devs=devs)

        if fig_rpe_metric is not None:
            self.addPlot("RPE per meter", fig_rpe_metric)

        if fig_rpe_time is not None:
            self.addPlot("RPE per second", fig_rpe_time)

        self.show()

    def show_multi_abs_deviations(
        self,
        deviation_list: List[AbsoluteTrajectoryDeviations],
        plot_settings: PlotSettings = PlotSettings(),
    ) -> None:
        """Plots multiple trajectory deviations


        Args:
            deviations (list[TrajectoryDeviations]): List of trajectory deviations.
        """
        self.tabs.clear()
        deviation_collection = DeviationCollection(deviations=deviation_list)

        pos_rot_dev_fig = plot_multiple_comb_deviations(deviation_list, plot_settings=plot_settings)
        self.addPlot("Trajectory Deviations", pos_rot_dev_fig)

        xyz_dev_fig, rpy_dev_fig = plot_multiple_deviations(deviation_collection, plot_settings=plot_settings)
        self.addPlot("XYZ Deviations", xyz_dev_fig)

        if rpy_dev_fig is not None:
            self.addPlot("RPY Deviations", rpy_dev_fig)

        fig_cdf = plot_edf(deviation_list, plot_settings=plot_settings)
        self.addPlot("Empirical Distribution Function", fig_cdf)

        fig_bars_pos = plot_bars(deviation_list, plot_settings=plot_settings, mode="positions")
        self.addPlot("Bar Plot (Positions)", fig_bars_pos)

        if rpy_dev_fig is not None:
            fig_bars_rot = plot_bars(deviation_list, plot_settings=plot_settings, mode="rotations")
            self.addPlot("Bar Plot (Rotations)", fig_bars_rot)

        pos_bias_fig, rpy_bias_fig = plot_bias_heatmap(deviation_collection, plot_settings=plot_settings)
        self.addPlot("Heatmap: Position Bias", pos_bias_fig)

        if rpy_bias_fig is not None:
            self.addPlot("Heatmap: RPY Bias", rpy_bias_fig)

        pos_rms_fig, rpy_rms_fig = plot_rms_heatmap(deviation_collection, plot_settings=plot_settings)
        self.addPlot("Heatmap: Position RMS", pos_rms_fig)

        if rpy_rms_fig is not None:
            self.addPlot("Heatmap: RPY RMS", rpy_rms_fig)
        self.show()

    def show_estimation(self, estimated_parameters: AlignmentParameters) -> None:
        self.tabs.clear()
        fig_covariance = plot_covariance_heatmap(estimated_parameters)
        self.addPlot("Parameter Covariance", fig_covariance)
        fig_correlation = plot_correlation_heatmap(estimated_parameters)
        self.addPlot("Parameter Correlation", fig_correlation)
        self.show()
