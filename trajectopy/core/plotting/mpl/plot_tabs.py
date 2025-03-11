"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""

import logging
from typing import List

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtGui, QtWidgets

from trajectopy.core.alignment.parameters import AlignmentParameters
from trajectopy.core.evaluation.ate_result import ATEResult
from trajectopy.core.evaluation.rpe_result import RPEResult
from trajectopy.plotting import (
    plot_ate,
    plot_ate_bars,
    plot_ate_edf,
    plot_compact_ate_hist,
    plot_correlation_heatmap,
    plot_covariance_heatmap,
    plot_rpe,
    plot_trajectories,
    scatter_ate,
)
from trajectopy.settings import MPLPlotSettings
from trajectopy.trajectory import Trajectory

logger = logging.getLogger("root")


class PlotTabs(QtWidgets.QMainWindow):
    """Frame for plotting multiple figures in tabs"""

    def __init__(self, parent, window_title: str = "Trajectopy - Viewer"):
        super().__init__(parent=parent)
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

    def show_trajectories(
        self, trajectories: List[Trajectory], mpl_plot_settings: MPLPlotSettings = MPLPlotSettings()
    ) -> None:
        """
        Plots Trajectories
        """
        if not trajectories:
            return

        self.tabs.clear()
        fig_pos, fig_xyz, fig_rpy = plot_trajectories(trajectories, dim=mpl_plot_settings.scatter_pos_dim)
        # create tab group
        self.addPlot("Trajectory", fig_pos)
        self.addPlot("XYZ", fig_xyz)
        if fig_rpy is not None:
            self.addPlot("RPY", fig_rpy)
        self.show()

    def show_single_deviations(
        self,
        ate_result: ATEResult,
        rpe_result: RPEResult = None,
        mpl_plot_settings: MPLPlotSettings = MPLPlotSettings(),
        title: str = "",
    ) -> None:
        """Plots single trajectory deviations"""
        self.tabs.clear()

        if not title:
            if ate_result is not None:
                title = ate_result.name
            elif rpe_result is not None:
                title = rpe_result.name
            else:
                title = "Deviations"

        if ate_result is not None:
            fig_ate_hist = plot_compact_ate_hist(ate_result=ate_result, plot_settings=mpl_plot_settings)
            self.addPlot("ATE Histograms", fig_ate_hist)

            fig_ate_line = plot_ate(ate_result, plot_settings=mpl_plot_settings)
            self.addPlot("ATE Line Plot", fig_ate_line)

            fig_ate_bars = plot_ate_bars([ate_result], plot_settings=mpl_plot_settings, mode="positions")
            self.addPlot("ATE Bars (Positions)", fig_ate_bars)

            if ate_result.has_orientation:
                fig_ate_bars_rot = plot_ate_bars([ate_result], plot_settings=mpl_plot_settings, mode="rotations")
                self.addPlot("ATE Bars (Rotations)", fig_ate_bars_rot)

            fig_cdf = plot_ate_edf(ate_result, plot_settings=mpl_plot_settings)
            self.addPlot("Empirical Distribution Function", fig_cdf)

            fig_scatter_pos, fig_scatter_rot = scatter_ate(ate_result, plot_settings=mpl_plot_settings)
            self.addPlot("ATE Scatter Plot (Positions)", fig_scatter_pos)
            if fig_scatter_rot is not None:
                self.addPlot("ATE Scatter Plot (Rotations)", fig_scatter_rot)

        if rpe_result is not None:
            fig_rpe_metric, fig_rpe_time = plot_rpe(rpe_result)

            if fig_rpe_metric is not None:
                self.addPlot("RPE per meter", fig_rpe_metric)

            if fig_rpe_time is not None:
                self.addPlot("RPE per second", fig_rpe_time)

        self.show()

    def show_multiple_deviations(
        self,
        ate_results: list[ATEResult],
        rpe_results: list[RPEResult],
        mpl_plot_settings: MPLPlotSettings = MPLPlotSettings(),
        title: str = "",
    ) -> None:
        """Plots single trajectory deviations"""
        self.tabs.clear()

        if not title:
            title = "Trajectory Comparison"

        if ate_results:
            fig_ate_line = plot_ate(ate_results, plot_settings=mpl_plot_settings)
            self.addPlot("ATE Line Plot", fig_ate_line)

            fig_ate_bars = plot_ate_bars(ate_results, plot_settings=mpl_plot_settings, mode="positions")
            self.addPlot("ATE Bars (Positions)", fig_ate_bars)

            ate_results_with_rot = [ate for ate in ate_results if ate.has_orientation]
            if ate_results_with_rot:
                fig_ate_bars_rot = plot_ate_bars(
                    ate_results_with_rot, plot_settings=mpl_plot_settings, mode="rotations"
                )
                self.addPlot("ATE Bars (Rotations)", fig_ate_bars_rot)

            fig_cdf = plot_ate_edf(ate_results, plot_settings=mpl_plot_settings)
            self.addPlot("Empirical Distribution Function", fig_cdf)

        if rpe_results:
            fig_rpe_metric, fig_rpe_time = plot_rpe(rpe_results)

            if fig_rpe_metric is not None:
                self.addPlot("RPE per meter", fig_rpe_metric)

            if fig_rpe_time is not None:
                self.addPlot("RPE per second", fig_rpe_time)

        self.show()

    def show_alignment_parameters(self, estimated_parameters: AlignmentParameters) -> None:
        self.tabs.clear()
        fig_covariance = plot_covariance_heatmap(estimated_parameters)
        self.addPlot("Parameter Covariance", fig_covariance)
        fig_correlation = plot_correlation_heatmap(estimated_parameters)
        self.addPlot("Parameter Correlation", fig_correlation)
        self.show()
