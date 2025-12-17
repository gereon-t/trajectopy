"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""

import logging
from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6 import QtGui, QtWidgets

from trajectopy.core.settings import MPLPlotSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.lib.alignment.parameters import AlignmentParameters
from trajectopy.results.ate_result import ATEResult
from trajectopy.results.rpe_result import RPEResult
from trajectopy.utils.common import get_axis_label
from trajectopy.visualization.mpl_plots import (
    plot_ate,
    plot_ate_3d,
    plot_ate_bars,
    plot_ate_dof,
    plot_ate_edf,
    plot_compact_ate_hist,
    plot_correlation_heatmap,
    plot_covariance_heatmap,
    plot_rpe,
    plot_trajectories,
    scatter_ate,
)

logger = logging.getLogger(__name__)


@dataclass
class PlotableDropdownItem:
    """Class for dropdown items in the plot tabs"""

    name: str
    data: np.ndarray
    color_data: np.ndarray
    colorbar_label: str
    x_label: str = "X"
    y_label: str = "Y"
    smooth: bool = False
    smooth_window: int = 5


class PlotTabs(QtWidgets.QMainWindow):
    """Frame for plotting multiple figures in tabs"""

    def __init__(self, parent, window_title: str = "Trajectopy - Viewer"):
        super().__init__(parent=parent)
        self.tabs = QtWidgets.QTabWidget()
        self.setWindowTitle(window_title)
        self.setCentralWidget(self.tabs)

        if (primary_screen := QtGui.QGuiApplication.primaryScreen()) is not None:
            screen_geometry = primary_screen.availableGeometry()
            self.resize(int(screen_geometry.width() * 0.8), int(screen_geometry.height() * 0.8) - 50)
        else:
            logger.warning("Could not determine screen size. Using default size.")

        self.center()

        self.canvases: list = []
        self.figure_handles: list = []
        self.toolbar_handles: list = []
        self.tab_handles: list = []
        self.current_window = -1

    def center(self):
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()

        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def add_plot(self, title: str, figure: Figure):
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

    def add_dropdown_plot(self, name: str, dropdown_items: list[PlotableDropdownItem]):
        """
        Adds a tab with a scatter plot and a dropdown to choose coloring.

        Parameters:
        - title: Title of the tab
        - data: Nx2 array (X, Y)
        - color_options: Dict mapping dropdown option labels to color arrays (length N)
        """
        new_tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(new_tab)
        dropdown = QtWidgets.QComboBox()
        dropdown.addItems(item.name for item in dropdown_items)

        figure = Figure()
        ax = figure.add_subplot(111)
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, new_tab)

        colorbar = None

        def update_plot(selected_key):
            nonlocal colorbar

            selected_item = next(item for item in dropdown_items if item.name == selected_key)
            c_list = selected_item.color_data
            if selected_item.smooth:
                c_list = np.convolve(
                    c_list,
                    np.ones(selected_item.smooth_window) / selected_item.smooth_window,
                    mode="same",
                )

            ax.clear()
            scatter = ax.scatter(selected_item.data[:, 0], selected_item.data[:, 1], c=c_list, cmap="RdYlBu_r")

            ax.set_xlabel(selected_item.x_label)
            ax.set_ylabel(selected_item.y_label)
            ax.set_title(selected_key)
            ax.axis("equal")

            figure.subplots_adjust(right=0.85)

            if colorbar:
                colorbar.update_normal(scatter)
                colorbar.set_label(selected_item.colorbar_label)
            else:
                colorbar = figure.colorbar(
                    scatter,
                    ax=ax,
                    pad=0.02,
                    fraction=0.046,
                    label=selected_item.colorbar_label,
                )

            canvas.draw()

        dropdown.currentTextChanged.connect(update_plot)

        layout.addWidget(dropdown)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        self.tabs.addTab(new_tab, name)
        self.tab_handles.append(new_tab)
        self.canvases.append(canvas)
        self.figure_handles.append(figure)
        self.toolbar_handles.append(toolbar)

        update_plot(dropdown.currentText())

    def show_trajectories(
        self, trajectories: list[Trajectory], mpl_plot_settings: MPLPlotSettings = MPLPlotSettings()
    ) -> None:
        """
        Plots Trajectories
        """
        if not trajectories:
            return

        x_label, y_label, _ = get_axis_label(trajectories)
        self.tabs.clear()
        fig_pos, fig_xyz, fig_rpy = plot_trajectories(trajectories, scatter_3d=mpl_plot_settings.scatter_3d)
        # create tab group
        self.add_plot("Trajectory", fig_pos)
        self.add_plot("XYZ", fig_xyz)
        if fig_rpy is not None:
            self.add_plot("RPY", fig_rpy)

        if mpl_plot_settings.velocity_tab:
            self.add_dropdown_plot(
                name="Velocity",
                dropdown_items=[
                    PlotableDropdownItem(
                        name=f"{traj.name} Velocity",
                        data=traj.positions.xyz,  # unsorted here
                        color_data=traj.absolute_velocity,
                        colorbar_label="Velocity [m/s]",
                        x_label=x_label,
                        y_label=y_label,
                        smooth=mpl_plot_settings.scatter_smooth,
                        smooth_window=mpl_plot_settings.scatter_smooth_window,
                    )
                    for traj in trajectories
                ],
            )

        if mpl_plot_settings.height_tab:
            self.add_dropdown_plot(
                name="Height",
                dropdown_items=[
                    PlotableDropdownItem(
                        name=f"{traj.name} Height",
                        data=traj.xyz,
                        color_data=traj.xyz[:, 2],
                        colorbar_label="Height [m]",
                        x_label=x_label,
                        y_label=y_label,
                        smooth=mpl_plot_settings.scatter_smooth,
                        smooth_window=mpl_plot_settings.scatter_smooth_window,
                    )
                    for traj in trajectories
                ],
            )
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
            x_label, y_label, _ = get_axis_label([ate_result.trajectory])
            fig_ate_hist = plot_compact_ate_hist(ate_result=ate_result, plot_settings=mpl_plot_settings)
            self.add_plot("ATE Histograms", fig_ate_hist)

            fig_ate_line = plot_ate(ate_result, plot_settings=mpl_plot_settings)
            self.add_plot("ATE Line Plot", fig_ate_line)

            fig_ate_dof = plot_ate_dof(ate_result, plot_settings=mpl_plot_settings)
            self.add_plot("ATE DOFs", fig_ate_dof)

            fig_ate_bars = plot_ate_bars([ate_result], plot_settings=mpl_plot_settings, mode="positions")
            self.add_plot("ATE Bars (Positions)", fig_ate_bars)

            fig_ate_3d = plot_ate_3d([ate_result], plot_settings=mpl_plot_settings)
            self.add_plot("ATE 3D Plot", fig_ate_3d)

            if ate_result.has_orientation:
                fig_ate_bars_rot = plot_ate_bars([ate_result], plot_settings=mpl_plot_settings, mode="rotations")
                self.add_plot("ATE Bars (Rotations)", fig_ate_bars_rot)

            fig_cdf = plot_ate_edf(ate_result, plot_settings=mpl_plot_settings)
            self.add_plot("Empirical Distribution Function", fig_cdf)

            fig_scatter_pos, fig_scatter_rot = scatter_ate(ate_result, plot_settings=mpl_plot_settings)
            self.add_plot("ATE Scatter Plot (Positions)", fig_scatter_pos)

            if fig_scatter_rot is not None:
                self.add_plot("ATE Scatter Plot (Rotations)", fig_scatter_rot)

            if mpl_plot_settings.dofs_tab:
                self.add_dropdown_plot(
                    name="DOFs",
                    dropdown_items=[
                        PlotableDropdownItem(
                            name="X",
                            data=ate_result.trajectory.xyz,
                            color_data=ate_result.pos_dev_x * mpl_plot_settings.unit_multiplier,
                            colorbar_label=f"Deviation {mpl_plot_settings.unit_str}",
                            x_label=x_label,
                            y_label=y_label,
                            smooth=mpl_plot_settings.scatter_smooth,
                            smooth_window=mpl_plot_settings.scatter_smooth_window,
                        ),
                        PlotableDropdownItem(
                            name="Y",
                            data=ate_result.trajectory.xyz,
                            color_data=ate_result.pos_dev_y * mpl_plot_settings.unit_multiplier,
                            colorbar_label=f"Deviation {mpl_plot_settings.unit_str}",
                            x_label=x_label,
                            y_label=y_label,
                            smooth=mpl_plot_settings.scatter_smooth,
                            smooth_window=mpl_plot_settings.scatter_smooth_window,
                        ),
                        PlotableDropdownItem(
                            name="Z",
                            data=ate_result.trajectory.xyz,
                            color_data=ate_result.pos_dev_z * mpl_plot_settings.unit_multiplier,
                            colorbar_label=f"Deviation {mpl_plot_settings.unit_str}",
                            x_label=x_label,
                            y_label=y_label,
                            smooth=mpl_plot_settings.scatter_smooth,
                            smooth_window=mpl_plot_settings.scatter_smooth_window,
                        ),
                        PlotableDropdownItem(
                            name="Along-Track",
                            data=ate_result.trajectory.xyz,
                            color_data=ate_result.pos_dev_along * mpl_plot_settings.unit_multiplier,
                            colorbar_label=f"Deviation {mpl_plot_settings.unit_str}",
                            x_label=x_label,
                            y_label=y_label,
                            smooth=mpl_plot_settings.scatter_smooth,
                            smooth_window=mpl_plot_settings.scatter_smooth_window,
                        ),
                        PlotableDropdownItem(
                            name="Horizontal Cross-Track",
                            data=ate_result.trajectory.xyz,
                            color_data=ate_result.pos_dev_cross_h * mpl_plot_settings.unit_multiplier,
                            colorbar_label=f"Deviation {mpl_plot_settings.unit_str}",
                            x_label=x_label,
                            y_label=y_label,
                            smooth=mpl_plot_settings.scatter_smooth,
                            smooth_window=mpl_plot_settings.scatter_smooth_window,
                        ),
                        PlotableDropdownItem(
                            name="Vertical Cross-Track",
                            data=ate_result.trajectory.xyz,
                            color_data=ate_result.pos_dev_cross_v * mpl_plot_settings.unit_multiplier,
                            colorbar_label=f"Deviation {mpl_plot_settings.unit_str}",
                            x_label=x_label,
                            y_label=y_label,
                            smooth=mpl_plot_settings.scatter_smooth,
                            smooth_window=mpl_plot_settings.scatter_smooth_window,
                        ),
                        PlotableDropdownItem(
                            name="Roll",
                            data=ate_result.trajectory.xyz,
                            color_data=np.rad2deg(ate_result.rot_dev_x),
                            colorbar_label="Deviation [°]",
                            x_label=x_label,
                            y_label=y_label,
                            smooth=mpl_plot_settings.scatter_smooth,
                            smooth_window=mpl_plot_settings.scatter_smooth_window,
                        ),
                        PlotableDropdownItem(
                            name="Pitch",
                            data=ate_result.trajectory.xyz,
                            color_data=np.rad2deg(ate_result.rot_dev_y),
                            colorbar_label="Deviation [°]",
                            x_label=x_label,
                            y_label=y_label,
                            smooth=mpl_plot_settings.scatter_smooth,
                            smooth_window=mpl_plot_settings.scatter_smooth_window,
                        ),
                        PlotableDropdownItem(
                            name="Yaw",
                            data=ate_result.trajectory.xyz,
                            color_data=np.rad2deg(ate_result.rot_dev_z),
                            colorbar_label="Deviation [°]",
                            x_label=x_label,
                            y_label=y_label,
                            smooth=mpl_plot_settings.scatter_smooth,
                            smooth_window=mpl_plot_settings.scatter_smooth_window,
                        ),
                    ],
                )

        if rpe_result is not None:
            fig_rpe_metric, fig_rpe_time = plot_rpe(rpe_result)

            if fig_rpe_metric is not None:
                self.add_plot("RPE per meter", fig_rpe_metric)

            if fig_rpe_time is not None:
                self.add_plot("RPE per second", fig_rpe_time)

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
            self.add_plot("ATE Line Plot", fig_ate_line)

            fig_ate_bars = plot_ate_bars(ate_results, plot_settings=mpl_plot_settings, mode="positions")
            self.add_plot("ATE Bars (Positions)", fig_ate_bars)

            fig_ate_3d = plot_ate_3d(ate_results, plot_settings=mpl_plot_settings)
            self.add_plot("ATE 3D Plot", fig_ate_3d)

            ate_results_with_rot = [ate for ate in ate_results if ate.has_orientation]
            if ate_results_with_rot:
                fig_ate_bars_rot = plot_ate_bars(
                    ate_results_with_rot, plot_settings=mpl_plot_settings, mode="rotations"
                )
                self.add_plot("ATE Bars (Rotations)", fig_ate_bars_rot)

            fig_cdf = plot_ate_edf(ate_results, plot_settings=mpl_plot_settings)
            self.add_plot("Empirical Distribution Function", fig_cdf)

        if rpe_results:
            fig_rpe_metric, fig_rpe_time = plot_rpe(rpe_results)

            if fig_rpe_metric is not None:
                self.add_plot("RPE per meter", fig_rpe_metric)

            if fig_rpe_time is not None:
                self.add_plot("RPE per second", fig_rpe_time)

        self.show()

    def show_alignment_parameters(self, estimated_parameters: AlignmentParameters) -> None:
        self.tabs.clear()
        fig_covariance = plot_covariance_heatmap(estimated_parameters)
        self.add_plot("Parameter Covariance", fig_covariance)
        fig_correlation = plot_correlation_heatmap(estimated_parameters)
        self.add_plot("Parameter Correlation", fig_correlation)
        self.show()
