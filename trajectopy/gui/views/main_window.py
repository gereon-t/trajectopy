import logging
import os
import shutil
from collections.abc import Callable
from tempfile import mkdtemp

import matplotlib
import numpy as np

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent

from trajectopy import __version__ as VERSION
from trajectopy.core.settings import MPLPlotSettings, PlotBackend, ReportSettings
from trajectopy.gui.managers.file_manager import FileManager
from trajectopy.gui.managers.plot_manager import PlotManager
from trajectopy.gui.managers.requests import (
    PlotRequest,
    PlotSettingsRequest,
    PlotSettingsRequestType,
    UIRequest,
    UIRequestType,
    generic_request_handler,
)
from trajectopy.gui.managers.session_manager import SessionManager
from trajectopy.gui.managers.trajectory_manager import TrajectoryManager
from trajectopy.gui.managers.ui_manager import UIManager
from trajectopy.gui.models.result_model import ResultTableModel
from trajectopy.gui.models.trajectory_model import TrajectoryTableModel
from trajectopy.gui.themes import DARK_STYLESHEET, LIGHT_STYLESHEET, MPL_COLORS
from trajectopy.gui.utils import center_window, set_dark_titlebar
from trajectopy.gui.views.about_window import AboutGUI
from trajectopy.gui.views.json_settings_view import JSONViewer
from trajectopy.gui.views.progress_window import ProgressWindow
from trajectopy.gui.views.result_table_view import ResultTableView
from trajectopy.gui.views.timeline_widget import TimelineDialog
from trajectopy.gui.views.trajectory_table_view import TrajectoryTableView
from trajectopy.utils.common import YEAR

logger = logging.getLogger(__name__)


class TrajectopyGUI(QtWidgets.QMainWindow):
    """
    Class representing the main window of the GUI

    This class is the main window of the GUI.
    It contains the table views for trajectories
    and results as well as the menu bar.
    """

    operation_started = Signal()
    operation_finished = Signal()
    ui_request = Signal(UIRequest)

    def __init__(
        self,
        single_thread: bool = False,
        report_settings_path: str = "",
        mpl_plot_settings_path: str = "",
        report_output_path: str = "",
        mapbox_token: str = "",
    ) -> None:
        QtWidgets.QMainWindow.__init__(self)

        self.REQUEST_MAPPING: dict[PlotSettingsRequestType, Callable[[PlotSettingsRequest], None]] = {
            PlotSettingsRequestType.EXPORT: self.handle_plot_settings_export,
            PlotSettingsRequestType.IMPORT: self.handle_plot_settings_import,
            PlotSettingsRequestType.RESET: self.handle_plot_settings_reset,
        }

        self._dark_theme = QtWidgets.QApplication.instance().styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
        self.trajectory_table_model = TrajectoryTableModel()
        self.result_table_model = ResultTableModel()
        self.setupUi()
        self.computation_thread: QtCore.QThread | None = None

        self.trajectory_manager = TrajectoryManager()
        self.ui_manager = UIManager(parent=self)
        self.file_manager = FileManager()
        self.session_manager = SessionManager()

        self.report_settings = self.get_report_settings(report_settings_path)
        self.report_settings.scatter_mapbox_token = mapbox_token or self.get_mapbox_token()
        self.report_output_path = self.get_report_directory(report_output_path)

        self.mpl_plot_settings = self.get_mpl_plot_settings(mpl_plot_settings_path)

        self.plot_manager = PlotManager(
            report_dir=self.report_output_path,
            plot_backend=PlotBackend.MPL,
            parent=self,
        )

        if not single_thread:
            self.computation_thread = QtCore.QThread(parent=self)
            self.trajectory_manager.moveToThread(self.computation_thread)
            self.file_manager.moveToThread(self.computation_thread)
            logger.info("Multithreading enabled")
        else:
            logger.info("Multithreading disabled")

        self.about_window = AboutGUI(parent=self, version_str=VERSION, year_str=YEAR)
        self.progress_window = ProgressWindow(parent=self)

        self.ui_request.connect(self.ui_manager.handle_request)
        self.setup_io_connections()
        self.setup_worker_connections()
        self.setup_progress_connections()
        self.setup_menu_bar()

        if self.computation_thread is not None:
            self.computation_thread.start()

        set_dark_titlebar(self, self._dark_theme)
        self.show()

    def get_report_directory(self, report_output_path) -> str:
        if not report_output_path:
            self.temp_dir = True
            return mkdtemp(prefix="trajectopy_reports_")
        else:
            self.temp_dir = False
            target_dir = os.path.abspath(report_output_path)
            os.makedirs(target_dir, exist_ok=True)
            return target_dir

    def get_report_settings(self, report_settings_path: str = "") -> ReportSettings:
        if report_settings_path:
            try:
                logger.info("Loaded report settings from %s", report_settings_path)
                return ReportSettings.from_file(report_settings_path)
            except Exception as e:
                logger.error("Could not load report settings file: %s", e)
                return ReportSettings()
        else:
            return ReportSettings()

    def get_mpl_plot_settings(self, mpl_plot_settings_path: str = "") -> MPLPlotSettings:
        if mpl_plot_settings_path:
            try:
                logger.info("Loaded MPL plot settings from %s", mpl_plot_settings_path)
                return MPLPlotSettings.from_file(mpl_plot_settings_path)
            except Exception as e:
                logger.error("Could not load MPL plot settings file: %s", e)
                return MPLPlotSettings()
        else:
            return MPLPlotSettings()

    def get_mapbox_token(self) -> str:
        mapbox_token = os.environ.get("MAPBOX_TOKEN")

        if mapbox_token:
            logger.info("Using mapbox token from environment variable")
            return mapbox_token

        mapbox_token_file = os.path.join(os.getcwd(), ".mapbox_token")
        if os.path.isfile(mapbox_token_file):
            with open(mapbox_token_file, encoding="utf-8") as f:
                mapbox_token = f.read()
                logger.info("Using mapbox token from file")
                return mapbox_token

        return ""

    def setup_menu_bar(self):
        """
        This method sets up the menu bar containing
        the session, plot settings and about menu
        """
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        session_menu = QtWidgets.QMenu("Session", parent=self)
        new_session_action = QAction("New", parent=self)
        new_session_action.triggered.connect(self.handle_new_session)
        session_menu.addAction(new_session_action)

        load_session_action = QAction("Load", parent=self)
        load_session_action.triggered.connect(self.handle_import_session)
        session_menu.addAction(load_session_action)

        save_session_action = QAction("Save", parent=self)
        save_session_action.triggered.connect(self.handle_export_session)
        session_menu.addAction(save_session_action)
        menubar.addMenu(session_menu)

        self.report_settings_action = QAction("Report Settings", parent=self)
        self.report_settings_action.triggered.connect(self.handle_show_report_settings)
        self.report_settings_action.setVisible(self.plot_manager.plot_backend == PlotBackend.PLOTLY)
        menubar.addAction(self.report_settings_action)

        self.plot_settings_action = QAction("Plot Settings", parent=self)
        self.plot_settings_action.triggered.connect(self.handle_show_mpl_settings)
        self.plot_settings_action.setVisible(self.plot_manager.plot_backend == PlotBackend.MPL)
        menubar.addAction(self.plot_settings_action)

        plot_backend_menu = QtWidgets.QMenu("Plotting Backend", parent=self)
        self.plotly_action = QAction("Plotly (HTML)", parent=self, checkable=True)
        self.matplotlib_action = QAction("Matplotlib", parent=self, checkable=True)
        self.matplotlib_action.setChecked(self.plot_manager.plot_backend == PlotBackend.MPL)
        self.plotly_action.setChecked(self.plot_manager.plot_backend == PlotBackend.PLOTLY)

        action_group = QActionGroup(self)
        action_group.addAction(self.plotly_action)
        action_group.addAction(self.matplotlib_action)

        plot_backend_menu.addAction(self.plotly_action)
        plot_backend_menu.addAction(self.matplotlib_action)
        menubar.addMenu(plot_backend_menu)

        # Connect signals to methods
        self.plotly_action.triggered.connect(self.set_plotly_backend)
        self.matplotlib_action.triggered.connect(self.set_matplotlib_backend)

        about_action = QAction("About", parent=self)
        about_action.triggered.connect(self.about_window.show)
        menubar.addAction(about_action)

        self._timeline_dialog = TimelineDialog(parent=self)

        theme_menu = QtWidgets.QMenu("Theme", parent=self)
        self._dark_theme_action = QAction("Dark", parent=self, checkable=True)
        self._light_theme_action = QAction("Light", parent=self, checkable=True)
        self._dark_theme_action.setChecked(self._dark_theme)
        self._light_theme_action.setChecked(not self._dark_theme)

        theme_group = QActionGroup(self)
        theme_group.addAction(self._dark_theme_action)
        theme_group.addAction(self._light_theme_action)
        self._dark_theme_action.triggered.connect(lambda: self._apply_theme(dark=True))
        self._light_theme_action.triggered.connect(lambda: self._apply_theme(dark=False))
        theme_menu.addAction(self._dark_theme_action)
        theme_menu.addAction(self._light_theme_action)
        menubar.addMenu(theme_menu)

        # Wire up system theme changes
        app = QtWidgets.QApplication.instance()
        if hasattr(app.styleHints(), "colorSchemeChanged"):
            app.styleHints().colorSchemeChanged.connect(self._on_system_theme_changed)

    @Slot(QtCore.Qt.ColorScheme)
    def _on_system_theme_changed(self, scheme: QtCore.Qt.ColorScheme) -> None:
        """Automatically switch theme when system color scheme changes."""
        self._apply_theme(dark=scheme == QtCore.Qt.ColorScheme.Dark)

    def _apply_theme(self, dark: bool) -> None:
        self._dark_theme = dark
        self._dark_theme_action.setChecked(dark)
        self._light_theme_action.setChecked(not dark)
        QtWidgets.QApplication.instance().setStyleSheet(DARK_STYLESHEET if dark else LIGHT_STYLESHEET)
        set_dark_titlebar(self, dark)
        self._update_preview_colors()
        self._preview_canvas.draw_idle()
        self._timeline_dialog.apply_theme(dark)

    def _update_preview_colors(self) -> None:
        colors = MPL_COLORS["dark" if self._dark_theme else "light"]
        self._preview_fig.patch.set_facecolor(colors["bg"])
        self._preview_ax.set_facecolor(colors["bg"])
        for spine in self._preview_ax.spines.values():
            spine.set_edgecolor(colors["spine"])
        self._preview_ax.tick_params(colors=colors["tick"], labelsize=7)
        self._preview_ax.xaxis.label.set_color(colors["tick"])
        self._preview_ax.yaxis.label.set_color(colors["tick"])

    def set_plotly_backend(self):
        self.plot_manager.set_plot_backend(PlotBackend.PLOTLY)
        self.report_settings_action.setVisible(True)
        self.plot_settings_action.setVisible(False)

    def set_matplotlib_backend(self):
        self.plot_manager.set_plot_backend(PlotBackend.MPL)
        self.report_settings_action.setVisible(False)
        self.plot_settings_action.setVisible(True)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        # Stop persistent Python workers first so no new work starts during shutdown.
        self.trajectory_manager.shutdown_executor()
        self.file_manager.shutdown_executor()

        if self.computation_thread and self.computation_thread.isRunning():
            self.computation_thread.requestInterruption()
            self.computation_thread.quit()
            if not self.computation_thread.wait(5000):
                logger.warning("Computation thread did not stop within timeout; close request aborted.")
                if a0 is not None:
                    a0.ignore()
                self.statusBar().showMessage("Background work still running. Please try closing again.")
                return

        if self.temp_dir:
            shutil.rmtree(self.report_output_path)
            logger.info("Removed temporary report directory %s", self.report_output_path)

        return super().closeEvent(a0)

    def setup_worker_connections(self):
        """This method sets up the signal connections between the worker thread and the main thread"""
        self.setup_trajectory_table_view_connections()
        self.setup_result_table_view_connections()
        self.setup_ui_manager_connections()
        self.setup_session_manager_connections()
        self.setup_file_manager_connections()
        self.setup_plot_manager_connections()
        self.setup_trajectory_manager_connections()
        self.setup_result_table_connections()
        self.setup_trajectory_table_connections()
        self.setup_plottings_connections()

    def setup_plot_manager_connections(self):
        self.plot_manager.ui_request.connect(self.ui_manager.handle_request)

    def setup_trajectory_table_connections(self):
        self.trajectory_table_model.file_request.connect(self.file_manager.handle_request)

    def setup_result_table_connections(self):
        self.result_table_model.ui_request.connect(self.ui_manager.handle_request)
        self.result_table_model.file_request.connect(self.file_manager.handle_request)

    def setup_trajectory_manager_connections(self):
        self.trajectory_manager.ui_request.connect(self.ui_manager.handle_request)
        self.trajectory_manager.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.trajectory_manager.result_model_request.connect(self.result_table_model.handle_request)

    def setup_file_manager_connections(self):
        self.file_manager.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.file_manager.result_model_request.connect(self.result_table_model.handle_request)
        self.file_manager.ui_request.connect(self.ui_manager.handle_request)

    def setup_session_manager_connections(self):
        self.session_manager.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.session_manager.result_model_request.connect(self.result_table_model.handle_request)
        self.session_manager.file_request.connect(self.file_manager.handle_request)
        self.session_manager.ui_request.connect(self.ui_manager.handle_request)
        self.session_manager.report_settings_request.connect(self.handle_report_settings_request)

    def setup_ui_manager_connections(self):
        self.ui_manager.trajectory_manager_request.connect(self.trajectory_manager.handle_request)
        self.ui_manager.file_request.connect(self.file_manager.handle_request)
        self.ui_manager.session_manager_request.connect(self.session_manager.handle_request)
        self.ui_manager.ui_request.connect(self.ui_manager.handle_request)
        self.ui_manager.result_model_request.connect(self.result_table_model.handle_request)

    def setup_result_table_view_connections(self):
        self.resultTableView.result_model_request.connect(self.result_table_model.handle_request)
        self.resultTableView.ui_request.connect(self.ui_manager.handle_request)
        self.resultTableView.file_request.connect(self.file_manager.handle_request)

    def setup_trajectory_table_view_connections(self):
        self.trajectoryTableView.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.trajectoryTableView.trajectory_manager_request.connect(self.trajectory_manager.handle_request)
        self.trajectoryTableView.ui_request.connect(self.ui_manager.handle_request)
        self.trajectoryTableView.file_request.connect(self.file_manager.handle_request)
        self.trajectoryTableView.result_model_request.connect(self.result_table_model.handle_request)
        self.trajectoryTableView.show_timeline.connect(self._show_timeline)

    def setup_progress_connections(self):
        if self.computation_thread is not None:
            # Give cross-thread managers a reference to the progress window
            # so the @show_progress decorator can invoke it directly via
            # QMetaObject.invokeMethod (bypasses signal dispatch).
            self.trajectory_manager._progress_window = self.progress_window
            self.file_manager._progress_window = self.progress_window

            # Status bar only — progress window is handled by the decorator.
            self.trajectory_manager.operation_started.connect(lambda: self.statusBar().showMessage("Processing..."))
            self.trajectory_manager.operation_finished.connect(lambda: self.statusBar().showMessage("Ready"))

            self.file_manager.operation_started.connect(lambda: self.statusBar().showMessage("Loading..."))
            self.file_manager.operation_finished.connect(lambda: self.statusBar().showMessage("Ready"))
        else:
            # Single-thread mode: all on main thread, use signal connections.
            self.trajectory_manager.operation_started.connect(self.progress_window.handle_show_request)
            self.trajectory_manager.operation_started.connect(lambda: self.statusBar().showMessage("Processing..."))
            self.trajectory_manager.operation_finished.connect(self.progress_window.handle_close_request)
            self.trajectory_manager.operation_finished.connect(lambda: self.statusBar().showMessage("Ready"))

            self.file_manager.operation_started.connect(self.progress_window.handle_show_request)
            self.file_manager.operation_started.connect(lambda: self.statusBar().showMessage("Loading..."))
            self.file_manager.operation_finished.connect(self.progress_window.handle_close_request)
            self.file_manager.operation_finished.connect(lambda: self.statusBar().showMessage("Ready"))

        self.session_manager.operation_started.connect(self.progress_window.handle_show_request)
        self.session_manager.operation_started.connect(lambda: self.statusBar().showMessage("Session operation..."))
        self.session_manager.operation_finished.connect(self.progress_window.handle_close_request)
        self.session_manager.operation_finished.connect(lambda: self.statusBar().showMessage("Ready"))

        self.plot_manager.operation_started.connect(self.progress_window.handle_show_request)
        self.plot_manager.operation_started.connect(lambda: self.statusBar().showMessage("Generating plot..."))
        self.plot_manager.operation_finished.connect(self.progress_window.handle_close_request)
        self.plot_manager.operation_finished.connect(lambda: self.statusBar().showMessage("Ready"))

    def setup_plottings_connections(self):
        self.resultTableView.plot_request.connect(self.inject_plot_settings)
        self.trajectoryTableView.plot_request.connect(self.inject_plot_settings)

    def setup_io_connections(self):
        self.trajectory_manager.update_view.connect(self.refresh)
        self.addTrajectoryButton.clicked.connect(
            lambda: self.ui_manager.handle_request(UIRequest(type=UIRequestType.IMPORT_TRAJ))
        )
        self.addResultButton.clicked.connect(
            lambda: self.ui_manager.handle_request(UIRequest(type=UIRequestType.IMPORT_RES))
        )

        # Update details panel on selection change
        self.trajectoryTableView.selectionModel().selectionChanged.connect(self._on_trajectory_selection_changed)
        self.resultTableView.selectionModel().selectionChanged.connect(self._on_result_selection_changed)
        # Also refresh details whenever layout changes (e.g., new items loaded)
        self.trajectory_table_model.layoutChanged.connect(self._on_trajectory_selection_changed)
        self.result_table_model.layoutChanged.connect(self._on_result_selection_changed)
        # Keep timeline in sync with trajectory model
        self.trajectory_table_model.layoutChanged.connect(self._on_trajectory_model_changed)

    @QtCore.Slot()
    def _on_trajectory_selection_changed(self) -> None:
        entries = self.trajectoryTableView.selected_entries
        self._preview_ax.clear()
        if not entries:
            self.details_text.setPlaceholderText("Select a trajectory or result to see details.")
            self.details_text.clear()
            self._preview_ax.set_visible(False)
            self._preview_canvas.draw_idle()
            return

        lines = []
        for entry in entries:
            t = entry.trajectory
            lines.append(f"Name:              {t.name}")
            lines.append(f"Poses:             {len(t)}")
            lines.append(f"Length [m]:        {t.total_length:.3f}")
            lines.append(f"Data rate [Hz]:    {t.data_rate:.3f}")
            lines.append(f"EPSG:              {t.positions.epsg}")
            lines.append(f"Orientation:       {'yes' if t.has_orientation else 'no'}")
            lines.append(f"Function of:       {t.index_label}")
            lines.append(f"Reference:         {'yes' if entry.set_as_reference else 'no'}")
            if len(entries) > 1:
                lines.append("\u2500" * 36)
        self.details_text.setPlainText("\n".join(lines))

        # Draw 2D XY preview
        self._preview_ax.set_visible(True)
        _MAX_PREVIEW_POINTS = 2000
        local_transformer = None
        for entry in entries:
            t = entry.trajectory

            if local_transformer is None and t.positions.local_transformer is not None:
                local_transformer = t.positions.local_transformer

            try:
                n = len(t.positions.xyz)
                if n > _MAX_PREVIEW_POINTS:
                    idx = np.linspace(0, n - 1, _MAX_PREVIEW_POINTS, dtype=int)
                    pos_subset = t.positions.copy()
                    pos_subset.xyz = t.positions.xyz[idx]
                else:
                    pos_subset = t.positions.copy()

                pos_subset.local_transformer = local_transformer
                local_xyz = pos_subset.to_local(inplace=False).xyz
                self._preview_ax.plot(local_xyz[:, 0], local_xyz[:, 1], linewidth=1.0, label=t.name)
            except Exception:
                logger.exception("Preview failed for trajectory '%s'", t.name)
                self._preview_ax.plot(t.positions.xyz[:, 0], t.positions.xyz[:, 1], linewidth=1.0, label=t.name)
        self._preview_ax.set_aspect("equal", adjustable="datalim")
        self._preview_ax.set_xlabel("X [m]")
        self._preview_ax.set_ylabel("Y [m]")
        if len(entries) > 1:
            self._preview_ax.legend(fontsize=7)
        self._update_preview_colors()
        self._preview_canvas.draw_idle()

    @QtCore.Slot()
    def _on_result_selection_changed(self) -> None:
        if (sel := self.resultTableView.selectionModel()) is None or not sel.selectedRows():
            return
        rows = [r.row() for r in sel.selectedRows()]
        items = [self.result_table_model.items[r] for r in rows if r < len(self.result_table_model.items)]
        if not items:
            return

        # Clear the trajectory 2D preview
        self._preview_ax.clear()
        self._preview_ax.set_visible(False)
        self._update_preview_colors()
        self._preview_canvas.draw_idle()

        lines = []
        for item in items:
            prop = getattr(item, "property_dict", {})
            if prop:
                longest_key = max((len(k) for k in prop), default=0)
                for k, v in prop.items():
                    lines.append(f"{k:<{longest_key}}  {v}")
            else:
                lines.append(f"Name:  {item.name}")
                lines.append(f"Type:  {type(item).__name__}")
            if len(items) > 1:
                lines.append("─" * 48)
        self.details_text.setPlainText("\n".join(lines))

    @QtCore.Slot()
    def _on_trajectory_model_changed(self) -> None:
        pass  # timeline is now driven by context menu selection

    @QtCore.Slot(object)
    def _show_timeline(self, entries) -> None:
        self._timeline_dialog.set_entries(entries)
        self._timeline_dialog.show()
        self._timeline_dialog.raise_()

    @QtCore.Slot()
    def refresh(self) -> None:
        self.trajectory_table_model.layoutChanged.emit()
        self.result_table_model.layoutChanged.emit()

    @Slot(PlotSettingsRequest)
    def handle_report_settings_request(self, request: PlotSettingsRequest) -> None:
        """Logic for handling a request."""
        generic_request_handler(self, request, passthrough_request=True)

    def handle_show_report_settings(self) -> None:
        viewer = JSONViewer(settings=self.report_settings, parent=self)
        viewer.show()

    def handle_show_mpl_settings(self) -> None:
        viewer = JSONViewer(settings=self.mpl_plot_settings, parent=self)
        viewer.show()

    def handle_plot_settings_export(self, request: PlotSettingsRequest) -> None:
        self.report_settings.to_file(os.path.join(request.file_path, "report_settings.json"))
        self.mpl_plot_settings.to_file(os.path.join(request.file_path, "mpl_settings.json"))

    def handle_plot_settings_import(self, request: PlotSettingsRequest) -> None:
        try:
            self.report_settings = ReportSettings.from_file(os.path.join(request.file_path, "report_settings.json"))
        except Exception as e:
            logger.warning("Could not load report settings file: %s", e)
            self.report_settings = ReportSettings()

        try:
            self.mpl_plot_settings = MPLPlotSettings.from_file(os.path.join(request.file_path, "mpl_settings.json"))
        except Exception as e:
            logger.warning("Could not load MPL plot settings file: %s", e)
            self.mpl_plot_settings = MPLPlotSettings()

    def handle_plot_settings_reset(self, _: PlotSettingsRequest) -> None:
        self.report_settings = ReportSettings()
        self.mpl_plot_settings = MPLPlotSettings()

    @QtCore.Slot(PlotRequest)
    def inject_plot_settings(self, plot_request: PlotRequest) -> None:
        logger.debug("Injecting plotting settings into plot request of type %s", plot_request.type)
        plot_request.report_settings = self.report_settings
        plot_request.mpl_plot_settings = self.mpl_plot_settings
        self.plot_manager.handle_request(plot_request)

    @QtCore.Slot()
    def handle_import_session(self) -> None:
        self.ui_manager.handle_request(UIRequest(type=UIRequestType.IMPORT_SESSION))

    @QtCore.Slot()
    def handle_export_session(self) -> None:
        self.ui_manager.handle_request(UIRequest(type=UIRequestType.EXPORT_SESSION))

    @QtCore.Slot()
    def handle_new_session(self) -> None:
        self.ui_manager.handle_request(UIRequest(type=UIRequestType.CONFIRM_RESET))

    def center(self):
        center_window(self)

    def setupUi(self):
        """This method sets up the GUI"""
        self.setObjectName("MainWindow")
        self.resize(1150, 700)
        self.center()
        self.setMinimumSize(QtCore.QSize(800, 500))

        # --- Central widget ---
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Horizontal splitter: left (tables) | right (details panel)
        self.h_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self.centralwidget)

        # --- LEFT PANE: vertical splitter with Trajectories + Results ---
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(0)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, left_widget)

        traj_group = QtWidgets.QGroupBox("Trajectories")
        traj_layout = QtWidgets.QVBoxLayout(traj_group)
        traj_layout.setContentsMargins(6, 6, 6, 6)

        self.trajectoryTableView = TrajectoryTableView(
            parent=traj_group,
            trajectory_table_model=self.trajectory_table_model,
        )
        header = self.trajectoryTableView.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        header.resizeSections(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        traj_layout.addWidget(self.trajectoryTableView)

        traj_button_layout = QtWidgets.QHBoxLayout()
        traj_button_layout.addStretch()
        self.addTrajectoryButton = QtWidgets.QPushButton("Add")
        self.addTrajectoryButton.setObjectName("addTrajectoryButton")
        traj_button_layout.addWidget(self.addTrajectoryButton)
        traj_layout.addLayout(traj_button_layout)
        self.splitter.addWidget(traj_group)

        result_group = QtWidgets.QGroupBox("Results")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        result_layout.setContentsMargins(6, 6, 6, 6)

        self.resultTableView = ResultTableView(parent=result_group, result_table_model=self.result_table_model)
        result_layout.addWidget(self.resultTableView)

        result_button_layout = QtWidgets.QHBoxLayout()
        result_button_layout.addStretch()
        self.addResultButton = QtWidgets.QPushButton("Add")
        self.addResultButton.setObjectName("addResultButton")
        result_button_layout.addWidget(self.addResultButton)
        result_layout.addLayout(result_button_layout)
        self.splitter.addWidget(result_group)

        left_layout.addWidget(self.splitter)
        self.h_splitter.addWidget(left_widget)

        # --- RIGHT PANE: details panel (text + 2D preview) ---
        details_widget = QtWidgets.QWidget()
        details_widget.setMinimumWidth(200)
        details_layout = QtWidgets.QVBoxLayout(details_widget)
        details_layout.setContentsMargins(8, 0, 0, 0)
        details_layout.setSpacing(4)

        details_header = QtWidgets.QLabel("Details")
        details_header.setStyleSheet("color: {muted}; padding: 4px;".format(muted="#888888"))
        details_layout.addWidget(details_header)

        details_v_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        self.details_text = QtWidgets.QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setPlaceholderText("Select a trajectory or result to see details.")
        details_v_splitter.addWidget(self.details_text)

        # Matplotlib 2D preview canvas
        self._preview_fig = Figure(figsize=(4, 3), tight_layout=True, facecolor="#1e1e1e")
        self._preview_ax = self._preview_fig.add_subplot(111)
        self._preview_ax.set_visible(False)
        self._preview_canvas = FigureCanvas(self._preview_fig)
        self._preview_canvas.setMinimumHeight(150)
        details_v_splitter.addWidget(self._preview_canvas)
        details_v_splitter.setStretchFactor(0, 1)
        details_v_splitter.setStretchFactor(1, 2)

        details_layout.addWidget(details_v_splitter)
        self.h_splitter.addWidget(details_widget)
        self.h_splitter.setStretchFactor(0, 3)
        self.h_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.h_splitter)
        self.setCentralWidget(self.centralwidget)
        self.statusBar().showMessage("Ready")
        self.statusBar().setSizeGripEnabled(False)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Trajectopy"))
