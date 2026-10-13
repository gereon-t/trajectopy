import logging
import os
from collections.abc import Callable
from typing import cast

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent

from trajectopy import __version__ as VERSION
from trajectopy.core.settings import PlotSettings
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
from trajectopy.gui.models.entries import AbsoluteDeviationEntry
from trajectopy.gui.models.result_model import ResultTableModel
from trajectopy.gui.models.trajectory_model import TrajectoryTableModel
from trajectopy.gui.themes import DARK_STYLESHEET, LIGHT_STYLESHEET
from trajectopy.gui.utils import center_window, set_dark_titlebar
from trajectopy.gui.views.about_window import AboutGUI
from trajectopy.gui.views.json_settings_view import JSONViewer
from trajectopy.gui.views.preview_plot import PreviewPlot
from trajectopy.gui.views.progress_window import ProgressWindow
from trajectopy.gui.views.result_table_view import ResultTableView
from trajectopy.gui.views.timeline_widget import TimelineDialog
from trajectopy.gui.views.trajectory_table_view import TrajectoryTableView
from trajectopy.gui.worker import BackgroundWorker
from trajectopy.utils.resources import YEAR
from trajectopy.visualization import style as plot_style

logger = logging.getLogger(__name__)

PREVIEW_SETTINGS_KEY = "details/preview_enabled"

PLOT_SETTINGS_FILENAME = "plot_settings.json"


def _application() -> QtWidgets.QApplication:
    return cast(QtWidgets.QApplication, QtWidgets.QApplication.instance())


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
        plot_settings_path: str = "",
    ) -> None:
        QtWidgets.QMainWindow.__init__(self)

        self.REQUEST_MAPPING: dict[PlotSettingsRequestType, Callable[[PlotSettingsRequest], None]] = {
            PlotSettingsRequestType.EXPORT: self.handle_plot_settings_export,
            PlotSettingsRequestType.IMPORT: self.handle_plot_settings_import,
            PlotSettingsRequestType.RESET: self.handle_plot_settings_reset,
        }

        self._dark_theme = _application().styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
        self._settings = QtCore.QSettings("trajectopy", "trajectopy")
        self._preview_enabled = self._settings.value(PREVIEW_SETTINGS_KEY, True, type=bool)
        # selection handler whose content the details panel currently shows; re-run when the preview is turned on
        self._refresh_details: Callable[[], None] | None = None
        plot_style.set_theme(self._dark_theme)
        self.trajectory_table_model = TrajectoryTableModel()
        self.result_table_model = ResultTableModel()
        self.setupUi()
        # Trajectory and file operations share one worker so that they run strictly one after another.
        self.worker = BackgroundWorker(inline=single_thread)
        logger.info("Multithreading %s", "disabled" if single_thread else "enabled")

        self.trajectory_manager = TrajectoryManager(worker=self.worker)
        self.ui_manager = UIManager(parent=self)
        self.file_manager = FileManager(worker=self.worker)
        self.session_manager = SessionManager()

        self.plot_settings = self.load_plot_settings(plot_settings_path)
        self.plot_manager = PlotManager(parent=self)

        self.about_window = AboutGUI(parent=self, version_str=VERSION, year_str=YEAR)
        self.progress_window = ProgressWindow(parent=self)

        self.ui_request.connect(self.ui_manager.handle_request)
        self.setup_io_connections()
        self.setup_worker_connections()
        self.setup_progress_connections()
        self.setup_menu_bar()

        set_dark_titlebar(self, self._dark_theme)
        self.show()

    @staticmethod
    def load_plot_settings(path: str = "") -> PlotSettings:
        if not path:
            return PlotSettings()
        try:
            settings = PlotSettings.from_file(path)
            logger.info("Loaded plot settings from %s", path)
            return settings
        except Exception as e:
            logger.error("Could not load plot settings file: %s", e)
            return PlotSettings()

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

        plot_settings_action = QAction("Plot Settings", parent=self)
        plot_settings_action.triggered.connect(self.handle_show_plot_settings)
        menubar.addAction(plot_settings_action)

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
        app = _application()
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
        _application().setStyleSheet(DARK_STYLESHEET if dark else LIGHT_STYLESHEET)
        set_dark_titlebar(self, dark)
        plot_style.set_theme(dark)
        self._preview.apply_theme()
        self._timeline_dialog.apply_theme(dark)

    @Slot(bool)
    def _set_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled = enabled
        self._settings.setValue(PREVIEW_SETTINGS_KEY, enabled)
        self._preview.setVisible(enabled)
        if not enabled:
            self._preview.clear_preview()
        elif self._refresh_details is not None:
            self._refresh_details()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if not self.worker.shutdown(timeout=5.0):
            logger.warning("Background operation did not finish within timeout; close request aborted.")
            if a0 is not None:
                a0.ignore()
            self.statusBar().showMessage("Background work still running. Please try closing again.")
            return

        self._preview.dispose()
        if a0 is not None:
            super().closeEvent(a0)

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
        self.session_manager.plot_settings_request.connect(self.handle_plot_settings_request)

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
        status_messages: dict[TrajectoryManager | FileManager | SessionManager | PlotManager, str] = {
            self.trajectory_manager: "Processing...",
            self.file_manager: "Loading...",
            self.session_manager: "Session operation...",
            self.plot_manager: "Generating plot...",
        }
        for manager, message in status_messages.items():
            manager.operation_started.connect(self.progress_window.handle_show_request)
            manager.operation_started.connect(lambda message=message: self.statusBar().showMessage(message))
            manager.operation_finished.connect(self.progress_window.handle_close_request)
            manager.operation_finished.connect(lambda: self.statusBar().showMessage("Ready"))

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

    @QtCore.Slot()
    def _on_trajectory_selection_changed(self) -> None:
        entries = self.trajectoryTableView.selected_entries
        if not entries:
            self.details_text.setPlaceholderText("Select a trajectory or result to see details.")
            self.details_text.clear()
            self._preview.clear_preview()
            return

        self._refresh_details = self._on_trajectory_selection_changed
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

        if self._preview_enabled:
            self._preview.show_trajectories([entry.trajectory for entry in entries])

    @QtCore.Slot()
    def _on_result_selection_changed(self) -> None:
        if (sel := self.resultTableView.selectionModel()) is None or not sel.selectedRows():
            return
        rows = [r.row() for r in sel.selectedRows()]
        items = [self.result_table_model.items[r] for r in rows if r < len(self.result_table_model.items)]
        if not items:
            return

        self._refresh_details = self._on_result_selection_changed
        ate_results = [item.deviations for item in items if isinstance(item, AbsoluteDeviationEntry)]
        if self._preview_enabled and ate_results:
            self._preview.show_ate_results(ate_results)
        else:
            self._preview.clear_preview()

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
    def handle_plot_settings_request(self, request: PlotSettingsRequest) -> None:
        """Logic for handling a request."""
        generic_request_handler(self, request)

    def handle_show_plot_settings(self) -> None:
        viewer = JSONViewer(settings=self.plot_settings, parent=self)
        viewer.show()

    def handle_plot_settings_export(self, request: PlotSettingsRequest) -> None:
        self.plot_settings.to_file(os.path.join(request.file_path, PLOT_SETTINGS_FILENAME))

    def handle_plot_settings_import(self, request: PlotSettingsRequest) -> None:
        path = os.path.join(request.file_path, PLOT_SETTINGS_FILENAME)
        self.plot_settings = self.load_plot_settings(path) if os.path.isfile(path) else PlotSettings()

    def handle_plot_settings_reset(self, _: PlotSettingsRequest) -> None:
        self.plot_settings = PlotSettings()

    @QtCore.Slot(PlotRequest)
    def inject_plot_settings(self, plot_request: PlotRequest) -> None:
        logger.debug("Injecting plotting settings into plot request of type %s", plot_request.type)
        plot_request.plot_settings = self.plot_settings
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
        self.trajectoryTableView.trajectory_context_menu.alignments_available = lambda: bool(
            self.result_table_model.alignment_entries
        )
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
        self._preview_toggle = QtWidgets.QCheckBox("Preview")
        self._preview_toggle.setChecked(self._preview_enabled)
        self._preview_toggle.setToolTip("Plots the selected trajectories and ATE results. Disable on slow systems.")
        self._preview_toggle.toggled.connect(self._set_preview_enabled)
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addWidget(details_header)
        header_layout.addStretch()
        header_layout.addWidget(self._preview_toggle)
        details_layout.addLayout(header_layout)

        details_v_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        self.details_text = QtWidgets.QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setPlaceholderText("Select a trajectory or result to see details.")
        details_v_splitter.addWidget(self.details_text)

        self._preview = PreviewPlot()
        self._preview.setMinimumHeight(150)
        self._preview.setVisible(self._preview_enabled)
        details_v_splitter.addWidget(self._preview)
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

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Trajectopy"))
