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
from trajectopy.gui.utils import app_settings, center_window, set_dark_titlebar
from trajectopy.gui.views.about_window import AboutGUI
from trajectopy.gui.views.json_settings_view import JSONViewer
from trajectopy.gui.views.preview_plot import PreviewPlot
from trajectopy.gui.views.progress_window import ProgressWindow
from trajectopy.gui.views.properties_panel import PropertiesPanel
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


def _count(number: int, singular: str, plural: str) -> str:
    return f"{number} {singular if number == 1 else plural}"


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
        self._settings = app_settings()
        self._preview_enabled = bool(self._settings.value(PREVIEW_SETTINGS_KEY, True, type=bool))
        # handler of the selection that the properties and the preview currently show
        self._show_selection: Callable[[], None] = self._on_trajectory_selection_changed
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

        # outside of the preview panel: stays reachable while the preview is hidden
        # (kept as attribute: PySide does not pass the ownership of corner widgets to the menu bar)
        self._preview_corner = QtWidgets.QWidget()
        self._preview_corner.setObjectName("previewToggleCorner")  # transparent on the menu bar (themes)
        toggle_layout = QtWidgets.QHBoxLayout(self._preview_corner)
        toggle_layout.setContentsMargins(0, 0, 10, 0)
        toggle_layout.addWidget(self._preview_toggle, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        # as high as the menu bar, so that the check box is vertically centered with the menu entries
        self._preview_corner.setFixedHeight(menubar.sizeHint().height())
        menubar.setCornerWidget(self._preview_corner, QtCore.Qt.Corner.TopRightCorner)

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
        self._preview_panel.setVisible(enabled)
        if enabled:
            self._show_selection()
        else:
            self._preview.clear_preview()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if not self._confirm_quit():
            if a0 is not None:
                a0.ignore()
            return

        self.statusBar().showMessage("Finishing the running operation …")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            if not self.worker.shutdown(timeout=5.0):
                # cannot be interrupted; the process exits as soon as it is done
                logger.warning("Background operation still running; the application exits once it has finished.")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._preview.dispose()
        if a0 is not None:
            super().closeEvent(a0)

    def _confirm_quit(self) -> bool:
        """Asks before discarding a session with entries or a running operation. Returns True to quit."""
        trajectories, results = self.trajectory_table_model.rowCount(), self.result_table_model.rowCount()
        busy = self.worker.is_busy
        if not (trajectories or results or busy):
            return True

        lines = []
        if trajectories or results:
            lines.append(
                f"The current session ({_count(trajectories, 'trajectory', 'trajectories')}, "
                f"{_count(results, 'result', 'results')}) will be lost. "
                "Use Session → Save to keep it."
            )
        if busy:
            lines.append("An operation is still running; its result will be discarded.")
        box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Question, "Quit Trajectopy", "Quit Trajectopy?", parent=self
        )
        box.setInformativeText("\n\n".join(lines))
        quit_button = box.addButton("Quit", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.exec()
        return box.clickedButton() is quit_button

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
        self.ui_manager.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
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
        status_messages: dict[TrajectoryManager | FileManager | SessionManager, str] = {
            self.trajectory_manager: "Processing...",
            self.file_manager: "Loading...",
            self.session_manager: "Session operation...",
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
        # Also refresh details when the shown entries change (processed, renamed, removed, ...); coalesced, so that
        # removing many rows redraws the preview only once
        self._selection_refresh = QtCore.QTimer(self)
        self._selection_refresh.setSingleShot(True)
        self._selection_refresh.setInterval(0)
        self._selection_refresh.timeout.connect(lambda: self._show_selection())
        for model in (self.trajectory_table_model, self.result_table_model):
            for signal in (model.dataChanged, model.rowsRemoved, model.modelReset, model.layoutChanged):
                signal.connect(lambda *_: self._selection_refresh.start())

    @QtCore.Slot()
    def _on_trajectory_selection_changed(self) -> None:
        self._show_selection = self._on_trajectory_selection_changed
        entries = self.trajectoryTableView.selected_entries
        self._properties.show_entries(entries)
        if entries and self._preview_enabled:
            self._preview.show_trajectories([entry.trajectory for entry in entries])
        else:
            self._preview.clear_preview()

    @QtCore.Slot()
    def _on_result_selection_changed(self) -> None:
        self._show_selection = self._on_result_selection_changed
        items = self.resultTableView.selected_entries
        self._properties.show_entries(items)
        ate_results = [item.deviations for item in items if isinstance(item, AbsoluteDeviationEntry)]
        if ate_results and self._preview_enabled:
            self._preview.show_ate_results(ate_results)
        else:
            self._preview.clear_preview()

    @QtCore.Slot(object)
    def _show_timeline(self, entries) -> None:
        self._timeline_dialog.set_entries(entries)
        self._timeline_dialog.show()
        self._timeline_dialog.raise_()

    @QtCore.Slot()
    def refresh(self) -> None:
        self.trajectory_table_model.refresh()
        self.result_table_model.refresh()

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
        # large enough for the three stacked tables and a useful preview, but never larger than the screen
        available = self.screen().availableGeometry()
        self.resize(min(1600, int(available.width() * 0.9)), min(1000, int(available.height() * 0.9)))
        self.center()
        self.setMinimumSize(QtCore.QSize(800, 500))

        self.trajectoryTableView = TrajectoryTableView(parent=None, trajectory_table_model=self.trajectory_table_model)
        self.addTrajectoryButton = QtWidgets.QPushButton("Add")
        self.addTrajectoryButton.setObjectName("addTrajectoryButton")
        self.addTrajectoryButton.setToolTip("Loads trajectory files (or drop them onto the table).")

        self.resultTableView = ResultTableView(parent=None, result_table_model=self.result_table_model)
        self.trajectoryTableView.trajectory_context_menu.alignments_available = lambda: bool(
            self.result_table_model.alignment_entries
        )
        self.addResultButton = QtWidgets.QPushButton("Add")
        self.addResultButton.setObjectName("addResultButton")
        self.addResultButton.setToolTip("Loads result files (or drop them onto the table).")

        self._properties = PropertiesPanel()

        # left: trajectories, results and the properties of the selection
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter.addWidget(self._group("Trajectories", self.trajectoryTableView, self.addTrajectoryButton))
        self.splitter.addWidget(self._group("Results", self.resultTableView, self.addResultButton))
        self.splitter.addWidget(self._group("Properties", self._properties))
        height = self.height()
        self.splitter.setSizes([int(height * 0.35), int(height * 0.25), int(height * 0.4)])

        # right: preview
        self._preview_toggle = QtWidgets.QCheckBox("Preview")
        self._preview_toggle.setChecked(self._preview_enabled)
        self._preview_toggle.setToolTip("Plots the selected trajectories and ATE results. Disable on slow systems.")
        self._preview_toggle.toggled.connect(self._set_preview_enabled)
        self._preview = PreviewPlot()
        self._preview.setMinimumHeight(150)
        # hidden together with the preview, so that the tables use the whole width
        self._preview_panel = QtWidgets.QWidget()
        self._preview_panel.setMinimumWidth(200)
        preview_layout = QtWidgets.QVBoxLayout(self._preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self._preview)

        self.h_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.h_splitter.addWidget(self.splitter)
        self.h_splitter.addWidget(self._preview_panel)
        # only once it has a parent, otherwise showing it opens a window of its own
        self._preview_panel.setVisible(self._preview_enabled)
        # the preview gets most of the width (also when the window is resized)
        self.h_splitter.setStretchFactor(0, 1)
        self.h_splitter.setStretchFactor(1, 4)
        self.h_splitter.setSizes([int(self.width() * 0.2), int(self.width() * 0.8)])

        central = QtWidgets.QWidget(self)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addWidget(self.h_splitter)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
        self.statusBar().setSizeGripEnabled(False)

        self.retranslateUi()

    @staticmethod
    def _group(
        title: str, widget: QtWidgets.QWidget, add_button: QtWidgets.QPushButton | None = None
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(widget)
        if add_button is not None:
            buttons = QtWidgets.QHBoxLayout()
            buttons.addStretch()
            buttons.addWidget(add_button)
            layout.addLayout(buttons)
        return group

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Trajectopy"))
