import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent

from trajectopy import __version__ as VERSION
from trajectopy.core.settings import PlotSettings
from trajectopy.gui.managers.file_manager import FileManager
from trajectopy.gui.managers.plot_manager import PlotManager
from trajectopy.gui.managers.project_manager import ProjectManager
from trajectopy.gui.managers.requests import (
    FileRequest,
    FileRequestType,
    PlotRequest,
    ProjectRequest,
    ProjectRequestType,
    UIRequest,
    UIRequestType,
)
from trajectopy.gui.managers.trajectory_manager import TrajectoryManager
from trajectopy.gui.managers.ui_manager import UIManager
from trajectopy.gui.models.entries import AbsoluteDeviationEntry
from trajectopy.gui.models.result_model import ResultTableModel
from trajectopy.gui.models.trajectory_model import TrajectoryTableModel
from trajectopy.gui.project import PROJECT_FILE_FILTER, PROJECT_SUFFIX, Project
from trajectopy.gui.themes import DARK_STYLESHEET, LIGHT_STYLESHEET
from trajectopy.gui.update_check import Release, UpdateChecker
from trajectopy.gui.utils import app_settings, browse_dir_dialog, center_window, set_dark_titlebar, show_msg_box
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

if TYPE_CHECKING:
    from trajectopy.gui.views.preview_plot_3d import PreviewPlot3D

logger = logging.getLogger(__name__)

PREVIEW_SETTINGS_KEY = "details/preview_enabled"
PREVIEW_3D_SETTINGS_KEY = "details/preview_3d"
PREVIEW_MODES = ("2D", "3D")

RECENT_PROJECTS_KEY = "projects/recent"
MAX_RECENT_PROJECTS = 10

UPDATE_CHECK_SETTINGS_KEY = "updates/check_at_startup"
UPDATE_CHECK_DELAY_MS = 2000  # after start-up, so that the request does not compete with it


def _application() -> QtWidgets.QApplication:
    return cast(QtWidgets.QApplication, QtWidgets.QApplication.instance())


def _same_path(path: str, other: str) -> bool:
    return os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(other))


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

        # the open project file; empty while the project has not been saved yet
        self._project_path = ""
        # changes since start; the project is modified while it differs from the count at the last save or open
        self._change_count = 0
        self._saved_change_count = 0
        self._pending_saves: list[int] = []  # change count of each save that is running or waiting to run
        self._after_save: Callable[[], None] | None = None
        self._closing = False

        self._dark_theme = _application().styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
        self._settings = app_settings()
        self._preview_enabled = bool(self._settings.value(PREVIEW_SETTINGS_KEY, True, type=bool))
        self._preview_3d_enabled = bool(self._settings.value(PREVIEW_3D_SETTINGS_KEY, False, type=bool))
        self._preview_3d: PreviewPlot3D | None = None  # created on first use
        self._check_updates_at_startup = bool(self._settings.value(UPDATE_CHECK_SETTINGS_KEY, True, type=bool))
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
        self.project_manager = ProjectManager(worker=self.worker)

        self.plot_settings = self.load_plot_settings(plot_settings_path)
        self.plot_manager = PlotManager(parent=self)

        self.about_window = AboutGUI(parent=self, version_str=VERSION, year_str=YEAR)
        self.progress_window = ProgressWindow(parent=self)

        self.ui_request.connect(self.ui_manager.handle_request)
        self.setup_io_connections()
        self.setup_worker_connections()
        self.setup_progress_connections()
        self.setup_menu_bar()

        self.setup_update_check()
        self.setup_change_tracking()
        self._update_title()

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
        the file, plot settings and about menu
        """
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        file_menu = QtWidgets.QMenu("File", parent=self)
        self._add_action(file_menu, "New", self.new_project, "Ctrl+N")
        self._add_action(file_menu, "Open…", self.open_project, "Ctrl+O")
        self._recent_menu = QtWidgets.QMenu("Open Recent", parent=self)
        self._recent_menu.setToolTipsVisible(True)
        file_menu.addMenu(self._recent_menu)
        self._update_recent_menu()
        file_menu.addSeparator()
        self._add_action(file_menu, "Save", self.save_project, "Ctrl+S")
        self._add_action(file_menu, "Save As…", self.save_project_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        self._add_action(file_menu, "Import Session Folder…", self.import_session_folder)
        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", self.close, "Ctrl+Q")
        menubar.addMenu(file_menu)

        plot_settings_action = QAction("Plot Settings", parent=self)
        plot_settings_action.triggered.connect(self.handle_show_plot_settings)
        menubar.addAction(plot_settings_action)

        help_menu = QtWidgets.QMenu("Help", parent=self)
        check_updates_action = QAction("Check for Updates…", parent=self)
        check_updates_action.triggered.connect(self.check_for_updates)
        help_menu.addAction(check_updates_action)
        self._check_updates_at_startup_action = QAction("Check for Updates at Startup", parent=self, checkable=True)
        self._check_updates_at_startup_action.setChecked(self._check_updates_at_startup)
        self._check_updates_at_startup_action.toggled.connect(self._set_check_updates_at_startup)
        help_menu.addAction(self._check_updates_at_startup_action)
        help_menu.addSeparator()
        about_action = QAction("About", parent=self)
        about_action.triggered.connect(self.about_window.show)
        help_menu.addAction(about_action)
        menubar.addMenu(help_menu)

        # outside of the preview panel: stays reachable while the preview is hidden
        # (kept as attribute: PySide does not pass the ownership of corner widgets to the menu bar)
        self._preview_corner = QtWidgets.QWidget()
        self._preview_corner.setObjectName("previewToggleCorner")  # transparent on the menu bar (themes)
        toggle_layout = QtWidgets.QHBoxLayout(self._preview_corner)
        toggle_layout.setContentsMargins(0, 0, 10, 0)
        toggle_layout.setSpacing(6)
        toggle_layout.addWidget(self._preview_toggle, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        toggle_layout.addWidget(self._preview_mode, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
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

    def setup_update_check(self) -> None:
        self._update_checker = UpdateChecker(self)
        self._update_checker.update_available.connect(self._show_update_available)
        self._update_checker.up_to_date.connect(self._report_up_to_date)
        self._update_checker.failed.connect(self._report_update_check_failure)
        self._manual_update_check = False

        # stays in the status bar (unlike messages, which are replaced by the next one)
        self._update_label = QtWidgets.QLabel()
        self._update_label.setOpenExternalLinks(True)
        self._update_label.hide()
        self.statusBar().addPermanentWidget(self._update_label)

        if self._check_updates_at_startup:
            QtCore.QTimer.singleShot(UPDATE_CHECK_DELAY_MS, self._update_checker.check)

    @Slot()
    def check_for_updates(self) -> None:
        """Checks for a newer release and reports the result in any case."""
        self._manual_update_check = True
        self._update_checker.check()

    @Slot(bool)
    def _set_check_updates_at_startup(self, enabled: bool) -> None:
        self._check_updates_at_startup = enabled
        self._settings.setValue(UPDATE_CHECK_SETTINGS_KEY, enabled)

    @Slot(object)
    def _show_update_available(self, release: Release) -> None:
        how = (
            "Download the installer from GitHub."
            if getattr(sys, "frozen", False)
            else 'Update with: pip install --upgrade "trajectopy[gui]"'
        )
        self._update_label.setText(f'<a href="{release.url}">Trajectopy {release.version} is available</a>')
        self._update_label.setToolTip(how)
        self._update_label.show()
        if self._manual_update_check:
            self._manual_update_check = False
            box = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Information,
                "Update Available",
                f"Trajectopy {release.version} is available (installed: {VERSION}).",
                parent=self,
            )
            box.setInformativeText(how)
            open_button = box.addButton("Open Download Page", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
            box.exec()
            if box.clickedButton() is open_button:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(release.url))

    @Slot()
    def _report_up_to_date(self) -> None:
        if self._manual_update_check:
            self._manual_update_check = False
            show_msg_box(f"Trajectopy {VERSION} is up to date.", self)

    @Slot(str)
    def _report_update_check_failure(self, reason: str) -> None:
        if self._manual_update_check:  # a failed check at startup (e.g. offline) is not worth a dialog
            self._manual_update_check = False
            show_msg_box(f"Could not check for updates:\n\n{reason}", self, error=True)

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
        if self._preview_3d is not None:
            self._preview_3d.apply_theme()
        self._timeline_dialog.apply_theme(dark)

    @Slot(bool)
    def _set_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled = enabled
        self._settings.setValue(PREVIEW_SETTINGS_KEY, enabled)
        self._preview_panel.setVisible(enabled)
        self._preview_mode.setEnabled(enabled)
        if enabled:
            self._show_selection()
        else:
            self._active_preview.clear_preview()

    @Slot(int)
    def _set_preview_mode(self, index: int) -> None:
        self._active_preview.clear_preview()
        self._preview_3d_enabled = PREVIEW_MODES[index] == "3D"
        self._settings.setValue(PREVIEW_3D_SETTINGS_KEY, self._preview_3d_enabled)
        self._preview_stack.setCurrentWidget(self._active_preview)
        if self._preview_enabled:
            self._show_selection()

    @property
    def _active_preview(self) -> "PreviewPlot | PreviewPlot3D":
        if not self._preview_3d_enabled:
            return self._preview
        if self._preview_3d is None:
            # created on first use only: needs OpenGL, which is not available (or slow) on every system
            from trajectopy.gui.views.preview_plot_3d import PreviewPlot3D

            self._preview_3d = PreviewPlot3D()
            self._preview_stack.addWidget(self._preview_3d)
        return self._preview_3d

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if not self._closing and not self._confirm_quit():
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
        if self._preview_3d is not None:
            self._preview_3d.dispose()
        if a0 is not None:
            super().closeEvent(a0)

    def _confirm_quit(self) -> bool:
        """Offers to save unsaved changes and asks before discarding a running operation. Returns True to quit."""
        if self.is_modified:
            answer = self._ask_save_changes()
            if answer == QtWidgets.QMessageBox.StandardButton.Save:
                self.save_project(then=self._close_after_save)
            return answer == QtWidgets.QMessageBox.StandardButton.Discard

        if not self.worker.is_busy:
            return True

        box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Question, "Quit Trajectopy", "Quit Trajectopy?", parent=self
        )
        box.setInformativeText("An operation is still running; its result will be discarded.")
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
        self.setup_project_manager_connections()
        self.setup_file_manager_connections()
        self.setup_plot_manager_connections()
        self.setup_trajectory_manager_connections()
        self.setup_result_table_connections()
        self.setup_plottings_connections()

    def setup_plot_manager_connections(self):
        self.plot_manager.ui_request.connect(self.ui_manager.handle_request)

    def setup_result_table_connections(self):
        self.result_table_model.ui_request.connect(self.ui_manager.handle_request)

    def setup_trajectory_manager_connections(self):
        self.trajectory_manager.ui_request.connect(self.ui_manager.handle_request)
        self.trajectory_manager.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.trajectory_manager.result_model_request.connect(self.result_table_model.handle_request)

    def setup_file_manager_connections(self):
        self.file_manager.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.file_manager.result_model_request.connect(self.result_table_model.handle_request)
        self.file_manager.ui_request.connect(self.ui_manager.handle_request)

    def setup_project_manager_connections(self):
        self.project_manager.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.project_manager.result_model_request.connect(self.result_table_model.handle_request)
        self.project_manager.ui_request.connect(self.ui_manager.handle_request)
        self.project_manager.plot_settings_loaded.connect(self._set_plot_settings)
        self.project_manager.project_loaded.connect(self._on_project_loaded)
        self.project_manager.project_saved.connect(self._on_project_saved)
        self.project_manager.save_failed.connect(self._on_save_failed)

    def setup_ui_manager_connections(self):
        self.ui_manager.trajectory_manager_request.connect(self.trajectory_manager.handle_request)
        self.ui_manager.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.ui_manager.file_request.connect(self.file_manager.handle_request)
        self.ui_manager.settings_applied.connect(self._mark_modified)
        self.ui_manager.ui_request.connect(self.ui_manager.handle_request)
        self.ui_manager.result_model_request.connect(self.result_table_model.handle_request)

    def setup_result_table_view_connections(self):
        self.resultTableView.result_model_request.connect(self.result_table_model.handle_request)
        self.resultTableView.ui_request.connect(self.ui_manager.handle_request)
        self.resultTableView.file_request.connect(self.file_manager.handle_request)
        self.resultTableView.project_dropped.connect(self.open_project)

    def setup_trajectory_table_view_connections(self):
        self.trajectoryTableView.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.trajectoryTableView.trajectory_manager_request.connect(self.trajectory_manager.handle_request)
        self.trajectoryTableView.ui_request.connect(self.ui_manager.handle_request)
        self.trajectoryTableView.file_request.connect(self.file_manager.handle_request)
        self.trajectoryTableView.project_dropped.connect(self.open_project)
        self.trajectoryTableView.result_model_request.connect(self.result_table_model.handle_request)
        self.trajectoryTableView.show_timeline.connect(self._show_timeline)

    def setup_progress_connections(self):
        status_messages: dict[TrajectoryManager | FileManager | ProjectManager, str] = {
            self.trajectory_manager: "Processing...",
            self.file_manager: "Loading...",
            self.project_manager: "Project operation...",
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
            self._active_preview.show_trajectories([entry.trajectory for entry in entries])
        else:
            self._active_preview.clear_preview()

    @QtCore.Slot()
    def _on_result_selection_changed(self) -> None:
        self._show_selection = self._on_result_selection_changed
        items = self.resultTableView.selected_entries
        self._properties.show_entries(items)
        ate_results = [item.deviations for item in items if isinstance(item, AbsoluteDeviationEntry)]
        if ate_results and self._preview_enabled:
            self._active_preview.show_ate_results(ate_results)
        else:
            self._active_preview.clear_preview()

    @QtCore.Slot(object)
    def _show_timeline(self, entries) -> None:
        self._timeline_dialog.set_entries(entries)
        self._timeline_dialog.show()
        self._timeline_dialog.raise_()

    @QtCore.Slot()
    def refresh(self) -> None:
        self.trajectory_table_model.refresh()
        self.result_table_model.refresh()

    def handle_show_plot_settings(self) -> None:
        viewer = JSONViewer(settings=self.plot_settings, parent=self)
        viewer.applied.connect(self._mark_modified)
        viewer.show()

    @Slot(object)
    def _set_plot_settings(self, plot_settings: PlotSettings) -> None:
        self.plot_settings = plot_settings

    @QtCore.Slot(PlotRequest)
    def inject_plot_settings(self, plot_request: PlotRequest) -> None:
        logger.debug("Injecting plotting settings into plot request of type %s", plot_request.type)
        plot_request.plot_settings = self.plot_settings
        self.plot_manager.handle_request(plot_request)

    def open_files(self, paths: list[str]) -> None:
        """Opens the given project (.tpy) and loads the given trajectory (.traj, .bag, ...) and result (.result) files
        into it (used at start-up)."""
        if missing := [path for path in paths if not os.path.isfile(path)]:
            show_msg_box("File not found:\n" + "\n".join(missing), self, error=True)
        existing = [os.path.abspath(path) for path in paths if path not in missing]
        projects = [path for path in existing if path.lower().endswith(PROJECT_SUFFIX)]
        if projects:
            if len(projects) > 1:
                logger.warning("Only one project can be open at a time; opening %s", projects[0])
            self.open_project(projects[0])
        results = [path for path in existing if path.lower().endswith(".result")]
        trajectories = [path for path in existing if path not in results and path not in projects]
        if trajectories:
            self.file_manager.handle_request(FileRequest(type=FileRequestType.READ_TRAJ, file_list=trajectories))
        if results:
            self.file_manager.handle_request(FileRequest(type=FileRequestType.READ_RES, file_list=results))

    # ----- projects -----

    @property
    def is_modified(self) -> bool:
        """Whether the project has unsaved changes. A new project without entries has none."""
        if not self._project_path and not (
            self.trajectory_table_model.rowCount() or self.result_table_model.rowCount()
        ):
            return False
        return self._change_count != self._saved_change_count

    def setup_change_tracking(self) -> None:
        """Counts every change of the tables; settings edited in place report their changes themselves."""
        for model in (self.trajectory_table_model, self.result_table_model):
            for signal in (
                model.rowsInserted,
                model.rowsRemoved,
                model.dataChanged,
                model.modelReset,
                model.layoutChanged,
            ):
                signal.connect(self._mark_modified)

    def _mark_modified(self, *_) -> None:
        self._change_count += 1
        self._update_title()

    def _update_title(self) -> None:
        name = Path(self._project_path).stem if self._project_path else "Untitled"
        self.setWindowTitle(f"{name}[*] - Trajectopy")
        self.setWindowModified(self.is_modified)

    def new_project(self) -> None:
        self._maybe_save(lambda: self._project_request(ProjectRequestType.NEW))

    def open_project(self, path: str = "") -> None:
        """Opens a project file; asks for one if no path is given."""
        if not path and not (path := self._ask_open_path()):
            return
        if not os.path.isfile(path):
            self._remove_recent_project(path)
            show_msg_box(f"Project not found:\n{path}", self, error=True)
            return
        path = os.path.abspath(path)
        self._maybe_save(lambda: self._project_request(ProjectRequestType.OPEN, path))

    def import_session_folder(self) -> None:
        """Opens a session folder saved by earlier versions of Trajectopy as a new (unsaved) project."""
        if directory := browse_dir_dialog(self):
            self._maybe_save(lambda: self._project_request(ProjectRequestType.IMPORT_SESSION_FOLDER, directory))

    def save_project(self, then: Callable[[], None] | None = None) -> None:
        """Saves the project to its file, or asks for a file if it has none yet. ``then`` runs once it is saved."""
        if self._project_path:
            self._save(self._project_path, then)
        else:
            self.save_project_as(then)

    def save_project_as(self, then: Callable[[], None] | None = None) -> None:
        if path := self._ask_save_path():
            self._save(path, then)

    def _save(self, path: str, then: Callable[[], None] | None) -> None:
        self._after_save = then
        self._pending_saves.append(self._change_count)
        project = Project(
            trajectories=list(self.trajectory_table_model.items),
            results=list(self.result_table_model.items),
            plot_settings=self.plot_settings,
        )
        self._project_request(ProjectRequestType.SAVE, path, project)

    def _project_request(
        self, request_type: ProjectRequestType, path: str = "", project: Project | None = None
    ) -> None:
        self.project_manager.handle_request(
            ProjectRequest(type=request_type, file_path=path, project=project or Project())
        )

    @Slot(str)
    def _on_project_loaded(self, path: str) -> None:
        self._project_path = path
        self._saved_change_count = self._change_count
        if path:
            self._add_recent_project(path)
        self._update_title()

    @Slot(str)
    def _on_project_saved(self, path: str) -> None:
        # changes made while the project was being saved remain unsaved
        self._saved_change_count = self._pending_saves.pop(0)
        self._project_path = path
        self._add_recent_project(path)
        self._update_title()
        if not self._pending_saves and (then := self._after_save) is not None:
            self._after_save = None
            then()

    @Slot()
    def _on_save_failed(self) -> None:
        self._pending_saves.pop(0)
        self._after_save = None

    def _close_after_save(self) -> None:
        self._closing = True
        # not within the close event that started the save (a save without worker thread finishes inside it)
        QtCore.QTimer.singleShot(0, self.close)

    def _maybe_save(self, then: Callable[[], None]) -> None:
        """Runs ``then`` once unsaved changes have been saved or discarded; not at all if the user cancels."""
        if not self.is_modified:
            then()
            return

        answer = self._ask_save_changes()
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            self.save_project(then=then)
        elif answer == QtWidgets.QMessageBox.StandardButton.Discard:
            then()

    def _ask_save_changes(self) -> QtWidgets.QMessageBox.StandardButton:
        name = Path(self._project_path).name if self._project_path else "Untitled"
        box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Question, "Trajectopy", f'Save the changes to "{name}"?', parent=self
        )
        box.setInformativeText("Your changes will be lost if you don't save them.")
        buttons = QtWidgets.QMessageBox.StandardButton
        box.setStandardButtons(buttons.Save | buttons.Discard | buttons.Cancel)
        box.setDefaultButton(buttons.Save)
        box.button(buttons.Discard).setText("Don't Save")
        return buttons(box.exec())

    def _ask_open_path(self) -> str:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Project", self._dialog_directory(), PROJECT_FILE_FILTER
        )
        return path

    def _ask_save_path(self) -> str:
        suggestion = self._project_path or os.path.join(self._dialog_directory(), f"Untitled{PROJECT_SUFFIX}")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Project As", suggestion, PROJECT_FILE_FILTER)
        if path and not path.lower().endswith(PROJECT_SUFFIX):
            path += PROJECT_SUFFIX
        return path

    def _dialog_directory(self) -> str:
        """Folder of the open project or of the most recent one."""
        recent = self._recent_projects()
        return os.path.dirname(self._project_path or (recent[0] if recent else ""))

    def _recent_projects(self) -> list[str]:
        value = self._settings.value(RECENT_PROJECTS_KEY, [])
        # depending on the platform, QSettings returns a list with one entry as plain string
        if isinstance(value, str):
            return [value]
        return [str(path) for path in value] if isinstance(value, list) else []

    def _set_recent_projects(self, paths: list[str]) -> None:
        self._settings.setValue(RECENT_PROJECTS_KEY, paths[:MAX_RECENT_PROJECTS])
        self._update_recent_menu()

    def _add_recent_project(self, path: str) -> None:
        path = os.path.abspath(path)
        self._set_recent_projects([path, *(other for other in self._recent_projects() if not _same_path(other, path))])

    def _remove_recent_project(self, path: str) -> None:
        self._set_recent_projects([other for other in self._recent_projects() if not _same_path(other, path)])

    def _update_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = self._recent_projects()
        for number, path in enumerate(recent, start=1):
            name = os.path.basename(path).replace("&", "&&")
            action = self._recent_menu.addAction(f"&{number} {name}" if number < 10 else name)
            action.setToolTip(path)
            action.setStatusTip(path)
            action.triggered.connect(lambda _=False, path=path: self.open_project(path))
        if recent:
            self._recent_menu.addSeparator()
            self._recent_menu.addAction("Clear Recent Projects").triggered.connect(
                lambda: self._set_recent_projects([])
            )
        else:
            self._recent_menu.addAction("No Recent Projects").setEnabled(False)

    def _add_action(self, menu: QtWidgets.QMenu, text: str, slot: Callable[[], object], shortcut: str = "") -> QAction:
        action = QAction(text, parent=self)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.triggered.connect(lambda: slot())  # without the checked state as argument
        menu.addAction(action)
        return action

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
        self._preview_mode = QtWidgets.QComboBox()
        self._preview_mode.addItems(PREVIEW_MODES)
        self._preview_mode.setToolTip("Top view (2D) or orbitable 3D view (OpenGL)")
        self._preview_mode.setEnabled(self._preview_enabled)
        self._preview = PreviewPlot()
        self._preview_stack = QtWidgets.QStackedWidget()
        self._preview_stack.setMinimumHeight(150)
        self._preview_stack.addWidget(self._preview)
        # hidden together with the preview, so that the tables use the whole width
        self._preview_panel = QtWidgets.QWidget()
        self._preview_panel.setMinimumWidth(200)
        preview_layout = QtWidgets.QVBoxLayout(self._preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self._preview_stack)
        self._preview_mode.setCurrentIndex(PREVIEW_MODES.index("3D" if self._preview_3d_enabled else "2D"))
        self._preview_stack.setCurrentWidget(self._active_preview)
        self._preview_mode.currentIndexChanged.connect(self._set_preview_mode)

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
