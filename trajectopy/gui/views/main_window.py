import html
import json
import logging
import os
from collections.abc import Callable
from typing import cast

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent

from trajectopy import __version__ as VERSION
from trajectopy.core.settings import PlotSettings
from trajectopy.gui import themes
from trajectopy.gui.managers.file_manager import FileManager
from trajectopy.gui.managers.plot_manager import PlotManager
from trajectopy.gui.managers.requests import (
    PlotRequest,
    PlotSettingsRequest,
    PlotSettingsRequestType,
    TrajectoryManagerRequest,
    TrajectoryManagerRequestType,
    TrajectoryModelRequest,
    TrajectoryModelRequestType,
    UIRequest,
    UIRequestType,
    generic_request_handler,
)
from trajectopy.gui.managers.session_manager import SessionManager
from trajectopy.gui.managers.trajectory_manager import TrajectoryManager
from trajectopy.gui.managers.ui_manager import UIManager
from trajectopy.gui.models import scalar_fields
from trajectopy.gui.models.entries import (
    AbsoluteDeviationEntry,
    AlignmentEntry,
    RelativeDeviationEntry,
    ResultEntry,
    TrajectoryEntry,
)
from trajectopy.gui.models.result_model import ResultTableModel
from trajectopy.gui.models.selection import TrajectorySelection
from trajectopy.gui.models.trajectory_model import TrajectoryTableModel
from trajectopy.gui.themes import DARK_STYLESHEET, LIGHT_STYLESHEET
from trajectopy.gui.utils import center_window, set_dark_titlebar
from trajectopy.gui.views.about_window import AboutGUI
from trajectopy.gui.views.json_settings_view import JSONViewer
from trajectopy.gui.views.progress_window import ProgressWindow
from trajectopy.gui.views.properties_panel import PropertiesPanel
from trajectopy.gui.views.result_table_view import ResultTableView
from trajectopy.gui.views.scene_view import SceneTrajectory
from trajectopy.gui.views.timeline_widget import TimelineDialog
from trajectopy.gui.views.trajectory_table_view import TrajectoryTableView
from trajectopy.gui.views.workspace import (
    BUILT_IN_LAYOUTS,
    DEFAULT_LAYOUT,
    VIEW_TYPES,
    Workspace,
    WorkspaceData,
    WorkspaceTrajectory,
)
from trajectopy.gui.worker import BackgroundWorker
from trajectopy.utils.resources import YEAR
from trajectopy.visualization import style as plot_style

logger = logging.getLogger(__name__)

PLOT_SETTINGS_FILENAME = "plot_settings.json"

GEOMETRY_KEY = "window/geometry"
SPLITTER_KEY = "window/splitter"
LAYOUTS_KEY = "workspace/layouts"  # JSON: layout name -> layout
CURRENT_LAYOUT_KEY = "workspace/current"  # JSON: {"name", "layout", "modified"}
ALIGN_KEY = "evaluation/align"

REFRESH_DELAY_MS = 30  # changes arriving together (e.g. several results) update the views once


def _application() -> QtWidgets.QApplication:
    return cast(QtWidgets.QApplication, QtWidgets.QApplication.instance())


class TrajectopyGUI(QtWidgets.QMainWindow):
    """
    Main window of the GUI.

    The list of trajectories and the selected trajectory are on the left; the workspace with the views the user
    arranged (comparison table, 3D view, plots) fills the rest. The trajectories' context menus contain all
    operations; "Evaluate All" compares every trajectory with the reference in one step.
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
        themes.set_dark(self._dark_theme)
        self._settings = QtCore.QSettings()  # organization and application name are set by the app
        self._color_modes: dict[str, str] = {}  # trajectory entry id -> scalar field shown in the 3D view
        self._known_results: set[str] = set()
        self._layout_name = DEFAULT_LAYOUT
        self._layout_modified = False
        self._plot_settings_version = 0
        plot_style.set_theme(self._dark_theme)

        self.trajectory_table_model = TrajectoryTableModel()
        self.result_table_model = ResultTableModel()
        self.plot_settings = self.load_plot_settings(plot_settings_path)

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(REFRESH_DELAY_MS)
        self._refresh_timer.timeout.connect(self._refresh)

        self.setupUi()
        # Trajectory and file operations share one worker so that they run strictly one after another.
        self.worker = BackgroundWorker(inline=single_thread)
        logger.info("Multithreading %s", "disabled" if single_thread else "enabled")

        self.trajectory_manager = TrajectoryManager(worker=self.worker)
        self.ui_manager = UIManager(parent=self)
        self.file_manager = FileManager(worker=self.worker)
        self.session_manager = SessionManager()
        self.plot_manager = PlotManager(parent=self)

        self.about_window = AboutGUI(parent=self, version_str=VERSION, year_str=YEAR)
        self.progress_window = ProgressWindow(parent=self)

        self.ui_request.connect(self.ui_manager.handle_request)
        self.setup_io_connections()
        self.setup_worker_connections()
        self.setup_progress_connections()
        self.setup_menu_bar()
        self._restore_window()

        set_dark_titlebar(self, self._dark_theme)
        self.show()
        self._refresh()

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

    # ------------------------------------------------------------------------------------------------ menus

    def setup_menu_bar(self):
        """Session, layout, plot settings, theme and about menus."""
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        session_menu = QtWidgets.QMenu("Session", parent=self)
        for text, slot in (
            ("New", self.handle_new_session),
            ("Load", self.handle_import_session),
            ("Save", self.handle_export_session),
        ):
            action = QAction(text, parent=self)
            action.triggered.connect(slot)
            session_menu.addAction(action)
        menubar.addMenu(session_menu)

        layout_menu = QtWidgets.QMenu("Layout", parent=self)
        layout_menu.addAction("Save", self._save_layout)
        layout_menu.addAction("Save As…", lambda: self._save_layout_as())
        layout_menu.addAction("Delete", self._delete_layout)
        layout_menu.addSeparator()
        layout_menu.addAction("Revert to Saved", lambda: self._load_layout(self._layout_name))
        menubar.addMenu(layout_menu)

        plot_settings_action = QAction("Plot Settings", parent=self)
        plot_settings_action.triggered.connect(self.handle_show_plot_settings)
        menubar.addAction(plot_settings_action)

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

        about_action = QAction("About", parent=self)
        about_action.triggered.connect(self.about_window.show)
        menubar.addAction(about_action)

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
        themes.set_dark(dark)
        _application().setStyleSheet(DARK_STYLESHEET if dark else LIGHT_STYLESHEET)
        set_dark_titlebar(self, dark)
        plot_style.set_theme(dark)
        self.workspace.apply_theme()
        self._update_selection_panel()  # the title's color swatch
        self._timeline_dialog.apply_theme(dark)

    # ---------------------------------------------------------------------------------------- window state

    def _restore_window(self) -> None:
        geometry = self._settings.value(GEOMETRY_KEY)
        if geometry is None or not self.restoreGeometry(geometry):
            self.center()
        if (splitter := self._settings.value(SPLITTER_KEY)) is not None:
            self._splitter.restoreState(splitter)

        try:
            current = json.loads(self._settings.value(CURRENT_LAYOUT_KEY, "") or "{}")
        except ValueError:
            current = {}
        name = current.get("name", DEFAULT_LAYOUT)
        if name not in self._layout_names():
            name = DEFAULT_LAYOUT
        if current.get("layout") and self.workspace.load(current["layout"]):
            self._set_layout_name(name, modified=bool(current.get("modified", False)))
        else:
            self._load_layout(name)

    def _save_window(self) -> None:
        self._settings.setValue(GEOMETRY_KEY, self.saveGeometry())
        self._settings.setValue(SPLITTER_KEY, self._splitter.saveState())
        current = {"name": self._layout_name, "layout": self.workspace.to_dict(), "modified": self._layout_modified}
        self._settings.setValue(CURRENT_LAYOUT_KEY, json.dumps(current))

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if not self.worker.shutdown(timeout=5.0):
            logger.warning("Background operation did not finish within timeout; close request aborted.")
            if a0 is not None:
                a0.ignore()
            self.statusBar().showMessage("Background work still running. Please try closing again.")
            return

        self._save_window()
        if a0 is not None:
            super().closeEvent(a0)

    # ------------------------------------------------------------------------------------------ layouts

    def _user_layouts(self) -> dict[str, dict]:
        try:
            layouts = json.loads(self._settings.value(LAYOUTS_KEY, "") or "{}")
        except ValueError:
            logger.warning("Saved layouts are not readable and are ignored.")
            return {}
        return layouts if isinstance(layouts, dict) else {}

    def _store_user_layouts(self, layouts: dict[str, dict]) -> None:
        self._settings.setValue(LAYOUTS_KEY, json.dumps(layouts))

    def _layout_names(self) -> list[str]:
        return [*BUILT_IN_LAYOUTS, *(name for name in self._user_layouts() if name not in BUILT_IN_LAYOUTS)]

    def _update_layout_box(self) -> None:
        self._layout_box.blockSignals(True)
        self._layout_box.clear()
        for name in self._layout_names():
            self._layout_box.addItem(f"{name} (built-in)" if name in BUILT_IN_LAYOUTS else name, name)
        self._layout_box.setCurrentIndex(self._layout_box.findData(self._layout_name))
        self._layout_box.blockSignals(False)

    def _set_layout_name(self, name: str, modified: bool = False) -> None:
        self._layout_name = name
        self._layout_modified = modified
        self._update_layout_box()
        self._modified_label.setVisible(modified)
        self._delete_layout_button.setEnabled(name not in BUILT_IN_LAYOUTS)

    def _load_layout(self, name: str) -> None:
        user_layouts = self._user_layouts()
        if name in user_layouts and self.workspace.load(user_layouts[name]):
            self._set_layout_name(name)
            return
        if name not in BUILT_IN_LAYOUTS:
            logger.warning("Layout '%s' is not available; using '%s'.", name, DEFAULT_LAYOUT)
            name = DEFAULT_LAYOUT
        self.workspace.load_built_in(name)
        self._set_layout_name(name)

    @Slot()
    def _on_layout_chosen(self) -> None:
        name = self._layout_box.currentData()
        if name is not None and (name != self._layout_name or self._layout_modified):
            self._load_layout(name)

    @Slot()
    def _on_workspace_changed(self) -> None:
        if not self._layout_modified:
            self._set_layout_name(self._layout_name, modified=True)

    def _save_layout(self) -> None:
        if self._layout_name in BUILT_IN_LAYOUTS:
            self._save_layout_as(suggestion=f"{self._layout_name} (mine)")
            return
        self._store_layout(self._layout_name)

    def _save_layout_as(self, suggestion: str = "") -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Layout", "Name of the layout:", text=suggestion)
        name = name.strip()
        if not ok or not name:
            return
        if name in BUILT_IN_LAYOUTS:
            QtWidgets.QMessageBox.information(
                self, "Save Layout", f"“{name}” is a built-in layout. Please choose another name."
            )
            return
        if name in self._user_layouts():
            answer = QtWidgets.QMessageBox.question(self, "Save Layout", f"Replace the layout “{name}”?")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._store_layout(name)

    def _store_layout(self, name: str) -> None:
        layouts = self._user_layouts()
        layouts[name] = self.workspace.to_dict()
        self._store_user_layouts(layouts)
        self._set_layout_name(name)
        self.statusBar().showMessage(f"Saved layout “{name}”", 4000)

    def _delete_layout(self) -> None:
        if self._layout_name in BUILT_IN_LAYOUTS:
            return
        answer = QtWidgets.QMessageBox.question(self, "Delete Layout", f"Delete the layout “{self._layout_name}”?")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        layouts = self._user_layouts()
        layouts.pop(self._layout_name, None)
        self._store_user_layouts(layouts)
        self._load_layout(DEFAULT_LAYOUT)

    # ----------------------------------------------------------------------------------------- connections

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
        self._add_trajectories_action.triggered.connect(
            lambda: self.ui_manager.handle_request(UIRequest(type=UIRequestType.IMPORT_TRAJ))
        )
        self._add_results_action.triggered.connect(
            lambda: self.ui_manager.handle_request(UIRequest(type=UIRequestType.IMPORT_RES))
        )

        self.trajectoryTableView.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.resultTableView.selectionModel().selectionChanged.connect(self._schedule_refresh)
        self.result_table_model.layoutChanged.connect(self._color_new_ate_results)
        for model in (self.trajectory_table_model, self.result_table_model):
            model.layoutChanged.connect(self._on_selection_changed)
        self.trajectory_table_model.layoutChanged.connect(self._update_reference_box)
        self.trajectory_table_model.visibility_changed.connect(self._schedule_refresh)
        self.workspace.changed.connect(self._on_workspace_changed)
        self.workspace.trajectory_activated.connect(self._select_trajectory)

    # ----------------------------------------------------------------------------------- selection & data

    @property
    def _selected_trajectories(self) -> list[TrajectoryEntry]:
        return self.trajectoryTableView.selected_entries

    @Slot()
    def _on_selection_changed(self) -> None:
        self._update_selection_panel()
        self._schedule_refresh()

    def _select_trajectory(self, entry_id: str) -> None:
        for row, entry in enumerate(self.trajectory_table_model.items):
            if entry.entry_id == entry_id:
                self.trajectoryTableView.selectRow(row)
                return

    @Slot()
    def _schedule_refresh(self) -> None:
        self._refresh_timer.start()

    def _latest_results(self) -> dict[str, dict[type, ResultEntry]]:
        """Most recent result of each kind per trajectory (entry id)."""
        latest: dict[str, dict[type, ResultEntry]] = {}
        for result in self.result_table_model.items:
            if result.source_id:
                latest.setdefault(result.source_id, {})[type(result)] = result
        return latest

    @Slot()
    def _refresh(self) -> None:
        """Gives the views the current data (checked trajectories, their results, the selection)."""
        trajectories = self.trajectory_table_model
        results = self.result_table_model.items
        latest = self._latest_results()

        checked: list[WorkspaceTrajectory] = []
        scene: list[SceneTrajectory] = []
        badges: dict[str, str] = {}
        for entry in trajectories.items:
            own = latest.get(entry.entry_id, {})
            ate = cast(AbsoluteDeviationEntry | None, own.get(AbsoluteDeviationEntry))
            if entry.set_as_reference:
                badges[entry.entry_id] = "reference"
            elif ate is not None:
                badges[entry.entry_id] = f"{ate.deviations.pos_dev_rms:.3f} m"
            if not trajectories.is_visible(entry):
                continue
            color = trajectories.color(entry)
            checked.append(
                WorkspaceTrajectory(
                    entry=entry,
                    color=color,
                    ate=ate,
                    rpe=cast(RelativeDeviationEntry | None, own.get(RelativeDeviationEntry)),
                    alignment=cast(AlignmentEntry | None, own.get(AlignmentEntry)),
                )
            )
            mode = self._color_modes.get(entry.entry_id, scalar_fields.SOLID)
            scene.append(
                SceneTrajectory(
                    key=entry.entry_id,
                    trajectory=entry.trajectory,
                    color=color,
                    field=scalar_fields.scalar_field(mode, entry, results),
                )
            )
        trajectories.set_badges(badges)
        if not trajectories.items:
            self._color_modes.clear()

        selected = {entry.entry_id for entry in self._selected_trajectories}
        selected |= {r.source_id for r in self.resultTableView.selected_entries if r.source_id}
        self.workspace.set_data(
            WorkspaceData(
                trajectories=checked,
                reference=trajectories.reference_entry,
                selected_ids=frozenset(selected),
                scene=scene,
                plot_settings=self.plot_settings,
                settings_version=self._plot_settings_version,
            )
        )

    @QtCore.Slot()
    def refresh(self) -> None:
        self.trajectory_table_model.layoutChanged.emit()
        self.result_table_model.layoutChanged.emit()

    # ---------------------------------------------------------------------------------- selection panel

    def _update_selection_panel(self) -> None:
        entries = self._selected_trajectories
        if not entries:
            self._selection_title.setText(
                f"<span style='color:{themes.color('muted')}; font-weight:400'>No trajectory selected</span>"
            )
        elif len(entries) == 1:
            swatch = self.trajectory_table_model.color(entries[0])
            self._selection_title.setText(f"<span style='color:{swatch}'>■</span>&nbsp;{html.escape(entries[0].name)}")
        else:
            self._selection_title.setText(f"{len(entries)} trajectories selected")
        self._properties.show_entries(entries)
        self._update_color_by(entries)
        self._filter_results()

    def _update_color_by(self, entries: list[TrajectoryEntry]) -> None:
        """Offers the fields all selected trajectories have; shows the current one if they share it."""
        self._color_by_row.setEnabled(bool(entries))
        self._color_by.clear()
        if not entries:
            return
        results = self.result_table_model.items
        choices = scalar_fields.field_choices(entries[0], results)
        for entry in entries[1:]:
            keys = {key for key, _ in scalar_fields.field_choices(entry, results)}
            choices = [(key, label) for key, label in choices if key in keys]
        for key, label in choices:
            self._color_by.addItem(label, key)
        modes = {self._color_modes.get(entry.entry_id, scalar_fields.SOLID) for entry in entries}
        self._color_by.setCurrentIndex(self._color_by.findData(modes.pop()) if len(modes) == 1 else -1)

    @Slot(int)
    def _on_color_by_chosen(self, index: int) -> None:
        key = self._color_by.itemData(index)
        for entry in self._selected_trajectories:
            self._color_modes[entry.entry_id] = key
        self._schedule_refresh()

    @Slot()
    def _color_new_ate_results(self) -> None:
        """A trajectory shown in its solid color switches to its ATE as soon as one is computed or loaded."""
        for entry in self.result_table_model.items:
            if entry.entry_id in self._known_results:
                continue
            self._known_results.add(entry.entry_id)
            if isinstance(entry, AbsoluteDeviationEntry) and entry.source_id:
                if self._color_modes.get(entry.source_id, scalar_fields.SOLID) == scalar_fields.SOLID:
                    self._color_modes[entry.source_id] = scalar_fields.ATE

    @Slot()
    def _filter_results(self) -> None:
        """Shows the results of the selected trajectories and those whose trajectory is not loaded."""
        loaded = {entry.entry_id for entry in self.trajectory_table_model.items}
        selected = {entry.entry_id for entry in self._selected_trajectories}
        show_all = self._show_all_results.isChecked()
        shown = 0
        for row, result in enumerate(self.result_table_model.items):
            visible = show_all or result.source_id in selected or result.source_id not in loaded
            self.resultTableView.setRowHidden(row, not visible)
            shown += visible
        total = len(self.result_table_model.items)
        self._results_label.setText(f"RESULTS ({shown} of {total})" if shown != total else f"RESULTS ({total})")

    # ------------------------------------------------------------------------------- reference & evaluation

    @Slot()
    def _update_reference_box(self) -> None:
        self._reference_box.blockSignals(True)
        self._reference_box.clear()
        self._reference_box.addItem("(none)", "")
        for entry in self.trajectory_table_model.items:
            self._reference_box.addItem(entry.name, entry.entry_id)
        reference = self.trajectory_table_model.reference_entry
        self._reference_box.setCurrentIndex(self._reference_box.findData(reference.entry_id if reference else ""))
        self._reference_box.blockSignals(False)

    @Slot(int)
    def _on_reference_chosen(self, index: int) -> None:
        entry_id = self._reference_box.itemData(index)
        entry = self.trajectory_table_model.get(entry_id) if entry_id else None
        if entry is None:
            request = TrajectoryModelRequest(type=TrajectoryModelRequestType.UNSET_REFERENCE)
        else:
            request = TrajectoryModelRequest(
                type=TrajectoryModelRequestType.SET_REFERENCE, selection=TrajectorySelection(entries=[entry])
            )
        self.trajectory_table_model.handle_request(request)

    @Slot()
    def evaluate_all(self) -> None:
        """ATE and RPE of every trajectory against the reference."""
        reference = self.trajectory_table_model.reference_entry
        if reference is None:
            self._message("Choose a reference trajectory first (in the toolbar or in its context menu).")
            return
        entries = [entry for entry in self.trajectory_table_model.items if entry is not reference]
        if not entries:
            self._message("There is no trajectory to compare with the reference.")
            return
        ate_type = (
            TrajectoryManagerRequestType.ATE_WITH_ALIGNMENT
            if self._align_action.isChecked()
            else TrajectoryManagerRequestType.ATE_NO_ALIGNMENT
        )
        for request_type in (ate_type, TrajectoryManagerRequestType.RPE):
            self.trajectory_manager.handle_request(
                TrajectoryManagerRequest(
                    type=request_type, selection=TrajectorySelection(entries=entries, reference_entry=reference)
                )
            )

    def _message(self, text: str) -> None:
        self.ui_manager.handle_request(UIRequest(type=UIRequestType.MESSAGE, message=text))

    # ---------------------------------------------------------------------------------------------- misc

    @QtCore.Slot(object)
    def _show_timeline(self, entries) -> None:
        self._timeline_dialog.set_entries(entries)
        self._timeline_dialog.show()
        self._timeline_dialog.raise_()

    @Slot(PlotSettingsRequest)
    def handle_plot_settings_request(self, request: PlotSettingsRequest) -> None:
        """Logic for handling a request."""
        generic_request_handler(self, request)
        self._plot_settings_changed()

    def handle_show_plot_settings(self) -> None:
        viewer = JSONViewer(settings=self.plot_settings, parent=self)
        # connected after the viewer's own handlers, so the views are redrawn with the applied settings
        viewer.apply_button.clicked.connect(self._plot_settings_changed)
        viewer.apply_and_close_button.clicked.connect(self._plot_settings_changed)
        viewer.show()

    def _plot_settings_changed(self) -> None:
        self._plot_settings_version += 1
        self._schedule_refresh()

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

    # ------------------------------------------------------------------------------------------------ UI

    def setupUi(self):
        """This method sets up the GUI"""
        self.setObjectName("MainWindow")
        self.setWindowTitle("Trajectopy")
        self.resize(1440, 900)
        self.setMinimumSize(QtCore.QSize(900, 560))

        self._setup_toolbar()

        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._create_sidebar())
        self._splitter.addWidget(self._create_workspace_panel())
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([340, 1100])
        self.setCentralWidget(self._splitter)

        self.statusBar().showMessage("Ready")
        self.statusBar().setSizeGripEnabled(False)

    def _setup_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Main", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)

        title = QtWidgets.QLabel("trajectopy")
        title.setObjectName("appTitle")
        toolbar.addWidget(title)
        self._add_trajectories_action = toolbar.addAction("Add Trajectories…")
        self._add_trajectories_action.setToolTip("Load trajectory files (or drop them onto the list)")
        self._add_results_action = toolbar.addAction("Add Results…")
        self._add_results_action.setToolTip("Load result files of earlier evaluations")
        for action in (self._add_trajectories_action, self._add_results_action):
            toolbar.widgetForAction(action).setObjectName("outlineButton")
        spacer = QtWidgets.QLabel()  # labels are transparent in the toolbar
        spacer.setFixedWidth(12)
        toolbar.addWidget(spacer)

        toolbar.addWidget(QtWidgets.QLabel("Reference"))
        self._reference_box = QtWidgets.QComboBox()
        self._reference_box.setMinimumContentsLength(18)
        self._reference_box.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._reference_box.setToolTip("The trajectory all others are compared with")
        self._reference_box.activated.connect(self._on_reference_chosen)
        toolbar.addWidget(self._reference_box)
        self._update_reference_box()

        evaluate_button = QtWidgets.QToolButton()
        evaluate_button.setObjectName("primaryButton")
        evaluate_button.setText("Evaluate All")
        evaluate_button.setToolTip("Compute ATE and RPE of every trajectory against the reference")
        evaluate_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        evaluate_button.clicked.connect(self.evaluate_all)
        evaluate_menu = QtWidgets.QMenu(evaluate_button)
        self._align_action = evaluate_menu.addAction("Align to the Reference Before Computing the ATE")
        self._align_action.setCheckable(True)
        self._align_action.setChecked(self._settings.value(ALIGN_KEY, True, type=bool))
        self._align_action.toggled.connect(lambda checked: self._settings.setValue(ALIGN_KEY, checked))
        evaluate_button.setMenu(evaluate_menu)
        toolbar.addWidget(evaluate_button)

    def _create_sidebar(self) -> QtWidgets.QWidget:
        self.trajectoryTableView = TrajectoryTableView(parent=None, trajectory_table_model=self.trajectory_table_model)
        self.resultTableView = ResultTableView(parent=None, result_table_model=self.result_table_model)
        self.trajectoryTableView.trajectory_context_menu.alignments_available = lambda: bool(
            self.result_table_model.alignment_entries
        )
        # the list shows the name (with visibility checkbox and color) and the evaluation summary
        for view, keep in (
            (self.trajectoryTableView, {0, self.trajectory_table_model.badge_column}),
            (self.resultTableView, {0, 1}),
        ):
            view.setShowGrid(False)
            header = view.horizontalHeader()
            header.setStretchLastSection(False)
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
            header.setMinimumSectionSize(40)
            for column in range(view.model().columnCount()):
                view.setColumnHidden(column, column not in keep)

        # sizing to the contents ignores the stylesheet's cell padding and would elide the type
        type_header = self.resultTableView.horizontalHeader()
        type_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        type_header.resizeSection(1, self.resultTableView.fontMetrics().horizontalAdvance("ATE Result") + 32)

        list_panel = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(list_panel)
        list_layout.setContentsMargins(12, 14, 12, 6)
        list_layout.setSpacing(6)
        list_layout.addWidget(_section_label("TRAJECTORIES"))
        list_layout.addWidget(self.trajectoryTableView)

        details = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(details)
        details_layout.setContentsMargins(12, 12, 12, 12)
        details_layout.setSpacing(10)
        self._selection_title = QtWidgets.QLabel("No trajectory selected")
        self._selection_title.setObjectName("selectionTitle")
        self._selection_title.setTextFormat(QtCore.Qt.TextFormat.RichText)
        details_layout.addWidget(self._selection_title)

        self._color_by = QtWidgets.QComboBox()
        self._color_by.setToolTip("Color of the selected trajectories in the 3D view")
        self._color_by.activated.connect(self._on_color_by_chosen)
        self._color_by_row = QtWidgets.QWidget()
        color_by_layout = QtWidgets.QHBoxLayout(self._color_by_row)
        color_by_layout.setContentsMargins(0, 0, 0, 0)
        color_by_label = QtWidgets.QLabel("Color by")
        color_by_label.setObjectName("mutedLabel")
        color_by_layout.addWidget(color_by_label)
        color_by_layout.addWidget(self._color_by, stretch=1)
        self._color_by_row.setEnabled(False)
        details_layout.addWidget(self._color_by_row)

        self._properties = PropertiesPanel()
        details_layout.addWidget(self._properties, stretch=3)

        results_header = QtWidgets.QHBoxLayout()
        self._results_label = _section_label("RESULTS")
        results_header.addWidget(self._results_label)
        results_header.addStretch()
        self._show_all_results = QtWidgets.QCheckBox("Show all")
        self._show_all_results.setToolTip("Show the results of all trajectories, not only of the selected ones")
        self._show_all_results.toggled.connect(self._filter_results)
        results_header.addWidget(self._show_all_results)
        details_layout.addLayout(results_header)
        details_layout.addWidget(self.resultTableView, stretch=2)

        sidebar = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        sidebar.setObjectName("sidebar")
        sidebar.setHandleWidth(1)
        sidebar.addWidget(list_panel)
        sidebar.addWidget(details)
        sidebar.setStretchFactor(0, 2)
        sidebar.setStretchFactor(1, 3)
        sidebar.setMinimumWidth(260)
        return sidebar

    def _create_workspace_panel(self) -> QtWidgets.QWidget:
        self.workspace = Workspace()

        bar_frame = QtWidgets.QFrame()
        bar_frame.setObjectName("layoutBar")
        bar = QtWidgets.QHBoxLayout(bar_frame)
        bar.setContentsMargins(12, 8, 12, 8)
        bar.setSpacing(8)
        layout_label = QtWidgets.QLabel("Layout")
        layout_label.setObjectName("mutedLabel")
        bar.addWidget(layout_label)
        self._layout_box = QtWidgets.QComboBox()
        self._layout_box.setMinimumContentsLength(16)
        self._layout_box.setToolTip("Saved arrangements of views")
        self._layout_box.activated.connect(self._on_layout_chosen)
        bar.addWidget(self._layout_box)
        self._modified_label = QtWidgets.QLabel("modified")
        self._modified_label.setObjectName("warningLabel")
        self._modified_label.setToolTip("The views were changed since the layout was loaded or saved")
        self._modified_label.hide()
        bar.addWidget(self._modified_label)
        for text, tip, slot in (
            ("Save", "Save the views in this layout (a built-in layout is saved as a copy)", self._save_layout),
            ("Save As…", "Save the views as a new layout", lambda: self._save_layout_as()),
        ):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        self._delete_layout_button = QtWidgets.QPushButton("Delete")
        self._delete_layout_button.setToolTip("Delete this layout (built-in layouts cannot be deleted)")
        self._delete_layout_button.clicked.connect(self._delete_layout)
        bar.addWidget(self._delete_layout_button)
        bar.addStretch()

        add_view = QtWidgets.QToolButton()
        add_view.setObjectName("outlineButton")
        add_view.setText("+  Add View")
        add_view.setToolTip("Add a table, the 3D view or a plot to the workspace")
        add_view.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        add_menu = QtWidgets.QMenu(add_view)
        for view_type in VIEW_TYPES:
            add_menu.addAction(view_type.label, lambda key=view_type.key: self.workspace.add_view(key))
        add_view.setMenu(add_menu)
        bar.addWidget(add_view)

        panel = QtWidgets.QWidget()
        panel.setObjectName("workspacePanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(bar_frame)
        layout.addWidget(self.workspace, stretch=1)
        return panel


def _section_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setObjectName("sectionLabel")
    font = label.font()
    font.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, 0.6)
    label.setFont(font)
    return label
