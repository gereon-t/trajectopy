"""End-to-end smoke test of the GUI wiring (managers, models, threading) without user interaction.

Runs headless (``QT_QPA_PLATFORM=offscreen``) and is skipped if the GUI extras are not installed.
"""

import os
import time
import unittest
from collections.abc import Callable
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtCore, QtWidgets
except ImportError:  # pragma: no cover - GUI extras not installed
    QtWidgets = None


TRAJECTORY_FILES = ["./test/data/generated_trajectory.traj", "./test/data/noisy_trajectory.traj"]


@unittest.skipIf(QtWidgets is None, "GUI dependencies not installed")
class TestGuiSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if QtWidgets.QApplication.instance() is None:
            # same as trajectopy.gui.app.main
            QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        # keep the window layout and view options of the user's own installation untouched
        cls.app.setOrganizationName("trajectopy-tests")
        cls.app.setApplicationName("trajectopy-tests")

    @classmethod
    def tearDownClass(cls) -> None:
        QtCore.QSettings().clear()

    def setUp(self) -> None:
        # Never block on modal dialogs; record messages instead.
        self.messages: list[str] = []
        for target in (
            "trajectopy.gui.managers.ui_manager.show_msg_box",
            "trajectopy.gui.views.json_settings_view.show_msg_box",
        ):
            patcher = mock.patch(target, side_effect=self.messages.append)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _wait_until(self, condition: Callable[[], bool], timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while not condition():
            self.app.processEvents()
            if time.monotonic() > deadline:
                self.fail(f"Timed out waiting for condition. Messages: {self.messages}")
            time.sleep(0.01)
        self.app.processEvents()

    def _run_workflow(self, single_thread: bool) -> None:
        from trajectopy.gui.managers.requests import (
            FileRequest,
            FileRequestType,
            TrajectoryManagerRequest,
            TrajectoryManagerRequestType,
            TrajectoryModelRequest,
            TrajectoryModelRequestType,
        )
        from trajectopy.gui.models.selection import TrajectorySelection
        from trajectopy.gui.views.main_window import TrajectopyGUI

        window = TrajectopyGUI(single_thread=single_thread)
        try:
            trajectories = window.trajectory_table_model
            results = window.result_table_model

            window.file_manager.handle_request(FileRequest(type=FileRequestType.READ_TRAJ, file_list=TRAJECTORY_FILES))
            self._wait_until(lambda: len(trajectories.items) == 2)

            reference, test = trajectories.items
            trajectories.handle_request(
                TrajectoryModelRequest(
                    type=TrajectoryModelRequestType.SET_REFERENCE,
                    selection=TrajectorySelection(entries=[reference]),
                )
            )
            self.assertIs(trajectories.reference_entry, reference)

            def run(request_type: TrajectoryManagerRequestType) -> None:
                window.trajectory_manager.handle_request(
                    TrajectoryManagerRequest(
                        type=request_type,
                        selection=TrajectorySelection(
                            entries=[trajectories.get(test.entry_id)], reference_entry=trajectories.reference_entry
                        ),
                    )
                )

            run(TrajectoryManagerRequestType.ATE_WITH_ALIGNMENT)
            self._wait_until(lambda: len(results.items) == 2)  # ATE result + alignment result

            run(TrajectoryManagerRequestType.RPE)
            self._wait_until(lambda: len(results.items) == 3)

            run(TrajectoryManagerRequestType.ALIGN)
            self._wait_until(lambda: len(trajectories.items) == 3)

            self._open_plots(window)
            self.assertEqual(self.messages, [])
        finally:
            window.close()
            self.app.processEvents()

    def _open_plots(self, window) -> None:
        """Opens every kind of plot window through the same request path as the views."""
        from trajectopy.gui.managers.requests import PlotRequest, PlotRequestType
        from trajectopy.gui.models.entries import AlignmentEntry
        from trajectopy.gui.models.selection import ResultSelection, TrajectorySelection

        results = window.result_table_model.items
        deviations = [entry for entry in results if not isinstance(entry, AlignmentEntry)]
        alignment = next(entry for entry in results if isinstance(entry, AlignmentEntry))
        requests = [
            PlotRequest(
                type=PlotRequestType.TRAJECTORIES,
                trajectory_selection=TrajectorySelection(entries=window.trajectory_table_model.items),
            ),
            PlotRequest(type=PlotRequestType.DEVIATIONS, result_selection=ResultSelection(entries=deviations)),
            PlotRequest(type=PlotRequestType.ALIGNMENT, result_selection=ResultSelection(entries=[alignment])),
        ]
        for request in requests:
            window.inject_plot_settings(request)
        self.app.processEvents()

        plot_windows = window.plot_manager._windows
        self.assertEqual(len(plot_windows), len(requests))
        for plot_window in plot_windows:
            for index in range(plot_window.tabs.count()):
                plot_window.tabs.setCurrentIndex(index)
                self.assertIsNotNone(plot_window.figure_widget(index))
            plot_window.close()

    def test_playback_window(self):
        """Playback must keep drawing when windows are closed and reopened (shared OpenGL resources)."""
        from trajectopy.gui.views.playback_window import PlaybackWindow
        from trajectopy.io import read_trajectory

        trajectories = [read_trajectory(path) for path in TRAJECTORY_FILES]
        draw_errors: list[str] = []
        patcher = mock.patch("pyqtgraph.debug.printExc", side_effect=lambda *a, **k: draw_errors.append(str(a)))
        patcher.start()
        self.addCleanup(patcher.stop)

        for _ in range(2):
            try:
                playback = PlaybackWindow(trajectories)
            except Exception as error:  # pragma: no cover - depends on the OpenGL support of the test machine
                self.skipTest(f"OpenGL not available: {error}")
            playback.show()
            self.app.processEvents()
            playback._sim_time = playback._max_duration / 2
            playback._update_frame(playback._sim_time)
            playback._toggle_play()
            self._wait_until(lambda playback=playback: playback._sim_time > playback._max_duration / 2, timeout=5.0)
            playback._view.grabFramebuffer()  # forces a paint of the OpenGL scene
            playback.close()
            self.app.processEvents()

        self.assertEqual(draw_errors, [])

    def test_every_request_type_has_a_handler(self):
        from trajectopy.gui.managers import requests
        from trajectopy.gui.views.main_window import TrajectopyGUI

        window = TrajectopyGUI(single_thread=True)
        try:
            handlers = {
                requests.TrajectoryManagerRequestType: window.trajectory_manager,
                requests.FileRequestType: window.file_manager,
                requests.UIRequestType: window.ui_manager,
                requests.SessionManagerRequestType: window.session_manager,
                requests.PlotRequestType: window.plot_manager,
                requests.TrajectoryModelRequestType: window.trajectory_table_model,
                requests.ResultModelRequestType: window.result_table_model,
                requests.PlotSettingsRequestType: window,
            }
            for request_enum, handler in handlers.items():
                with self.subTest(request_type=request_enum.__name__):
                    missing = [member.name for member in request_enum if member not in handler.REQUEST_MAPPING]
                    self.assertEqual(missing, [])
        finally:
            window.close()

    def test_settings_editor_roundtrip(self):
        from trajectopy.core.settings import PlotSettings, ProcessingSettings
        from trajectopy.gui.views.json_settings_view import JSONViewer

        for settings in (ProcessingSettings(), PlotSettings()):
            with self.subTest(settings=type(settings).__name__):
                expected = type(settings).from_dict(settings.to_dict())
                viewer = JSONViewer(parent=None, settings=settings)
                viewer.apply_changes()
                self.assertEqual(settings, expected)
                self.assertEqual(self.messages, [])
                viewer.close()

    def test_evaluate_all_and_views(self):
        """ "Evaluate All" fills the comparison and every view type can show the results."""
        from trajectopy.gui.managers.requests import FileRequest, FileRequestType
        from trajectopy.gui.models.entries import AbsoluteDeviationEntry, RelativeDeviationEntry
        from trajectopy.gui.views.main_window import TrajectopyGUI
        from trajectopy.gui.views.workspace import VIEW_TYPES, FigureContent

        window = TrajectopyGUI(single_thread=True)
        try:
            trajectories, results = window.trajectory_table_model, window.result_table_model
            window.file_manager.handle_request(FileRequest(type=FileRequestType.READ_TRAJ, file_list=TRAJECTORY_FILES))
            self._wait_until(lambda: len(trajectories.items) == 2)
            reference, test = trajectories.items

            window._on_reference_chosen(window._reference_box.findData(reference.entry_id))
            self.assertIs(trajectories.reference_entry, reference)
            window.evaluate_all()
            self._wait_until(lambda: any(isinstance(r, RelativeDeviationEntry) for r in results.items))
            ate = next(r for r in results.items if isinstance(r, AbsoluteDeviationEntry))
            self.assertEqual(ate.source_id, test.entry_id)
            window._refresh()
            self.assertEqual(trajectories.badges[reference.entry_id], "reference")
            self.assertTrue(trajectories.badges[test.entry_id].endswith(" m"))

            for view_type in VIEW_TYPES:
                with self.subTest(view=view_type.key):
                    dock = window.workspace.add_view(view_type.key)
                    if isinstance(dock.content, FigureContent):
                        self.assertIsNotNone(dock.content._widget)
                    window.workspace.remove_view(dock)
            self.assertEqual(self.messages, [])
        finally:
            window.close()
            self.app.processEvents()

    def test_layouts_are_saved_and_restored(self):
        from trajectopy.gui.views.main_window import TrajectopyGUI

        window = TrajectopyGUI(single_thread=True)
        try:
            window._load_layout("Trajectory check")
            window.workspace.add_view("ate_histograms")
            self.assertTrue(window._layout_modified)
            with mock.patch("PySide6.QtWidgets.QInputDialog.getText", return_value=("My layout", True)):
                window._save_layout()  # a built-in layout is saved as a copy under a new name
            self.assertEqual(window._layout_name, "My layout")
            self.assertFalse(window._layout_modified)
            saved = [dock.view_type for dock in window.workspace.docks]
        finally:
            window.close()
            self.app.processEvents()

        window = TrajectopyGUI(single_thread=True)
        try:
            self.assertEqual(window._layout_name, "My layout")
            self.assertEqual([dock.view_type for dock in window.workspace.docks], saved)
            window._load_layout("Evaluation")
            self.assertEqual([dock.view_type for dock in window.workspace.docks], ["comparison", "scene", "ate"])
            window._load_layout("My layout")
            self.assertEqual([dock.view_type for dock in window.workspace.docks], saved)
        finally:
            window.close()
            self.app.processEvents()

    def test_divide_into_laps(self):
        from trajectopy.gui.managers.requests import TrajectoryManagerRequest, TrajectoryManagerRequestType
        from trajectopy.gui.models.entries import TrajectoryEntry
        from trajectopy.gui.models.selection import TrajectorySelection
        from trajectopy.gui.views.main_window import TrajectopyGUI
        from trajectopy.io import read_trajectory
        from trajectopy.processing.sorting import sort_spatially

        window = TrajectopyGUI(single_thread=True)
        try:
            # Two laps of the generated (closed loop) trajectory, sorted by path length.
            lap = read_trajectory("./test/data/generated_trajectory.traj")
            sorted_traj = sort_spatially(lap, inplace=False)
            entry = TrajectoryEntry(full_filename="", trajectory=sorted_traj)

            window.trajectory_manager.handle_request(
                TrajectoryManagerRequest(
                    type=TrajectoryManagerRequestType.DIVIDE_INTO_LAPS,
                    selection=TrajectorySelection(entries=[entry]),
                )
            )
            self._wait_until(lambda: len(window.trajectory_table_model.items) >= 1)
            self.assertEqual(self.messages, [])
        finally:
            window.close()

    def test_workflow_multi_threaded(self):
        self._run_workflow(single_thread=False)

    def test_workflow_single_threaded(self):
        self._run_workflow(single_thread=True)


if __name__ == "__main__":
    unittest.main()
