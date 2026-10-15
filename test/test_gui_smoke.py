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

    def setUp(self) -> None:
        # Never touch the user's real GUI settings (window layout): use a temporary settings file.
        import tempfile

        settings_dir = tempfile.TemporaryDirectory()
        self.addCleanup(settings_dir.cleanup)
        settings_file = os.path.join(settings_dir.name, "settings.ini")
        patcher = mock.patch(
            "trajectopy.gui.views.main_window.app_settings",
            side_effect=lambda: QtCore.QSettings(settings_file, QtCore.QSettings.Format.IniFormat),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        # Never block on modal dialogs; record messages instead.
        self.messages: list[str] = []
        for target in (
            "trajectopy.gui.managers.ui_manager.show_msg_box",
            "trajectopy.gui.views.json_settings_view.show_msg_box",
        ):
            patcher = mock.patch(target, side_effect=lambda message, *_, **__: self.messages.append(message))
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch("trajectopy.gui.views.main_window.TrajectopyGUI._confirm_quit", return_value=True)
        self.confirm_quit = patcher.start()
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

    def test_disabled_preview_frees_its_space(self):
        from trajectopy.gui.views.main_window import TrajectopyGUI

        window = TrajectopyGUI(single_thread=True)
        try:
            window.resize(1200, 700)
            self.app.processEvents()
            preview_width = window.h_splitter.sizes()[1]
            self.assertGreater(preview_width, 0)

            window._preview_toggle.setChecked(False)  # the toggle stays reachable in the menu bar
            self.app.processEvents()
            self.assertEqual(window.h_splitter.sizes()[1], 0)

            window._preview_toggle.setChecked(True)
            self.app.processEvents()
            self.assertEqual(window.h_splitter.sizes()[1], preview_width)
        finally:
            window.close()

    def test_properties_panel_shows_the_selection(self):
        from trajectopy.gui.models.entries import TrajectoryEntry
        from trajectopy.gui.views.main_window import TrajectopyGUI
        from trajectopy.io import read_trajectory

        window = TrajectopyGUI(single_thread=True)
        try:
            for file in TRAJECTORY_FILES:
                window.trajectory_table_model.add(
                    TrajectoryEntry(full_filename=file, trajectory=read_trajectory(file))
                )
            window.trajectoryTableView.selectAll()
            self.app.processEvents()
            model = window._properties.model
            names = [entry.name for entry in window.trajectory_table_model.items]
            self.assertEqual(
                [model.headerData(i, QtCore.Qt.Orientation.Horizontal) for i in range(3)], ["Property", *names]
            )
            self.assertIn("Number of Poses", [item.name for item in model.items])

        finally:
            window.close()

    def test_quit_can_be_cancelled(self):
        from trajectopy.gui.views.main_window import TrajectopyGUI

        window = TrajectopyGUI(single_thread=True)
        self.confirm_quit.return_value = False
        window.close()
        self.assertTrue(window.isVisible())
        self.confirm_quit.return_value = True
        window.close()
        self.assertFalse(window.isVisible())

    def test_rename_asks_for_the_name_in_the_main_window(self):
        from trajectopy.gui.managers.requests import UIRequest, UIRequestType
        from trajectopy.gui.models.entries import TrajectoryEntry
        from trajectopy.gui.models.selection import TrajectorySelection
        from trajectopy.gui.views.main_window import TrajectopyGUI
        from trajectopy.io import read_trajectory

        window = TrajectopyGUI(single_thread=True)
        try:
            entry = TrajectoryEntry(full_filename=TRAJECTORY_FILES[0], trajectory=read_trajectory(TRAJECTORY_FILES[0]))
            window.trajectory_table_model.add(entry)
            request = UIRequest(type=UIRequestType.RENAME_TRAJ, trajectory_selection=TrajectorySelection([entry]))
            with mock.patch.object(QtWidgets.QInputDialog, "getText", return_value=("  Renamed  ", True)) as ask:
                window.ui_manager.handle_request(request)
            self.assertIs(ask.call_args.args[0], window)  # modal to the main window, not a free-floating dialog
            self.assertEqual(window.trajectory_table_model.items[0].name, "Renamed")

            with mock.patch.object(QtWidgets.QInputDialog, "getText", return_value=("Other", False)):
                window.ui_manager.handle_request(request)
            self.assertEqual(window.trajectory_table_model.items[0].name, "Renamed")  # cancelled
        finally:
            window.close()

    def test_failed_requests_are_reported_as_errors(self):
        from trajectopy.gui.managers.requests import UIRequest, UIRequestType, generic_request_handler

        class FailingHandler(QtCore.QObject):
            ui_request = QtCore.Signal(UIRequest)

            def __init__(self) -> None:
                super().__init__()
                self.REQUEST_MAPPING = {UIRequestType.PLAYBACK: self.fail}

            def fail(self, _) -> None:
                raise ValueError("no poses")

        handler, reported = FailingHandler(), []
        handler.ui_request.connect(reported.append)
        generic_request_handler(handler, UIRequest(type=UIRequestType.PLAYBACK))
        self.assertTrue(reported[0].error)
        self.assertEqual(reported[0].message, "Playback failed:\n\nno poses")

    def test_workflow_multi_threaded(self):
        self._run_workflow(single_thread=False)

    def test_workflow_single_threaded(self):
        self._run_workflow(single_thread=True)


if __name__ == "__main__":
    unittest.main()
