"""Table models keep the selection on the right entries, and sub-windows are freed and left open independently.

Runs headless (``QT_QPA_PLATFORM=offscreen``) and is skipped if the GUI extras are not installed.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets

    from trajectopy.core.settings import PlotSettings
    from trajectopy.gui.managers.plot_manager import PlotManager
    from trajectopy.gui.managers.requests import (
        PlotRequest,
        PlotRequestType,
        TrajectoryModelRequest,
        TrajectoryModelRequestType,
        UIRequest,
        UIRequestType,
    )
    from trajectopy.gui.managers.ui_manager import UIManager
    from trajectopy.gui.models.entries import TrajectoryEntry
    from trajectopy.gui.models.selection import TrajectorySelection
    from trajectopy.gui.models.trajectory_model import TrajectoryTableModel
    from trajectopy.gui.views.trajectory_table_view import TrajectoryTableView
except ImportError:  # pragma: no cover - GUI extras not installed
    QtWidgets = None

from test.testdata import noisy_trajectory


def _entry(name: str, group_id: str | None = None) -> "TrajectoryEntry":
    trajectory = noisy_trajectory.copy()
    trajectory.name = name
    entry = TrajectoryEntry(full_filename="", trajectory=trajectory)
    if group_id is not None:
        entry.group_id = group_id
    return entry


@unittest.skipIf(QtWidgets is None, "GUI dependencies not installed")
class TestTableModelSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.model = TrajectoryTableModel()
        self.view = TrajectoryTableView(parent=None, trajectory_table_model=self.model)
        self.entries = [_entry(f"T{i}") for i in range(4)]
        for entry in self.entries:
            self.model.add(entry)

    def selected_names(self) -> list[str]:
        return [entry.name for entry in self.view.selected_entries]

    def test_selection_stays_on_entry_when_row_above_is_removed(self):
        self.view.selectRow(1)
        self.model.remove([self.entries[0].entry_id])
        self.assertEqual(self.selected_names(), ["T1"])

    def test_removing_the_selected_entry_clears_the_selection(self):
        self.view.selectRow(2)
        self.model.remove([self.entries[2].entry_id])
        self.assertEqual(self.selected_names(), [])
        self.assertEqual([entry.name for entry in self.model.items], ["T0", "T1", "T3"])

    def test_selection_follows_entry_when_reordered(self):
        self.view.selectRow(1)
        order = [self.entries[i].entry_id for i in (3, 2, 1, 0)]
        self.model.sort_items(TrajectoryModelRequest(type=TrajectoryModelRequestType.SORT, index_list=order))
        self.assertEqual([entry.name for entry in self.model.items], ["T3", "T2", "T1", "T0"])
        self.assertEqual(self.selected_names(), ["T1"])

    def test_entries_missing_in_the_order_are_kept(self):
        order = [self.entries[2].entry_id, self.entries[0].entry_id]
        self.model.sort_items(TrajectoryModelRequest(type=TrajectoryModelRequestType.SORT, index_list=order))
        self.assertEqual([entry.name for entry in self.model.items], ["T2", "T0", "T1", "T3"])

    def test_remove_group_removes_only_that_group(self):
        grouped = [_entry("G0", group_id="group"), _entry("G1", group_id="group")]
        for entry in grouped:
            self.model.add(entry)
        self.view.selectRow(3)
        self.model.remove_group("group")
        self.assertEqual([entry.name for entry in self.model.items], ["T0", "T1", "T2", "T3"])
        self.assertEqual(self.selected_names(), ["T3"])

    def test_reset_clears_the_selection(self):
        self.view.selectRow(0)
        self.model.reset()
        self.assertEqual(self.model.rowCount(), 0)
        self.assertEqual(self.selected_names(), [])


@unittest.skipIf(QtWidgets is None, "GUI dependencies not installed")
class TestSubWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.main_window = QtWidgets.QMainWindow()
        self.addCleanup(self.main_window.deleteLater)
        self.selection = TrajectorySelection(entries=[_entry("T0")])

    def _process_deferred_deletes(self) -> None:
        for _ in range(5):
            self.app.sendPostedEvents(None, 0)  # 0: QEvent.DeferredDelete included
            self.app.processEvents()

    def test_closed_plot_windows_are_freed(self):
        manager = PlotManager(parent=self.main_window)
        request = PlotRequest(
            type=PlotRequestType.TRAJECTORIES, trajectory_selection=self.selection, plot_settings=PlotSettings()
        )
        for _ in range(3):
            manager.handle_request(request)
            manager._windows[-1].close()
            self._process_deferred_deletes()
        self.assertEqual(manager._windows, [])
        self.assertEqual(self.main_window.findChildren(QtWidgets.QMainWindow), [])

    def test_opening_a_window_leaves_the_others_open(self):
        manager = UIManager(parent=self.main_window)
        manager.handle_request(UIRequest(type=UIRequestType.TRAJ_SETTINGS, trajectory_selection=self.selection))
        manager.handle_request(UIRequest(type=UIRequestType.DOF_ORGANIZER, trajectory_selection=self.selection))
        windows = self.main_window.findChildren(QtWidgets.QMainWindow)
        self.assertEqual(len(windows), 2)
        self.assertTrue(all(window.isVisible() for window in windows))
        for window in windows:
            window.close()
        self._process_deferred_deletes()
        self.assertEqual(self.main_window.findChildren(QtWidgets.QMainWindow), [])


if __name__ == "__main__":
    unittest.main()
