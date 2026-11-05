"""Table models keep the selection on the right entries, and sub-windows are freed and left open independently.

Runs headless (``QT_QPA_PLATFORM=offscreen``) and is skipped if the GUI extras are not installed.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtCore, QtWidgets

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
        self.view.select_entries([self.entries[1]])
        self.model.remove([self.entries[0].entry_id])
        self.assertEqual(self.selected_names(), ["T1"])

    def test_removing_the_selected_entry_clears_the_selection(self):
        self.view.select_entries([self.entries[2]])
        self.model.remove([self.entries[2].entry_id])
        self.assertEqual(self.selected_names(), [])
        self.assertEqual([entry.name for entry in self.model.items], ["T0", "T1", "T3"])

    def test_selection_follows_entry_when_reordered(self):
        self.view.select_entries([self.entries[1]])
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
        self.view.select_entries([self.entries[3]])
        self.model.remove_group("group")
        self.assertEqual([entry.name for entry in self.model.items], ["T0", "T1", "T2", "T3"])
        self.assertEqual(self.selected_names(), ["T3"])

    def test_derived_entries_are_grouped_after_their_source(self):
        derived = self.entries[1].derive(noisy_trajectory.copy())
        self.model.add(derived)
        self.assertEqual(self.model.items.index(derived), 2)
        self.assertEqual(self.model.rowCount(), 4)  # T0, group of T1, T2, T3
        group = self.model.index(1, 0)
        self.assertEqual(self.model.group_at(group), self.entries[1].group_id)
        self.assertEqual(self.model.rowCount(group), 2)
        self.assertEqual(self.model.data(group), "T1 (2)")

    def test_selecting_a_group_selects_its_entries(self):
        self.model.add(self.entries[1].derive(noisy_trajectory.copy()))
        group = self.model.index(1, 0)
        self.view.selectionModel().select(
            group, QtCore.QItemSelectionModel.SelectionFlag.Select | QtCore.QItemSelectionModel.SelectionFlag.Rows
        )
        self.assertEqual(len(self.view.selected_entries), 2)
        self.assertEqual(set(self.view.selection.groups), {self.entries[1].group_id})

    def test_group_rename_ungroup(self):
        ids = {self.entries[0].entry_id, self.entries[2].entry_id}
        self.view.select_entries([self.entries[2]])
        self.model.group_entries(ids, "Run 1")
        self.assertEqual([entry.name for entry in self.model.items], ["T0", "T2", "T1", "T3"])
        self.assertEqual(list(self.model.groups().values()), ["Run 1"])
        self.assertEqual(self.selected_names(), ["T2"])  # the selection follows the entry into the group

        group_id = self.entries[0].group_id
        self.model.rename_groups({group_id}, "Run 2")
        self.assertEqual(self.model.data(self.model.index(0, 0)), "Run 2 (2)")

        self.model.move_to_group({self.entries[3].entry_id}, group_id)
        self.assertEqual([entry.name for entry in self.model.items], ["T0", "T2", "T3", "T1"])
        self.assertEqual(self.entries[3].group_name, "Run 2")

        self.model.ungroup_entries({self.entries[0].entry_id, self.entries[2].entry_id})
        self.assertEqual(list(self.model.groups().values()), ["Run 2"])  # a named group stays with one entry
        self.model.ungroup_entries({self.entries[3].entry_id})
        self.assertEqual(self.model.groups(), {})
        self.assertEqual(self.model.rowCount(), 4)

    def test_dropping_entries(self):
        self.model.group_entries({self.entries[0].entry_id, self.entries[1].entry_id}, "Run")
        data = self.model.mimeData([self.model.entry_index(self.entries[3].entry_id)])
        group = self.model.group_index(self.entries[0].group_id)
        self.assertTrue(self.model.dropMimeData(data, QtCore.Qt.DropAction.MoveAction, -1, -1, group))
        self.assertEqual(self.entries[3].group_id, self.entries[0].group_id)

        # onto an entry outside of groups: a group with that entry
        data = self.model.mimeData([self.model.entry_index(self.entries[3].entry_id)])
        target = self.model.entry_index(self.entries[2].entry_id)
        self.assertTrue(self.model.dropMimeData(data, QtCore.Qt.DropAction.MoveAction, -1, -1, target))
        self.assertEqual(self.entries[3].group_id, self.entries[2].group_id)
        self.assertEqual(len(self.model.groups()), 2)

        # between the rows of the top level: out of the group
        data = self.model.mimeData([self.model.entry_index(self.entries[3].entry_id)])
        self.assertTrue(self.model.dropMimeData(data, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()))
        self.assertEqual(len(self.model.groups()), 1)

    def test_collapsed_groups_stay_collapsed(self):
        self.model.group_entries({self.entries[0].entry_id, self.entries[1].entry_id}, "Run")
        group = self.model.group_index(self.entries[0].group_id)
        self.assertTrue(self.view.isExpanded(group))
        self.view.collapse(group)
        self.model.remove([self.entries[3].entry_id])
        self.assertFalse(self.view.isExpanded(self.model.group_index(self.entries[0].group_id)))

    def test_reset_clears_the_selection(self):
        self.view.select_entries([self.entries[0]])
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

    def test_windows_open_where_the_main_window_is(self):
        from trajectopy.gui.utils import center_window

        available = self.main_window.screen().availableGeometry()
        self.main_window.setGeometry(available.left() + 50, available.top() + 50, 600, 400)
        self.main_window.show()
        window = QtWidgets.QMainWindow(self.main_window)
        window.resize(200, 100)
        center_window(window)
        self.assertEqual(window.frameGeometry().center(), self.main_window.frameGeometry().center())

        self.main_window.move(available.left(), available.top())  # partly beyond the screen if centered
        window.resize(available.width(), 100)
        center_window(window)
        self.assertEqual(window.frameGeometry().left(), available.left())
        self.main_window.hide()

    def _visible_windows(self) -> list:
        return [window for window in self.main_window.findChildren(QtWidgets.QMainWindow) if window.isVisible()]

    def _close_all(self) -> None:
        for window in self.main_window.findChildren(QtWidgets.QMainWindow):
            window.close()
        self._process_deferred_deletes()

    def test_settings_of_a_trajectory_open_only_once(self):
        manager = UIManager(parent=self.main_window)
        request = UIRequest(type=UIRequestType.TRAJ_SETTINGS, trajectory_selection=self.selection)
        manager.handle_request(request)
        first = self._visible_windows()
        manager.handle_request(request)

        self.assertEqual(self._visible_windows(), first)
        self.assertEqual(len(first), 1)

        # once closed, it opens again
        self._close_all()
        manager.handle_request(request)
        self.assertEqual(len(self._visible_windows()), 1)
        self._close_all()

    def test_a_window_for_the_current_selection_replaces_the_open_one(self):
        manager = UIManager(parent=self.main_window)
        request = UIRequest(type=UIRequestType.DOF_ORGANIZER, trajectory_selection=self.selection)
        manager.handle_request(request)
        first = self._visible_windows()
        manager.handle_request(request)
        second = self._visible_windows()

        self.assertEqual(len(second), 1)
        self.assertIsNot(second[0], first[0])
        self._close_all()

    def test_applying_settings_keeps_values_outside_of_the_usual_range(self):
        from trajectopy.core.settings import ProcessingSettings
        from trajectopy.gui.views.json_settings_view import JSONViewer

        settings = ProcessingSettings()
        settings.alignment.preprocessing.max_samples = 250_000
        settings.alignment.preprocessing.time_start = -2.5
        viewer = JSONViewer(parent=self.main_window, settings=settings)
        viewer.apply_changes()
        viewer.close()
        self._process_deferred_deletes()

        self.assertEqual(settings.alignment.preprocessing.max_samples, 250_000)
        self.assertEqual(settings.alignment.preprocessing.time_start, -2.5)


@unittest.skipIf(QtWidgets is None, "GUI dependencies not installed")
class TestUniqueIds(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_a_result_file_opened_twice_gives_two_independent_entries(self):
        import tempfile

        from trajectopy.gui.models.entries import AbsoluteDeviationEntry
        from trajectopy.gui.models.result_model import ResultTableModel
        from trajectopy.processing.evaluation import ate

        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "ate.result")
            AbsoluteDeviationEntry(deviations=ate(noisy_trajectory, noisy_trajectory, align=False)).to_file(filename)
            first, second = AbsoluteDeviationEntry.from_file(filename), AbsoluteDeviationEntry.from_file(filename)

        model = ResultTableModel()
        model.add(first)
        model.add(second)
        model.remove([second.entry_id])

        self.assertEqual(model.items, [first])

    def test_operation_results_keep_the_current_reference(self):
        import copy

        from trajectopy.gui.managers.requests import TrajectoryModelRequest, TrajectoryModelRequestType

        model = TrajectoryTableModel()
        first, second = _entry("first"), _entry("second")
        first.set_as_reference = True
        model.add(first)
        model.add(second)
        result = copy.deepcopy(first)  # e.g. an EPSG change, computed while the reference is changed
        model.handle_request(
            TrajectoryModelRequest(
                type=TrajectoryModelRequestType.SET_REFERENCE, selection=TrajectorySelection(entries=[second])
            )
        )
        model.handle_request(
            TrajectoryModelRequest(
                type=TrajectoryModelRequestType.UPDATE, selection=TrajectorySelection(entries=[result])
            )
        )

        self.assertEqual([entry.set_as_reference for entry in model.items], [False, True])


if __name__ == "__main__":
    unittest.main()
