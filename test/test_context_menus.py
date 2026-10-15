"""Context menus: entries stay visible with a reason when unavailable, and shortcuts trigger the menu entries."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:  # pragma: no cover - GUI extras not installed
    QtWidgets = None


@unittest.skipIf(QtWidgets is None, "GUI dependencies not installed")
class TestContextMenus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        from test.testdata import generated_trajectory
        from trajectopy.gui.models.entries import TrajectoryEntry
        from trajectopy.gui.views.trajectory_context_menu import TrajectoryContextMenu

        self.trajectory = TrajectoryEntry(full_filename="", trajectory=generated_trajectory.copy())
        self.reference = TrajectoryEntry(full_filename="", trajectory=generated_trajectory.copy())
        self.reference.set_as_reference = True
        self.menu = TrajectoryContextMenu()
        self.menu.alignments_available = lambda: False

    def _selection(self, *entries, reference=None):
        from trajectopy.gui.models.selection import TrajectorySelection

        return TrajectorySelection(entries=list(entries), reference_entry=reference)

    def _actions(self) -> dict:
        from trajectopy.gui.views.menu_helpers import iter_actions

        return {action.text(): action for action in iter_actions(self.menu)}

    def test_reference_entries_explain_why_they_are_unavailable(self):
        self.menu.populate(self._selection(self.trajectory))
        actions = self._actions()
        for text in ("Absolute Error (ATE), Aligned First", "Estimate Alignment", "Interactive Alignment…"):
            self.assertFalse(actions[text].isEnabled(), text)
            self.assertIn("Set a reference first", actions[text].toolTip())
        self.assertTrue(any(text.startswith("ⓘ") for text in actions))  # visible note in the submenu
        self.assertIn("no alignments", actions["Apply Existing Alignment…"].toolTip())

        self.menu.populate(self._selection(self.trajectory, reference=self.reference))
        actions = self._actions()
        self.assertTrue(actions["Estimate Alignment"].isEnabled())
        self.assertFalse(any(text.startswith("ⓘ") for text in actions))

    def test_selection_size(self):
        self.menu.populate(self._selection(self.trajectory))
        actions = self._actions()
        self.assertFalse(actions["Merge"].isEnabled())
        self.assertIn("two or more", actions["Merge"].toolTip())
        self.assertTrue(actions["Rename"].isEnabled())

        self.menu.populate(self._selection(self.trajectory, self.reference, reference=self.reference))
        actions = self._actions()
        self.assertTrue(actions["Merge"].isEnabled())
        self.assertFalse(actions["Rename"].isEnabled())
        self.assertIn("single trajectory", actions["Interactive Alignment…"].toolTip())

    def test_every_entry_explains_itself(self):
        self.menu.populate(self._selection(self.trajectory, reference=self.reference))
        self_explanatory = {"Rename", "Remove"}
        for text, action in self._actions().items():
            if text not in self_explanatory:
                self.assertTrue(action.toolTip() and action.toolTip() != text.replace("&", ""), text)

    def test_shortcuts_trigger_the_menu_entries(self):
        from trajectopy.gui.managers.requests import TrajectoryModelRequestType, UIRequestType

        received = []
        self.menu.trajectory_model_request.connect(received.append)
        self.menu.ui_request.connect(received.append)
        selection = self._selection(self.trajectory)

        def press(key, modifiers=QtCore.Qt.KeyboardModifier.NoModifier) -> bool:
            event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key, modifiers)
            return self.menu.handle_key_press(selection, event)

        self.assertTrue(press(QtCore.Qt.Key.Key_R))
        self.assertTrue(press(QtCore.Qt.Key.Key_F2))
        self.assertTrue(press(QtCore.Qt.Key.Key_Delete))
        self.assertTrue(press(QtCore.Qt.Key.Key_C, QtCore.Qt.KeyboardModifier.ControlModifier))
        self.assertFalse(press(QtCore.Qt.Key.Key_M))  # merge needs two trajectories
        self.assertFalse(press(QtCore.Qt.Key.Key_J))
        self.assertEqual(
            [request.type for request in received],
            [
                TrajectoryModelRequestType.SET_REFERENCE,
                UIRequestType.RENAME_TRAJ,  # asks for the name first
                TrajectoryModelRequestType.REMOVE,
                TrajectoryModelRequestType.COPY,
            ],
        )

    def test_result_menu(self):
        from test.testdata import generated_trajectory
        from trajectopy.gui.models.entries import AbsoluteDeviationEntry
        from trajectopy.gui.models.selection import ResultSelection
        from trajectopy.gui.views.menu_helpers import iter_actions
        from trajectopy.gui.views.result_context_menu import ResultContextMenu
        from trajectopy.processing import evaluation

        deviations = evaluation.ate(trajectory=generated_trajectory, other=generated_trajectory, align=False)
        entry = AbsoluteDeviationEntry(deviations=deviations)
        menu = ResultContextMenu()
        menu.populate(ResultSelection(entries=[entry, entry]))
        actions = {action.text(): action for action in iter_actions(menu)}
        self.assertTrue(actions["Plot"].isEnabled())  # deviations are plotted together
        self.assertFalse(actions["Export…"].isEnabled())
        self.assertIn("single result", actions["Export…"].toolTip())


if __name__ == "__main__":
    unittest.main()
