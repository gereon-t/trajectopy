import os
import sys
import tempfile
import unittest
from unittest import mock

try:
    from PySide6 import QtCore, QtWidgets

    from trajectopy.gui import file_dialog, utils
    from trajectopy.gui.file_dialog import Mode
    from trajectopy.visualization.figure import ensure_app
except ImportError:  # GUI dependencies not installed
    raise unittest.SkipTest("GUI dependencies not installed") from None

QT_DIALOG = QtWidgets.QFileDialog.Option.DontUseNativeDialog
ACCEPTED = QtWidgets.QDialog.DialogCode.Accepted


class TestFileDialogs(unittest.TestCase):
    def setUp(self) -> None:
        ensure_app()
        # keep the settings of the tests away from the user's settings
        self._settings_dir = tempfile.TemporaryDirectory()
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        QtCore.QSettings.setPath(
            QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, self._settings_dir.name
        )
        self.folder = tempfile.TemporaryDirectory()
        self.file = os.path.join(self.folder.name, "poses.traj")
        open(self.file, "w").close()  # noqa: SIM115

    def tearDown(self) -> None:
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.NativeFormat)
        self.folder.cleanup()
        self._settings_dir.cleanup()

    def _run(self, mode: Mode, selected: list[str], accepted: bool = True, **kwargs):
        shown: list[QtWidgets.QFileDialog] = []

        def fake_exec(dialog: QtWidgets.QFileDialog) -> int:
            shown.append(dialog)
            return ACCEPTED if accepted else QtWidgets.QDialog.DialogCode.Rejected

        with (
            mock.patch.object(QtWidgets.QFileDialog, "exec", fake_exec),
            mock.patch.object(QtWidgets.QFileDialog, "selectedFiles", return_value=selected),
        ):
            paths = file_dialog.run_file_dialog(None, mode, "Open", "Trajectory File (*.traj)", **kwargs)
        return paths, shown[0]

    def test_qt_dialog_on_windows(self):
        """The native Windows dialog takes seconds to return after a file was chosen."""
        self.assertEqual(bool(file_dialog.file_dialog_options() & QT_DIALOG), sys.platform == "win32")

    def test_returns_the_selection_or_nothing(self):
        self.assertEqual(self._run(Mode.OPEN_MANY, [self.file])[0], [self.file])
        self.assertEqual(self._run(Mode.OPEN_MANY, [self.file], accepted=False)[0], [])

    def test_opens_in_the_folder_used_last_for_the_kind_of_file(self):
        self._run(Mode.OPEN_MANY, [self.file])
        _, dialog = self._run(Mode.OPEN, [])
        # samefile: the temporary folder may be given in the short 8.3 form on Windows (e.g. RUNNER~1)
        self.assertTrue(os.path.samefile(dialog.directory().absolutePath(), self.folder.name))
        if dialog.testOption(QT_DIALOG):  # only Qt's own dialog (Windows) lists the recent folders in its sidebar
            sidebar = [url.toLocalFile() for url in dialog.sidebarUrls()]
            self.assertTrue(any(os.path.isdir(path) and os.path.samefile(path, self.folder.name) for path in sidebar))

    @unittest.skipUnless(sys.platform == "win32", "Qt's own dialog is used on Windows only")
    def test_qt_dialog_is_adapted(self):
        _, dialog = self._run(Mode.SAVE, [self.file])
        view = dialog.findChild(QtWidgets.QTreeView, "treeView")
        self.assertTrue(view.isColumnHidden(2))  # the type column
        buttons = dialog.findChild(QtWidgets.QDialogButtonBox, "buttonBox")
        cancel = [b for b in buttons.buttons() if buttons.buttonRole(b) == buttons.ButtonRole.RejectRole]
        self.assertEqual(cancel[0].objectName(), "secondaryButton")
        self.assertEqual(dialog.defaultSuffix(), "traj")
        self.assertEqual(dialog.labelText(QtWidgets.QFileDialog.DialogLabel.Accept), "Save")

    def test_helpers_return_single_paths(self):
        with mock.patch.object(utils, "run_file_dialog", return_value=[self.file]) as run:
            self.assertEqual(utils.open_file_dialog(None), self.file)
            self.assertEqual(utils.save_file_dialog(None), self.file)
            self.assertEqual(utils.read_file_dialog(None), [self.file])
            self.assertEqual(utils.browse_dir_dialog(None), self.file)
        self.assertEqual(
            [call.args[1] for call in run.call_args_list], [Mode.OPEN, Mode.SAVE, Mode.OPEN_MANY, Mode.DIRECTORY]
        )


if __name__ == "__main__":
    unittest.main()
