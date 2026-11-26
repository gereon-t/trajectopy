import sys
import unittest
from unittest import mock

from PySide6 import QtWidgets

from trajectopy.gui import utils

QT_DIALOG = QtWidgets.QFileDialog.Option.DontUseNativeDialog


class TestFileDialogs(unittest.TestCase):
    def test_qt_dialog_on_windows(self):
        """The native Windows dialog takes seconds to return after a file was chosen."""
        expected = sys.platform == "win32"
        self.assertEqual(bool(utils.file_dialog_options() & QT_DIALOG), expected)

    def test_all_dialogs_use_the_options(self):
        dialog = QtWidgets.QFileDialog
        with (
            mock.patch.object(dialog, "getOpenFileNames", return_value=(["a"], "")) as open_files,
            mock.patch.object(dialog, "getOpenFileName", return_value=("a", "")) as open_file,
            mock.patch.object(dialog, "getSaveFileName", return_value=("a", "")) as save_file,
            mock.patch.object(dialog, "getExistingDirectory", return_value="d") as directory,
        ):
            self.assertEqual(utils.read_file_dialog(None), ["a"])
            self.assertEqual(utils.open_file_dialog(None), "a")
            self.assertEqual(utils.save_file_dialog(None), "a")
            self.assertEqual(utils.browse_dir_dialog(None), "d")
        options = utils.file_dialog_options()
        for call in (open_files, open_file, save_file):
            self.assertEqual(call.call_args.kwargs["options"], options)
        self.assertTrue(directory.call_args.args[3] & options == options)


if __name__ == "__main__":
    unittest.main()
