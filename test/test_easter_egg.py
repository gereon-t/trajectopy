"""The jingle of the About window: click counting and playback with the players of the operating systems."""

import os
import unittest
from unittest import mock

from trajectopy.gui import easter_egg

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestClickCounter(unittest.TestCase):
    def test_quick_clicks_complete_the_series(self):
        counter = easter_egg.ClickCounter(clicks=3, max_interval=1.0)
        self.assertEqual([counter.click(now) for now in (0.0, 0.5, 1.0)], [False, False, True])
        self.assertFalse(counter.click(1.5))  # the count starts over

    def test_a_pause_restarts_the_count(self):
        counter = easter_egg.ClickCounter(clicks=3, max_interval=1.0)
        self.assertEqual([counter.click(now) for now in (0.0, 0.5, 2.0, 2.5, 3.0)], [False, False, False, False, True])


class TestJingle(unittest.TestCase):
    def test_asset_is_a_small_mp3(self):
        self.assertLess(os.path.getsize(easter_egg.JINGLE_FILE_PATH), 60_000)
        with open(easter_egg.JINGLE_FILE_PATH, "rb") as file:
            header = file.read(2)
        self.assertEqual((header[0], header[1] & 0xE0), (0xFF, 0xE0))  # MPEG audio frame sync

    def test_windows_uses_the_media_control_interface(self):
        with mock.patch.object(easter_egg.sys, "platform", "win32"), mock.patch.object(easter_egg, "mci") as mci:
            easter_egg.play_jingle("C:/a b/jingle.mp3")
        self.assertEqual(
            [call.args[0] for call in mci.call_args_list],
            [
                "close trajectopy_jingle",
                'open "C:/a b/jingle.mp3" type mpegvideo alias trajectopy_jingle',
                "play trajectopy_jingle",
            ],
        )

    def test_macos_uses_afplay(self):
        with (
            mock.patch.object(easter_egg.sys, "platform", "darwin"),
            mock.patch.object(easter_egg.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
            mock.patch.object(easter_egg.subprocess, "Popen") as popen,
        ):
            easter_egg.play_jingle("jingle.mp3")
        self.assertEqual(popen.call_args.args[0], ["/usr/bin/afplay", "jingle.mp3"])

    def test_linux_uses_the_first_available_player(self):
        with (
            mock.patch.object(easter_egg.sys, "platform", "linux"),
            mock.patch.object(
                easter_egg.shutil, "which", side_effect=lambda name: "/bin/paplay" if name == "paplay" else None
            ),
            mock.patch.object(easter_egg.subprocess, "Popen") as popen,
        ):
            easter_egg.play_jingle("jingle.mp3")
        self.assertEqual(popen.call_args.args[0], ["/bin/paplay", "jingle.mp3"])

    def test_without_player_nothing_happens(self):
        with (
            mock.patch.object(easter_egg.sys, "platform", "linux"),
            mock.patch.object(easter_egg.shutil, "which", return_value=None),
            mock.patch.object(easter_egg.subprocess, "Popen") as popen,
        ):
            easter_egg.play_jingle("jingle.mp3")
        popen.assert_not_called()

    def test_errors_do_not_escape(self):
        with (
            mock.patch.object(easter_egg.sys, "platform", "win32"),
            mock.patch.object(easter_egg, "mci", side_effect=OSError("no device")),
        ):
            easter_egg.play_jingle("missing.mp3")  # logged, not raised


class TestAboutWindow(unittest.TestCase):
    def test_seven_clicks_on_the_logo_play_the_jingle(self):
        try:
            from PySide6 import QtCore, QtWidgets
            from PySide6.QtTest import QTest

            from trajectopy.gui.views import about_window
        except ImportError:  # pragma: no cover - GUI extras not installed
            self.skipTest("GUI extras not installed")
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = about_window.AboutGUI(None, "7.0.0", "2026")
        with mock.patch.object(about_window, "play_jingle") as play:
            for _ in range(easter_egg.CLICKS_TO_PLAY - 1):
                QTest.mouseClick(window.icon, QtCore.Qt.MouseButton.LeftButton)
            play.assert_not_called()
            QTest.mouseClick(window.icon, QtCore.Qt.MouseButton.LeftButton)
            play.assert_called_once()
        window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
