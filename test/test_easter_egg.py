"""The jingle of the About window: click counting, decoding, reverb and playback."""

import os
import tempfile
import unittest
import wave
from unittest import mock

import numpy as np

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
    def test_asset_is_small_and_decodes(self):
        self.assertLess(os.path.getsize(easter_egg.JINGLE_FILE_PATH), 80_000)
        signal, sample_rate = easter_egg.read_jingle()
        self.assertEqual(sample_rate, 16000)
        self.assertAlmostEqual(len(signal) / sample_rate, 9.2, places=1)
        self.assertLessEqual(np.max(np.abs(signal)), 1.0)
        self.assertGreater(np.max(np.abs(signal)), 0.5)

    def test_decoder_follows_the_ima_adpcm_steps(self):
        # code 7 adds step/8 + step/4 + step/2 + step (integer division): 0 + 1 + 3 + 7 = 11 for the first step
        # size 7, then raises the step index by 8 to the step size 16: 2 + 4 + 8 + 16 = 30
        np.testing.assert_array_equal(easter_egg.decode_adpcm(bytes([0x77]), 2), [11, 41])

    def test_reverb_adds_a_stereo_tail(self):
        signal = np.zeros(1000)
        signal[0] = 1.0
        result = easter_egg.add_reverb(signal, 1000, decay=0.5)
        self.assertEqual(result.shape, (1000 + 500 + 25 - 1, 2))
        self.assertAlmostEqual(float(np.max(np.abs(result))), 0.9)
        self.assertFalse(np.allclose(result[:, 0], result[:, 1]))  # wide, not mono

    def test_playback_writes_a_wav_file(self):
        with mock.patch.object(easter_egg, "_play_file") as play_file:
            easter_egg._play()
        (path,), _ = play_file.call_args
        with wave.open(path) as file:
            self.assertEqual((file.getnchannels(), file.getframerate()), (2, 16000))
            self.assertGreater(file.getnframes() / file.getframerate(), 11.0)  # clip and reverb tail

    def test_errors_do_not_escape(self):
        with mock.patch.object(easter_egg, "read_jingle", side_effect=OSError("missing")):
            easter_egg._play()  # logged, not raised

    def test_write_wav_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "clip.wav")
            easter_egg.write_wav(path, np.array([[2.0, -2.0]]), 8000)
            with wave.open(path) as file:
                np.testing.assert_array_equal(np.frombuffer(file.readframes(1), "<i2"), [32767, -32767])


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
