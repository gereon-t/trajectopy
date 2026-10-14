"""3D playback in the browser (CesiumJS page with CZML), tested without a browser."""

import json
import os
import re
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

from test.test_kml import _ate_result, _georeferenced
from trajectopy.io import cesium


def _page(write, *args, **kwargs) -> tuple[str, list[dict]]:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "playback.html")
        write(*args, path, **kwargs)
        with open(path, encoding="utf-8") as file:
            page = file.read()
    czml = json.loads(re.search(r"const czml = (.*);\nconst trail", page).group(1))
    return page, czml


def _trail(page: str) -> dict:
    return json.loads(re.search(r"const trail = (.*);\nconst ionToken", page).group(1))


def _packets(czml: list[dict], prefix: str) -> list[dict]:
    return [packet for packet in czml if packet["id"].startswith(prefix)]


class TestCesiumPage(unittest.TestCase):
    def test_trajectory_playback(self):
        trajectory = _georeferenced()
        page, czml = _page(cesium.write_trajectory_html, trajectory, color="#1f77b4")
        document, point = czml[0], _packets(czml, "position")[0]
        start, stop = document["clock"]["interval"].split("/")
        self.assertTrue(start.startswith("2023-11-14T22:13:20"))
        self.assertEqual(document["clock"]["currentTime"], stop)  # starts with the whole line visible

        samples = point["position"]["cartographicDegrees"]
        self.assertEqual(len(samples), 4 * len(trajectory))  # time, longitude, latitude, height per pose
        self.assertAlmostEqual(samples[-4], trajectory.timestamps[-1] - trajectory.timestamps[0], places=3)
        self.assertAlmostEqual(samples[1], 7.08, delta=0.05)  # Bonn

        lines = _packets(czml, "line")
        self.assertGreater(len(lines), 1)  # split into pieces that appear one after another
        self.assertTrue(
            all(line["polyline"]["material"]["solidColor"]["color"]["rgba"] == [31, 119, 180, 255] for line in lines)
        )
        poses = sum(len(line["polyline"]["positions"]["cartographicDegrees"]) // 3 for line in lines)
        self.assertEqual(poses - (len(lines) - 1), len(trajectory))  # complete, consecutive lines share a pose
        self.assertIn(cesium.CESIUM_URL, page)
        self.assertIn('const ionToken = "";', page)

    def test_ate_playback_is_colored_with_legend(self):
        page, czml = _page(cesium.write_ate_html, _ate_result(), field="position", max_std=0)
        colors = {
            tuple(line["polyline"]["material"]["solidColor"]["color"]["rgba"]) for line in _packets(czml, "line")
        }
        self.assertGreater(len(colors), 5)  # the deviation grows along the trajectory: many color classes
        self.assertIn("Position deviation [m]", page)
        self.assertIn("≥ ", page)

    def test_options(self):
        page, _ = _page(cesium.write_trajectory_html, _georeferenced(), map_style="satellite", ion_token=" abc ")
        self.assertIn('const ionToken = "abc";', page)
        self.assertIn('const mapStyle = "satellite";', page)
        with self.assertRaises(ValueError):
            _page(cesium.write_trajectory_html, _georeferenced(), map_style="google")
        with self.assertRaises(ValueError):
            _page(cesium.write_trajectory_html, _georeferenced(), color="blue")
        with self.assertRaises(ValueError):
            _page(cesium.write_ate_html, _ate_result(with_orientation=False), field="yaw")

    def test_line_never_runs_ahead_of_the_position(self):
        """At every pose, the complete lines plus the page's trail cover exactly the poses up to the position."""
        import bisect

        page, czml = _page(cesium.write_ate_html, _ate_result(), field="position", max_std=0)
        trail = _trail(page)
        times = trail["t"]
        start = czml[0]["clock"]["interval"].split("/")[0]
        line_start = {}  # pose index from which a line is shown (its availability)
        for line, last in enumerate(trail["last"]):
            availability_start = _packets(czml, f"line{line}")[0]["availability"].split("/")[0]
            self.assertTrue(availability_start >= start)
            line_start[line] = last - 1
        for pose in range(len(times)):
            covered = set()
            for line, (first, last) in enumerate(zip(trail["first"], trail["last"])):
                if line_start[line] <= pose:  # complete
                    covered.update(range(first, last))
            line = bisect.bisect_right(trail["first"], pose) - 1  # the rule of the page (lastAtOrBefore)
            if pose < trail["last"][line] - 1:  # incomplete: drawn by the trail
                covered.update(range(trail["first"][line], pose + 1))
            self.assertEqual(covered, set(range(pose + 1)), f"pose {pose}")

    def test_heights(self):
        trajectory = _georeferenced()
        page, czml = _page(cesium.write_trajectory_html, trajectory)
        self.assertIsNone(_trail(page)["h"])  # default: clamped to the ground
        self.assertEqual(_packets(czml, "position")[0]["point"]["heightReference"], "CLAMP_TO_GROUND")

        page, czml = _page(cesium.write_trajectory_html, trajectory, heights="ellipsoidal")
        heights = _trail(page)["h"]
        self.assertAlmostEqual(heights[0], trajectory.xyz[0, 2], places=2)  # UTM heights taken as they are
        self.assertEqual(_packets(czml, "position")[0]["point"]["heightReference"], "NONE")
        line = _packets(czml, "line")[0]["polyline"]
        self.assertNotIn("clampToGround", line)
        self.assertAlmostEqual(line["positions"]["cartographicDegrees"][2], trajectory.xyz[0, 2], places=2)

        with self.assertRaises(ValueError):
            _page(cesium.write_trajectory_html, trajectory, heights="orthometric")

    def test_geoid_heights(self):
        """Heights above sea level are converted with the EGM2008 geoid (about +47 m in Bonn)."""
        trajectory = _georeferenced()
        try:
            page, _ = _page(cesium.write_trajectory_html, trajectory, heights="geoid")
        except ValueError as error:  # no internet connection and the grid is not cached
            self.skipTest(str(error))
        self.assertAlmostEqual(_trail(page)["h"][0] - trajectory.xyz[0, 2], 47.2, delta=0.5)

    def test_geoid_model_unavailable(self):
        group = mock.Mock(best_available=False)
        with (
            mock.patch.object(cesium, "TransformerGroup", return_value=group),
            self.assertRaisesRegex(ValueError, "geoid model is not available"),
        ):
            _page(cesium.write_trajectory_html, _georeferenced(), heights="geoid")

    def test_photorealistic_needs_a_token(self):
        with self.assertRaises(ValueError):
            _page(cesium.write_trajectory_html, _georeferenced(), map_style="photorealistic")
        page, _ = _page(cesium.write_trajectory_html, _georeferenced(), map_style="photorealistic", ion_token="t")
        self.assertIn('const mapStyle = "photorealistic";', page)

    def test_camera_can_follow_the_position(self):
        page, _ = _page(cesium.write_trajectory_html, _georeferenced())
        self.assertIn('<input type="checkbox" id="follow" checked>', page)

    def test_map_can_be_switched_on_the_page(self):
        page, _ = _page(cesium.write_trajectory_html, _georeferenced(), map_style="satellite")
        for style in cesium.MAP_STYLES:
            self.assertIn(f'<option value="{style}">', page)
        self.assertIn('const mapStyle = "satellite";', page)  # the initial map

    def test_names_cannot_break_the_page(self):
        trajectory = _georeferenced()
        trajectory.name = '</script><script>alert("x")</script>'
        page, czml = _page(cesium.write_trajectory_html, trajectory)
        self.assertNotIn("</script><script>alert", page)  # cannot end the script: "</" is escaped in data
        self.assertEqual(czml[0]["name"], trajectory.name)


class TestShowInBrowser(unittest.TestCase):
    def test_serves_only_the_page(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "playback.html")
            cesium.write_trajectory_html(_georeferenced(), path)
            with open(os.path.join(directory, "secret.txt"), "w") as file:
                file.write("secret")

            def browser(url: str) -> None:  # fetch like a browser would
                self.page = urllib.request.urlopen(url, timeout=5).read()

            with mock.patch.object(cesium.webbrowser, "open", side_effect=browser):
                url = cesium.show_in_browser(path, wait=5)
            self.assertTrue(url.startswith("http://127.0.0.1:"))
            with open(path, "rb") as file:
                self.assertEqual(self.page, file.read())
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(url.rsplit("/", 1)[0] + "/secret.txt", timeout=5)


class TestPlaybackDialog(unittest.TestCase):
    def test_dialog_remembers_map_and_token(self):
        try:
            from PySide6 import QtCore, QtWidgets
        except ImportError:  # pragma: no cover - GUI extras not installed
            self.skipTest("GUI dependencies not installed")
        from trajectopy.gui.views import cesium_dialog

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            ini = os.path.join(directory, "settings.ini")
            settings = lambda: QtCore.QSettings(ini, QtCore.QSettings.Format.IniFormat)  # noqa: E731
            with mock.patch.object(cesium_dialog, "app_settings", side_effect=settings):
                dialog = cesium_dialog.CesiumPlaybackDialog(
                    fields=[("position", "Position"), ("yaw", "Yaw")], field="yaw"
                )
                self.assertEqual(dialog.field, "yaw")
                dialog.map_box.setCurrentIndex(dialog.map_box.findData("satellite"))
                dialog.token_edit.setText("token123")
                dialog.heights_box.setCurrentIndex(dialog.heights_box.findData("ellipsoidal"))
                dialog.accept()

                dialog = cesium_dialog.CesiumPlaybackDialog()
                self.assertEqual(dialog.map_style, "satellite")
                self.assertEqual(dialog.ion_token, "token123")
                self.assertEqual(dialog.heights, "ellipsoidal")
                self.assertEqual(dialog.field, "")  # trajectories: a color instead of a field


if __name__ == "__main__":
    unittest.main()
