"""Google Earth export: time-stamped track (playback) in a chosen color."""

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np

from test.testdata import generated_trajectory
from trajectopy.core.positions import Positions
from trajectopy.core.trajectory import Trajectory
from trajectopy.io.kml import KML_GX_NAMESPACE, KML_NAMESPACE, kml_color
from trajectopy.utils.definitions import Sorting

NS = {"kml": KML_NAMESPACE, "gx": KML_GX_NAMESPACE}
UTM_OFFSET = np.array([364800.0, 5621500.0, 60.0])  # Bonn, Germany in UTM zone 32N


def _georeferenced() -> Trajectory:
    trajectory = generated_trajectory.copy()
    trajectory.positions = Positions(xyz=trajectory.xyz + UTM_OFFSET, epsg=25832)
    trajectory.timestamps = trajectory.timestamps + 1.7e9  # November 2023
    return trajectory


def _export(trajectory: Trajectory, **kwargs) -> ET.Element:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "track.kml")
        trajectory.to_kml(path, **kwargs)
        return ET.parse(path).getroot()


class TestKml(unittest.TestCase):
    def test_track_has_a_timestamp_per_position(self):
        trajectory = _georeferenced()
        root = _export(trajectory)
        track = root.find(".//gx:Track", NS)
        self.assertIsNotNone(track)
        whens = [element.text for element in track.findall("kml:when", NS)]
        coords = [element.text for element in track.findall("gx:coord", NS)]
        self.assertEqual(len(whens), len(trajectory))
        self.assertEqual(len(coords), len(trajectory))

        times = [datetime.strptime(when, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc) for when in whens]
        self.assertEqual(times, sorted(times))
        self.assertAlmostEqual(times[0].timestamp(), trajectory.timestamps[0], places=3)

        longitude, latitude, _ = map(float, coords[0].split())
        self.assertAlmostEqual(longitude, 7.1, delta=0.1)  # Bonn
        self.assertAlmostEqual(latitude, 50.7, delta=0.1)

    def test_spatially_sorted_trajectory_is_played_back_in_time_order(self):
        trajectory = _georeferenced()
        reversed_order = trajectory.copy()
        reversed_order.mask(np.arange(len(trajectory))[::-1])
        reversed_order.set_sorting(Sorting.PATH_LENGTH)
        whens = [element.text for element in _export(reversed_order).findall(".//kml:when", NS)]
        self.assertEqual(whens, sorted(whens))

    def test_repeated_positions_are_kept(self):
        """Later laps over the same positions must not be dropped (playback timing)."""
        trajectory = _georeferenced()
        two_laps = Trajectory(
            positions=Positions(xyz=np.vstack([trajectory.xyz, trajectory.xyz]), epsg=25832),
            timestamps=np.concatenate([trajectory.timestamps, trajectory.timestamps + 1000.0]),
        )
        self.assertEqual(len(_export(two_laps).findall(".//gx:coord", NS)), 2 * len(trajectory))

    def test_color(self):
        root = _export(_georeferenced(), color="#1f77b4")
        colors = {element.text for element in root.findall(".//kml:Style//kml:color", NS)}
        self.assertEqual(colors, {"ffb4771f"})  # line and points, KML order aabbggrr

    def test_kml_color(self):
        self.assertEqual(kml_color("#ff0000"), "ff0000ff")
        self.assertEqual(kml_color("00ff0080"), "8000ff00")
        for invalid in ("red", "#12345", "#gg0000"):
            with self.subTest(color=invalid), self.assertRaises(ValueError):
                kml_color(invalid)

    def test_unknown_coordinate_system(self):
        with self.assertRaises(ValueError):
            _export(generated_trajectory)

    def test_gui_export_uses_the_chosen_color(self):
        try:
            from PySide6 import QtGui, QtWidgets
        except ImportError:  # pragma: no cover - GUI extras not installed
            self.skipTest("GUI dependencies not installed")
        from unittest import mock

        from trajectopy.gui.managers.requests import UIRequest, UIRequestType
        from trajectopy.gui.managers.ui_manager import UIManager
        from trajectopy.gui.models.entries import TrajectoryEntry
        from trajectopy.gui.models.selection import TrajectorySelection

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        request = UIRequest(
            type=UIRequestType.EXPORT_KML,
            trajectory_selection=TrajectorySelection(entries=[TrajectoryEntry("", _georeferenced())]),
        )
        manager = UIManager()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "track.kml")
            with (
                mock.patch.object(QtWidgets.QColorDialog, "getColor", return_value=QtGui.QColor("#00ff00")),
                mock.patch("trajectopy.gui.managers.ui_manager.save_file_dialog", return_value=path),
            ):
                manager.handle_request(request)
            colors = {element.text for element in ET.parse(path).getroot().findall(".//kml:Style//kml:color", NS)}
            self.assertEqual(colors, {"ff00ff00"})

            os.remove(path)
            with (
                mock.patch.object(QtWidgets.QColorDialog, "getColor", return_value=QtGui.QColor()),  # canceled
                mock.patch("trajectopy.gui.managers.ui_manager.save_file_dialog", return_value=path) as save,
            ):
                manager.handle_request(request)
            save.assert_not_called()
            self.assertFalse(os.path.exists(path))


def _ate_result(with_orientation: bool = True):
    from trajectopy.processing.evaluation import ate

    reference = _georeferenced()
    estimate = reference.copy()
    rng = np.random.default_rng(0)
    xyz = estimate.xyz.copy()
    xyz[:, 0] += np.linspace(0.0, 2.0, len(xyz))  # deviation growing along the trajectory
    xyz[:, 1] += rng.normal(0.0, 0.01, len(xyz))
    estimate.positions = Positions(xyz=xyz, epsg=25832)
    if not with_orientation:
        estimate.rotations = None
        reference.rotations = None
    return ate(estimate, reference, align=False)


class TestAteKml(unittest.TestCase):
    def _export(self, result, **kwargs) -> ET.Element:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ate.kml")
            result.to_kml(path, **kwargs)
            return ET.parse(path).getroot()

    def test_fields_depend_on_orientation(self):
        from trajectopy.io.kml import ate_kml_fields

        with_rotation = {field.key for field in ate_kml_fields(_ate_result())}
        without_rotation = {field.key for field in ate_kml_fields(_ate_result(with_orientation=False))}
        self.assertTrue({"position", "x", "along_track", "rotation", "yaw"} <= with_rotation)
        self.assertNotIn("rotation", without_rotation)
        self.assertIn("vertical_cross_track", without_rotation)

    def test_line_is_colored_by_the_field(self):
        from trajectopy.utils.colors import value_color

        result = _ate_result()
        root = self._export(result, field="position", max_std=0)
        legend = root.find("kml:Document/kml:Folder", NS)
        self.assertEqual(legend.find("kml:name", NS).text, "Position deviation [m]")
        classes = legend.findall("kml:Folder", NS)
        self.assertEqual(len(classes), 11)
        self.assertTrue(classes[0].find("kml:name", NS).text.startswith("≤"))

        # deviations grow along the trajectory: it starts in the lowest (blue) and ends in the highest (red) class
        placemarks = sorted(
            legend.iter(f"{{{KML_NAMESPACE}}}Placemark"), key=lambda p: p.find(".//kml:begin", NS).text
        )
        style = {
            s.get("id"): s.find("kml:LineStyle/kml:color", NS).text for s in root.iter(f"{{{KML_NAMESPACE}}}Style")
        }
        first_style = placemarks[0].find("kml:styleUrl", NS).text.lstrip("#")
        last_style = placemarks[-1].find("kml:styleUrl", NS).text.lstrip("#")
        self.assertEqual(style[first_style], kml_color(value_color(0.5 / 11)))
        self.assertEqual(style[last_style], kml_color(value_color(10.5 / 11)))

        # the colored runs together cover the whole trajectory (consecutive runs share their boundary pose)
        points = sum(len(p.find(".//kml:coordinates", NS).text.split()) for p in placemarks)
        self.assertEqual(points - (len(placemarks) - 1), len(result.trajectory))

    def test_playback(self):
        result = _ate_result()
        root = self._export(result)
        whens = [element.text for element in root.findall(".//gx:Track/kml:when", NS)]
        self.assertEqual(len(whens), len(result.trajectory))
        self.assertEqual(whens, sorted(whens))

    def test_unavailable_field(self):
        with self.assertRaises(ValueError):
            self._export(_ate_result(with_orientation=False), field="yaw")
        with self.assertRaises(ValueError):
            self._export(_ate_result(), field="speed")

    def test_gui_export_uses_the_chosen_field(self):
        try:
            from PySide6 import QtWidgets
        except ImportError:  # pragma: no cover - GUI extras not installed
            self.skipTest("GUI dependencies not installed")
        from unittest import mock

        from trajectopy.gui.managers.requests import UIRequest, UIRequestType
        from trajectopy.gui.managers.ui_manager import UIManager
        from trajectopy.gui.models.entries import AbsoluteDeviationEntry
        from trajectopy.gui.models.selection import ResultSelection

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        request = UIRequest(
            type=UIRequestType.EXPORT_KML,
            result_selection=ResultSelection(entries=[AbsoluteDeviationEntry(deviations=_ate_result())]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ate.kml")
            with (
                mock.patch.object(QtWidgets.QInputDialog, "getItem", return_value=("Yaw deviation [°]", True)),
                mock.patch("trajectopy.gui.managers.ui_manager.save_file_dialog", return_value=path),
            ):
                UIManager().handle_request(request)
            legend = ET.parse(path).getroot().find("kml:Document/kml:Folder/kml:name", NS)
            self.assertEqual(legend.text, "Yaw deviation [°]")


if __name__ == "__main__":
    unittest.main()
