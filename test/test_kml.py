"""Google Earth export: time-stamped track (playback), colors, heights and the export dialog."""

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from unittest import mock

import numpy as np

from test.testdata import generated_trajectory
from trajectopy.core.positions import Positions
from trajectopy.core.trajectory import Trajectory
from trajectopy.io.kml import KML_GX_NAMESPACE, KML_NAMESPACE, ate_kml_fields, kml_color, trajectory_kml_fields
from trajectopy.utils.colors import value_color
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


def _export_ate(result, **kwargs) -> ET.Element:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "ate.kml")
        result.to_kml(path, **kwargs)
        return ET.parse(path).getroot()


def _gui():
    """QtWidgets and a factory of GUI settings stored in a directory (not the user's settings)."""
    try:
        from PySide6 import QtCore, QtWidgets
    except ImportError:  # pragma: no cover - GUI extras not installed
        raise unittest.SkipTest("GUI dependencies not installed") from None

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def settings(directory: str):
        ini = os.path.join(directory, "settings.ini")
        return lambda: QtCore.QSettings(ini, QtCore.QSettings.Format.IniFormat)

    return QtWidgets, settings


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

    def test_color_and_line_width(self):
        root = _export(_georeferenced(), color="#1f77b4", line_width=5)
        colors = {element.text for element in root.findall(".//kml:Style/*/kml:color", NS)}
        self.assertEqual(colors, {"ffb4771f", "80b4771f"})  # line and points, KML order aabbggrr; curtain
        self.assertEqual(root.find(".//kml:LineStyle/kml:width", NS).text, "5")

    def test_kml_color(self):
        self.assertEqual(kml_color("#ff0000"), "ff0000ff")
        self.assertEqual(kml_color("00ff0080"), "8000ff00")
        for invalid in ("red", "#12345", "#gg0000"):
            with self.subTest(color=invalid), self.assertRaises(ValueError):
                kml_color(invalid)

    def test_unknown_coordinate_system(self):
        with self.assertRaises(ValueError):
            _export(generated_trajectory)

    def test_colored_by_a_field(self):
        trajectory = _georeferenced()
        root = _export(trajectory, field="speed", colormap="viridis", color_classes=5)
        legend = root.find("kml:Document/kml:Folder", NS)
        self.assertEqual(legend.find("kml:name", NS).text, "Speed [m/s]")
        self.assertEqual(len(legend.findall("kml:Folder", NS)), 5)
        line_colors = {s.find("kml:LineStyle/kml:color", NS).text for s in root.iter(f"{{{KML_NAMESPACE}}}Style")}
        self.assertIn(kml_color(value_color(0.1, "viridis")), line_colors)
        self.assertEqual(len(root.findall(".//gx:Track/kml:when", NS)), len(trajectory))  # playback

    def test_all_trajectory_fields(self):
        for field in trajectory_kml_fields():
            with self.subTest(field=field.key):
                root = _export(_georeferenced(), field=field.key)
                self.assertTrue(root.findall(".//kml:LineString", NS))

    def test_speed_is_metric(self):
        from trajectopy.io.kml import _speed
        from trajectopy.utils.geometry import gradient_3d

        trajectory = _georeferenced()
        expected = np.linalg.norm(gradient_3d(xyz=trajectory.xyz, tstamps=trajectory.timestamps), axis=1)
        np.testing.assert_allclose(_speed(trajectory), expected, rtol=0.01)  # UTM scale is close to 1

    def test_invalid_options(self):
        for options in (
            {"heights": "orthometric"},
            {"field": "acceleration"},
            {"field": "speed", "colormap": "jet"},
            {"field": "speed", "value_range": (1.0, 1.0)},
            {"field": "speed", "color_classes": 0},
        ):
            with self.subTest(**options), self.assertRaises(ValueError):
                _export(_georeferenced(), **options)


class TestKmlHeights(unittest.TestCase):
    def test_clamped_to_the_ground_by_default(self):
        root = _export(_georeferenced(), extrude=True)
        track = root.find(".//gx:Track", NS)
        self.assertEqual(track.find("kml:altitudeMode", NS).text, "clampToGround")
        self.assertIsNone(track.find("kml:extrude", NS))  # nothing to extrude on the ground

    def test_offset_above_the_ground(self):
        root = _export(_georeferenced(), height_offset=10.0, extrude=True)
        track = root.find(".//gx:Track", NS)
        self.assertEqual(track.find("kml:altitudeMode", NS).text, "relativeToGround")
        self.assertEqual(track.find("kml:extrude", NS).text, "1")
        self.assertEqual(float(track.find("gx:coord", NS).text.split()[2]), 10.0)

    def test_heights_above_sea_level_are_taken_as_they_are(self):
        trajectory = _georeferenced()
        root = _export(trajectory, heights="sea_level", height_offset=1.5)
        track = root.find(".//gx:Track", NS)
        self.assertEqual(track.find("kml:altitudeMode", NS).text, "absolute")
        altitude = float(track.find("gx:coord", NS).text.split()[2])
        self.assertAlmostEqual(altitude, trajectory.xyz[0, 2] + 1.5, places=2)

    def test_colored_line_at_heights(self):
        result = _ate_result()
        root = _export_ate(result, heights="sea_level", extrude=True)
        line = root.find(".//kml:LineString", NS)
        self.assertEqual(line.find("kml:altitudeMode", NS).text, "absolute")
        self.assertEqual(line.find("kml:extrude", NS).text, "1")
        self.assertIsNone(line.find("kml:tessellate", NS))  # straight lines between the poses, at their heights
        _, _, altitude = map(float, line.find("kml:coordinates", NS).text.split()[0].split(","))
        first = np.argmin(result.trajectory.timestamps)
        self.assertAlmostEqual(altitude, result.trajectory.xyz[first, 2], places=2)

    def test_ellipsoidal_heights_are_converted_to_sea_level(self):
        """Google Earth uses heights above the EGM96 geoid (about 47 m above the ellipsoid in Bonn)."""
        trajectory = _georeferenced()
        try:
            root = _export(trajectory, heights="ellipsoidal")
        except ValueError as error:  # no internet connection and the grid is not cached
            self.skipTest(str(error))
        altitude = float(root.find(".//gx:Track/gx:coord", NS).text.split()[2])
        self.assertAlmostEqual(trajectory.xyz[0, 2] - altitude, 47.0, delta=1.0)

    def test_geoid_model_unavailable(self):
        from trajectopy.io import kml

        group = mock.Mock(best_available=False)
        with (
            mock.patch.object(kml, "TransformerGroup", return_value=group),
            self.assertRaisesRegex(ValueError, "geoid model is not available"),
        ):
            _export(_georeferenced(), heights="ellipsoidal")


class TestAteKml(unittest.TestCase):
    def test_fields_depend_on_orientation(self):
        with_rotation = {field.key for field in ate_kml_fields(_ate_result())}
        without_rotation = {field.key for field in ate_kml_fields(_ate_result(with_orientation=False))}
        self.assertTrue({"position", "x", "along_track", "rotation", "yaw"} <= with_rotation)
        self.assertNotIn("rotation", without_rotation)
        self.assertIn("vertical_cross_track", without_rotation)

    def test_line_is_colored_by_the_field(self):
        result = _ate_result()
        root = _export_ate(result, field="position", max_std=0)
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

    def test_fixed_value_range(self):
        root = _export_ate(_ate_result(), value_range=(0.0, 4.0), color_classes=4)
        names = [f.find("kml:name", NS).text for f in root.findall("kml:Document/kml:Folder/kml:Folder", NS)]
        self.assertEqual(names, ["≤ 1.0 m", "1.0 – 2.0 m", "2.0 – 3.0 m", "≥ 3.0 m"])

    def test_playback(self):
        result = _ate_result()
        root = _export_ate(result)
        whens = [element.text for element in root.findall(".//gx:Track/kml:when", NS)]
        self.assertEqual(len(whens), len(result.trajectory))
        self.assertEqual(whens, sorted(whens))

    def test_unavailable_field(self):
        with self.assertRaises(ValueError):
            _export_ate(_ate_result(with_orientation=False), field="yaw")
        with self.assertRaises(ValueError):
            _export_ate(_ate_result(), field="speed")


class TestKmlGui(unittest.TestCase):
    def test_trajectory_export_uses_the_dialog_options(self):
        QtWidgets, settings = _gui()
        from PySide6 import QtGui

        from trajectopy.gui.managers.requests import UIRequest, UIRequestType
        from trajectopy.gui.managers.ui_manager import UIManager
        from trajectopy.gui.models.entries import TrajectoryEntry
        from trajectopy.gui.models.selection import TrajectorySelection
        from trajectopy.gui.views import kml_dialog

        request = UIRequest(
            type=UIRequestType.EXPORT_KML,
            trajectory_selection=TrajectorySelection(entries=[TrajectoryEntry("", _georeferenced())]),
        )

        def choose(dialog):
            dialog.color = QtGui.QColor("#00ff00")
            dialog.offset_spin.setValue(5.0)
            return QtWidgets.QDialog.DialogCode.Accepted

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "track.kml")
            with (
                mock.patch.object(kml_dialog, "app_settings", side_effect=settings(directory)),
                mock.patch.object(kml_dialog.KmlExportDialog, "exec", new=choose),
                mock.patch("trajectopy.gui.managers.ui_manager.save_file_dialog", return_value=path),
            ):
                UIManager().handle_request(request)
            root = ET.parse(path).getroot()
            self.assertEqual(root.find(".//kml:LineStyle/kml:color", NS).text, "ff00ff00")
            self.assertEqual(root.find(".//gx:Track/kml:altitudeMode", NS).text, "relativeToGround")

            os.remove(path)
            with (
                mock.patch.object(kml_dialog, "app_settings", side_effect=settings(directory)),
                mock.patch.object(
                    kml_dialog.KmlExportDialog, "exec", return_value=QtWidgets.QDialog.DialogCode.Rejected
                ),
                mock.patch("trajectopy.gui.managers.ui_manager.save_file_dialog", return_value=path) as save,
            ):
                UIManager().handle_request(request)
            save.assert_not_called()
            self.assertFalse(os.path.exists(path))

    def test_ate_export_uses_the_chosen_field(self):
        QtWidgets, settings = _gui()
        from trajectopy.gui.managers.requests import UIRequest, UIRequestType
        from trajectopy.gui.managers.ui_manager import UIManager
        from trajectopy.gui.models.entries import AbsoluteDeviationEntry
        from trajectopy.gui.models.selection import ResultSelection
        from trajectopy.gui.views import kml_dialog

        request = UIRequest(
            type=UIRequestType.EXPORT_KML,
            result_selection=ResultSelection(entries=[AbsoluteDeviationEntry(deviations=_ate_result())]),
        )

        def choose(dialog):
            dialog.field_box.setCurrentIndex(dialog.field_box.findData("yaw"))
            return QtWidgets.QDialog.DialogCode.Accepted

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ate.kml")
            with (
                mock.patch.object(kml_dialog, "app_settings", side_effect=settings(directory)),
                mock.patch.object(kml_dialog.KmlExportDialog, "exec", new=choose),
                mock.patch("trajectopy.gui.managers.ui_manager.save_file_dialog", return_value=path),
            ):
                UIManager().handle_request(request)
            legend = ET.parse(path).getroot().find("kml:Document/kml:Folder/kml:name", NS)
            self.assertEqual(legend.text, "Yaw deviation [°]")

    def test_dialog_remembers_the_options(self):
        _, settings = _gui()
        from trajectopy.gui.views import kml_dialog

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(kml_dialog, "app_settings", side_effect=settings(directory)),
        ):
            dialog = kml_dialog.KmlExportDialog(fields=trajectory_kml_fields())
            self.assertIsNone(dialog.field)  # one color by default
            self.assertFalse(dialog.colormap_box.isEnabled())
            self.assertFalse(dialog.extrude_check.isEnabled())  # nothing to extrude on the ground
            dialog.field_box.setCurrentIndex(dialog.field_box.findData("speed"))
            dialog.colormap_box.setCurrentIndex(dialog.colormap_box.findData("viridis"))
            dialog.fixed_range_check.setChecked(True)
            dialog.range_max_spin.setValue(20.0)
            dialog.heights_box.setCurrentIndex(dialog.heights_box.findData("ellipsoidal"))
            dialog.extrude_check.setChecked(True)
            dialog.accept()

            dialog = kml_dialog.KmlExportDialog(fields=trajectory_kml_fields())
            options = dialog.options()
            self.assertEqual(options["field"], "speed")
            self.assertEqual(options["colormap"], "viridis")
            self.assertEqual(options["value_range"], (0.0, 20.0))
            self.assertEqual(options["heights"], "ellipsoidal")
            self.assertTrue(options["extrude"])

            # ATE results remember their field separately
            ate_dialog = kml_dialog.KmlExportDialog(fields=ate_kml_fields(_ate_result()), ate=True)
            self.assertEqual(ate_dialog.field, "position")

    def test_dialog_rejects_an_empty_range(self):
        QtWidgets, settings = _gui()
        from trajectopy.gui.views import kml_dialog

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(kml_dialog, "app_settings", side_effect=settings(directory)),
            mock.patch.object(QtWidgets.QMessageBox, "warning") as warning,
        ):
            dialog = kml_dialog.KmlExportDialog(fields=trajectory_kml_fields())
            dialog.field_box.setCurrentIndex(dialog.field_box.findData("speed"))
            dialog.fixed_range_check.setChecked(True)
            dialog.range_min_spin.setValue(5.0)
            dialog.range_max_spin.setValue(5.0)
            dialog.accept()
            warning.assert_called_once()
            self.assertNotEqual(dialog.result(), QtWidgets.QDialog.DialogCode.Accepted)


if __name__ == "__main__":
    unittest.main()
