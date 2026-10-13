"""Map backgrounds, tested with a fake tile server (no network access)."""

import math
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtGui

    from trajectopy import visualization as vis
    from trajectopy.visualization import basemap
    from trajectopy.visualization.figure import ensure_app
except ImportError:  # pragma: no cover - plotting extras not installed
    basemap = None

import numpy as np

from test.testdata import generated_trajectory
from trajectopy.core.positions import Positions
from trajectopy.core.settings import PlotSettings
from trajectopy.core.trajectory import Trajectory

# Bonn, Germany in UTM zone 32N
UTM_OFFSET = np.array([364800.0, 5621500.0, 60.0])


def _georeferenced(trajectory: Trajectory) -> Trajectory:
    geo = trajectory.copy()
    geo.positions = Positions(xyz=trajectory.xyz + UTM_OFFSET, epsg=25832)
    return geo


def _slippy_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Reference implementation of the standard web map tile index (OpenStreetMap wiki)."""
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


@unittest.skipIf(basemap is None, "plotting dependencies not installed")
class TestBasemap(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_app()

    def setUp(self) -> None:
        self.requested: list[tuple[str, int, int, int]] = []

        def fake_tile(style_name: str, zoom: int, x: int, y: int) -> QtGui.QImage:
            self.requested.append((style_name, zoom, x, y))
            tile = QtGui.QImage(basemap.TILE_SIZE, basemap.TILE_SIZE, QtGui.QImage.Format.Format_RGB32)
            tile.fill(QtGui.QColor("#00ff00"))
            return tile

        patcher = mock.patch.object(basemap, "_fetch_tile", side_effect=fake_tile)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_requested_tiles_match_standard_tile_indices(self):
        lon, lat = 7.0970, 50.7320
        from pyproj import Transformer

        mercator = np.array([Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)])
        map_image = basemap.load_map_image(mercator)
        self.assertIsNotNone(map_image)
        zoom = self.requested[0][1]
        self.assertIn(_slippy_tile(lon, lat, zoom), {(x, y) for _, _, x, y in self.requested})
        # the stitched image covers the point
        self.assertLess(map_image.left, mercator[0, 0])
        self.assertGreater(map_image.right, mercator[0, 0])
        self.assertLess(map_image.bottom, mercator[0, 1])
        self.assertGreater(map_image.top, mercator[0, 1])

    def test_map_frame_preserves_local_distances(self):
        trajectory = _georeferenced(generated_trajectory)
        mercator = basemap.web_mercator_xy(trajectory)
        local = basemap.MapFrame.around(mercator).to_local(mercator)
        original = np.linalg.norm(np.diff(trajectory.xyz[:, :2], axis=0), axis=1).sum()
        on_map = np.linalg.norm(np.diff(local, axis=0), axis=1).sum()
        self.assertAlmostEqual(on_map / original, 1.0, delta=2e-3)  # UTM scale factor and frame approximation

    def test_export_contains_map(self):
        figure = vis.positions_figure([_georeferenced(generated_trajectory)], PlotSettings(background_map=True))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "map.png")
            figure.export(path, width=400)
            image = QtGui.QImage(path)
        self.assertTrue(self.requested)
        green = sum(
            QtGui.QColor(image.pixel(x, y)).green() > 200 and QtGui.QColor(image.pixel(x, y)).red() < 50
            for x in range(0, image.width(), 10)
            for y in range(0, image.height(), 10)
        )
        self.assertGreater(green, 100, "map tiles are not visible in the export")

    def test_map_styles(self):
        trajectory = _georeferenced(generated_trajectory)
        for map_style in basemap.TILE_SOURCES:
            with self.subTest(map_style=map_style), tempfile.TemporaryDirectory() as directory:
                settings = PlotSettings(background_map=True, map_style=map_style)
                vis.ate_map_figure([_ate(trajectory)], settings).export(os.path.join(directory, "m.png"), width=300)
                self.assertEqual(self.requested[-1][0], map_style)

    def test_not_georeferenced_falls_back_without_map(self):
        figure = vis.positions_figure([generated_trajectory], PlotSettings(background_map=True))
        with self.assertLogs("trajectopy.visualization.plots", level="WARNING"), tempfile.TemporaryDirectory() as d:
            figure.export(os.path.join(d, "plain.png"), width=300)
        self.assertEqual(self.requested, [])

    def test_interactive_map_loads_in_background(self):
        figure = vis.positions_figure([_georeferenced(generated_trajectory)], PlotSettings(background_map=True))
        widget = figure.create_widget()
        layers = [item for item in widget.graphics.ci.getItem(0, 0).items if isinstance(item, basemap.MapLayer)]
        self.assertEqual(len(layers), 1)
        app = ensure_app()
        for _ in range(200):
            app.processEvents()
            if not layers[0].pixmap().isNull():
                break
            import time

            time.sleep(0.01)
        self.assertFalse(layers[0].pixmap().isNull())
        widget.deleteLater()

    def test_late_tiles_for_closed_plot_are_ignored(self):
        figure = vis.positions_figure([_georeferenced(generated_trajectory)], PlotSettings(background_map=True))
        widget = figure.create_widget()
        layer = next(i for i in widget.graphics.ci.getItem(0, 0).items if isinstance(i, basemap.MapLayer))
        widget.graphics.clear()  # deletes the layer's C++ object
        mercator = basemap.web_mercator_xy(_georeferenced(generated_trajectory))
        layer.set_map(basemap.load_map_image(mercator))  # must not raise
        widget.deleteLater()


def _ate(trajectory: Trajectory):
    from trajectopy.processing.evaluation import ate

    return ate(trajectory, trajectory.copy(), align=False)


if __name__ == "__main__":
    unittest.main()
