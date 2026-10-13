import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtGui

    from trajectopy import visualization as vis
    from trajectopy.visualization.axes import index_axis
    from trajectopy.visualization.figure import ensure_app
except ImportError:  # pragma: no cover - plotting extras not installed
    vis = None

import numpy as np

from test.testdata import generated_trajectory, noisy_trajectory
from trajectopy.core.settings import PlotSettings
from trajectopy.processing.evaluation import ate, rpe


@unittest.skipIf(vis is None, "plotting dependencies not installed")
class TestVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_app()
        cls.ate_result, cls.alignment = ate(
            noisy_trajectory.copy(), generated_trajectory.copy(), return_alignment=True
        )
        cls.rpe_result = rpe(noisy_trajectory.copy(), generated_trajectory.copy())

    def _render_all(self, figures: list) -> None:
        """Exports every variant of every figure and shows each variant in an interactive widget."""
        self.assertTrue(figures)
        with tempfile.TemporaryDirectory() as directory:
            paths = vis.export_figures(figures, directory, width=400)
            self.assertEqual(len(paths), sum(len(figure.variants) for figure in figures))
            for path in paths:
                self.assertEqual(QtGui.QImage(path).width(), 400, path)
        for figure in figures:
            widget = figure.create_widget()
            for variant in figure.variants:
                widget.selector.setCurrentText(variant)
            widget.deleteLater()

    def test_trajectory_figures(self):
        figures = vis.trajectory_figures([generated_trajectory, noisy_trajectory])
        self.assertEqual([figure.title for figure in figures], ["Positions", "XYZ", "RPY", "Map"])
        self._render_all(figures)

    def test_deviation_figures_single_and_multiple(self):
        for ate_results, rpe_results in (([self.ate_result], [self.rpe_result]), ([self.ate_result] * 2, [])):
            with self.subTest(results=len(ate_results)):
                self._render_all(vis.deviation_figures(ate_results, rpe_results))

    def test_deviation_figures_only_rpe(self):
        figures = vis.deviation_figures(rpe_results=[self.rpe_result])
        self.assertTrue(all(figure.title.startswith("RPE") for figure in figures))
        self._render_all(figures)

    def test_alignment_figures(self):
        self._render_all(vis.alignment_figures(self.alignment.position_parameters))

    def test_settings_variants(self):
        settings = PlotSettings(
            ate_unit_is_mm=True,
            directed_ate=True,
            ate_remove_above=0.5,
            hist_percentile=0.9,
            smoothing_window=5,
            use_relative_timestamps=True,
            colorbar_max_std=0.0,
        )
        self._render_all(vis.deviation_figures([self.ate_result], [self.rpe_result], settings))
        self._render_all(vis.trajectory_figures([generated_trajectory], settings))

    def test_figures_do_not_modify_results(self):
        pos_dev = self.ate_result.abs_dev.pos_dev.copy()
        xyz = self.ate_result.trajectory.xyz.copy()
        settings = PlotSettings(ate_unit_is_mm=True, ate_remove_above=float(np.median(self.ate_result.pos_dev_comb)))
        self._render_all(vis.deviation_figures([self.ate_result], settings=settings))
        np.testing.assert_array_equal(self.ate_result.abs_dev.pos_dev, pos_dev)
        np.testing.assert_array_equal(self.ate_result.trajectory.xyz, xyz)

    def test_results_without_deviations_left_are_skipped(self):
        with self.assertLogs("trajectopy.visualization.plots", level="WARNING"):
            figures = vis.deviation_figures([self.ate_result], settings=PlotSettings(ate_remove_above=1e-9))
        self.assertEqual(figures, [])

    def test_unix_time_axis(self):
        unix_traj = generated_trajectory.copy()
        unix_traj.timestamps = unix_traj.timestamps + 1.7e9
        self.assertTrue(index_axis([unix_traj]).is_datetime)
        self.assertFalse(index_axis([unix_traj], use_relative_timestamps=True).is_datetime)
        unix_ate = ate(unix_traj, unix_traj.copy(), align=False)
        self._render_all([vis.xyz_figure([unix_traj]), vis.ate_figure([unix_ate])])

    def test_svg_export(self):
        from xml.dom import minidom

        figures = [
            vis.ate_figure([self.ate_result]),
            vis.ate_statistics_figure([self.ate_result]),
            vis.parameter_matrix_figure(self.alignment.position_parameters, correlation=True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for paths in [vis.export_figures(figures, directory, export_format="svg")]:
                for path in paths:
                    document = minidom.parse(path)
                    self.assertTrue(document.getElementsByTagName("path"), path)

    def test_svg_close_path_commands_are_kept(self):
        from xml.dom import minidom

        from trajectopy.visualization import _svg_export

        node = minidom.parseString('<g><path d="M0,0 L1,0 L1,1 Z M2,2 L3,3 Z"/></g>').documentElement
        removed = _svg_export._close_path_positions(node)
        self.assertNotIn("Z", node.getElementsByTagName("path")[0].getAttribute("d"))
        _svg_export._restore_close_path_commands(removed)
        self.assertEqual(node.getElementsByTagName("path")[0].getAttribute("d"), "M0,0 L1,0 L1,1 Z M2,2 L3,3 Z")

    def test_matrix_colors_follow_theme(self):
        from trajectopy.visualization import style

        self.assertEqual(style.readable_text_color(QtGui.QColor("#000000")), "#ffffff")
        self.assertEqual(style.readable_text_color(QtGui.QColor("#ffffff")), "#000000")
        try:
            style.set_theme(dark=True)
            dark_center = style.diverging_colormap().map(0.5, mode="qcolor")
            self._render_all([vis.parameter_matrix_figure(self.alignment.position_parameters, correlation=True)])
        finally:
            style.set_theme(dark=False)
        light_center = style.diverging_colormap().map(0.5, mode="qcolor")
        # the neutral color of the matrix blends into the respective background
        self.assertLess(dark_center.lightness(), 80)
        self.assertGreater(light_center.lightness(), 200)

    def test_plot_window_renders_tabs_lazily(self):
        window = vis.PlotWindow(vis.trajectory_figures([generated_trajectory]))
        self.assertIsNotNone(window.figure_widget(0))
        self.assertIsNone(window.figure_widget(1))
        window.tabs.setCurrentIndex(1)
        self.assertIsNotNone(window.figure_widget(1))
        window.close()


if __name__ == "__main__":
    unittest.main()
