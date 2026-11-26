"""The public API of trajectopy.

The public API consists of the names exported by the namespaces below. Everything else, e.g.
``trajectopy.core.trajectory`` or ``trajectopy.processing._lib``, is an implementation detail that may change
in any release. Changing these lists is an API change: adding a name requires a minor release, removing or
renaming one requires a deprecation period and a major release (see the API policy in the documentation).
"""

import importlib
import inspect
import unittest

from trajectopy.utils.deprecation import TrajectopyDeprecationWarning


def plotting_available() -> bool:
    """Whether the plotting extras, which trajectopy.visualization needs, are installed."""
    try:
        import pyqtgraph  # noqa: F401
        import PySide6  # noqa: F401
    except ImportError:
        return False
    return True


def importable_namespaces() -> list[str]:
    """The public namespaces that can be imported with the installed dependencies."""
    return [name for name in PUBLIC_API if name != "trajectopy.visualization" or plotting_available()]


PUBLIC_API = {
    "trajectopy": [
        "ATEResult",
        "AbsoluteTrajectoryDeviations",
        "AlignmentError",
        "AlignmentEstimationSettings",
        "AlignmentParameters",
        "AlignmentParametersError",
        "AlignmentPreprocessing",
        "AlignmentResult",
        "AlignmentSettings",
        "AlignmentStochastics",
        "ApproximationError",
        "ApproximationSettings",
        "ComponentCorrelation",
        "DeviationCorrelation",
        "HelmertTransformation",
        "InterpolationMethod",
        "Leverarm",
        "MatchingMethod",
        "MatchingSettings",
        "PairDistanceUnit",
        "Parameter",
        "PlotSettings",
        "PointSetError",
        "Pose",
        "PositionApproximationMethod",
        "Positions",
        "ProcessingSettings",
        "RPEResult",
        "RelativeComparisonSettings",
        "RelativeTrajectoryDeviations",
        "RotationApproximationMethod",
        "SensorRotationParameters",
        "Settings",
        "Sorting",
        "SortingSettings",
        "TrajectopyError",
        "TrajectopyIOError",
        "Trajectory",
        "TrajectoryError",
        "__version__",
        "adopt_first_pose",
        "align",
        "apply_alignment",
        "approximate",
        "ate",
        "average_trajectories",
        "divide_into_laps",
        "estimate_alignment",
        "format_table",
        "interpolate",
        "match_trajectories",
        "merge_trajectories",
        "rpe",
        "show_versions",
        "sort_spatially",
        "summary_table",
        "write_report",
    ],
    "trajectopy.io": [
        "TrajectoryFormat",
        "detect_format",
        "list_topics",
        "read_alignment_parameters",
        "read_alignment_result",
        "read_ate_result",
        "read_euroc",
        "read_kitti",
        "read_rpe_result",
        "read_sensor_rotation_parameters",
        "read_trajectory",
        "read_trajectory_string",
        "read_tum",
        "trajectories_from_rosbag",
        "trajectory_to_string",
        "write_alignment_parameters",
        "write_alignment_result",
        "write_ate_kml",
        "write_ate_result",
        "write_kml",
        "write_rpe_result",
        "write_sensor_rotation_parameters",
        "write_trajectory",
    ],
    "trajectopy.exceptions": [
        "TrajectopyDeprecationWarning",
        "AlignmentError",
        "AlignmentParametersError",
        "ApproximationError",
        "PointSetError",
        "TrajectopyError",
        "TrajectopyIOError",
        "TrajectoryError",
    ],
    "trajectopy.visualization": [
        "Figure",
        "PlotWindow",
        "SpeedFit",
        "alignment_figures",
        "ate_boxplot_figure",
        "ate_components_figure",
        "ate_correlation_figure",
        "ate_distribution_figure",
        "ate_figure",
        "ate_histogram_figure",
        "ate_map_figure",
        "ate_speed_figure",
        "ate_statistics_figure",
        "deviation_figures",
        "export_figures",
        "parameter_matrix_figure",
        "positions_figure",
        "rpe_figures",
        "rpy_figure",
        "set_theme",
        "show",
        "smoothed_speed",
        "speed_fit",
        "trajectory_figures",
        "trajectory_map_figure",
        "xyz_figure",
    ],
}

DEPRECATED = {
    "trajectopy": ["ComparisonMethod", "IOError", "Rotations"],
    "trajectopy.exceptions": ["IOError"],
    "trajectopy.io": ["HeaderData"],
}


class TestPublicApi(unittest.TestCase):
    def test_exported_names_are_unchanged(self):
        for module_name, expected in PUBLIC_API.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(sorted(module.__all__), sorted(expected))

    def test_exported_names_resolve_and_are_documented(self):
        for module_name in importable_namespaces():
            module = importlib.import_module(module_name)
            for name in PUBLIC_API[module_name]:
                with self.subTest(name=f"{module_name}.{name}"):
                    value = getattr(module, name)
                    if inspect.isclass(value) or inspect.isfunction(value):
                        self.assertTrue(inspect.getdoc(value), f"{module_name}.{name} has no docstring")

    def test_every_name_can_be_imported_first(self):
        """Each public name resolves in a fresh interpreter, without another module imported before it (an
        import cycle shows only then)."""
        import subprocess
        import sys

        import trajectopy

        names = [name for name in PUBLIC_API["trajectopy"] if name != "__version__"]
        first_of_module = {trajectopy._LAZY_IMPORTS[name]: name for name in reversed(names)}
        for module, name in first_of_module.items():
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-c", f"import trajectopy; trajectopy.{name}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_dir_lists_the_public_names(self):
        import trajectopy

        self.assertTrue(set(PUBLIC_API["trajectopy"]) <= set(dir(trajectopy)))

    def test_deprecated_names_warn(self):
        for module_name, names in DEPRECATED.items():
            module = importlib.import_module(module_name)
            for name in names:
                with self.subTest(name=f"{module_name}.{name}"), self.assertWarns(TrajectopyDeprecationWarning):
                    getattr(module, name)

    def test_star_import_does_not_shadow_builtins(self):
        import builtins

        for module_name, names in PUBLIC_API.items():
            with self.subTest(module=module_name):
                self.assertFalse(set(names) & set(dir(builtins)) - {"__version__"})

    def test_keyword_only_options(self):
        """Options after the data arguments are keyword-only, so that they can be reordered or extended."""
        import trajectopy as tpy

        for function in (tpy.align, tpy.apply_alignment, tpy.match_trajectories, tpy.sort_spatially, tpy.approximate):
            with self.subTest(function=function.__name__):
                parameter = inspect.signature(function).parameters["inplace"]
                self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameter.default, False)


if __name__ == "__main__":
    unittest.main()
