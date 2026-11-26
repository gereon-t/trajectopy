"""
Trajectopy - Trajectory Evaluation in Python

A comprehensive package for trajectory analysis, alignment, and comparison.

Key Components
--------------

**Core Classes:**
    - Trajectory: Main class for representing and manipulating trajectories
    - Positions: Position container with EPSG code; orientations are scipy Rotation objects
    - Settings: Configuration for trajectory processing

**Processing Functions:**
    - align: Align trajectories using various methods
    - ate: Absolute Trajectory Error evaluation
    - rpe: Relative Pose Error evaluation
    - match_trajectories: Match poses between trajectories
    - interpolate: Temporal interpolation of trajectories
    - merge_trajectories, average_trajectories: Combine multiple trajectories
    - sort_spatially: Spatial sorting of trajectory poses
    - approximate: Smoothing by piecewise cubic approximation

**Result Classes:**
    - ATEResult: Absolute Trajectory Error results
    - RPEResult: Relative Pose Error results
    - AlignmentResult: Alignment parameters and statistics

**Parameters:**
    - AlignmentParameters: 3D transformation parameters (translation, rotation, scale)
    - SensorRotationParameters: Sensor orientation parameters
    - Parameter: Base class for stochastic parameters with variance

**Visualization** (``pip install "trajectopy[plot]"``):
    - trajectopy.visualization: Figures of trajectories, ATE/RPE results and alignment parameters,
      shown in an interactive window (``show``) or exported to PNG/SVG (``export_figures``).

**File I/O:**
    - trajectopy.io: Reading and writing of trajectories, results and alignment parameters.
      ``Trajectory.from_file`` / ``to_file`` and the result classes' equivalents delegate to it.

Conventions
-----------
    - Functions and methods that transform a trajectory return a new object and never modify their
      inputs. Pass the keyword-only argument ``inplace=True`` to modify the input instead; the modified
      input is returned then, too. Always use the return value, e.g. ``traj = tpy.align(traj, ref)``.
    - Every processing function takes one ``settings`` argument: its own settings type or the complete
      ``ProcessingSettings``. ``None`` means the default settings.
    - Options (e.g. ``MatchingMethod``) can be given as strings, e.g. ``method="nearest_temporal"``.
    - File functions accept ``str`` and ``os.PathLike`` and overwrite existing files by default.
    - The public API consists of the names exported here and by ``trajectopy.io``,
      ``trajectopy.visualization`` and ``trajectopy.exceptions``; see the API policy in the documentation.

Quick Start
-----------
    >>> import trajectopy as tpy
    >>>
    >>> # Load trajectories
    >>> traj_test = tpy.Trajectory.from_file("test.traj")
    >>> traj_ref = tpy.Trajectory.from_file("reference.traj")
    >>>
    >>> # Evaluate (the ATE aligns the trajectories first)
    >>> settings = tpy.ProcessingSettings()
    >>> ate_result = tpy.ate(traj_test, traj_ref, settings)
    >>> rpe_result = tpy.rpe(traj_test, traj_ref, settings)
    >>> traj_test_aligned = tpy.apply_alignment(traj_test, ate_result.alignment)
"""

import importlib
import logging
from typing import TYPE_CHECKING

from trajectopy.__version__ import __version__

# trajectopy logs, but leaves the configuration of the output to the application
logging.getLogger(__name__).addHandler(logging.NullHandler())

if TYPE_CHECKING:
    from trajectopy.core.parameters import (
        AlignmentParameters,
        HelmertTransformation,
        Leverarm,
        Parameter,
        SensorRotationParameters,
    )
    from trajectopy.core.positions import Positions
    from trajectopy.core.settings import (
        AlignmentEstimationSettings,
        AlignmentPreprocessing,
        AlignmentSettings,
        AlignmentStochastics,
        ApproximationSettings,
        InterpolationMethod,
        MatchingMethod,
        MatchingSettings,
        PairDistanceUnit,
        PlotSettings,
        PositionApproximationMethod,
        ProcessingSettings,
        RelativeComparisonSettings,
        RotationApproximationMethod,
        Settings,
        SortingSettings,
    )
    from trajectopy.core.trajectory import Pose, Trajectory
    from trajectopy.exceptions import (
        AlignmentError,
        AlignmentParametersError,
        ApproximationError,
        PointSetError,
        TrajectopyError,
        TrajectopyIOError,
        TrajectoryError,
    )
    from trajectopy.processing.alignment import adopt_first_pose, align, apply_alignment, estimate_alignment
    from trajectopy.processing.approximation import approximate
    from trajectopy.processing.evaluation import ate, rpe
    from trajectopy.processing.interpolation import interpolate
    from trajectopy.processing.matching import match_trajectories
    from trajectopy.processing.merging import average_trajectories, merge_trajectories
    from trajectopy.processing.sorting import divide_into_laps, sort_spatially
    from trajectopy.report import write_report
    from trajectopy.results.alignment_result import AlignmentResult
    from trajectopy.results.ate_result import AbsoluteTrajectoryDeviations, ATEResult
    from trajectopy.results.correlation import ComponentCorrelation, DeviationCorrelation
    from trajectopy.results.rpe_result import RelativeTrajectoryDeviations, RPEResult
    from trajectopy.results.summary import format_table, summary_table
    from trajectopy.utils.definitions import Sorting
    from trajectopy.utils.versions import show_versions

# The public names below are imported on first access (PEP 562): importing numpy, scipy and pandas takes
# seconds on a cold start, which the GUI covers with its splash screen and `trajectopy --version` does not
# need at all.
_LAZY_IMPORTS = {
    "AlignmentParameters": "trajectopy.core.parameters",
    "HelmertTransformation": "trajectopy.core.parameters",
    "Leverarm": "trajectopy.core.parameters",
    "Parameter": "trajectopy.core.parameters",
    "SensorRotationParameters": "trajectopy.core.parameters",
    "Positions": "trajectopy.core.positions",
    "AlignmentEstimationSettings": "trajectopy.core.settings",
    "AlignmentPreprocessing": "trajectopy.core.settings",
    "AlignmentSettings": "trajectopy.core.settings",
    "AlignmentStochastics": "trajectopy.core.settings",
    "ApproximationSettings": "trajectopy.core.settings",
    "InterpolationMethod": "trajectopy.core.settings",
    "MatchingMethod": "trajectopy.core.settings",
    "MatchingSettings": "trajectopy.core.settings",
    "PairDistanceUnit": "trajectopy.core.settings",
    "PlotSettings": "trajectopy.core.settings",
    "PositionApproximationMethod": "trajectopy.core.settings",
    "ProcessingSettings": "trajectopy.core.settings",
    "RelativeComparisonSettings": "trajectopy.core.settings",
    "RotationApproximationMethod": "trajectopy.core.settings",
    "Settings": "trajectopy.core.settings",
    "SortingSettings": "trajectopy.core.settings",
    "Pose": "trajectopy.core.trajectory",
    "Trajectory": "trajectopy.core.trajectory",
    "AlignmentError": "trajectopy.exceptions",
    "AlignmentParametersError": "trajectopy.exceptions",
    "ApproximationError": "trajectopy.exceptions",
    "PointSetError": "trajectopy.exceptions",
    "TrajectopyError": "trajectopy.exceptions",
    "TrajectopyIOError": "trajectopy.exceptions",
    "TrajectoryError": "trajectopy.exceptions",
    "adopt_first_pose": "trajectopy.processing.alignment",
    "align": "trajectopy.processing.alignment",
    "apply_alignment": "trajectopy.processing.alignment",
    "estimate_alignment": "trajectopy.processing.alignment",
    "approximate": "trajectopy.processing.approximation",
    "ate": "trajectopy.processing.evaluation",
    "rpe": "trajectopy.processing.evaluation",
    "interpolate": "trajectopy.processing.interpolation",
    "match_trajectories": "trajectopy.processing.matching",
    "average_trajectories": "trajectopy.processing.merging",
    "merge_trajectories": "trajectopy.processing.merging",
    "divide_into_laps": "trajectopy.processing.sorting",
    "sort_spatially": "trajectopy.processing.sorting",
    "AlignmentResult": "trajectopy.results.alignment_result",
    "ATEResult": "trajectopy.results.ate_result",
    "AbsoluteTrajectoryDeviations": "trajectopy.results.ate_result",
    "ComponentCorrelation": "trajectopy.results.correlation",
    "DeviationCorrelation": "trajectopy.results.correlation",
    "RPEResult": "trajectopy.results.rpe_result",
    "RelativeTrajectoryDeviations": "trajectopy.results.rpe_result",
    "summary_table": "trajectopy.results.summary",
    "format_table": "trajectopy.results.summary",
    "write_report": "trajectopy.report",
    "Sorting": "trajectopy.utils.definitions",
    "show_versions": "trajectopy.utils.versions",
}


# Deprecated top-level names: name -> (module, attribute, message)
_DEPRECATED = {
    "IOError": ("trajectopy.exceptions", "TrajectopyIOError", "use trajectopy.TrajectopyIOError instead"),
    "ComparisonMethod": ("trajectopy.core.settings", "ComparisonMethod", "it is an internal detail of the file I/O"),
    "Rotations": (
        "scipy.spatial.transform",
        "Rotation",
        "use scipy.spatial.transform.Rotation, which trajectopy uses for orientations",
    ),
}


def __getattr__(name: str):
    if name in _DEPRECATED:
        from trajectopy.utils.deprecation import warn_deprecated

        deprecated_module, attribute, message = _DEPRECATED[name]
        warn_deprecated(f"trajectopy.{name} is deprecated, {message}.")
        return getattr(importlib.import_module(deprecated_module), attribute)
    module = _LAZY_IMPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # the next access does not go through __getattr__ anymore
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    # Version
    "__version__",
    "show_versions",
    # Core
    "Trajectory",
    "Pose",
    "Positions",
    "Settings",
    "ProcessingSettings",
    "AlignmentSettings",
    "AlignmentEstimationSettings",
    "AlignmentPreprocessing",
    "AlignmentStochastics",
    "MatchingSettings",
    "RotationApproximationMethod",
    "PositionApproximationMethod",
    "ApproximationSettings",
    "RelativeComparisonSettings",
    "SortingSettings",
    "PlotSettings",
    "PairDistanceUnit",
    "MatchingMethod",
    "InterpolationMethod",
    # Results
    "ATEResult",
    "AbsoluteTrajectoryDeviations",
    "DeviationCorrelation",
    "ComponentCorrelation",
    "RPEResult",
    "RelativeTrajectoryDeviations",
    "AlignmentResult",
    # Parameters
    "AlignmentParameters",
    "HelmertTransformation",
    "Leverarm",
    "SensorRotationParameters",
    "Parameter",
    # Processing
    "estimate_alignment",
    "apply_alignment",
    "align",
    "adopt_first_pose",
    "approximate",
    "ate",
    "rpe",
    "interpolate",
    "merge_trajectories",
    "average_trajectories",
    "match_trajectories",
    "sort_spatially",
    "divide_into_laps",
    # Summaries and reports
    "summary_table",
    "format_table",
    "write_report",
    # Exceptions
    "TrajectopyError",
    "TrajectoryError",
    "AlignmentError",
    "AlignmentParametersError",
    "TrajectopyIOError",
    "ApproximationError",
    "PointSetError",
    # Sorting
    "Sorting",
]
