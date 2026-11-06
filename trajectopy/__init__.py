"""
Trajectopy - Trajectory Evaluation in Python

A comprehensive package for trajectory analysis, alignment, and comparison.

Key Components
--------------

**Core Classes:**
    - Trajectory: Main class for representing and manipulating trajectories
    - Positions, Rotations: Position and rotation data containers
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
    - Functions that transform a trajectory accept ``inplace``. With ``inplace=False`` the inputs are
      never modified and a new object is returned; with ``inplace=True`` the modified input is returned.
      Always use the return value, e.g. ``traj = tpy.align(traj, ref, inplace=False)``.
    - Evaluation functions (``ate``, ``rpe``, ``estimate_alignment``) never modify their inputs.
    - Settings arguments default to ``None``, which means "use the default settings".

Quick Start
-----------
    >>> import trajectopy as tpy
    >>>
    >>> # Load trajectories
    >>> traj_test = tpy.Trajectory.from_file("test.traj")
    >>> traj_ref = tpy.Trajectory.from_file("reference.traj")
    >>>
    >>> # Align and evaluate
    >>> traj_test_aligned = tpy.align(traj_test, traj_ref)
    >>> ate_result = tpy.ate(traj_test_aligned, traj_ref)
    >>> rpe_result = tpy.rpe(traj_test_aligned, traj_ref)
"""

import importlib
from typing import TYPE_CHECKING

from trajectopy.__version__ import __version__

if TYPE_CHECKING:
    from trajectopy.core.parameters import (
        AlignmentParameters,
        Parameter,
        SensorRotationParameters,
    )
    from trajectopy.core.positions import Positions
    from trajectopy.core.rotations import Rotations
    from trajectopy.core.settings import (
        AlignmentEstimationSettings,
        AlignmentPreprocessing,
        AlignmentSettings,
        AlignmentStochastics,
        ApproximationSettings,
        ComparisonMethod,
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
    from trajectopy.core.trajectory import Trajectory
    from trajectopy.exceptions import (
        AlignmentError,
        AlignmentParametersError,
        ApproximationError,
        IOError,  # Deprecated: use TrajectopyIOError
        PointSetError,
        TrajectopyError,
        TrajectopyIOError,
        TrajectoryError,
    )
    from trajectopy.processing.alignment import align, apply_alignment, estimate_alignment
    from trajectopy.processing.approximation import approximate
    from trajectopy.processing.evaluation import ate, rpe
    from trajectopy.processing.interpolation import interpolate
    from trajectopy.processing.matching import match_trajectories
    from trajectopy.processing.merging import average_trajectories, merge_trajectories
    from trajectopy.processing.sorting import divide_into_laps, sort_spatially
    from trajectopy.results.alignment_result import AlignmentResult
    from trajectopy.results.ate_result import ATEResult
    from trajectopy.results.correlation import DeviationCorrelation
    from trajectopy.results.rpe_result import RPEResult
    from trajectopy.utils.definitions import Sorting

# The public names below are imported on first access (PEP 562): importing numpy, scipy and pandas takes
# seconds on a cold start, which the GUI covers with its splash screen and `trajectopy --version` does not
# need at all.
_LAZY_IMPORTS = {
    "AlignmentParameters": "trajectopy.core.parameters",
    "Parameter": "trajectopy.core.parameters",
    "SensorRotationParameters": "trajectopy.core.parameters",
    "Positions": "trajectopy.core.positions",
    "Rotations": "trajectopy.core.rotations",
    "AlignmentEstimationSettings": "trajectopy.core.settings",
    "AlignmentPreprocessing": "trajectopy.core.settings",
    "AlignmentSettings": "trajectopy.core.settings",
    "AlignmentStochastics": "trajectopy.core.settings",
    "ApproximationSettings": "trajectopy.core.settings",
    "ComparisonMethod": "trajectopy.core.settings",
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
    "Trajectory": "trajectopy.core.trajectory",
    "AlignmentError": "trajectopy.exceptions",
    "AlignmentParametersError": "trajectopy.exceptions",
    "ApproximationError": "trajectopy.exceptions",
    "IOError": "trajectopy.exceptions",
    "PointSetError": "trajectopy.exceptions",
    "TrajectopyError": "trajectopy.exceptions",
    "TrajectopyIOError": "trajectopy.exceptions",
    "TrajectoryError": "trajectopy.exceptions",
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
    "DeviationCorrelation": "trajectopy.results.correlation",
    "RPEResult": "trajectopy.results.rpe_result",
    "Sorting": "trajectopy.utils.definitions",
}


def __getattr__(name: str):
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
    # Core
    "Trajectory",
    "Positions",
    "Rotations",
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
    "ComparisonMethod",
    "MatchingMethod",
    "InterpolationMethod",
    # Results
    "ATEResult",
    "DeviationCorrelation",
    "RPEResult",
    "AlignmentResult",
    # Parameters
    "AlignmentParameters",
    "SensorRotationParameters",
    "Parameter",
    # Processing
    "estimate_alignment",
    "apply_alignment",
    "align",
    "approximate",
    "ate",
    "rpe",
    "interpolate",
    "merge_trajectories",
    "average_trajectories",
    "match_trajectories",
    "sort_spatially",
    "divide_into_laps",
    # Exceptions
    "TrajectopyError",
    "TrajectoryError",
    "AlignmentError",
    "AlignmentParametersError",
    "TrajectopyIOError",
    "IOError",  # Deprecated: use TrajectopyIOError
    "ApproximationError",
    "PointSetError",
    # Sorting
    "Sorting",
]
