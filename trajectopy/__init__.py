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
    - approximate_cubic: Cubic approximation of trajectory data

**Result Classes:**
    - ATEResult: Absolute Trajectory Error results
    - RPEResult: Relative Pose Error results
    - AlignmentResult: Alignment parameters and statistics

**Parameters:**
    - AlignmentParameters: 3D transformation parameters (translation, rotation, scale)
    - SensorRotationParameters: Sensor orientation parameters
    - Parameter: Base class for stochastic parameters with variance

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

from trajectopy.__version__ import __version__
from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import (
    AlignmentEstimationSettings,
    AlignmentPreprocessing,
    AlignmentSettings,
    AlignmentStochastics,
    ApproximationSettings,
    ComparisonMethod,
    ExportSettings,
    InterpolationMethod,
    MatchingMethod,
    MatchingSettings,
    MPLPlotSettings,
    PairDistanceUnit,
    PlotBackend,
    ProcessingSettings,
    RelativeComparisonSettings,
    ReportSettings,
    Settings,
    SortingSettings,
)
from trajectopy.core.trajectory import Trajectory
from trajectopy.exceptions import (
    AlignmentError,
    AlignmentParametersError,
    ApproximationError,
    IOError,
    PointSetError,
    TrajectopyError,
    TrajectoryError,
)
from trajectopy.processing.alignment import align, apply_alignment, estimate_alignment
from trajectopy.processing.approximation import approximate_cubic
from trajectopy.processing.evaluation import ate, rpe
from trajectopy.processing.interpolation import interpolate
from trajectopy.processing.lib.alignment.parameters import (
    AlignmentParameters,
    Parameter,
    SensorRotationParameters,
)
from trajectopy.processing.matching import match_trajectories
from trajectopy.processing.merging import average_trajectories, merge_trajectories
from trajectopy.processing.sorting import sort_spatially
from trajectopy.results.alignment_result import AlignmentResult
from trajectopy.results.ate_result import ATEResult
from trajectopy.results.rpe_result import RPEResult

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
    "ReportSettings",
    "ApproximationSettings",
    "RelativeComparisonSettings",
    "MPLPlotSettings",
    "SortingSettings",
    "ExportSettings",
    "PairDistanceUnit",
    "ComparisonMethod",
    "MatchingMethod",
    "InterpolationMethod",
    "PlotBackend",
    # Results
    "ATEResult",
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
    "approximate_cubic",
    "ate",
    "rpe",
    "interpolate",
    "merge_trajectories",
    "average_trajectories",
    "match_trajectories",
    "sort_spatially",
    # Exceptions
    "TrajectopyError",
    "TrajectoryError",
    "AlignmentError",
    "AlignmentParametersError",
    "IOError",
    "ApproximationError",
    "PointSetError",
]
