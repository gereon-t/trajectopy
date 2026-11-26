# API Reference

Welcome to the Trajectopy API reference. This section provides detailed documentation for all public modules, classes, and functions.

All names documented here are available from the public namespaces `trajectopy`, `trajectopy.io`,
`trajectopy.visualization` and `trajectopy.exceptions`; import them from there rather than from the modules
they are defined in. See the [API policy](../api_policy.md) for the conventions and the stability guarantees.

## Core Modules

- **[Trajectory](core/trajectory.md)** - Main trajectory class for representing and manipulating trajectory data
- **[Positions](core/positions.md)** - Position data container
- **[Settings](core/settings.md)** - Configuration classes for trajectory processing
- **[Parameters](core/parameters.md)** - Alignment parameters and stochastic models

## Processing Functions

- **[Alignment](processing/alignment.md)** - Trajectory alignment functions
- **[Evaluation](processing/evaluation.md)** - ATE and RPE evaluation metrics
- **[Interpolation](processing/interpolation.md)** - Temporal interpolation
- **[Matching](processing/matching.md)** - Pose matching between trajectories
- **[Merging](processing/merging.md)** - Merge and average multiple trajectories
- **[Sorting](processing/sorting.md)** - Spatial sorting of trajectories
- **[Approximation](processing/approximation.md)** - Trajectory approximation algorithms

## File I/O

- **[TUM, KITTI & EuRoC](io/formats.md)** - `trajectopy.io.read_trajectory` and `write_trajectory` for all formats
- **[Trajectory I/O](io/trajectory_io.md)** - The trajectopy ASCII format
- **[Results I/O](io/results_io.md)** - Read and write ATE/RPE/alignment results and parameters
- **[Google Earth (KML)](io/kml.md)** - Export trajectories and ATE results to Google Earth
- **[ROS Bags](io/rosbag.md)** - Read trajectories from ROS1 and ROS2 bag files

## Results

- **[ATEResult](results/ate_result.md)** - Absolute Trajectory Error results
- **[RPEResult](results/rpe_result.md)** - Relative Pose Error results
- **[AlignmentResult](results/alignment_result.md)** - Alignment parameters and statistics
- **[DeviationCorrelation](results/correlation.md)** - Temporal correlation of ATE deviations

## Visualization

- **[Figures](visualization/figures.md)** - Plots of trajectories, ATE/RPE results and alignment parameters
- **[Showing & Exporting](visualization/display.md)** - Interactive plot window, image export and plot theme
