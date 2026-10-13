# API Reference

Welcome to the Trajectopy API reference. This section provides detailed documentation for all public modules, classes, and functions.

## Core Modules

- **[Trajectory](core/trajectory.md)** - Main trajectory class for representing and manipulating trajectory data
- **[Positions](core/positions.md)** - Position data container
- **[Rotations](core/rotations.md)** - Rotation data container  
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

- **[Trajectory I/O](io/trajectory_io.md)** - Read and write trajectories (ASCII, KML)
- **[Results I/O](io/results_io.md)** - Read and write ATE/RPE/alignment results and parameters

## Results

- **[ATEResult](results/ate_result.md)** - Absolute Trajectory Error results
- **[RPEResult](results/rpe_result.md)** - Relative Pose Error results
- **[AlignmentResult](results/alignment_result.md)** - Alignment parameters and statistics

## Visualization

- **[Figures](visualization/figures.md)** - Plots of trajectories, ATE/RPE results and alignment parameters
- **[Showing & Exporting](visualization/display.md)** - Interactive plot window and image export
