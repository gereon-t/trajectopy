# User Guide

This guide covers the core functionality of Trajectopy: loading data, alignment, evaluation, and visualization.

## 1. Loading Data

Trajectories are loaded using the `Trajectory` class. All ASCII text files are supported as long as they meet the requirements described in [File Formats](./file_formats.md).

```python
from trajectopy import Trajectory

# Load from file
traj = Trajectory.from_file("path/to/trajectory.traj")
```

## 2. Alignment

To compare two trajectories (e.g., estimated vs. ground truth), they often need to be aligned first.

```python
from trajectopy import ProcessingSettings, estimate_alignment, apply_alignment

# Settings
settings = ProcessingSettings()

# Estimate alignment
alignment_result = estimate_alignment(
    trajectory=traj_est,
    other=traj_ref,
    alignment_settings=settings.alignment,
    matching_settings=settings.matching,
)

# Apply alignment
traj_est_aligned = apply_alignment(traj_est, alignment_result)
```

## 3. Evaluation

Trajectopy provides standard metrics for trajectory evaluation.

### Absolute Trajectory Error (ATE)

ATE measures the global consistency of the trajectory.

```python
from trajectopy import ate

ate_result = ate(other=traj_ref, trajectory=traj_est_aligned)
print(f"Position ATE: {ate_result.pos_ate}")
```

### Relative Pose Error (RPE)

RPE measures the local accuracy (drift) over fixed time or distance intervals.

```python
from trajectopy import rpe

rpe_result = rpe(other=traj_ref, trajectory=traj_est_aligned)
print(f"Position RPE: {rpe_result.pos_rpe}")
```

## 4. Visualization & Reporting

You can generate interactive HTML reports using Plotly.

```python
from trajectopy.visualization import plotly_reports

# Create a report
report_html = plotly_reports.create_deviation_report(
    ate_result=ate_result,
    rpe_result=rpe_result
)

# Save and show
plotly_reports.show_report(report_text=report_html, filepath="./report.html")
```

## 5. GUI Application

For a graphical interface, simply run:

```bash
trajectopy
```

This opens the application where you can drag & drop files, align them, and view results interactively.
