# User Guide

This guide covers the core functionality of Trajectopy: loading data, alignment, evaluation, and visualization. All functionality is available both via the Python API and the GUI application. The examples below use the Python API.

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
from trajectopy import Trajectory, ProcessingSettings, estimate_alignment, apply_alignment

gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

# default settings
processing_settings = ProcessingSettings()

# Estimate alignment
alignment_result = estimate_alignment(
    trajectory=est_traj,
    other=gt_traj,
    alignment_settings=processing_settings.alignment,
    matching_settings=processing_settings.matching,
)

# Apply alignment
est_traj_aligned = apply_alignment(est_traj, alignment_result)
```

The function `align` combines these two steps:

```python
from trajectopy import Trajectory, align

gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

est_traj_aligned = align(trajectory=est_traj, other=gt_traj, inplace=False)
```

## 3. Evaluation

Trajectopy provides standard metrics for trajectory evaluation.

### Absolute Trajectory Error (ATE)

ATE measures the global consistency of the trajectory by building differences between corresponding poses. Note that `ate()` already includes alignment by default. This can be disabled by setting the function parameter `align=False`.

```python
from trajectopy import Trajectory, ate, ProcessingSettings

gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

# default settings
processing_settings = ProcessingSettings()

ate_result = ate(other=gt_traj, trajectory=est_traj, processing_settings=processing_settings)
print(f"Position ATE: {ate_result.pos_ate}")
```

### Relative Pose Error (RPE)

RPE measures the pose drift over fixed time or distance intervals.

```python
from trajectopy import Trajectory, rpe, ProcessingSettings

gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

# default settings
processing_settings = ProcessingSettings()

rpe_result = rpe(other=gt_traj, trajectory=est_traj, processing_settings=processing_settings)
print(f"Position RPE: {rpe_result.pos_rpe}")
```

## 4. Visualization & Reporting

### Plotly Reports

You can generate interactive HTML reports using Plotly.

#### Trajectory Visualization

```python
from trajectopy import Trajectory, ReportSettings
from trajectopy.visualization import plotly_reports

gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

report_settings = ReportSettings(scatter_axis_order="xy", ate_unit_is_mm=False)
traj_report = plotly_reports.create_trajectory_report(
    trajectories=[gt_traj, est_traj], report_settings=report_settings
)
plotly_reports.show_report(traj_report, filepath="reports/report.html")
```

#### Evaluation Report

```python
from trajectopy import ProcessingSettings, ReportSettings, Trajectory, ate, rpe
from trajectopy.visualization import plotly_reports

gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

# default settings
processing_settings = ProcessingSettings()

ate_result = ate(other=gt_traj, trajectory=est_traj, processing_settings=processing_settings)
rpe_result = rpe(other=gt_traj, trajectory=est_traj, processing_settings=processing_settings)

report_settings = ReportSettings(ate_unit_is_mm=False, scatter_marker_size=8)
report = plotly_reports.create_deviation_report(
    ate_result=ate_result, rpe_result=rpe_result, report_settings=report_settings
)
plotly_reports.show_report(report_text=report, filepath="reports/report.html")
```

#### Alignment Report

```python
from trajectopy import ProcessingSettings, Trajectory, estimate_alignment
from trajectopy.visualization import plotly_reports

gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

# default settings
processing_settings = ProcessingSettings()

alignment_result = estimate_alignment(
    trajectory=est_traj,
    other=gt_traj,
    alignment_settings=processing_settings.alignment,
    matching_settings=processing_settings.matching,
)

report = plotly_reports.create_alignment_report(
    alignment_parameters=alignment_result.position_parameters,
    name=alignment_result.name,
)
plotly_reports.show_report(report_text=report, filepath="reports/report.html")
```

### Matplotlib Visualization

Additionally, you can visualize trajectories directly using Matplotlib.

```python
import matplotlib.pyplot as plt

from trajectopy import Trajectory, ate, estimate_alignment, rpe
from trajectopy.visualization import mpl_plots

gt = Trajectory.from_file("./example_data/KITTI_gt.traj")
orb = Trajectory.from_file("./example_data/KITTI_ORB.traj")
sptam = Trajectory.from_file("./example_data/KITTI_SPTAM.traj")
mpl_plots.plot_trajectories([gt, orb])

alignment_result = estimate_alignment(gt, orb)
ate_result_orb = ate(gt, orb)
ate_result_sptam = ate(gt, sptam)
rpe_result = rpe(gt, orb)

# All available plotting functions
mpl_plots.plot_covariance_heatmap(alignment_result.position_parameters)
mpl_plots.plot_correlation_heatmap(alignment_result.position_parameters)

mpl_plots.plot_ate_3d([ate_result_orb, ate_result_sptam])
mpl_plots.plot_compact_ate_hist(ate_result_orb)
mpl_plots.plot_ate([ate_result_orb, ate_result_sptam])
mpl_plots.plot_ate_bars([ate_result_orb, ate_result_sptam], mode="positions")
mpl_plots.plot_ate_bars([ate_result_orb, ate_result_sptam], mode="rotations")

mpl_plots.plot_ate_edf(ate_result_orb)

mpl_plots.scatter_ate(ate_result_orb)

mpl_plots.plot_rpe(rpe_result)

plt.show()
```

## 5. GUI Application

For a graphical interface, simply run:

```bash
trajectopy
```

This opens the application where you can drag & drop files, align them, and view results interactively.
