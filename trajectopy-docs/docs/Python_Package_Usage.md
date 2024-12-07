In addition to the GUI, trajectopy can also be used as a Python package. The package provides a set of functions to evaluate and visualize the quality of trajectory. Trajectopy is designed to be easy to use and to provide a high level of customization.


## Absolute Trajectory Error (ATE)

Computing the Absolute Trajectory Error (ATE) between two trajectories is a common evaluation metric in the field of visual odometry and SLAM. The ATE is calculated by aligning the two trajectories and then computing the pose-wise difference between the aligned trajectories.
    
```python
import trajectopy as tpy

gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

ate_result = tpy.ate(trajectory_gt=gt_traj, trajectory_est=est_traj)

```

## Relative Pose Error (RPE)

The Relative Pose Error (RPE) is another common evaluation metric. The RPE measures the relative pose difference between two trajectories at different time or distance intervals.

```python
import trajectopy as tpy

gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

rpe_result = tpy.rpe(trajectory_gt=gt_traj, trajectory_est=est_traj)

```

## Single Trajectory Evaluation Report

The results of a single trajectory comparison can be visualized in a HTML report.

```python
# Import
gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

# Evaluation
ate_result = tpy.ate(trajectory_gt=gt_traj, trajectory_est=est_traj)
rpe_result = tpy.rpe(trajectory_gt=gt_traj, trajectory_est=est_traj)

# Report
report_settings = tpy.ReportSettings(ate_unit_is_mm=False, scatter_marker_size=8)
report = tpy.create_deviation_report(ate_result=ate_result, rpe_result=rpe_result, report_settings=report_settings)
tpy.show_report(report_text=report, filepath="reports/report.html") 
```

## Multi Trajectory Evaluation Report

Multiple trajectory comparisons can be visualized in a single HTML report.

```python
import trajectopy as tpy

# Import
gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj_1 = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
est_traj_2 = tpy.Trajectory.from_file("./example_data/KITTI_SPTAM.traj")

# Evaluation 1
ate_result_1 = tpy.ate(trajectory_gt=gt_traj, trajectory_est=est_traj_1)
rpe_result_1 = tpy.rpe(trajectory_gt=gt_traj, trajectory_est=est_traj_1)

# Evaluation 2
ate_result_2 = tpy.ate(trajectory_gt=gt_traj, trajectory_est=est_traj_2)
rpe_result_2 = tpy.rpe(trajectory_gt=gt_traj, trajectory_est=est_traj_2)

# Report
multi_report = tpy.create_deviation_report(
    ate_result=[ate_result_1, ate_result_2], rpe_result=[rpe_result_1, rpe_result_2]
)
tpy.show_report(report_text=multi_report, filepath="reports/report.html")

```

## Matplotlib Plotting

In addition to the HTML-based plotting, trajectopy also supports Matplotlib-based plotting.
In the below example, all available plotting functions are demonstrated.

```python
import matplotlib.pyplot as plt
import trajectopy as tpy

# importing
gt = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
orb = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
sptam = tpy.Trajectory.from_file("./example_data/KITTI_SPTAM.traj")
tpy.plot_trajectories([gt, orb])

# evaluation
alignment = tpy.estimate_alignment(gt, orb)
ate_result_orb = tpy.ate(gt, orb)
ate_result_sptam = tpy.ate(gt, sptam)
rpe_result = tpy.rpe(gt, orb)

# alignment parameter plots
tpy.plot_covariance_heatmap(alignment.position_parameters)
tpy.plot_correlation_heatmap(alignment.position_parameters)

# ATE plots
tpy.plot_compact_ate_hist(ate_result_orb)
tpy.plot_ate([ate_result_orb, ate_result_sptam])
tpy.plot_ate_bars([ate_result_orb, ate_result_sptam], mode="positions")
tpy.plot_ate_bars([ate_result_orb, ate_result_sptam], mode="rotations")
tpy.plot_ate_edf(ate_result_orb)
tpy.scatter_ate(ate_result_orb)

# RPE plot
tpy.plot_rpe(rpe_result)

plt.show()
```

