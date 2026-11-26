# Getting Started

This tutorial evaluates the trajectories of two visual SLAM systems, ORB-SLAM and S-PTAM, against the ground truth
of a [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) sequence. It takes about ten minutes and
introduces the concepts that the [user guide](user_guide/index.md) explains in detail. The files are in the
[`example_data`](https://github.com/gereon-t/trajectopy/tree/main/example_data) folder of the repository; download
them, or clone the repository, and run the examples in the folder that contains `example_data`.

Install trajectopy with the plotting extra first (see [Installation](installation.md)):

```console
pip install "trajectopy[plot]"
```

## 1. Load the Trajectories

A [`Trajectory`](api/trajectory.md#trajectopy.Trajectory) holds timestamps, positions and orientations. The
files contain a header that describes their columns; trajectopy also recognizes TUM, KITTI and EuRoC files.

```python
import trajectopy as tpy

reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
orb = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
sptam = tpy.Trajectory.from_file("./example_data/KITTI_SPTAM.traj")
print(repr(orb))
```

```text
<Trajectory 'KITTI ORB': 4541 poses with orientations, 3705.098 m, EPSG 0, sorted by time>
```

The trajectory has 4541 poses with orientations, it is 3.7 km long, its positions are given in a local coordinate
system (EPSG code 0) and its poses are ordered by time. Its data are NumPy arrays:

```python
print(orb.timestamps[:3], orb.xyz.shape)
```

```text
[0. 1. 2.] (4541, 3)
```

## 2. Align the Estimate

A SLAM system starts in a coordinate frame of its own. Before its poses can be compared with the reference, the
transformation between both frames is estimated: the *alignment*. By default, it is a rigid body transformation
(translation and rotation), estimated by least squares from all corresponding poses.

```python
alignment = tpy.estimate_alignment(orb, reference)
translation = alignment.position_parameters
print(f"{translation.sim_trans_x.value:.3f} m, {translation.sim_trans_y.value:.3f} m, {translation.sim_trans_z.value:.3f} m")

aligned = tpy.apply_alignment(orb, alignment)
```

```text
-1.363 m, 3.357 m, -0.328 m
```

Like every function of trajectopy, `apply_alignment` returns a new trajectory and leaves `orb` unchanged.

## 3. Absolute Trajectory Error

The *absolute trajectory error* (ATE) compares corresponding poses of the aligned estimate and the reference.
`ate` matches the poses, aligns the estimate (as above) and computes the deviation of every pose:

```python
ate_result = tpy.ate(orb, reference)
print(ate_result.describe()[["pos_mean", "pos_rms", "pos_max"]].round(3))
```

```text
pos_mean    1.153
pos_rms     1.304
pos_max     3.638
Name: KITTI ORB vs. KITTI Ground Truth, dtype: float64
```

On average, the positions of ORB-SLAM deviate by 1.15 m from the reference. The result also holds the deviation of
every pose, along the axes (`pos_dev_x`, …) and along the trajectory (`pos_dev_along`, `pos_dev_cross_h`, …).

## 4. Relative Pose Error

The *relative pose error* (RPE) measures the drift: how much the relative motion between poses that are, e.g.,
100 m apart deviates from that of the reference. It needs no alignment.

```python
rpe_result = tpy.rpe(orb, reference)
print(f"{rpe_result.pos_rpe:.3f} {rpe_result.pos_drift_unit}")
```

```text
0.694 %
```

ORB-SLAM drifts by about 0.7 % of the distance travelled.

## 5. Compare Both Estimates

`summary_table` puts the statistics of several results side by side:

```python
ate_results = [ate_result, tpy.ate(sptam, reference)]
table = tpy.summary_table(ate_results)
print(tpy.format_table(table[["pos_mean [m]", "pos_rms [m]"]], "markdown", precision=3))
```

```text
| name | pos_mean [m] | pos_rms [m] |
| --- | --- | --- |
| KITTI ORB vs. KITTI Ground Truth | 1.153 | 1.304 |
| KITTI SPTAM vs. KITTI Ground Truth | 3.489 | 3.739 |
```

## 6. Plot the Deviations

[`trajectopy.visualization`](api/visualization.md) creates the figures of the GUI. They are shown in an interactive
window with `vis.show(figures)` or saved as images:

```python
from trajectopy import visualization as vis

figures = vis.deviation_figures(ate_results, settings=tpy.PlotSettings(ate_unit_is_mm=True))
paths = vis.export_figures(figures, "plots")
```

## 7. Save the Results

Results and trajectories are saved as text files, which the GUI opens, too:

```python
ate_result.to_file("orb_ate.result")
aligned.to_file("orb_aligned.traj")
tpy.write_report("report.html", ate_results, [rpe_result])
```

`report.html` is a self-contained report with all statistics and figures.

## Next Steps

- The [user guide](user_guide/index.md) explains trajectories, alignment, evaluation, plots and file formats in
  detail.
- The [GUI](user_guide/gui.md) offers everything above without code: run `trajectopy`.
- The [command line](user_guide/command_line.md) evaluates trajectories in scripts and pipelines:
  `trajectopy evaluate example_data/KITTI_ORB.traj --ref example_data/KITTI_gt.traj`.
