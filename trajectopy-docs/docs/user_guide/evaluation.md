# Evaluation

Trajectopy evaluates a trajectory against a reference with two complementary metrics:

- the **absolute trajectory error (ATE)**: the deviations of corresponding poses after an alignment. It measures
  the global accuracy of a trajectory.
- the **relative pose error (RPE)**: the deviations of the relative motion between poses that are a given
  distance or time apart. It measures the drift and needs no alignment.

All deviations are *evaluated trajectory minus reference*.

## Absolute Trajectory Error (ATE)

```python
import numpy as np
import trajectopy as tpy

reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

ate_result = tpy.ate(estimate, reference)
print(ate_result)
```

```text
<ATEResult 'KITTI ORB vs. KITTI Ground Truth': 4541 poses, position ATE 1.1533 m, rotation ATE 0.6204 deg>
```

`ate` matches the poses of both trajectories, [aligns](alignment.md) the estimate to the reference and computes
the deviations of each pose. The alignment is kept in the result (`ate_result.alignment`); pass `align=False` to
evaluate the trajectory as it is, e.g. if it is already given in the frame of the reference (then
`ate_result.alignment` holds identity parameters).

The settings of matching and alignment are part of the [processing settings](settings.md#processing-settings);
`ate` takes the complete `ProcessingSettings`:

```python
settings = tpy.ProcessingSettings()
settings.alignment.estimation_settings.scale = True
settings.matching.method = tpy.MatchingMethod.NEAREST_TEMPORAL

ate_result = tpy.ate(estimate, reference, settings)
```

### Deviations

The [`ATEResult`](../api/results.md#trajectopy.ATEResult) holds the deviation of every pose, both along the
axes of the coordinate system and along the trajectory:

| Attribute | Deviation |
| --- | --- |
| `pos_dev_x`, `pos_dev_y`, `pos_dev_z` | Position deviations along x, y and z [m] |
| `pos_dev_along` | Along-track: in the direction of travel; positive if the trajectory is ahead [m] |
| `pos_dev_cross_h` | Horizontal cross-track: positive to the right of the direction of travel [m] |
| `pos_dev_cross_v` | Vertical cross-track: positive upwards [m] |
| `pos_dev_comb` | 3D position deviation (length of the deviation vector) [m] |
| `rot_dev_x`, `rot_dev_y`, `rot_dev_z` | Roll, pitch and yaw of the deviation rotation [rad] |
| `rot_dev_comb` | Rotation angle of the deviation rotation [rad] |

The along- and cross-track directions are taken from the orientations of the reference, or, if it has none,
from those of the evaluated trajectory, or, if neither has orientations, from the direction of travel (see the
[frame definition](settings.md#ate-frame-definition)).

!!! warning "Body frame"
    With orientations, the along-track direction is the x axis of the body frame. If the x axis of the
    trajectories does not point forward, the along- and cross-track deviations are mixed up. This is the case
    for the KITTI example data used on this page, whose body y axis points forward: there, `pos_dev_along` is
    actually a cross-track deviation.

`to_dataframe()` returns the deviations together with the timestamps, path lengths (`arc_lengths`) and
positions of the evaluated trajectory. In it, the rotation deviations are quaternions (`rot_dev_x/y/z/w`),
unlike the attributes `rot_dev_x/y/z`, which are roll, pitch and yaw.

### Statistics

```python
print(ate_result.describe())
```

`describe()` returns the main statistics as a pandas Series, in meters and radians: the number of poses; for the
positions the mean (`pos_mean`, the position ATE), median, RMS, standard deviation, 95th percentile, minimum,
maximum and the bias of every component; for the rotations the mean, RMS, standard deviation, 95th percentile
and maximum. `summary_table` and `segment_statistics` give rotations in degrees. Each statistic is also available as an attribute, e.g. `pos_ate`, `pos_dev_rms`,
`pos_dev_max`, `pos_bias_along` or `rot_ate`; `pos_dev_percentile(q)` returns any percentile.

### Confidence Intervals

Deviations of trajectories are correlated in time: neighboring poses do not deviate independently. The usual
standard error of the mean (`std / sqrt(n)`) is therefore far too optimistic. `confidence_interval` computes a
block bootstrap whose blocks are twice as long as the correlation time of the deviations, e.g. to decide whether
two estimates differ significantly:

```python
low, high = ate_result.confidence_interval(quantity="position", statistic="mean")  # or "rms"; "rotation" in radians
print(f"Position ATE: {ate_result.pos_ate:.3f} m (95 % confidence interval {low:.3f} to {high:.3f} m)")
```

### Segments

`segment_statistics` evaluates parts of the trajectory separately: consecutive windows of time or path length,
or segments given by a label for each pose (e.g. tunnel / open sky):

```python
table = ate_result.segment_statistics(segment_length=500.0, by="path_length")  # pandas DataFrame
print(table[["start", "end", "pos_rms", "pos_max"]])

labels = np.where(ate_result.trajectory.path_lengths < 1000.0, "first kilometer", "rest")
table = ate_result.segment_statistics(labels=labels)
```

### Temporal Correlation

Deviations from a reference are usually correlated over seconds to minutes, e.g. those of GNSS/INS.
`ate_result.correlation` describes this for every deviation component (`along`, `cross_h`, `cross_v`, `x`, `y`,
`z` and, with orientations, `roll`, `pitch` and `yaw`):

```python
cross_track = ate_result.correlation.components["cross_h"]
print(cross_track.white_noise_share)  # share of the variance that is uncorrelated white noise
print(cross_track.correlation_time)  # correlation time of the correlated part [s]
print(cross_track.effective_samples)  # approximate number of independent deviations
print(ate_result.correlation.cross_correlation_directed)  # correlation of along / cross-h / cross-v
```

Deviations are usually white noise (e.g. measurement noise of the reference) plus a correlated part (e.g. the
GNSS/INS error). The white noise makes the autocorrelation drop at once between the first two lags; then the
correlated part decays. Its share is extrapolated to lag 0 from the first lags, and its correlation time is the
lag at which the autocorrelation drops to 1/e of that share. Deviations whose first lag is not significantly
correlated are white noise (correlation time 0).

The autocorrelation is computed in chronological order, also for spatially sorted results. Gaps (e.g. losses of
a total station's lock) are respected, and irregular sampling is allowed. It is computed up to a tenth of the
number of deviations, at most up to 600 s, since larger lags rely on too few pairs; longer correlation times are
reported as infinite.

A correlation time can only be determined reliably from data that is much longer than it. With a reference such
as a synchronized total station, the cross-track deviations are the most reliable measure, because timing
errors affect only the along-track direction.

## Relative Pose Error (RPE)

```python
rpe_result = tpy.rpe(estimate, reference)
print(rpe_result)
print(rpe_result.describe())  # drift statistics per pair distance
```

The RPE compares the relative motion between pairs of poses that are a given distance (or time) apart:

1. Find pairs of reference poses that are, e.g., 100 m apart, and the corresponding pairs of the evaluated
   trajectory.
2. Compute the relative pose within each pair, $\Delta P = P_{1}^{-1} \cdot P_{N}$, for the reference and for
   the evaluated trajectory.
3. Compute the difference of both relative poses, and its translation and rotation.
4. Divide them by the distance (or time) between the poses of the pair.

If one of the trajectories has no orientations, only the position drift is computed, as the difference of the
lengths of the pairs (distances between their poses) instead of the relative poses.

By default, pairs are formed for distances of 100 m to 800 m in steps of 100 m. The position drift is given in
percent (pairs by distance) or m/s (pairs by time) (`pos_drift_unit`). The rotation values of the result are in
radians per 100 m or per second; `rot_drift_unit` is the display unit after conversion to degrees, e.g.
`np.rad2deg(rpe_result.rot_rpe)`. The pair distances are set in the
[relative comparison settings](settings.md#relative-comparison-settings):

```python
settings = tpy.RelativeComparisonSettings(
    pair_min_distance=10.0, pair_max_distance=60.0, pair_distance_step=10.0, pair_distance_unit="second"
)
rpe_result = tpy.rpe(estimate, reference, settings)  # with the default matching settings
print(f"{rpe_result.pos_rpe:.3f} {rpe_result.pos_drift_unit}")
```

## Comparing Several Results

To benchmark several estimates against one reference, `summary_table` puts the main statistics of all results
into one table (one row per result), which `format_table` writes as Markdown, CSV or LaTeX:

```python
estimates = [
    tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj"),
    tpy.Trajectory.from_file("./example_data/KITTI_SPTAM.traj"),
]
ate_results = [tpy.ate(estimate, reference) for estimate in estimates]
rpe_results = [tpy.rpe(estimate, reference) for estimate in estimates]

table = tpy.summary_table(ate_results, rpe_results)  # pandas DataFrame
print(tpy.format_table(table, table_format="markdown"))  # or "csv", "latex"
```

`write_report` creates a self-contained HTML file with the summary, the statistics of each result, all figures
and the processing settings, e.g. to archive the evaluation of a paper:

```python
tpy.write_report("report.html", ate_results, rpe_results, settings=tpy.ProcessingSettings())
```

## Saving Results

Results are saved as text files, which the GUI can open, too:

```python
ate_result.to_file("ate.result")
ate_result = tpy.ATEResult.from_file("ate.result")

rpe_result.to_file("rpe.result")
rpe_result = tpy.RPEResult.from_file("rpe.result")
```

Results read from files contain no alignment (`ate_result.alignment` is None).
