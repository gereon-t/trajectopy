# Alignment

Two trajectories can only be compared pose by pose if they are given in the same coordinate frame and with the
same time reference. A SLAM estimate, for example, starts in a frame of its own, and the clock of a sensor may be
offset from that of the reference. The *alignment* estimates the transformation between both trajectories.

## Parameters

The alignment estimates up to 11 parameters with a least squares adjustment of matched poses (see the
[paper](https://doi.org/10.1515/jag-2024-0040) for the model):

| Parameters | Meaning | Estimated by default |
| --- | --- | --- |
| Translation x, y, z | Shift between the frames [m] | yes |
| Rotation x, y, z | Rotation between the frames [rad] | yes |
| Scale | Scale factor, e.g. for monocular visual odometry | no |
| Time shift | Offset between the clocks [s] | no |
| Leverarm x, y, z | Offset between the reference points of both platforms, in the body frame [m]; needs orientations | no |

Additionally, a constant *sensor rotation* (mounting rotation between the body frames) can be computed from the
orientations after the least squares adjustment.

Which parameters are estimated is set in the [alignment settings](settings.md#alignment-settings); the
settings also contain the standard deviations that weight the observations and a preprocessing step that
selects the poses to use. The leverarm needs the orientations of the trajectory to align; estimating it
without them raises `ValueError`.

The time shift is the offset of the clock of the aligned trajectory: `apply_alignment` subtracts it from its
timestamps. A positive time shift means that the timestamps of the trajectory are too large, e.g. a time shift
of 0.1 s means that the pose stamped 10.1 s was taken when the reference shows 10.0 s.

## Estimating and Applying an Alignment

```python
import trajectopy as tpy

reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

alignment = tpy.estimate_alignment(estimate, reference)
print(alignment)
print(alignment.position_parameters)

aligned = tpy.apply_alignment(estimate, alignment)
```

`estimate_alignment` matches the poses of both trajectories (see [Matching](#matching)), estimates the
parameters and returns an [`AlignmentResult`](../api/results.md#trajectopy.AlignmentResult). `apply_alignment`
applies it to a trajectory, not necessarily the one it was estimated for, e.g. to all trajectories recorded with
the same sensor setup. `align` does both in one step:

```python
aligned = tpy.align(estimate, reference)
```

Both functions take `AlignmentSettings` or the complete `ProcessingSettings`; only the latter also sets the
matching (with `AlignmentSettings`, the default matching settings are used).

An [ATE evaluation](evaluation.md#absolute-trajectory-error-ate) aligns the trajectories itself and keeps the
alignment in its result (`ate_result.alignment`).

### Choosing the Parameters

```python
settings = tpy.AlignmentSettings()
settings.estimation_settings.scale = True  # monocular visual odometry has no metric scale
settings.estimation_settings.time_shift = True

alignment = tpy.estimate_alignment(estimate, reference, settings)
```

`AlignmentEstimationSettings.from_components` enables groups of parameters at once; `similarity` enables the
translation, the rotation and the scale:

```python
settings.estimation_settings = tpy.AlignmentEstimationSettings.from_components(
    similarity=True, time_shift=True, leverarm=False
)
```

### Inspecting the Result

The estimated parameters are [`AlignmentParameters`](../api/results.md#trajectopy.AlignmentParameters): each
parameter has a value and a variance, and the parameters share a covariance matrix.

```python
parameters = alignment.position_parameters
print(parameters.sim_trans_x.value, parameters.sim_trans_x.variance)
print(parameters.time_shift.value)
print(parameters.sim3_matrix)  # 4x4 similarity transformation
print(alignment.converged)  # False if the least squares adjustment did not converge
```

If the adjustment does not converge, the parameters keep their initial values.

`trajectopy.visualization.alignment_figures` plots the covariance and correlation matrices, which show whether
parameters could be told apart with the data at hand (see [Visualization](visualization.md)).

Alignments are saved and loaded with `to_file` and `AlignmentResult.from_file`.

### Initial Values

`initial_parameters` starts the adjustment at given values instead of an approximate solution of its own, e.g.
from a manual alignment in the GUI. Parameters that are not estimated keep these values. A large time offset
(e.g. between unrelated clocks) can be given this way; then only the remaining offset is estimated:

```python
prior = tpy.AlignmentParameters()
prior.time_shift.value = 0.0  # e.g. a known offset between the clocks in seconds
alignment = tpy.estimate_alignment(estimate, reference, initial_parameters=prior)
```

## Simpler Alignments

`adopt_first_pose` shifts a trajectory so that its first position coincides with the first position of the
reference, and rotates its orientations so that its first orientation coincides with the first orientation of
the reference. The positions are only shifted, not rotated. Both trajectories should be matched first, so that
their first poses correspond.

```python
matched, matched_reference = tpy.match_trajectories(estimate, reference)
first_pose_aligned = tpy.adopt_first_pose(matched, matched_reference)
```

## Matching

Alignment and ATE compare *corresponding* poses. `match_trajectories` finds them with the method of the
[matching settings](settings.md#matching-settings) and returns copies of both trajectories that contain only the
matched poses, in the same order:

| Method | Matching |
| --- | --- |
| `interpolation` (default) | The trajectory is interpolated at the timestamps of the reference |
| `nearest_temporal` | Pose with the nearest timestamp, within `max_time_diff` |
| `nearest_spatial` | Nearest position, within `max_distance`; needs no common time reference |
| `nearest_spatial_interpolated` | Projection onto a line through the `k_nearest` nearest reference positions |

```python
matched, matched_reference = tpy.match_trajectories(
    estimate, reference, tpy.MatchingSettings(method="nearest_temporal")
)
print(len(matched) == len(matched_reference))
```

If the time spans of both trajectories do not overlap at all, the `interpolation` and `nearest_temporal`
methods first estimate the time offset roughly by matching the positions. Clocks that are offset but still
overlap are not detected; estimate the time shift with the alignment, or give it as initial value.
