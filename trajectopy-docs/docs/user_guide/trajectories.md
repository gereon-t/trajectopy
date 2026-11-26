# Trajectories

A [`Trajectory`](../api/trajectory.md#trajectopy.Trajectory) holds a sequence of poses: timestamps, positions
and, optionally, orientations. Everything in trajectopy, from the alignment to the plots, works on trajectories.

## Creating Trajectories

### From Files

`Trajectory.from_file` reads trajectopy, TUM, KITTI and EuRoC files and ASCII files whose columns are described
by a header (see [File Formats](file_formats.md)). The format is recognized automatically.

```python
import numpy as np
import trajectopy as tpy

trajectory = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
print(repr(trajectory))
```

```text
<Trajectory 'KITTI Ground Truth': 4541 poses with orientations, 3724.187 m, EPSG 0, sorted by time>
```

The same short summary is shown in interactive sessions; `print(trajectory)` shows a longer one with the data
rate.

ROS bags contain one trajectory per topic; they are read with
[`trajectopy.io.trajectories_from_rosbag`](../api/io.md#trajectopy.io.trajectories_from_rosbag) (see
[ROS Bags](file_formats.md#ros-bags)).

### From Arrays

The constructor takes the positions as an array of shape (N, 3), the orientations as quaternions (x, y, z, w) of
shape (N, 4) or as a `scipy.spatial.transform.Rotation`, and the timestamps in seconds:

```python
timestamps = np.arange(0.0, 10.0, 0.1)
positions = np.column_stack([timestamps, np.sin(timestamps), np.zeros_like(timestamps)])
quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (len(timestamps), 1))  # identity

curve = tpy.Trajectory(positions, quaternions, timestamps, name="Curve")
```

`Trajectory.from_arrays` additionally accepts Euler angles (`rpy=`, roll, pitch and yaw in radians) and the EPSG
code of the positions:

```python
rpy = np.column_stack([np.zeros_like(timestamps), np.zeros_like(timestamps), timestamps / 10])
curve = tpy.Trajectory.from_arrays(positions, rpy=rpy, timestamps=timestamps, name="Curve")
```

Without timestamps, the pose numbers 0, 1, 2, … are used. Without orientations, the trajectory has positions only;
then, rotation deviations and the leverarm cannot be computed.

Poses with the same timestamp are removed (the first one is kept), and the poses are put into chronological order.

## Accessing the Data

| Attribute | Contents |
| --- | --- |
| `timestamps` | Timestamps in seconds, shape (N,) |
| `xyz` | Positions, shape (N, 3); the same as `positions.xyz` |
| `positions` | [`Positions`](../api/trajectory.md#trajectopy.Positions): the positions and their EPSG code |
| `rotations` | Orientations as `scipy.spatial.transform.Rotation`, or None |
| `quat`, `rpy` | Orientations as quaternions (x, y, z, w) or Euler angles (roll, pitch, yaw in radians) |
| `se3` | Poses as 4×4 homogeneous matrices |
| `path_lengths` | Distance travelled since the first pose, in meters |
| `velocity_xyz`, `absolute_velocity` | Velocity vectors and speed in m/s (derived from the positions unless given) |
| `position_std` | Standard deviations of the positions along x, y and z in meters, or None |
| `name`, `sorting` | Name, and whether the poses are ordered by time or by path length |
| `total_length`, `data_rate`, `time_start`, `time_end` | Summary values |

```python
print(len(trajectory), trajectory.has_orientation, trajectory.total_length)

table = trajectory.to_dataframe()  # pandas DataFrame with one row per pose
```

Indexing works like with NumPy arrays: an integer returns a single [`Pose`](../api/trajectory.md#trajectopy.Pose),
a slice, a boolean mask or an index array returns a new trajectory:

```python
first = trajectory[0]
print(first.timestamp, first.position)

every_tenth = trajectory[::10]
fast = trajectory[trajectory.absolute_velocity > 10.0]
```

Iterating over a trajectory yields its poses. `equals` compares the data of two trajectories within a tolerance
(`==` also compares their names).

## Selecting Poses

| Method | Result |
| --- | --- |
| `crop(start, end)` | Poses between two timestamps (or path lengths, see [Sorting](#time-and-path-length)); `inverse=True` removes them instead |
| `mask(mask)` | Poses selected by a boolean mask or index array (the same as indexing) |
| `intersect(index)` | Poses within the time span of another index, without large gaps |

```python
first_minute = trajectory.crop(0.0, 60.0)
without_first_minute = trajectory.crop(0.0, 60.0, inverse=True)
```

Like all methods that change a trajectory, they return a new trajectory and leave the original unchanged, unless
`inplace=True` is passed.

## Coordinate Systems

The positions of a trajectory are given in a coordinate system with an [EPSG code](https://epsg.io)
(`trajectory.positions.epsg`), e.g. 25832 for UTM zone 32N or 4326 for latitude and longitude. EPSG code 0 means
a local system without known datum, e.g. of a SLAM system; such positions cannot be transformed into other
systems.

`Positions.to_epsg` transforms the positions:

```python
utm = tpy.Positions(np.column_stack([364000.0 + positions[:, 0], 5621000.0 + positions[:, 1], positions[:, 2]]), epsg=25832)
georeferenced = tpy.Trajectory(utm, quaternions, timestamps, name="UTM")

geographic = georeferenced.copy()
geographic.positions = geographic.positions.to_epsg(4326)  # latitude, longitude, height
print(geographic.xyz[0])
```

Georeferenced positions also have a *local frame*: a tangential east-north-up frame at their mean, in which path
lengths are computed. `positions.to_local()` returns the positions in it.

## Transforming

`transform` applies a 4×4 transformation matrix (rotation, translation and, optionally, scale) to the positions,
orientations and velocities:

```python
shift = np.eye(4)
shift[:3, 3] = [10.0, 0.0, 0.0]
shifted = trajectory.transform(shift)
```

To transform a trajectory onto a reference, estimate the transformation with an [alignment](alignment.md).

## Time and Path Length

The poses of a trajectory are ordered by time (`Sorting.TIME`) or, after spatial sorting, by path length
(`Sorting.PATH_LENGTH`). The order determines the *index* of the trajectory (`trajectory.index`): the
timestamps or the path lengths. Cropping, interpolation, approximation and the plots work on the index.

Spatial sorting orders the poses of a trajectory that drives along the same track several times by their
position on the track, e.g. to compare laps with each other. It reconstructs the path from the positions alone
(see the [sorting settings](settings.md#sorting-settings)):

<!-- skip-test: needs a trajectory driving along a closed track -->
```python
sorted_trajectory = tpy.sort_spatially(trajectory)
laps = tpy.divide_into_laps(sorted_trajectory)  # one trajectory per lap
```

`set_sorting` switches between both orders of a trajectory that has been sorted spatially before.

## Resampling, Smoothing and Combining

| Function | Result |
| --- | --- |
| [`interpolate`](../api/processing.md#trajectopy.interpolate) | Trajectory interpolated at given timestamps (linear for positions, SLERP for orientations) |
| [`approximate`](../api/processing.md#trajectopy.approximate) | Smoothed trajectory: piecewise cubic polynomials for the positions, moving average for the orientations |
| [`merge_trajectories`](../api/processing.md#trajectopy.merge_trajectories) | One trajectory with the poses of several, ordered by time |
| [`average_trajectories`](../api/processing.md#trajectopy.average_trajectories) | Pose-wise mean of several trajectories with the same number of poses |

```python
resampled = tpy.interpolate(trajectory, np.arange(0.0, 100.0, 0.5))
smoothed = tpy.approximate(trajectory, tpy.ApproximationSettings(position_interval_size=5.0))
```

## Saving

`to_file` writes the trajectopy format, or TUM, KITTI or EuRoC depending on the file extension or the `format`
argument:

```python
trajectory.to_file("copy.traj")
trajectory.to_file("copy.tum")  # TUM format
trajectory.to_file("poses.txt", format="kitti")
```

See [File Formats](file_formats.md#files-written-by-trajectopy) for the content of the files.
