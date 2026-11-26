# File Formats

Trajectopy reads trajectories from:

- **ASCII files** (`.traj`, `.txt`, `.csv`, …): one pose per line; a header describes the columns.
- **TUM, KITTI and EuRoC files**, the formats of common benchmarks, which are recognized automatically.
- **ROS bags**: ROS1 (`.bag`) and ROS2 (`.mcap`, `.db3`), with poses, odometry, transforms, paths and GNSS fixes.

In Python, `Trajectory.from_file` and [`trajectopy.io.read_trajectory`](../api/io.md#trajectopy.io.read_trajectory)
read all file formats; [`trajectopy.io.trajectories_from_rosbag`](../api/io.md#trajectopy.io.trajectories_from_rosbag)
reads ROS bags.

## ASCII Files

### Structure

Each line holds one pose. Lines starting with `#` are header lines, which describe how the data is interpreted.
Without a header, the columns are `t,px,py,pz,qx,qy,qz,qw` (time, position, quaternion), separated by commas:

```
1000.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
1000.1,0.1,0.0,0.0,0.0,0.0,0.0,1.0
1000.2,0.2,0.0,0.0,0.0,0.0,0.0,1.0
```

With a header:

```
#name MyTrajectory
#epsg 25832
#fields t,px,py,pz,qx,qy,qz,qw
1000.0,364000.0,5621000.0,60.0,0.0,0.0,0.0,1.0
1000.1,364000.1,5621000.0,60.0,0.0,0.0,0.0,1.0
```

### Header Lines

| Header | Description | Default | Options |
|--------|-------------|---------|---------|
| `#name` | Display name | `Trajectory` | Any string |
| `#epsg` | EPSG code of the positions | `0` | Any EPSG code; `0` = local coordinates without a known datum |
| `#fields` | Meaning of the columns | `t,px,py,pz,qx,qy,qz,qw` | Comma-separated [field names](#field-names) |
| `#delimiter` | Column separator, in quotes | `','` | Any single character, e.g. `#delimiter ';'` |
| `#nframe` | Navigation frame of the data | `enu` | Order of east, north and up/down, e.g. `enu`, `ned`, `neu` |
| `#rot_unit` | Unit of Euler angles | `rad` | `rad`, `deg` |
| `#euler_rot_seq` | Axis sequence of Euler angles | `xyz` | Up to 3 of `x`, `y`, `z` for extrinsic or `X`, `Y`, `Z` for intrinsic rotations (not mixed) |
| `#time_format` | Format of the time column(s) | `unix` | `unix`, `datetime`, `gps_sow` |
| `#time_offset` | Offset in seconds added to all timestamps | `0.0` | Any number |
| `#datetime_format` | Format of date/time strings | `%Y-%m-%d %H:%M:%S.%f` | [strftime format](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes) |
| `#datetime_timezone` | Time system of date/time strings | `UTC` | `UTC` or `GPS` |
| `#gps_week` | GPS week of `gps_sow` timestamps | `0` | Integer |
| `#sorting` | Order of the poses | `time` | `time`, `path_length` (or `arc_length`) |

Notes:

- **Delimiter**: the character must be in quotes; without them, the line is ignored and a comma is used.
  Whitespace-separated files (spaces or tabs) are detected automatically and need no `#delimiter` line.
- **Navigation frame**: positions, velocities and orientations are converted to ENU. Positions are only
  converted if `#epsg` is `0`; with an EPSG code, their axes are defined by the coordinate system.
- **GPS time**: `gps_sow` timestamps are seconds of the GPS week given by `#gps_week`; they are converted to UTC
  (including leap seconds). `#datetime_timezone GPS` does the same for date/time strings.
- **Time systems**: date/time strings must be UTC or GPS time; local time zones are not supported (other values
  are read as UTC, with a warning). A constant offset can be corrected with `#time_offset`.
- Header lines may also appear after the data (e.g. comments), but files with all header lines at the top load
  faster.

### Field Names

| Field | Description | Can repeat? |
|-------|-------------|-------------|
| `t` | Time | Yes, for date and time in separate columns (`#time_format datetime`) |
| `l` | Path length (arc length) in meters | No |
| `px`, `py`, `pz` | Position (in the order of the coordinate system, e.g. latitude, longitude, height for EPSG 4326) | No |
| `qx`, `qy`, `qz`, `qw` | Orientation as quaternion | No |
| `ex`, `ey`, `ez` | Orientation as Euler angles (roll, pitch, yaw; see `#rot_unit` and `#euler_rot_seq`) | No |
| `vx`, `vy`, `vz` | Velocity in meters per second | No |
| `sx`, `sy`, `sz` | Standard deviations of the positions in meters, used to weight the alignment (see `use_position_std`) | No |
| `-` | Column that is not used | Yes |

Only `px`, `py` and `pz` are required. Without time column, the line number is used as timestamp; without
orientation, the trajectory has positions only (no rotation deviations, no leverarm estimation). If both
quaternions and Euler angles are given, the quaternions are used.

### Examples

**Positions only**

```
#name PositionOnly
#fields t,px,py,pz
1000.0,0.0,0.0,0.0
1000.1,0.1,0.0,0.0
1000.2,0.2,0.0,0.0
```

**Euler angles in degrees**

```
#name EulerTrajectory
#fields t,px,py,pz,ex,ey,ez
#rot_unit deg
1000.0,0.0,0.0,0.0,0.0,0.0,0.0
1000.1,0.1,0.0,0.0,0.0,0.0,5.0
```

**Space-separated, with unused columns**

```
#name SpaceSeparated
#fields t,-,-,px,py,pz
1000.0 17 ok 0.0 0.0 0.0
1000.1 17 ok 0.1 0.0 0.0
```

**Geographic coordinates (latitude, longitude, ellipsoidal height)**

```
#name GNSS
#epsg 4979
#fields t,px,py,pz
1000.0,50.7374,7.0982,120.5
1000.1,50.7375,7.0983,121.2
```

**Date and time strings**

```
#name DatetimeTrajectory
#fields t,px,py,pz,qx,qy,qz,qw
#time_format datetime
#datetime_format %Y-%m-%d %H:%M:%S.%f
#datetime_timezone UTC
2024-01-01 12:00:00.000,0.0,0.0,0.0,0.0,0.0,0.0,1.0
2024-01-01 12:00:00.100,0.1,0.0,0.0,0.0,0.0,0.0,1.0
```

Date and time can also be in separate columns: `#fields t,t,px,py,pz` (they are joined with a space).

**GPS time (seconds of week), NED frame, semicolon-separated**

```
#name GPSTime
#fields t,px,py,pz,qx,qy,qz,qw
#delimiter ';'
#nframe ned
#time_format gps_sow
#gps_week 2250
345600.0;0.0;0.0;0.0;0.0;0.0;0.0;1.0
345600.1;0.1;0.0;0.0;0.0;0.0;0.0;1.0
```

### Large Files

Files with millions of poses load several times faster with the optional pyarrow reader:
`pip install "trajectopy[fast]"`. It is used automatically when installed, for files without date/time strings.

## TUM, KITTI and EuRoC Files

Files of common benchmarks are recognized automatically by `Trajectory.from_file`, the GUI and the command line,
so they need no header:

| Format | Layout | Recognized by |
| --- | --- | --- |
| TUM | `timestamp tx ty tz qx qy qz qw`, separated by spaces | the default columns of trajectopy files |
| KITTI | one row-major 3×4 pose matrix (12 values) per line, no timestamps | 12 values per line |
| EuRoC | `timestamp [ns], px, py, pz, qw, qx, qy, qz[, vx, vy, vz, ...]` | a first line starting with `#timestamp` and commas |

The format can also be given explicitly, and trajectories can be written in these formats:

```python
import numpy as np
import trajectopy as tpy
from trajectopy.io import detect_format, read_kitti, write_trajectory

trajectory = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
write_trajectory(trajectory, "poses.txt", format="kitti")  # or "tum", "euroc", "trajectopy"
print(detect_format("poses.txt"))

trajectory = tpy.Trajectory.from_file("poses.txt", format="kitti")
np.savetxt("times.txt", trajectory.timestamps)
trajectory = read_kitti("poses.txt", timestamps="times.txt")  # KITTI poses are numbered otherwise
```

Without a `#name` header line, a trajectory is named after its file.

## ROS Bags

ROS1 bags (`.bag`) and ROS2 bags (`.mcap` or `.db3` files, or a bag directory with `metadata.yaml`) can be opened
in the GUI like ASCII files, or read in Python:

<!-- skip-test: needs a ROS bag -->
```python
from trajectopy.io import list_topics, trajectories_from_rosbag

print(list_topics("recording.mcap"))  # topics with a supported message type
trajectories = trajectories_from_rosbag("recording.mcap", topics=["/odom"])  # all supported topics by default
```

Each topic becomes one trajectory, named after the topic; the timestamps are taken from the message headers.

| Message type | Trajectory |
| --- | --- |
| `geometry_msgs/PoseStamped` | positions and orientations |
| `geometry_msgs/PoseWithCovarianceStamped`, `nav_msgs/Odometry` | positions and orientations, with position standard deviations from the covariance |
| `geometry_msgs/PointStamped` | positions |
| `geometry_msgs/TransformStamped`, `tf2_msgs/TFMessage` (`/tf`) | one trajectory per frame pair, named `topic: parent -> child` |
| `nav_msgs/Path` | the poses of the last message (paths are usually republished as a whole) |
| `sensor_msgs/NavSatFix` | latitude, longitude and ellipsoidal height (EPSG 4326) with standard deviations; fixes without a position are skipped |

Messages of other types are skipped.

## Position Standard Deviations

Trajectories can carry the standard deviations of their positions (`Trajectory.position_std`, in meters along x,
y and z). They are read from the covariances of ROS messages and from the fields `sx`, `sy`, `sz` of trajectopy
files, and written to those fields. The alignment weights each position by them instead of the global
`std_*` values of the [alignment stochastics](settings.md#alignment-settings) (see `use_position_std`); the
horizontal variance is the mean of the x and y variances.

## Files Written by Trajectopy

- **Trajectories** (`Trajectory.to_file`, *Export…* in the GUI) are ASCII files as described above, with the
  fields `t,l,px,py,pz,qx,qy,qz,qw,vx,vy,vz` (without `qx`…`qw` for trajectories without orientation, and
  followed by `sx,sy,sz` for trajectories with position standard deviations). The command
  `trajectopy convert` writes the TUM, KITTI and EuRoC formats, too.
- **Reports** (`.html`): self-contained evaluation reports (`trajectopy.write_report`, `--report` of the command
  line, *Export Report or Summary Table…* in the GUI), and summary tables (`.md`, `.csv`, `.tex`).
- **Results** (`.result`): ATE, RPE and alignment results (`ATEResult.to_file`, `RPEResult.to_file`,
  `AlignmentResult.to_file`). They are text files with a header, too, and can be opened in the GUI.
- **Settings** (`.json`): [processing settings](settings.md#processing-settings) and
  [plot settings](settings.md#plot-settings).
- **Projects** (`.tpy`, GUI): all trajectories, results and settings of the GUI, see
  [GUI](gui.md#projects). Projects are stored in a binary format (an uncompressed NumPy
  archive), so that even large projects are saved and opened within seconds. To exchange data with other
  programs, export the trajectories and results instead. Projects of Trajectopy 5.1 (zip archives of text files)
  can still be opened, and session folders of earlier versions can be imported.

## See Also

- [Trajectories](trajectories.md): creating trajectories from arrays, loading and saving them in Python
- [`trajectopy.io`](../api/io.md): all functions for reading and writing files
