# File Formats

Comprehensive guide to trajectory file formats supported by Trajectopy.

## Overview

Trajectopy supports:

- **ASCII/CSV files** - Primary format (`.traj`, `.txt`, `.csv`)
- **ROS bag files** - ROS1 and ROS2 (`.bag`)

## ASCII File Format

### Default Structure

```
time, px, py, pz, qx, qy, qz, qw
```

**Example:**
```
1000.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
1000.1,0.1,0.0,0.0,0.0,0.0,0.0,1.0
1000.2,0.2,0.0,0.0,0.0,0.0,0.0,1.0
```

### Header Configuration

Headers start with `#` and configure how data is interpreted:

```
#name MyTrajectory
#epsg 32632
#fields t,px,py,pz,qx,qy,qz,qw
#delimiter ,
1000.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
```

### Supported Header Fields

| Header | Description | Default | Options |
|--------|-------------|---------|---------|
| `#name` | Trajectory display name | Filename | Any string |
| `#epsg` | EPSG coordinate system code | 0 | Any valid EPSG code |
| `#fields` | Column definitions | `t,px,py,pz,qx,qy,qz,qw` | See field table |
| `#delimiter` | Column separator | `,` | Any character |
| `#nframe` | Navigation frame | `enu` | `enu`, `ned` |
| `#rot_unit` | Rotation unit | `rad` | `rad`, `deg` |
| `#time_format` | Time format | `unix` | `unix`, `datetime`, `gps_sow` |
| `#time_offset` | Time offset (seconds) | `0.0` | Any float |
| `#datetime_format` | Datetime string format | `%Y-%m-%d %H:%M:%S.%f` | strftime format |
| `#datetime_timezone` | Timezone | `UTC` | TZ name or `GPS` |
| `#gps_week` | GPS week (for gps_sow) | `0` | Integer |
| `#sorting` | Data sorting | `time` | `time`, `arc_length` |

### Field Names

| Field | Description | Type | Can repeat? |
|-------|-------------|------|-------------|
| `t` | Time/timestamp | str / float | Yes |
| `l` | Arc length | float | No |
| `px` | Position X / Latitude | float | No |
| `py` | Position Y / Longitude | float | No |
| `pz` | Position Z / Altitude | float | No |
| `qx` | Quaternion X | float | No |
| `qy` | Quaternion Y | float | No |
| `qz` | Quaternion Z | float | No |
| `qw` | Quaternion W | float | No |
| `ex` | Euler angle X (roll) | float | No |
| `ey` | Euler angle Y (pitch) | float | No |
| `ez` | Euler angle Z (yaw) | float | No |
| `vx` | Velocity X | float | No |
| `vy` | Velocity Y | float | No |
| `vz` | Velocity Z | float | No |

## Format Examples

### Position-Only Trajectory

```
#name PositionOnly
#fields t,px,py,pz
1000.0,0.0,0.0,0.0
1000.1,0.1,0.0,0.0
1000.2,0.2,0.0,0.0
```

### Euler Angles (Degrees)

```
#name EulerTrajectory
#fields t,px,py,pz,ex,ey,ez
#rot_unit deg
1000.0,0.0,0.0,0.0,0.0,0.0,0.0
1000.1,0.1,0.0,0.0,0.0,0.0,5.0
```

### Space-Separated

```
#name SpaceSeparated
#fields t,px,py,pz,qx,qy,qz,qw
#delimiter  
1000.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
1000.1 0.1 0.0 0.0 0.0 0.0 0.0 1.0
```

### Geographic Coordinates (Lat/Lon)

```
#name GPS
#epsg 4326
#fields t,px,py,pz
1000.0,50.7374,7.0982,120.5
1000.1,50.7375,7.0983,121.2
```

### Datetime Timestamps

```
#name DatetimeTrajectory
#fields t,px,py,pz,qx,qy,qz,qw
#time_format datetime
#datetime_format %Y-%m-%d %H:%M:%S.%f
#datetime_timezone UTC
2024-01-01 12:00:00.000,0.0,0.0,0.0,0.0,0.0,0.0,1.0
2024-01-01 12:00:00.100,0.1,0.0,0.0,0.0,0.0,0.0,1.0
```

### GPS Time (SOW)

```
#name GPSTime
#fields t,px,py,pz,qx,qy,qz,qw
#time_format gps_sow
#gps_week 2250
345600.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
345600.1,0.1,0.0,0.0,0.0,0.0,0.0,1.0
```

## Creating Custom Trajectories

### From NumPy Arrays

```python
import trajectopy as tpy
import numpy as np

# Create timestamps and positions
timestamps = np.arange(0, 10, 0.1)
positions = np.column_stack(
    [
        timestamps,  # x increases with time
        np.sin(timestamps),  # y is sinusoidal
        np.zeros_like(timestamps),  # z is zero
    ]
)

# Create rotations (identity quaternions)
quaternions = np.tile([0, 0, 0, 1], (len(timestamps), 1))

# Create trajectory
traj = tpy.Trajectory(
    timestamps=timestamps,
    xyz=positions,
    quat=quaternions,
    name="CustomTrajectory",
)

# Save
traj.to_file("custom.traj")
```

## See Also

- **[User Guide](./user_guide.md)** - GUI file import
- **[API Reference](./api/core/trajectory.md)** - Trajectory class details
