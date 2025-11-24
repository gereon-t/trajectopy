
## <kbd>class</kbd> `Trajectory`
Class representing a trajectory containing synchronized position, orientation, and time data. 



**Attributes:**
 
 - <b>`positions`</b> (Positions):  Container for spatial coordinates and coordinate reference system (EPSG) data. 
 - <b>`rotations`</b> (Rotations):  Container for orientation data (quaternions), or None if not provided. 
 - <b>`timestamps`</b> (np.ndarray):  A 1D array of timestamps corresponding to each pose. 
 - <b>`name`</b> (str):  An identifier string for the trajectory. 
 - <b>`path_lengths`</b> (np.ndarray):  A 1D array of cumulative path lengths starting from zero. 
 - <b>`sorting`</b> (Sorting):  The current sorting strategy (Sorting.TIME or Sorting.ARC_LENGTH). 

Properties: 
 - <b>`velocity_xyz`</b> (np.ndarray):  The 3D velocity vector $[v_x, v_y, v_z]$ for each pose. 
 - <b>`absolute_velocity`</b> (np.ndarray):  The scalar speed (magnitude of velocity) for each pose. 
 - <b>`se3`</b> (List[np.ndarray]):  A list of $4 \times 4$ homogeneous transformation matrices representing the full pose. 
 - <b>`xyz`</b> (np.ndarray):  The position coordinates sorted according to the current sorting strategy. 
 - <b>`quat`</b> (np.ndarray):  The orientation quaternions $[x, y, z, w]$ sorted according to the current sorting strategy. 
 - <b>`rpy`</b> (np.ndarray):  The Euler angles (roll, pitch, yaw) sorted according to the current sorting strategy. 
 - <b>`total_length`</b> (float):  The total distance covered by the trajectory in meters. 
 - <b>`data_rate`</b> (float):  The average sampling frequency of the trajectory in Hz. 
 - <b>`index`</b> (np.ndarray):  The current independent variable array (time or path-length) used for parametrization. 

### <kbd>method</kbd> `Trajectory.__init__`

```python
__init__(
    xyz: numpy.ndarray,
    timestamps: numpy.ndarray | None = None,
    quat: numpy.ndarray | None = None,
    rpy: numpy.ndarray | None = None,
    epsg: int = 0,
    name: str = '',
    path_lengths: Optional[numpy.ndarray] = None,
    velocity_xyz: Optional[numpy.ndarray] = None,
    sorting: trajectopy.definitions.Sorting = <Sorting.TIME: 'time'>
) → None
```

Initialize a Trajectory object. 



**Args:**
 
 - <b>`xyz`</b> (np.ndarray):  N x 3 array of position coordinates. 
 - <b>`timestamps`</b> (np.ndarray | None, optional):  Array of timestamps. If None, a range index is used. 
 - <b>`quat`</b> (np.ndarray | None, optional):  N x 4 array of quaternions (x, y, z, w). Mutually exclusive with `rpy`. 
 - <b>`rpy`</b> (np.ndarray | None, optional):  N x 3 array of Roll-Pitch-Yaw angles. Mutually exclusive with `quat`. 
 - <b>`epsg`</b> (int, optional):  EPSG code for the coordinate system. Defaults to 0. 
 - <b>`name`</b> (str, optional):  Name of the trajectory. Defaults to generic counter name. 
 - <b>`path_lengths`</b> (Union[np.ndarray, None], optional):  Pre-calculated path lengths. If None, they are computed from xyz. 
 - <b>`velocity_xyz`</b> (Union[np.ndarray, None], optional):  Pre-calculated 3D velocities. If None, they are computed via gradient. 
 - <b>`sorting`</b> (Sorting, optional):  Definition of the sorting logic (TIME or ARC_LENGTH). Defaults to Sorting.TIME. 



**Raises:**
 
 - <b>`TrajectoryError`</b>:  If both `quat` and `rpy` are provided, or if array dimensions do not match. 


---

#### <kbd>property</kbd> Trajectory.absolute_velocity

Returns the norm (magnitude) of the 3D velocity vectors. 

---

#### <kbd>property</kbd> Trajectory.data_rate

Calculates the average data rate (frequency in Hz) based on timestamp differences. 

---

#### <kbd>property</kbd> Trajectory.datetimes

Returns the timestamps converted to Pandas datetime objects (unit='s'). 

---

#### <kbd>property</kbd> Trajectory.has_orientation

Returns True if valid rotation data is available. 

---

#### <kbd>property</kbd> Trajectory.index

Returns the independent variable currently parameterizing the trajectory. This is either the Timestamp vector or the Path Length vector, depending on `self.sorting`. 

---

#### <kbd>property</kbd> Trajectory.index_label

Returns the label string of the current index (e.g., 'time [s]'). 

---

#### <kbd>property</kbd> Trajectory.index_unit

Returns the unit string of the current index ('s' for Time, 'm' for Path Length). 

---

#### <kbd>property</kbd> Trajectory.is_unix_time

Checks if the supplied trajectories have (likely) unix timestamps as seconds and converts them to datetime objects. 

---

#### <kbd>property</kbd> Trajectory.quat

Returns the quaternions sorted according to the current `sorting` strategy. Returns zeros if no rotations are present. 

---

#### <kbd>property</kbd> Trajectory.rpy

Returns the Roll-Pitch-Yaw angles sorted according to the current `sorting` strategy. 

---

#### <kbd>property</kbd> Trajectory.se3

Returns a list of SE3 poses (4x4 homogeneous transformation matrices). 

---

#### <kbd>property</kbd> Trajectory.sort_switching_index

Returns an array of indices that would switch the current sorting (e.g., unsort the data). 

---

#### <kbd>property</kbd> Trajectory.sorting_index

Returns the indices used to sort the trajectory based on the current `sorting` attribute (Time or Path Length). 

---

#### <kbd>property</kbd> Trajectory.total_length

Returns the total cumulative path length of the trajectory in meters. 

---

#### <kbd>property</kbd> Trajectory.velocity_xyz

Returns the 3D velocity vectors. If not set manually, they are computed via gradient of the positions over time. 

---

#### <kbd>property</kbd> Trajectory.xyz

Returns the XYZ coordinates sorted according to the current `sorting` strategy. Note: This differs from `self.positions.xyz`, which retains the original order. 



---

### <kbd>method</kbd> `Trajectory.copy`

```python
copy() → Trajectory
```

Returns a deep copy of the trajectory instance. 

---

### <kbd>method</kbd> `Trajectory.crop`

```python
crop(
    t_start: float,
    t_end: float,
    inverse: bool = False,
    inplace: bool = True
) → Trajectory
```

Crops (or cuts) the trajectory based on a time window. 



**Args:**
 
 - <b>`t_start`</b> (float):  Start timestamp of the window. 
 - <b>`t_end`</b> (float):  End timestamp of the window. 
 - <b>`inverse`</b> (bool, optional):  If True, removes data *inside* the window (cutting).  If False, keeps data *inside* the window (cropping). Defaults to False. 
 - <b>`inplace`</b> (bool, optional):  If True, modifies self. If False, returns a new instance. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  The modified or new trajectory instance. 

---

### <kbd>classmethod</kbd> `Trajectory.from_file`

```python
from_file(filename: str, io_stream: bool = False) → Trajectory
```

Create a trajectory instance from a file. 

The file is expected to be a CSV-like format. It handles extraction of timestamps, xyz positions, rotations, path lengths, and velocities via `trajectory_io`. 



**Args:**
 
 - <b>`filename`</b> (str):  Path to the file or string content if io_stream is True. 
 - <b>`io_stream`</b> (bool, optional):  If True, `filename` is treated as the raw string content  of the file/stream. Defaults to False. 



**Returns:**
 
 - <b>`Trajectory`</b>:  The loaded trajectory object. 

---

### <kbd>method</kbd> `Trajectory.init_path_lengths`

```python
init_path_lengths()
```

Computes cumulative path lengths based on Euclidean distances between consecutive local coordinates. 

---

### <kbd>method</kbd> `Trajectory.intersect`

```python
intersect(
    timestamps: numpy.ndarray,
    max_gap_size: float = 10.0,
    inplace: bool = True
) → Trajectory
```

Filters the trajectory to overlap with a reference timestamp vector. 

This method finds the common time span between self and the reference `timestamps`, crops self to that span, and then filters points that are either exact matches or exist within valid gaps defined by `max_gap_size`. 



**Args:**
 
 - <b>`timestamps`</b> (np.ndarray):  The reference timestamps to intersect with. 
 - <b>`max_gap_size`</b> (float, optional):  The maximum allowed time gap (in seconds) between  reference timestamps to include trajectory points. Defaults to 10.0. 
 - <b>`inplace`</b> (bool, optional):  If True, modifies self. Defaults to True. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the time spans do not overlap. 



**Returns:**
 
 - <b>`Trajectory`</b>:  The intersected trajectory. 

---

### <kbd>method</kbd> `Trajectory.mask`

```python
mask(mask: Union[list, numpy.ndarray], inplace: bool = True) → Trajectory
```

Applies a boolean mask or index array to filter all trajectory components. 

Filtered components include: timestamps, positions, rotations, path lengths, and velocities. 



**Args:**
 
 - <b>`mask`</b> (Union[list, np.ndarray]):  Boolean array or list of indices to keep. 
 - <b>`inplace`</b> (bool, optional):  If True, modifies self. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  The masked trajectory. 

---

### <kbd>method</kbd> `Trajectory.overlaps_with`

```python
overlaps_with(other: 'Trajectory') → bool
```

Checks if the time span of this trajectory overlaps with another. 



**Args:**
 
 - <b>`other`</b> (Trajectory):  The trajectory to compare against. 



**Returns:**
 
 - <b>`bool`</b>:  True if the time ranges overlap, False otherwise. 

---

### <kbd>method</kbd> `Trajectory.to_dataframe`

```python
to_dataframe(sort_by: str = '') → DataFrame
```

Exports the trajectory to a Pandas DataFrame. 

Columns usually include: time, path_length, pos_x, pos_y, pos_z, speed_x, speed_y, speed_z, and rotation columns (rot_x/y/z/w) if available. 



**Args:**
 
 - <b>`sort_by`</b> (str, optional):  Column name to sort by. If empty, uses `self.sorting`. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  A dataframe containing the trajectory data. 

---

### <kbd>method</kbd> `Trajectory.to_file`

```python
to_file(filename: str, mode: str = 'w') → None
```

Writes the trajectory to an ASCII file using the format defined in `to_string`. 



**Args:**
 
 - <b>`filename`</b> (str):  The output file path. 
 - <b>`mode`</b> (str, optional):  File open mode. Defaults to "w". 

---

### <kbd>method</kbd> `Trajectory.to_kml`

```python
to_kml(filename: str, precision: float = 1e-06) → str
```

Exports the trajectory to a Google Earth KML file. 

Requires the trajectory to have a valid EPSG code so it can be converted to WGS84 (EPSG:4326). 



**Args:**
 
 - <b>`filename`</b> (str):  The output filename (e.g., "track.kml"). 
 - <b>`precision`</b> (float, optional):  Coordinate precision in degrees for rounding/simplification. Defaults to 1e-6. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the trajectory does not have a known EPSG code. 

---

### <kbd>method</kbd> `Trajectory.to_string`

```python
to_string() → str
```

Serializes the trajectory to a CSV-formatted string with metadata headers. 

Headers included: #epsg, #name, #nframe, #sorting, #fields. 

---

### <kbd>method</kbd> `Trajectory.transform`

```python
transform(transformation: numpy.ndarray, inplace: bool = True) → Trajectory
```

Applies a rigid body transformation to the trajectory poses. 



**Args:**
 
 - <b>`transformation`</b> (np.ndarray):  A 4x4 homogeneous transformation matrix. 
 - <b>`inplace`</b> (bool, optional):  If True, modifies self. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  The transformed trajectory. 

