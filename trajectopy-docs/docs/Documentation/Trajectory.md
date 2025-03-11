
## <kbd>class</kbd> `Trajectory`
Class representing a trajectory, i.e. position and orientation of a plattform over time 


- Position-Computations are always done in a local frame 
- Time stamps are always in UTC time 
- Rotations are always defined in a East-North-Up frame 



**Attributes:**
 
 - <b>`pos`</b> (PointSet):  Position of the trajectory 
 - <b>`rot`</b> (RotationSet):  Orientation of the trajectory 
 - <b>`tstamps`</b> (np.ndarray):  Time stamps of the trajectory 
 - <b>`name`</b> (str):  Name of the trajectory 
 - <b>`arc_lengths`</b> (np.ndarray):  Arc lengths of the trajectory 
 - <b>`speed_3d`</b> (np.ndarray):  3D speed of the trajectory 
 - <b>`sorting`</b> (Sorting):  Sorting of the trajectory 

Methods: 


 - <b>`__init__`</b>:  Initialize trajectory 
 - <b>`__str__`</b>:  Returns string describing trajectory 
 - <b>`__repr__`</b>:  Returns string representation of trajectory 
 - <b>`__len__`</b>:  Return number of poses 
 - <b>`__eq__`</b>:  Check if two trajectories are equal 
 - <b>`init_arc_lengths`</b>:  Initialize arc lengths 
 - <b>`copy`</b>:  Deep copy of itself 
 - <b>`from_file`</b>:  Create trajectory from file 
 - <b>`sort_switching_index`</b>:  Returns the index that switches the sorting of the trajectory 
 - <b>`sorting_index`</b>:  Returns the index that sorts the trajectory 
 - <b>`function_of`</b>:  Returns the function of the trajectory 
 - <b>`function_of_unit`</b>:  Returns the unit of the function of the trajectory 
 - <b>`function_of_label`</b>:  Returns the label of the function of the trajectory 
 - <b>`xyz`</b>:  Returns the xyz coordinates of the trajectory 
 - <b>`quat`</b>:  Returns the quaternion of the trajectory 
 - <b>`rpy`</b>:  Returns the roll, pitch, yaw of the trajectory 
 - <b>`to_dataframe`</b>:  Returns a pandas dataframe containing tstamps, xyz, quat and speed_3d of the trajectory 
 - <b>`to_file`</b>:  Writes trajectory to ascii file 
 - <b>`from_numpy`</b>:  Initialize trajectory using numpy arrays 
 - <b>`se3`</b>:  Returns SE3 pose list 
 - <b>`se3.setter`</b>:  Sets position and rotation from se3 list 
 - <b>`data_rate`</b>:  Returns data rate 
 - <b>`total_length`</b>:  Return the total trajectory arc_length 
 - <b>`speed_3d`</b>:  Returns computed speeds or custom speeds 
 - <b>`speed_3d.setter`</b>:  Sets custom speeds 
 - <b>`speed`</b>:  Returns trajectory speeds calculated using consecutive point distances 
 - <b>`crop`</b>:  Crops trajectory to timespan defined by t_start and t_end 
 - <b>`interpolate`</b>:  Interpolates a trajectory to specified timestamps 
 - <b>`_interpolate_rotations`</b>:  Function for rotation interpolation of a trajectory 
 - <b>`_interpolate_positions`</b>:  Function for position interpolation of a trajectory 
 - <b>`match_timestamps`</b>:  Truncates trajectory to only those poses where the timestamps exactly match "tstamps" 
 - <b>`intersect`</b>:  Intersects trajectory with a given timestamp vector 
 - <b>`apply_index`</b>:  Applies index to the trajectory 
 - <b>`apply_transformation`</b>:  Applies transformation to trajectory 

### <kbd>method</kbd> `Trajectory.__init__`

```python
__init__(
    pos: trajectopy.pointset.PointSet,
    rot: Optional[trajectopy.rotationset.RotationSet] = None,
    tstamps: Optional[numpy.ndarray] = None,
    name: str = '',
    arc_lengths: Optional[numpy.ndarray] = None,
    speed_3d: Optional[numpy.ndarray] = None,
    sorting: trajectopy.sorting.Sorting = <Sorting.TIME: 'time'>
) → None
```






---

#### <kbd>property</kbd> Trajectory.data_rate

Returns data rate 

---

#### <kbd>property</kbd> Trajectory.datetimes

Returns the datetime of the trajectory 

---

#### <kbd>property</kbd> Trajectory.function_of

Returns the function of the trajectory 

---

#### <kbd>property</kbd> Trajectory.function_of_label

Returns the label of the function of the trajectory 

---

#### <kbd>property</kbd> Trajectory.function_of_unit

Returns the unit of the function of the trajectory 

---

#### <kbd>property</kbd> Trajectory.has_orientation

Returns True if orientation is available 

---

#### <kbd>property</kbd> Trajectory.is_unix_time

Returns True if time is in unix format 

---

#### <kbd>property</kbd> Trajectory.quat

Returns the quaternion of the trajectory 

In contrast to the rot.as_quat() attribute, this method reflects the current sorting of the trajectory. 

---

#### <kbd>property</kbd> Trajectory.rpy

Returns the roll, pitch, yaw of the trajectory 

In contrast to the rot.as_euler(seq="xyz") attribute, this method reflects the current sorting of the trajectory. 

---

#### <kbd>property</kbd> Trajectory.se3

Returns SE3 pose list 

---

#### <kbd>property</kbd> Trajectory.sort_switching_index

Returns the index that switches the sorting of the trajectory 

---

#### <kbd>property</kbd> Trajectory.sorting_index

Returns the index that sorts the trajectory 

---

#### <kbd>property</kbd> Trajectory.speed

Returns trajectory speeds calculated using consecutive point distances 

---

#### <kbd>property</kbd> Trajectory.speed_3d

Returns computed speeds or custom speeds 

---

#### <kbd>property</kbd> Trajectory.total_length

Return the total trajectory arc_length. 

---

#### <kbd>property</kbd> Trajectory.xyz

Returns the xyz coordinates of the trajectory 

In contrast to the pos.xyz attribute, this method reflects the current sorting of the trajectory. 



---

### <kbd>method</kbd> `Trajectory.adopt_first_orientation`

```python
adopt_first_orientation(
    trajectory: 'Trajectory',
    inplace: bool = True
) → Trajectory
```

Transform trajectory so that the first orientation is identical in both 



**Args:**
 
 - <b>`trajectory`</b> (Trajectory):  Target Trajectory 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Transformed trajectory 

---

### <kbd>method</kbd> `Trajectory.adopt_first_pose`

```python
adopt_first_pose(trajectory: 'Trajectory', inplace: bool = True) → Trajectory
```

Transform trajectory so that the first pose is identical in both 



**Args:**
 
 - <b>`trajectory`</b> (Trajectory):  Target Trajectory 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Transformed trajectory 

---

### <kbd>method</kbd> `Trajectory.adopt_first_position`

```python
adopt_first_position(
    trajectory: 'Trajectory',
    inplace: bool = True
) → Trajectory
```

Transform trajectory so that the first position is identical in both 



**Args:**
 
 - <b>`trajectory`</b> (Trajectory):  Target Trajectory 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Transformed trajectory 

---

### <kbd>method</kbd> `Trajectory.apply_alignment`

```python
apply_alignment(
    alignment_result: trajectopy.core.alignment.result.AlignmentResult,
    inplace: bool = True
) → Trajectory
```

Transforms trajectory using alignment parameters. 

After computing the alignment parameters needed to align two trajectories, they can be applied to arbitrary trajectories. 



**Args:**
  alignment_result (AlignmentResult) 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Aligned trajectory 

---

### <kbd>method</kbd> `Trajectory.apply_index`

```python
apply_index(
    index: Union[list, numpy.ndarray],
    inplace: bool = True
) → Trajectory
```

Applies index to the trajectory 

This will be done either in-place or using a new instance of a trajectory. The index can be used to filter and / or sort the components of the trajectory. 

Those components are: 
- timestamps (tstamps) 
- positions (xyz) 
- rotations (rot) 
- arc lengths (arc_lengths) 
- sorting index (_sort_index) 



**Args:**
 
 - <b>`index`</b> (Union[list, np.ndarray]):  index that should be applied 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Trajectory with index applied. 

---

### <kbd>method</kbd> `Trajectory.apply_transformation`

```python
apply_transformation(
    transformation: numpy.ndarray,
    inplace: bool = True
) → Trajectory
```

Applies transformation to trajectory 



**Args:**
 
 - <b>`transformation`</b> (np.ndarray):  4x4 Transformation matrix 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Transformed trajectory 

---

### <kbd>method</kbd> `Trajectory.approximate`

```python
approximate(
    approximation_settings: trajectopy.settings.ApproximationSettings = ApproximationSettings(fe_int_size=0.15, fe_min_obs=25, rot_approx_win_size=0.15),
    inplace: bool = True
) → Trajectory
```

Approximates the trajectory using piecewise cubic polynomial. 



**Args:**
 
 - <b>`approximation_settings`</b> (ApproximationSettings):  Approximation settings. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Approximated trajectory. 

---

### <kbd>method</kbd> `Trajectory.copy`

```python
copy() → Trajectory
```

Deep copy of itself 

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

Crops trajectory to timespan defined by t_start and t_end 



**Args:**
 
 - <b>`t_start`</b> (float):  Start timestamp of desired time span 
 - <b>`t_end`</b> (float):  End timestamp of desired time span 
 - <b>`inverse`</b> (bool, optional):  If true, 'crop' turns  into 'cut', i.e. everthing  outside of t_start and t_end  will be removed.  Defaults to False. 
 - <b>`inplace`</b> (bool, optional):  Perform crop in-place.  Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Cropped trajectory 

---

### <kbd>method</kbd> `Trajectory.divide_into_laps`

```python
divide_into_laps(
    sorting_settings: trajectopy.settings.SortingSettings = SortingSettings(discard_missing=True, voxel_size=0.05, movement_threshold=0.005, k_nearest=4),
    return_lap_indices: bool = False
) → Union[List[ForwardRef('Trajectory')], Tuple[List[ForwardRef('Trajectory')], numpy.ndarray]]
```

Divides the trajectory into laps. 



**Args:**
 
 - <b>`sorting_settings`</b> (SortingSettings):  Sorting settings. 



**Returns:**
 
 - <b>`List[Trajectory]`</b>:  List of trajectories, each representing a lap. 

---

### <kbd>classmethod</kbd> `Trajectory.from_file`

```python
from_file(filename: str, io_stream: bool = False) → Trajectory
```

Create trajectory from file 

The file must be a csv file containing columns for at least the timestamp, x, y and z coordinates of the trajectory. Those fields must be named "t", "px", "py" and "pz" in the header using the #fields tag. However, by default a trajectory with "t,px,py,pz,qx,qy,qz,qw" fields is assumed. Additional fields include the arc length, specified by "l", and the speed, specified by "vx", "vy" and "vz". The delimiter can be specified using the #delimiter tag. The default delimiter is a comma. 



**Args:**
 
 - <b>`filename`</b> (str):  path to file 
 - <b>`io_stream`</b> (bool, optional):  If true, the file is read from a stream. 



**Returns:**
 
 - <b>`Trajectory`</b>:  trajectory object 

---

### <kbd>classmethod</kbd> `Trajectory.from_numpy`

```python
from_numpy(
    xyz: numpy.ndarray,
    quat: numpy.ndarray,
    tstamps: numpy.ndarray,
    epsg: int = 0
) → Trajectory
```

Initialize trajectory using numpy arrays 

---

### <kbd>method</kbd> `Trajectory.init_arc_lengths`

```python
init_arc_lengths()
```





---

### <kbd>method</kbd> `Trajectory.interpolate`

```python
interpolate(
    tstamps: Union[list, numpy.ndarray],
    inplace: bool = True
) → Trajectory
```

Interpolates a trajectory to specified timestamps 

This method removes timestamps from tstamps if they lie outside of the timestamp range of the trajectory (self). Since providing values for those timestamps would require an extrapolation and not an interpolation, this behaviour is consistent with the definition of this method. 



**Args:**
 
 - <b>`tstamps`</b> (list):  Interpolation timestamps 
 - <b>`inplace`</b> (bool, optional):  Perform in-place interpolation.  Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Interpolated trajectory 

---

### <kbd>method</kbd> `Trajectory.intersect`

```python
intersect(
    tstamps: numpy.ndarray,
    max_gap_size: float = 2.0,
    inplace: bool = True
) → Trajectory
```

Intersects trajectory with a given timestamp vector 

After intersection, the trajectory covers the same timespan as 'tstamps'. Further, gaps larger than 'max_gap_size' are removed. If two consecutive timespans in tstamps have a difference of more than 'max_gap_size' seconds, they are considered as the limits of a gap. All timestamps of the trajectory that lie within this gap will be removed. 



**Args:**
 
 - <b>`tstamps`</b> (np.ndarray):  Intersection timespans 
 - <b>`max_gap_size`</b> (float, optional):  Maximum allowed gap between timespans.  If Defaults to 0.5. 
 - <b>`inplace`</b> (bool, optional):  Perform intersection in-place.  Defaults to True. 



**Raises:**
 
 - <b>`ValueError`</b>:  If timespans do not overlap. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Intersected trajectory 

---

### <kbd>method</kbd> `Trajectory.match_timestamps`

```python
match_timestamps(tstamps: numpy.ndarray, inplace: bool = True) → Trajectory
```

Truncates trajectory to only those poses where the timestamps exactly match "tstamps" 



**Args:**
 
 - <b>`tstamps`</b> (np.ndarray):  Input timestamps 
 - <b>`inplace`</b> (bool, optional):  Perform matching in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Trajectory with matched timestamps 

---

### <kbd>method</kbd> `Trajectory.sort_spatially`

```python
sort_spatially(
    sorting_settings: trajectopy.settings.SortingSettings = SortingSettings(discard_missing=True, voxel_size=0.05, movement_threshold=0.005, k_nearest=4),
    inplace: bool = True
) → Trajectory
```

Sorts the trajectory spatially. 



**Args:**
 
 - <b>`sorting_settings`</b> (SortingSettings):  Sorting settings. 
 - <b>`inplace`</b> (bool, optional):  Whether to sort the trajectory in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Sorted trajectory. 

---

### <kbd>method</kbd> `Trajectory.to_dataframe`

```python
to_dataframe(sort_by: str = '') → DataFrame
```

Returns a pandas dataframe containing tstamps, xyz, quat and speed_3d of the trajectory. 

The dataframe is sorted by the current sorting attribute (time or arc_length). 



**Args:**
 
 - <b>`sort_by`</b> (str, optional):  Column to sort by. This  overrides the current sort_by  attribute. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  Trajectory as dataframe 

---

### <kbd>method</kbd> `Trajectory.to_file`

```python
to_file(filename: str, mode: str = 'w') → None
```

Writes trajectory to ascii file 



**Args:**
 
 - <b>`filename`</b> (str):  Output filename 

---

### <kbd>method</kbd> `Trajectory.to_kml`

```python
to_kml(filename: str, precision: float = 1e-06) → str
```

Create a KML file from a trajectory. 



**Args:**
 
 - <b>`trajectory`</b> (Trajectory):  Trajectory to be exported. 
 - <b>`filename`</b> (str):  Filename of the KML file. 
 - <b>`precision`</b> (float, optional):  Precision of the exported positions in degree. Defaults to 1e-6. 

---

### <kbd>method</kbd> `Trajectory.to_string`

```python
to_string() → str
```

Writes trajectory to a string instead of a file. 

