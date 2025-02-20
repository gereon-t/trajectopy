
## <kbd>class</kbd> `ATEResult`
This class represents a set of absolute trajectory deviations 

Absolute trajectory deviations describe absolute pose deviations between two trajectories. The deviations are calculated by comparing pairs of positions and orientations in the test and reference trajectory. 



**Attributes:**
 
 - <b>`trajectory`</b> (Trajectory):  The trajectory the deviations are belonging to 
 - <b>`abs_dev`</b> (AbsoluteTrajectoryDeviations):  The absolute trajectory deviations 
 - <b>`name`</b> (str):  The name of the result 

### <kbd>method</kbd> `ATEResult.__init__`

```python
__init__(
    trajectory: trajectopy.core.trajectory.Trajectory,
    abs_dev: trajectopy.core.evaluation.deviations.AbsoluteTrajectoryDeviations,
    name: str = ''
) → None
```






---

#### <kbd>property</kbd> ATEResult.columns

Returns the column names of the dataframe 

---

#### <kbd>property</kbd> ATEResult.has_orientation

Returns True if orientation is available 

---

#### <kbd>property</kbd> ATEResult.pos_ate

Returns mean of 3d position deviations 

---

#### <kbd>property</kbd> ATEResult.pos_bias_along

Returns along track bias 

---

#### <kbd>property</kbd> ATEResult.pos_bias_cross_h

Returns horizontal cross track bias 

---

#### <kbd>property</kbd> ATEResult.pos_bias_cross_v

Returns vertical cross track bias 

---

#### <kbd>property</kbd> ATEResult.pos_bias_x

Returns x bias 

---

#### <kbd>property</kbd> ATEResult.pos_bias_y

Returns y bias 

---

#### <kbd>property</kbd> ATEResult.pos_bias_z

Returns z bias 

---

#### <kbd>property</kbd> ATEResult.pos_dev_along

Returns deviations of along track deviations 

---

#### <kbd>property</kbd> ATEResult.pos_dev_comb

Returns position deviations combined using the L2 norm 

---

#### <kbd>property</kbd> ATEResult.pos_dev_cross_h

Returns deviations of horizontal cross track deviations 

---

#### <kbd>property</kbd> ATEResult.pos_dev_cross_v

Returns deviations of vertical cross track deviations 

---

#### <kbd>property</kbd> ATEResult.pos_dev_max

Returns max of 3d position deviations 

---

#### <kbd>property</kbd> ATEResult.pos_dev_median

Returns min of 3d position deviations 

---

#### <kbd>property</kbd> ATEResult.pos_dev_min

Returns min of 3d position deviations 

---

#### <kbd>property</kbd> ATEResult.pos_dev_rms

Returns RMS of 3d positions 

---

#### <kbd>property</kbd> ATEResult.pos_dev_std

Returns std of 3d position deviations 

---

#### <kbd>property</kbd> ATEResult.pos_dev_x

Returns x deviations 

---

#### <kbd>property</kbd> ATEResult.pos_dev_y

Returns y deviations 

---

#### <kbd>property</kbd> ATEResult.pos_dev_z

Returns z deviations 

---

#### <kbd>property</kbd> ATEResult.pos_rms_along

Returns RMS of along track deviations 

---

#### <kbd>property</kbd> ATEResult.pos_rms_cross_h

Returns RMS of horizontal cross track deviations 

---

#### <kbd>property</kbd> ATEResult.pos_rms_cross_v

Returns RMS of vertical cross track deviations 

---

#### <kbd>property</kbd> ATEResult.pos_rms_x

Returns RMS of x deviations 

---

#### <kbd>property</kbd> ATEResult.pos_rms_y

Returns RMS of y deviations 

---

#### <kbd>property</kbd> ATEResult.pos_rms_z

Returns RMS of z deviations 

---

#### <kbd>property</kbd> ATEResult.property_dict





---

#### <kbd>property</kbd> ATEResult.rot_ate

Returns mean of rotation deviations 

---

#### <kbd>property</kbd> ATEResult.rot_bias_x

Returns roll bias 

---

#### <kbd>property</kbd> ATEResult.rot_bias_y

Returns pitch bias 

---

#### <kbd>property</kbd> ATEResult.rot_bias_z

Returns yaw bias 

---

#### <kbd>property</kbd> ATEResult.rot_dev_comb

Returns rotation deviations as single rotation angles 

---

#### <kbd>property</kbd> ATEResult.rot_dev_max

Returns max of rotations 

---

#### <kbd>property</kbd> ATEResult.rot_dev_median

Returns median of rotations 

---

#### <kbd>property</kbd> ATEResult.rot_dev_min

Returns min of rotations 

---

#### <kbd>property</kbd> ATEResult.rot_dev_rms

Returns RMS of rotations 

---

#### <kbd>property</kbd> ATEResult.rot_dev_std

Returns STD of rotations 

---

#### <kbd>property</kbd> ATEResult.rot_dev_x

Returns roll deviations 

---

#### <kbd>property</kbd> ATEResult.rot_dev_y

Returns pitch deviations 

---

#### <kbd>property</kbd> ATEResult.rot_dev_z

Returns yaw deviations 

---

#### <kbd>property</kbd> ATEResult.rot_rms_x

Returns RMS of roll deviations 

---

#### <kbd>property</kbd> ATEResult.rot_rms_y

Returns RMS of pitch deviations 

---

#### <kbd>property</kbd> ATEResult.rot_rms_z

Returns RMS of yaw deviations 


---

#### <kbd>handler</kbd> ATEResult.rot_bias_xyz

---

#### <kbd>handler</kbd> ATEResult.rot_dev_xyz


---

### <kbd>classmethod</kbd> `ATEResult.from_file`

```python
from_file(filename: str)
```





---

### <kbd>method</kbd> `ATEResult.to_dataframe`

```python
to_dataframe() → DataFrame
```

Exports results as pandas dataframe 

---

### <kbd>method</kbd> `ATEResult.to_file`

```python
to_file(filename: str, mode: str = 'a') → None
```

Exports results as csv 


## <kbd>class</kbd> `RPEResult`
This class represents a set of relative trajectory deviations 

Relative trajectory deviations describe relative pose deviations between two trajectories. The deviations are calculated by comparing pairs of positions and orientations in the test and reference trajectory. 



**Attributes:**
 
 - <b>`rpe_dev`</b> (RelativeTrajectoryDeviations):  The relative trajectory deviations. 
 - <b>`name`</b> (str):  Name of the deviation set. 

### <kbd>method</kbd> `RPEResult.__init__`

```python
__init__(
    rpe_dev: trajectopy.core.evaluation.deviations.RelativeTrajectoryDeviations,
    name: str
) → None
```






---

#### <kbd>property</kbd> RPEResult.all_pair_distances





---

#### <kbd>property</kbd> RPEResult.all_rot_devs





---

#### <kbd>property</kbd> RPEResult.columns





---

#### <kbd>property</kbd> RPEResult.drift_factor





---

#### <kbd>property</kbd> RPEResult.dynamic_pos_dict





---

#### <kbd>property</kbd> RPEResult.dynamic_rot_dict





---

#### <kbd>property</kbd> RPEResult.has_rot_dev





---

#### <kbd>property</kbd> RPEResult.mean_pair_distances





---

#### <kbd>property</kbd> RPEResult.num_pairs





---

#### <kbd>property</kbd> RPEResult.pair_distance_unit





---

#### <kbd>property</kbd> RPEResult.pos_dev_all





---

#### <kbd>property</kbd> RPEResult.pos_dev_max





---

#### <kbd>property</kbd> RPEResult.pos_dev_mean





---

#### <kbd>property</kbd> RPEResult.pos_dev_median





---

#### <kbd>property</kbd> RPEResult.pos_dev_min





---

#### <kbd>property</kbd> RPEResult.pos_drift_unit





---

#### <kbd>property</kbd> RPEResult.pos_rpe





---

#### <kbd>property</kbd> RPEResult.pos_std





---

#### <kbd>property</kbd> RPEResult.property_dict

Returns a dictionary containing the properties of the deviation set relevant for time based comparisons. This is the case when pose-pairs are defined by a time difference. 

---

#### <kbd>property</kbd> RPEResult.rot_dev_max





---

#### <kbd>property</kbd> RPEResult.rot_dev_mean





---

#### <kbd>property</kbd> RPEResult.rot_dev_median





---

#### <kbd>property</kbd> RPEResult.rot_dev_min





---

#### <kbd>property</kbd> RPEResult.rot_drift_unit





---

#### <kbd>property</kbd> RPEResult.rot_rpe

Returns the average rotation drift in radians per 100 meters. 

---

#### <kbd>property</kbd> RPEResult.rot_std





---

#### <kbd>property</kbd> RPEResult.step







---

### <kbd>method</kbd> `RPEResult.compute_metric`

```python
compute_metric(
    key: str,
    func: Callable[[Any], float],
    factor: float = 1.0
) → List[float]
```





---

### <kbd>classmethod</kbd> `RPEResult.from_file`

```python
from_file(filename: str)
```

Reads a set of relative trajectory deviations from a file. 

---

### <kbd>method</kbd> `RPEResult.get_all`

```python
get_all(key: str) → List[float]
```





---

### <kbd>method</kbd> `RPEResult.to_dataframe`

```python
to_dataframe() → DataFrame
```





---

### <kbd>method</kbd> `RPEResult.to_file`

```python
to_file(filename: str, mode: str = 'a') → None
```





