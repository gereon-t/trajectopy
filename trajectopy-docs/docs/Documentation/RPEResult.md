
## <kbd>module</kbd> `trajectopy.core.evaluation.rpe_result`
Trajectopy - Trajectory Evaluation in Python 

Gereon Tombrink, 2025 tombrink@igg.uni-bonn.de 



---

### <kbd>class</kbd> `RPEResult`
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

##### <kbd>property</kbd> RPEResult.all_pair_distances





---

##### <kbd>property</kbd> RPEResult.all_rot_devs





---

##### <kbd>property</kbd> RPEResult.columns





---

##### <kbd>property</kbd> RPEResult.drift_factor





---

##### <kbd>property</kbd> RPEResult.dynamic_pos_dict





---

##### <kbd>property</kbd> RPEResult.dynamic_rot_dict





---

##### <kbd>property</kbd> RPEResult.has_rot_dev





---

##### <kbd>property</kbd> RPEResult.mean_pair_distances





---

##### <kbd>property</kbd> RPEResult.num_pairs





---

##### <kbd>property</kbd> RPEResult.pair_distance_unit





---

##### <kbd>property</kbd> RPEResult.pos_dev_all





---

##### <kbd>property</kbd> RPEResult.pos_dev_max





---

##### <kbd>property</kbd> RPEResult.pos_dev_mean





---

##### <kbd>property</kbd> RPEResult.pos_dev_median





---

##### <kbd>property</kbd> RPEResult.pos_dev_min





---

##### <kbd>property</kbd> RPEResult.pos_drift_unit





---

##### <kbd>property</kbd> RPEResult.pos_rpe





---

##### <kbd>property</kbd> RPEResult.pos_std





---

##### <kbd>property</kbd> RPEResult.property_dict

Returns a dictionary containing the properties of the deviation set relevant for time based comparisons. This is the case when pose-pairs are defined by a time difference. 

---

##### <kbd>property</kbd> RPEResult.rot_dev_max





---

##### <kbd>property</kbd> RPEResult.rot_dev_mean





---

##### <kbd>property</kbd> RPEResult.rot_dev_median





---

##### <kbd>property</kbd> RPEResult.rot_dev_min





---

##### <kbd>property</kbd> RPEResult.rot_drift_unit





---

##### <kbd>property</kbd> RPEResult.rot_rpe

Returns the average rotation drift in radians per 100 meters. 

---

##### <kbd>property</kbd> RPEResult.rot_std





---

##### <kbd>property</kbd> RPEResult.step







---

#### <kbd>method</kbd> `RPEResult.compute_metric`

```python
compute_metric(
    key: str,
    func: Callable[[Any], float],
    factor: float = 1.0
) → List[float]
```





---

#### <kbd>classmethod</kbd> `RPEResult.from_file`

```python
from_file(filename: str)
```

Reads a set of relative trajectory deviations from a file. 

---

#### <kbd>method</kbd> `RPEResult.get_all`

```python
get_all(key: str) → List[float]
```





---

#### <kbd>method</kbd> `RPEResult.to_dataframe`

```python
to_dataframe() → DataFrame
```





---

#### <kbd>method</kbd> `RPEResult.to_file`

```python
to_file(filename: str, mode: str = 'a') → None
```






