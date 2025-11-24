
## <kbd>module</kbd> `trajectopy.tools.interpolation`





---

### <kbd>function</kbd> `interpolate`

```python
interpolate(
    trajectory: trajectopy.trajectory.Trajectory,
    timestamps: Union[list, numpy.ndarray],
    method: trajectopy.settings.InterpolationMethod = <InterpolationMethod.LINEAR: 'linear'>,
    inplace: bool = True
) → Trajectory
```

Interpolates a trajectory to specified timestamps using the given method. 



**Args:**
 
 - <b>`trajectory`</b> (Trajectory):  Trajectory to interpolate 
 - <b>`timestamps`</b> (list):  Interpolation timestamps 
 - <b>`method`</b> (InterpolationMethod, optional):  Interpolation method. Defaults to "linear". 
 - <b>`inplace`</b> (bool, optional):  Perform in-place interpolation. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Interpolated trajectory 


