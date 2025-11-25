
## <kbd>module</kbd> `trajectopy.processing.approximation`





---

### <kbd>function</kbd> `approximate_cubic`

```python
approximate_cubic(
    trajectory: trajectopy.core.trajectory.Trajectory,
    approximation_settings: trajectopy.core.settings.CubicApproximationSettings = CubicApproximationSettings(position_interval_size=0.15, position_min_observations=25, rotation_window_size=0.15),
    inplace: bool = False
) → Trajectory
```

Approximate the trajectory using piecewise cubic polynomials based on the provided settings. 



**Args:**
 
 - <b>`traj`</b> (Trajectory):  The trajectory to approximate. 
 - <b>`settings`</b> (ApproximationSettings):  Settings for the approximation. 
 - <b>`inplace`</b> (bool):  If True, modify the trajectory in place. Defaults to False. 



**Returns:**
 
 - <b>`Trajectory`</b>:  The approximated trajectory. 


