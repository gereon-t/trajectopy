
## <kbd>module</kbd> `trajectopy.matching`
Trajectopy - Trajectory Evaluation in Python 

Gereon Tombrink, 2025 tombrink@igg.uni-bonn.de 


---

### <kbd>function</kbd> `match_trajectories`

```python
match_trajectories(
    traj_from: trajectopy.trajectory.Trajectory,
    traj_to: trajectopy.trajectory.Trajectory,
    settings: trajectopy.settings.MatchingSettings = MatchingSettings(method=<MatchingMethod.INTERPOLATION: 'interpolation'>, max_time_diff=0.01, max_distance=0.0, k_nearest=10),
    inplace: bool = True
) → Tuple[trajectopy.trajectory.Trajectory, trajectopy.trajectory.Trajectory]
```

Matches two trajectories using the specified method 



**Args:**
 
 - <b>`traj_from`</b> (Trajectory):  Trajectory to match 
 - <b>`traj_to`</b> (Trajectory):  Reference trajectory 
 - <b>`settings`</b> (MatchingSettings, optional):  Matching settings. Defaults to MatchingSettings(). 
 - <b>`inplace`</b> (bool, optional):  Whether to modify the input trajectories. Defaults to True. 

Supported methods: 


- MatchingMethod.INTERPOLATION: Interpolates the test trajectory onto the reference trajectory using its timestamps 
- MatchingMethod.NEAREST_TEMPORAL: Find the nearest temporal match without interpolation 
- MatchingMethod.NEAREST_SPATIAL: Find the nearest spatial match without interpolation 
- MatchingMethod.NEAREST_SPATIAL_INTERPOLATED: Find the nearest n spatial matches and spatially interpolate using a 3d line 

Nearest Spatial 

This method matches two trajectories by finding the nearest pose in the target trajectory for each pose in the source trajectory. The distance between two poses is computed using the Euclidean distance between their positions. 

Nearest Temporal 

This method matches two trajectories using their timestamps by finding the nearest timestamp in the target trajectory for each timestamp in the source trajectory. 

Interpolation 

This method matches two trajectories by interpolating the timestamps of one trajectory to the timestamps of the other trajectory. The interpolation is linear for both positions and rotations (SLERP). 

Nearest Spatial Interpolated 

This method matches both trajectories spatially by requesting the nearest k positions from the reference trajectory for each pose in the test trajectory. Then, an interpolation is performed using a 3d line fit of the k nearest positions. After this operation, both trajectories will have the length of the test trajectory. This method does not support rotation matching. 



**Returns:**
 
 - <b>`Tuple[Trajectory, Trajectory]`</b>:  Matched trajectories 


---

### <kbd>function</kbd> `match_trajectories_interpolation`

```python
match_trajectories_interpolation(
    traj_test: trajectopy.trajectory.Trajectory,
    traj_ref: trajectopy.trajectory.Trajectory
) → Tuple[trajectopy.trajectory.Trajectory, trajectopy.trajectory.Trajectory]
```

Ensures that both trajectories are sampled in the same way 

This method will intersect both trajectories with each other and then approximate the trajectory with the higher data rate onto the other trajectory. The sorting and the arc lengths of both trajectories are identical after the call of this method. 



**Args:**
 
 - <b>`traj_test`</b> (Trajectory):  Test trajectory 
 - <b>`traj_ref`</b> (Trajectory):  Reference trajectory 



**Returns:**
 
 - <b>`Tuple[Trajectory, Trajectory]`</b>:  Both trajectories with the  same sampling. The instance  which called this method is  the first returned trajectory. 


---

### <kbd>function</kbd> `match_trajectories_temporal`

```python
match_trajectories_temporal(
    traj_test: trajectopy.trajectory.Trajectory,
    traj_ref: trajectopy.trajectory.Trajectory,
    max_distance: float = 0.01
) → Tuple[trajectopy.trajectory.Trajectory, trajectopy.trajectory.Trajectory]
```

This method matches both trajectories temporally 

After this operation, both trajectories will have the length of the test trajectory. This means, that the reference trajectory may be modified. 



**Args:**
 
 - <b>`traj_test`</b> (Trajectory):  Test trajectory 
 - <b>`traj_ref`</b> (Trajectory):  Reference trajectory 
 - <b>`max_distance`</b> (float, optional):  Maximum distance between two timestamps.  Defaults to 0.1. 



**Returns:**
 
 - <b>`Tuple[Trajectory, Trajectory]`</b>:  Matched trajectories 


---

### <kbd>function</kbd> `match_trajectories_spatial`

```python
match_trajectories_spatial(
    traj_test: trajectopy.trajectory.Trajectory,
    traj_ref: trajectopy.trajectory.Trajectory,
    max_distance: float = 0.0
) → Tuple[trajectopy.trajectory.Trajectory, trajectopy.trajectory.Trajectory]
```

This method matches both trajectories spatially 

After this operation, both trajectories will have the length of the test trajectory. This means, that the reference trajectory may be modified. 



**Args:**
 
 - <b>`traj_from`</b> (Trajectory):  Test trajectory 
 - <b>`traj_to`</b> (Trajectory):  Reference trajectory 
 - <b>`max_distance`</b> (float, optional):  Maximum distance between two poses.  Defaults to None. This means all  matches are accepted. 



**Returns:**
 
 - <b>`Tuple[Trajectory, Trajectory]`</b>:  Matched trajectories 


---

### <kbd>function</kbd> `match_trajectories_spatial_interpolation`

```python
match_trajectories_spatial_interpolation(
    traj_test: trajectopy.trajectory.Trajectory,
    traj_ref: trajectopy.trajectory.Trajectory,
    max_distance: float = 0.0,
    k_nearest: int = 10
) → Tuple[trajectopy.trajectory.Trajectory, trajectopy.trajectory.Trajectory]
```

This method matches both trajectories spatially by requesting the nearest two poses from the reference trajectory for each pose in the test trajectory. Then, an interpolation is performed between the two nearest poses. 

After this operation, both trajectories will have the length of the test trajectory. This means, that the reference trajectory may be modified. 



**Args:**
 
 - <b>`traj_from`</b> (Trajectory):  Test trajectory 
 - <b>`traj_to`</b> (Trajectory):  Reference trajectory 
 - <b>`max_distance`</b> (float, optional):  Maximum distance between two poses.  Defaults to None. This means all  matches are accepted. 
 - <b>`k_nearest`</b> (int, optional):  Number of nearest poses to request from  the reference trajectory. Defaults to 10. 



**Returns:**
 
 - <b>`Tuple[Trajectory, Trajectory]`</b>:  Matched trajectories 


---

### <kbd>function</kbd> `rough_timestamp_matching`

```python
rough_timestamp_matching(
    traj_ref: trajectopy.trajectory.Trajectory,
    traj_test: trajectopy.trajectory.Trajectory,
    max_distance: float = 0.0
) → float
```

This method roughly matches two trajectories temporally 

**Args:**
 
 - <b>`traj_from`</b> (Trajectory):  Test trajectory 
 - <b>`traj_to`</b> (Trajectory):  Reference trajectory 



**Returns:**
 
 - <b>`float`</b>:  Mean time offset 


