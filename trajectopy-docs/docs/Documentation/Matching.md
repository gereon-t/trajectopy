
## <kbd>function</kbd> `match_trajectories`

```python
match_trajectories(
    traj_from: trajectopy.core.trajectory.Trajectory,
    traj_to: trajectopy.core.trajectory.Trajectory,
    settings: trajectopy.core.settings.matching.MatchingSettings = MatchingSettings(method=<MatchingMethod.INTERPOLATION: 'interpolation'>, max_time_diff=0.01, max_distance=0.0, k_nearest=10),
    inplace: bool = True
) → Tuple[trajectopy.core.trajectory.Trajectory, trajectopy.core.trajectory.Trajectory]
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
