
## <kbd>module</kbd> `trajectopy.tools.sorting`





---

### <kbd>function</kbd> `sort_spatially`

```python
sort_spatially(
    trajectory: trajectopy.trajectory.Trajectory,
    sorting_settings: trajectopy.settings.SortingSettings = SortingSettings(voxel_size=0.05, movement_threshold=0.005, k_nearest=4),
    inplace: bool = True
) → Trajectory
```

Sorts the trajectory spatially. 



**Args:**
 
 - <b>`trajectory`</b> (Trajectory):  Trajectory to sort. 
 - <b>`sorting_settings`</b> (SortingSettings):  Sorting settings. 
 - <b>`inplace`</b> (bool, optional):  Whether to sort the trajectory in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Sorted trajectory. 


---

### <kbd>function</kbd> `divide_into_laps`

```python
divide_into_laps(
    trajectory: trajectopy.trajectory.Trajectory,
    sorting_settings: trajectopy.settings.SortingSettings = SortingSettings(voxel_size=0.05, movement_threshold=0.005, k_nearest=4),
    return_lap_indices: bool = False
) → Union[List[trajectopy.trajectory.Trajectory], Tuple[List[trajectopy.trajectory.Trajectory], numpy.ndarray]]
```

Divides the trajectory into laps. 



**Args:**
 
 - <b>`trajectory`</b> (Trajectory):  Trajectory to divide. 
 - <b>`sorting_settings`</b> (SortingSettings):  Sorting settings. 
 - <b>`return_lap_indices`</b> (bool, optional):  Whether to return lap indices. Defaults to False. 



**Returns:**
 
 - <b>`List[Trajectory]`</b>:  List of trajectories, each representing a lap. 


