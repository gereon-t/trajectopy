
## <kbd>function</kbd> `merge_trajectories`

```python
merge_trajectories(
    trajectories: List[trajectopy.core.trajectory.Trajectory]
) → Trajectory
```

Merges a list of trajectories into one trajectory. 

This function ignores EPSG codes and merges the trajectories based on their timestamps. Therefore, all trajectories should be in the same coordinate system before merging. 



**Args:**
 
 - <b>`list[Trajectory]`</b>:  List of trajectories to merge. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Merged trajectory. 
