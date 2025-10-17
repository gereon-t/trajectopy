
## <kbd>module</kbd> `trajectopy.sorting`
Trajectopy - Trajectory Evaluation in Python 

Gereon Tombrink, 2025 tombrink@igg.uni-bonn.de 


---

### <kbd>function</kbd> `sort_spatially`

```python
sort_spatially(
    xyz_unsorted: numpy.ndarray,
    settings: trajectopy.settings.SortingSettings = SortingSettings(voxel_size=0.05, movement_threshold=0.005, k_nearest=4)
) → Tuple[List[int], numpy.ndarray]
```

Reconstructs the spatial sorting of the given points 

Spatially sorts the positions by constructing the minimum-spanning-tree of the positions. Finally, by performing up to 3 breadth-first-searches within the mst, the spatial sorting can be reconstructed 

This functionality is only useful if the positions describe a closed loop without intersections. 

This method can also take care of inserting missing points and assures that the direction of travel is kept during sorting. 



**Args:**
 
 - <b>`xyz_unsorted`</b> (np.ndarray):  unsorted positions 
 - <b>`settings`</b> (SortingSettings, optional):  sorting settings. Defaults to SortingSettings(). 



**Returns:**
 
 - <b>`list`</b>:  Sort index 
 - <b>`np.ndarray`</b>:  Arc lengths along the path 


---

### <kbd>function</kbd> `detect_direction`

```python
detect_direction(xyz: numpy.ndarray) → int
```

Detects the direction of travel using polar coordinates 

This only works if the points describe a somewhat circular trajectory which circulates around the origin of the coordinate frame. 



**Args:**
 
 - <b>`xyz`</b> (np.ndarray):  input positions 



**Returns:**
 
 - <b>`int`</b>:  -1 / 1 indicating direction of travel 


---

### <kbd>function</kbd> `complete_lap_dist`

```python
complete_lap_dist(xyz, dist_th: float = 0.5) → bool
```

Function to determine if lap is complete A lap is considered as complete, if the distance between the starting point and the end point is below a specified distance 


---

### <kbd>class</kbd> `Sorting`








