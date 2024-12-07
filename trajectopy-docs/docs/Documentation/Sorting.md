
## <kbd>class</kbd> `Sorting`








## <kbd>function</kbd> `sort_spatially`

```python
sort_spatially(
    xyz_unsorted: numpy.ndarray,
    settings: trajectopy.core.settings.sorting.SortingSettings = SortingSettings(discard_missing=True, voxel_size=0.05, movement_threshold=0.005, k_nearest=4)
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
