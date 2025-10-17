
## <kbd>module</kbd> `trajectopy.evaluation`
Trajectopy - Trajectory Evaluation in Python 

Gereon Tombrink, 2025 tombrink@igg.uni-bonn.de 


---

### <kbd>function</kbd> `ate`

```python
ate(
    trajectory_gt: trajectopy.trajectory.Trajectory,
    trajectory_est: trajectopy.trajectory.Trajectory,
    settings: trajectopy.settings.ProcessingSettings = ProcessingSettings(alignment=AlignmentSettings(preprocessing=AlignmentPreprocessing(min_speed=0.0, time_start=0.0, time_end=0.0), estimation_settings=AlignmentEstimationSettings(translation_x=True, translation_y=True, translation_z=True, rotation_x=True, rotation_y=True, rotation_z=True, scale=False, time_shift=False, leverarm_x=False, leverarm_y=False, leverarm_z=False, sensor_rotation=False), stochastics=AlignmentStochastics(std_xy_from=1.0, std_z_from=1.0, std_xy_to=1.0, std_z_to=1.0, std_roll_pitch=0.017453292519943295, std_yaw=0.017453292519943295, std_speed=1.0, error_probability=0.05, variance_estimation=False), metric_threshold=0.0001, time_threshold=0.0001), matching=MatchingSettings(method=<MatchingMethod.INTERPOLATION: 'interpolation'>, max_time_diff=0.01, max_distance=0.0, max_gap_size=10.0, k_nearest=10), relative_comparison=RelativeComparisonSettings(pair_min_distance=100.0, pair_max_distance=800.0, pair_distance_step=100.0, pair_distance_unit=<PairDistanceUnit.METER: 'meter'>, use_all_pose_pairs=True), approximation=ApproximationSettings(position_interval_size=0.15, position_min_observations=25, rotation_window_size=0.15), sorting=SortingSettings(voxel_size=0.05, movement_threshold=0.005, k_nearest=4)),
    return_alignment: bool = False,
    align: bool = True
) → Union[trajectopy.core.evaluation.ate_result.ATEResult, Tuple[trajectopy.core.evaluation.ate_result.ATEResult, trajectopy.core.alignment.result.AlignmentResult]]
```

Computes the absolute trajectory error (ATE) between two trajectories. 



**Args:**
 
 - <b>`trajectory_gt`</b> (Trajectory):  Ground truth trajectory. 
 - <b>`trajectory_est`</b> (Trajectory):  Estimated trajectory. 
 - <b>`settings`</b> (ProcessingSettings, optional):  Processing settings. 
 - <b>`return_alignment`</b> (bool, optional):  Whether to return the alignment result. 

Description: 

The ATE is computed by first matching the estimated trajectory to the ground truth trajectory. Then, the alignment between the two trajectories is estimated. The estimated trajectory is aligned to the ground truth trajectory using the estimated alignment. Finally, the ATE is computed by comparing the aligned estimated trajectory to the ground truth trajectory. 



**Returns:**
 
 - <b>`ATEResult`</b>:  Result of the ATE computation. 


---

### <kbd>function</kbd> `rpe`

```python
rpe(
    trajectory_gt: trajectopy.trajectory.Trajectory,
    trajectory_est: trajectopy.trajectory.Trajectory,
    settings: trajectopy.settings.ProcessingSettings = ProcessingSettings(alignment=AlignmentSettings(preprocessing=AlignmentPreprocessing(min_speed=0.0, time_start=0.0, time_end=0.0), estimation_settings=AlignmentEstimationSettings(translation_x=True, translation_y=True, translation_z=True, rotation_x=True, rotation_y=True, rotation_z=True, scale=False, time_shift=False, leverarm_x=False, leverarm_y=False, leverarm_z=False, sensor_rotation=False), stochastics=AlignmentStochastics(std_xy_from=1.0, std_z_from=1.0, std_xy_to=1.0, std_z_to=1.0, std_roll_pitch=0.017453292519943295, std_yaw=0.017453292519943295, std_speed=1.0, error_probability=0.05, variance_estimation=False), metric_threshold=0.0001, time_threshold=0.0001), matching=MatchingSettings(method=<MatchingMethod.INTERPOLATION: 'interpolation'>, max_time_diff=0.01, max_distance=0.0, max_gap_size=10.0, k_nearest=10), relative_comparison=RelativeComparisonSettings(pair_min_distance=100.0, pair_max_distance=800.0, pair_distance_step=100.0, pair_distance_unit=<PairDistanceUnit.METER: 'meter'>, use_all_pose_pairs=True), approximation=ApproximationSettings(position_interval_size=0.15, position_min_observations=25, rotation_window_size=0.15), sorting=SortingSettings(voxel_size=0.05, movement_threshold=0.005, k_nearest=4))
) → RPEResult
```

Computes the relative pose error (RPE) between two trajectories. 



**Args:**
 
 - <b>`trajectory_gt`</b> (Trajectory):  Ground truth trajectory. 
 - <b>`trajectory_est`</b> (Trajectory):  Estimated trajectory. 
 - <b>`settings`</b> (ProcessingSettings, optional):  Processing settings. 

Description: 

The RPE is computed by comparing the relative poses between the estimated and ground truth trajectories. The pose distances are either defined in meters or in seconds depending on the settings. 



**Returns:**
 
 - <b>`RPEResult`</b>:  Result of the RPE computation. 


