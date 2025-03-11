
## <kbd>module</kbd> `trajectopy.alignment`
Trajectopy - Trajectory Evaluation in Python 

Gereon Tombrink, 2025 tombrink@igg.uni-bonn.de 


---

### <kbd>function</kbd> `estimate_alignment`

```python
estimate_alignment(
    traj_from: trajectopy.trajectory.Trajectory,
    traj_to: trajectopy.trajectory.Trajectory,
    alignment_settings: trajectopy.settings.AlignmentSettings = AlignmentSettings(preprocessing=AlignmentPreprocessing(min_speed=0.0, time_start=0.0, time_end=0.0), estimation_settings=AlignmentEstimationSettings(trans_x=True, trans_y=True, trans_z=True, rot_x=True, rot_y=True, rot_z=True, scale=False, time_shift=False, use_x_speed=True, use_y_speed=True, use_z_speed=True, lever_x=False, lever_y=False, lever_z=False, sensor_rotation=False, auto_update=False), stochastics=AlignmentStochastics(std_xy_from=1.0, std_z_from=1.0, std_xy_to=1.0, std_z_to=1.0, std_roll_pitch=0.017453292519943295, std_yaw=0.017453292519943295, std_speed=1.0, error_probability=0.05, variance_estimation=False), metric_threshold=0.0001, time_threshold=0.0001),
    matching_settings: trajectopy.settings.MatchingSettings = MatchingSettings(method=<MatchingMethod.INTERPOLATION: 'interpolation'>, max_time_diff=0.01, max_distance=0.0, k_nearest=10)
) → AlignmentResult
```

Aligns two trajectories 

Performs a 
- Helmert 
- Leverarm 
- Time shift 

estimation depending on the configuration. After this, the estimated parameters are applied to the 'traj_from' trajectory. 



**Args:**
 
- traj_from (Trajectory) 
- traj_to (Trajectory) 
- alignment_settings (AlignmentSettings, optional): Settings for the alignment process. Defaults to AlignmentSettings(). 
- matching_settings (MatchingSettings, optional): Settings for the matching process. Defaults to MatchingSettings(). 



**Returns:**
 
 - <b>`AlignmentResult`</b>:  Result of the alignment process 


