
## <kbd>function</kbd> `estimate_alignment`

```python
estimate_alignment(
    traj_from: trajectopy.core.trajectory.Trajectory,
    traj_to: trajectopy.core.trajectory.Trajectory,
    alignment_settings: trajectopy.core.settings.alignment.AlignmentSettings = AlignmentSettings(preprocessing=AlignmentPreprocessing(min_speed=0.0, time_start=0.0, time_end=0.0), estimation_settings=AlignmentEstimationSettings(trans_x=True, trans_y=True, trans_z=True, rot_x=True, rot_y=True, rot_z=True, scale=False, time_shift=False, use_x_speed=True, use_y_speed=True, use_z_speed=True, lever_x=False, lever_y=False, lever_z=False, sensor_rotation=False, auto_update=False), stochastics=AlignmentStochastics(std_xy_from=1.0, std_z_from=1.0, std_xy_to=1.0, std_z_to=1.0, std_roll_pitch=0.017453292519943295, std_yaw=0.017453292519943295, std_speed=1.0, error_probability=0.05, variance_estimation=False), metric_threshold=0.0001, time_threshold=0.0001),
    matching_settings: trajectopy.core.settings.matching.MatchingSettings = MatchingSettings(method=<MatchingMethod.INTERPOLATION: 'interpolation'>, max_time_diff=0.01, max_distance=0.0, k_nearest=10)
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

## <kbd>class</kbd> `AlignmentResult`
Class to store the result of an alignment. 



**Attributes:**
 
 - <b>`name`</b> (str):  Name of the result. 
 - <b>`position_parameters`</b> (AlignmentParameters):  Estimated position alignment parameters. 
 - <b>`rotation_parameters`</b> (SensorRotationParameters):  Estimated rotation alignment parameters. 
 - <b>`estimation_of`</b> (AlignmentEstimationSettings):  Settings defining which parameters were estimated. 
 - <b>`converged`</b> (bool):  Whether the estimation converged. 

### <kbd>method</kbd> `AlignmentResult.__init__`

```python
__init__(
    name: str = 'Alignment Result',
    position_parameters: trajectopy.core.alignment.parameters.AlignmentParameters = <factory>,
    rotation_parameters: trajectopy.core.alignment.parameters.SensorRotationParameters = <factory>,
    estimation_of: trajectopy.core.settings.alignment.AlignmentEstimationSettings = <factory>,
    converged: bool = True
) → None
```








---

### <kbd>classmethod</kbd> `AlignmentResult.from_file`

```python
from_file(filename: str) → AlignmentResult
```

Load the result from a file. 



**Args:**
 
 - <b>`filename`</b> (str):  Path to the file. 



**Returns:**
 
 - <b>`AlignmentResult`</b>:  The loaded result. 

---

### <kbd>method</kbd> `AlignmentResult.to_file`

```python
to_file(filename: str) → None
```

Save the result to a file. 



**Args:**
 
 - <b>`filename`</b> (str):  Path to the file. 


## <kbd>class</kbd> `AlignmentData`
Class holding the observation data required for Alignment 





**Attributes:**
 
 - <b>`traj_from`</b> (Trajectory):  Source trajectory 
 - <b>`traj_to`</b> (Trajectory):  Target trajectory 
 - <b>`alignment_settings`</b> (AlignmentSettings):  Settings for the alignment 
 - <b>`matching_settings`</b> (MatchingSettings):  Settings for the matching 

### <kbd>method</kbd> `AlignmentData.__init__`

```python
__init__(
    traj_from: trajectopy.core.trajectory.Trajectory,
    traj_to: trajectopy.core.trajectory.Trajectory,
    alignment_settings: trajectopy.core.settings.alignment.AlignmentSettings,
    matching_settings: trajectopy.core.settings.matching.MatchingSettings
) → None
```






---

#### <kbd>property</kbd> AlignmentData.est_euler_x





---

#### <kbd>property</kbd> AlignmentData.est_euler_y





---

#### <kbd>property</kbd> AlignmentData.est_euler_z





---

#### <kbd>property</kbd> AlignmentData.est_obs_vector





---

#### <kbd>property</kbd> AlignmentData.est_rpy_from





---

#### <kbd>property</kbd> AlignmentData.est_speed





---

#### <kbd>property</kbd> AlignmentData.est_speed_x





---

#### <kbd>property</kbd> AlignmentData.est_speed_y





---

#### <kbd>property</kbd> AlignmentData.est_speed_z





---

#### <kbd>property</kbd> AlignmentData.est_x_from





---

#### <kbd>property</kbd> AlignmentData.est_x_to





---

#### <kbd>property</kbd> AlignmentData.est_xyz_from





---

#### <kbd>property</kbd> AlignmentData.est_xyz_to





---

#### <kbd>property</kbd> AlignmentData.est_y_from





---

#### <kbd>property</kbd> AlignmentData.est_y_to





---

#### <kbd>property</kbd> AlignmentData.est_z_from





---

#### <kbd>property</kbd> AlignmentData.est_z_to





---

#### <kbd>property</kbd> AlignmentData.euler_x





---

#### <kbd>property</kbd> AlignmentData.euler_y





---

#### <kbd>property</kbd> AlignmentData.euler_z





---

#### <kbd>property</kbd> AlignmentData.group_stds

Returns the mean standard deviation for each group 

---

#### <kbd>property</kbd> AlignmentData.num_obs_per_epoch

Returns the number of observations per epoch depending on the enabled estimation modes. 

---

#### <kbd>property</kbd> AlignmentData.number_of_epochs





---

#### <kbd>property</kbd> AlignmentData.obs_vector





---

#### <kbd>property</kbd> AlignmentData.res_vector





---

#### <kbd>property</kbd> AlignmentData.rpy_from





---

#### <kbd>property</kbd> AlignmentData.sigma_ll





---

#### <kbd>property</kbd> AlignmentData.speed





---

#### <kbd>property</kbd> AlignmentData.speed_x





---

#### <kbd>property</kbd> AlignmentData.speed_y





---

#### <kbd>property</kbd> AlignmentData.speed_z





---

#### <kbd>property</kbd> AlignmentData.tstamps





---

#### <kbd>property</kbd> AlignmentData.var_vector





---

#### <kbd>property</kbd> AlignmentData.x_from





---

#### <kbd>property</kbd> AlignmentData.x_to





---

#### <kbd>property</kbd> AlignmentData.xyz_from





---

#### <kbd>property</kbd> AlignmentData.xyz_to





---

#### <kbd>property</kbd> AlignmentData.y_from





---

#### <kbd>property</kbd> AlignmentData.y_to





---

#### <kbd>property</kbd> AlignmentData.z_from





---

#### <kbd>property</kbd> AlignmentData.z_to







---

### <kbd>method</kbd> `AlignmentData.build_obs_vector`

```python
build_obs_vector(
    xyz_from: numpy.ndarray,
    xyz_to: numpy.ndarray,
    rot_from: Optional[trajectopy.core.rotationset.RotationSet],
    speed: Optional[numpy.ndarray]
) → ndarray
```

Creates the observation vector required for the alignment adjustment. 



**Args:**
 
 - <b>`xyz_from`</b> (np.ndarray):  Positions to be aligned. 
 - <b>`xyz_to`</b> (np.ndarray):  Target Positions. 
 - <b>`rpy_from`</b> (np.ndarray):  The roll, pitch, and yaw angles of the platform.  Those angles should describe the rotation of the  body-fixed coordinate system with respect to the  inertial coordinate system. 



**Returns:**
 
 - <b>`np.ndarray`</b>:  The observation vector required for the alignment adjustment. 

---

### <kbd>method</kbd> `AlignmentData.build_res_vector`

```python
build_res_vector() → ndarray
```





---

### <kbd>method</kbd> `AlignmentData.build_var_vector`

```python
build_var_vector() → ndarray
```

Sets up the variance vector 

Its size depends on whether the leverarm should be estimated or not. In this case, not only the source and the target positions are relevant but also the platform orientations. Also, when estimating the time shift, the platform speed is also considered. 



**Returns:**
 
 - <b>`np.ndarray`</b>:  variance vector 

---

### <kbd>method</kbd> `AlignmentData.get_est_obs_group`

```python
get_est_obs_group(key: str) → Tuple[numpy.ndarray, ...]
```





---

### <kbd>method</kbd> `AlignmentData.get_obs_group`

```python
get_obs_group(key: str) → Tuple[numpy.ndarray, ...]
```





---

### <kbd>method</kbd> `AlignmentData.get_res_group`

```python
get_res_group(key: str) → Tuple[numpy.ndarray, ...]
```





---

### <kbd>method</kbd> `AlignmentData.get_var_group`

```python
get_var_group(key: str) → Tuple[numpy.ndarray, ...]
```





---

### <kbd>method</kbd> `AlignmentData.get_variance_estimation_subset`

```python
get_variance_estimation_subset(num_obs: int = 200) → AlignmentData
```

Returns a subset of the alignment data for variance estimation 

This method will return a subset of the alignment data that contains 'num_obs' observations. The subset is selected by finding the epoch with the highest standard deviation of the observations. The idea is that a higher variance in the observations will lead to a better estimation of the variances. 

The motivation behind this is that the variance estimation is memory and time consuming. Therefore, a subset of the observations is used for the estimation. 



**Args:**
 
 - <b>`num_obs`</b> (int, optional):  Subet size. Defaults to 200. 



**Returns:**
 
 - <b>`AlignmentData`</b>:  Cropped alignment data 

---

### <kbd>method</kbd> `AlignmentData.set_obs_group`

```python
set_obs_group(key: str, values: numpy.ndarray) → None
```





---

### <kbd>method</kbd> `AlignmentData.set_res_group`

```python
set_res_group(key: str, values: numpy.ndarray) → None
```





---

### <kbd>method</kbd> `AlignmentData.set_var_group`

```python
set_var_group(key: str, values: numpy.ndarray) → None
```





---

### <kbd>method</kbd> `AlignmentData.setup`

```python
setup() → None
```

Prepare two trajectories for alignment. 

This method will filter the trajectories by speed and resample both trajectories to the same sampling. 

