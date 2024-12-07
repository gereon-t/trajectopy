
# <kbd>class</kbd> `ATEResult`
This class represents a set of absolute trajectory deviations 

Absolute trajectory deviations describe absolute pose deviations between two trajectories. The deviations are calculated by comparing pairs of positions and orientations in the test and reference trajectory. 



**Attributes:**
 
 - <b>`trajectory`</b> (Trajectory):  The trajectory the deviations are belonging to 
 - <b>`abs_dev`</b> (AbsoluteTrajectoryDeviations):  The absolute trajectory deviations 
 - <b>`name`</b> (str):  The name of the result 

### <kbd>method</kbd> `ATEResult.__init__`

```python
__init__(
    trajectory: trajectopy.core.trajectory.Trajectory,
    abs_dev: trajectopy.core.evaluation.deviations.AbsoluteTrajectoryDeviations,
    name: str = ''
) → None
```






---

### <kbd>property</kbd> ATEResult.columns

Returns the column names of the dataframe 

---

### <kbd>property</kbd> ATEResult.has_orientation

Returns True if orientation is available 

---

### <kbd>property</kbd> ATEResult.pos_ate

Returns mean of 3d position deviations 

---

### <kbd>property</kbd> ATEResult.pos_bias_along

Returns along track bias 

---

### <kbd>property</kbd> ATEResult.pos_bias_cross_h

Returns horizontal cross track bias 

---

### <kbd>property</kbd> ATEResult.pos_bias_cross_v

Returns vertical cross track bias 

---

### <kbd>property</kbd> ATEResult.pos_bias_x

Returns x bias 

---

### <kbd>property</kbd> ATEResult.pos_bias_y

Returns y bias 

---

### <kbd>property</kbd> ATEResult.pos_bias_z

Returns z bias 

---

### <kbd>property</kbd> ATEResult.pos_dev_along

Returns deviations of along track deviations 

---

### <kbd>property</kbd> ATEResult.pos_dev_comb

Returns position deviations combined using the L2 norm 

---

### <kbd>property</kbd> ATEResult.pos_dev_cross_h

Returns deviations of horizontal cross track deviations 

---

### <kbd>property</kbd> ATEResult.pos_dev_cross_v

Returns deviations of vertical cross track deviations 

---

### <kbd>property</kbd> ATEResult.pos_dev_max

Returns max of 3d position deviations 

---

### <kbd>property</kbd> ATEResult.pos_dev_median

Returns min of 3d position deviations 

---

### <kbd>property</kbd> ATEResult.pos_dev_min

Returns min of 3d position deviations 

---

### <kbd>property</kbd> ATEResult.pos_dev_rms

Returns RMS of 3d positions 

---

### <kbd>property</kbd> ATEResult.pos_dev_std

Returns std of 3d position deviations 

---

### <kbd>property</kbd> ATEResult.pos_dev_x

Returns x deviations 

---

### <kbd>property</kbd> ATEResult.pos_dev_y

Returns y deviations 

---

### <kbd>property</kbd> ATEResult.pos_dev_z

Returns z deviations 

---

### <kbd>property</kbd> ATEResult.pos_rms_along

Returns RMS of along track deviations 

---

### <kbd>property</kbd> ATEResult.pos_rms_cross_h

Returns RMS of horizontal cross track deviations 

---

### <kbd>property</kbd> ATEResult.pos_rms_cross_v

Returns RMS of vertical cross track deviations 

---

### <kbd>property</kbd> ATEResult.pos_rms_x

Returns RMS of x deviations 

---

### <kbd>property</kbd> ATEResult.pos_rms_y

Returns RMS of y deviations 

---

### <kbd>property</kbd> ATEResult.pos_rms_z

Returns RMS of z deviations 

---

### <kbd>property</kbd> ATEResult.property_dict





---

### <kbd>property</kbd> ATEResult.rot_ate

Returns mean of rotation deviations 

---

### <kbd>property</kbd> ATEResult.rot_bias_x

Returns roll bias 

---

### <kbd>property</kbd> ATEResult.rot_bias_y

Returns pitch bias 

---

### <kbd>property</kbd> ATEResult.rot_bias_z

Returns yaw bias 

---

### <kbd>property</kbd> ATEResult.rot_dev_comb

Returns rotation deviations as single rotation angles 

---

### <kbd>property</kbd> ATEResult.rot_dev_max

Returns max of rotations 

---

### <kbd>property</kbd> ATEResult.rot_dev_median

Returns median of rotations 

---

### <kbd>property</kbd> ATEResult.rot_dev_min

Returns min of rotations 

---

### <kbd>property</kbd> ATEResult.rot_dev_rms

Returns RMS of rotations 

---

### <kbd>property</kbd> ATEResult.rot_dev_std

Returns STD of rotations 

---

### <kbd>property</kbd> ATEResult.rot_dev_x

Returns roll deviations 

---

### <kbd>property</kbd> ATEResult.rot_dev_y

Returns pitch deviations 

---

### <kbd>property</kbd> ATEResult.rot_dev_z

Returns yaw deviations 

---

### <kbd>property</kbd> ATEResult.rot_rms_x

Returns RMS of roll deviations 

---

### <kbd>property</kbd> ATEResult.rot_rms_y

Returns RMS of pitch deviations 

---

### <kbd>property</kbd> ATEResult.rot_rms_z

Returns RMS of yaw deviations 



---

## <kbd>classmethod</kbd> `ATEResult.from_file`

```python
from_file(filename: str)
```





---

## <kbd>method</kbd> `ATEResult.to_dataframe`

```python
to_dataframe() → DataFrame
```

Exports results as pandas dataframe 

---

## <kbd>method</kbd> `ATEResult.to_file`

```python
to_file(filename: str, mode: str = 'a') → None
```

Exports results as csv 


# <kbd>class</kbd> `AlignmentData`
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

### <kbd>property</kbd> AlignmentData.est_euler_x





---

### <kbd>property</kbd> AlignmentData.est_euler_y





---

### <kbd>property</kbd> AlignmentData.est_euler_z





---

### <kbd>property</kbd> AlignmentData.est_obs_vector





---

### <kbd>property</kbd> AlignmentData.est_rpy_from





---

### <kbd>property</kbd> AlignmentData.est_speed





---

### <kbd>property</kbd> AlignmentData.est_speed_x





---

### <kbd>property</kbd> AlignmentData.est_speed_y





---

### <kbd>property</kbd> AlignmentData.est_speed_z





---

### <kbd>property</kbd> AlignmentData.est_x_from





---

### <kbd>property</kbd> AlignmentData.est_x_to





---

### <kbd>property</kbd> AlignmentData.est_xyz_from





---

### <kbd>property</kbd> AlignmentData.est_xyz_to





---

### <kbd>property</kbd> AlignmentData.est_y_from





---

### <kbd>property</kbd> AlignmentData.est_y_to





---

### <kbd>property</kbd> AlignmentData.est_z_from





---

### <kbd>property</kbd> AlignmentData.est_z_to





---

### <kbd>property</kbd> AlignmentData.euler_x





---

### <kbd>property</kbd> AlignmentData.euler_y





---

### <kbd>property</kbd> AlignmentData.euler_z





---

### <kbd>property</kbd> AlignmentData.group_stds

Returns the mean standard deviation for each group 

---

### <kbd>property</kbd> AlignmentData.num_obs_per_epoch

Returns the number of observations per epoch depending on the enabled estimation modes. 

---

### <kbd>property</kbd> AlignmentData.number_of_epochs





---

### <kbd>property</kbd> AlignmentData.obs_vector





---

### <kbd>property</kbd> AlignmentData.res_vector





---

### <kbd>property</kbd> AlignmentData.rpy_from





---

### <kbd>property</kbd> AlignmentData.sigma_ll





---

### <kbd>property</kbd> AlignmentData.speed





---

### <kbd>property</kbd> AlignmentData.speed_x





---

### <kbd>property</kbd> AlignmentData.speed_y





---

### <kbd>property</kbd> AlignmentData.speed_z





---

### <kbd>property</kbd> AlignmentData.tstamps





---

### <kbd>property</kbd> AlignmentData.var_vector





---

### <kbd>property</kbd> AlignmentData.x_from





---

### <kbd>property</kbd> AlignmentData.x_to





---

### <kbd>property</kbd> AlignmentData.xyz_from





---

### <kbd>property</kbd> AlignmentData.xyz_to





---

### <kbd>property</kbd> AlignmentData.y_from





---

### <kbd>property</kbd> AlignmentData.y_to





---

### <kbd>property</kbd> AlignmentData.z_from





---

### <kbd>property</kbd> AlignmentData.z_to







---

## <kbd>method</kbd> `AlignmentData.build_obs_vector`

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

## <kbd>method</kbd> `AlignmentData.build_res_vector`

```python
build_res_vector() → ndarray
```





---

## <kbd>method</kbd> `AlignmentData.build_var_vector`

```python
build_var_vector() → ndarray
```

Sets up the variance vector 

Its size depends on whether the leverarm should be estimated or not. In this case, not only the source and the target positions are relevant but also the platform orientations. Also, when estimating the time shift, the platform speed is also considered. 



**Returns:**
 
 - <b>`np.ndarray`</b>:  variance vector 

---

## <kbd>method</kbd> `AlignmentData.get_est_obs_group`

```python
get_est_obs_group(key: str) → Tuple[numpy.ndarray, ...]
```





---

## <kbd>method</kbd> `AlignmentData.get_obs_group`

```python
get_obs_group(key: str) → Tuple[numpy.ndarray, ...]
```





---

## <kbd>method</kbd> `AlignmentData.get_res_group`

```python
get_res_group(key: str) → Tuple[numpy.ndarray, ...]
```





---

## <kbd>method</kbd> `AlignmentData.get_var_group`

```python
get_var_group(key: str) → Tuple[numpy.ndarray, ...]
```





---

## <kbd>method</kbd> `AlignmentData.get_variance_estimation_subset`

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

## <kbd>method</kbd> `AlignmentData.set_obs_group`

```python
set_obs_group(key: str, values: numpy.ndarray) → None
```





---

## <kbd>method</kbd> `AlignmentData.set_res_group`

```python
set_res_group(key: str, values: numpy.ndarray) → None
```





---

## <kbd>method</kbd> `AlignmentData.set_var_group`

```python
set_var_group(key: str, values: numpy.ndarray) → None
```





---

## <kbd>method</kbd> `AlignmentData.setup`

```python
setup() → None
```

Prepare two trajectories for alignment. 

This method will filter the trajectories by speed and resample both trajectories to the same sampling. 


# <kbd>class</kbd> `AlignmentEstimationSettings`
Dataclass defining which parameters to estimate during the alignment process 

### <kbd>method</kbd> `AlignmentEstimationSettings.__init__`

```python
__init__(
    trans_x: bool = True,
    trans_y: bool = True,
    trans_z: bool = True,
    rot_x: bool = True,
    rot_y: bool = True,
    rot_z: bool = True,
    scale: bool = False,
    time_shift: bool = False,
    use_x_speed: bool = True,
    use_y_speed: bool = True,
    use_z_speed: bool = True,
    lever_x: bool = False,
    lever_y: bool = False,
    lever_z: bool = False,
    sensor_rotation: bool = False,
    auto_update: bool = False
) → None
```






---

### <kbd>property</kbd> AlignmentEstimationSettings.all_disabled





---

### <kbd>property</kbd> AlignmentEstimationSettings.all_lq_disabled

Returns True if all parameters estimated with Least-Squares are disabled 

---

### <kbd>property</kbd> AlignmentEstimationSettings.enabled_lq_parameter_filter

Returns a list of bools indicating which parameters estimated within LQ are enabled 

---

### <kbd>property</kbd> AlignmentEstimationSettings.helmert_enabled





---

### <kbd>property</kbd> AlignmentEstimationSettings.helmert_filter





---

### <kbd>property</kbd> AlignmentEstimationSettings.leverarm_enabled





---

### <kbd>property</kbd> AlignmentEstimationSettings.leverarm_filter





---

### <kbd>property</kbd> AlignmentEstimationSettings.lq_parameter_filter





---

### <kbd>property</kbd> AlignmentEstimationSettings.short_mode_str

Returns a short string describing the enabled parameters 

---

### <kbd>property</kbd> AlignmentEstimationSettings.time_shift_enabled





---

### <kbd>property</kbd> AlignmentEstimationSettings.time_shift_filter







---

## <kbd>classmethod</kbd> `AlignmentEstimationSettings.all`

```python
all(
    sensor_rotation: bool = True,
    auto_update: bool = False
) → AlignmentEstimationSettings
```





---

## <kbd>classmethod</kbd> `AlignmentEstimationSettings.from_bool_list`

```python
from_bool_list(bool_list: List[bool]) → AlignmentEstimationSettings
```





---

## <kbd>classmethod</kbd> `AlignmentEstimationSettings.from_components`

```python
from_components(
    similarity: bool = False,
    time_shift: bool = False,
    leverarm: bool = False,
    sensor_rotation: bool = False,
    auto_update: bool = False
) → AlignmentEstimationSettings
```






# <kbd>class</kbd> `AlignmentParameters`
Parameter set for spatio-temporal alignment 

### <kbd>method</kbd> `AlignmentParameters.__init__`

```python
__init__(
    enabled: bool = True,
    covariance_matrix: numpy.ndarray = <factory>,
    sim_trans_x: trajectopy.core.alignment.parameters.Parameter = <factory>,
    sim_trans_y: trajectopy.core.alignment.parameters.Parameter = <factory>,
    sim_trans_z: trajectopy.core.alignment.parameters.Parameter = <factory>,
    sim_rot_x: trajectopy.core.alignment.parameters.Parameter = <factory>,
    sim_rot_y: trajectopy.core.alignment.parameters.Parameter = <factory>,
    sim_rot_z: trajectopy.core.alignment.parameters.Parameter = <factory>,
    sim_scale: trajectopy.core.alignment.parameters.Parameter = <factory>,
    time_shift: trajectopy.core.alignment.parameters.Parameter = <factory>,
    lever_x: trajectopy.core.alignment.parameters.Parameter = <factory>,
    lever_y: trajectopy.core.alignment.parameters.Parameter = <factory>,
    lever_z: trajectopy.core.alignment.parameters.Parameter = <factory>
) → None
```






---

### <kbd>property</kbd> AlignmentParameters.any_enabled





---

### <kbd>property</kbd> AlignmentParameters.enabled_bool_list





---

### <kbd>property</kbd> AlignmentParameters.enabled_indices





---

### <kbd>property</kbd> AlignmentParameters.helmert





---

### <kbd>property</kbd> AlignmentParameters.leverarm





---

### <kbd>property</kbd> AlignmentParameters.num_enabled





---

### <kbd>property</kbd> AlignmentParameters.sim3_matrix





---

### <kbd>property</kbd> AlignmentParameters.values





---

### <kbd>property</kbd> AlignmentParameters.values_enabled





---

### <kbd>property</kbd> AlignmentParameters.variances





---

### <kbd>property</kbd> AlignmentParameters.variances_enabled







---

## <kbd>method</kbd> `AlignmentParameters.apply_settings`

```python
apply_settings(
    settings: trajectopy.core.settings.alignment.AlignmentEstimationSettings
) → None
```

Applies the estimation settings to the parameters by enabling or disabling them 

---

## <kbd>method</kbd> `AlignmentParameters.disable`

```python
disable() → None
```





---

## <kbd>method</kbd> `AlignmentParameters.enable`

```python
enable() → None
```





---

## <kbd>classmethod</kbd> `AlignmentParameters.from_file`

```python
from_file(filename: str) → AlignmentParameters
```

Reads the alignment parameters from a file 



**Args:**
 
 - <b>`filename`</b> (str):  Path to the file 



**Returns:**
 
 - <b>`AlignmentParameters`</b>:  AlignmentParameters instance 

---

## <kbd>classmethod</kbd> `AlignmentParameters.from_settings`

```python
from_settings(
    settings: trajectopy.core.settings.alignment.AlignmentEstimationSettings
) → AlignmentParameters
```





---

## <kbd>method</kbd> `AlignmentParameters.get_covariance_matrix`

```python
get_covariance_matrix(enabled_only: bool = True) → ndarray
```





---

## <kbd>method</kbd> `AlignmentParameters.params_labels`

```python
params_labels(enabled_only: bool = True, lower_case: bool = True) → List[str]
```





---

## <kbd>method</kbd> `AlignmentParameters.set_covariance_matrix`

```python
set_covariance_matrix(cov_matrix: numpy.ndarray) → None
```





---

## <kbd>method</kbd> `AlignmentParameters.to_dataframe`

```python
to_dataframe() → DataFrame
```





---

## <kbd>method</kbd> `AlignmentParameters.to_dict`

```python
to_dict(enabled_only: bool = True) → Dict[str, List[float]]
```





---

## <kbd>method</kbd> `AlignmentParameters.to_estimation_settings`

```python
to_estimation_settings() → AlignmentEstimationSettings
```





---

## <kbd>method</kbd> `AlignmentParameters.to_file`

```python
to_file(filename: str) → None
```

Writes the alignment parameters to a file 

All parameters are written to the file no matter if they are enabled or not. If they are not enabled, the default value is written. Enabled parameters are marked with a 1, disabled with a 0. The order of the parameters is: 


- Translation x 
- Translation y 
- Translation z 
- Rotation x 
- Rotation y 
- Rotation z 
- Scale 
- Time Shift 
- Leverarm x 
- Leverarm y 
- Leverarm z 

Besides the parameters, the covariance matrix is written to the file. The covariance matrix is written row by row next to the parameters. 



**Args:**
 
 - <b>`filename`</b> (str):  Path to the file 

---

## <kbd>method</kbd> `AlignmentParameters.to_name_list`

```python
to_name_list(enabled_only: bool = True, lower_case: bool = True) → List[str]
```





---

## <kbd>method</kbd> `AlignmentParameters.to_string_list`

```python
to_string_list(enabled_only: bool = True) → List[str]
```





---

## <kbd>method</kbd> `AlignmentParameters.validate_covariance`

```python
validate_covariance()
```

Checks dimensions of covariance matrix and sets variances accordingly 


# <kbd>class</kbd> `AlignmentPreprocessing`
Dataclass defining alignment preprocessing configuration 

### <kbd>method</kbd> `AlignmentPreprocessing.__init__`

```python
__init__(
    min_speed: float = 0.0,
    time_start: float = 0.0,
    time_end: float = 0.0
) → None
```









# <kbd>class</kbd> `AlignmentResult`
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

## <kbd>classmethod</kbd> `AlignmentResult.from_file`

```python
from_file(filename: str) → AlignmentResult
```

Load the result from a file. 



**Args:**
 
 - <b>`filename`</b> (str):  Path to the file. 



**Returns:**
 
 - <b>`AlignmentResult`</b>:  The loaded result. 

---

## <kbd>method</kbd> `AlignmentResult.to_file`

```python
to_file(filename: str) → None
```

Save the result to a file. 



**Args:**
 
 - <b>`filename`</b> (str):  Path to the file. 


# <kbd>class</kbd> `AlignmentSettings`
Dataclass defining alignment configuration 



**Args:**
 


    - mode (str): Mode of the H(elmert)-L(everarm)-T(ime) transformation  Depending on the presence of the letters "h", "l", "t"  inside this string, the alignment will estimate the  corresponding parameters 
    - std_xx (float): Standard deviations in their corresponding unit  of the supported observation_groups: 
        - xy_from (source positions) 
        - z_from 
        - xy_to (target positions) 
        - z_to 
        - roll_pitch (platform orientations) 
        - yaw 

### <kbd>method</kbd> `AlignmentSettings.__init__`

```python
__init__(
    preprocessing: trajectopy.core.settings.alignment.AlignmentPreprocessing = <factory>,
    estimation_settings: trajectopy.core.settings.alignment.AlignmentEstimationSettings = <factory>,
    stochastics: trajectopy.core.settings.alignment.AlignmentStochastics = <factory>,
    metric_threshold: float = 0.0001,
    time_threshold: float = 0.0001
) → None
```









# <kbd>class</kbd> `AlignmentStochastics`
Dataclass defining alignment stochastics configuration 

### <kbd>method</kbd> `AlignmentStochastics.__init__`

```python
__init__(
    std_xy_from: float = 1.0,
    std_z_from: float = 1.0,
    std_xy_to: float = 1.0,
    std_z_to: float = 1.0,
    std_roll_pitch: float = 0.017453292519943295,
    std_yaw: float = 0.017453292519943295,
    std_speed: float = 1.0,
    error_probability: float = 0.05,
    variance_estimation: bool = False
) → None
```






---

### <kbd>property</kbd> AlignmentStochastics.var_roll_pitch





---

### <kbd>property</kbd> AlignmentStochastics.var_speed_to





---

### <kbd>property</kbd> AlignmentStochastics.var_xy_from





---

### <kbd>property</kbd> AlignmentStochastics.var_xy_to





---

### <kbd>property</kbd> AlignmentStochastics.var_yaw





---

### <kbd>property</kbd> AlignmentStochastics.var_z_from





---

### <kbd>property</kbd> AlignmentStochastics.var_z_to








# <kbd>class</kbd> `Any`
Special type indicating an unconstrained type. 


- Any is compatible with every type. 
- Any assumed to have all methods. 
- All values assumed to be instances of Any. 

Note that all the above statements are true from the point of view of static type checkers. At runtime, Any should not be used with instance checks. 





# <kbd>class</kbd> `ApproximationSettings`
Dataclass defining approximation configuration 

### <kbd>method</kbd> `ApproximationSettings.__init__`

```python
__init__(
    fe_int_size: float = 0.15,
    fe_min_obs: int = 25,
    rot_approx_win_size: float = 0.15
) → None
```









# <kbd>class</kbd> `ComparisonMethod`








# <kbd>class</kbd> `Enum`
Create a collection of name/value pairs. 

Example enumeration: 

``` class Color(Enum):```
...     RED = 1
...     BLUE = 2
...     GREEN = 3

Access them by:

- attribute access:
```

``` Color.RED``` ```
<Color.RED: 1> 


- value lookup: 

``` Color(1)```
<Color.RED: 1>


- name lookup:

``` Color['RED']``` <Color.RED: 1> 

Enumerations can be iterated over, and know how many members they have: 

``` len(Color)```
3

``` list(Color)``` [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] 

Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details. 

### <kbd>method</kbd> `Enum.__init__`

```python
__init__(*args, **kwds)
```









# <kbd>class</kbd> `ExportSettings`
ExportSettings(format: str = 'png', height: int = 500, width: int = 800, scale: int = 1) 

### <kbd>method</kbd> `ExportSettings.__init__`

```python
__init__(
    format: str = 'png',
    height: int = 500,
    width: int = 800,
    scale: int = 1
) → None
```








---

## <kbd>method</kbd> `ExportSettings.to_config`

```python
to_config() → dict
```






# <kbd>class</kbd> `HelmertTransformation`
Parameter set for a similarity transformation 

### <kbd>method</kbd> `HelmertTransformation.__init__`

```python
__init__(
    enabled: bool = True,
    covariance_matrix: numpy.ndarray = <factory>,
    trans_x: trajectopy.core.alignment.parameters.Parameter = <factory>,
    trans_y: trajectopy.core.alignment.parameters.Parameter = <factory>,
    trans_z: trajectopy.core.alignment.parameters.Parameter = <factory>,
    rot_x: trajectopy.core.alignment.parameters.Parameter = <factory>,
    rot_y: trajectopy.core.alignment.parameters.Parameter = <factory>,
    rot_z: trajectopy.core.alignment.parameters.Parameter = <factory>,
    scale: trajectopy.core.alignment.parameters.Parameter = <factory>
) → None
```






---

### <kbd>property</kbd> HelmertTransformation.any_enabled





---

### <kbd>property</kbd> HelmertTransformation.enabled_bool_list





---

### <kbd>property</kbd> HelmertTransformation.enabled_indices





---

### <kbd>property</kbd> HelmertTransformation.num_enabled





---

### <kbd>property</kbd> HelmertTransformation.rotation





---

### <kbd>property</kbd> HelmertTransformation.rotation_matrix





---

### <kbd>property</kbd> HelmertTransformation.rotation_set





---

### <kbd>property</kbd> HelmertTransformation.translation





---

### <kbd>property</kbd> HelmertTransformation.values





---

### <kbd>property</kbd> HelmertTransformation.values_enabled





---

### <kbd>property</kbd> HelmertTransformation.variances





---

### <kbd>property</kbd> HelmertTransformation.variances_enabled







---

## <kbd>method</kbd> `HelmertTransformation.apply_to`

```python
apply_to(xyz: numpy.ndarray) → ndarray
```





---

## <kbd>method</kbd> `HelmertTransformation.disable`

```python
disable() → None
```





---

## <kbd>method</kbd> `HelmertTransformation.enable`

```python
enable() → None
```





---

## <kbd>method</kbd> `HelmertTransformation.get_covariance_matrix`

```python
get_covariance_matrix(enabled_only: bool = True) → ndarray
```





---

## <kbd>method</kbd> `HelmertTransformation.set_covariance_matrix`

```python
set_covariance_matrix(cov_matrix: numpy.ndarray) → None
```





---

## <kbd>method</kbd> `HelmertTransformation.to_dict`

```python
to_dict(enabled_only: bool = True) → Dict[str, List[float]]
```





---

## <kbd>method</kbd> `HelmertTransformation.to_name_list`

```python
to_name_list(enabled_only: bool = True, lower_case: bool = True) → List[str]
```





---

## <kbd>method</kbd> `HelmertTransformation.to_string_list`

```python
to_string_list(enabled_only: bool = True) → List[str]
```





---

## <kbd>method</kbd> `HelmertTransformation.validate_covariance`

```python
validate_covariance()
```

Checks dimensions of covariance matrix and sets variances accordingly 


# <kbd>class</kbd> `Leverarm`
Parameter set for a leverarm 

### <kbd>method</kbd> `Leverarm.__init__`

```python
__init__(
    enabled: bool = True,
    covariance_matrix: numpy.ndarray = <factory>,
    x: trajectopy.core.alignment.parameters.Parameter = <factory>,
    y: trajectopy.core.alignment.parameters.Parameter = <factory>,
    z: trajectopy.core.alignment.parameters.Parameter = <factory>
) → None
```






---

### <kbd>property</kbd> Leverarm.any_enabled





---

### <kbd>property</kbd> Leverarm.enabled_bool_list





---

### <kbd>property</kbd> Leverarm.enabled_indices





---

### <kbd>property</kbd> Leverarm.num_enabled





---

### <kbd>property</kbd> Leverarm.values





---

### <kbd>property</kbd> Leverarm.values_enabled





---

### <kbd>property</kbd> Leverarm.variances





---

### <kbd>property</kbd> Leverarm.variances_enabled







---

## <kbd>method</kbd> `Leverarm.apply_to`

```python
apply_to(xyz: numpy.ndarray, quat_body: numpy.ndarray) → ndarray
```

Applies the leverarm to a set of positions using orientations 



**Args:**
 
 - <b>`xyz`</b> (np.ndarray):  Positions 
 - <b>`quat_body`</b> (np.ndarray):  Orientations 



**Returns:**
 
 - <b>`np.ndarray`</b>:  Leverarm applied positions 

---

## <kbd>method</kbd> `Leverarm.disable`

```python
disable() → None
```





---

## <kbd>method</kbd> `Leverarm.enable`

```python
enable() → None
```





---

## <kbd>method</kbd> `Leverarm.get_covariance_matrix`

```python
get_covariance_matrix(enabled_only: bool = True) → ndarray
```





---

## <kbd>method</kbd> `Leverarm.set_covariance_matrix`

```python
set_covariance_matrix(cov_matrix: numpy.ndarray) → None
```





---

## <kbd>method</kbd> `Leverarm.to_dict`

```python
to_dict(enabled_only: bool = True) → Dict[str, List[float]]
```





---

## <kbd>method</kbd> `Leverarm.to_name_list`

```python
to_name_list(enabled_only: bool = True, lower_case: bool = True) → List[str]
```





---

## <kbd>method</kbd> `Leverarm.to_string_list`

```python
to_string_list(enabled_only: bool = True) → List[str]
```





---

## <kbd>method</kbd> `Leverarm.validate_covariance`

```python
validate_covariance()
```

Checks dimensions of covariance matrix and sets variances accordingly 


# <kbd>class</kbd> `MPLPlotSettings`
Dataclass defining plot configuration 

### <kbd>method</kbd> `MPLPlotSettings.__init__`

```python
__init__(
    scatter_cbar_show_zero: bool = True,
    scatter_cbar_steps: int = 4,
    scatter_no_axis: bool = False,
    scatter_sigma_factor: float = 3.0,
    ate_unit_is_mm: bool = False,
    hist_as_stairs: bool = False,
    heatmap_spacing: float = 1.0,
    show_directed_devs: bool = False
) → None
```






---

### <kbd>property</kbd> MPLPlotSettings.unit_multiplier





---

### <kbd>property</kbd> MPLPlotSettings.unit_str








# <kbd>class</kbd> `MatchingMethod`








# <kbd>class</kbd> `MatchingSettings`
MatchingSettings(method: trajectopy.core.settings.matching.MatchingMethod = <MatchingMethod.INTERPOLATION: 'interpolation'>, max_time_diff: float = 0.01, max_distance: float = 0.0, k_nearest: int = 10) 

### <kbd>method</kbd> `MatchingSettings.__init__`

```python
__init__(
    method: trajectopy.core.settings.matching.MatchingMethod = <MatchingMethod.INTERPOLATION: 'interpolation'>,
    max_time_diff: float = 0.01,
    max_distance: float = 0.0,
    k_nearest: int = 10
) → None
```








---

## <kbd>method</kbd> `MatchingSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```





---

## <kbd>method</kbd> `MatchingSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```






# <kbd>class</kbd> `PairDistanceUnit`








# <kbd>class</kbd> `PointSet`
Class representing a point set 



The pointset class can hold one or multiple 3d positions with a corresponding EPSG code for datum information. When creating a pointset instance, a transformation pipeline is created based on the passed points, which can transform the points into a local system tangent to the ellipsoid (grs80). Such a local datum is represented within this class with an EPSG code of 0 and is mainly suitable for local calculations within the pointset. 

All transformations, including the local transformation, are carried out using pyproj, a python toolbox for projections and transformations. 

### <kbd>method</kbd> `PointSet.__init__`

```python
__init__(
    xyz: Union[numpy.ndarray, list],
    epsg: int = 0,
    local_transformer: pyproj.transformer.Transformer = None,
    init_local_transformer: bool = True,
    epsg_local_cart: int = 4936,
    epsg_local_geod: int = 4937
) → None
```

Initialize PointSet and create local transformer 

If a pointset is initialized directly with an EPSG code of 0, such a local transformer cannot be constructed, and a transformation into other EPSG codes is therefore not possible. Use this setting, if you dont have any information about the datum of the passed points. However, if the local transformer is already known, it can be provided during the initialization of the pointset using the local_transformer variable. 



**Args:**
 
 - <b>`xyz`</b> (np.ndarray):  1- / 2- dimensional numpy array  containing the coordinated of the  input positions 
 - <b>`epsg`</b> (int, optional):  EPSG code of the datum of the input  positions. Defaults to 0. 
 - <b>`local_transformer`</b> (Transformer, optional):  pyproj transformer  that describes the  transformation to  a local coordinate  system.  Defaults to None. 
 - <b>`init_local_transformer`</b> (bool, optional):  Specifies if a local  transformer should be  initialized.  Defaults to True. 
 - <b>`epsg_local_cart`</b> (int, optional):  EPSG code of the earth-centered  datum that is used to construct  the local transformation pipeline.  In a first step, the coordinates  are transformed to this coordinate  frame. In this coordinate frame  they are reduced by their mean  position.  Defaults to 4936. 
 - <b>`epsg_local_geod`</b> (int, optional):  In the final step of the local  transformation pipeline, the  positions reduced by their mean  are rotated into a local system  tangent to the ellipsoid.  The ellipsoid is defined by this  parameter using an EPSG code.  Both EPSG codes, epsg_local_cart  and epsg_local_geod should refer  to the same datum (here ETRS89).  Defaults to 4937. 



**Raises:**
 
 - <b>`PointSetError`</b>:  Gets raised, if input xyz is not a numpy array 


---

### <kbd>property</kbd> PointSet.crs

Coordinate Reference System 



**Returns:**
 
 - <b>`CRS`</b>:  pyproj CRS object that represents the current coordinate system 

---

### <kbd>property</kbd> PointSet.x

x property 

The x/y/z properties will either return a one-dimensional numpy array or a single float / int depending on whether there is more than one point in the pointset 



**Returns:**
  Union[int, float, np.ndarray] 

---

### <kbd>property</kbd> PointSet.xyz

xyz property returning the points within the pointset 



**Returns:**
 
 - <b>`np.ndarray`</b>:  2-dimensional numpy array 

---

### <kbd>property</kbd> PointSet.y

y property 

The x/y/z properties will either return a one-dimensional numpy array or a single float / int depending on whether there is more than one point in the pointset 



**Returns:**
  Union[int, float, np.ndarray] 

---

### <kbd>property</kbd> PointSet.z

z property 

The x/y/z properties will either return a one-dimensional numpy array or a single float / int depending on whether there is more than one point in the pointset 



**Returns:**
  Union[int, float, np.ndarray] 



---

## <kbd>method</kbd> `PointSet.build_local_transformer`

```python
build_local_transformer()
```





---

## <kbd>method</kbd> `PointSet.copy`

```python
copy() → PointSet
```

Deep copy 

---

## <kbd>method</kbd> `PointSet.mean`

```python
mean(inplace: bool = False) → PointSet
```

Computes the mean of all points within the pointset 



**Args:**
 
 - <b>`inplace`</b> (bool, optional):  if true, the pointset gets  replaced by a single mean  position. Defaults to False. 



**Returns:**
 
 - <b>`PointSet`</b>:  Contains the mean position 

---

## <kbd>method</kbd> `PointSet.round_to`

```python
round_to(prec: float) → PointSet
```

Rounds all points to a given precision 



**Args:**
 
 - <b>`prec`</b> (float):  desired rounding precision 



**Returns:**
 
 - <b>`PointSet`</b>:  Contains the rounded positions 

---

## <kbd>method</kbd> `PointSet.to_epsg`

```python
to_epsg(target_epsg: int, inplace: bool = True) → PointSet
```

Performs a coordinate transformation using a target crs 

This method will construct the required pyproj transformer and applies it in order to transform the pointset to the target ESPG code. 



**Args:**
 
 - <b>`target_epsg`</b> (int):  EPSG code of target CRS 
 - <b>`inplace`</b> (bool, optional):  perform transformation in place.  Defaults to True. 



**Raises:**
 


 - <b>`PointSetError`</b>:  Gets raised if it is not possible to recover  from a local datum since local transformer  is unknown 



**Returns:**
 
 - <b>`PointSet`</b>:  transformed pointset 

---

## <kbd>method</kbd> `PointSet.to_local`

```python
to_local(inplace: bool = True) → PointSet
```

Transform pointset to a local frame tangential to the  (grs80) ellipsoid 

This is equivalent to an transformation to an EPSG of 0 



**Args:**
 
 - <b>`inplace`</b> (bool, optional):  perform transformation in place.  Defaults to True. 



**Returns:**
 
 - <b>`PointSet`</b>:  2-dimensional PointSet containing xyz of the  transformed points 


# <kbd>class</kbd> `ProcessingSettings`
Settings for processing the trajectory. 

### <kbd>method</kbd> `ProcessingSettings.__init__`

```python
__init__(
    alignment: trajectopy.core.settings.alignment.AlignmentSettings = <factory>,
    matching: trajectopy.core.settings.matching.MatchingSettings = <factory>,
    relative_comparison: trajectopy.core.settings.comparison.RelativeComparisonSettings = <factory>,
    approximation: trajectopy.core.settings.approximation.ApproximationSettings = <factory>,
    sorting: trajectopy.core.settings.sorting.SortingSettings = <factory>
) → None
```









# <kbd>class</kbd> `RPEResult`
This class represents a set of relative trajectory deviations 

Relative trajectory deviations describe relative pose deviations between two trajectories. The deviations are calculated by comparing pairs of positions and orientations in the test and reference trajectory. 



**Attributes:**
 
 - <b>`rpe_dev`</b> (RelativeTrajectoryDeviations):  The relative trajectory deviations. 
 - <b>`name`</b> (str):  Name of the deviation set. 

### <kbd>method</kbd> `RPEResult.__init__`

```python
__init__(
    rpe_dev: trajectopy.core.evaluation.deviations.RelativeTrajectoryDeviations,
    name: str
) → None
```






---

### <kbd>property</kbd> RPEResult.all_pair_distances





---

### <kbd>property</kbd> RPEResult.all_rot_devs





---

### <kbd>property</kbd> RPEResult.columns





---

### <kbd>property</kbd> RPEResult.drift_factor





---

### <kbd>property</kbd> RPEResult.dynamic_pos_dict





---

### <kbd>property</kbd> RPEResult.dynamic_rot_dict





---

### <kbd>property</kbd> RPEResult.has_rot_dev





---

### <kbd>property</kbd> RPEResult.mean_pair_distances





---

### <kbd>property</kbd> RPEResult.num_pairs





---

### <kbd>property</kbd> RPEResult.pair_distance_unit





---

### <kbd>property</kbd> RPEResult.pos_dev_all





---

### <kbd>property</kbd> RPEResult.pos_dev_max





---

### <kbd>property</kbd> RPEResult.pos_dev_mean





---

### <kbd>property</kbd> RPEResult.pos_dev_median





---

### <kbd>property</kbd> RPEResult.pos_dev_min





---

### <kbd>property</kbd> RPEResult.pos_drift_unit





---

### <kbd>property</kbd> RPEResult.pos_rpe





---

### <kbd>property</kbd> RPEResult.pos_std





---

### <kbd>property</kbd> RPEResult.property_dict

Returns a dictionary containing the properties of the deviation set relevant for time based comparisons. This is the case when pose-pairs are defined by a time difference. 

---

### <kbd>property</kbd> RPEResult.rot_dev_max





---

### <kbd>property</kbd> RPEResult.rot_dev_mean





---

### <kbd>property</kbd> RPEResult.rot_dev_median





---

### <kbd>property</kbd> RPEResult.rot_dev_min





---

### <kbd>property</kbd> RPEResult.rot_drift_unit





---

### <kbd>property</kbd> RPEResult.rot_rpe

Returns the average rotation drift in radians per 100 meters. 

---

### <kbd>property</kbd> RPEResult.rot_std





---

### <kbd>property</kbd> RPEResult.step







---

## <kbd>method</kbd> `RPEResult.compute_metric`

```python
compute_metric(
    key: str,
    func: Callable[[Any], float],
    factor: float = 1.0
) → List[float]
```





---

## <kbd>classmethod</kbd> `RPEResult.from_file`

```python
from_file(filename: str)
```

Reads a set of relative trajectory deviations from a file. 

---

## <kbd>method</kbd> `RPEResult.get_all`

```python
get_all(key: str) → List[float]
```





---

## <kbd>method</kbd> `RPEResult.to_dataframe`

```python
to_dataframe() → DataFrame
```





---

## <kbd>method</kbd> `RPEResult.to_file`

```python
to_file(filename: str, mode: str = 'a') → None
```






# <kbd>class</kbd> `RelativeComparisonSettings`
A class representing the settings for relative trajectory comparison. 



**Attributes:**
 
 - <b>`relative_pair_min_distance`</b> (float):  The minimum distance between two poses in a relative pair. 
 - <b>`relative_pair_max_distance`</b> (float):  The maximum distance between two poses in a relative pair. 
 - <b>`relative_pair_distance_step`</b> (float):  The step size for the distance between two poses in a relative pair. 
 - <b>`relative_pair_distance_unit`</b> (PairDistanceUnit):  The unit of measurement for the distance between two poses in a relative pair. 
 - <b>`use_all_pose_pairs`</b> (bool):  Whether to use all possible pose pairs for relative comparison. 

### <kbd>method</kbd> `RelativeComparisonSettings.__init__`

```python
__init__(
    pair_min_distance: float = 100.0,
    pair_max_distance: float = 800.0,
    pair_distance_step: float = 100.0,
    pair_distance_unit: trajectopy.core.settings.comparison.PairDistanceUnit = <PairDistanceUnit.METER: 'meter'>,
    use_all_pose_pairs: bool = True
) → None
```








---

## <kbd>method</kbd> `RelativeComparisonSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```





---

## <kbd>method</kbd> `RelativeComparisonSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```






# <kbd>class</kbd> `ReportSettings`
ReportSettings class represents the settings for generating reports. 


- `single_plot_height` (int): The height of a single plot. Default value is 450. 
- `two_subplots_height` (int): The height of two subplots. Default value is 540. 
- `three_subplots_height` (int): The height of three subplots. Default value is 750. 
- `scatter_max_std` (float): The upper colorbar limit is set to the mean plus this value times the standard deviation of the data. This is useful to prevent outliers from dominating the colorbar. Default value is 4.0. 
- `ate_unit_is_mm` (bool): Indicates whether the unit of Absolute Trajectory Error (ATE) is millimeters. Default value is False. 
- `directed_ate` (bool): Indicates whether the ATE is split into along-, horizontal-cross- and vertical-cross-track direction. Default value is True. 
- `histogram_opacity` (float): The opacity of the histogram bars. Default value is 0.7. 
- `histogram_bargap` (float): The gap between histogram bars. Default value is 0.1. 
- `histogram_barmode` (str): The mode of displaying histogram bars. Default value is "overlay". 
- `histogram_yaxis_title` (str): The title of the y-axis in the histogram. Default value is "Count". 
- `plot_mode` (str): The mode of displaying plots. Default value is "lines+markers". 
- `scatter_mode` (str): The mode of displaying scatter plots. Default value is "markers". 
- `scatter_colorscale` (str): The colorscale for scatter plots. Default value is "RdYlBu_r". 
- `scatter_axis_order` (str): The order of the axes in scatter plots. Default value is "xy". If 3d plotting is desired, also specify "z". 
- `scatter_marker_size` (int): The size of markers in scatter plots. Default value is 5. 
- `scatter_detailed` (bool): Indicates whether to show scatter plots for each degree of freedom. Default value is False. 
- `scatter_mapbox` (bool): Indicates whether to use mapbox for scatter plots. Default value is False. 
- `scatter_mapbox_style` (str): The mapbox style for scatter plots. Default value is "open-street-map". 
- `scatter_mapbox_zoom` (int): The zoom level for scatter plots. Default value is 15. 
- `pos_x_name` (str): The name of the x-coordinate in position data. Default value is "x". 
- `pos_y_name` (str): The name of the y-coordinate in position data. Default value is "y". 
- `pos_z_name` (str): The name of the z-coordinate in position data. Default value is "z". 
- `pos_x_unit` (str): The unit of the x-coordinate in position data. Default value is "m". 
- `pos_y_unit` (str): The unit of the y-coordinate in position data. Default value is "m". 
- `pos_z_unit` (str): The unit of the z-coordinate in position data. Default value is "m". 
- `pos_dir_dev_x_name` (str): The name of the along-track direction deviation in position data. Default value is "along". 
- `pos_dir_dev_y_name` (str): The name of the horizontal-cross-track direction deviation in position data. Default value is "cross-h". 
- `pos_dir_dev_z_name` (str): The name of the vertical-cross-track direction deviation in position data. Default value is "cross-v". 
- `rot_x_name` (str): The name of the roll angle in rotation data. Default value is "roll". 
- `rot_y_name` (str): The name of the pitch angle in rotation data. Default value is "pitch". 
- `rot_z_name` (str): The name of the yaw angle in rotation data. Default value is "yaw". 
- `rot_unit` (str): The unit of rotation angles. Default value is "°". 
- `single_plot_export` (ExportSettings): The export settings for single plots. Default value is an instance of ExportSettings with width=800 and height=450. 
- `two_subplots_export` (ExportSettings): The export settings for two subplots. Default value is an instance of ExportSettings with width=800 and height=540. 
- `three_subplots_export` (ExportSettings): The export settings for three subplots. Default value is an instance of ExportSettings with width=800 and height=750. 

### <kbd>method</kbd> `ReportSettings.__init__`

```python
__init__(
    single_plot_height: int = 640,
    two_subplots_height: int = 750,
    three_subplots_height: int = 860,
    scatter_max_std: float = 4.0,
    ate_unit_is_mm: bool = False,
    directed_ate: bool = True,
    histogram_opacity: float = 0.7,
    histogram_bargap: float = 0.1,
    histogram_barmode: str = 'overlay',
    histogram_yaxis_title: str = 'Count',
    plot_mode: str = 'lines+markers',
    scatter_mode: str = 'markers',
    scatter_colorscale: str = 'RdYlBu_r',
    scatter_axis_order: str = 'xy',
    scatter_marker_size: int = 5,
    scatter_detailed: bool = False,
    scatter_mapbox: bool = False,
    scatter_mapbox_style: str = 'open-street-map',
    scatter_mapbox_zoom: int = 15,
    scatter_mapbox_token: str = '',
    pos_x_name: str = 'x',
    pos_y_name: str = 'y',
    pos_z_name: str = 'z',
    pos_x_unit: str = 'm',
    pos_y_unit: str = 'm',
    pos_z_unit: str = 'm',
    pos_dir_dev_x_name: str = 'along',
    pos_dir_dev_y_name: str = 'cross-h',
    pos_dir_dev_z_name: str = 'cross-v',
    rot_x_name: str = 'roll',
    rot_y_name: str = 'pitch',
    rot_z_name: str = 'yaw',
    rot_unit: str = '°',
    single_plot_export: trajectopy.core.settings.report.ExportSettings = <factory>,
    two_subplots_export: trajectopy.core.settings.report.ExportSettings = <factory>,
    three_subplots_export: trajectopy.core.settings.report.ExportSettings = <factory>
) → None
```









# <kbd>class</kbd> `RotationSet`
Class representing rotations 

This class is essentially just a wrapper around the parent scipy.spatial.transform.Rotation class. It extends its functionality by introducing + / - operators as well as the ability to create a deepcopy and to output rotation angles. 

Furthermore, it ensures that the naming is consistent with the pointset class. 


---

### <kbd>property</kbd> RotationSet.rotangle

Returns minimum rotation angle(s) 



---

## <kbd>method</kbd> `RotationSet.copy`

```python
copy() → RotationSet
```





---

## <kbd>classmethod</kbd> `RotationSet.from_euler`

```python
from_euler(seq: str, angles: numpy.ndarray, degrees: bool = False) → RotationSet
```





---

## <kbd>classmethod</kbd> `RotationSet.from_quat`

```python
from_quat(quat: numpy.ndarray) → RotationSet
```






# <kbd>class</kbd> `SensorRotationParameters`
Parameter set for sensor b-frame rotation 

### <kbd>method</kbd> `SensorRotationParameters.__init__`

```python
__init__(
    enabled: bool = True,
    covariance_matrix: numpy.ndarray = <factory>,
    sensor_rot_x: trajectopy.core.alignment.parameters.Parameter = <factory>,
    sensor_rot_y: trajectopy.core.alignment.parameters.Parameter = <factory>,
    sensor_rot_z: trajectopy.core.alignment.parameters.Parameter = <factory>
) → None
```






---

### <kbd>property</kbd> SensorRotationParameters.any_enabled





---

### <kbd>property</kbd> SensorRotationParameters.enabled_bool_list





---

### <kbd>property</kbd> SensorRotationParameters.enabled_indices





---

### <kbd>property</kbd> SensorRotationParameters.num_enabled





---

### <kbd>property</kbd> SensorRotationParameters.rotation





---

### <kbd>property</kbd> SensorRotationParameters.rotation_matrix





---

### <kbd>property</kbd> SensorRotationParameters.rotation_set





---

### <kbd>property</kbd> SensorRotationParameters.values





---

### <kbd>property</kbd> SensorRotationParameters.values_enabled





---

### <kbd>property</kbd> SensorRotationParameters.variances





---

### <kbd>property</kbd> SensorRotationParameters.variances_enabled







---

## <kbd>method</kbd> `SensorRotationParameters.disable`

```python
disable() → None
```





---

## <kbd>method</kbd> `SensorRotationParameters.enable`

```python
enable() → None
```





---

## <kbd>classmethod</kbd> `SensorRotationParameters.from_file`

```python
from_file(filename: str) → SensorRotationParameters
```





---

## <kbd>method</kbd> `SensorRotationParameters.get_covariance_matrix`

```python
get_covariance_matrix(enabled_only: bool = True) → ndarray
```





---

## <kbd>method</kbd> `SensorRotationParameters.set_covariance_matrix`

```python
set_covariance_matrix(cov_matrix: numpy.ndarray) → None
```





---

## <kbd>method</kbd> `SensorRotationParameters.to_dict`

```python
to_dict(enabled_only: bool = True) → Dict[str, List[float]]
```





---

## <kbd>method</kbd> `SensorRotationParameters.to_file`

```python
to_file(filename: str) → None
```





---

## <kbd>method</kbd> `SensorRotationParameters.to_name_list`

```python
to_name_list(enabled_only: bool = True, lower_case: bool = True) → List[str]
```





---

## <kbd>method</kbd> `SensorRotationParameters.to_string_list`

```python
to_string_list(enabled_only: bool = True) → List[str]
```





---

## <kbd>method</kbd> `SensorRotationParameters.validate_covariance`

```python
validate_covariance()
```

Checks dimensions of covariance matrix and sets variances accordingly 


# <kbd>class</kbd> `Settings`
Base Class for Settings 

### <kbd>method</kbd> `Settings.__init__`

```python
__init__() → None
```








---

## <kbd>method</kbd> `Settings.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

## <kbd>method</kbd> `Settings.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

## <kbd>classmethod</kbd> `Settings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

## <kbd>classmethod</kbd> `Settings.from_file`

```python
from_file(path: str) → Settings
```





---

## <kbd>method</kbd> `Settings.to_dict`

```python
to_dict() → dict
```





---

## <kbd>method</kbd> `Settings.to_file`

```python
to_file(path: str)
```





---

## <kbd>method</kbd> `Settings.to_json`

```python
to_json()
```





---

## <kbd>method</kbd> `Settings.update_from_dict`

```python
update_from_dict(dct: dict)
```






# <kbd>class</kbd> `Sorting`








# <kbd>class</kbd> `SortingSettings`
This class stores all sorting settings 

### <kbd>method</kbd> `SortingSettings.__init__`

```python
__init__(
    discard_missing: bool = True,
    voxel_size: float = 0.05,
    movement_threshold: float = 0.005,
    k_nearest: int = 4
) → None
```









# <kbd>class</kbd> `TimeFormat`








# <kbd>class</kbd> `Trajectory`
Class representing a trajectory, i.e. position and orientation of a plattform over time 

Position-Computations are always done in a local frame Time stamps are always in UTC time Rotations are always defined in a East-North-Up frame 

### <kbd>method</kbd> `Trajectory.__init__`

```python
__init__(
    pos: pointset.pointset.PointSet,
    rot: Optional[trajectopy.core.rotationset.RotationSet] = None,
    tstamps: Optional[numpy.ndarray] = None,
    name: str = '',
    arc_lengths: Optional[numpy.ndarray] = None,
    speed_3d: Optional[numpy.ndarray] = None,
    sorting: trajectopy.core.sorting.Sorting = <Sorting.TIME: 'time'>
) → None
```






---

### <kbd>property</kbd> Trajectory.data_rate

Returns data rate 

---

### <kbd>property</kbd> Trajectory.function_of

Returns the function of the trajectory 

---

### <kbd>property</kbd> Trajectory.function_of_label

Returns the label of the function of the trajectory 

---

### <kbd>property</kbd> Trajectory.function_of_unit

Returns the unit of the function of the trajectory 

---

### <kbd>property</kbd> Trajectory.has_orientation

Returns True if orientation is available 

---

### <kbd>property</kbd> Trajectory.quat

Returns the quaternion of the trajectory 

In contrast to the rot.as_quat() attribute, this method reflects the current sorting of the trajectory. 

---

### <kbd>property</kbd> Trajectory.rpy

Returns the roll, pitch, yaw of the trajectory 

In contrast to the rot.as_euler(seq="xyz") attribute, this method reflects the current sorting of the trajectory. 

---

### <kbd>property</kbd> Trajectory.se3

Returns SE3 pose list 

---

### <kbd>property</kbd> Trajectory.sort_switching_index

Returns the index that switches the sorting of the trajectory 

---

### <kbd>property</kbd> Trajectory.sorting_index

Returns the index that sorts the trajectory 

---

### <kbd>property</kbd> Trajectory.speed

Returns trajectory speeds calculated using consecutive point distances 

---

### <kbd>property</kbd> Trajectory.speed_3d

Returns computed speeds or custom speeds 

---

### <kbd>property</kbd> Trajectory.total_length

Return the total trajectory arc_length. 

---

### <kbd>property</kbd> Trajectory.xyz

Returns the xyz coordinates of the trajectory 

In contrast to the pos.xyz attribute, this method reflects the current sorting of the trajectory. 



---

## <kbd>method</kbd> `Trajectory.adopt_first_orientation`

```python
adopt_first_orientation(
    trajectory: 'Trajectory',
    inplace: bool = True
) → Trajectory
```

Transform trajectory so that the first orientation is identical in both 



**Args:**
 
 - <b>`trajectory`</b> (Trajectory):  Target Trajectory 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Transformed trajectory 

---

## <kbd>method</kbd> `Trajectory.adopt_first_pose`

```python
adopt_first_pose(trajectory: 'Trajectory', inplace: bool = True) → Trajectory
```

Transform trajectory so that the first pose is identical in both 



**Args:**
 
 - <b>`trajectory`</b> (Trajectory):  Target Trajectory 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Transformed trajectory 

---

## <kbd>method</kbd> `Trajectory.adopt_first_position`

```python
adopt_first_position(
    trajectory: 'Trajectory',
    inplace: bool = True
) → Trajectory
```

Transform trajectory so that the first position is identical in both 



**Args:**
 
 - <b>`trajectory`</b> (Trajectory):  Target Trajectory 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Transformed trajectory 

---

## <kbd>method</kbd> `Trajectory.apply_alignment`

```python
apply_alignment(
    alignment_result: trajectopy.core.alignment.result.AlignmentResult,
    inplace: bool = True
) → Trajectory
```

Transforms trajectory using alignment parameters. 

After computing the alignment parameters needed to align two trajectories, they can be applied to arbitrary trajectories. 



**Args:**
  alignment_result (AlignmentResult) 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Aligned trajectory 

---

## <kbd>method</kbd> `Trajectory.apply_index`

```python
apply_index(
    index: Union[list, numpy.ndarray],
    inplace: bool = True
) → Trajectory
```

Applies index to the trajectory 

This will be done either in-place or using a new instance of a trajectory. The index can be used to filter and / or sort the components of the trajectory. 

Those components are: 
- timestamps (tstamps) 
- positions (xyz) 
- rotations (rot) 
- arc lengths (arc_lengths) 
- sorting index (_sort_index) 



**Args:**
 
 - <b>`index`</b> (Union[list, np.ndarray]):  index that should be applied 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Trajectory with index applied. 

---

## <kbd>method</kbd> `Trajectory.apply_transformation`

```python
apply_transformation(
    transformation: numpy.ndarray,
    inplace: bool = True
) → Trajectory
```

Applies transformation to trajectory 



**Args:**
 
 - <b>`transformation`</b> (np.ndarray):  4x4 Transformation matrix 
 - <b>`inplace`</b> (bool, optional):  Perform in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Transformed trajectory 

---

## <kbd>method</kbd> `Trajectory.approximate`

```python
approximate(
    approximation_settings: trajectopy.core.settings.approximation.ApproximationSettings = ApproximationSettings(fe_int_size=0.15, fe_min_obs=25, rot_approx_win_size=0.15),
    inplace: bool = True
) → Trajectory
```

Approximates the trajectory using piecewise cubic polynomial. 



**Args:**
 
 - <b>`approximation_settings`</b> (ApproximationSettings):  Approximation settings. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Approximated trajectory. 

---

## <kbd>method</kbd> `Trajectory.copy`

```python
copy() → Trajectory
```

Deep copy of itself 

---

## <kbd>method</kbd> `Trajectory.crop`

```python
crop(
    t_start: float,
    t_end: float,
    inverse: bool = False,
    inplace: bool = True
) → Trajectory
```

Crops trajectory to timespan defined by t_start and t_end 



**Args:**
 
 - <b>`t_start`</b> (float):  Start timestamp of desired time span 
 - <b>`t_end`</b> (float):  End timestamp of desired time span 
 - <b>`inverse`</b> (bool, optional):  If true, 'crop' turns  into 'cut', i.e. everthing  outside of t_start and t_end  will be removed.  Defaults to False. 
 - <b>`inplace`</b> (bool, optional):  Perform crop in-place.  Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Cropped trajectory 

---

## <kbd>classmethod</kbd> `Trajectory.from_file`

```python
from_file(filename: str, io_stream: bool = False) → Trajectory
```

Create trajectory from file 

The file must be a csv file containing columns for at least the timestamp, x, y and z coordinates of the trajectory. Those fields must be named "t", "px", "py" and "pz" in the header using the #fields tag. However, by default a trajectory with "t,px,py,pz,qx,qy,qz,qw" fields is assumed. Additional fields include the arc length, specified by "l", and the speed, specified by "vx", "vy" and "vz". The delimiter can be specified using the #delimiter tag. The default delimiter is a comma. 



**Args:**
 
 - <b>`filename`</b> (str):  path to file 
 - <b>`io_stream`</b> (bool, optional):  If true, the file is read from a stream. 



**Returns:**
 
 - <b>`Trajectory`</b>:  trajectory object 

---

## <kbd>classmethod</kbd> `Trajectory.from_numpy`

```python
from_numpy(
    xyz: numpy.ndarray,
    quat: numpy.ndarray,
    tstamps: numpy.ndarray,
    epsg: int = 0
) → Trajectory
```

Initialize trajectory using numpy arrays 

---

## <kbd>method</kbd> `Trajectory.init_arc_lengths`

```python
init_arc_lengths()
```





---

## <kbd>method</kbd> `Trajectory.interpolate`

```python
interpolate(
    tstamps: Union[list, numpy.ndarray],
    inplace: bool = True
) → Trajectory
```

Interpolates a trajectory to specified timestamps 

This method removes timestamps from tstamps if they lie outside of the timestamp range of the trajectory (self). Since providing values for those timestamps would require an extrapolation and not an interpolation, this behaviour is consistent with the definition of this method. 



**Args:**
 
 - <b>`tstamps`</b> (list):  Interpolation timestamps 
 - <b>`inplace`</b> (bool, optional):  Perform in-place interpolation.  Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Interpolated trajectory 

---

## <kbd>method</kbd> `Trajectory.intersect`

```python
intersect(
    tstamps: numpy.ndarray,
    max_gap_size: float = 2.0,
    inplace: bool = True
) → Trajectory
```

Intersects trajectory with a given timestamp vector 

After intersection, the trajectory covers the same timespan as 'tstamps'. Further, gaps larger than 'max_gap_size' are removed. If two consecutive timespans in tstamps have a difference of more than 'max_gap_size' seconds, they are considered as the limits of a gap. All timestamps of the trajectory that lie within this gap will be removed. 



**Args:**
 
 - <b>`tstamps`</b> (np.ndarray):  Intersection timespans 
 - <b>`max_gap_size`</b> (float, optional):  Maximum allowed gap between timespans.  If Defaults to 0.5. 
 - <b>`inplace`</b> (bool, optional):  Perform intersection in-place.  Defaults to True. 



**Raises:**
 
 - <b>`ValueError`</b>:  If timespans do not overlap. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Intersected trajectory 

---

## <kbd>method</kbd> `Trajectory.match_timestamps`

```python
match_timestamps(tstamps: numpy.ndarray, inplace: bool = True) → Trajectory
```

Truncates trajectory to only those poses where the timestamps exactly match "tstamps" 



**Args:**
 
 - <b>`tstamps`</b> (np.ndarray):  Input timestamps 
 - <b>`inplace`</b> (bool, optional):  Perform matching in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Trajectory with matched timestamps 

---

## <kbd>method</kbd> `Trajectory.sort_spatially`

```python
sort_spatially(
    sorting_settings: trajectopy.core.settings.sorting.SortingSettings = SortingSettings(discard_missing=True, voxel_size=0.05, movement_threshold=0.005, k_nearest=4),
    inplace: bool = True
) → Trajectory
```

Sorts the trajectory spatially. 



**Args:**
 
 - <b>`sorting_settings`</b> (SortingSettings):  Sorting settings. 
 - <b>`inplace`</b> (bool, optional):  Whether to sort the trajectory in-place. Defaults to True. 



**Returns:**
 
 - <b>`Trajectory`</b>:  Sorted trajectory. 

---

## <kbd>method</kbd> `Trajectory.to_dataframe`

```python
to_dataframe(sort_by: str = '') → DataFrame
```

Returns a pandas dataframe containing tstamps, xyz, quat and speed_3d of the trajectory. 

The dataframe is sorted by the current sorting attribute (time or arc_length). 



**Args:**
 
 - <b>`sort_by`</b> (str, optional):  Column to sort by. This  overrides the current sort_by  attribute. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  Trajectory as dataframe 

---

## <kbd>method</kbd> `Trajectory.to_file`

```python
to_file(filename: str, mode: str = 'w') → None
```

Writes trajectory to ascii file 

The first line will always be the epsg information. After that, the trajectory data is written. 



**Args:**
 
 - <b>`filename`</b> (str):  Output filename 


# <kbd>class</kbd> `Unit`
Enumeration class handling different units 





# <kbd>function</kbd> `ate`

```python
ate(
    trajectory_gt: trajectopy.core.trajectory.Trajectory,
    trajectory_est: trajectopy.core.trajectory.Trajectory,
    settings: trajectopy.core.settings.processing.ProcessingSettings = ProcessingSettings(alignment=AlignmentSettings(preprocessing=AlignmentPreprocessing(min_speed=0.0, time_start=0.0, time_end=0.0), estimation_settings=AlignmentEstimationSettings(trans_x=True, trans_y=True, trans_z=True, rot_x=True, rot_y=True, rot_z=True, scale=False, time_shift=False, use_x_speed=True, use_y_speed=True, use_z_speed=True, lever_x=False, lever_y=False, lever_z=False, sensor_rotation=False, auto_update=False), stochastics=AlignmentStochastics(std_xy_from=1.0, std_z_from=1.0, std_xy_to=1.0, std_z_to=1.0, std_roll_pitch=0.017453292519943295, std_yaw=0.017453292519943295, std_speed=1.0, error_probability=0.05, variance_estimation=False), metric_threshold=0.0001, time_threshold=0.0001), matching=MatchingSettings(method=<MatchingMethod.INTERPOLATION: 'interpolation'>, max_time_diff=0.01, max_distance=0.0, k_nearest=10), relative_comparison=RelativeComparisonSettings(pair_min_distance=100.0, pair_max_distance=800.0, pair_distance_step=100.0, pair_distance_unit=<PairDistanceUnit.METER: 'meter'>, use_all_pose_pairs=True), approximation=ApproximationSettings(fe_int_size=0.15, fe_min_obs=25, rot_approx_win_size=0.15), sorting=SortingSettings(discard_missing=True, voxel_size=0.05, movement_threshold=0.005, k_nearest=4)),
    return_alignment: bool = False
) → Union[trajectopy.core.evaluation.ate_result.ATEResult, Tuple[trajectopy.core.evaluation.ate_result.ATEResult, trajectopy.core.alignment.result.AlignmentResult]]
```

Computes the absolute trajectory error (ATE) between two trajectories. 



**Args:**
 
 - <b>`trajectory_gt`</b> (Trajectory):  Ground truth trajectory. 
 - <b>`trajectory_est`</b> (Trajectory):  Estimated trajectory. 
 - <b>`settings`</b> (ProcessingSettings, optional):  Processing settings. 
 - <b>`return_alignment`</b> (bool, optional):  Whether to return the alignment result. 

Description: The ATE is computed by first matching the estimated trajectory to the ground truth trajectory. Then, the alignment between the two trajectories is estimated. The estimated trajectory is aligned to the ground truth trajectory using the estimated alignment. Finally, the ATE is computed by comparing the aligned estimated trajectory to the ground truth trajectory. 



**Returns:**
 
 - <b>`ATEResult`</b>:  Result of the ATE computation. 

# <kbd>class</kbd> `auto`
Instances are replaced with an appropriate value in Enum class suites. 

### <kbd>method</kbd> `auto.__init__`

```python
__init__(value=_auto_null)
```









# <kbd>function</kbd> `comparison_method_from_string`

```python
comparison_method_from_string(string: str) → <enum 'ComparisonMethod'>
```





# <kbd>function</kbd> `create_alignment_report`

```python
create_alignment_report(
    alignment_parameters: trajectopy.core.alignment.parameters.AlignmentParameters,
    name: str = 'Alignment',
    report_settings: trajectopy.core.settings.report.ReportSettings = ReportSettings(single_plot_height=640, two_subplots_height=750, three_subplots_height=860, scatter_max_std=4.0, ate_unit_is_mm=False, directed_ate=True, histogram_opacity=0.7, histogram_bargap=0.1, histogram_barmode='overlay', histogram_yaxis_title='Count', plot_mode='lines+markers', scatter_mode='markers', scatter_colorscale='RdYlBu_r', scatter_axis_order='xy', scatter_marker_size=5, scatter_detailed=False, scatter_mapbox=False, scatter_mapbox_style='open-street-map', scatter_mapbox_zoom=15, scatter_mapbox_token='', pos_x_name='x', pos_y_name='y', pos_z_name='z', pos_x_unit='m', pos_y_unit='m', pos_z_unit='m', pos_dir_dev_x_name='along', pos_dir_dev_y_name='cross-h', pos_dir_dev_z_name='cross-v', rot_x_name='roll', rot_y_name='pitch', rot_z_name='yaw', rot_unit='°', single_plot_export=ExportSettings(format='png', height=540, width=800, scale=1), two_subplots_export=ExportSettings(format='png', height=540, width=800, scale=1), three_subplots_export=ExportSettings(format='png', height=750, width=800, scale=1))
) → str
```

Render trajectory alignment heatmaps. 

Heatmaps are covariance matrix and correlation matrix. 



**Args:**
 
 - <b>`alignment_parameters`</b> (AlignmentParameters):  Estimated alignment parameters. 
 - <b>`name`</b> (str, optional):  Name of the report. Defaults to "Alignment". 
 - <b>`report_settings`</b> (ReportSettings, optional):  Report settings. Defaults to ReportSettings() 



**Returns:**
 HTML string of the rendered report including the heatmap plots. 

# <kbd>function</kbd> `create_deviation_report`

```python
create_deviation_report(
    ate_result: Union[trajectopy.core.evaluation.ate_result.ATEResult, list[trajectopy.core.evaluation.ate_result.ATEResult]],
    rpe_result: Optional[trajectopy.core.evaluation.rpe_result.RPEResult, list[trajectopy.core.evaluation.rpe_result.RPEResult]],
    report_settings: trajectopy.core.settings.report.ReportSettings = ReportSettings(single_plot_height=640, two_subplots_height=750, three_subplots_height=860, scatter_max_std=4.0, ate_unit_is_mm=False, directed_ate=True, histogram_opacity=0.7, histogram_bargap=0.1, histogram_barmode='overlay', histogram_yaxis_title='Count', plot_mode='lines+markers', scatter_mode='markers', scatter_colorscale='RdYlBu_r', scatter_axis_order='xy', scatter_marker_size=5, scatter_detailed=False, scatter_mapbox=False, scatter_mapbox_style='open-street-map', scatter_mapbox_zoom=15, scatter_mapbox_token='', pos_x_name='x', pos_y_name='y', pos_z_name='z', pos_x_unit='m', pos_y_unit='m', pos_z_unit='m', pos_dir_dev_x_name='along', pos_dir_dev_y_name='cross-h', pos_dir_dev_z_name='cross-v', rot_x_name='roll', rot_y_name='pitch', rot_z_name='yaw', rot_unit='°', single_plot_export=ExportSettings(format='png', height=540, width=800, scale=1), two_subplots_export=ExportSettings(format='png', height=540, width=800, scale=1), three_subplots_export=ExportSettings(format='png', height=750, width=800, scale=1))
)
```

Create a HTML deviation report containing the absolute trajectory error and relative pose error results. 



**Args:**
 
 - <b>`ate_result`</b> (Union[ATEResult, list[ATEResult]]):  The absolute trajectory error results (one or multiple) 
 - <b>`rpe_result`</b> (Union[RPEResult, list[RPEResult]]):  The relative pose error results (one or multiple) 
 - <b>`report_settings`</b> (ReportSettings):  The report settings, defaults to ReportSettings() 



**Returns:**
 
 - <b>`str`</b>:  The deviation report 

# <kbd>function</kbd> `create_trajectory_report`

```python
create_trajectory_report(
    trajectories: List[trajectopy.core.trajectory.Trajectory],
    report_settings: trajectopy.core.settings.report.ReportSettings = ReportSettings(single_plot_height=640, two_subplots_height=750, three_subplots_height=860, scatter_max_std=4.0, ate_unit_is_mm=False, directed_ate=True, histogram_opacity=0.7, histogram_bargap=0.1, histogram_barmode='overlay', histogram_yaxis_title='Count', plot_mode='lines+markers', scatter_mode='markers', scatter_colorscale='RdYlBu_r', scatter_axis_order='xy', scatter_marker_size=5, scatter_detailed=False, scatter_mapbox=False, scatter_mapbox_style='open-street-map', scatter_mapbox_zoom=15, scatter_mapbox_token='', pos_x_name='x', pos_y_name='y', pos_z_name='z', pos_x_unit='m', pos_y_unit='m', pos_z_unit='m', pos_dir_dev_x_name='along', pos_dir_dev_y_name='cross-h', pos_dir_dev_z_name='cross-v', rot_x_name='roll', rot_y_name='pitch', rot_z_name='yaw', rot_unit='°', single_plot_export=ExportSettings(format='png', height=540, width=800, scale=1), two_subplots_export=ExportSettings(format='png', height=540, width=800, scale=1), three_subplots_export=ExportSettings(format='png', height=750, width=800, scale=1))
) → str
```

Render a HTML report containing trajectory plots. 



**Args:**
 
 - <b>`trajectories`</b>:  List of trajectories to render. 
 - <b>`report_settings`</b>:  Report settings. 



**Returns:**
 HTML string of the rendered report including the trajectory plots. 

# <kbd>function</kbd> `dataclass`

```python
dataclass(
    cls=None,
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
    match_args=True,
    kw_only=False,
    slots=False,
    weakref_slot=False
)
```

Add dunder methods based on the fields defined in the class. 

Examines PEP 526 __annotations__ to determine fields. 

If init is true, an __init__() method is added to the class. If repr is true, a __repr__() method is added. If order is true, rich comparison dunder methods are added. If unsafe_hash is true, a __hash__() method is added. If frozen is true, fields may not be assigned to after instance creation. If match_args is true, the __match_args__ tuple is added. If kw_only is true, then by default all fields are keyword-only. If slots is true, a new class with a __slots__ attribute is returned. 

# <kbd>function</kbd> `estimate_alignment`

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
  traj_from (Trajectory)  traj_to (Trajectory) 
 - <b>`alignment_settings`</b> (AlignmentSettings, optional):  Settings for the alignment process. Defaults to AlignmentSettings(). 
 - <b>`matching_settings`</b> (MatchingSettings, optional):  Settings for the matching process. Defaults to MatchingSettings(). 



**Returns:**
 
 - <b>`AlignmentResult`</b>:  Result of the alignment process 

# <kbd>function</kbd> `field`

```python
field(
    default=<dataclasses._MISSING_TYPE object at 0x104d76350>,
    default_factory=<dataclasses._MISSING_TYPE object at 0x104d76350>,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=<dataclasses._MISSING_TYPE object at 0x104d76350>
)
```

Return an object to identify dataclass fields. 

default is the default value of the field.  default_factory is a 0-argument function called to initialize a field's value.  If init is true, the field will be a parameter to the class's __init__() function.  If repr is true, the field will be included in the object's repr().  If hash is true, the field will be included in the object's hash().  If compare is true, the field will be used in comparison functions.  metadata, if specified, must be a mapping which is stored but not otherwise examined by dataclass.  If kw_only is true, the field will become a keyword-only parameter to __init__(). 

It is an error to specify both default and default_factory. 

# <kbd>function</kbd> `match_trajectories`

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

Nearest Spatial This method matches two trajectories by finding the nearest pose in the target trajectory for each pose in the source trajectory. The distance between two poses is computed using the Euclidean distance between their positions. 

Nearest Temporal This method matches two trajectories using their timestamps by finding the nearest timestamp in the target trajectory for each timestamp in the source trajectory. 

Interpolation This method matches two trajectories by interpolating the timestamps of one trajectory to the timestamps of the other trajectory. The interpolation is linear for both positions and rotations (SLERP). 

Nearest Spatial Interpolated This method matches both trajectories spatially by requesting the nearest k positions from the reference trajectory for each pose in the test trajectory. Then, an interpolation is performed using a 3d line fit of the k nearest positions. After this operation, both trajectories will have the length of the test trajectory. This method does not support rotation matching. 



**Returns:**
 
 - <b>`Tuple[Trajectory, Trajectory]`</b>:  Matched trajectories 

# <kbd>function</kbd> `merge_trajectories`

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

# <kbd>function</kbd> `plot_ate`

```python
plot_ate(
    ate_results: Union[trajectopy.core.evaluation.ate_result.ATEResult, List[trajectopy.core.evaluation.ate_result.ATEResult]],
    plot_settings: trajectopy.core.settings.mpl_settings.MPLPlotSettings = MPLPlotSettings(scatter_cbar_show_zero=True, scatter_cbar_steps=4, scatter_no_axis=False, scatter_sigma_factor=3.0, ate_unit_is_mm=False, hist_as_stairs=False, heatmap_spacing=1.0, show_directed_devs=False)
) → Figure
```

Plots ATE for the given ATEResult(s) as a line plot using matplotlib. If available, the plot contains the position and rotation deviations. The x-axis depends on the sorting of the trajectory. 



**Args:**
 
 - <b>`ate_results`</b> (Union[ATEResult, List[ATEResult]]):  ATE result(s) to plot. 
 - <b>`plot_settings`</b> (MPLPlotSettings, optional):  Plot settings. Defaults to MPLPlotSettings(). 



**Returns:**
 
 - <b>`Figure`</b>:  Figure containing the plot. 

# <kbd>function</kbd> `plot_ate_bars`

```python
plot_ate_bars(
    ate_results: List[trajectopy.core.evaluation.ate_result.ATEResult],
    plot_settings: trajectopy.core.settings.mpl_settings.MPLPlotSettings = MPLPlotSettings(scatter_cbar_show_zero=True, scatter_cbar_steps=4, scatter_no_axis=False, scatter_sigma_factor=3.0, ate_unit_is_mm=False, hist_as_stairs=False, heatmap_spacing=1.0, show_directed_devs=False),
    mode: str = 'positions'
) → Figure
```

Plots multiple ATE results as bars for different characteristics (min, max, mean, median, rms, std) using matplotlib. 



**Args:**
 
 - <b>`ate_result`</b> (List[ATEResult]):  List of ATE results. 
 - <b>`plot_settings`</b> (MPLPlotSettings, optional):  Plot settings. Defaults to MPLPlotSettings(). 
 - <b>`mode`</b> (str, optional):  Mode to plot. Either 'positions' or 'rotations'. Defaults to 'positions'. 



**Returns:**
 
 - <b>`Figure`</b>:  Bar plot figure. 

# <kbd>function</kbd> `plot_ate_edf`

```python
plot_ate_edf(
    ate_results: Union[trajectopy.core.evaluation.ate_result.ATEResult, List[trajectopy.core.evaluation.ate_result.ATEResult]],
    plot_settings: trajectopy.core.settings.mpl_settings.MPLPlotSettings = MPLPlotSettings(scatter_cbar_show_zero=True, scatter_cbar_steps=4, scatter_no_axis=False, scatter_sigma_factor=3.0, ate_unit_is_mm=False, hist_as_stairs=False, heatmap_spacing=1.0, show_directed_devs=False)
) → Figure
```

Plots ATE EDF for the given ATEResult(s) as a line plot using matplotlib. The EDF (Empirical Distribution Function) shows the cummulative probability of the deviations. Using this plot, one can easily see how many percent of the deviations are below a certain value. 



**Args:**
 
 - <b>`ate_results`</b> (Union[ATEResult, List[ATEResult]]):  ATE result to plot. 
 - <b>`plot_settings`</b> (MPLPlotSettings, optional):  Plot settings. Defaults to MPLPlotSettings(). 



**Returns:**
 
 - <b>`Figure`</b>:  Figure containing the plot. 

# <kbd>function</kbd> `plot_compact_ate_hist`

```python
plot_compact_ate_hist(
    ate_result: trajectopy.core.evaluation.ate_result.ATEResult,
    plot_settings: trajectopy.core.settings.mpl_settings.MPLPlotSettings = MPLPlotSettings(scatter_cbar_show_zero=True, scatter_cbar_steps=4, scatter_no_axis=False, scatter_sigma_factor=3.0, ate_unit_is_mm=False, hist_as_stairs=False, heatmap_spacing=1.0, show_directed_devs=False)
) → Figure
```

Plots compact ATE histograms for the given ATEResult. The plot contains histograms for the position deviations and, if available, the rotation deviations. 



**Args:**
 
 - <b>`ate_result`</b> (ATEResult):  ATE result to plot. 
 - <b>`plot_settings`</b> (MPLPlotSettings, optional):  Plot settings. Defaults to MPLPlotSettings(). 



**Returns:**
 
 - <b>`Figure`</b>:  Figure containing the plot. 

# <kbd>function</kbd> `plot_correlation_heatmap`

```python
plot_correlation_heatmap(
    estimated_parameters: trajectopy.core.alignment.parameters.AlignmentParameters,
    enabled_only: bool = True
) → Figure
```

Plots the correlation heatmap of the alignment parameters using matplotlib. 



**Args:**
 
 - <b>`estimated_parameters`</b> (AlignmentParameters):  Estimated parameters. 
 - <b>`enabled_only`</b> (bool, optional):  Whether to consider only enabled parameters. Defaults to True. 



**Returns:**
 
 - <b>`plt.Figure`</b>:  Correlation heatmap figure. 

# <kbd>function</kbd> `plot_covariance_heatmap`

```python
plot_covariance_heatmap(
    estimated_parameters: trajectopy.core.alignment.parameters.AlignmentParameters,
    enabled_only: bool = True
) → Figure
```

Plots the covariance heatmap of the alignment parameters using matplotlib. 



**Args:**
 
 - <b>`estimated_parameters`</b> (AlignmentParameters):  Estimated parameters. 
 - <b>`enabled_only`</b> (bool, optional):  Whether to consider only enabled parameters. Defaults to True. 



**Returns:**
 
 - <b>`plt.Figure`</b>:  Covariance heatmap figure. 

# <kbd>function</kbd> `plot_rpe`

```python
plot_rpe(
    rpe_results: List[trajectopy.core.evaluation.rpe_result.RPEResult]
) → Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]
```

Plots the RPE results as a line plot with violin plots for the position and rotation deviations. 

Depending on the pair distance unit, the unit of the position deviations is either in meters/meters (%) or meters/seconds. The unit of the rotation deviations is respectively in degrees/m or degrees/second. 



**Args:**
 
 - <b>`rpe_results`</b> (list[RelativeTrajectoryDeviations]):  list of RelativeTrajectoryDeviations 



**Returns:**
 
 - <b>`Tuple[Figure, Figure]`</b>:  metric and time RPE plots 

# <kbd>function</kbd> `plot_trajectories`

```python
plot_trajectories(
    trajectories: List[trajectopy.core.trajectory.Trajectory],
    dim: int = 2
) → Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure, Optional[matplotlib.figure.Figure]]
```

Plots the trajectories in 2d or 3d using matplotlib. 



**Args:**
 
 - <b>`trajectories`</b> (List[Trajectory]):  List of trajectories to plot. 
 - <b>`dim`</b> (int, optional):  Dimension of the plot. Defaults to 2. 

This function creates one 2D or 3D plot for the xy(z) coordinates of the trajectories, one subplot for the xyz coordinates and one subplot for the rpy angles. 



**Returns:**
 
 - <b>`Tuple[Figure, Figure, Union[Figure, None]]`</b>:  Figures for the position, xyz and rpy plots. 

# <kbd>function</kbd> `rpe`

```python
rpe(
    trajectory_gt: trajectopy.core.trajectory.Trajectory,
    trajectory_est: trajectopy.core.trajectory.Trajectory,
    settings: trajectopy.core.settings.processing.ProcessingSettings = ProcessingSettings(alignment=AlignmentSettings(preprocessing=AlignmentPreprocessing(min_speed=0.0, time_start=0.0, time_end=0.0), estimation_settings=AlignmentEstimationSettings(trans_x=True, trans_y=True, trans_z=True, rot_x=True, rot_y=True, rot_z=True, scale=False, time_shift=False, use_x_speed=True, use_y_speed=True, use_z_speed=True, lever_x=False, lever_y=False, lever_z=False, sensor_rotation=False, auto_update=False), stochastics=AlignmentStochastics(std_xy_from=1.0, std_z_from=1.0, std_xy_to=1.0, std_z_to=1.0, std_roll_pitch=0.017453292519943295, std_yaw=0.017453292519943295, std_speed=1.0, error_probability=0.05, variance_estimation=False), metric_threshold=0.0001, time_threshold=0.0001), matching=MatchingSettings(method=<MatchingMethod.INTERPOLATION: 'interpolation'>, max_time_diff=0.01, max_distance=0.0, k_nearest=10), relative_comparison=RelativeComparisonSettings(pair_min_distance=100.0, pair_max_distance=800.0, pair_distance_step=100.0, pair_distance_unit=<PairDistanceUnit.METER: 'meter'>, use_all_pose_pairs=True), approximation=ApproximationSettings(fe_int_size=0.15, fe_min_obs=25, rot_approx_win_size=0.15), sorting=SortingSettings(discard_missing=True, voxel_size=0.05, movement_threshold=0.005, k_nearest=4))
) → RPEResult
```

Computes the relative pose error (RPE) between two trajectories. 



**Args:**
 
 - <b>`trajectory_gt`</b> (Trajectory):  Ground truth trajectory. 
 - <b>`trajectory_est`</b> (Trajectory):  Estimated trajectory. 
 - <b>`settings`</b> (ProcessingSettings, optional):  Processing settings. 

Description: The RPE is computed by comparing the relative poses between the estimated and ground truth trajectories. The pose distances are either defined in meters or in seconds depending on the settings. 



**Returns:**
 
 - <b>`RPEResult`</b>:  Result of the RPE computation. 

# <kbd>function</kbd> `scatter_ate`

```python
scatter_ate(
    ate_result: trajectopy.core.evaluation.ate_result.ATEResult,
    plot_settings: trajectopy.core.settings.mpl_settings.MPLPlotSettings = MPLPlotSettings(scatter_cbar_show_zero=True, scatter_cbar_steps=4, scatter_no_axis=False, scatter_sigma_factor=3.0, ate_unit_is_mm=False, hist_as_stairs=False, heatmap_spacing=1.0, show_directed_devs=False)
)
```

Plots the ATE results as a scatter plot with color-coded deviations. 



**Args:**
 
 - <b>`ate_result`</b> (ATEResult):  ATE result to plot. 
 - <b>`plot_settings`</b> (MPLPlotSettings, optional):  Plot settings. Defaults to MPLPlotSettings(). 

# <kbd>function</kbd> `show_report`

```python
show_report(report_text: str, filepath: str = '') → None
```

This function writes a report to a file and opens it in the default web browser. 



**Args:**
 
 - <b>`report_text`</b> (str):  The report text 
 - <b>`filepath`</b> (str, optional):  The file path to save the report. If not given, a random file name will be generated. 

# <kbd>function</kbd> `sort_spatially`

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

# <kbd>function</kbd> `write_report`

```python
write_report(output_file: str, report_text: str) → None
```

Writes a report to the given output file. 



**Args:**
 
 - <b>`output_file`</b> (str):  The output file path 
 - <b>`report_text`</b> (str):  The report text 
