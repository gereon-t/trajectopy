
## <kbd>module</kbd> `trajectopy.core.alignment.parameters`
Trajectopy - Trajectory Evaluation in Python 

Gereon Tombrink, 2025 tombrink@igg.uni-bonn.de 

**Global Variables**
---------------
- **UNIT_FORMAT_RULES**


---

### <kbd>class</kbd> `AlignmentParametersError`








---

### <kbd>class</kbd> `Parameter`
Class representing a Parameter 

A parameter holds a stochastic value with some variance. It can be enabled and disabled. 

If a parameter is disabled, its value value is equal to the default value. The default value of a parameter should be chosen in a way so that the parameter has no effect on computations. For example, a default scale value of 1 will not affect any computations this scale parameter is involved in. The same holds true for a rotation or translation of 0. The disabling of parameters is meant to be used to exclude parameter from being estimated during least-squares adjustment. 

### <kbd>method</kbd> `Parameter.__init__`

```python
__init__(
    value: float,
    variance: float = 0.0,
    default: float = 0.0,
    enabled: bool = True,
    name: str = '',
    unit: trajectopy.definitions.Unit = <Unit.NONE: 6>
) → None
```






---

##### <kbd>property</kbd> Parameter.value





---

##### <kbd>property</kbd> Parameter.variance







---

#### <kbd>method</kbd> `Parameter.disable`

```python
disable() → None
```





---

#### <kbd>method</kbd> `Parameter.enable`

```python
enable() → None
```






---

### <kbd>class</kbd> `ParameterSet`
Abstract class representing a set of parameters 

This class groups related parameters. For example, 3 parameters for a 3d translation. 

### <kbd>method</kbd> `ParameterSet.__init__`

```python
__init__(
    enabled: bool = True,
    covariance_matrix: numpy.ndarray = <factory>
) → None
```






---

##### <kbd>property</kbd> ParameterSet.any_enabled





---

##### <kbd>property</kbd> ParameterSet.enabled_bool_list





---

##### <kbd>property</kbd> ParameterSet.enabled_indices





---

##### <kbd>property</kbd> ParameterSet.num_enabled





---

##### <kbd>property</kbd> ParameterSet.values





---

##### <kbd>property</kbd> ParameterSet.values_enabled





---

##### <kbd>property</kbd> ParameterSet.variances





---

##### <kbd>property</kbd> ParameterSet.variances_enabled







---

#### <kbd>method</kbd> `ParameterSet.disable`

```python
disable() → None
```





---

#### <kbd>method</kbd> `ParameterSet.enable`

```python
enable() → None
```





---

#### <kbd>method</kbd> `ParameterSet.get_covariance_matrix`

```python
get_covariance_matrix(enabled_only: bool = True) → ndarray
```





---

#### <kbd>method</kbd> `ParameterSet.set_covariance_matrix`

```python
set_covariance_matrix(cov_matrix: numpy.ndarray) → None
```





---

#### <kbd>method</kbd> `ParameterSet.to_dict`

```python
to_dict(enabled_only: bool = True) → Dict[str, List[float]]
```





---

#### <kbd>method</kbd> `ParameterSet.to_name_list`

```python
to_name_list(enabled_only: bool = True, lower_case: bool = True) → List[str]
```





---

#### <kbd>method</kbd> `ParameterSet.to_string_list`

```python
to_string_list(enabled_only: bool = True) → List[str]
```





---

#### <kbd>method</kbd> `ParameterSet.validate_covariance`

```python
validate_covariance()
```

Checks dimensions of covariance matrix and sets variances accordingly 


---

### <kbd>class</kbd> `HelmertTransformation`
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

##### <kbd>property</kbd> HelmertTransformation.any_enabled





---

##### <kbd>property</kbd> HelmertTransformation.enabled_bool_list





---

##### <kbd>property</kbd> HelmertTransformation.enabled_indices





---

##### <kbd>property</kbd> HelmertTransformation.num_enabled





---

##### <kbd>property</kbd> HelmertTransformation.rotation





---

##### <kbd>property</kbd> HelmertTransformation.rotation_matrix





---

##### <kbd>property</kbd> HelmertTransformation.rotation_set





---

##### <kbd>property</kbd> HelmertTransformation.translation





---

##### <kbd>property</kbd> HelmertTransformation.values





---

##### <kbd>property</kbd> HelmertTransformation.values_enabled





---

##### <kbd>property</kbd> HelmertTransformation.variances





---

##### <kbd>property</kbd> HelmertTransformation.variances_enabled







---

#### <kbd>method</kbd> `HelmertTransformation.apply_to`

```python
apply_to(xyz: numpy.ndarray) → ndarray
```





---

#### <kbd>method</kbd> `HelmertTransformation.disable`

```python
disable() → None
```





---

#### <kbd>method</kbd> `HelmertTransformation.enable`

```python
enable() → None
```





---

#### <kbd>method</kbd> `HelmertTransformation.get_covariance_matrix`

```python
get_covariance_matrix(enabled_only: bool = True) → ndarray
```





---

#### <kbd>method</kbd> `HelmertTransformation.set_covariance_matrix`

```python
set_covariance_matrix(cov_matrix: numpy.ndarray) → None
```





---

#### <kbd>method</kbd> `HelmertTransformation.to_dict`

```python
to_dict(enabled_only: bool = True) → Dict[str, List[float]]
```





---

#### <kbd>method</kbd> `HelmertTransformation.to_name_list`

```python
to_name_list(enabled_only: bool = True, lower_case: bool = True) → List[str]
```





---

#### <kbd>method</kbd> `HelmertTransformation.to_string_list`

```python
to_string_list(enabled_only: bool = True) → List[str]
```





---

#### <kbd>method</kbd> `HelmertTransformation.validate_covariance`

```python
validate_covariance()
```

Checks dimensions of covariance matrix and sets variances accordingly 


---

### <kbd>class</kbd> `Leverarm`
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

##### <kbd>property</kbd> Leverarm.any_enabled





---

##### <kbd>property</kbd> Leverarm.enabled_bool_list





---

##### <kbd>property</kbd> Leverarm.enabled_indices





---

##### <kbd>property</kbd> Leverarm.num_enabled





---

##### <kbd>property</kbd> Leverarm.values





---

##### <kbd>property</kbd> Leverarm.values_enabled





---

##### <kbd>property</kbd> Leverarm.variances





---

##### <kbd>property</kbd> Leverarm.variances_enabled







---

#### <kbd>method</kbd> `Leverarm.apply_to`

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

#### <kbd>method</kbd> `Leverarm.disable`

```python
disable() → None
```





---

#### <kbd>method</kbd> `Leverarm.enable`

```python
enable() → None
```





---

#### <kbd>method</kbd> `Leverarm.get_covariance_matrix`

```python
get_covariance_matrix(enabled_only: bool = True) → ndarray
```





---

#### <kbd>method</kbd> `Leverarm.set_covariance_matrix`

```python
set_covariance_matrix(cov_matrix: numpy.ndarray) → None
```





---

#### <kbd>method</kbd> `Leverarm.to_dict`

```python
to_dict(enabled_only: bool = True) → Dict[str, List[float]]
```





---

#### <kbd>method</kbd> `Leverarm.to_name_list`

```python
to_name_list(enabled_only: bool = True, lower_case: bool = True) → List[str]
```





---

#### <kbd>method</kbd> `Leverarm.to_string_list`

```python
to_string_list(enabled_only: bool = True) → List[str]
```





---

#### <kbd>method</kbd> `Leverarm.validate_covariance`

```python
validate_covariance()
```

Checks dimensions of covariance matrix and sets variances accordingly 


---

### <kbd>class</kbd> `AlignmentParameters`
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

##### <kbd>property</kbd> AlignmentParameters.any_enabled





---

##### <kbd>property</kbd> AlignmentParameters.enabled_bool_list





---

##### <kbd>property</kbd> AlignmentParameters.enabled_indices





---

##### <kbd>property</kbd> AlignmentParameters.helmert





---

##### <kbd>property</kbd> AlignmentParameters.leverarm





---

##### <kbd>property</kbd> AlignmentParameters.num_enabled





---

##### <kbd>property</kbd> AlignmentParameters.sim3_matrix





---

##### <kbd>property</kbd> AlignmentParameters.values





---

##### <kbd>property</kbd> AlignmentParameters.values_enabled





---

##### <kbd>property</kbd> AlignmentParameters.variances





---

##### <kbd>property</kbd> AlignmentParameters.variances_enabled







---

#### <kbd>method</kbd> `AlignmentParameters.apply_settings`

```python
apply_settings(settings: trajectopy.settings.AlignmentEstimationSettings) → None
```

Applies the estimation settings to the parameters by enabling or disabling them 

---

#### <kbd>method</kbd> `AlignmentParameters.disable`

```python
disable() → None
```





---

#### <kbd>method</kbd> `AlignmentParameters.enable`

```python
enable() → None
```





---

#### <kbd>classmethod</kbd> `AlignmentParameters.from_file`

```python
from_file(filename: str) → AlignmentParameters
```

Reads the alignment parameters from a file 



**Args:**
 
 - <b>`filename`</b> (str):  Path to the file 



**Returns:**
 
 - <b>`AlignmentParameters`</b>:  AlignmentParameters instance 

---

#### <kbd>classmethod</kbd> `AlignmentParameters.from_settings`

```python
from_settings(
    settings: trajectopy.settings.AlignmentEstimationSettings
) → AlignmentParameters
```





---

#### <kbd>method</kbd> `AlignmentParameters.get_covariance_matrix`

```python
get_covariance_matrix(enabled_only: bool = True) → ndarray
```





---

#### <kbd>method</kbd> `AlignmentParameters.params_labels`

```python
params_labels(enabled_only: bool = True, lower_case: bool = True) → List[str]
```





---

#### <kbd>method</kbd> `AlignmentParameters.set_covariance_matrix`

```python
set_covariance_matrix(cov_matrix: numpy.ndarray) → None
```





---

#### <kbd>method</kbd> `AlignmentParameters.to_dataframe`

```python
to_dataframe() → DataFrame
```





---

#### <kbd>method</kbd> `AlignmentParameters.to_dict`

```python
to_dict(enabled_only: bool = True) → Dict[str, List[float]]
```





---

#### <kbd>method</kbd> `AlignmentParameters.to_estimation_settings`

```python
to_estimation_settings() → AlignmentEstimationSettings
```





---

#### <kbd>method</kbd> `AlignmentParameters.to_file`

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

#### <kbd>method</kbd> `AlignmentParameters.to_name_list`

```python
to_name_list(enabled_only: bool = True, lower_case: bool = True) → List[str]
```





---

#### <kbd>method</kbd> `AlignmentParameters.to_string_list`

```python
to_string_list(enabled_only: bool = True) → List[str]
```





---

#### <kbd>method</kbd> `AlignmentParameters.validate_covariance`

```python
validate_covariance()
```

Checks dimensions of covariance matrix and sets variances accordingly 


---

### <kbd>class</kbd> `SensorRotationParameters`
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

##### <kbd>property</kbd> SensorRotationParameters.any_enabled





---

##### <kbd>property</kbd> SensorRotationParameters.enabled_bool_list





---

##### <kbd>property</kbd> SensorRotationParameters.enabled_indices





---

##### <kbd>property</kbd> SensorRotationParameters.num_enabled





---

##### <kbd>property</kbd> SensorRotationParameters.rotation





---

##### <kbd>property</kbd> SensorRotationParameters.rotation_matrix





---

##### <kbd>property</kbd> SensorRotationParameters.rotation_set





---

##### <kbd>property</kbd> SensorRotationParameters.values





---

##### <kbd>property</kbd> SensorRotationParameters.values_enabled





---

##### <kbd>property</kbd> SensorRotationParameters.variances





---

##### <kbd>property</kbd> SensorRotationParameters.variances_enabled







---

#### <kbd>method</kbd> `SensorRotationParameters.disable`

```python
disable() → None
```





---

#### <kbd>method</kbd> `SensorRotationParameters.enable`

```python
enable() → None
```





---

#### <kbd>classmethod</kbd> `SensorRotationParameters.from_file`

```python
from_file(filename: str) → SensorRotationParameters
```





---

#### <kbd>method</kbd> `SensorRotationParameters.get_covariance_matrix`

```python
get_covariance_matrix(enabled_only: bool = True) → ndarray
```





---

#### <kbd>method</kbd> `SensorRotationParameters.set_covariance_matrix`

```python
set_covariance_matrix(cov_matrix: numpy.ndarray) → None
```





---

#### <kbd>method</kbd> `SensorRotationParameters.to_dict`

```python
to_dict(enabled_only: bool = True) → Dict[str, List[float]]
```





---

#### <kbd>method</kbd> `SensorRotationParameters.to_file`

```python
to_file(filename: str) → None
```





---

#### <kbd>method</kbd> `SensorRotationParameters.to_name_list`

```python
to_name_list(enabled_only: bool = True, lower_case: bool = True) → List[str]
```





---

#### <kbd>method</kbd> `SensorRotationParameters.to_string_list`

```python
to_string_list(enabled_only: bool = True) → List[str]
```





---

#### <kbd>method</kbd> `SensorRotationParameters.validate_covariance`

```python
validate_covariance()
```

Checks dimensions of covariance matrix and sets variances accordingly 


