
## <kbd>module</kbd> `trajectopy.settings`




**Global Variables**
---------------
- **METRIC_THRESHOLD**
- **TIME_THRESHOLD**

---

### <kbd>function</kbd> `comparison_method_from_string`

```python
comparison_method_from_string(string: str) → <enum 'ComparisonMethod'>
```






---

### <kbd>class</kbd> `Settings`
Base Class for Settings 

### <kbd>method</kbd> `Settings.__init__`

```python
__init__() → None
```








---

#### <kbd>method</kbd> `Settings.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

#### <kbd>method</kbd> `Settings.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

#### <kbd>classmethod</kbd> `Settings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `Settings.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `Settings.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `Settings.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `Settings.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `Settings.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `AlignmentPreprocessing`
Dataclass defining alignment preprocessing configuration 

### <kbd>method</kbd> `AlignmentPreprocessing.__init__`

```python
__init__(
    min_speed: float = 0.0,
    time_start: float = 0.0,
    time_end: float = 0.0
) → None
```








---

#### <kbd>method</kbd> `AlignmentPreprocessing.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

#### <kbd>method</kbd> `AlignmentPreprocessing.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

#### <kbd>classmethod</kbd> `AlignmentPreprocessing.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `AlignmentPreprocessing.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `AlignmentPreprocessing.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `AlignmentPreprocessing.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `AlignmentPreprocessing.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `AlignmentPreprocessing.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `AlignmentEstimationSettings`
Dataclass defining which parameters to estimate during the alignment process 

### <kbd>method</kbd> `AlignmentEstimationSettings.__init__`

```python
__init__(
    translation_x: bool = True,
    translation_y: bool = True,
    translation_z: bool = True,
    rotation_x: bool = True,
    rotation_y: bool = True,
    rotation_z: bool = True,
    scale: bool = False,
    time_shift: bool = False,
    leverarm_x: bool = False,
    leverarm_y: bool = False,
    leverarm_z: bool = False,
    sensor_rotation: bool = False
) → None
```






---

##### <kbd>property</kbd> AlignmentEstimationSettings.all_disabled





---

##### <kbd>property</kbd> AlignmentEstimationSettings.all_lq_disabled

Returns True if all parameters estimated with Least-Squares are disabled 

---

##### <kbd>property</kbd> AlignmentEstimationSettings.enabled_lq_parameter_filter

Returns a list of bools indicating which parameters estimated within LQ are enabled 

---

##### <kbd>property</kbd> AlignmentEstimationSettings.helmert_enabled





---

##### <kbd>property</kbd> AlignmentEstimationSettings.helmert_filter





---

##### <kbd>property</kbd> AlignmentEstimationSettings.leverarm_enabled





---

##### <kbd>property</kbd> AlignmentEstimationSettings.leverarm_filter





---

##### <kbd>property</kbd> AlignmentEstimationSettings.lq_parameter_filter





---

##### <kbd>property</kbd> AlignmentEstimationSettings.short_mode_str

Returns a short string describing the enabled parameters 

---

##### <kbd>property</kbd> AlignmentEstimationSettings.time_shift_enabled





---

##### <kbd>property</kbd> AlignmentEstimationSettings.time_shift_filter







---

#### <kbd>classmethod</kbd> `AlignmentEstimationSettings.all`

```python
all(sensor_rotation: bool = True) → AlignmentEstimationSettings
```





---

#### <kbd>method</kbd> `AlignmentEstimationSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

#### <kbd>method</kbd> `AlignmentEstimationSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

#### <kbd>classmethod</kbd> `AlignmentEstimationSettings.from_bool_list`

```python
from_bool_list(bool_list: List[bool]) → AlignmentEstimationSettings
```





---

#### <kbd>classmethod</kbd> `AlignmentEstimationSettings.from_components`

```python
from_components(
    similarity: bool = False,
    time_shift: bool = False,
    leverarm: bool = False,
    sensor_rotation: bool = False
) → AlignmentEstimationSettings
```





---

#### <kbd>classmethod</kbd> `AlignmentEstimationSettings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `AlignmentEstimationSettings.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `AlignmentEstimationSettings.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `AlignmentEstimationSettings.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `AlignmentEstimationSettings.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `AlignmentEstimationSettings.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `AlignmentStochastics`
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

##### <kbd>property</kbd> AlignmentStochastics.var_roll_pitch





---

##### <kbd>property</kbd> AlignmentStochastics.var_speed_to





---

##### <kbd>property</kbd> AlignmentStochastics.var_xy_from





---

##### <kbd>property</kbd> AlignmentStochastics.var_xy_to





---

##### <kbd>property</kbd> AlignmentStochastics.var_yaw





---

##### <kbd>property</kbd> AlignmentStochastics.var_z_from





---

##### <kbd>property</kbd> AlignmentStochastics.var_z_to







---

#### <kbd>method</kbd> `AlignmentStochastics.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

#### <kbd>method</kbd> `AlignmentStochastics.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

#### <kbd>classmethod</kbd> `AlignmentStochastics.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `AlignmentStochastics.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `AlignmentStochastics.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `AlignmentStochastics.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `AlignmentStochastics.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `AlignmentStochastics.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `AlignmentSettings`
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
    preprocessing: trajectopy.settings.AlignmentPreprocessing = <factory>,
    estimation_settings: trajectopy.settings.AlignmentEstimationSettings = <factory>,
    stochastics: trajectopy.settings.AlignmentStochastics = <factory>,
    metric_threshold: float = 0.0001,
    time_threshold: float = 0.0001
) → None
```








---

#### <kbd>method</kbd> `AlignmentSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

#### <kbd>method</kbd> `AlignmentSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

#### <kbd>classmethod</kbd> `AlignmentSettings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `AlignmentSettings.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `AlignmentSettings.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `AlignmentSettings.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `AlignmentSettings.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `AlignmentSettings.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `CubicApproximationSettings`
Dataclass defining approximation configuration 

### <kbd>method</kbd> `CubicApproximationSettings.__init__`

```python
__init__(
    position_interval_size: float = 0.15,
    position_min_observations: int = 25,
    rotation_window_size: float = 0.15
) → None
```








---

#### <kbd>method</kbd> `CubicApproximationSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

#### <kbd>method</kbd> `CubicApproximationSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

#### <kbd>classmethod</kbd> `CubicApproximationSettings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `CubicApproximationSettings.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `CubicApproximationSettings.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `CubicApproximationSettings.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `CubicApproximationSettings.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `CubicApproximationSettings.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `PairDistanceUnit`








---

### <kbd>class</kbd> `ComparisonMethod`








---

### <kbd>class</kbd> `RelativeComparisonSettings`
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
    pair_distance_unit: trajectopy.settings.PairDistanceUnit = <PairDistanceUnit.METER: 'meter'>,
    use_all_pose_pairs: bool = True
) → None
```








---

#### <kbd>method</kbd> `RelativeComparisonSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```





---

#### <kbd>method</kbd> `RelativeComparisonSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```





---

#### <kbd>classmethod</kbd> `RelativeComparisonSettings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `RelativeComparisonSettings.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `RelativeComparisonSettings.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `RelativeComparisonSettings.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `RelativeComparisonSettings.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `RelativeComparisonSettings.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `MatchingMethod`








---

### <kbd>class</kbd> `InterpolationMethod`








---

### <kbd>class</kbd> `MatchingSettings`
MatchingSettings(method: trajectopy.settings.MatchingMethod = <MatchingMethod.INTERPOLATION: 'interpolation'>, max_time_diff: float = 0.01, max_distance: float = 0.0, max_gap_size: float = 10.0, k_nearest: int = 2) 

### <kbd>method</kbd> `MatchingSettings.__init__`

```python
__init__(
    method: trajectopy.settings.MatchingMethod = <MatchingMethod.INTERPOLATION: 'interpolation'>,
    max_time_diff: float = 0.01,
    max_distance: float = 0.0,
    max_gap_size: float = 10.0,
    k_nearest: int = 2
) → None
```








---

#### <kbd>method</kbd> `MatchingSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```





---

#### <kbd>method</kbd> `MatchingSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```





---

#### <kbd>classmethod</kbd> `MatchingSettings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `MatchingSettings.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `MatchingSettings.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `MatchingSettings.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `MatchingSettings.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `MatchingSettings.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `MPLPlotSettings`
Dataclass defining plot configuration 

### <kbd>method</kbd> `MPLPlotSettings.__init__`

```python
__init__(
    colorbar_show_zero_crossing: bool = True,
    colorbar_steps: int = 4,
    colorbar_max_std: float = 3.0,
    scatter_hide_axes: bool = False,
    scatter_3d: bool = False,
    scatter_smooth: bool = False,
    scatter_smooth_window: int = 5,
    ate_unit_is_mm: bool = False,
    ate_remove_above: float = 0.0,
    hist_as_stairs: bool = False,
    hist_percentile: float = 1.0,
    directed_ate: bool = False,
    dofs_tab: bool = True,
    velocity_tab: bool = True,
    height_tab: bool = True
) → None
```






---

##### <kbd>property</kbd> MPLPlotSettings.unit_multiplier





---

##### <kbd>property</kbd> MPLPlotSettings.unit_str







---

#### <kbd>method</kbd> `MPLPlotSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

#### <kbd>method</kbd> `MPLPlotSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

#### <kbd>classmethod</kbd> `MPLPlotSettings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `MPLPlotSettings.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `MPLPlotSettings.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `MPLPlotSettings.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `MPLPlotSettings.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `MPLPlotSettings.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `PlotBackend`








---

### <kbd>class</kbd> `SortingSettings`
This class stores all sorting settings 

### <kbd>method</kbd> `SortingSettings.__init__`

```python
__init__(
    voxel_size: float = 0.05,
    movement_threshold: float = 0.005,
    k_nearest: int = 4
) → None
```








---

#### <kbd>method</kbd> `SortingSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

#### <kbd>method</kbd> `SortingSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

#### <kbd>classmethod</kbd> `SortingSettings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `SortingSettings.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `SortingSettings.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `SortingSettings.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `SortingSettings.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `SortingSettings.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `ProcessingSettings`
Settings for processing the trajectory. 

### <kbd>method</kbd> `ProcessingSettings.__init__`

```python
__init__(
    alignment: trajectopy.settings.AlignmentSettings = <factory>,
    matching: trajectopy.settings.MatchingSettings = <factory>,
    relative_comparison: trajectopy.settings.RelativeComparisonSettings = <factory>,
    approximation: trajectopy.settings.CubicApproximationSettings = <factory>,
    sorting: trajectopy.settings.SortingSettings = <factory>
) → None
```








---

#### <kbd>method</kbd> `ProcessingSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

#### <kbd>method</kbd> `ProcessingSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

#### <kbd>classmethod</kbd> `ProcessingSettings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `ProcessingSettings.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `ProcessingSettings.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `ProcessingSettings.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `ProcessingSettings.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `ProcessingSettings.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `ExportSettings`
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

#### <kbd>method</kbd> `ExportSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

#### <kbd>method</kbd> `ExportSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

#### <kbd>classmethod</kbd> `ExportSettings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `ExportSettings.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `ExportSettings.to_config`

```python
to_config() → dict
```





---

#### <kbd>method</kbd> `ExportSettings.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `ExportSettings.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `ExportSettings.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `ExportSettings.update_from_dict`

```python
update_from_dict(dct: dict)
```






---

### <kbd>class</kbd> `ReportSettings`
ReportSettings class represents the settings for generating reports. 


- `single_plot_height` (int): The height of a single plot. Default value is 450. 
- `two_subplots_height` (int): The height of two subplots. Default value is 540. 
- `three_subplots_height` (int): The height of three subplots. Default value is 750. 
- `scatter_max_std` (float): The upper colorbar limit is set to the mean plus this value times the standard deviation of the data. This is useful to prevent outliers from dominating the colorbar. Default value is 4.0. 
- `ate_unit_is_mm` (bool): Indicates whether the unit of Absolute Trajectory Error (ATE) is millimeters. Default value is False. 
- `ate_remove_above` (float): Cap ATE at this value, if set to 0.0, no cap is applied. Default value is 0.0. 
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
- `scatter_show_individual_dofs` (bool): Indicates whether to show scatter plots for each degree of freedom. Default value is False. 
- `scatter_plot_on_map` (bool): Indicates whether to use mapbox for scatter plots. Default value is False. 
- `scatter_mapbox_style` (str): The mapbox style for scatter plots. Default value is "open-street-map". 
- `scatter_mapbox_zoom` (int): The zoom level for scatter plots. Default value is 15. 
- `pos_x_name` (str): The name of the x-coordinate in position data. Default value is "x". 
- `pos_y_name` (str): The name of the y-coordinate in position data. Default value is "y". 
- `pos_z_name` (str): The name of the z-coordinate in position data. Default value is "z". 
- `pos_x_unit` (str): The unit of the x-coordinate in position data. Default value is "m". 
- `pos_y_unit` (str): The unit of the y-coordinate in position data. Default value is "m". 
- `pos_z_unit` (str): The unit of the z-coordinate in position data. Default value is "m". 
- `directed_pos_dev_x_name` (str): The name of the along-track direction deviation in position data. Default value is "along". 
- `directed_pos_dev_y_name` (str): The name of the horizontal-cross-track direction deviation in position data. Default value is "cross-h". 
- `directed_pos_dev_z_name` (str): The name of the vertical-cross-track direction deviation in position data. Default value is "cross-v". 
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
    single_plot_height: int = 750,
    two_subplots_height: int = 750,
    three_subplots_height: int = 860,
    scatter_max_std: float = 4.0,
    ate_unit_is_mm: bool = False,
    ate_remove_above: float = 0.0,
    directed_ate: bool = False,
    histogram_opacity: float = 0.7,
    histogram_bargap: float = 0.1,
    histogram_barmode: str = 'overlay',
    histogram_yaxis_title: str = 'Count',
    plot_mode: str = 'lines+markers',
    scatter_mode: str = 'markers',
    scatter_colorscale: str = 'RdYlBu_r',
    scatter_axis_order: str = 'xy',
    scatter_marker_size: int = 5,
    scatter_show_individual_dofs: bool = False,
    scatter_smooth: bool = False,
    scatter_smooth_window: int = 5,
    scatter_plot_on_map: bool = False,
    scatter_mapbox_style: str = 'open-street-map',
    scatter_mapbox_zoom: int = 15,
    scatter_mapbox_token: str = '',
    pos_x_name: str = 'x',
    pos_y_name: str = 'y',
    pos_z_name: str = 'z',
    pos_x_unit: str = 'm',
    pos_y_unit: str = 'm',
    pos_z_unit: str = 'm',
    directed_pos_dev_x_name: str = 'along',
    directed_pos_dev_y_name: str = 'cross-h',
    directed_pos_dev_z_name: str = 'cross-v',
    rot_x_name: str = 'roll',
    rot_y_name: str = 'pitch',
    rot_z_name: str = 'yaw',
    rot_unit: str = '°',
    single_plot_export: trajectopy.settings.ExportSettings = <factory>,
    two_subplots_export: trajectopy.settings.ExportSettings = <factory>,
    three_subplots_export: trajectopy.settings.ExportSettings = <factory>
) → None
```








---

#### <kbd>method</kbd> `ReportSettings.decoder`

```python
decoder(name: str, value: Any) → Any
```

Decoder for json deserialization of dataclasses 

---

#### <kbd>method</kbd> `ReportSettings.encoder`

```python
encoder(name: str, value: Any) → Any
```

Encoder for json serialization of dataclasses 

---

#### <kbd>classmethod</kbd> `ReportSettings.from_dict`

```python
from_dict(dct: dict) → Settings
```





---

#### <kbd>classmethod</kbd> `ReportSettings.from_file`

```python
from_file(path: str) → Settings
```





---

#### <kbd>method</kbd> `ReportSettings.to_dict`

```python
to_dict() → dict
```





---

#### <kbd>method</kbd> `ReportSettings.to_file`

```python
to_file(path: str)
```





---

#### <kbd>method</kbd> `ReportSettings.to_json`

```python
to_json()
```





---

#### <kbd>method</kbd> `ReportSettings.update_from_dict`

```python
update_from_dict(dct: dict)
```






