
## <kbd>module</kbd> `trajectopy.report`
Trajectopy - Trajectory Evaluation in Python 

Gereon Tombrink, 2025 tombrink@igg.uni-bonn.de 

**Global Variables**
---------------
- **TEMPLATES_PATH**

---

### <kbd>function</kbd> `create_alignment_report`

```python
create_alignment_report(
    alignment_parameters: trajectopy.core.alignment.parameters.AlignmentParameters,
    name: str = 'Alignment',
    report_settings: trajectopy.settings.ReportSettings = ReportSettings(single_plot_height=640, two_subplots_height=750, three_subplots_height=860, scatter_max_std=4.0, ate_unit_is_mm=False, directed_ate=False, histogram_opacity=0.7, histogram_bargap=0.1, histogram_barmode='overlay', histogram_yaxis_title='Count', plot_mode='lines+markers', scatter_mode='markers', scatter_colorscale='RdYlBu_r', scatter_axis_order='xy', scatter_marker_size=5, scatter_detailed=False, scatter_mapbox=False, scatter_mapbox_style='open-street-map', scatter_mapbox_zoom=15, scatter_mapbox_token='', pos_x_name='x', pos_y_name='y', pos_z_name='z', pos_x_unit='m', pos_y_unit='m', pos_z_unit='m', pos_dir_dev_x_name='along', pos_dir_dev_y_name='cross-h', pos_dir_dev_z_name='cross-v', rot_x_name='roll', rot_y_name='pitch', rot_z_name='yaw', rot_unit='°', single_plot_export=ExportSettings(format='png', height=540, width=800, scale=1), two_subplots_export=ExportSettings(format='png', height=540, width=800, scale=1), three_subplots_export=ExportSettings(format='png', height=750, width=800, scale=1))
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


---

### <kbd>function</kbd> `create_trajectory_report`

```python
create_trajectory_report(
    trajectories: List[trajectopy.trajectory.Trajectory],
    report_settings: trajectopy.settings.ReportSettings = ReportSettings(single_plot_height=640, two_subplots_height=750, three_subplots_height=860, scatter_max_std=4.0, ate_unit_is_mm=False, directed_ate=False, histogram_opacity=0.7, histogram_bargap=0.1, histogram_barmode='overlay', histogram_yaxis_title='Count', plot_mode='lines+markers', scatter_mode='markers', scatter_colorscale='RdYlBu_r', scatter_axis_order='xy', scatter_marker_size=5, scatter_detailed=False, scatter_mapbox=False, scatter_mapbox_style='open-street-map', scatter_mapbox_zoom=15, scatter_mapbox_token='', pos_x_name='x', pos_y_name='y', pos_z_name='z', pos_x_unit='m', pos_y_unit='m', pos_z_unit='m', pos_dir_dev_x_name='along', pos_dir_dev_y_name='cross-h', pos_dir_dev_z_name='cross-v', rot_x_name='roll', rot_y_name='pitch', rot_z_name='yaw', rot_unit='°', single_plot_export=ExportSettings(format='png', height=540, width=800, scale=1), two_subplots_export=ExportSettings(format='png', height=540, width=800, scale=1), three_subplots_export=ExportSettings(format='png', height=750, width=800, scale=1))
) → str
```

Render a HTML report containing trajectory plots. 



**Args:**
 
 - <b>`trajectories`</b>:  List of trajectories to render. 
 - <b>`report_settings`</b>:  Report settings. 



**Returns:**
 HTML string of the rendered report including the trajectory plots. 


---

### <kbd>function</kbd> `create_deviation_report`

```python
create_deviation_report(
    ate_result: Union[trajectopy.core.evaluation.ate_result.ATEResult, list[trajectopy.core.evaluation.ate_result.ATEResult]],
    rpe_result: Optional[trajectopy.core.evaluation.rpe_result.RPEResult, list[trajectopy.core.evaluation.rpe_result.RPEResult]],
    report_settings: trajectopy.settings.ReportSettings = ReportSettings(single_plot_height=640, two_subplots_height=750, three_subplots_height=860, scatter_max_std=4.0, ate_unit_is_mm=False, directed_ate=False, histogram_opacity=0.7, histogram_bargap=0.1, histogram_barmode='overlay', histogram_yaxis_title='Count', plot_mode='lines+markers', scatter_mode='markers', scatter_colorscale='RdYlBu_r', scatter_axis_order='xy', scatter_marker_size=5, scatter_detailed=False, scatter_mapbox=False, scatter_mapbox_style='open-street-map', scatter_mapbox_zoom=15, scatter_mapbox_token='', pos_x_name='x', pos_y_name='y', pos_z_name='z', pos_x_unit='m', pos_y_unit='m', pos_z_unit='m', pos_dir_dev_x_name='along', pos_dir_dev_y_name='cross-h', pos_dir_dev_z_name='cross-v', rot_x_name='roll', rot_y_name='pitch', rot_z_name='yaw', rot_unit='°', single_plot_export=ExportSettings(format='png', height=540, width=800, scale=1), two_subplots_export=ExportSettings(format='png', height=540, width=800, scale=1), three_subplots_export=ExportSettings(format='png', height=750, width=800, scale=1))
)
```

Create a HTML deviation report containing the absolute trajectory error and relative pose error results. 



**Args:**
 
 - <b>`ate_result`</b> (Union[ATEResult, list[ATEResult]]):  The absolute trajectory error results (one or multiple) 
 - <b>`rpe_result`</b> (Union[RPEResult, list[RPEResult]]):  The relative pose error results (one or multiple) 
 - <b>`report_settings`</b> (ReportSettings):  The report settings, defaults to ReportSettings() 



**Returns:**
 
 - <b>`str`</b>:  The deviation report 


---

### <kbd>function</kbd> `write_report`

```python
write_report(output_file: str, report_text: str) → None
```

Writes a report to the given output file. 



**Args:**
 
 - <b>`output_file`</b> (str):  The output file path 
 - <b>`report_text`</b> (str):  The report text 


---

### <kbd>function</kbd> `show_report`

```python
show_report(report_text: str, filepath: str = '') → None
```

This function writes a report to a file and opens it in the default web browser. 



**Args:**
 
 - <b>`report_text`</b> (str):  The report text 
 - <b>`filepath`</b> (str, optional):  The file path to save the report. If not given, a random file name will be generated. 


