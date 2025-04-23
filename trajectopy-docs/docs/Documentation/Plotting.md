
## <kbd>module</kbd> `trajectopy.plotting`
Trajectopy - Trajectory Evaluation in Python 

Gereon Tombrink, 2025 tombrink@igg.uni-bonn.de 


---

### <kbd>function</kbd> `plot_trajectories`

```python
plot_trajectories(
    trajectories: List[trajectopy.trajectory.Trajectory],
    scatter_3d: bool = False
) → Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure, Optional[matplotlib.figure.Figure]]
```

Plots the trajectories in 2d or 3d using matplotlib. 



**Args:**
 
 - <b>`trajectories`</b> (List[Trajectory]):  List of trajectories to plot. 
 - <b>`dim`</b> (int, optional):  Dimension of the plot. Defaults to 2. 

This function creates one 2D or 3D plot for the xy(z) coordinates of the trajectories, one subplot for the xyz coordinates and one subplot for the rpy angles. 



**Returns:**
 
 - <b>`Tuple[Figure, Figure, Union[Figure, None]]`</b>:  Figures for the position, xyz and rpy plots. 


---

### <kbd>function</kbd> `plot_correlation_heatmap`

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


---

### <kbd>function</kbd> `plot_covariance_heatmap`

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


---

### <kbd>function</kbd> `plot_ate_bars`

```python
plot_ate_bars(
    ate_results: List[trajectopy.core.evaluation.ate_result.ATEResult],
    plot_settings: trajectopy.settings.MPLPlotSettings = MPLPlotSettings(colorbar_show_zero_crossing=True, colorbar_steps=4, colorbar_max_std=3.0, scatter_hide_axes=False, scatter_3d=False, ate_unit_is_mm=False, hist_as_stairs=False, directed_ate=False),
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


---

### <kbd>function</kbd> `plot_compact_ate_hist`

```python
plot_compact_ate_hist(
    ate_result: trajectopy.core.evaluation.ate_result.ATEResult,
    plot_settings: trajectopy.settings.MPLPlotSettings = MPLPlotSettings(colorbar_show_zero_crossing=True, colorbar_steps=4, colorbar_max_std=3.0, scatter_hide_axes=False, scatter_3d=False, ate_unit_is_mm=False, hist_as_stairs=False, directed_ate=False)
) → Figure
```

Plots compact ATE histograms for the given ATEResult. The plot contains histograms for the position deviations and, if available, the rotation deviations. 



**Args:**
 
 - <b>`ate_result`</b> (ATEResult):  ATE result to plot. 
 - <b>`plot_settings`</b> (MPLPlotSettings, optional):  Plot settings. Defaults to MPLPlotSettings(). 



**Returns:**
 
 - <b>`Figure`</b>:  Figure containing the plot. 


---

### <kbd>function</kbd> `plot_ate`

```python
plot_ate(
    ate_results: Union[trajectopy.core.evaluation.ate_result.ATEResult, List[trajectopy.core.evaluation.ate_result.ATEResult]],
    plot_settings: trajectopy.settings.MPLPlotSettings = MPLPlotSettings(colorbar_show_zero_crossing=True, colorbar_steps=4, colorbar_max_std=3.0, scatter_hide_axes=False, scatter_3d=False, ate_unit_is_mm=False, hist_as_stairs=False, directed_ate=False)
) → Figure
```

Plots ATE for the given ATEResult(s) as a line plot using matplotlib. If available, the plot contains the position and rotation deviations. The x-axis depends on the sorting of the trajectory. 



**Args:**
 
 - <b>`ate_results`</b> (Union[ATEResult, List[ATEResult]]):  ATE result(s) to plot. 
 - <b>`plot_settings`</b> (MPLPlotSettings, optional):  Plot settings. Defaults to MPLPlotSettings(). 



**Returns:**
 
 - <b>`Figure`</b>:  Figure containing the plot. 


---

### <kbd>function</kbd> `plot_ate_edf`

```python
plot_ate_edf(
    ate_results: Union[trajectopy.core.evaluation.ate_result.ATEResult, List[trajectopy.core.evaluation.ate_result.ATEResult]],
    plot_settings: trajectopy.settings.MPLPlotSettings = MPLPlotSettings(colorbar_show_zero_crossing=True, colorbar_steps=4, colorbar_max_std=3.0, scatter_hide_axes=False, scatter_3d=False, ate_unit_is_mm=False, hist_as_stairs=False, directed_ate=False)
) → Figure
```

Plots ATE EDF for the given ATEResult(s) as a line plot using matplotlib. The EDF (Empirical Distribution Function) shows the cummulative probability of the deviations. Using this plot, one can easily see how many percent of the deviations are below a certain value. 



**Args:**
 
 - <b>`ate_results`</b> (Union[ATEResult, List[ATEResult]]):  ATE result to plot. 
 - <b>`plot_settings`</b> (MPLPlotSettings, optional):  Plot settings. Defaults to MPLPlotSettings(). 



**Returns:**
 
 - <b>`Figure`</b>:  Figure containing the plot. 


---

### <kbd>function</kbd> `plot_rpe`

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


---

### <kbd>function</kbd> `scatter_ate`

```python
scatter_ate(
    ate_result: trajectopy.core.evaluation.ate_result.ATEResult,
    plot_settings: trajectopy.settings.MPLPlotSettings = MPLPlotSettings(colorbar_show_zero_crossing=True, colorbar_steps=4, colorbar_max_std=3.0, scatter_hide_axes=False, scatter_3d=False, ate_unit_is_mm=False, hist_as_stairs=False, directed_ate=False)
) → Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]
```

Plots the ATE results as a scatter plot with color-coded deviations. 



**Args:**
 
 - <b>`ate_result`</b> (ATEResult):  ATE result to plot. 
 - <b>`plot_settings`</b> (MPLPlotSettings, optional):  Plot settings. Defaults to MPLPlotSettings(). 


