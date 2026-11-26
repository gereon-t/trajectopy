# Visualization

[`trajectopy.visualization`](../api/visualization.md) provides the same figures that the GUI shows. Plotting
requires the plotting extra (included in the GUI installation):

```console
pip install "trajectopy[plot]"
```

## Figures

A [`Figure`](../api/visualization.md#trajectopy.visualization.Figure) is a page of plots. Some figures offer
*variants*, e.g. the deviation component a map is colored by. Figures are created by functions that take the
trajectories or results to show and the [plot settings](settings.md#plot-settings):

```python
import trajectopy as tpy
from trajectopy import visualization as vis

reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
orb = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
sptam = tpy.Trajectory.from_file("./example_data/KITTI_SPTAM.traj")

ate_results = [tpy.ate(orb, reference), tpy.ate(sptam, reference)]
rpe_results = [tpy.rpe(orb, reference), tpy.rpe(sptam, reference)]

settings = tpy.PlotSettings(ate_unit_is_mm=True)

# Collections of figures (exactly what the GUI shows)
trajectory_figures = vis.trajectory_figures([reference, orb], settings)
deviation_figures = vis.deviation_figures(ate_results, rpe_results, settings)
alignment_figures = vis.alignment_figures(ate_results[0].alignment.position_parameters, settings)

# Single figures
ate_figure = vis.ate_figure(ate_results, settings)
```

Several results are drawn into the same figure, so that they can be compared directly.

| Function | Content |
| --- | --- |
| `positions_figure` | Top view of trajectories |
| `xyz_figure` / `rpy_figure` | Coordinates / orientations over time or path length |
| `trajectory_map_figure` | Top view colored by speed or height |
| `trajectory_figures` | All of the above |
| `ate_figure` | Position (and rotation) deviations over time or path length |
| `ate_components_figure` | Deviations of the single degrees of freedom |
| `ate_histogram_figure` | Histograms of the deviations |
| `ate_statistics_figure` | Minimum, maximum, mean, median, RMS and standard deviation |
| `ate_boxplot_figure` | Box plots of the deviations, one box per result |
| `ate_distribution_figure` | Empirical cumulative distribution of the deviations |
| `ate_correlation_figure` | Temporal correlation of the deviations and their standard deviation over time |
| `ate_map_figure` | Top view colored by any deviation component |
| `ate_speed_figure` | Deviations over speed, with the time offset estimated from their slope |
| `rpe_figures` | Relative pose error over the pair distance |
| `deviation_figures` | All ATE and RPE figures |
| `parameter_matrix_figure` | Covariance or correlation matrix of estimated alignment parameters |
| `alignment_figures` | Covariance and correlation matrix |

`ate_speed_figure` reveals synchronization errors: a time offset Δt between the trajectories shifts the
positions along the track by speed times Δt. `speed_fit` and `smoothed_speed` compute the fitted line and the
smoothed speed it is based on.

## Showing Figures

`show` opens an interactive window with one tab per figure and blocks until it is closed:

<!-- skip-test: opens a window -->
```python
vis.show(deviation_figures, title="KITTI Evaluation")
vis.show(vis.ate_map_figure(ate_results, settings))
```

In the plot window, the mouse wheel zooms, dragging with the left button pans, dragging with the right button
zooms the x and y axes separately, and the *A* button in the lower left corner fits the plot to the data again.
Stacked plots (e.g. x, y and z over time) share their x-axis. *Export Plot…* and *Export All…* in the toolbar
save the figures.

Inside a running Qt application, pass `block=False`; the window is returned and stays open.

## Exporting Figures

Figures are exported to PNG, JPG or SVG files, also without a display (e.g. on a server):

```python
paths = vis.export_figures(deviation_figures, "plots", export_format="svg")  # all figures and variants

ate_figure.export("ate.png", width=2400)
vis.ate_map_figure(ate_results, settings).export("ate_map.png", variant=f"{ate_results[0].name}: Position")
```

The variant names of a figure are listed in `figure.variants`.

## Appearance

The [plot settings](settings.md#plot-settings) control units, line widths, fonts, the color scale and the export
format. `set_theme(dark=True)` switches figures created afterwards to a dark background.
