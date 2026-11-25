# User Guide

This guide covers the core functionality of Trajectopy: loading data, alignment, evaluation and visualization,
first with the Python API and then in the [GUI](#6-using-the-gui), which offers the same functionality.
The examples use the files in the [`example_data`](https://github.com/gereon-t/trajectopy/tree/main/example_data)
folder of the repository.

In all functions that compare two trajectories, `trajectory` is the trajectory to evaluate (or align) and `other`
is the reference (ground truth).

## 1. Loading and Saving Trajectories

Trajectories are loaded with the `Trajectory` class. ASCII files with any column layout can be read, as long as
their header describes it (see [File Formats](file_formats.md)).

```python
import trajectopy as tpy

trajectory = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
print(trajectory.name, len(trajectory), trajectory.has_orientation)

trajectory.to_file("copy.traj")  # trajectopy ASCII format
```

Files in the TUM, KITTI and EuRoC formats are recognized automatically
(see [File Formats](file_formats.md#tum-kitti-and-euroc-files)). Trajectories can also be created from NumPy
arrays (`Trajectory.from_arrays`, see [File Formats](file_formats.md#creating-trajectories-in-python)) and read
from ROS1 and ROS2 bags (poses, odometry, TF, paths and GNSS fixes, see
[File Formats](file_formats.md#ros-bags)):

```python
from trajectopy.io.rosbag import trajectories_from_rosbag

trajectories = trajectories_from_rosbag("recording.mcap")  # one trajectory per topic
```

## 2. Alignment

To compare two trajectories (e.g., estimated vs. ground truth) that are given in different coordinate frames,
they have to be aligned first. The alignment is estimated with a least squares adjustment from matched poses;
which parameters are estimated (similarity transformation, leverarm, time shift) is defined by the
[alignment settings](settings.md#alignment-settings).

```python
import trajectopy as tpy

gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

settings = tpy.ProcessingSettings()  # default settings

# Estimate the alignment of est_traj to gt_traj
alignment_result = tpy.estimate_alignment(
    trajectory=est_traj,
    other=gt_traj,
    alignment_settings=settings.alignment,
    matching_settings=settings.matching,
)
print(alignment_result.position_parameters)

# Apply it; inplace=False returns an aligned copy and leaves est_traj unchanged
est_traj_aligned = tpy.apply_alignment(est_traj, alignment_result, inplace=False)
```

The function `align` combines both steps:

```python
est_traj_aligned = tpy.align(trajectory=est_traj, other=gt_traj, inplace=False)
```

`apply_alignment` and `align` modify the given trajectory unless `inplace=False` is passed.

## 3. Evaluation

### Absolute Trajectory Error (ATE)

The ATE measures the global consistency of a trajectory with the differences between corresponding poses of
both trajectories. `ate()` aligns the trajectory to the reference first; pass `align=False` to evaluate it as it is.

```python
import trajectopy as tpy

gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

ate_result = tpy.ate(trajectory=est_traj, other=gt_traj)
print(f"Position ATE: {ate_result.pos_ate:.3f} m, rotation ATE: {ate_result.rot_ate:.3f} rad")

# Also return the alignment that was used
ate_result, alignment_result = tpy.ate(trajectory=est_traj, other=gt_traj, return_alignment=True)
```

The result contains the deviations of each matched pose (e.g. `pos_dev_x`, `pos_dev_along`) and their
statistics (e.g. `pos_dev_rms`, `pos_dev_max`, `pos_bias_x`); `to_dataframe()` returns the deviations as a table.

### Relative Pose Error (RPE)

The RPE measures the drift of a trajectory: the deviations between the relative poses of pose pairs that are a
given distance or time apart, normalized by that distance or time (see
[relative comparison settings](settings.md#relative-comparison-settings)). It does not require the trajectories to
be aligned.

```python
rpe_result = tpy.rpe(trajectory=est_traj, other=gt_traj)
print(f"Position RPE: {rpe_result.pos_rpe:.3f} {rpe_result.pos_drift_unit}")
print(f"Rotation RPE: {rpe_result.rot_rpe:.5f} {rpe_result.rot_drift_unit}")
```

### Custom Settings

All processing steps are configured with `ProcessingSettings`; see [Settings](settings.md) for all options.

```python
settings = tpy.ProcessingSettings()
settings.alignment.estimation_settings.scale = False
settings.alignment.estimation_settings.time_shift = True
settings.matching.method = tpy.MatchingMethod.NEAREST_TEMPORAL
settings.relative_comparison.pair_distance_unit = tpy.PairDistanceUnit.SECOND
settings.relative_comparison.pair_min_distance = 10.0
settings.relative_comparison.pair_max_distance = 60.0
settings.relative_comparison.pair_distance_step = 10.0

ate_result = tpy.ate(trajectory=est_traj, other=gt_traj, processing_settings=settings)
rpe_result = tpy.rpe(trajectory=est_traj, other=gt_traj, processing_settings=settings)

settings.to_file("settings.json")  # the GUI reads this format, too
settings = tpy.ProcessingSettings.from_file("settings.json")
```

### Saving Results

Results and alignments are saved as text files, which the GUI can open, too:

```python
ate_result.to_file("ate.result")
ate_result = tpy.ATEResult.from_file("ate.result")

rpe_result.to_file("rpe.result")
alignment_result.to_file("alignment.result")
```

### Percentiles, Confidence Intervals and Segments

Besides mean, median, RMS and maximum, ATE results offer percentiles and confidence intervals. Deviations of
trajectories are correlated in time, so the usual standard error (`std / sqrt(n)`) is far too optimistic; the
confidence intervals come from a block bootstrap whose blocks are twice as long as the correlation time of the
deviations. They show, for example, whether two estimates differ significantly.

```python
print(ate_result.pos_dev_p95)                     # 95th percentile of the position deviations [m]
print(ate_result.pos_dev_percentile(99))
low, high = ate_result.confidence_interval("position", "mean")  # or "rms"; "rotation" in radians
```

`segment_statistics` evaluates parts of the trajectory separately: consecutive windows of time or path length,
or segments given by a label of each deviation (e.g. tunnel / open sky). The figure *ATE Segments* shows the
windows.

```python
table = ate_result.segment_statistics(segment_length=100.0, by="path_length")  # pandas DataFrame
labels = np.where(in_tunnel, "tunnel", "open sky")                               # one label per deviation
table = ate_result.segment_statistics(labels=labels)
```

### Comparing Several Results

To benchmark several estimates against one reference, `summary_table` puts the main statistics of all results
into one table (one row per result), which `format_table` writes as Markdown, CSV or LaTeX. `write_report`
creates a self-contained HTML file with the summary, the statistics of each result, all figures and the
processing settings, e.g. to archive the evaluation of a paper.

```python
ate_results = [tpy.ate(trajectory=est, other=gt_traj) for est in (orb, sptam)]
rpe_results = [tpy.rpe(trajectory=est, other=gt_traj) for est in (orb, sptam)]

table = tpy.summary_table(ate_results, rpe_results)  # pandas DataFrame
print(tpy.format_table(table, "markdown"))           # or "csv", "latex"
tpy.write_report("report.html", ate_results=ate_results, rpe_results=rpe_results, settings=settings)
```

The notebook [`example_scripts/evaluation_walkthrough.ipynb`](https://github.com/gereon-t/trajectopy/blob/main/example_scripts/evaluation_walkthrough.ipynb)
walks through such a comparison with its output.

## 4. Visualization

Plotting requires the plotting extra (included in the GUI installation):

```bash
pip install "trajectopy[plot]"
```

`trajectopy.visualization` provides the same figures that the GUI shows. A figure is a page of plots;
some figures offer *variants*, e.g. the quantity a trajectory is colored by. Figures can be explored in
an interactive window or exported to PNG, JPG or SVG files (also without a display).

```python
import trajectopy as tpy
from trajectopy import visualization as vis

gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
orb_traj = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
sptam_traj = tpy.Trajectory.from_file("./example_data/KITTI_SPTAM.traj")

ate_results = [tpy.ate(orb_traj, gt_traj), tpy.ate(sptam_traj, gt_traj)]
rpe_results = [tpy.rpe(orb_traj, gt_traj), tpy.rpe(sptam_traj, gt_traj)]
ate_result, alignment = tpy.ate(orb_traj, gt_traj, return_alignment=True)

settings = tpy.PlotSettings(ate_unit_is_mm=True)

# Collections of figures (exactly what the GUI shows)
trajectory_figures = vis.trajectory_figures([gt_traj, orb_traj], settings)
deviation_figures = vis.deviation_figures(ate_results, rpe_results, settings)  # one or several results
alignment_figures = vis.alignment_figures(alignment.position_parameters, settings)

# Interactive window with one tab per figure (blocks until the window is closed)
vis.show(deviation_figures, title="KITTI Evaluation")

# Export all figures and variants, e.g. for a paper or a report
vis.export_figures(deviation_figures, "plots", export_format="svg")

# Single figures
vis.show(vis.ate_figure(ate_results, settings))
vis.ate_map_figure([ate_result], settings).export("ate_map.png", variant="Position", width=2400)
```

In the plot window, the mouse wheel zooms, dragging with the left button pans, dragging with the right button
zooms the x and y axes separately, and the *A* button in the lower left corner fits the plot to the data again.
Stacked plots (e.g. x, y and z over time) share their x-axis. *Export Plot…* and *Export All…* in the toolbar
save the figures.

Available figures:

| Function | Content |
| --- | --- |
| `positions_figure` | Top view of trajectories |
| `xyz_figure` / `rpy_figure` | Coordinates / orientations over time or path length |
| `trajectory_map_figure` | Top view colored by velocity or height |
| `ate_figure` | Position (and rotation) deviations over time or path length |
| `ate_components_figure` | Deviations of the single degrees of freedom |
| `ate_histogram_figure` | Histograms of the deviations |
| `ate_statistics_figure` | Min, max, mean, median, RMS and standard deviation |
| `ate_distribution_figure` | Empirical cumulative distribution of the deviations |
| `ate_correlation_figure` | Temporal correlation of the deviations and their standard deviation over time |
| `ate_map_figure` | Top view colored by any deviation component |
| `ate_speed_figure` | Deviations over speed (along-track by default) with the time offset estimated from their slope |
| `rpe_figures` | Relative pose error over the pair distance |
| `parameter_matrix_figure` | Covariance / correlation of estimated alignment parameters |

All figures use the [plot settings](settings.md#plot-settings).

### Temporal Correlation of Deviations

Deviations of a trajectory from a reference are usually correlated in time over seconds to minutes, for example
for GNSS/INS. Then, neighboring epochs do not provide independent information. `ate_result.correlation` describes
this for every deviation component (along-track, cross-track, x, y, z and, with orientations, roll, pitch and yaw):

```python
correlation = ate_result.correlation
cross_track = correlation.components["cross_h"]
print(cross_track.white_noise_share)  # share of the variance that is uncorrelated white noise
print(cross_track.correlation_time)  # correlation time of the correlated part [s]
print(cross_track.effective_samples)  # approximate number of independent deviations
print(correlation.cross_correlation_directed)  # correlation of along / cross-h / cross-v at the same epochs
```

Deviations are usually white noise (e.g. measurement noise of the reference, uncorrelated from one epoch to the
next) plus a correlated part (e.g. the GNSS/INS error). The white noise makes the autocorrelation drop at once
between the first two lags; then the correlated part decays. Its share is extrapolated to lag 0 from the first
lags, and its correlation time is the lag at which the autocorrelation drops to 1/e of that share. Without this,
a large white noise share would make the correlated part look like it lasted only a fraction of a sampling
interval. Deviations whose first lag is not significantly correlated are white noise (correlation time 0).

The autocorrelation is computed in chronological order, also for spatially sorted results, and gaps (e.g. losses
of a total station's lock) are respected: no deviation is correlated across a gap. Irregular sampling is allowed. The autocorrelation is computed up to a tenth of the number of deviations
(n / 10 sampling intervals), at most up to 600 s, since larger lags rely on too few pairs of deviations; longer
correlation times are reported as "> max. lag".
The correlation times and the number of independent deviations are also shown in the properties of ATE results,
and the plots of ATE results include the autocorrelation and the standard deviation over time.

A correlation time can only be determined reliably from data that is much longer than it: from one hour of data,
a correlation time of 30 s is uncertain by about 10 to 20 %. With a reference such as a synchronized total station,
the cross-track deviations are the most reliable measure, because timing errors affect only the along-track
direction.

## 5. Google Earth

Georeferenced trajectories (trajectories with an EPSG code) and ATE results can be exported as KML and opened in
Google Earth, which shows them in their real-world context: on imagery and terrain, and, where available,
between photorealistic 3D buildings. In the GUI, choose *Export to Google Earth (KML)…* in the context menu of a
trajectory or an ATE result; the dialog offers all options below and remembers them.

```python
# trajectory and ate_result must be georeferenced (EPSG code other than 0); the KITTI example data is not
trajectory.to_kml("track.kml", color="#1f77b4")  # one color: "#RRGGBB" or "#RRGGBBAA"
trajectory.to_kml("track.kml", field="speed", heights="ellipsoidal", extrude=True)
ate_result.to_kml("ate.kml", field="along_track", colormap="viridis", value_range=(-0.5, 0.5))
```

The trajectory is exported as a time-stamped track: the time slider of Google Earth plays it back in real
time order (also for spatially sorted trajectories). Playback needs Google Earth Pro (desktop); Google Earth on
the web shows the trajectory, but has no time slider.

**Colors**

- Trajectories are drawn in one color (`color`), or colored by a field (`field`): `speed`, `height` or `time`.
- ATE results are colored by one of their deviations (`field`): `position`, `x`, `y`, `z`, `along_track`,
  `horizontal_cross_track`, `vertical_cross_track` and, for trajectories with orientations, `rotation`, `roll`,
  `pitch` and `yaw`.
- `colormap`: `"rdylbu"` (blue = low, red = high, like the plots) or `"viridis"` (colorblind-safe).
- Like in the plots, the color scale is clipped to the mean ± 3 standard deviations (`max_std`, 0 = minimum to
  maximum). With `value_range=(lower, upper)`, it has fixed limits instead, so that the colors of several
  exports are comparable.
- `color_classes`: number of colors (default 11). The Places panel of Google Earth lists them with their value
  ranges (the legend); each class can be switched on and off. During playback, the colored line grows with the
  time slider.
- `line_width`: line width in pixels (default 3 for trajectories, 4 for ATE results).

**Heights**

- `heights="ground"` (default): the trajectory is clamped to the terrain of Google Earth; its own heights are
  ignored. With a `height_offset`, it floats that many meters above the terrain.
- `heights="ellipsoidal"`: the trajectory is drawn at its own heights, which are heights above the WGS84
  ellipsoid (e.g. from GNSS). Google Earth expects heights above sea level, so they are converted with the
  EGM96 geoid model (downloaded once from cdn.proj.org).
- `heights="sea_level"`: the trajectory is drawn at its own heights, which are already heights above sea
  level (e.g. orthometric or normal heights of a national height system). They are used as they are; the
  difference between national height systems and Google Earth's EGM96 is usually below a meter or two.
- With the wrong choice, the trajectory floats about 30-50 m above the ground or sinks into it.
  `height_offset` (meters) is added in all modes, e.g. to correct for the antenna height.
- `extrude=True` connects the trajectory to the ground with a semi-transparent vertical curtain, which makes
  heights easy to read (not for trajectories clamped to the ground).

## 6. Using the GUI

Start the GUI from the Start menu (Windows app) or with `trajectopy` (Python package, see
[Command Line](index.md#command-line)).

### A Typical Workflow

1. **Load trajectories**: drop files onto the trajectory table or press *Add*. On Windows, `.traj` files can also be
   opened by double-click (if chosen during installation), and files can be passed on the command line.
2. **Set the reference**: right-click the ground truth and choose *Set as Reference* (`R`).
3. **Compare**: select one or more other trajectories, right-click and choose
   *Compare with Reference → Absolute Error (ATE), with Alignment*. The result appears in the results table.
4. **Inspect**: select the result to see its statistics in the properties table and its deviations in the
   preview; press `V` (or *Plot*) for all plots.
5. **Save**: *File → Save* (`Ctrl+S`) keeps everything for later in a project file; *Export…* (`E`) saves single
   trajectories or results.

### Main Window

The left side lists the trajectories (name, EPSG code, reference) and the results (name, type). Below them,
the properties table shows all properties of the selected trajectories or results side by side; *Export…* saves
them as CSV. The preview on the right plots the selected trajectories, or the selected ATE results colored by
their deviations. The drop-down next to the *Preview* checkbox in the top right corner switches between the top
view (*2D*) and an orbitable *3D* view (drag to rotate, wheel to zoom, middle-drag to pan). On slow systems, the
preview can be turned off with the checkbox; the tables and the properties then use the whole width.

Trajectory files (`.traj` and other ASCII files, ROS1 `.bag` files) are loaded into the trajectory table,
`.result` files into the results table. Long operations run in the background; a progress window shows that
Trajectopy is busy.

**Groups**: trajectories computed from another one (aligned, smoothed, sorted, … copies) are grouped with it
automatically, so that the results of one measurement run stay together. Groups can be collapsed; selecting a group
selects all of its entries. *Group → New Group…* (`Ctrl+G`) in the context menu groups any trajectories or results,
*Rename Group…* names a group, *Ungroup* (`Ctrl+Shift+G`) takes entries out of their groups and *Move to Group* moves
them into another group. Entries can also be dragged onto a group (or an entry) to move them there, or between the
rows to take them out of their group. Groups are saved in the project.

### Menu Bar

| Menu | Entry | Description |
| --- | --- | --- |
| File | New (`Ctrl+N`) | Starts a new, empty project with default plot settings. |
| | Open… (`Ctrl+O`) | Opens a project file (`.tpy`); replaces the current project. |
| | Open Recent | The last ten projects. |
| | Save (`Ctrl+S`) | Saves the project to its file (asks for a file the first time). |
| | Save As… (`Ctrl+Shift+S`) | Saves the project to a new file. |
| | Import Session Folder… | Opens a session folder saved by earlier versions of Trajectopy as a new project. |
| | Exit (`Ctrl+Q`) | Closes Trajectopy. |
| View | Log (`Ctrl+L`) | Shows or hides the log panel at the bottom of the main window. |
| Plot Settings | | Opens the [plot settings](settings.md#plot-settings), which apply to all plots opened afterwards. |
| Help | Check for Updates… | Checks GitHub for a newer version. |
| | Check for Updates at Startup | Checks quietly at every start and shows a link in the status bar if a newer version is available (on by default). |
| | About | Version and license information. |
| Theme | Dark / Light | Color theme. It starts with the system setting and follows when the system setting changes. |

**Projects** keep everything in one file (`.tpy`): all trajectories with their processing settings, the
reference, all results, the order of the tables and the plot settings. The title bar shows the name of the
project and a `*` while it has unsaved changes. Before these are lost (*New*, *Open*, closing Trajectopy),
Trajectopy offers to save them. A project file can also be opened by dropping it onto one of the tables, by
double-click on Windows (if chosen during installation) or on the command line. Project files are binary, so
that large projects are saved and opened quickly; use *Export…* for files that other programs can read. Projects
saved by Trajectopy 5.1 can still be opened.

**Log**: Trajectopy writes what it does, warnings and errors to a log. *View → Log* (`Ctrl+L`) opens it at the
bottom of the main window; warnings are shown in orange, errors in red, and the context menu copies or clears the
messages. The log shows everything since Trajectopy was started, also if it is opened later, and it reopens at the
next start if it was open. The Windows app has no console window, so the log is the place to look when something
does not work as expected (Python installations also print the messages to the terminal they were started from).

**Updates**: the update check only contacts GitHub (it sends no data except the installed version in the request
header); it can be turned off in the Help menu. Python installations are updated with
`pip install --upgrade "trajectopy[gui]"`, the Windows app with the installer of the new version.

Closing Trajectopy offers to save unsaved changes and asks for confirmation while an operation is running.

### Processing Settings of a Trajectory

Every trajectory has its own [processing settings](settings.md#processing-settings) (alignment, matching,
relative comparison, approximation, sorting), which are used when it is processed. Edit them with *Settings…*
(`S`) in its context menu; the editor can also import and export them as JSON. When a trajectory file is
loaded, a JSON file with the same name in the same folder (e.g. `run1.json` next to `run1.traj`) is used as its
settings.

### Context Menus

All actions are in the context menus of the selected trajectories or results (right-click). Hover over an entry
to see what it does. Entries that are not available for the current selection are greyed out and their hint
explains why, e.g. *Compare with Reference* needs a reference: right-click the reference trajectory and choose
*Set as Reference*.

The trajectory context menu is grouped by task:

| Group | Entries |
| --- | --- |
| View | *Plot*, *Playback in 3D* (body frame axes at the current pose), *Timeline* (time spans of the trajectories, e.g. to check their overlap) |
| Reference | *Set as Reference* / *Unset Reference* |
| Compare with Reference | *Absolute Error (ATE), without Alignment*, *Absolute Error (ATE), with Alignment*, *Relative Pose Error (RPE)*, *Extract Matching Poses* |
| Align with Reference | *Estimate Alignment*, *Interactive Alignment…* (see below), *Match First Pose*, *Apply Existing Alignment…* |
| Coordinate System | *Transform to EPSG…*, *Adapt EPSG from Reference*, *Set EPSG Code (No Transformation)…* |
| Process | *Approximate*, *Resample to Time Grid…*, *Rearrange Axes / Angles…*, *Sort Spatially*, *Switch Order: Time ↔ Path*, *Divide into Laps*, *Merge*, *Average* |
| Entry | *Settings…*, *Rename*, *Duplicate*, *Export…*, *Export to Google Earth (KML)…*, *Group*, *Remove*, *Remove Group* |

Processing adds new trajectories (e.g. *Approximate* adds a smoothed copy) and comparisons add results; only the
coordinate system entries, *Rearrange Axes / Angles…* and *Switch Order: Time ↔ Path* change the trajectory itself.

The results context menu offers *Plot*, *Rename*, *Duplicate*, *Export…*, *Group*, *Remove* and, for ATE results,
*Export to Google Earth (KML)…*. *Export Report or Summary Table…* saves the selected ATE and RPE results as an
HTML report or as a summary table (Markdown, CSV or LaTeX) with one row per result. ATE results of spatially sorted trajectories can also be switched between
chronological and spatial order with *Switch Order: Time ↔ Path*, e.g. to plot the deviations over time. For alignments, *Choose Parameters to Apply…* selects which of the estimated
parameters are used when the alignment is applied to a trajectory (*Apply Existing Alignment…*).

### Keyboard Shortcuts

Keyboard shortcuts are shown next to the entries of the context menus. They are active when the corresponding
table has focus and do what the menu entry does for the selected rows.

| Shortcut | Trajectories table | Results table |
| --- | --- | --- |
| `V` | Plot | Plot |
| `R` / `Shift+R` | Set as reference / unset reference | |
| `T` | Transform to EPSG | |
| `M` | Merge (two or more trajectories) | |
| `S` | Settings | |
| `F2` (or `U`) | Rename | Rename |
| `Ctrl+D` (or `Ctrl+C`) | Duplicate | Duplicate |
| `E` | Export | Export |
| `Delete` | Remove | Remove |
| `Ctrl+G` / `Ctrl+Shift+G` | New group / ungroup | New group / ungroup |

### Interactive Alignment

Set a reference, then choose *Align with Reference → Interactive Alignment…* in the context menu of a trajectory.
The window helps in two situations: when the automatic alignment fails (unrelated frames, a different scale, a
large time offset), and when you want to understand *what* is wrong with an alignment, e.g. to calibrate a
leverarm or a time offset. It works in three steps; every step can be undone (`Ctrl+Z`, redo with `Ctrl+Y`).

**1 · Place roughly.** Bring the trajectory onto the reference, in space and in time.

- **Auto align** tries several rough placements (the current one, the first pose, a fit on poses at equal time,
  the centers, the time from the speed profiles), keeps the closest and refines it by least squares.
- **Drag** the trajectory in the view to move it; `Ctrl`+drag rotates and `Shift`+drag scales it about the pivot
  (+). Rotations turn about the viewing direction (top, side or front view). Double-click places the pivot.
  Keyboard: arrows move, `Q`/`E` rotate (`Shift`: 10× larger, `Ctrl`: 10× finer).
- **Pick point pairs**: click a point on the trajectory and the reference position it belongs to. One pair fixes
  the position, two add the heading (and the scale with *fit scale*), three or more the full rotation.
- **Match first pose** puts the first pose onto the reference's first pose.
- **Sync time** finds the time offset by comparing the speed profiles, which do not depend on the coordinate
  frame. You can also drag left/right in the speed plot (*Over time*).

**2 · Diagnose and correct.** Once the trajectories overlap, the *deviation at equal time* remains: the position of
the trajectory minus that of the reference at the same timestamp. It is split into *along-track* (direction of
travel of the reference), *cross-track* (to its left) and *vertical*. Every alignment parameter leaves its own
signature in it:

| Error | Signature | View |
| --- | --- | --- |
| Time shift | along-track deviation proportional to the speed, none at standstills | *Time shift* |
| Leverarm | constant in the body frame, also at standstills | *Shift and leverarm* |
| Shift | constant in the world frame: in the body frame it changes with the heading | *Shift and leverarm* |
| Heading | tangential deviation (about the pivot) proportional to the distance from it | *Rotation and scale* |
| Scale | radial deviation proportional to the distance from the pivot | *Rotation and scale* |
| Roll, pitch | vertical deviation growing along x or y | *Rotation and scale* |

The **diagnosis** lists the parameters that explain the deviation, most important first. It considers only the
parameters checked with *estimate* (under *Values and estimated parameters*, preselected from the trajectory
settings): *Explains* is the part of the deviation (RMS) that only this parameter accounts for compared to the
other estimated ones, *Correction* its error. If a parameter that is not estimated would reduce the deviation
considerably, the diagnosis points it out. A parameter is listed only if it explains clearly more than noise
would (taking into account that the deviation of trajectories is correlated from pose to pose) and if it
matters compared to the deviation that remains after correcting everything; so correcting removes rows but
does not bring up new ones made of noise. The corrections are determined for the estimated parameters jointly, so that effects that look alike are attributed correctly; for a vehicle, for example, a
leverarm along the direction of travel and a time shift both deviate along-track, but only the time shift grows
with the speed. Parameters that cannot be told apart with the data at hand are named (e.g. the vertical leverarm
and the vertical shift of a platform that is always level).

Selecting a row opens the view with the signature of the parameter. By default, a view removes the effects of all
other parameters (*Hide effects of other parameters*), so that its plot shows just this signature plus what no parameter
explains; uncheck it to see the actual deviation. Hovering over the data marks the poses of that time in the main
view, and the *deviation arrows* there show the pattern on the map (parallel: shift; along the path: time shift;
swirl: heading; away from the pivot: scale).

The plots keep their axes while you drag, so that nothing jumps; they adapt when the data leaves the visible range
or fills only a small part of it. After zooming or panning a plot, double-click it to let its axes follow the data
again. The ranking of the parameters is updated once you pause, so that the rows do not reorder under the mouse;
the values of the placement (*Values and estimated parameters*) are folded away until you open them. Messages
appear in the status bar at the bottom.

Corrections are made with *Correct* in the diagnosis, with *Correct all* (all estimated parameters together),
or with the buttons of the views,
e.g. *Correct shift and leverarm*, which uses only the standstills, where a time shift has no effect. Every
correction is estimated together with all other parameters, so the order in which they are applied does not
matter.

**3 · Least squares.** *Refine by least squares* runs the rigorous spatio-temporal least squares alignment of the
parameters checked with *estimate* (preselected from the trajectory settings), starting from the placement shown;
it runs in the background and can be cancelled. The values of all parameters can also be typed in.
**Apply** aligns the trajectory as shown and saves the alignment; if the placement comes unchanged from the least
squares alignment, its statistics (e.g. the covariance) are saved, too.

## 7. Command Line

Besides starting the GUI, the `trajectopy` command evaluates trajectories without Qt, e.g. in scripts, on servers
or in CI pipelines. All commands read trajectopy, TUM, KITTI and EuRoC files and ROS bags (select topics with
`--topic`), and take [processing settings](settings.md) with `--settings`. `trajectopy <command> --help` lists all
options.

```bash
# ATE of one or more estimates; prints a summary
trajectopy ate orb.traj sptam.traj --ref gt.traj

# ATE and RPE with custom settings, a summary table, the result files, all figures and an HTML report
trajectopy evaluate orb.traj sptam.traj --ref gt.traj --settings settings.json \
    --table summary.md --output-dir results --figures figures --report report.html

# RPE of an odometry topic of a ROS2 bag against a TUM file, without alignment
trajectopy rpe recording.mcap --topic /odom --ref gt.tum --no-align

# Align a trajectory and save it (any format) together with the alignment parameters
trajectopy align estimate.traj --ref gt.traj -o aligned.traj --parameters alignment.result

# Convert between formats and coordinate systems
trajectopy convert poses.txt --format kitti --to tum -o poses.tum
trajectopy convert gnss.traj --epsg 25832 -o gnss_utm.traj

# Show the trajectories of a file and the topics of a bag
trajectopy info recording.bag
```

The format of the summary table follows the file extension of `--table` (`.md`, `.tex`, otherwise CSV) or
`--table-format`. Errors are reported with exit code 1 (2 for invalid input), so the commands can be used in
pipelines.
