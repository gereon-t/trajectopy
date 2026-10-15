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

Trajectories can also be created from NumPy arrays (`Trajectory.from_arrays`, see
[File Formats](file_formats.md#creating-trajectories-in-python)) and read from ROS bags:

```python
from trajectopy.io.rosbag import trajectories_from_rosbag

trajectories = trajectories_from_rosbag("recording.bag")  # one trajectory per topic
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
| `ate_map_figure` | Top view colored by any deviation component |
| `rpe_figures` | Relative pose error over the pair distance |
| `parameter_matrix_figure` | Covariance / correlation of estimated alignment parameters |

All figures use the [plot settings](settings.md#plot-settings).

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
   *Compare with Reference → Absolute Error (ATE), Aligned First*. The result appears in the results table.
4. **Inspect**: select the result to see its statistics in the properties table and its deviations in the
   preview; press `V` (or *Plot*) for all plots.
5. **Save**: *File → Save* (`Ctrl+S`) keeps everything for later in a project file; *Export…* (`E`) saves single
   trajectories or results.

### Main Window

The left side lists the trajectories (name, EPSG code, reference) and the results (name, type). Below them,
the properties table shows all properties of the selected trajectories or results side by side; *Export…* saves
them as CSV. The preview on the right plots the selected trajectories, or the selected ATE results colored by
their deviations. On slow systems, it can be turned off with the *Preview* checkbox in the top right corner;
the tables and the properties then use the whole width.

Trajectory files (`.traj` and other ASCII files, ROS1 `.bag` files) are loaded into the trajectory table,
`.result` files into the results table. Long operations run in the background; a progress window shows that
Trajectopy is busy.

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
| Plot Settings | | Opens the [plot settings](settings.md#plot-settings), which apply to all plots opened afterwards. |
| Help | Check for Updates… | Checks GitHub for a newer version. |
| | Check for Updates at Startup | Checks quietly at every start and shows a link in the status bar if a newer version is available (on by default). |
| | About | Version and license information. |
| Theme | Dark / Light | Color theme. It starts with the system setting and follows when the system setting changes. |

**Projects** keep everything in one file (`.tpy`): all trajectories with their processing settings, the
reference, all results, the order of the tables and the plot settings. The title bar shows the name of the
project and a `*` while it has unsaved changes. Before these are lost (*New*, *Open*, closing Trajectopy),
Trajectopy offers to save them. A project file can also be opened by dropping it onto one of the tables, by
double-click on Windows (if chosen during installation) or on the command line. It is a zip archive of the
same `.traj`, `.json` and `.result` files that *Export…* writes, so these can be extracted, too.

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
| Compare with Reference | *Absolute Error (ATE), Aligned First*, *Absolute Error (ATE), as Is*, *Relative Pose Error (RPE)*, *Extract Matching Poses* |
| Align with Reference | *Estimate Alignment*, *Interactive Alignment…* (see below), *Match First Pose*, *Apply Existing Alignment…* |
| Coordinate System | *Transform to EPSG…*, *Adapt EPSG from Reference*, *Set EPSG Code (No Transformation)…* |
| Process | *Smooth (Approximate)*, *Resample to Time Grid…*, *Rearrange Axes / Angles…*, *Sort Spatially*, *Switch Order: Time ↔ Path*, *Divide into Laps*, *Merge*, *Average* |
| Entry | *Settings…*, *Rename*, *Duplicate*, *Export…*, *Export to Google Earth (KML)…*, *Remove*, *Remove Including Derived* |

Processing adds new trajectories (e.g. *Smooth* adds a smoothed copy) and comparisons add results; only the
coordinate system entries, *Rearrange Axes / Angles…* and *Switch Order: Time ↔ Path* change the trajectory itself.

The results context menu offers *Plot*, *Rename*, *Duplicate*, *Export…*, *Remove* and, for ATE results,
*Export to Google Earth (KML)…*. For alignments, *Choose Parameters to Apply…* selects which of the estimated
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

### Interactive Alignment

If the automatic alignment cannot find a solution on its own (e.g. trajectories in unrelated frames or
with a large time offset), a trajectory can be aligned manually: set a reference, then choose
*Align with Reference → Interactive Alignment…* in the context menu of the trajectory.

1. **Auto align** runs the spatio-temporal least squares alignment and shows its result. Only the parameters
   checked with *estimate* are estimated (preselected from the trajectory settings); the others keep their
   values. The start values are chosen automatically: the current values, the first pose matched, a fit on
   poses matched in time or the centers matched, whatever brings the trajectory closest to the reference.
   The leverarm can only be estimated for trajectories with orientations.
2. If Auto align does not find the right solution, **place the trajectory manually** and press
   *Refine manual alignment*: it runs the same least squares alignment, but starts exactly from the values shown.
   *Match first pose* puts the first position onto the reference's first position and aligns the initial
   direction of travel. It does not use the timestamps, so it also works with an unknown time offset.
   In the plots, drag ● to move and ◆ to rotate the trajectory. The top view rotates about the vertical
   axis (yaw), the side view about y (pitch) and the front view about x (roll).
   Single parameters can be fine-tuned with the jog sliders (drag and hold: the further you drag, the faster
   the value changes; they snap back when released), by typing values or with the spin box arrows / mouse
   wheel (Ctrl: 10× steps), or with the keyboard after clicking into the plots (arrows, PgUp/PgDn, Q/E;
   Shift: 10× faster, Ctrl: 10× finer).
3. **Apply** aligns the trajectory with the values shown and saves the alignment. If they come unchanged from
   the least squares alignment, the saved alignment includes its statistics (e.g. the covariance).

Position, heading and scale are relative to the trajectory's center, so rotating does not move the
trajectory away even for georeferenced coordinates.
