# User Guide

This guide covers the core functionality of Trajectopy: loading data, alignment, evaluation, and visualization. All functionality is available both via the Python API and the GUI application. The examples below use the Python API.

## 1. Loading Data

Trajectories are loaded using the `Trajectory` class. All ASCII text files are supported as long as they meet the requirements described in [File Formats](./file_formats.md).

```python
from trajectopy import Trajectory

# Load from file
traj = Trajectory.from_file("path/to/trajectory.traj")
```

## 2. Alignment

To compare two trajectories (e.g., estimated vs. ground truth), they often need to be aligned first.

```python
from trajectopy import Trajectory, ProcessingSettings, estimate_alignment, apply_alignment

gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

# default settings
processing_settings = ProcessingSettings()

# Estimate alignment
alignment_result = estimate_alignment(
    trajectory=est_traj,
    other=gt_traj,
    alignment_settings=processing_settings.alignment,
    matching_settings=processing_settings.matching,
)

# Apply alignment
est_traj_aligned = apply_alignment(est_traj, alignment_result)
```

The function `align` combines these two steps:

```python
from trajectopy import Trajectory, align

gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

est_traj_aligned = align(trajectory=est_traj, other=gt_traj, inplace=False)
```

## 3. Evaluation

Trajectopy provides standard metrics for trajectory evaluation.

### Absolute Trajectory Error (ATE)

ATE measures the global consistency of the trajectory by building differences between corresponding poses. Note that `ate()` already includes alignment by default. This can be disabled by setting the function parameter `align=False`.

```python
from trajectopy import Trajectory, ate, ProcessingSettings

gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

# default settings
processing_settings = ProcessingSettings()

# align = True is the default, but can be disabled if you want to evaluate the unaligned trajectories
ate_result = ate(other=gt_traj, trajectory=est_traj, processing_settings=processing_settings, align=True)
print(f"Position ATE: {ate_result.pos_ate}")
```

### Relative Pose Error (RPE)

RPE measures the pose drift over fixed time or distance intervals.

```python
from trajectopy import Trajectory, rpe, ProcessingSettings

gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

# default settings
processing_settings = ProcessingSettings()

rpe_result = rpe(other=gt_traj, trajectory=est_traj, processing_settings=processing_settings)
print(f"Position RPE: {rpe_result.pos_rpe}")
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

### Maps

Top views of georeferenced trajectories (trajectories with an EPSG code) can be drawn on a map, in the GUI
and in scripts:

```python
settings = tpy.PlotSettings(background_map=True, map_style="satellite")  # or "osm"
vis.show(vis.ate_map_figure(ate_results, settings))
```

- Map tiles are loaded from OpenStreetMap or Esri World Imagery without an API key. This requires an
  internet connection and sends the approximate area of your trajectories to the tile server. Tiles are
  cached on disk (`%LOCALAPPDATA%	rajectopy	iles` on Windows, `~/.cache/trajectopy/tiles` otherwise) and
  refreshed after 30 days.
- With a map, positions are shown in local east/north meters around the center of the trajectories
  (true north; the axes can be rotated slightly against a UTM grid due to meridian convergence).
- Trajectories without an EPSG code are drawn without a map.

### Google Earth

Georeferenced trajectories can be exported as KML and opened in Google Earth: `trajectory.to_kml("track.kml")`
or "Export to Google Earth (KML)" in the trajectory context menu of the GUI.

## 5. GUI Application

For a graphical interface, simply run:

```bash
trajectopy
```

This opens the application where you can drag & drop files, align them, and view results interactively.

### Manual Alignment

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

### Context Menus

All actions are in the context menu of the selected trajectories or results (right-click). Hover over an entry
to see what it does. Entries that are not available for the current selection are greyed out and their hint
explains why, e.g. *Compare with Reference* needs a reference: right-click the reference trajectory and choose
*Set as Reference*.

The trajectory context menu is grouped by task:

- **Plot**, **Playback in 3D**, **Timeline** and **Properties**
- **Set as Reference**, **Compare with Reference** (ATE, RPE, extracting matching poses) and
  **Align with Reference** (estimated, interactive, first pose, or applying an existing alignment)
- **Coordinate System**: transforming positions or declaring their EPSG code
- **Process**: smoothing, resampling, rearranging axes, sorting, dividing into laps, merging and averaging
- **Settings**, **Rename**, **Duplicate**, **Export** and **Remove**

### Keyboard Shortcuts

Keyboard shortcuts are shown next to the entries of the context menus. They are active when the corresponding
table has focus and do what the menu entry does for the selected rows.

| Shortcut | Trajectories table | Results table |
| --- | --- | --- |
| `V` | Plot | Plot |
| `P` | Properties | Properties |
| `R` / `Shift+R` | Set as reference / unset reference | |
| `T` | Transform to EPSG | |
| `M` | Merge (two or more trajectories) | |
| `S` | Settings | |
| `F2` (or `U`) | Rename | Rename |
| `Ctrl+D` (or `Ctrl+C`) | Duplicate | Duplicate |
| `E` | Export | Export |
| `Delete` | Remove | Remove |
