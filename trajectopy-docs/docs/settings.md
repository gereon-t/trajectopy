# Settings

Trajectopy has two kinds of settings:

- **Processing settings** (`ProcessingSettings`): how trajectories are aligned, matched, compared, smoothed
  and sorted. In the GUI, every trajectory has its own (*Settings…* in its context menu).
- **Plot settings** (`PlotSettings`): how results are plotted. In the GUI, they apply to all plots
  (*Plot Settings* in the menu bar).

Both can be created in Python, changed attribute by attribute, and saved to or loaded from JSON files:

```python
import trajectopy as tpy

settings = tpy.ProcessingSettings()
settings.alignment.estimation_settings.time_shift = True
settings.matching.method = tpy.MatchingMethod.NEAREST_TEMPORAL
settings.to_file("settings.json")

settings = tpy.ProcessingSettings.from_file("settings.json")
plot_settings = tpy.PlotSettings(ate_unit_is_mm=True, line_width=2)
```

In a JSON file, the settings are nested like the attributes; options of enumerations are written in lower case
(e.g. `"method": "nearest_temporal"`). Settings missing in a file keep their default values.

## Processing Settings

| Attribute | Contents |
| :--- | :--- |
| `alignment` | Trajectory alignment: similarity transformation, time shift, leverarm |
| `matching` | Matching of corresponding poses of two trajectories |
| `relative_comparison` | Relative trajectory comparison (RPE) |
| `approximation` | Smoothing (approximation) of trajectories |
| `sorting` | Spatial sorting of trajectories |

### Alignment Settings

The alignment estimates the parameters that transform a trajectory onto a reference with a least squares
adjustment of matched poses. Its settings have four parts.

#### Preprocessing (`alignment.preprocessing`)

| Setting | Description | Default |
| --- | --- | --- |
| `min_speed` | Only poses faster than this (m/s) are used. | `0.0` |
| `time_start` | Only poses after this time are used, in seconds relative to the first common timestamp of both trajectories. | `0.0` |
| `time_end` | Only poses before this time are used, in seconds relative to the first common timestamp (`0` = until the end). | `0.0` |
| `max_samples` | Maximum number of matched poses used for the alignment. For larger trajectories, a new random subset of this size is drawn on each run, which keeps the least squares adjustment fast and memory-friendly (`0` = all poses). | `5000` |

#### Estimated Parameters (`alignment.estimation_settings`)

| Setting | Description | Default |
| --- | --- | --- |
| `translation_x`, `translation_y`, `translation_z` | Translation of the similarity transformation | `True` |
| `rotation_x`, `rotation_y`, `rotation_z` | Rotation of the similarity transformation | `True` |
| `scale` | Scale of the similarity transformation | `False` |
| `time_shift` | Time shift between both trajectories | `False` |
| `leverarm_x`, `leverarm_y`, `leverarm_z` | Leverarm between both trajectories (needs orientations) | `False` |
| `sensor_rotation` | Constant offsets between the roll, pitch and yaw angles of both trajectories. They are computed independently of the least squares adjustment. | `False` |

#### Stochastics (`alignment.stochastics`)

Standard deviations of the observations, which weight them in the least squares adjustment.

| Setting | Description | Default |
| --- | --- | --- |
| `std_xy_from`, `std_z_from` | Horizontal / vertical position of the trajectory to align (m) | `1.0` |
| `std_xy_to`, `std_z_to` | Horizontal / vertical position of the reference (m) | `1.0` |
| `std_roll_pitch` | Roll and pitch (rad) | `0.01745` (1°) |
| `std_yaw` | Yaw (rad) | `0.01745` (1°) |
| `std_speed` | Speed (m/s) | `1.0` |
| `error_probability` | Error probability of the statistical tests | `0.05` |
| `variance_estimation` | Estimate the variance factor a posteriori | `False` |

#### Convergence (`alignment`)

Usually, these can be left at their default values.

| Setting | Description | Default |
| --- | --- | --- |
| `metric_threshold` | Iteration threshold of the least squares adjustment for the metric parameters | `0.0001` |
| `time_threshold` | Iteration threshold of the least squares adjustment for the time shift (s) | `0.0001` |

### Matching Settings

Matching finds the corresponding poses of two trajectories, for the alignment and for the ATE.

| Setting | Description | Default |
| --- | --- | --- |
| `matching.method` | Matching method (`MatchingMethod`), see below | `INTERPOLATION` |
| `matching.max_time_diff` | Maximum time difference of poses matched by their timestamps (s) | `0.01` |
| `matching.max_distance` | Maximum distance of poses matched spatially (m; `0` = no limit) | `0.0` |
| `matching.max_gap_size` | Maximum gap within a trajectory that is bridged by interpolation (s; `INTERPOLATION` method) | `10.0` |
| `matching.k_nearest` | Number of nearest neighbors for `NEAREST_SPATIAL_INTERPOLATED` | `2` |

#### Matching Methods

- **`INTERPOLATION`**: the poses of one trajectory are interpolated at the timestamps of the other. Positions are
  interpolated linearly, rotations with SLERP.
- **`NEAREST_TEMPORAL`**: each pose is matched with the pose of the other trajectory with the nearest timestamp
  (within `max_time_diff`).
- **`NEAREST_SPATIAL`**: each pose is matched with the nearest position of the other trajectory (Euclidean distance,
  within `max_distance`). This needs no common time reference.
- **`NEAREST_SPATIAL_INTERPOLATED`**: for each pose of the trajectory to evaluate, the `k_nearest` positions of the
  reference are requested and a 3D line is fitted through them; the pose is matched with its projection onto that
  line. The matched rotations are the chordal L2 mean of the neighbors' rotations. Afterwards, both trajectories
  have the length of the trajectory to evaluate.

### Relative Comparison Settings

| Setting | Description | Default |
| --- | --- | --- |
| `relative_comparison.pair_min_distance` | Smallest distance between the poses of a pair | `100.0` |
| `relative_comparison.pair_max_distance` | Largest distance between the poses of a pair | `800.0` |
| `relative_comparison.pair_distance_step` | Step between the pair distances | `100.0` |
| `relative_comparison.pair_distance_unit` | Unit of the pair distances (`PairDistanceUnit`): `METER` or `SECOND` | `METER` |
| `relative_comparison.use_all_pose_pairs` | Use all (overlapping) pose pairs instead of consecutive (non-overlapping) ones | `True` |

#### RPE Background

For this metric, relative pose-pair differences are compared. The distance between two poses can be specified by the user and can be either time- or distance-based. The comparison involves finding pose pairs separated by a specific distance or time interval, computing the relative translation and rotation between the reference and estimated pose pairs, and calculating the translational and rotational difference normalized by the distance or time that separated the poses.

1. Find pose pair separated by e.g. 100 m in reference trajectory. This pair represents the start and end poses of a sequence of size $N$.
2. Find the corresponding pose pair in estimated trajectory
3. Compute relative translation and rotation between the reference pose pair

    $\Delta_{P~ref} = P_{ref, 1}^{-1} \cdot P_{ref, N}$

4. Compute relative translation and rotation between the estimated pose pair

    $\Delta_{P~est} = P_{est, 1}^{-1} \cdot P_{est, N}$

5. Compute transformation between 3) and 4)

6. Compute translation and rotation error from 5)

7. Divide 6) by the distance or the time that separated both poses (e.g. 100 m).

This metric does not require both trajectories to be aligned. Units are m/m (%) and deg/m for distance-based
comparison, and m/s and deg/s for time-based comparison.

Example: with the default settings (100 m to 800 m in steps of 100 m), pose pairs are formed for the distances
100 m, 200 m, …, 800 m.

### Approximation Settings

Trajectopy currently supports piece-wise cubic approximation for position components and moving average for rotation components. Piece-wise cubic means that the trajectory is divided into intervals of a specified size, and within each interval, a cubic polynomial is fitted to the position data. The fitted polynomial is then used to approximate the position values within that interval.

The interval and window sizes refer to the index of the trajectory: seconds for trajectories sorted by time,
meters for spatially sorted trajectories.

| Setting | Description | Default |
| --- | --- | --- |
| `approximation.position_approximation_method` | Method for the positions (`PositionApproximationMethod`): `CUBIC` | `CUBIC` |
| `approximation.position_interval_size` | Size of the intervals of the piece-wise cubic approximation | `0.15` |
| `approximation.position_min_observations` | Minimum number of observations per interval | `25` |
| `approximation.rotation_approximation_method` | Method for the rotations (`RotationApproximationMethod`): `AVERAGE_IN_WINDOW` (moving average, see `rotation_window_size`) or `AVERAGE_PER_PATH_LENGTH` (for trajectories that pass the same track several times: averages the rotations of all laps at the same path length, see `max_path_length_gap`) | `AVERAGE_IN_WINDOW` |
| `approximation.rotation_window_size` | Size of the moving average window for the rotations | `0.15` |
| `approximation.max_path_length_gap` | Maximum path length gap (m) that is bridged when the rotations of several laps are averaged | `0.15` |

### Sorting Settings

Sorting involves two steps:

1. Smoothing the trajectory using Moving Least Squares (MLS). The k nearest neighbors of each point are used to fit a 3D line, and the point is projected onto this line. This step reduces noise and outliers in the trajectory.
2. Shortest path approximation by constructing a minimum spanning tree (MST) based on the smoothed points.

| Setting | Description | Default |
| --- | --- | --- |
| `sorting.voxel_size` | Size of the voxel grid (m). Instead of the raw points, the centroids of the occupied voxels are used for the nearest neighbor searches of the MLS. | `0.05` |
| `sorting.movement_threshold` | The MLS iterations stop when all points move less than this (m). | `0.005` |
| `sorting.k_nearest` | Number of nearest voxels used by the MLS | `4` |

## Plot Settings

All plots, in the GUI and in `trajectopy.visualization`, use the same `PlotSettings`:

| Setting | Description | Default |
| --- | --- | --- |
| `ate_unit_is_mm` | Show position deviations in millimeters instead of meters | `False` |
| `directed_ate` | Split position deviations into along-track, horizontal cross-track and vertical cross-track instead of x, y and z (see below) | `False` |
| `ate_remove_above` | Exclude position deviations above this value (m) from the plots (`0` = keep all) | `0.0` |
| `hist_percentile` | Show histogram data up to this percentile of the absolute values; useful for data with outliers | `1.0` |
| `colorbar_max_std` | Clip color-coded values to the mean ± this many standard deviations, so that outliers don't dominate the color scale (`0` = no clipping) | `3.0` |
| `smoothing_window` | Moving-average window applied to color-coded values (`1` = no smoothing) | `1` |
| `use_relative_timestamps` | Show time axes in seconds relative to the first timestamp instead of absolute (UTC) time | `False` |
| `line_width` | Line width (pixels) | `5` |
| `marker_size` | Marker size of color-coded trajectory maps (pixels) | `5.0` |
| `font_size` | Font size of axis labels and titles (points) | `10` |
| `export_format` | File format of exported plots: `png`, `jpg` or `svg` | `png` |
| `export_width` | Width of exported raster images (pixels) | `1600` |

Plot settings are saved with a GUI session (`plot_settings.json`) and can be loaded at startup with
`trajectopy --plot-settings plot_settings.json`.

### ATE Frame Definition (regarding `directed_ate`)

The ATE can be split into along-, horizontal-cross- and vertical-cross-track directions by setting `directed_ate` to `True`. The along-track direction is defined as positive in the direction of travel. The horizontal cross-track direction is defined as positive to the right of the along-track direction. The vertical cross-track direction is defined as positive upwards. The following image illustrates the frame definition.

<img src="https://raw.githubusercontent.com/gereon-t/trajectopy/main/.images/along_cross_frames.png" alt="ate_frames" width="400"/>
