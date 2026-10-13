## Processing Settings

The processing settings allow you to configure various processing steps that can be applied to trajectories before or during analysis. The available processing options are summarized in the table below:

| Setting | Description |
| :--- | :--- |
| `alignment` | Settings for trajectory alignment (Similarity transformation (helmert), time shift, lever arm, etc.) |
| `matching` | Settings for matching poses between two trajectories |
| `relative_comparison` | Settings for relative trajectory comparison (RPE) |
| `approximation` | Settings for trajectory approximation and smoothing |
| `sorting` | Settings for spatially sorting trajectories |

### Alignment Settings

#### Alignment Preprocessing Settings

- `min_speed` (float): Only poses with a speed above this threshold are considered for alignment (meters/second).
- `time_start` (float): Only poses with a timestamp above this threshold are considered for alignment. The timestamp is given in seconds and is relative to the first common timestamp of both matched trajectories.
- `time_end` (float): Only poses with a timestamp below this threshold are considered for alignment. The timestamp is given in seconds and is relative to the first common timestamp of both matched trajectories.

#### Alignment Estimation Settings

- `translation_x` (boolean): Enable or disable x-translation of the similarity transformation.
- `translation_y` (boolean): Enable or disable y-translation of the similarity transformation.
- `translation_z` (boolean): Enable or disable z-translation of the similarity transformation.
- `rotation_x` (boolean): Enable or disable rotation around the X-axis of the similarity transformation.
- `rotation_y` (boolean): Enable or disable rotation around the Y-axis of the similarity transformation.
- `rotation_z` (boolean): Enable or disable rotation around the Z-axis of the similarity transformation.
- `scale` (boolean): Enable or disable scaling of the similarity transformation.
- `time_shift` (boolean): Enable or disable the estimation of time shift.
- `leverarm_x` (boolean): Enable or disable estimation of lever arm in the X-axis.
- `leverarm_y` (boolean): Enable or disable estimation of lever arm in the Y-axis.
- `leverarm_z` (boolean): Enable or disable estimation of lever arm in the Z-axis.
- `sensor_rotation` (boolean): Enable or disable computation of sensor rotation offsets. Independent of the least squares adjustment, constant offsets between the roll, pitch and yaw angles of both trajectories are computed.

#### Alignment Stochastics Settings

- `std_xy_from` (float): Standard deviation of XY source position components in meters.
- `std_z_from` (float): Standard deviation of Z source position component in meters.
- `std_xy_to` (float): Standard deviation of XY target position components in meters.
- `std_z_to` (float): Standard deviation of Z target position component in meters.
- `std_roll_pitch` (float): Standard deviation of roll and pitch in radians.
- `std_yaw` (float): Standard deviation of yaw in radians.
- `std_speed_to` (float): Standard deviation of platform speed in (meters per second).
- `error_probability` (float): Probability of error used for stochastic testing.
- `variance_estimation` (boolean): Enable or disable the estimation of the variance factor for a-posteriori variance computation.

#### Alignment Threshold Settings

Usually, these settings can be left at their default values.

- `metric_threshold` (float): Iteration threshold for the least squares adjustment regarding the metric parameters.
- `time_threshold` (float): Iteration threshold in seconds for the least squares adjustment regarding the time shift parameter.


### Matching Settings

- `method` (`MatchingMethod`): The method used for trajectory matching. Choices: `MatchingMethod.NEAREST_SPATIAL`, `MatchingMethod.NEAREST_TEMPORAL`, `MatchingMethod.INTERPOLATION`, `MatchingMethod.NEAREST_SPATIAL_INTERPOLATED`. The methods are described below.
- `max_time_diff` (float): Maximum allowed time difference in seconds when matching two trajectories using their timestamps.
- `max_distance` (float): Maximum allowed distance in meters between matched positions during spatial matching.
- `max_gap_size` (float): Maximum allowed gap size in seconds within a trajectory during matching.
- `k_nearest` (integer): The number of nearest neighbors to consider during spatial interpolation matching.

### Matching Methods

#### Nearest Spatial

This method matches two trajectories by finding the nearest pose in the target trajectory for each pose in the source trajectory. The distance between two poses is computed using the Euclidean distance between their positions.

#### Nearest Temporal

This method matches two trajectories using their timestamps by finding the nearest timestamp in the target trajectory for each timestamp in the source trajectory.

#### Interpolation

This method matches two trajectories by interpolating the timestamps of one trajectory to the timestamps of the other trajectory. The interpolation is linear for both positions and rotations (SLERP).

#### Nearest Spatial Interpolated

This method matches both trajectories spatially by requesting the nearest k positions from the reference trajectory for each pose in the test trajectory. Then, an interpolation is performed using a 3d line fit of the k nearest positions. After this operation, both trajectories will have the length of the test trajectory. The matched rotations are computed by averaging the quaternions of the k nearest neighbors using the chordal L2 mean.


### Relative Comparison Settings

- `pair_min_distance` (float): Minimum pose pair distance to be considered during RPE (Relative Pose Error) computation.

- `pair_max_distance` (float): Maximum pose pair distance to be considered during RPE computation.

- `pair_distance_step` (float): Step in which the pose pair distance increases.

- `pair_distance_unit` (`Unit`): Unit of the pose pair distance. Choices: `Unit.METER`, `Unit.SECOND`.

- `use_all_pose_pairs` (boolean): If enabled, overlapping pose pairs will be used for relative metrics calculation.


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

This metric does not require both trajectories to be aligned. Units are m/m: %, deg/m for distance based comparison and m/s, deg/s for time-based comparison. 

Example:

- Minimum pose distance: 100
- Maximum pose distance: 800
- Distance step: 100
- Distance unit: Meter

Results in pose distances: [100 m, 200 m, 300 m, 400 m, 500 m, 600 m, 700 m, 800 m]

Furthermore, the user can choose to either use consecutive pose pairs (non-overlapping) or all posible pairs (overlapping).

### Approximation Settings

Trajectopy currently supports piece-wise cubic approximation for position components and moving average for rotation components. Piece-wise cubic means that the trajectory is divided into intervals of a specified size, and within each interval, a cubic polynomial is fitted to the position data. The fitted polynomial is then used to approximate the position values within that interval.

- `position_interval_size` (float): Size of the position intervals in meters for cubic approximation. Default value is 0.15 meters.
- `position_min_observations` (int): Minimum number of observations required in each position interval for cubic approximation. Default value is 25.
- `rotation_window_size` (float): Size of the rotation smoothing window in meters for rotation approximation (not cubic!). Default value is 0.15 meters.

### Sorting Settings

Sorting involves two steps:

1. Smoothing the trajectory using Moving Least Squares (MLS). The k nearest neighbors of each point are used to fit a 3D line, and the point is projected onto this line. This step reduces noise and outliers in the trajectory.
2. Shortest path approximation by constructing a minimum spanning tree (MST) based on the smoothed points.

For sorting, the following settings can be adjusted:

- `voxel_size` (float): Moving Least Squares (MLS) setting. Instead of querying the raw input points, a voxel grid is created and the centroids of the occupied voxels are used for nearest neighbor searches. This setting specifies the size of the voxel grid for downsampling. Default value is 0.05 meters.
- `movement_threshold` (float): Moving Least Squares (MLS) setting. This threshold defines the maximum allowed movement of points between two iterations of the MLS algorithm. If all points move less than this threshold, the MLS algorithm terminates. Default value is 0.005 meters.
- `k_nearest` (int): Number of nearest voxels to consider during Moving Least Squares (MLS) smoothing. Default value is 4.


## Plot Settings

All plots, in the GUI ("Plot Settings" in the menu bar) and in `trajectopy.visualization`, use the
same `PlotSettings`:

- `ate_unit_is_mm` (bool): Show position deviations in millimeters instead of meters. Default value is False.
- `directed_ate` (bool): Show position deviations split into along-track, horizontal cross-track and vertical cross-track instead of x, y and z. Default value is False.
- `ate_remove_above` (float): Exclude position deviations above this value in meters from the plots (0.0 = keep all). Default value is 0.0.
- `hist_percentile` (float): Show histogram data up to this percentile of the absolute values. Useful for data with outliers. Default value is 1.0.
- `colorbar_max_std` (float): Clip color-coded values to the mean ± this many standard deviations, so that outliers don't dominate the color scale (0.0 = no clipping). Default value is 3.0.
- `smoothing_window` (int): Moving-average window applied to color-coded values (1 = no smoothing). Default value is 1.
- `use_relative_timestamps` (bool): Show time axes in seconds relative to the first timestamp instead of absolute (UTC) time. Default value is False.
- `line_width` (float): Line width in pixels. Default value is 1.5.
- `marker_size` (float): Marker size in pixels of color-coded trajectory maps. Default value is 4.
- `font_size` (int): Font size in points of axis labels and titles. Default value is 10.
- `export_format` (str): File format of exported plots ("png", "jpg" or "svg"). Default value is "png".
- `export_width` (int): Width in pixels of exported raster images. Default value is 1600.
- `background_map` (bool): Show a map below the top views of georeferenced trajectories (see [Maps](user_guide.md#maps)). Default value is False.
- `map_style` (str): Map shown by `background_map`: "osm" (OpenStreetMap) or "satellite" (Esri World Imagery). Default value is "osm".

Plot settings are saved with a GUI session (`plot_settings.json`) and can be loaded at startup with
`trajectopy --plot-settings plot_settings.json`.

### ATE Frame Definition (regarding `directed_ate`)

The ATE can be split into along-, horizontal-cross- and vertical-cross-track directions by setting `directed_ate` to `True`. The along-track direction is defined as positive in the direction of travel. The horizontal cross-track direction is defined as positive to the right of the along-track direction. The vertical cross-track direction is defined as positive upwards. The following image illustrates the frame definition.

<img src="https://raw.githubusercontent.com/gereon-t/trajectopy/main/.images/along_cross_frames.png" alt="ate_frames" width="400"/>
