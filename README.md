<div align="center">
    <h1>Trajectopy - Trajectory Evaluation in Python</h1>
    <a href="https://github.com/gereon-t/trajectopy/releases"><img src="https://img.shields.io/github/v/release/gereon-t/trajectopy?label=version" /></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8.2+-blue.svg" /></a>
    <a href="https://github.com/gereon-t/trajectopy/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gereon-t/trajectopy" /></a>
    <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <br />
    <a href="https://github.com/gereon-t/trajectopy"><img src="https://img.shields.io/badge/Windows-0078D6?st&logo=windows&logoColor=white" /></a>
    <a href="https://github.com/gereon-t/trajectopy"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://github.com/gereon-t/trajectopy"><img src="https://img.shields.io/badge/mac%20os-000000?&logo=apple&logoColor=white" /></a>
    <br />
    <a href="https://www.gug.uni-bonn.de/en/"><img src=.images/uni-bonn.svg height="50"/></a>
    <a href="https://www.gug.uni-bonn.de/en/"><img src=.images/igg.png height="50"/></a>

#### Trajectopy is a Python package with an intuitive graphical user interface for empirical trajectory evaluation.

This repository contains the source code of the trajectopy front end PyQt6 GUI. The back-end is part of the trajectopy-core package, see  [https://github.com/gereon-t/trajectopy-core](https://github.com/gereon-t/trajectopy-core).

<p align="center">
  <img style="border-radius: 10px;" src=.images/trajectopy_gif_low_quality.gif>
</p>

</div>


## Key Features

Trajectopy offers a range of features, including:

- __Interactive GUI__: A user-friendly interface that enables seamless interaction with your trajectory data, making it easy to visualize, align, and compare trajectories.
- __Alignment__: An advanced trajectory alignment algorithm that can be tailored to the specific application and supports a similarity transformation, a leverarm and a time shift estimation.
- __Comparison__: Absolute and relative trajectory comparison metrics (__ATE and RPE__) that can be computed using various pose-matching methods.
- __Data Import/Export__: Support for importing and exporting data, ensuring compatibility with your existing workflows.
- __Customizable HTML-based Visualization__: Powered by [Plotly](https://plotly.com/), trajectopy offers a range of interactive plots that can be customized to your needs. ([Demo](https://htmlpreview.github.io/?https://github.com/gereon-t/trajectopy-core/blob/main/example_data/report.html))


## Table of Contents
- [Installation](#installation)
- [Command Line Options](#command-line-options)
- [Importing Trajectories](#importing-trajectories)
- [Processing Settings](#processing-settings)


## Installation

```console
pip install trajectopy
```

Run
  
```console
trajectopy
```

## Command Line Options
```console	
usage: trajectopy [-h] [--version] [--single_thread] [--report_settings REPORT_SETTINGS] [--report_path REPORT_PATH]
```
  
```console
options:
    -h, --help            show this help message and exit
    --version, -v
    --single_thread       Disable multithreading
    --report_settings REPORT_SETTINGS, -s REPORT_SETTINGS
                            Path to JSON report settings file
                            that will override the default settings.
    --report_path REPORT_PATH, -o REPORT_PATH
                            Output directory for all reports of one session.
```
Since version 2.0.1, trajectopy allows users to customize the report output path and settings. By default, reports are saved in the "reports" folder of the current working directory.

The report settings file must include all available settings. You can find a sample file [here](https://github.com/gereon-t/trajectopy/blob/main/example_data/custom.json).

Example:
```console
trajectopy -s custom.json -o ./custom_report_location
```


## Importing Trajectories

Trajectories can be imported using the "Add" button below the trajectory table or by dragging files into the area of the trajectory table.
Trajectory files must be ASCII files with a csv-like layout, by default, trajectopy filters for the ".traj" extension. The default column structure that can be read without any configuration is the following:

| time | position x | position y | position z | quaternion x | quaternion y | quaternion z | quaternion w |
|---|---|---|---|---|---|---|---|


Columns are expected to be separated by commas by default.

It is recommended to provide a header at the beginning of the trajectory file. Header entries always begin with a "#".
Below you can find a table of all allowed header entries and their meaning.

| Header | Description  |
|---|---|
| #name | The name provided here is displayed in the table view and in plots of the trajectory |
| #epsg | [EPSG Code](https://epsg.io/) of the datum of the input positions. Required, if geodetic datum transformations are desired. Default: 0, meaning local coordinates without any known geodetic datum |
| #fields | Describes the columns of the ASCII trajectory file. Separated with commas. <table>  <thead>  <th>field name</th>  <th>Meaning</th>  </tr>  </thead>  <tbody>  <tr>  <td>t</td>  <td>time</td>  </tr>  <tr>  <td>l</td>  <td>arc lengths in meters</td>  </tr>  <tr>  <td>px</td>  <td>position x / lat (degrees only)</td>  </tr>  <tr>  <td>py</td>  <td>position y / lon (degrees only) </td>  </tr>  <tr>  <td>pz</td>  <td>position z</td>  </tr> <tr>  <td>qx</td>  <td>quaternion x</td>  </tr> <tr>  <td>qy</td>  <td>quaternion y</td>  </tr> <tr>  <td>qz</td>  <td>quaternion z</td>  </tr> <tr>  <td>qw</td>  <td>quaternion w</td>  </tr> </tr> <tr>  <td>ex</td>  <td>euler angle x</td>  </tr> </tr> <tr>  <td>ey</td>  <td>euler angle y</td>  </tr> </tr> <tr>  <td>ez</td>  <td>euler angle z</td>  </tr> </tr> <tr>  <td>vx</td>  <td>speed x</td>  </tr> </tr> <tr>  <td>vy</td>  <td>speed y</td>  </tr> </tr> <tr>  <td>vz</td>  <td>speed z</td>  </tr> </tr> </tbody>  </table> Example: "#fields t,px,py,pz" Note: The only column that is allowed to appear multiple times is the "t" column. |
| #delimiter | Delimiter used to separate the columns within the file. Default: "," |
| #nframe | Definition of the navigation-frame the orientations of the trajectory refer to. Choices: "enu": East North Up or "ned": North East Down. Default: "enu" |
| #rot_unit | Unit of the orientations. Choices: "deg": Degree, "rad": Radians. Default: "rad" |
| #time_format | Format of the timestamps / dates. Choices: "unix": Unix timestamps (since 01-01-1970), "datetime": Human readable date-times. Default: "unix" |
| #time_offset | Offset in seconds that is applied to the imported timestamps. Default: 0.0 |
| #datetime_format | Format of the datetimes. Only relevant if "time_format" is "datetime". Default: "%Y-%m-%d %H:%M:%S.%f" |
| #datetime_timezone | Time zone of the timestamps. During import, all timestamps are converted to UTC considering the input time zone. Choices: [Time zone](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones) or "GPS" |
| #sorting | Sorting of the input data. Choices: "chrono": Chronologically sorted data (usually the case), "spatial": Spatially sorted data, i.e. along the arc length. Default: "chrono" |

**New since 0.9.2**: Experimental ROS bag support for geometry_msgs/msg/PoseStamped messages. Files must have a ".bag" extension. Poses must have positions and orientations. One file can contain multiple trajectories published under different topics.


## Processing Settings

Trajectopy offers a range of processing options that can be applied to the imported trajectories. These options are:

| Option | Description |
|---|---|
| Alignment | Alignment of two trajectories using least squares adjustment. The implemented approach can handle a similarity transformation (translation, rotation, scale), a lever arm (3d vector), and a time shift (scalar). Each parameter can be included or exluded from the adjustment depending on the individual sensor modalities using the `AlignmentSettings`. In addition, preprocessing steps and stochastics can also be configured. |
| Matching | Matching of two trajectories to establish pose-to-pose correspondencies. After matching both trajectories will have the same number of poses. You can choose from different matching methods in the `MatchingSettings`. |
| Comparison | Comparison of two trajectories using absolute (ATE) and relative (RPE) metrics. The relative comparison can be configured using the `RelativeComparisonSettings`. |

## Alignment Settings

### Preprocessing Settings

- `min_speed` (float): Only poses with a speed above this threshold are considered for alignment.
- `time_start` (float): Only poses with a timestamp above this threshold are considered for alignment. The timestamp is given in seconds and is relative to the first common timestamp of both matched trajectories.
- `time_end` (float): Only poses with a timestamp below this threshold are considered for alignment. The timestamp is given in seconds and is relative to the first common timestamp of both matched trajectories.

### Estimation Settings

- `trans_x` (boolean): Enable or disable x-translation of the similarity transformation.
- `trans_y` (boolean): Enable or disable y-translation of the similarity transformation.
- `trans_z` (boolean): Enable or disable z-translation of the similarity transformation.
- `rot_x` (boolean): Enable or disable rotation around the X-axis of the similarity transformation.
- `rot_y` (boolean): Enable or disable rotation around the Y-axis of the similarity transformation.
- `rot_z` (boolean): Enable or disable rotation around the Z-axis of the similarity transformation.
- `scale` (boolean): Enable or disable scaling of the similarity transformation.
- `time_shift` (boolean): Enable or disable the estimation of time shift.
- `use_x_speed` (boolean): Enable or disable the use of X-axis speed for time shift estimation.
- `use_y_speed` (boolean): Enable or disable the use of Y-axis speed for time shift estimation.
- `use_z_speed` (boolean): Enable or disable the use of Z-axis speed for time shift estimation.
- `lever_x` (boolean): Enable or disable estimation of lever arm in the X-axis.
- `lever_y` (boolean): Enable or disable estimation of lever arm in the Y-axis.
- `lever_z` (boolean): Enable or disable estimation of lever arm in the Z-axis.
- `sensor_rotation` (boolean): Enable or disable estimation of sensor rotation. Independent of the least squares adjustment, a constant rotational offset can be computed between the rotations of both trajectories after the alignment.
- `auto_update` (boolean): If set to `True`, the estimation settings are automatically updated and parameters are disabled if they are not significant (experimental).

### Stochastics Settings

- `std_xy_from` (float): Standard deviation of XY source position components in meters.
- `std_z_from` (float): Standard deviation of Z source position component in meters.
- `std_xy_to` (float): Standard deviation of XY target position components in meters.
- `std_z_to` (float): Standard deviation of Z target position component in meters.
- `std_roll_pitch` (float): Standard deviation of roll and pitch in radians.
- `std_yaw` (float): Standard deviation of yaw in radians.
- `std_speed_to` (float): Standard deviation of platform speed in (meters per second).
- `error_probability` (float): Probability of error used for stochastic testing.

### Threshold Settings

Usually, these settings can be left at their default values.

- `metric_threshold` (float): Iteration threshold for the least squares adjustment regarding the metric parameters.
- `time_threshold` (float): Iteration threshold for the least squares adjustment regarding the time shift parameter.


## Matching Settings

- `method` (`MatchingMethod`): The method used for trajectory matching. Choices: `MatchingMethod.NEAREST_SPATIAL`, `MatchingMethod.NEAREST_TEMPORAL`, `MatchingMethod.INTERPOLATION`, `MatchingMethod.NEAREST_SPATIAL_INTERPOLATED`. The methods are described below.
- `max_time_diff` (float): Maximum allowed time difference when matching two trajectories using their timestamps.
- `max_distance` (float): Maximum allowed distance between matched positions during spatial matching.
- `k_nearest` (integer): The number of nearest neighbors to consider during spatial interpolation matching.

### Matching Methods

#### Nearest Spatial

This method matches two trajectories by finding the nearest pose in the target trajectory for each pose in the source trajectory. The distance between two poses is computed using the Euclidean distance between their positions.

#### Nearest Temporal

This method matches two trajectories using their timestamps by finding the nearest timestamp in the target trajectory for each timestamp in the source trajectory.

#### Interpolation

This method matches two trajectories by interpolating the timestamps of one trajectory to the timestamps of the other trajectory. The interpolation is linear for both positions and rotations (SLERP).

#### Nearest Spatial Interpolated

This method matches both trajectories spatially by requesting the nearest k positions from the reference trajectory for each pose in the test trajectory. Then, an interpolation is performed using a 3d line fit of the k nearest positions. After this operation, both trajectories will have the length of the test trajectory. This method does not support rotation matching.


## Relative Comparison Settings

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

### Report Settings

#### Visualization Settings

- `scatter_max_std` (float): The upper colorbar limit is set to the mean plus this value times the standard deviation of the data. This is useful to prevent outliers from dominating the colorbar. Default value is 4.0.
- `ate_unit_is_mm` (bool): Indicates whether the unit of Absolute Trajectory Error (ATE) is millimeters. Default value is False.
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
- `scatter_detailed` (bool): Indicates whether to show scatter plots for each degree of freedom. Default value is False.

#### Position Units and Names

- `pos_x_name` (string): Name for the X-axis position. Default: "x".
- `pos_y_name` (string): Name for the Y-axis position. Default: "y".
- `pos_z_name` (string): Name for the Z-axis position. Default: "z".
- `pos_x_unit` (string): Unit for the X-axis position, Default: "m".
- `pos_y_unit` (string): Unit for the Y-axis position, Default: "m".
- `pos_z_unit` (string): Unit for the Z-axis position, Default: "m".

#### Rotation Units and Names

- `rot_x_name` (string): Name for the roll rotation. Default: "roll".
- `rot_y_name` (string): Name for the pitch rotation. Default: "pitch".
- `rot_z_name` (string): Name for the yaw rotation. Default: "yaw".
- `rot_unit` (string): Unit symbol for rotation. Default: "°".

#### Export Settings

- `single_plot_export` (ExportSettings): The export settings for single plots. Default value is an instance of ExportSettings with width=800 and height=450.
- `two_subplots_export` (ExportSettings): The export settings for two subplots. Default value is an instance of ExportSettings with width=800 and height=540.
- `three_subplots_export` (ExportSettings): The export settings for three subplots. Default value is an instance of ExportSettings with width=800 and height=750.
- `single_plot_height` (int): The height of a single plot. Default value is 450.
- `two_subplots_height` (int): The height of two subplots. Default value is 540.
- `three_subplots_height` (int): The height of three subplots. Default value is 750.


#### Export Settings

- `format` (string): The export format. Choices: "png", "svg", "jpeg", "webp". Default: "png".
- `height` (integer): The export height in pixels. Default: 500.
- `width` (integer): The export width in pixels. Default: 800.
- `scale` (integer): The export scale. Default: 6.