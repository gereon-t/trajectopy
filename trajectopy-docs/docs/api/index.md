# API Reference

This reference describes the public API of trajectopy: the names of the namespaces `trajectopy`, `trajectopy.io`, `trajectopy.visualization` and `trajectopy.exceptions`. Import them from these namespaces; the [API policy](../api_policy.md) describes the conventions of the API and which changes to expect between versions.

```python
import trajectopy as tpy
from trajectopy import io, visualization as vis
```

## [Trajectories](trajectory.md)

| Name | Description |
| --- | --- |
| [`Trajectory`](trajectory.md#trajectopy.Trajectory) | Class representing a trajectory containing synchronized position, orientation, and time data. |
| [`Pose`](trajectory.md#trajectopy.Pose) | A single pose of a trajectory, as returned by indexing a trajectory with an integer or iterating over it. |
| [`Positions`](trajectory.md#trajectopy.Positions) | Class representing (georeferenced) positions. |

## [Processing](processing.md)

| Name | Description |
| --- | --- |
| [`estimate_alignment`](processing.md#trajectopy.estimate_alignment) | Estimates the alignment between two trajectories. |
| [`apply_alignment`](processing.md#trajectopy.apply_alignment) | Transforms trajectory using alignment parameters. |
| [`align`](processing.md#trajectopy.align) | Aligns the trajectory with another trajectory. |
| [`adopt_first_pose`](processing.md#trajectopy.adopt_first_pose) | Transform trajectory so that the first pose is identical in both. |
| [`match_trajectories`](processing.md#trajectopy.match_trajectories) | Matches two trajectories using the specified method. |
| [`interpolate`](processing.md#trajectopy.interpolate) | Interpolates a trajectory to specified timestamps using the given method. |
| [`approximate`](processing.md#trajectopy.approximate) | Approximate the trajectory using piecewise cubic polynomials based on the provided settings. |
| [`sort_spatially`](processing.md#trajectopy.sort_spatially) | Sorts the trajectory spatially. This is only useful for trajectories that describe a closed loop without intersections. |
| [`divide_into_laps`](processing.md#trajectopy.divide_into_laps) | Divides the trajectory into laps. This is only useful for trajectories that describe a closed loop without intersections. |
| [`merge_trajectories`](processing.md#trajectopy.merge_trajectories) | Merges a list of trajectories into one trajectory. |
| [`average_trajectories`](processing.md#trajectopy.average_trajectories) | Averages a list of trajectories into one trajectory. |

## [Evaluation](evaluation.md)

| Name | Description |
| --- | --- |
| [`ate`](evaluation.md#trajectopy.ate) | Computes the absolute trajectory error (ATE) between two trajectories. |
| [`rpe`](evaluation.md#trajectopy.rpe) | Computes the relative pose error (RPE) between two trajectories. |
| [`summary_table`](evaluation.md#trajectopy.summary_table) | ATE and RPE summaries side by side, joined by the result names (outer join, so that results with only an ATE or only an RPE keep their row). |
| [`format_table`](evaluation.md#trajectopy.format_table) | Formats a summary table as CSV, Markdown or LaTeX (without additional dependencies). |
| [`write_report`](evaluation.md#trajectopy.write_report) | Writes a self-contained HTML report of an evaluation. |

## [Results](results.md)

| Name | Description |
| --- | --- |
| [`ATEResult`](results.md#trajectopy.ATEResult) | This class represents a set of absolute trajectory deviations |
| [`AbsoluteTrajectoryDeviations`](results.md#trajectopy.AbsoluteTrajectoryDeviations) | Deviations of every pose of an ATE result (`ATEResult.abs_dev`). |
| [`RPEResult`](results.md#trajectopy.RPEResult) | This class represents a set of relative trajectory deviations |
| [`RelativeTrajectoryDeviations`](results.md#trajectopy.RelativeTrajectoryDeviations) | Deviations of the pose pairs of an RPE result (`RPEResult.rpe_dev`). |
| [`DeviationCorrelation`](results.md#trajectopy.DeviationCorrelation) | Temporal correlation of the deviations of an ATE result. |
| [`ComponentCorrelation`](results.md#trajectopy.ComponentCorrelation) | Temporal correlation of one deviation component. |
| [`AlignmentResult`](results.md#trajectopy.AlignmentResult) | Class to store the result of an alignment. |
| [`AlignmentParameters`](results.md#trajectopy.AlignmentParameters) | Parameters of the spatio-temporal alignment of two trajectories. |
| [`HelmertTransformation`](results.md#trajectopy.HelmertTransformation) | Parameters of a 7-parameter similarity transformation (`AlignmentParameters.helmert`). |
| [`Leverarm`](results.md#trajectopy.Leverarm) | Lever arm between two trajectories (`AlignmentParameters.leverarm`). |
| [`SensorRotationParameters`](results.md#trajectopy.SensorRotationParameters) | Constant rotation between the body frames of two trajectories (mounting rotation). |
| [`Parameter`](results.md#trajectopy.Parameter) | Class representing a stochastic parameter with variance. |

## [Settings](settings.md)

| Name | Description |
| --- | --- |
| [`Settings`](settings.md#trajectopy.Settings) | Base class for all settings dataclasses. |
| [`ProcessingSettings`](settings.md#trajectopy.ProcessingSettings) | Complete configuration for trajectory processing. |
| [`AlignmentSettings`](settings.md#trajectopy.AlignmentSettings) | Complete configuration for trajectory alignment. |
| [`AlignmentPreprocessing`](settings.md#trajectopy.AlignmentPreprocessing) | Configuration for alignment preprocessing filters. |
| [`AlignmentEstimationSettings`](settings.md#trajectopy.AlignmentEstimationSettings) | Configuration for parameters to estimate during trajectory alignment. |
| [`AlignmentStochastics`](settings.md#trajectopy.AlignmentStochastics) | Configuration for alignment stochastic model (observation uncertainties). |
| [`MatchingSettings`](settings.md#trajectopy.MatchingSettings) | Configuration for trajectory matching. |
| [`RelativeComparisonSettings`](settings.md#trajectopy.RelativeComparisonSettings) | Configuration for relative trajectory comparison (RPE - Relative Pose Error). |
| [`ApproximationSettings`](settings.md#trajectopy.ApproximationSettings) | Configuration for trajectory approximation (smoothing). |
| [`SortingSettings`](settings.md#trajectopy.SortingSettings) | Configuration for trajectory sorting and downsampling using Moving Least Squares. |
| [`PlotSettings`](settings.md#trajectopy.PlotSettings) | Configuration of all plots (GUI and :mod:`trajectopy.visualization`). |
| [`MatchingMethod`](settings.md#trajectopy.MatchingMethod) | Method for matching poses between two trajectories. |
| [`InterpolationMethod`](settings.md#trajectopy.InterpolationMethod) | Method for interpolation. |
| [`PairDistanceUnit`](settings.md#trajectopy.PairDistanceUnit) | Unit of measurement for pose pair distances in relative comparison. |
| [`PositionApproximationMethod`](settings.md#trajectopy.PositionApproximationMethod) | Method for position approximation. |
| [`RotationApproximationMethod`](settings.md#trajectopy.RotationApproximationMethod) | Method for rotation approximation. |
| [`Sorting`](settings.md#trajectopy.Sorting) | Order of the poses of a trajectory: chronological or along the path (spatially sorted). |

## [Input and Output](io.md)

| Name | Description |
| --- | --- |
| [`io.read_trajectory`](io.md#trajectopy.io.read_trajectory) | Reads a trajectory file in any supported format. |
| [`io.write_trajectory`](io.md#trajectopy.io.write_trajectory) | Writes a trajectory in one of the supported formats. |
| [`io.detect_format`](io.md#trajectopy.io.detect_format) | Recognizes the format of a trajectory file from its first lines. |
| [`io.TrajectoryFormat`](io.md#trajectopy.io.TrajectoryFormat) | File formats of trajectories. |
| [`io.read_tum`](io.md#trajectopy.io.read_tum) | Reads a trajectory in the TUM format (``timestamp tx ty tz qx qy qz qw``). |
| [`io.read_kitti`](io.md#trajectopy.io.read_kitti) | Reads a trajectory in the KITTI odometry format (one row-major 3x4 pose matrix per line). |
| [`io.read_euroc`](io.md#trajectopy.io.read_euroc) | Reads a trajectory in the EuRoC ground truth format. |
| [`io.read_trajectory_string`](io.md#trajectopy.io.read_trajectory_string) | Reads a trajectory from the string content of a trajectopy ASCII file. |
| [`io.trajectory_to_string`](io.md#trajectopy.io.trajectory_to_string) | Returns a trajectory in the trajectopy ASCII format, as :func:`write_trajectory` writes it. |
| [`io.list_topics`](io.md#trajectopy.io.list_topics) | The topics of a bag with a supported message type, mapped to their message type. |
| [`io.trajectories_from_rosbag`](io.md#trajectopy.io.trajectories_from_rosbag) | Creates trajectories from a ROS1 or ROS2 bag. |
| [`io.read_ate_result`](io.md#trajectopy.io.read_ate_result) | Reads an ATE result written by :func:`write_ate_result`. |
| [`io.write_ate_result`](io.md#trajectopy.io.write_ate_result) | Writes an ATE result as CSV file with a metadata header, which :func:`read_ate_result` and the GUI can read. |
| [`io.read_rpe_result`](io.md#trajectopy.io.read_rpe_result) | Reads an RPE result written by :func:`write_rpe_result`. |
| [`io.write_rpe_result`](io.md#trajectopy.io.write_rpe_result) | Writes an RPE result as CSV file with a metadata header, which :func:`read_rpe_result` and the GUI can read. |
| [`io.read_alignment_result`](io.md#trajectopy.io.read_alignment_result) | Reads an alignment result written by :func:`write_alignment_result`. |
| [`io.write_alignment_result`](io.md#trajectopy.io.write_alignment_result) | Writes an alignment result to a file that :func:`read_alignment_result` and the GUI can read. |
| [`io.read_alignment_parameters`](io.md#trajectopy.io.read_alignment_parameters) | Reads alignment parameters written by :func:`write_alignment_parameters`. |
| [`io.write_alignment_parameters`](io.md#trajectopy.io.write_alignment_parameters) | Writes alignment parameters together with their covariance matrix to a CSV file. |
| [`io.read_sensor_rotation_parameters`](io.md#trajectopy.io.read_sensor_rotation_parameters) | Reads the sensor rotation parameters of an alignment result file. |
| [`io.write_sensor_rotation_parameters`](io.md#trajectopy.io.write_sensor_rotation_parameters) | Appends sensor rotation parameters to a file, e.g. to an alignment result written before. |
| [`io.write_kml`](io.md#trajectopy.io.write_kml) | Exports a trajectory to a Google Earth KML file. |
| [`io.write_ate_kml`](io.md#trajectopy.io.write_ate_kml) | Exports an ATE result to a Google Earth KML file, colored by one of its deviation fields. |

## [Visualization](visualization.md)

| Name | Description |
| --- | --- |
| [`visualization.trajectory_figures`](visualization.md#trajectopy.visualization.trajectory_figures) | All figures describing the given trajectories. |
| [`visualization.positions_figure`](visualization.md#trajectopy.visualization.positions_figure) | Top view (x/y) of the trajectories. |
| [`visualization.xyz_figure`](visualization.md#trajectopy.visualization.xyz_figure) | Coordinates of the trajectories over time or path length. |
| [`visualization.rpy_figure`](visualization.md#trajectopy.visualization.rpy_figure) | Roll, pitch and yaw of the trajectories over time or path length (None without orientations). |
| [`visualization.trajectory_map_figure`](visualization.md#trajectopy.visualization.trajectory_map_figure) | Top view of each trajectory colored by velocity or height. |
| [`visualization.deviation_figures`](visualization.md#trajectopy.visualization.deviation_figures) | All figures describing absolute (ATE) and relative (RPE) evaluation results. |
| [`visualization.ate_figure`](visualization.md#trajectopy.visualization.ate_figure) | Position (and rotation) deviations over time or path length. |
| [`visualization.ate_components_figure`](visualization.md#trajectopy.visualization.ate_components_figure) | Deviations of the single degrees of freedom over time or path length. |
| [`visualization.ate_histogram_figure`](visualization.md#trajectopy.visualization.ate_histogram_figure) | Histograms of the deviations of the single degrees of freedom. |
| [`visualization.ate_statistics_figure`](visualization.md#trajectopy.visualization.ate_statistics_figure) | Min, max, mean, median, RMS and standard deviation of the deviations as bar charts. |
| [`visualization.ate_boxplot_figure`](visualization.md#trajectopy.visualization.ate_boxplot_figure) | Box plots of the position (and rotation) deviations, one box per result, to compare several results. |
| [`visualization.ate_distribution_figure`](visualization.md#trajectopy.visualization.ate_distribution_figure) | Empirical cumulative distribution of the position (and rotation) deviations. |
| [`visualization.ate_correlation_figure`](visualization.md#trajectopy.visualization.ate_correlation_figure) | Temporal correlation of the deviations: autocorrelation over the time lag (with the level that defines the correlation time of each component, 1/e of its correlated share, and the correlation time as a point) and the standard deviation in consecutive time windows. |
| [`visualization.ate_map_figure`](visualization.md#trajectopy.visualization.ate_map_figure) | Top view of the evaluated trajectories colored by their deviations. |
| [`visualization.ate_speed_figure`](visualization.md#trajectopy.visualization.ate_speed_figure) | Position deviations over the speed of the evaluated trajectory, to reveal synchronization errors. |
| [`visualization.rpe_figures`](visualization.md#trajectopy.visualization.rpe_figures) | Relative pose errors over the pair distance, one figure per pair distance unit. |
| [`visualization.alignment_figures`](visualization.md#trajectopy.visualization.alignment_figures) | Covariance and correlation of estimated alignment parameters. |
| [`visualization.parameter_matrix_figure`](visualization.md#trajectopy.visualization.parameter_matrix_figure) | Covariance or correlation matrix of estimated alignment parameters. |
| [`visualization.smoothed_speed`](visualization.md#trajectopy.visualization.smoothed_speed) | Speed of ``trajectory`` in m/s from its velocity vectors averaged over ``window`` seconds. |
| [`visualization.speed_fit`](visualization.md#trajectopy.visualization.speed_fit) | Fits a straight line to deviations (in meters) over speed (in m/s) by least squares. |
| [`visualization.SpeedFit`](visualization.md#trajectopy.visualization.SpeedFit) | Straight line fitted by `speed_fit`. |
| [`visualization.Figure`](visualization.md#trajectopy.visualization.Figure) | A titled page of plots. |
| [`visualization.show`](visualization.md#trajectopy.visualization.show) | Shows figures in a plot window. |
| [`visualization.PlotWindow`](visualization.md#trajectopy.visualization.PlotWindow) | Window showing figures as tabs. Tabs are rendered when they are first opened. |
| [`visualization.export_figures`](visualization.md#trajectopy.visualization.export_figures) | Exports all figures (and all their variants) to image files in ``directory``. |
| [`visualization.set_theme`](visualization.md#trajectopy.visualization.set_theme) | Selects the theme of all figures rendered from now on (also of figures created before). |

## [Exceptions and Warnings](exceptions.md)

| Name | Description |
| --- | --- |
| [`exceptions.TrajectopyError`](exceptions.md#trajectopy.exceptions.TrajectopyError) | Base exception for trajectopy. |
| [`exceptions.TrajectoryError`](exceptions.md#trajectopy.exceptions.TrajectoryError) | Trajectory-specific errors. |
| [`exceptions.AlignmentError`](exceptions.md#trajectopy.exceptions.AlignmentError) | Alignment-specific errors. |
| [`exceptions.AlignmentParametersError`](exceptions.md#trajectopy.exceptions.AlignmentParametersError) | Alignment parameters errors. |
| [`exceptions.ApproximationError`](exceptions.md#trajectopy.exceptions.ApproximationError) | Approximation-specific errors. |
| [`exceptions.PointSetError`](exceptions.md#trajectopy.exceptions.PointSetError) | Point set errors. |
| [`exceptions.TrajectopyIOError`](exceptions.md#trajectopy.exceptions.TrajectopyIOError) | I/O errors. |
| [`exceptions.TrajectopyDeprecationWarning`](exceptions.md#trajectopy.exceptions.TrajectopyDeprecationWarning) | Warning for parts of the API that will be removed in the next major release. |

## [Utilities](utilities.md)

| Name | Description |
| --- | --- |
| [`show_versions`](utilities.md#trajectopy.show_versions) | Prints the versions of trajectopy, Python, the operating system and the dependencies. |
