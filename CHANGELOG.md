# Changelog

All notable changes to trajectopy are documented here. trajectopy follows
[semantic versioning](https://semver.org); see the
[API policy](https://gereon-t.github.io/trajectopy/api_policy/) for what is covered by it.

## 7.0.0

Version 7 settles the public API, so that it can grow without breaking changes. It defines which names are
public, applies the same conventions to every function and removes deprecated and redundant names without a
transition period: code of version 6 that uses them fails with an error. Changed defaults, above all that
functions no longer modify their input, do not raise errors; they are listed under *Breaking changes*. The
table of removed names and the migration guide below list the replacements.

### Highlights

- **Inputs are never modified by default.** Every function and method with an `inplace` argument returns a
  new object unless `inplace=True` is passed. `inplace` is keyword-only.
- **One settings argument.** All processing functions take a single `settings` argument: their own settings
  type or the complete `ProcessingSettings`. The same settings object can be passed everywhere:
  `tpy.ate(est, ref, settings)`, `tpy.align(est, ref, settings)`, `tpy.match_trajectories(est, ref, settings)`.
- **`ate_result.alignment`** holds the alignment that was applied before the deviations were computed.
- **Options accept strings**, e.g. `tpy.MatchingSettings(method="nearest_temporal")` or
  `tpy.interpolate(trajectory, index, method="linear")`, also when assigned to a setting
  (`settings.matching.method = "nearest_temporal"`). The lookup of these enumerations ignores the case.
- **`Trajectory`**
  - accepts arrays in the constructor: `tpy.Trajectory(xyz, quat, timestamps, name="...")`;
  - supports numpy-like indexing: `trajectory[10]` returns a `tpy.Pose`, `trajectory[10:20]`,
    `trajectory[mask]` and `trajectory[[0, 5]]` return new trajectories;
  - has a one-line `repr`, `equals(other, rtol=..., atol=...)` and `Trajectory.from_string`;
  - writes all formats: `trajectory.to_file("out.tum")`.
- **Results** have one-line `repr`s and `describe()`: `ATEResult.describe()` returns the statistics as a
  `pandas.Series`, `RPEResult.describe()` the drift statistics per pair distance as a `pandas.DataFrame`.
- **File I/O**: all reading and writing functions accept `pathlib.Path`. `trajectopy.io.read_trajectory`
  and `write_trajectory` handle all formats (trajectopy, TUM, KITTI, EuRoC); `trajectopy.io` also exports
  `detect_format`, `read_tum`, `read_kitti`, `read_euroc`, `list_topics` and `trajectories_from_rosbag`.
- **Public API**: the names exported by `trajectopy`, `trajectopy.io`, `trajectopy.visualization` and
  `trajectopy.exceptions` form the public API, guarded by a test. New: `tpy.Pose`, `tpy.adopt_first_pose`,
  `tpy.show_versions` and `trajectopy.exceptions.TrajectopyDeprecationWarning`. The types that results consist
  of are public and documented, too: `tpy.AbsoluteTrajectoryDeviations` (`ATEResult.abs_dev`),
  `tpy.RelativeTrajectoryDeviations` (`RPEResult.rpe_dev`), `tpy.ComponentCorrelation`
  (`DeviationCorrelation.components`), `tpy.HelmertTransformation` and `tpy.Leverarm`
  (`AlignmentParameters.helmert`, `.leverarm`) and `trajectopy.visualization.SpeedFit`.
- trajectopy attaches a `NullHandler` to its logger and leaves the logging configuration to applications.

### New Features

Besides the API changes above, version 7 adds:

- **GUI**: an interactive alignment window (drag, point pairs, time synchronization, diagnosis of the alignment
  parameters and least squares refinement), which replaces the manual alignment window; export of HTML reports
  and summary tables; and a faster file dialog on Windows.
- **Command line**: the commands `ate`, `rpe`, `evaluate`, `align`, `convert` and `info` evaluate, align and convert
  trajectories without the GUI.
- **Evaluation**: percentiles (`pos_dev_percentile`), confidence intervals that account for correlated
  deviations (`confidence_interval`), statistics per segment (`segment_statistics`), summary tables of several
  results (`summary_table`, `format_table`) and self-contained HTML reports (`write_report`).
- **Plots**: `ate_boxplot_figure`.
- **File formats**: TUM, KITTI and EuRoC files are read (recognized automatically, with a log message) and
  written; ROS2 bags and further ROS message types (odometry, transforms, paths, GNSS fixes) are read.
- **Alignment**: positions are weighted by their standard deviations (`Trajectory.position_std`, read from ROS
  covariances and the `sx`, `sy`, `sz` columns of trajectopy files).

### Breaking changes

Compared with version 6.2.1. A script compared the public API of both versions name by name and parameter by
parameter; every difference that can break existing code is listed here.

**Functions no longer modify their input by default**

- `inplace` is keyword-only everywhere and defaults to `False`, so that the input is left unchanged and the
  result is returned. It defaulted to `True` for `Trajectory.crop`, `Trajectory.intersect`, `Trajectory.mask`,
  `Trajectory.transform`, `Positions.to_epsg`, `Positions.to_local`, `align`, `apply_alignment`, `interpolate`,
  `match_trajectories` and `sort_spatially`; it was positional for these and for `approximate` and
  `Positions.mean`.

**One settings argument**

- `estimate_alignment`, `align`, `ate`, `rpe`, `match_trajectories`, `sort_spatially` and `approximate` take a
  single `settings` argument instead of `alignment_settings`, `matching_settings`, `processing_settings`,
  `sorting_settings` and `approximation_settings` (see the table below).

**Options are keyword-only**

Only the data (trajectories, results, figures, file names, indices) and the settings are positional arguments.
The following arguments must now be passed by keyword:

- Processing: `estimate_alignment(initial_parameters=)`, `ate(align=)`, `interpolate(method=, max_gap_size=)`,
  `divide_into_laps(return_lap_indices=)`.
- `Trajectory(name=, path_lengths=, velocity_xyz=, sorting=)`, `Trajectory.crop(inverse=)`,
  `Trajectory.intersect(max_gap_size=)`, `Trajectory.to_file(mode=)`, `Trajectory.to_kml(precision=, color=)`,
  `Trajectory.to_dataframe(sort_by=)`.
- `Positions(local_transformer=, init_local_transformer=, epsg_local_cart=, epsg_local_geod=)`.
- Results: `ATEResult(name=)`, `RPEResult(name=)`, all fields of `AlignmentResult`, `ATEResult.to_file(mode=)`,
  `RPEResult.to_file(mode=)`, `ATEResult.to_kml(field=, max_std=)`, `RPEResult.compute_metric(factor=)`,
  `DeviationCorrelation.from_ate_result(max_lag=, window=)`.
- Parameters and settings: `Parameter(default=, enabled=, name=, unit=)`; `enabled_only` (and `lower_case`) of
  `get_covariance_matrix`, `to_dict`, `to_string_list`, `to_name_list` and `AlignmentParameters.params_labels`;
  `AlignmentEstimationSettings.from_components(similarity=, time_shift=, leverarm=, sensor_rotation=)` and
  `AlignmentEstimationSettings.all(sensor_rotation=)`.
- `trajectopy.io`: `write_trajectory(mode=)`, `write_ate_result(mode=)`, `write_rpe_result(mode=)` and all
  options of `write_kml` and `write_ate_kml` (`color`, `field`, `colormap`, `max_std`, `value_range`,
  `color_classes`, `heights`, `height_offset`, `extrude`, `line_width`, `precision`).
- `trajectopy.visualization`: `Figure.export(variant=, width=)`, `Figure.create_widget(use_opengl=)`,
  `PlotWindow(title=, export_format=, export_width=, use_opengl=, parent=)`, `export_figures(export_format=,
  width=)`, `show(title=, block=, use_opengl=)`, `set_theme(dark=)`, `ate_speed_figure(speed_window=)`,
  `parameter_matrix_figure(correlation=)`, `smoothed_speed(window=)`.

**Changed behavior**

- `ATEResult.to_file`, `RPEResult.to_file` and `AlignmentResult.to_file` (and `trajectopy.io.write_ate_result`,
  `write_rpe_result` and `write_alignment_result`) overwrite existing files instead of appending to them; pass
  `mode="a"` to append.
- `ate(..., align=False)` sets `ate_result.alignment` to None.
- `trajectopy.io.read_trajectory` and `Trajectory.from_file` recognize TUM, KITTI and EuRoC files; `write_trajectory`
  and `Trajectory.to_file` choose the format from the file extension (`.tum`, `.kitti`) and write the trajectopy
  format otherwise.
- Orientations are `scipy.spatial.transform.Rotation` objects: `trajectory.rotations`,
  `ATEResult.abs_dev.rot_dev`, `AlignmentParameters.helmert.rotation_set` and
  `SensorRotationParameters.rotation_set`. The `Rotations` class (a subclass of scipy's `Rotation`) and its
  module `trajectopy.core.rotations` are gone. Compose rotations with `*` and `inv()` instead of `+` and `-`
  (`a - b` is `a * b.inv()`), and use `magnitude()` instead of `rotangle`.
- `repr(trajectory)` is a one-line summary; `str(trajectory)` still prints the summary table.
- Options are converted with `str()` to their value, e.g. `str(tpy.Sorting.TIME) == "time"`.
- File contents that do not match their header or format raise `TrajectopyIOError` and alignments that cannot be
  estimated raise `AlignmentError` (both are also `ValueError`s, see *Fixes*).

**Removed names and modules**

- The names and arguments in the table *Removed and renamed names* below.
- The modules deprecated since version 5: `trajectopy.readers` (use `trajectopy.io`), `trajectopy.utils.common`
  and `trajectopy.processing.lib.alignment.parameters` (use `trajectopy.AlignmentParameters`).
- The internal package `trajectopy.processing.lib` was renamed to `trajectopy.processing._lib` to mark it as
  private.

### Documentation

- The documentation is restructured: installation, a user guide with one page per topic (trajectories,
  alignment, evaluation, visualization, Google Earth, file formats, settings, GUI, command line), an API
  reference organized by the public namespaces, the API policy and the release notes.
- Every public function, class, method and property is documented with its arguments and return value. Every
  public function and class has examples, which run as doctests, and links to related functions (See Also).
- The examples of the user guide and the scripts in `example_scripts` are run by the tests.

### Fixes

- `trajectopy.match_trajectories` failed with an import error if it was the first processing function used in
  a session.
- `AlignmentParameters.to_file` (and `trajectopy.io.write_alignment_parameters`) wrote files that
  `AlignmentParameters.from_file` could not read; the parameters are now written with their enabled flags and
  17 significant digits.
- An unknown `#time_format` in a file header raises a `TrajectopyIOError` that lists the valid formats instead
  of a `KeyError`.
- `TrajectopyIOError` and `AlignmentError` are raised now: for files whose structure does not match their
  header or format (e.g. a wrong number of columns) and for alignments that cannot be estimated (e.g. no poses
  left after the preprocessing), where a plain `ValueError` was raised before. Both are also `ValueError`s, so
  existing error handling keeps working.
- `Trajectory(..., sorting=)`, `Trajectory.set_sorting` and `ATEResult.set_sorting` accept the sorting as a
  string (`"time"`, `"path_length"`, case-insensitive), like all other options; before, a string broke the
  `repr` and the file export of the trajectory.
- `Positions.mean(inplace=True)` returns the modified object, like all other functions with `inplace`.
- `Trajectory(...)` copies the arrays and the `Positions` object it is created from; before, it reordered the
  caller's positions in place when the timestamps were not in chronological order.
- The poses returned when iterating over a trajectory hold their velocity as a tuple, like those returned by
  indexing; before, it was a NumPy array, so that comparing poses failed.
- Alignment result files keep whether the estimation converged (`#converged false`); before, every alignment
  read from a file counted as converged.

### Removed and renamed names

| Version 6 | Version 7 |
| --- | --- |
| `estimate_alignment(..., alignment_settings=, matching_settings=)`, same for `align` | `settings=` (`AlignmentSettings` or `ProcessingSettings`) |
| `ate(..., processing_settings=)`, `rpe(..., processing_settings=)` | `settings=` |
| `match_trajectories(..., matching_settings=)` | `settings=` |
| `sort_spatially(..., sorting_settings=)` | `settings=` |
| `approximate(..., approximation_settings=)` | `settings=` |
| `ate(..., return_alignment=True)` | `ate_result.alignment` |
| `Trajectory.from_file(text, io_stream=True)` | `Trajectory.from_string(text)` |
| `trajectopy.IOError`, `trajectopy.exceptions.IOError` | `trajectopy.TrajectopyIOError` |
| `trajectopy.ComparisonMethod` | none (an internal detail of the file I/O) |
| `trajectopy.io.HeaderData` | none (an internal detail of reading files) |
| `MatchingMethod.UNKNOWN` | none (it was never a matching method); settings files with `"method": "unknown"` can no longer be read |
| `trajectopy.Rotations` | `scipy.spatial.transform.Rotation` |

### Migrating from version 6

```python
# version 6
aligned = tpy.align(est, ref, alignment_settings=s.alignment, matching_settings=s.matching)  # modified est
ate_result, alignment = tpy.ate(est, ref, processing_settings=s, return_alignment=True)
trajectory.crop(10, 20)  # modified trajectory

# version 7
aligned = tpy.align(est, ref, s)  # est is unchanged
ate_result = tpy.ate(est, ref, s)
alignment = ate_result.alignment
trajectory = trajectory.crop(10, 20)  # or trajectory.crop(10, 20, inplace=True)
```

Code that relied on functions modifying their input without using the return value has to use the return
value (or pass `inplace=True`). Removed arguments raise a `TypeError` and removed names an `AttributeError`
(an `ImportError` with `from ... import`).
