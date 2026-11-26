# Changelog

All notable changes to trajectopy are documented here. trajectopy follows
[semantic versioning](https://semver.org); see the
[API policy](https://gereon-t.github.io/trajectopy/api_policy/) for what is covered by it.

## 7.0.0

Version 7 settles the public API, so that it can grow without breaking changes. It defines which names are
public, applies the same conventions to every function and removes deprecated and redundant names without a
transition period: code written for version 6 fails with a clear error where it needs to be changed. The table
of removed names and the migration guide below list the replacements.

### Highlights

- **Inputs are never modified by default.** Every function and method with an `inplace` argument returns a
  new object unless `inplace=True` is passed. `inplace` is keyword-only.
- **One settings argument.** All processing functions take a single `settings` argument: their own settings
  type or the complete `ProcessingSettings`. The same settings object can be passed everywhere:
  `tpy.ate(est, ref, settings)`, `tpy.align(est, ref, settings)`, `tpy.match_trajectories(est, ref, settings)`.
- **`ate_result.alignment`** holds the alignment that was applied before the deviations were computed.
- **Options accept strings**, e.g. `tpy.MatchingSettings(method="nearest_temporal")` or
  `tpy.interpolate(trajectory, index, method="linear")`, also when assigned to a setting
  (`settings.matching.method = "nearest_temporal"`). All option enumerations share the base
  `OptionEnum`, whose lookup ignores the case.
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

- `inplace` defaults to `False` everywhere and is keyword-only. This affects `Trajectory.crop`, `intersect`,
  `mask` and `transform`, `Positions.to_epsg` and `to_local`, `align`, `apply_alignment`,
  `adopt_first_pose`, `interpolate`, `match_trajectories`, `sort_spatially` and `approximate`.
- Only the data (trajectories, results, figures, file names, indices) and the settings are positional
  arguments; all options are keyword-only. For example: `estimate_alignment(..., initial_parameters=)`,
  `ate(..., align=)`, `interpolate(..., method=)`, `Trajectory(..., name=)`, `Positions(..., local_transformer=)`,
  `Trajectory.crop(..., inverse=)`, `Trajectory.intersect(..., max_gap_size=)`, `Trajectory.to_file(..., mode=)`,
  `Trajectory.to_kml(..., precision=, color=)`, the `mode` of all `to_file` methods and `write_*_result`
  functions, `ATEResult.confidence_interval(quantity=, statistic=)`, `ATEResult.segment_statistics(
  segment_length=, by=)`, `ATEResult.to_kml(..., field=)`, `format_table(..., table_format=)`,
  `summary_table(..., confidence_intervals=)`, `divide_into_laps(..., return_lap_indices=)`, the options of
  `io.write_kml`, `io.write_ate_kml`, `io.read_tum`, `io.read_kitti` and `io.read_euroc`, and in
  `trajectopy.visualization` `ate_speed_figure(..., speed_window=)`, `parameter_matrix_figure(..., correlation=)`,
  `smoothed_speed(..., window=)`, `Figure.export(..., variant=, width=)`, `export_figures(..., export_format=,
  width=)`, `show(..., title=, block=, use_opengl=)`, `Figure.create_widget(use_opengl=)` and
  `PlotWindow(..., title=)`. The same holds for `Trajectory.to_dataframe(sort_by=)`,
  `RPEResult.compute_metric(..., factor=)`, `io.trajectories_from_rosbag(..., topics=)`,
  `AlignmentEstimationSettings.from_components(...)` and `.all(sensor_rotation=)`, the `enabled_only` and
  `lower_case` arguments of the parameter sets (`to_dict`, `to_name_list`, `to_string_list`,
  `get_covariance_matrix`, `AlignmentParameters.params_labels`) and the `name` (and `alignment`) of the
  `ATEResult` and `RPEResult` constructors.
- `ate(..., align=False)` sets `ate_result.alignment` to None instead of empty alignment parameters.
- `ATEResult.to_file`, `RPEResult.to_file` and `AlignmentResult.to_file` (and the corresponding `write_*`
  functions) overwrite existing files; pass `mode="a"` to append.
- `trajectopy.io.read_trajectory` recognizes the file format; `write_trajectory` picks it from the file
  extension (`.tum`, `.kitti`) and writes the trajectopy format otherwise.
- Orientations are `scipy.spatial.transform.Rotation` objects: `trajectory.rotations`,
  `ATEResult.abs_dev.rot_dev`, `AlignmentParameters.helmert.rotation_set` and
  `SensorRotationParameters.rotation_set`. The `Rotations` class, a
  subclass of scipy's `Rotation`, is gone.
  Compose rotations with `*` and `inv()` instead of `+` and `-` (`a - b` is `a * b.inv()`), and use
  `magnitude()` instead of `rotangle`.
- Options are converted with `str()` to their value, e.g. `str(tpy.Sorting.TIME) == "time"`.
- Removed the modules deprecated since version 5: `trajectopy.readers` (use `trajectopy.io`),
  `trajectopy.utils.common` and `trajectopy.processing.lib.alignment.parameters`
  (use `trajectopy.AlignmentParameters`).

- The internal package `trajectopy.processing.lib` was renamed to `trajectopy.processing._lib` to mark it as
  private.

### Documentation

- The documentation is restructured like that of numpy and pandas: installation, a user guide with one page per
  topic (trajectories, alignment, evaluation, visualization, Google Earth, file formats, settings, GUI, command
  line), an API reference organized by the public namespaces, the API policy and the release notes.
- Every public function, class, method and property is documented with its arguments and return value. Every
  public function and class has examples, which run as doctests, and links to related functions (See Also).
- The examples of the user guide and the scripts in `example_scripts` are run by the tests.

### Fixes

- `trajectopy.match_trajectories` failed with an import error if it was the first processing function used in
  a session.
- `AlignmentParameters.to_file` (and `trajectopy.io.write_alignment_parameters`) wrote files that
  `AlignmentParameters.from_file` could not read; the parameters are now written with their enabled flags and
  17 significant digits.
- An unknown `#time_format` in a file header raises a `ValueError` that lists the valid formats instead of a
  `KeyError`.
- `Trajectory(..., sorting=)`, `Trajectory.set_sorting` and `ATEResult.set_sorting` accept the sorting as a
  string (`"time"`, `"path_length"`, case-insensitive), like all other options; before, a string broke the
  `repr` and the file export of the trajectory.
- `Positions.mean(inplace=True)` returns the modified object, like all other functions with `inplace`.
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
| `MatchingMethod.UNKNOWN` | none (it was never a matching method) |
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
that names them.
