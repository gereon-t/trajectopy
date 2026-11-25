# Changelog

All notable changes to trajectopy are documented here. trajectopy follows
[semantic versioning](https://semver.org); see the
[API policy](https://gereon-t.github.io/trajectopy/api_policy/) for what is covered by it.

## 7.0.0

Version 7 settles the public API, so that it can grow without breaking changes. It defines which names are
public, applies the same conventions to every function and removes long deprecated aliases. Most code that
uses keyword arguments keeps working and shows deprecation warnings that name the replacement.

### Highlights

- **Inputs are never modified by default.** Every function and method with an `inplace` argument returns a
  new object unless `inplace=True` is passed. `inplace` is keyword-only.
- **One settings argument.** All processing functions take a single `settings` argument: their own settings
  type or the complete `ProcessingSettings`. The same settings object can be passed everywhere:
  `tpy.ate(est, ref, settings)`, `tpy.align(est, ref, settings)`, `tpy.match_trajectories(est, ref, settings)`.
- **`ate_result.alignment`** holds the alignment that was applied before the deviations were computed.
- **Options accept strings**, e.g. `tpy.MatchingSettings(method="nearest_temporal")` or
  `tpy.interpolate(trajectory, index, method="linear")`. All option enumerations share the base
  `OptionEnum`, whose lookup ignores the case.
- **`Trajectory`**
  - accepts arrays in the constructor: `tpy.Trajectory(xyz, quat, timestamps, name="...")`;
  - supports numpy-like indexing: `trajectory[10]` returns a `tpy.Pose`, `trajectory[10:20]`,
    `trajectory[mask]` and `trajectory[[0, 5]]` return new trajectories;
  - has a one-line `repr`, `equals(other, rtol=..., atol=...)` and `Trajectory.from_string`;
  - writes all formats: `trajectory.to_file("out.tum")`.
- **File I/O**: all reading and writing functions accept `pathlib.Path`. `trajectopy.io.read_trajectory`
  and `write_trajectory` handle all formats (trajectopy, TUM, KITTI, EuRoC); `trajectopy.io` also exports
  `detect_format`, `read_tum`, `read_kitti`, `read_euroc`, `list_topics` and `trajectories_from_rosbag`.
- **Public API**: the names exported by `trajectopy`, `trajectopy.io`, `trajectopy.visualization` and
  `trajectopy.exceptions` form the public API, guarded by a test. New: `tpy.Pose`, `tpy.adopt_first_pose`,
  `tpy.show_versions` and `trajectopy.exceptions.TrajectopyDeprecationWarning`.
- trajectopy attaches a `NullHandler` to its logger and leaves the logging configuration to applications.

### Breaking changes

- `inplace` defaults to `False` everywhere and is keyword-only. This affects `Trajectory.crop`, `intersect`,
  `mask` and `transform`, `Positions.to_epsg` and `to_local`, `align`, `apply_alignment`,
  `adopt_first_pose`, `interpolate`, `match_trajectories`, `sort_spatially` and `approximate`.
- Arguments after the data arguments are keyword-only, e.g. `estimate_alignment(..., initial_parameters=)`,
  `ate(..., align=)`, `Trajectory(..., name=)`, `Positions(..., local_transformer=)`,
  `divide_into_laps(..., return_lap_indices=)`, `Trajectory.crop(..., inverse=)` and
  `Trajectory.intersect(..., max_gap_size=)`.
- `ATEResult.to_file`, `RPEResult.to_file` and `AlignmentResult.to_file` (and the corresponding `write_*`
  functions) overwrite existing files; pass `mode="a"` to append.
- `trajectopy.io.read_trajectory` recognizes the file format; `write_trajectory` picks it from the file
  extension (`.tum`, `.kitti`) and writes the trajectopy format otherwise.
- `Rotations` wraps a `scipy.spatial.transform.Rotation` instead of subclassing it, so that trajectopy does not
  depend on scipy's internals. It offers the same constructors, conversions and operations; use
  `rotations.to_scipy()` where a scipy `Rotation` is needed (e.g. for `Slerp`) and `Rotations.from_scipy()`
  for the opposite direction. `Rotations.random` takes `rng` instead of `random_state`.
- Options are converted with `str()` to their value, e.g. `str(tpy.Sorting.TIME) == "time"`.
- Removed the modules deprecated since version 5: `trajectopy.readers` (use `trajectopy.io`),
  `trajectopy.utils.common` and `trajectopy.processing.lib.alignment.parameters`
  (use `trajectopy.AlignmentParameters`).

- The internal package `trajectopy.processing.lib` was renamed to `trajectopy.processing._lib` to mark it as
  private.

### Deprecations

These keep working in version 7 and will be removed in version 8:

| Deprecated | Replacement |
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
value (or pass `inplace=True`). Run your tests with `-W error::trajectopy.exceptions.TrajectopyDeprecationWarning`
to find the remaining uses of deprecated arguments.
