# API Policy

This page describes what belongs to the public API of trajectopy, which conventions it follows and how it
changes between releases.

## Public API

The public API consists of the names exported by these namespaces:

| Namespace | Content |
| --- | --- |
| `trajectopy` | Trajectories, settings, processing functions (alignment, matching, evaluation, ...), results, exceptions |
| `trajectopy.io` | Reading and writing of trajectories (trajectopy, TUM, KITTI and EuRoC files; reading of ROS bags), results, alignment parameters and KML files |
| `trajectopy.visualization` | Figures of trajectories, results and alignment parameters (needs `trajectopy[plot]`) |
| `trajectopy.exceptions` | The exceptions raised by trajectopy and its deprecation warning |

Each of these namespaces lists its names in `__all__`. Everything else, for example
`trajectopy.core.trajectory`, `trajectopy.processing._lib`, `trajectopy.gui` and `trajectopy.cli`, is an
implementation detail. Such modules remain importable, but they can change in any release.
Methods and attributes whose names start with an underscore are private as well.

```python
import trajectopy as tpy
from trajectopy import io, visualization as vis
```

## Conventions

The whole public API follows the same conventions, so that it can grow without breaking changes:

- **Inputs are never modified.** Functions and methods that transform a trajectory return a new object. The
  keyword-only argument `inplace=True` modifies the input instead (and returns it, too). Always use the
  return value: `trajectory = tpy.align(trajectory, reference)`. Exceptions are the methods `Trajectory.set_sorting`, `ATEResult.set_sorting` and `ATEResult.remove_ate_above`,
  which change the object itself.
- **One settings argument.** Every processing function with settings takes a single `settings` argument,
  either of its own settings type (e.g. `MatchingSettings`) or the complete `ProcessingSettings`, from which it
  uses the part it needs. So the same `ProcessingSettings` can be passed to all of them. `ate`, which needs
  the matching and the alignment settings, takes only `ProcessingSettings`. `None` means the defaults.
- **Options are enumerations that accept strings.** `MatchingSettings(method="nearest_temporal")` and
  `tpy.interpolate(trajectory, index, method="linear")` work as well as the enumeration members; the lookup
  ignores the case.
- **Keyword-only options.** Only the data (trajectories, results, figures, file names, indices) and the
  settings are positional arguments; all options are keyword-only, so that new options can be added and
  options can be reordered without breaking calls.
- **Results carry everything.** Results are returned as objects instead of tuples whose shape depends on a
  flag; for example, the alignment used by `tpy.ate` is available as `ate_result.alignment`. The only
  remaining exception is `divide_into_laps(..., return_lap_indices=True)`.
- **Orientations** are [`scipy.spatial.transform.Rotation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
  objects.
- **Files.** All functions that read or write files accept `str` and `os.PathLike` (e.g. `pathlib.Path`).
  Writers overwrite existing files; those with a `mode` argument append with `mode="a"`. Only
  `write_sensor_rotation_parameters` always appends (to an alignment result file).
- **Errors.** Invalid arguments raise `TypeError` or `ValueError`. Errors specific to trajectopy raise the
  subclasses of `trajectopy.TrajectopyError`: `TrajectoryError` and `PointSetError` for trajectory and position
  data that do not fit together or cannot be transformed (e.g. a different number of positions and timestamps,
  an unknown EPSG code), `TrajectopyIOError` for files whose structure does not match their header or format
  (e.g. a wrong number of columns; values that cannot be parsed raise the `ValueError` of the parser),
  `AlignmentError` for alignments that cannot be estimated, `AlignmentParametersError` for invalid covariance
  matrices of alignment parameters and
  `ApproximationError` for failed approximations. `TrajectopyIOError` and `AlignmentError` are also
  `ValueError`s. Errors of the file system (e.g. `FileNotFoundError`) are passed on unchanged, and
  `trajectopy.visualization` raises `ImportError` without the plotting extra.

## Versioning

trajectopy follows [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`.

- **Patch releases** fix bugs.
- **Minor releases** add functionality in a backwards compatible way and may deprecate parts of the API.
- **Major releases** remove deprecated parts of the API. Other breaking changes are avoided.

The GUI and the command line interface are versioned together with the library.

## Deprecations

Parts of the public API are not removed without warning. They are first deprecated: they keep working,
but emit a `trajectopy.exceptions.TrajectopyDeprecationWarning` (a subclass of `DeprecationWarning`) that
names the replacement.
Deprecated parts are removed in the next major release, at the earliest one minor release after the
deprecation.

This policy applies from version 7 on. Version 7 itself removed the deprecated and redundant names of version 6
without a transition period; the [release notes](release_notes.md) list their replacements.

To make sure that your code is ready for the next major release, turn the warnings into errors in your
tests, e.g. with pytest:

```toml
[tool.pytest.ini_options]
filterwarnings = ["error::trajectopy.exceptions.TrajectopyDeprecationWarning"]
```

## Bug reports

Please add the output of `trajectopy.show_versions()` to bug reports.
