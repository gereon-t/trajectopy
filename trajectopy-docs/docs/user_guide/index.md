# User Guide

The user guide explains the concepts of trajectopy and shows how to use them, from loading trajectories to
reports of several evaluated estimates. New to trajectopy? Start with the [Getting Started](../getting_started.md)
tutorial. The [API reference](../api/index.md) describes every class and function
in detail.

| Topic | Contents |
| --- | --- |
| [Trajectories](trajectories.md) | Creating, loading, saving, selecting and transforming trajectories; coordinate systems |
| [Alignment](alignment.md) | Estimating and applying the transformation between two trajectories |
| [Evaluation](evaluation.md) | Absolute trajectory error (ATE), relative pose error (RPE), statistics, reports |
| [Visualization](visualization.md) | Interactive plots and image export |
| [Google Earth](google_earth.md) | Export of trajectories and ATE results as KML |
| [File Formats](file_formats.md) | Trajectory files, ROS bags and the files trajectopy writes |
| [Settings](settings.md) | All processing and plot settings |
| [GUI](gui.md) | The graphical user interface |
| [Command Line](command_line.md) | Evaluation, alignment and conversion from the shell |
| [Glossary](glossary.md) | The terms used in trajectopy |

The examples use the files in the [`example_data`](https://github.com/gereon-t/trajectopy/tree/main/example_data)
folder of the repository: the ground truth of a KITTI sequence and the estimates of two visual SLAM systems,
ORB-SLAM and S-PTAM.

## Conventions

Trajectopy is imported as `tpy`; the plotting module as `vis`:

```python
import trajectopy as tpy
from trajectopy import visualization as vis
```

The public API consists of the names in the namespaces `trajectopy`, `trajectopy.io`,
`trajectopy.visualization` and `trajectopy.exceptions`. Import from these namespaces, not from the modules the
names are defined in; the [API policy](../api_policy.md) describes which changes to expect between versions.

All functions follow the same conventions:

- **Trajectory and reference**: functions that compare two trajectories take the trajectory to evaluate (or to
  align) first and the reference second; the parameters are called `trajectory` and `other`.
- **Inputs are not modified**: functions and methods that change a trajectory return a new one. Pass
  `inplace=True` to change the given trajectory instead. Always use the return value:
  `trajectory = tpy.align(trajectory, reference)`. Exceptions are the methods `Trajectory.set_sorting`, `ATEResult.set_sorting` and `ATEResult.remove_ate_above`,
  which change the object itself.
- **One settings argument**: processing functions with settings take a `settings` argument, either their own
  settings type (e.g. `MatchingSettings`) or the complete `ProcessingSettings`, of which they use the part they
  need (`ate`, which needs several parts, takes only `ProcessingSettings`). `None` means the default settings.
- **Options are keyword arguments**: only the data (trajectories, results, file names, indices) and the
  settings are positional arguments; all options are passed by name, e.g. `tpy.interpolate(trajectory, index,
  method="linear")`.
- **Options as strings**: options such as the matching method are enumerations, but strings are accepted as
  well, e.g. `tpy.MatchingSettings(method="nearest_temporal")`.
- **Units**: positions, path lengths and position deviations are in meters, timestamps in seconds, angles in
  radians, unless stated otherwise.
- **Deviations**: deviations are always *evaluated trajectory minus reference*.
- **Files**: functions that read or write files accept strings and `pathlib.Path` objects. Writers overwrite
  existing files; those with a `mode` argument append to them with `mode="a"`. Only
  `write_sensor_rotation_parameters` and `SensorRotationParameters.to_file` always append (to an alignment
  result file).
