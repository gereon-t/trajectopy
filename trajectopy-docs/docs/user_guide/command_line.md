# Command Line

The `trajectopy` command starts the GUI. With a subcommand, it evaluates, aligns and converts trajectories without
a GUI, e.g. in scripts, on servers or in CI pipelines. `trajectopy <command> --help` lists the options of a command.

## Starting the GUI

```console
trajectopy [-h] [--version] [--single-thread] [--plot-settings PLOT_SETTINGS] [files ...]
```

| Argument | Description |
| --- | --- |
| `files` | Project (`.tpy`), trajectory (`.traj`, `.bag`, `.mcap`, …) or result (`.result`) files to open at start. Trajectories and results are added to the project; of several projects, only the first is opened. Bag directories cannot be opened this way. |
| `--version`, `-v` | Show the version and exit. |
| `--single-thread` | Process in the GUI thread instead of a background thread, e.g. for debugging. The GUI does not respond while processing. On by default when trajectopy runs under a debugger. |
| `--plot-settings PLOT_SETTINGS` | JSON file with [plot settings](settings.md#plot-settings); settings missing in the file keep their default values. If the file cannot be read, the GUI logs the error and starts with the default plot settings. |
| `-h`, `--help` | Show the help and exit. |

```console
trajectopy ground_truth.traj slam.traj
trajectopy flight.tpy
trajectopy --plot-settings plot_settings.json
```

A project contains its own plot settings, which replace those given with `--plot-settings` when it is opened.
The Windows app has no console, so it prints nothing: neither `--version` and `--help` nor the output of the
commands below; the version is shown in *Help → About*. The apps for macOS and Linux print to the terminal they
are started from (on macOS, start the program `Trajectopy.app/Contents/MacOS/Trajectopy`). For scripts and
pipelines, the Python package is the better choice.

## Commands

| Command | Description |
| --- | --- |
| `ate` | Absolute trajectory error of one or more trajectories against a reference |
| `rpe` | Relative pose error (drift) of one or more trajectories against a reference |
| `evaluate` | ATE and RPE of one or more trajectories against a reference |
| `align` | Align trajectories to a reference and print the parameters |
| `convert` | Convert trajectories between formats (and coordinate systems) |
| `info` | Show a summary of trajectory files and the topics of ROS bags |

All commands read trajectopy, TUM, KITTI and EuRoC files and ROS bags (`.bag`, `.mcap`, `.db3` or a bag
directory). A reference bag with several supported topics needs `--ref-topic`. Errors are printed as one line
and reported with exit code 1 (2 for invalid arguments or input, e.g. a missing file), so the commands can be used
in pipelines; with `--verbose`, errors with exit code 1 show their full traceback instead.

Output files named after a trajectory or result (`<name>` below) replace every character other than letters,
digits, `-`, `_` and `.` with `_`, e.g. `KITTI ORB vs. KITTI Ground Truth` becomes
`KITTI_ORB_vs._KITTI_Ground_Truth`.

```console
# ATE of two estimates; prints a summary table
trajectopy ate orb.traj sptam.traj --ref gt.traj

# ATE and RPE with custom settings: summary table, result files, figures and an HTML report
trajectopy evaluate orb.traj sptam.traj --ref gt.traj --settings settings.json \
    --table summary.md --output-dir results --figures figures --report report.html

# RPE of an odometry topic of a ROS2 bag against a TUM file
trajectopy rpe recording.mcap --topic /odom --ref gt.tum

# Align a trajectory and save it together with the alignment parameters
trajectopy align estimate.traj --ref gt.traj -o aligned.traj --parameters alignment.result

# Convert between formats and coordinate systems
trajectopy convert poses.txt --format kitti --to tum -o poses.tum
trajectopy convert gnss.traj --epsg 25832 -o gnss_utm.traj

# Show the trajectories of a file and the topics of a bag
trajectopy info recording.bag
```

### Options of All Commands

| Option | Description |
| --- | --- |
| `files` | Trajectory files or ROS bags |
| `--topic TOPIC` | Topic to read from ROS bags (repeatable); all supported topics by default |
| `--format FORMAT` | Format of the trajectory files: `auto` (default), `trajectopy`, `tum`, `kitti` or `euroc` |
| `--quiet`, `-q` | Print only warnings and errors |
| `--verbose` | Print debug messages and the traceback of errors (except invalid input) |

### `ate`, `rpe` and `evaluate`

| Option | Description |
| --- | --- |
| `--ref`, `-r REF` | Reference (ground truth) trajectory file or ROS bag (required) |
| `--ref-topic TOPIC` | Topic of the reference in a ROS bag |
| `--ref-format FORMAT` | Format of the reference file |
| `--settings`, `-s FILE` | JSON file with [processing settings](settings.md#processing-settings) |
| `--no-align` | Evaluate without estimating an alignment first |
| `--table`, `-t FILE` | Write the summary table to this file |
| `--table-format FORMAT` | `csv`, `markdown` or `latex`; by default derived from the extension of `--table` (`.md`, `.tex`, otherwise CSV) |
| `--no-ci` | Leave out the confidence intervals (faster) |
| `--output-dir`, `-o DIR` | Write the result files (`.result`) to this directory |
| `--figures DIR` | Export all figures to this directory (requires `trajectopy[plot]`) |
| `--plot-settings FILE` | JSON [plot settings](settings.md#plot-settings) for figures and the report |
| `--report FILE` | Write a self-contained HTML report to this file |
| `--no-report-figures` | Leave the figures out of the report |

`--output-dir` and `--figures` are created if they do not exist. `--output-dir` receives
`<name>_ate.result` and `<name>_rpe.result` for each trajectory, where `<name>` is the name of the result
(`<trajectory> vs. <reference>`), and for each
ATE result (unless `--no-align` is given) its alignment as `<name>_alignment.result`. Existing files are
overwritten.

Unless `--no-align` is given, `rpe` and `evaluate` compute the RPE of the aligned trajectory, with the alignment
of the [settings](settings.md#processing-settings). The RPE of `trajectopy.rpe` in Python and of the GUI uses the
trajectory as it is. Both agree for the default rigid body alignment, which does not change relative poses; with
scale, time shift, lever arm or sensor rotation, they differ. `--quiet` also suppresses the summary table, which is not printed
either when it is written with `--table`.

### `align`

| Option | Description |
| --- | --- |
| `--ref`, `-r REF` | Reference trajectory file or ROS bag (required) |
| `--ref-topic TOPIC`, `--ref-format FORMAT` | Topic and format of the reference |
| `--settings`, `-s FILE` | JSON file with [processing settings](settings.md#processing-settings) |
| `--output`, `-o PATH` | Write the aligned trajectory to this file (or directory for several trajectories) |
| `--to FORMAT` | Format of the aligned trajectory: `trajectopy` (default), `tum`, `kitti` or `euroc` |
| `--parameters FILE` | Write the alignment parameters to this `.result` file |

With a directory as `--output`, the aligned trajectories are written as `<name>_aligned.traj` (`.tum`, `.kitti`
or `.csv` with `--to tum`, `kitti` or `euroc`), where `<name>` is the name of the trajectory. Several
trajectories need an existing directory as `--output`; `--parameters` then holds the alignment of the last one.

### `convert`

| Option | Description |
| --- | --- |
| `--output`, `-o PATH` | Output file, or directory for several trajectories (required) |
| `--to FORMAT` | Output format: `trajectopy` (default), `tum`, `kitti` or `euroc` |
| `--epsg EPSG` | Transform the positions to this EPSG code |

With a directory as `--output`, each trajectory is written as `<name>.traj` (`.tum`, `.kitti` or `.csv` with
`--to tum`, `kitti` or `euroc`), where `<name>` is the name of the trajectory. Several trajectories need an
existing directory as `--output`.

### `info`

`info` has only the options of all commands.
