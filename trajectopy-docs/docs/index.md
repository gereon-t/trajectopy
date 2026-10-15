<div align="center">
    <h1>Trajectopy - Trajectory Evaluation in Python</h1>
    <a href="https://github.com/gereon-t/trajectopy/releases"><img src="https://img.shields.io/github/v/release/gereon-t/trajectopy?label=version" /></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" /></a>
    <a href="https://github.com/gereon-t/trajectopy/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gereon-t/trajectopy" /></a>
    <a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg"></a>
    <br />
    <a href="https://github.com/gereon-t/trajectopy"><img src="https://img.shields.io/badge/Windows-0078D6?st&logo=windows&logoColor=white" /></a>
    <a href="https://github.com/gereon-t/trajectopy"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://github.com/gereon-t/trajectopy"><img src="https://img.shields.io/badge/mac%20os-000000?&logo=apple&logoColor=white" /></a>

<h4>Trajectopy is a Python package with an optional graphical user interface for empirical trajectory evaluation. </h4>

<p align="center">
  <img style="border-radius: 10px;" src="https://raw.githubusercontent.com/gereon-t/trajectopy/main/.images/trajectopy_gif_low_quality.gif">
</p>

</div>


## Key Features

- __Interactive GUI__: Load, plot, align and compare trajectories without writing code. Every action explains
  itself in the context menus, and trajectories can be aligned interactively if an automatic alignment is not possible.
- __Alignment__: A spatio-temporal least squares alignment that can be tailored to the application: similarity
  transformation, leverarm and time shift, each of them optional.
- __Comparison__: Absolute and relative trajectory error (__ATE and RPE__), computed with various pose-matching methods.
- __Data Import/Export__: ASCII files with configurable columns, time formats and coordinate systems
  (any EPSG code), and ROS bags.
- __Fast, Interactive Visualization__: Powered by [pyqtgraph](https://www.pyqtgraph.org/), the same plots are
  available in the GUI and in Python scripts, stay fluid for large trajectories, and can be exported to PNG, JPG
  or SVG. Georeferenced trajectories and ATE results can be exported to Google Earth, colored by speed, height or
  deviations, on the terrain or at their own heights, with playback.

## Installation

### Windows App

Download `Trajectopy-<version>-setup.exe` from the [latest release](https://github.com/gereon-t/trajectopy/releases/latest)
and run it. No Python installation or administrator rights are needed. The installer adds Trajectopy to the Start
menu and, optionally, opens `.traj` files with it on double-click. The app reports new versions (Help → Check for
Updates); installing a newer version over an existing one updates it.

As the installer is not signed, Windows SmartScreen may warn about an unknown publisher: choose
*More info → Run anyway*.

### Python Package

| Command | Contents |
| --- | --- |
| `pip install "trajectopy[gui]"` | Python package and GUI (recommended) |
| `pip install "trajectopy[plot]"` | Python package with plotting, without the GUI |
| `pip install trajectopy` | Python package only |

Trajectopy requires Python 3.10 or newer.

**Faster loading of large files**: trajectory files with millions of poses load several times faster with the
optional `fast` extra, which installs [pyarrow](https://arrow.apache.org/docs/python/) for reading. It can be
combined with the other extras:

```console
pip install "trajectopy[gui,fast]"
```

## Quick Start

### Python

Load two trajectories and compute the absolute and relative trajectory error. The example files are in the
[`example_data`](https://github.com/gereon-t/trajectopy/tree/main/example_data) folder of the repository.

```python
import trajectopy as tpy

reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

# ATE: the estimate is aligned to the reference first (similarity transformation)
ate_result = tpy.ate(trajectory=estimate, other=reference)
print(f"Position ATE: {ate_result.pos_ate:.3f} m")

# RPE: drift per 100 m to 800 m (see the relative comparison settings)
rpe_result = tpy.rpe(trajectory=estimate, other=reference)
print(f"Position RPE: {rpe_result.pos_rpe:.3f} {rpe_result.pos_drift_unit}")
```

The [User Guide](user_guide.md) covers alignment, evaluation, plots and the GUI in detail.

### GUI

Start the Windows app from the Start menu, or run:

```console
trajectopy
```

Drop trajectory files onto the trajectory table, right-click the ground truth and choose *Set as Reference*,
then right-click another trajectory and choose *Compare with Reference*. See [Using the GUI](user_guide.md#6-using-the-gui).

## Command Line

```console
trajectopy [-h] [--version] [--single-thread] [--plot-settings PLOT_SETTINGS] [files ...]
```

| Argument | Description |
| --- | --- |
| `files` | Trajectory (`.traj`, `.bag`) or result (`.result`) files to open at start. |
| `--version`, `-v` | Show the version and exit. |
| `--single-thread` | Process in the GUI thread instead of a background thread, e.g. for debugging. The GUI does not respond while processing. |
| `--plot-settings PLOT_SETTINGS` | JSON file with [plot settings](settings.md#plot-settings); settings missing in the file keep their default values. |
| `-h`, `--help` | Show the help and exit. |

Examples:

```console
trajectopy ground_truth.traj slam.traj
trajectopy --plot-settings my_session/plot_settings.json
```

A session saved from the GUI contains its plot settings in `plot_settings.json`.

## Citation

If you use Trajectopy for academic work, please cite our [paper](https://www.degruyter.com/document/doi/10.1515/jag-2024-0040/html):

```bibtex
@article{Tombrink2024,
  title = {Spatio-temporal trajectory alignment for trajectory evaluation},
  author = {Gereon Tombrink and Ansgar Dreier and Lasse Klingbeil and Heiner Kuhlmann},
  journal = {Journal of Applied Geodesy},
  year = {2024},
  doi = {10.1515/jag-2024-0040},
  url = {https://doi.org/10.1515/jag-2024-0040},
  codeurl = {https://github.com/gereon-t/trajectopy}
}
```

## License

Trajectopy is released under the GNU General Public License v3.0.

## Contributing

Contributions are welcome! Please see our [GitHub repository](https://github.com/gereon-t/trajectopy) for details.

---

*Developed at the Institute of Geodesy and Geoinformation, University of Bonn*
