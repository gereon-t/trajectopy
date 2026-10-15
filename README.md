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
    <div>
      <img src=".images/uni_bonn.svg" height="50" />
      <img src=".images/igg.gif" height="50" />
    </div>
<h4>Trajectopy is a Python package with an optional graphical user interface for empirical trajectory evaluation. </h4>

<p align="center">
  <img style="border-radius: 10px;" src="https://raw.githubusercontent.com/gereon-t/trajectopy/main/.images/trajectopy_gif_low_quality.gif">
</p>

</div>

## Installation

**Windows app**: download `Trajectopy-<version>-setup.exe` from the [latest release](https://github.com/gereon-t/trajectopy/releases/latest) and run it. No Python installation or administrator rights are needed; `.traj` files can then be opened by double-click, and the app reports new versions (Help → Check for Updates).

**Python package** (Python 3.10 or newer):

| Command | Contents |
| --- | --- |
| `pip install "trajectopy[gui]"` | Python package and GUI (recommended) |
| `pip install "trajectopy[plot]"` | Python package with plotting, without the GUI |
| `pip install trajectopy` | Python package only |

Loading large trajectory files (millions of poses) is several times faster with the `fast` extra, which can be combined with the others: `pip install "trajectopy[gui,fast]"`.

## Quick Start

```python
import trajectopy as tpy

reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

ate_result = tpy.ate(trajectory=estimate, other=reference)  # aligns the estimate first
print(f"Position ATE: {ate_result.pos_ate:.3f} m")
```

Or start the GUI with `trajectopy` (or from the Start menu), drop trajectory files onto it, right-click the ground truth → *Set as Reference*, then right-click another trajectory → *Compare with Reference*.

## Documentation

<a href="https://gereon-t.github.io/trajectopy/" target="_blank">https://gereon-t.github.io/trajectopy/</a>

## Key Features

- __Interactive GUI__: Load, plot, align and compare trajectories without writing code. Every action explains itself in the context menus, and trajectories can be aligned interactively if an automatic alignment is not possible.
- __Alignment__: A spatio-temporal least squares alignment that can be tailored to the application: similarity transformation, leverarm and time shift, each of them optional.
- __Comparison__: Absolute and relative trajectory error (__ATE and RPE__), computed with various pose-matching methods.
- __Data Import/Export__: ASCII files with configurable columns, time formats and coordinate systems (any EPSG code), and ROS bags.
- __Fast, Interactive Visualization__: Powered by [pyqtgraph](https://www.pyqtgraph.org/), the same plots are available in the GUI and in Python scripts, stay fluid for large trajectories, and can be exported to PNG, JPG or SVG. Georeferenced trajectories and ATE results can be exported to Google Earth, colored by speed, height or deviations, on the terrain or at their own heights, with playback.

## Web Application (Docker)

A simple web application is available at [gereon-t/trajectopy-web](https://github.com/gereon-t/trajectopy-web) that allows you to use the core functionality of Trajectopy using Docker.

## Citation

If you use this library for any academic work, please cite our original [paper](https://www.degruyter.com/document/doi/10.1515/jag-2024-0040/html).

```bibtex
@article{Tombrink2024,
url = {https://doi.org/10.1515/jag-2024-0040},
title = {Spatio-temporal trajectory alignment for trajectory evaluation},
author = {Gereon Tombrink and Ansgar Dreier and Lasse Klingbeil and Heiner Kuhlmann},
journal = {Journal of Applied Geodesy},
doi = {doi:10.1515/jag-2024-0040},
year = {2024},
codeurl = {https://github.com/gereon-t/trajectopy},
}
```
