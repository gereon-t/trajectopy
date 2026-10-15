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

Trajectopy offers a range of features, including:

- __Interactive GUI__: A user-friendly interface that enables seamless interaction with your trajectory data, making it easy to visualize, align, and compare trajectories.
- __Alignment__: An advanced trajectory alignment algorithm that can be tailored to the specific application and supports a similarity transformation, a leverarm and a time shift estimation.
- __Comparison__: Absolute and relative trajectory comparison metrics (__ATE and RPE__) that can be computed using various pose-matching methods.
- __Data Import/Export__: Support for importing and exporting data, ensuring compatibility with your existing workflows.
- __Fast, Interactive Visualization__: Powered by [pyqtgraph](https://www.pyqtgraph.org/), the same plots are available in the GUI and in Python scripts, stay fluid for large trajectories, and can be exported to PNG, JPG or SVG. Georeferenced trajectories and ATE results can be exported to Google Earth, colored by speed, height or deviations, on the terrain or at their own heights, with playback.

## Installation (with GUI)

It is recommended to install trajectopy with the GUI using the following command:

```console
pip install "trajectopy[gui]"
```

## Installation (without GUI)

To install trajectopy without the GUI, use the following command:

```console
pip install trajectopy
```

Now you can use trajectopy as a Python package in your scripts. To also create plots from your scripts
(without installing the GUI), install the plotting extra:

```console
pip install "trajectopy[plot]"
```

## Faster Loading of Large Files

Trajectory files with millions of poses load several times faster with the optional `fast` extra
(it installs [pyarrow](https://arrow.apache.org/docs/python/) for reading). It can be combined with the other extras:

```console
pip install "trajectopy[gui,fast]"
```

## Python Package Quick Start

Here is a minimal example to load two trajectories, align them, and calculate the Absolute Trajectory Error (ATE).

```python
import trajectopy as tpy

# 1. Load trajectories
traj_ref = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
traj_est = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

# 2. Evaluate (ATE already includes alignment)
ate_result = tpy.ate(other=traj_ref, trajectory=traj_est)

# 3. Print results
print(f"Position ATE: {ate_result.pos_ate:.3f} m")
```

For more detailed usage, see the [User Guide](user_guide.md).

## GUI Application

Trajectopy also includes a graphical user interface. To launch it, run:

```bash
trajectopy
```

## Command Line Options (GUI version only)
```console
trajectopy --help
```

```console
usage: trajectopy [-h] [--version] [--single-thread] [--plot-settings PLOT_SETTINGS]

Trajectopy - Trajectory Evaluation in Python

options:
  -h, --help            show this help message and exit
  --version, -v
  --single-thread       Disable multithreading
  --plot-settings PLOT_SETTINGS
                        Path to a JSON plot settings file that overrides the default plot settings.
```

The plot settings file contains the [plot settings](settings.md#plot-settings); settings that are
missing from the file keep their default values. A session saved from the GUI contains its plot settings
in `plot_settings.json`.

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
