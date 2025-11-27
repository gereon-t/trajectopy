<div align="center">
    <h1>Trajectopy - Trajectory Evaluation in Python</h1>
    <a href="https://github.com/gereon-t/trajectopy/releases"><img src="https://img.shields.io/github/v/release/gereon-t/trajectopy?label=version" /></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8.2+-blue.svg" /></a>
    <a href="https://github.com/gereon-t/trajectopy/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gereon-t/trajectopy" /></a>
    <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
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
- __Customizable Visualization__: Powered by [Plotly](https://plotly.com/) or [Matplotlib](https://matplotlib.org/), trajectopy offers a range of interactive plots that can be customized to your needs. ([Demo](https://htmlpreview.github.io/?https://github.com/gereon-t/trajectopy/blob/main/example_data/report.html))

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

Now you can use trajectopy as a Python package in your scripts.

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
usage: trajectopy [-h] [--version] [--single-thread] [--report-settings REPORT_SETTINGS] [--mpl-settings MPL_SETTINGS] [--report-path REPORT_PATH] [--mapbox-token MAPBOX_TOKEN]

Trajectopy - Trajectory Evaluation in Python

options:
  -h, --help            show this help message and exit
  --version, -v
  --single-thread       Disable multithreading
  --report-settings REPORT_SETTINGS
                        Path to JSON report settings file that will override the default settings.
  --mpl-settings MPL_SETTINGS
                        Path to JSON matplotlib plot settings file that will override the default settings.
  --report-path, -o REPORT_PATH
                        Output directory for all reports of one session. If not specified, a temporary directory will be used.
  --mapbox-token MAPBOX_TOKEN
                        Mapbox token to use Mapbox map styles in trajectory plots.
```
Trajectopy allows users to customize the report output path and settings. By default, reports are stored in a temporary directory that will be deleted when the program exits. If you want to keep the reports, you can specify a custom output path using the `--report-path` option. The report settings can be customized using a JSON file.
The report settings file must include all available settings. You can find a sample file [here](https://github.com/gereon-t/trajectopy/blob/main/example_data/report_settings.json).
In addition, you can customize the Matplotlib plot settings using a JSON file with the `--mpl-settings` option. A sample file can be found [here](https://github.com/gereon-t/trajectopy/blob/main/example_data/matplotlib_settings.json).

Example:
```console
trajectopy --report-settings ./report_settings.json -o ./persistent_report_directory
```

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
