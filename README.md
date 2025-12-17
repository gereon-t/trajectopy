<div align="center">
    <h1>Trajectopy - Trajectory Evaluation in Python</h1>
    <a href="https://github.com/gereon-t/trajectopy/releases"><img src="https://img.shields.io/github/v/release/gereon-t/trajectopy?label=version" /></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" /></a>
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

Using [Mapbox](https://www.mapbox.com/), you can visualize your trajectories on a map:

<p align="center">
  <img style="border-radius: 10px;" src=".images/plot.png">
</p>

</div>

## Installation

Full version (with GUI):

```bash
pip install "trajectopy[gui]"
```

Python package only:

```bash
pip install trajectopy
```

## Documentation

<a href="https://gereon-t.github.io/trajectopy/" target="_blank">https://gereon-t.github.io/trajectopy/</a>

## Key Features

Trajectopy offers a range of features, including:

- __Interactive GUI__: A user-friendly interface that enables seamless interaction with your trajectory data, making it easy to visualize, align, and compare trajectories.
- __Alignment__: An advanced trajectory alignment algorithm that can be tailored to the specific application and supports a similarity transformation, a leverarm and a time shift estimation.
- __Comparison__: Absolute and relative trajectory comparison metrics (__ATE and RPE__) that can be computed using various pose-matching methods.
- __Data Import/Export__: Support for importing and exporting data, ensuring compatibility with your existing workflows.
- __Customizable Visualization__: Powered by [Plotly](https://plotly.com/) or [Matplotlib](https://matplotlib.org/), trajectopy offers a range of interactive plots that can be customized to your needs. ([Demo](https://htmlpreview.github.io/?https://github.com/gereon-t/trajectopy/blob/main/example_data/report.html))

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
