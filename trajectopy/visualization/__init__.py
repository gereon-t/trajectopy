"""Plots of trajectories, evaluation results and alignment parameters.

All plots are built with pyqtgraph and are the same in scripts and in the GUI. Install the plotting
dependencies with ``pip install "trajectopy[plot]"``.

Example:
    >>> import trajectopy as tpy
    >>> from trajectopy import visualization as vis
    >>> ate_result = tpy.ate(traj_est, traj_ref)
    >>> figures = vis.deviation_figures([ate_result], settings=tpy.PlotSettings(ate_unit_is_mm=True))
    >>> vis.export_figures(figures, "plots")   # PNG files, works headless
    >>> vis.show(figures)                     # interactive window
"""

import importlib
from typing import TYPE_CHECKING, Any

# Public names and the submodule defining them. They are imported on first use, so that the
# Qt-independent helpers (e.g. ``trajectopy.visualization.axes``) work without the plotting extras.
_EXPORTS = {
    "Figure": "figure",
    "PlotWindow": "figure",
    "export_figures": "figure",
    "show": "figure",
    "set_theme": "style",
    "alignment_figures": "plots",
    "ate_components_figure": "plots",
    "ate_distribution_figure": "plots",
    "ate_figure": "plots",
    "ate_histogram_figure": "plots",
    "ate_map_figure": "plots",
    "ate_statistics_figure": "plots",
    "deviation_figures": "plots",
    "parameter_matrix_figure": "plots",
    "positions_figure": "plots",
    "rpe_figures": "plots",
    "rpy_figure": "plots",
    "trajectory_figures": "plots",
    "trajectory_map_figure": "plots",
    "xyz_figure": "plots",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.{_EXPORTS[name]}")
    except ImportError as error:
        raise ImportError('Plotting requires the plotting dependencies: pip install "trajectopy[plot]"') from error
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])


if TYPE_CHECKING:  # pragma: no cover - for static type checkers and IDEs
    from trajectopy.visualization.figure import Figure, PlotWindow, export_figures, show
    from trajectopy.visualization.plots import (
        alignment_figures,
        ate_components_figure,
        ate_distribution_figure,
        ate_figure,
        ate_histogram_figure,
        ate_map_figure,
        ate_statistics_figure,
        deviation_figures,
        parameter_matrix_figure,
        positions_figure,
        rpe_figures,
        rpy_figure,
        trajectory_figures,
        trajectory_map_figure,
        xyz_figure,
    )
    from trajectopy.visualization.style import set_theme
