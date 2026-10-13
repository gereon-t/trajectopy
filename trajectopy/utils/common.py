"""Deprecated: the helpers formerly defined here were split into focused modules.

- :mod:`trajectopy.utils.geometry`
- :mod:`trajectopy.utils.linalg`
- :mod:`trajectopy.utils.numeric`
- :mod:`trajectopy.utils.formatting`
- :mod:`trajectopy.utils.resources`
- :mod:`trajectopy.visualization.axes` (plotting helpers)

This module re-exports them for backwards compatibility.
"""

from typing import Any

from trajectopy.utils.formatting import dict2table, list2box, number_to_string
from trajectopy.utils.geometry import (
    common_span,
    fit_line_2d,
    fit_line_3d,
    gradient_3d,
    lengths_from_xyz,
    nearest_point,
    skew_symmetric_matrix,
)
from trajectopy.utils.linalg import least_squares, sparse_least_squares
from trajectopy.utils.numeric import rms, rndodd, round_to_precision, shrink_data
from trajectopy.utils.resources import FULL_ICON_FILE_PATH, ICON_BG_FILE_PATH, ICON_FILE_PATH, YEAR, resource_path

_VISUALIZATION_HELPERS = {"get_sorting", "get_axis_label"}


def __getattr__(name: str) -> Any:
    # Plotting helpers live in the visualization layer; import lazily to keep utils free of upward dependencies.
    if name in _VISUALIZATION_HELPERS:
        from trajectopy.visualization import axes

        return getattr(axes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FULL_ICON_FILE_PATH",
    "ICON_BG_FILE_PATH",
    "ICON_FILE_PATH",
    "YEAR",
    "common_span",
    "dict2table",
    "fit_line_2d",
    "fit_line_3d",
    "gradient_3d",
    "least_squares",
    "lengths_from_xyz",
    "list2box",
    "nearest_point",
    "number_to_string",
    "resource_path",
    "rms",
    "rndodd",
    "round_to_precision",
    "shrink_data",
    "skew_symmetric_matrix",
    "sparse_least_squares",
]
