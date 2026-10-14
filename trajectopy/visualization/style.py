"""Shared look of all plots: colors, colormaps and the light/dark theme."""

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui

from trajectopy.utils.colors import VALUE_COLORS

# Qualitative palette for lines of different trajectories / results (Tableau 10).
PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

# Colors of the three components of a vector (x/roll, y/pitch, z/yaw).
AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")

# Sequential map for color-coded magnitudes (blue = low, red = high), shared with the Google Earth export.
VALUE_COLORMAP = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(VALUE_COLORS)), color=list(VALUE_COLORS))


@dataclass(frozen=True)
class Theme:
    background: str
    foreground: str
    grid_alpha: float
    # Diverging map for signed matrices such as correlations; its neutral center matches the background.
    diverging_colormap: str


LIGHT = Theme(background="#ffffff", foreground="#3c3c3c", grid_alpha=0.25, diverging_colormap="CET-D1")
DARK = Theme(background="#1e1e1e", foreground="#bbbbbb", grid_alpha=0.2, diverging_colormap="CET-D4")

_current_theme = LIGHT

ANTIALIAS_MAX_POINTS = 50_000


def set_theme(dark: bool) -> None:
    """Selects the theme for plots created from now on."""
    global _current_theme
    _current_theme = DARK if dark else LIGHT
    # Antialiasing is enabled per curve (see ``antialias``), because Qt draws antialiased wide lines very slowly.
    pg.setConfigOptions(background=_current_theme.background, foreground=_current_theme.foreground, antialias=False)


def antialias(num_points: int) -> bool:
    """Whether a curve with ``num_points`` points should be drawn antialiased.

    Antialiasing looks better but makes Qt's raster engine orders of magnitude slower for long curves.
    """
    return num_points <= ANTIALIAS_MAX_POINTS


def theme() -> Theme:
    return _current_theme


def diverging_colormap() -> pg.ColorMap:
    return pg.colormap.get(_current_theme.diverging_colormap)


def readable_text_color(background: QtGui.QColor) -> str:
    """Black or white, whichever is easier to read on ``background``."""
    luminance = 0.299 * background.redF() + 0.587 * background.greenF() + 0.114 * background.blueF()
    return "#000000" if luminance > 0.5 else "#ffffff"


def color(index: int) -> str:
    """Color of the ``index``-th data series."""
    return PALETTE[index % len(PALETTE)]


def pen(index_or_color: int | str, width: float = 1.5, style=None) -> QtGui.QPen:
    line_color = color(index_or_color) if isinstance(index_or_color, int) else index_or_color
    line_pen = pg.mkPen(line_color, width=width)
    if style is not None:
        line_pen.setStyle(style)
    return line_pen


def brush(index_or_color: int | str, alpha: int = 255) -> QtGui.QBrush:
    fill = QtGui.QColor(color(index_or_color) if isinstance(index_or_color, int) else index_or_color)
    fill.setAlpha(alpha)
    return pg.mkBrush(fill)


set_theme(dark=False)
