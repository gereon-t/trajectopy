"""Map backgrounds for top-view plots, built from web map tiles.

Only the standard library (``urllib``), Qt (image decoding) and pyproj (reprojection) are used. Tiles are
fetched in parallel, cached on disk and stitched into one image. Interactive plots load them in the
background; exports wait for them.

Plots with a map background are drawn in a local metric frame tangent to Web Mercator (see
:class:`MapFrame`), so that shapes and distances are preserved locally and the tiles line up exactly.
"""

import contextvars
import logging
import math
import os
import time
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.__version__ import __version__
from trajectopy.core.trajectory import Trajectory
from trajectopy.visualization import style

logger = logging.getLogger(__name__)

USER_AGENT = f"trajectopy/{__version__} (+https://github.com/gereon-t/trajectopy)"

TILE_SIZE = 256
EARTH_RADIUS = 6378137.0
ORIGIN_SHIFT = math.pi * EARTH_RADIUS  # half the Web Mercator extent
MAX_ZOOM = 19
MAX_TILES = 64
TARGET_IMAGE_WIDTH = 2048  # pixels across the trajectories' extent
TILE_TIMEOUT_S = 10.0

CACHE_MAX_AGE_S = 30 * 24 * 3600


@dataclass(frozen=True)
class TileSource:
    url: str  # with {z}, {x} and {y} placeholders
    attribution: str
    max_zoom: int


# Tile servers that can be used without an API key. Requests identify trajectopy (User-Agent) and tiles
# are cached on disk, as required by the OpenStreetMap tile usage policy.
TILE_SOURCES = {
    "osm": TileSource(
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution="© OpenStreetMap contributors",
        max_zoom=19,
    ),
    "satellite": TileSource(
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution="Imagery © Esri, Maxar, Earthstar Geographics, GIS User Community",
        max_zoom=19,
    ),
}
# Street maps are dimmed by this amount in the dark theme so that they don't outshine the data.
DARK_THEME_DIMMING = 0.55

# Set while a figure is exported: map tiles are then loaded synchronously.
load_synchronously: contextvars.ContextVar[bool] = contextvars.ContextVar("load_synchronously", default=False)

# Separate pools: map loads wait for tile downloads and must not occupy the threads the downloads need.
_tile_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="TrajectopyTiles")
_map_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TrajectopyMaps")


# ------------------------------------------------------------------------------------------- coordinates


def web_mercator_xy(trajectory: Trajectory) -> np.ndarray | None:
    """Horizontal positions in Web Mercator (EPSG:3857), or None if the trajectory is not georeferenced."""
    positions = trajectory.positions
    if positions.epsg == 0 and positions.local_transformer is None:
        return None
    try:
        return positions.to_epsg(3857, inplace=False).xyz[:, :2]
    except Exception as error:
        logger.warning("Could not transform '%s' to Web Mercator: %s", trajectory.name, error)
        return None


@dataclass(frozen=True)
class MapFrame:
    """Local east/north frame in meters, tangent to Web Mercator at ``center``.

    Web Mercator is conformal; dividing by the scale factor at the center turns its coordinates into
    local meters (accurate to well below 0.1 % within several kilometers of the center).
    """

    center: tuple[float, float]
    scale: float

    @classmethod
    def around(cls, mercator_xy: np.ndarray) -> "MapFrame":
        lower, upper = mercator_xy.min(axis=0), mercator_xy.max(axis=0)
        center = (lower + upper) / 2
        latitude = math.atan(math.sinh(center[1] / EARTH_RADIUS))
        return cls(center=(float(center[0]), float(center[1])), scale=math.cos(latitude))

    def to_local(self, mercator_xy: np.ndarray) -> np.ndarray:
        return (mercator_xy - np.asarray(self.center)) * self.scale


# ------------------------------------------------------------------------------------------------- tiles


def _cache_directory() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "trajectopy", "tiles")


def _tile_meters(zoom: int) -> float:
    return 2 * ORIGIN_SHIFT / 2**zoom


def choose_zoom(width_m: float, height_m: float, max_zoom: int = MAX_ZOOM) -> int:
    """Highest zoom level that covers the area with about TARGET_IMAGE_WIDTH pixels and at most MAX_TILES."""
    extent = max(width_m, height_m, 1.0)
    zoom = int(math.floor(math.log2(TARGET_IMAGE_WIDTH * 2 * ORIGIN_SHIFT / (TILE_SIZE * extent))))
    zoom = max(0, min(zoom, max_zoom))
    while zoom > 0:
        tiles_x = math.ceil(width_m / _tile_meters(zoom)) + 1
        tiles_y = math.ceil(height_m / _tile_meters(zoom)) + 1
        if tiles_x * tiles_y <= MAX_TILES:
            break
        zoom -= 1
    return zoom


def _fetch_tile(style_name: str, zoom: int, x: int, y: int) -> QtGui.QImage | None:
    path = os.path.join(_cache_directory(), style_name, str(zoom), str(x), f"{y}.tile")
    if not os.path.isfile(path) or time.time() - os.path.getmtime(path) > CACHE_MAX_AGE_S:
        url = TILE_SOURCES[style_name].url.format(z=zoom, x=x, y=y)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=TILE_TIMEOUT_S) as response:
                data = response.read()
        except Exception as error:
            logger.warning("Could not load map tile %s: %s", url, error)
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            file.write(data)
    image = QtGui.QImage(path)
    return None if image.isNull() else image


@dataclass(frozen=True)
class MapImage:
    """Stitched tiles and the Web Mercator rectangle (left, bottom, right, top) they cover."""

    image: QtGui.QImage
    left: float
    bottom: float
    right: float
    top: float


def load_map_image(
    mercator_xy: np.ndarray, style_name: str = "osm", dark: bool = False, margin: float = 0.1
) -> MapImage | None:
    """Downloads (or reads from the cache) the tiles covering ``mercator_xy`` and stitches them."""
    source = TILE_SOURCES[style_name]
    lower, upper = mercator_xy.min(axis=0), mercator_xy.max(axis=0)
    padding = (upper - lower) * margin + 20.0
    lower, upper = lower - padding, upper + padding
    width_m, height_m = upper - lower
    zoom = choose_zoom(float(width_m), float(height_m), max_zoom=source.max_zoom)
    tile_m = _tile_meters(zoom)
    x0, x1 = (int((value + ORIGIN_SHIFT) // tile_m) for value in (lower[0], upper[0]))
    y0, y1 = (int((ORIGIN_SHIFT - value) // tile_m) for value in (upper[1], lower[1]))

    keys = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    tiles = list(_tile_executor.map(lambda key: _fetch_tile(style_name, zoom, *key), keys))
    if not any(tile is not None for tile in tiles):
        return None

    image = QtGui.QImage((x1 - x0 + 1) * TILE_SIZE, (y1 - y0 + 1) * TILE_SIZE, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor(style.theme().background))
    painter = QtGui.QPainter(image)
    for (x, y), tile in zip(keys, tiles):
        if tile is not None:
            painter.drawImage((x - x0) * TILE_SIZE, (y - y0) * TILE_SIZE, tile)
    if dark and style_name != "satellite":
        dimming = QtGui.QColor(style.theme().background)
        dimming.setAlphaF(DARK_THEME_DIMMING)
        painter.fillRect(image.rect(), dimming)
    painter.end()

    return MapImage(
        image=image,
        left=x0 * tile_m - ORIGIN_SHIFT,
        right=(x1 + 1) * tile_m - ORIGIN_SHIFT,
        top=ORIGIN_SHIFT - y0 * tile_m,
        bottom=ORIGIN_SHIFT - (y1 + 1) * tile_m,
    )


# -------------------------------------------------------------------------------------------- plot items


class _MapLoader(QtCore.QObject):
    loaded = QtCore.Signal(object)


class MapLayer(QtWidgets.QGraphicsPixmapItem):
    """Map image in a plot that uses the local frame ``frame``."""

    def __init__(self, frame: MapFrame) -> None:
        super().__init__()
        self.frame = frame
        self.setZValue(-1000)
        self.setTransformationMode(QtCore.Qt.TransformationMode.SmoothTransformation)

    def set_map(self, map_image: MapImage | None) -> None:
        if map_image is None:
            return
        try:
            self._show(map_image)
        except RuntimeError:  # the plot was closed before the tiles arrived
            pass

    def _show(self, map_image: MapImage) -> None:
        left, top = self.frame.to_local(np.array([map_image.left, map_image.top]))
        right, bottom = self.frame.to_local(np.array([map_image.right, map_image.bottom]))
        self.setPixmap(QtGui.QPixmap.fromImage(map_image.image))
        transform = QtGui.QTransform()
        transform.translate(left, top)
        transform.scale((right - left) / map_image.image.width(), (bottom - top) / map_image.image.height())
        self.setTransform(transform)


def add_map_background(
    plot: pg.PlotItem, frame: MapFrame, mercator_xy: Sequence[np.ndarray], style_name: str = "osm"
) -> None:
    """Adds a map below the data of ``plot`` (drawn in ``frame``) and the tile attribution."""
    if style_name not in TILE_SOURCES:
        raise ValueError(f"Unknown map style '{style_name}', choose one of {sorted(TILE_SOURCES)}")
    layer = MapLayer(frame)
    plot.addItem(layer, ignoreBounds=True)
    box = QtGui.QColor(style.theme().background)
    box.setAlpha(190)
    attribution = pg.TextItem(
        TILE_SOURCES[style_name].attribution, color=style.theme().foreground, anchor=(1, 1), fill=pg.mkBrush(box)
    )
    attribution.setParentItem(plot.getViewBox())
    plot.getViewBox().sigResized.connect(
        lambda view_box: attribution.setPos(view_box.width() - 4, view_box.height() - 2)
    )
    font = attribution.textItem.font()
    font.setPointSizeF(7)
    attribution.setFont(font)

    points = np.vstack(mercator_xy)
    dark = style.theme() is style.DARK
    if load_synchronously.get():
        layer.set_map(load_map_image(points, style_name, dark))
        return

    loader = _MapLoader()
    loader.loaded.connect(layer.set_map)
    layer.loader = loader  # type: ignore[attr-defined]  # keep the loader alive as long as the layer
    _map_executor.submit(lambda: loader.loaded.emit(load_map_image(points, style_name, dark)))
