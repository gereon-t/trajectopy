"""Google Earth (KML) export of trajectories and ATE results.

Trajectories are written as time-stamped tracks (``gx:Track``) that the time slider of Google Earth plays back.
Optionally, the trajectory is additionally drawn as a line colored by a field (speed, height or time for
trajectories, a deviation for ATE results): the line is split into runs of consecutive poses of the same color
class. Each class is a folder (named by its value range, so that the Places panel of Google Earth shows the
legend and the classes can be switched on and off), and each run appears at its time, so that the colored line
grows during playback.

Vertically, the trajectory is either clamped to the terrain of Google Earth or drawn at its own heights (see
:data:`HEIGHT_MODES`). Google Earth expects heights above sea level (EGM96 geoid); ellipsoidal heights are
converted with the geoid model, which pyproj downloads from the PROJ CDN (cdn.proj.org) where needed.
"""

import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyproj
from pyproj.transformer import TransformerGroup

from trajectopy.core.trajectory import Trajectory
from trajectopy.results.ate_result import ATEResult
from trajectopy.utils.colors import COLORMAPS, color_range, value_color
from trajectopy.utils.geometry import gradient_3d

KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
KML_GX_NAMESPACE = "http://www.google.com/kml/ext/2.2"  # Google extensions: gx:Track for time-stamped tracks
KML_POINT_ICON = "https://maps.google.com/mapfiles/kml/shapes/shaded_dot.png"  # white dot, tinted by the color
DEFAULT_KML_COLOR = "#ff0000"
DEFAULT_COLOR_CLASSES = 11
# How the trajectory is placed vertically:
# "ground": clamped to the terrain (its heights are ignored); with a height offset, that far above the terrain,
# "ellipsoidal": at its heights, which are heights above the WGS84 ellipsoid (e.g. GNSS); converted to heights
#                above sea level with the EGM96 geoid model,
# "sea_level": at its heights, which are already heights above sea level (orthometric / normal heights).
HEIGHT_MODES = ("ground", "ellipsoidal", "sea_level")

_KML = f"{{{KML_NAMESPACE}}}"
_GX = f"{{{KML_GX_NAMESPACE}}}"


def kml_color(color: str) -> str:
    """Converts "#RRGGBB" or "#RRGGBBAA" to the KML color format "aabbggrr".

    Raises:
        ValueError: If the color is not given in one of these formats.
    """
    value = color.strip().removeprefix("#")
    try:
        int(value, 16)
    except ValueError:
        value = ""
    if len(value) not in (6, 8):
        raise ValueError(f'Invalid color "{color}": use "#RRGGBB" or "#RRGGBBAA", e.g. "#ff0000" for red.')
    red, green, blue, alpha = value[0:2], value[2:4], value[4:6], value[6:8] or "ff"
    return f"{alpha}{blue}{green}{red}".lower()


# ------------------------------------------------------------------------------------------------- fields


@dataclass(frozen=True)
class KmlField:
    """A per-pose value of a trajectory or an ATE result that can color the Google Earth export."""

    key: str
    label: str
    unit: str
    values: Callable[[Any], np.ndarray]
    needs_orientation: bool = False


def _speed(trajectory: Trajectory) -> np.ndarray:
    """Speed from the positions in a metric frame (earth-centered), in the order of ``trajectory``."""
    order = np.argsort(trajectory.timestamps, kind="stable")
    ecef = trajectory.positions.to_epsg(4978, inplace=False).xyz[order]
    speed = np.empty(len(order))
    speed[order] = np.linalg.norm(gradient_3d(xyz=ecef, tstamps=trajectory.timestamps[order]), axis=1)
    return speed


TRAJECTORY_KML_FIELDS = (
    KmlField("speed", "Speed", "m/s", _speed),
    KmlField("height", "Height", "m", lambda t: t.positions.to_epsg(4326, inplace=False).xyz[:, 2]),
    KmlField("time", "Time", "s", lambda t: t.timestamps - t.timestamps.min()),
)

ATE_KML_FIELDS = (
    KmlField("position", "Position deviation", "m", lambda r: r.pos_dev_comb),
    KmlField("x", "Position deviation x", "m", lambda r: r.pos_dev_x),
    KmlField("y", "Position deviation y", "m", lambda r: r.pos_dev_y),
    KmlField("z", "Position deviation z", "m", lambda r: r.pos_dev_z),
    KmlField("along_track", "Along-track deviation", "m", lambda r: r.pos_dev_along),
    KmlField("horizontal_cross_track", "Horizontal cross-track deviation", "m", lambda r: r.pos_dev_cross_h),
    KmlField("vertical_cross_track", "Vertical cross-track deviation", "m", lambda r: r.pos_dev_cross_v),
    KmlField("rotation", "Rotation deviation", "°", lambda r: np.rad2deg(r.rot_dev_comb), True),
    KmlField("roll", "Roll deviation", "°", lambda r: np.rad2deg(r.rot_dev_x), True),
    KmlField("pitch", "Pitch deviation", "°", lambda r: np.rad2deg(r.rot_dev_y), True),
    KmlField("yaw", "Yaw deviation", "°", lambda r: np.rad2deg(r.rot_dev_z), True),
)


def trajectory_kml_fields() -> list[KmlField]:
    """The fields that can color the Google Earth export of a trajectory."""
    return list(TRAJECTORY_KML_FIELDS)


def ate_kml_fields(ate_result: ATEResult) -> list[KmlField]:
    """The fields of ``ate_result`` that can color its Google Earth export."""
    return [field for field in ATE_KML_FIELDS if ate_result.has_orientation or not field.needs_orientation]


@dataclass(frozen=True)
class _ColorScale:
    """How a field is mapped to colors (see :func:`write_kml`)."""

    colormap: str
    max_std: float
    value_range: tuple[float, float] | None
    color_classes: int

    def __post_init__(self) -> None:
        if self.colormap not in COLORMAPS:
            raise ValueError(f'Unknown colormap "{self.colormap}", choose one of {sorted(COLORMAPS)}.')
        if self.color_classes < 1:
            raise ValueError("At least one color class is required.")
        if self.value_range is not None and not self.value_range[0] < self.value_range[1]:
            raise ValueError(f"Invalid value range {self.value_range}: the lower limit must be below the upper one.")

    def limits(self, values: np.ndarray) -> tuple[float, float]:
        if self.value_range is not None:
            return float(self.value_range[0]), float(self.value_range[1])
        return color_range(values, self.max_std)


# ------------------------------------------------------------------------------------------------ export


def write_kml(
    trajectory: Trajectory,
    filename: str,
    color: str = DEFAULT_KML_COLOR,
    field: str | None = None,
    colormap: str = "rdylbu",
    max_std: float = 3.0,
    value_range: tuple[float, float] | None = None,
    color_classes: int = DEFAULT_COLOR_CLASSES,
    heights: str = "ground",
    height_offset: float = 0.0,
    extrude: bool = False,
    line_width: float = 3.0,
    precision: float = 1e-6,
) -> None:
    """Exports a trajectory to a Google Earth KML file.

    The trajectory is written as a time-stamped track (``gx:Track``): Google Earth shows its path and plays
    it back with the time slider. Optionally, it is colored by a field (speed, height or time).

    Requires the trajectory to have a valid EPSG code so it can be converted to WGS84 (EPSG:4326).

    Args:
        trajectory (Trajectory): Trajectory to export.
        filename (str): The output filename (e.g., "track.kml").
        color (str, optional): Color of the track and its points as "#RRGGBB" or "#RRGGBBAA", if it is not
            colored by a field. Defaults to red.
        field (str, optional): Key of the field that colors the line, see :func:`trajectory_kml_fields`:
            "speed", "height" or "time". Defaults to none (one color).
        colormap (str, optional): Color scale of the field, see :data:`trajectopy.utils.colors.COLORMAPS`:
            "rdylbu" (blue = low, red = high, like the plots) or "viridis" (colorblind-safe). Defaults to "rdylbu".
        max_std (float, optional): The color scale is clipped to mean ± ``max_std`` standard deviations
            (0 = minimum to maximum). Ignored if ``value_range`` is given. Defaults to 3.0, like the plots.
        value_range (tuple[float, float], optional): Fixed (lower, upper) limits of the color scale, e.g. to
            compare several exports; values beyond them get the color of the nearest limit. Defaults to none.
        color_classes (int, optional): Number of colors of the scale. Defaults to 11.
        heights (str, optional): Vertical placement, see :data:`HEIGHT_MODES`: "ground" (clamped to the
            terrain), "ellipsoidal" (the trajectory's heights are ellipsoidal heights, e.g. GNSS; converted with
            the EGM96 geoid, which needs an internet connection the first time) or "sea_level" (the heights are
            above sea level). Defaults to "ground".
        height_offset (float, optional): Meters added to the heights; with "ground", the trajectory floats
            this far above the terrain. Defaults to 0.
        extrude (bool, optional): Connect the trajectory to the ground with a vertical curtain (not with
            "ground" and no offset). Defaults to False.
        line_width (float, optional): Line width in pixels. Defaults to 3.
        precision (float, optional): Coordinate precision in degrees. Defaults to 1e-6 (about 0.1 m).

    Raises:
        ValueError: If the trajectory does not have a known EPSG code, an option is invalid, or the geoid model
            is not available.
    """
    scale = _ColorScale(colormap, max_std, value_range, color_classes)
    placement = _placement(trajectory, heights, height_offset)
    root, document = _document(trajectory.name)
    if field is None:
        _add_track_style(document, "track", kml_color(color), line_width)
        _add_track(document, trajectory.name, "track", placement, extrude, precision)
    else:
        selected = _field(trajectory_kml_fields(), field)
        values = np.asarray(selected.values(trajectory), dtype=float)[placement.order]
        _add_colored_line(
            document, trajectory.name, selected, values, placement, scale, extrude, line_width, precision
        )
    _write(root, filename)


def write_ate_kml(
    ate_result: ATEResult,
    filename: str,
    field: str = "position",
    colormap: str = "rdylbu",
    max_std: float = 3.0,
    value_range: tuple[float, float] | None = None,
    color_classes: int = DEFAULT_COLOR_CLASSES,
    heights: str = "ground",
    height_offset: float = 0.0,
    extrude: bool = False,
    line_width: float = 4.0,
    precision: float = 1e-6,
) -> None:
    """Exports an ATE result to a Google Earth KML file, colored by one of its deviation fields.

    The evaluated trajectory is drawn as a colored line, and as a time-stamped track that the time slider of
    Google Earth plays back. The Places panel lists the color classes with their value ranges (legend).

    Args:
        ate_result (ATEResult): Result to export. Its trajectory needs a known EPSG code.
        filename (str): The output filename (e.g., "ate.kml").
        field (str, optional): Key of the deviation field that colors the line, see :func:`ate_kml_fields`,
            e.g. "position", "along_track" or "yaw". Defaults to "position".
        colormap (str, optional): Color scale, see :func:`write_kml`. Defaults to "rdylbu".
        max_std (float, optional): The color scale is clipped to mean ± ``max_std`` standard deviations
            (0 = minimum to maximum). Ignored if ``value_range`` is given. Defaults to 3.0, like the plots.
        value_range (tuple[float, float], optional): Fixed (lower, upper) limits of the color scale.
            Defaults to none.
        color_classes (int, optional): Number of colors of the scale. Defaults to 11.
        heights (str, optional): Vertical placement, see :func:`write_kml`. Defaults to "ground".
        height_offset (float, optional): Meters added to the heights, see :func:`write_kml`. Defaults to 0.
        extrude (bool, optional): Connect the line to the ground with a vertical curtain. Defaults to False.
        line_width (float, optional): Line width in pixels. Defaults to 4.
        precision (float, optional): Coordinate precision in degrees. Defaults to 1e-6 (about 0.1 m).

    Raises:
        ValueError: If the trajectory has no known EPSG code, the field is not available, an option is invalid,
            or the geoid model is not available.
    """
    selected = _field(ate_kml_fields(ate_result), field)
    scale = _ColorScale(colormap, max_std, value_range, color_classes)
    placement = _placement(ate_result.trajectory, heights, height_offset)
    values = np.asarray(selected.values(ate_result), dtype=float)[placement.order]
    root, document = _document(ate_result.name)
    _add_colored_line(document, ate_result.name, selected, values, placement, scale, extrude, line_width, precision)
    _write(root, filename)


# --------------------------------------------------------------------------------------------- placement


@dataclass(frozen=True)
class _Placement:
    """Poses in chronological order with unique timestamps (as needed for playback), as written to KML."""

    lon_lat_alt: np.ndarray  # (N, 3); the altitude is meaningless for "clampToGround"
    timestamps: np.ndarray
    order: np.ndarray  # indices of these poses in the trajectory
    altitude_mode: str  # "clampToGround", "relativeToGround" or "absolute" (above sea level)


def _placement(trajectory: Trajectory, heights: str, height_offset: float) -> _Placement:
    if heights not in HEIGHT_MODES:
        raise ValueError(f'Unknown heights "{heights}", choose one of {list(HEIGHT_MODES)}.')
    if trajectory.positions.local_transformer is None:
        raise ValueError(
            "Trajectory must be defined in a well-known coordinate system (EPSG code) to be exported to KML. "
        )
    order = np.argsort(trajectory.timestamps, kind="stable")
    _, first_of_each_time = np.unique(trajectory.timestamps[order], return_index=True)
    order = order[np.sort(first_of_each_time)]
    lat, lon, height = trajectory.positions.to_epsg(4326, inplace=False).xyz[order].T

    if heights == "ground":
        altitude = np.full(len(order), float(height_offset))
        mode = "relativeToGround" if height_offset else "clampToGround"
    else:
        if heights == "ellipsoidal":
            height = _ellipsoidal_to_sea_level(lat, lon, height)
        altitude = height + height_offset
        mode = "absolute"
    return _Placement(np.column_stack([lon, lat, altitude]), trajectory.timestamps[order], order, mode)


def _ellipsoidal_to_sea_level(lat: np.ndarray, lon: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Heights above the EGM96 geoid, which Google Earth uses as sea level."""
    network_enabled = pyproj.network.is_network_enabled()
    pyproj.network.set_network_enabled(True)  # the geoid grid is downloaded (and cached) where needed
    try:
        group = TransformerGroup("EPSG:4979", "EPSG:4326+5773", always_xy=True)  # ellipsoidal -> EGM96 height
        if not group.best_available:
            raise ValueError(
                "The EGM96 geoid model is not available (it is downloaded from cdn.proj.org on first use). "
                'Check the internet connection, or use heights="sea_level" or "ground".'
            )
        _, _, sea_level = group.transformers[0].transform(lon, lat, height)
    finally:
        pyproj.network.set_network_enabled(network_enabled)
    return np.asarray(sea_level, dtype=float)


# ---------------------------------------------------------------------------------------------- elements


def _field(fields: list[KmlField], key: str) -> KmlField:
    available = {field.key: field for field in fields}
    if key not in available:
        raise ValueError(f'Unknown or unavailable field "{key}", choose one of {sorted(available)}.')
    return available[key]


def _document(name: str) -> tuple[ET.Element, ET.Element]:
    ET.register_namespace("", KML_NAMESPACE)
    ET.register_namespace("gx", KML_GX_NAMESPACE)
    root = ET.Element(f"{_KML}kml")
    document = ET.SubElement(root, f"{_KML}Document")
    ET.SubElement(document, f"{_KML}name").text = name
    return root, document


def _add_track_style(document: ET.Element, style_id: str, color: str, line_width: float) -> None:
    style = ET.SubElement(document, f"{_KML}Style", id=style_id)
    icon_style = ET.SubElement(style, f"{_KML}IconStyle")
    ET.SubElement(icon_style, f"{_KML}color").text = color
    ET.SubElement(icon_style, f"{_KML}scale").text = "0.7"
    ET.SubElement(ET.SubElement(icon_style, f"{_KML}Icon"), f"{_KML}href").text = KML_POINT_ICON
    line_style = ET.SubElement(style, f"{_KML}LineStyle")
    ET.SubElement(line_style, f"{_KML}color").text = color
    ET.SubElement(line_style, f"{_KML}width").text = f"{line_width:g}"
    poly_style = ET.SubElement(style, f"{_KML}PolyStyle")  # the curtain of extruded lines, half transparent
    ET.SubElement(poly_style, f"{_KML}color").text = "80" + color[2:]


def _add_altitude(element: ET.Element, placement: _Placement, extrude: bool) -> None:
    if extrude and placement.altitude_mode != "clampToGround":
        ET.SubElement(element, f"{_KML}extrude").text = "1"
    if placement.altitude_mode == "clampToGround":
        ET.SubElement(element, f"{_KML}tessellate").text = "1"  # follows the terrain between the poses
    ET.SubElement(element, f"{_KML}altitudeMode").text = placement.altitude_mode


def _add_track(
    document: ET.Element, name: str, style_id: str, placement: _Placement, extrude: bool, precision: float
) -> None:
    placemark = ET.SubElement(document, f"{_KML}Placemark")
    ET.SubElement(placemark, f"{_KML}name").text = name
    ET.SubElement(placemark, f"{_KML}styleUrl").text = f"#{style_id}"
    track = ET.SubElement(placemark, f"{_GX}Track")
    if extrude and placement.altitude_mode != "clampToGround":
        ET.SubElement(track, f"{_KML}extrude").text = "1"
    ET.SubElement(track, f"{_KML}altitudeMode").text = placement.altitude_mode
    for timestamp in placement.timestamps:
        ET.SubElement(track, f"{_KML}when").text = _when(timestamp)
    decimals = _decimals(precision)
    for longitude, latitude, altitude in placement.lon_lat_alt:
        ET.SubElement(track, f"{_GX}coord").text = f"{longitude:.{decimals}f} {latitude:.{decimals}f} {altitude:.2f}"


def _add_colored_line(
    document: ET.Element,
    name: str,
    field: KmlField,
    values: np.ndarray,
    placement: _Placement,
    scale: _ColorScale,
    extrude: bool,
    line_width: float,
    precision: float,
) -> None:
    """The line colored by ``values`` with its legend, and the playback track (moving point only)."""
    color_classes = scale.color_classes
    lower, upper = scale.limits(values)
    with np.errstate(invalid="ignore"):
        fractions = np.nan_to_num((values - lower) / (upper - lower), nan=0.0)
    classes = np.clip((fractions * color_classes).astype(int), 0, color_classes - 1)

    for index in range(color_classes):
        color = kml_color(value_color((index + 0.5) / color_classes, scale.colormap))
        _add_track_style(document, f"class{index}", color, line_width)
    _add_track_style(document, "track", "ffffffff", line_width=0)  # moving point only; the colored line shows the path

    legend = ET.SubElement(document, f"{_KML}Folder")
    ET.SubElement(legend, f"{_KML}name").text = f"{field.label} [{field.unit}]"
    ET.SubElement(legend, f"{_KML}open").text = "1"
    bounds = np.linspace(lower, upper, color_classes + 1)
    folders = []
    for index in range(color_classes):
        folder = ET.SubElement(legend, f"{_KML}Folder")
        ET.SubElement(folder, f"{_KML}name").text = _class_name(bounds, index, field.unit)
        folders.append(folder)

    decimals = _decimals(precision)
    run_starts = np.flatnonzero(np.r_[True, np.diff(classes) != 0])
    run_ends = np.r_[run_starts[1:], len(classes)]  # exclusive; each run is connected to the next pose
    for start, end in zip(run_starts, run_ends):
        stop = min(end + 1, len(classes))
        if stop - start < 2:
            continue  # a single last pose has no segment
        placemark = ET.SubElement(folders[classes[start]], f"{_KML}Placemark")
        ET.SubElement(placemark, f"{_KML}styleUrl").text = f"#class{classes[start]}"
        ET.SubElement(ET.SubElement(placemark, f"{_KML}TimeSpan"), f"{_KML}begin").text = _when(
            placement.timestamps[start]
        )
        line = ET.SubElement(placemark, f"{_KML}LineString")
        _add_altitude(line, placement, extrude)
        ET.SubElement(line, f"{_KML}coordinates").text = " ".join(
            f"{lon:.{decimals}f},{lat:.{decimals}f},{alt:.2f}" for lon, lat, alt in placement.lon_lat_alt[start:stop]
        )

    _add_track(document, f"{name} (playback)", "track", placement, extrude=False, precision=precision)


def _when(timestamp: float) -> str:
    return pd.Timestamp(timestamp, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _decimals(precision: float) -> int:
    return max(0, int(-np.floor(np.log10(precision))))


def _class_name(bounds: np.ndarray, index: int, unit: str) -> str:
    """Value range of a color class; the outer classes also contain the values beyond the (clipped) scale."""
    span = bounds[-1] - bounds[0]
    decimals = max(0, int(np.ceil(-np.log10(span / (len(bounds) - 1)))) + 1) if span > 0 else 3
    low, high = f"{bounds[index]:.{decimals}f}", f"{bounds[index + 1]:.{decimals}f}"
    if len(bounds) == 2:
        return f"{low} – {high} {unit}"
    if index == 0:
        return f"≤ {high} {unit}"
    if index == len(bounds) - 2:
        return f"≥ {low} {unit}"
    return f"{low} – {high} {unit}"


def _write(root: ET.Element, filename: str) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space=" ", level=0)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
