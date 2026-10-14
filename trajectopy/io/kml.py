"""Google Earth (KML) export of trajectories and ATE results.

Trajectories are written as time-stamped tracks (``gx:Track``) that the time slider of Google Earth plays back.
ATE results are additionally drawn as a line colored by a deviation field: the line is split into runs of
consecutive poses of the same color class. Each class is a folder (named by its value range, so that the
Places panel of Google Earth shows the legend and the classes can be switched on and off), and each run
appears at its time, so that the colored line grows during playback.
"""

import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trajectopy.core.trajectory import Trajectory
from trajectopy.results.ate_result import ATEResult
from trajectopy.utils.colors import color_range, value_color

KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
KML_GX_NAMESPACE = "http://www.google.com/kml/ext/2.2"  # Google extensions: gx:Track for time-stamped tracks
KML_POINT_ICON = "https://maps.google.com/mapfiles/kml/shapes/shaded_dot.png"  # white dot, tinted by the color
DEFAULT_KML_COLOR = "#ff0000"
DEFAULT_COLOR_CLASSES = 11

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


# --------------------------------------------------------------------------------------------- ATE fields


@dataclass(frozen=True)
class AteKmlField:
    """A deviation field of an ATE result that can color the Google Earth export."""

    key: str
    label: str
    unit: str
    values: Callable[[ATEResult], np.ndarray]
    needs_orientation: bool = False


ATE_KML_FIELDS = (
    AteKmlField("position", "Position deviation", "m", lambda r: r.pos_dev_comb),
    AteKmlField("x", "Position deviation x", "m", lambda r: r.pos_dev_x),
    AteKmlField("y", "Position deviation y", "m", lambda r: r.pos_dev_y),
    AteKmlField("z", "Position deviation z", "m", lambda r: r.pos_dev_z),
    AteKmlField("along_track", "Along-track deviation", "m", lambda r: r.pos_dev_along),
    AteKmlField("horizontal_cross_track", "Horizontal cross-track deviation", "m", lambda r: r.pos_dev_cross_h),
    AteKmlField("vertical_cross_track", "Vertical cross-track deviation", "m", lambda r: r.pos_dev_cross_v),
    AteKmlField("rotation", "Rotation deviation", "°", lambda r: np.rad2deg(r.rot_dev_comb), True),
    AteKmlField("roll", "Roll deviation", "°", lambda r: np.rad2deg(r.rot_dev_x), True),
    AteKmlField("pitch", "Pitch deviation", "°", lambda r: np.rad2deg(r.rot_dev_y), True),
    AteKmlField("yaw", "Yaw deviation", "°", lambda r: np.rad2deg(r.rot_dev_z), True),
)


def ate_kml_fields(ate_result: ATEResult) -> list[AteKmlField]:
    """The fields of ``ate_result`` that can color its Google Earth export."""
    return [field for field in ATE_KML_FIELDS if ate_result.has_orientation or not field.needs_orientation]


# ------------------------------------------------------------------------------------------------ export


def write_kml(trajectory: Trajectory, filename: str, precision: float = 1e-6, color: str = DEFAULT_KML_COLOR) -> None:
    """Exports a trajectory to a Google Earth KML file.

    The trajectory is written as a time-stamped track (``gx:Track``): Google Earth shows its path and plays
    it back with the time slider. The positions are clamped to the ground.

    Requires the trajectory to have a valid EPSG code so it can be converted to WGS84 (EPSG:4326).

    Args:
        trajectory (Trajectory): Trajectory to export.
        filename (str): The output filename (e.g., "track.kml").
        precision (float, optional): Coordinate precision in degrees. Defaults to 1e-6 (about 0.1 m).
        color (str, optional): Color of the track and its points as "#RRGGBB" or "#RRGGBBAA".
            Defaults to red.

    Raises:
        ValueError: If the trajectory does not have a known EPSG code or the color is invalid.
    """
    color_kml = kml_color(color)
    lat_lon, timestamps, _ = _chronological_wgs84(trajectory)
    root, document = _document(trajectory.name)
    _add_track_style(document, "track", color_kml, line_width=2)
    _add_track(document, trajectory.name, "track", lat_lon, timestamps, precision)
    _write(root, filename)


def write_ate_kml(
    ate_result: ATEResult,
    filename: str,
    field: str = "position",
    max_std: float = 3.0,
    color_classes: int = DEFAULT_COLOR_CLASSES,
    precision: float = 1e-6,
) -> None:
    """Exports an ATE result to a Google Earth KML file, colored by one of its deviation fields.

    The evaluated trajectory is drawn as a line in the colors of the plots (blue = low, red = high), and as a
    time-stamped track that the time slider of Google Earth plays back. The Places panel lists the color
    classes with their value ranges (legend). Positions are clamped to the ground.

    Args:
        ate_result (ATEResult): Result to export. Its trajectory needs a known EPSG code.
        filename (str): The output filename (e.g., "ate.kml").
        field (str, optional): Key of the deviation field that colors the line, see :func:`ate_kml_fields`,
            e.g. "position", "along_track" or "yaw". Defaults to "position".
        max_std (float, optional): The color scale is clipped to mean ± ``max_std`` standard deviations
            (0 = minimum to maximum). Defaults to 3.0, like the plots.
        color_classes (int, optional): Number of colors of the scale. Defaults to 11.
        precision (float, optional): Coordinate precision in degrees. Defaults to 1e-6 (about 0.1 m).

    Raises:
        ValueError: If the trajectory has no known EPSG code or the field is not available.
    """
    fields = {available.key: available for available in ate_kml_fields(ate_result)}
    if field not in fields:
        raise ValueError(f'Unknown or unavailable field "{field}", choose one of {sorted(fields)}.')
    selected = fields[field]
    if color_classes < 1:
        raise ValueError("At least one color class is required.")

    lat_lon, timestamps, order = _chronological_wgs84(ate_result.trajectory)
    values = np.asarray(selected.values(ate_result), dtype=float)[order]
    lower, upper = color_range(values, max_std)
    classes = np.clip(((values - lower) / (upper - lower) * color_classes).astype(int), 0, color_classes - 1)

    root, document = _document(ate_result.name)
    for index in range(color_classes):
        color = kml_color(value_color((index + 0.5) / color_classes))
        _add_track_style(document, f"class{index}", color, line_width=4)
    _add_track_style(document, "track", "ffffffff", line_width=0)  # moving point only; the colored line shows the path

    title = f"{selected.label} [{selected.unit}]"
    deviations = ET.SubElement(document, f"{_KML}Folder")
    ET.SubElement(deviations, f"{_KML}name").text = title
    ET.SubElement(deviations, f"{_KML}open").text = "1"
    bounds = np.linspace(lower, upper, color_classes + 1)
    folders = []
    for index in range(color_classes):
        folder = ET.SubElement(deviations, f"{_KML}Folder")
        ET.SubElement(folder, f"{_KML}name").text = _class_name(bounds, index, selected.unit)
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
        ET.SubElement(ET.SubElement(placemark, f"{_KML}TimeSpan"), f"{_KML}begin").text = _when(timestamps[start])
        line = ET.SubElement(placemark, f"{_KML}LineString")
        ET.SubElement(line, f"{_KML}tessellate").text = "1"
        ET.SubElement(line, f"{_KML}coordinates").text = " ".join(
            f"{lon:.{decimals}f},{lat:.{decimals}f},0" for lat, lon in lat_lon[start:stop]
        )

    _add_track(document, f"{ate_result.name} (playback)", "track", lat_lon, timestamps, precision)
    _write(root, filename)


# --------------------------------------------------------------------------------------------- helpers


def _chronological_wgs84(trajectory: Trajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latitudes/longitudes and timestamps in chronological order with unique timestamps (as needed for
    playback), and the indices of these poses in ``trajectory``."""
    if trajectory.positions.local_transformer is None:
        raise ValueError(
            "Trajectory must be defined in a well-known coordinate system (EPSG code) to be exported to KML. "
        )
    order = np.argsort(trajectory.timestamps, kind="stable")
    _, first_of_each_time = np.unique(trajectory.timestamps[order], return_index=True)
    order = order[np.sort(first_of_each_time)]
    wgs84 = trajectory.positions.to_epsg(4326, inplace=False).xyz[order, :2]
    return wgs84, trajectory.timestamps[order], order


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


def _add_track(
    document: ET.Element, name: str, style_id: str, lat_lon: np.ndarray, timestamps: np.ndarray, precision: float
) -> None:
    placemark = ET.SubElement(document, f"{_KML}Placemark")
    ET.SubElement(placemark, f"{_KML}name").text = name
    ET.SubElement(placemark, f"{_KML}styleUrl").text = f"#{style_id}"
    track = ET.SubElement(placemark, f"{_GX}Track")
    ET.SubElement(track, f"{_KML}altitudeMode").text = "clampToGround"
    for timestamp in timestamps:
        ET.SubElement(track, f"{_KML}when").text = _when(timestamp)
    decimals = _decimals(precision)
    for latitude, longitude in lat_lon:
        ET.SubElement(track, f"{_GX}coord").text = f"{longitude:.{decimals}f} {latitude:.{decimals}f} 0"


def _when(timestamp: float) -> str:
    return pd.Timestamp(timestamp, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _decimals(precision: float) -> int:
    return max(0, int(-np.floor(np.log10(precision))))


def _class_name(bounds: np.ndarray, index: int, unit: str) -> str:
    """Value range of a color class; the outer classes also contain the values beyond the (clipped) scale."""
    span = bounds[-1] - bounds[0]
    decimals = max(0, int(np.ceil(-np.log10(span / (len(bounds) - 1)))) + 1) if span > 0 else 3
    low, high = f"{bounds[index]:.{decimals}f}", f"{bounds[index + 1]:.{decimals}f}"
    if index == 0:
        return f"≤ {high} {unit}"
    if index == len(bounds) - 2:
        return f"≥ {low} {unit}"
    return f"{low} – {high} {unit}"


def _write(root: ET.Element, filename: str) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space=" ", level=0)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
