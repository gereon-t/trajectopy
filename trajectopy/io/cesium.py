"""3D playback in the browser: export of trajectories and ATE results as a standalone CesiumJS page.

The page contains the data as CZML (Cesium's format for time-dynamic data) and loads CesiumJS from a CDN, so
it opens in any browser without installing anything. The timeline and the animation widget of Cesium play
the trajectory back: a point moves along it and the (colored) line grows behind it.

Browsers don't run Cesium's background workers on pages opened directly from disk (``file://``), so the page
has to be served by a web server. :func:`show_in_browser` does that with a minimal local server that only serves
the exported page; once loaded, the page needs no server anymore (CesiumJS and the tiles come from the internet).

Without a Cesium ion access token, the globe shows OpenStreetMap or Esri World Imagery tiles. With a token
(free account at https://ion.cesium.com), it additionally shows terrain and 3D buildings (Cesium OSM Buildings),
or - map style "photorealistic" - Google's photorealistic 3D tiles (textured 3D meshes like in Google Earth).

By default, the trajectory is clamped to the ground (terrain or 3D tiles). With ``heights="ellipsoidal"`` or
``heights="geoid"``, it is drawn at its own heights instead (see :data:`HEIGHT_MODES`).

The map can be switched on the page; the map style of the export is the initial one. During playback, the
camera follows the moving position ("Follow the position" on the page).
"""

import html
import http.server
import json
import os
import threading
import webbrowser

import numpy as np
import pyproj
from pyproj.transformer import TransformerGroup

from trajectopy.core.trajectory import Trajectory
from trajectopy.io.kml import (
    DEFAULT_COLOR_CLASSES,
    _chronological_wgs84,
    _class_name,
    _when,
    ate_kml_fields,
)
from trajectopy.results.ate_result import ATEResult
from trajectopy.utils.colors import color_range, value_color

CESIUM_VERSION = "1.119.0"
CESIUM_URL = f"https://cdn.jsdelivr.net/npm/cesium@{CESIUM_VERSION}/Build/Cesium"
# "photorealistic" needs a Cesium ion access token (Google Photorealistic 3D Tiles, streamed via Cesium ion)
MAP_STYLES = ("osm", "satellite", "photorealistic")
# How the trajectory is placed vertically:
# "ground": clamped to the terrain / 3D tiles (its heights are ignored),
# "ellipsoidal": at its heights, which are heights above the WGS84 ellipsoid (e.g. GNSS),
# "geoid": at its heights, which are heights above sea level (EGM2008 geoid); they are converted to ellipsoidal
#          heights with the geoid model, which pyproj downloads from the PROJ CDN (cdn.proj.org) where needed.
HEIGHT_MODES = ("ground", "ellipsoidal", "geoid")
# Consecutive poses of one color are split into lines of at most this many poses. Each line appears once the
# position has passed it; the part between the last complete line and the position is drawn by the page itself
# (the "trail"), so that the line ends exactly at the moving position.
MAX_RUN_POSES = 25


def write_trajectory_html(
    trajectory: Trajectory,
    filename: str,
    color: str = "#ff0000",
    map_style: str = "osm",
    ion_token: str = "",
    heights: str = "ground",
) -> None:
    """Exports a trajectory as a web page with 3D playback (CesiumJS).

    Args:
        trajectory (Trajectory): Trajectory to export. Needs a known EPSG code.
        filename (str): The output filename (e.g., "playback.html").
        color (str, optional): Color of the trajectory as "#RRGGBB". Defaults to red.
        map_style (str, optional): "osm" (OpenStreetMap), "satellite" (Esri World Imagery) or "photorealistic"
            (Google Photorealistic 3D Tiles, needs ``ion_token``). Defaults to "osm".
        ion_token (str, optional): Cesium ion access token for terrain and 3D buildings. Defaults to none.
        heights (str, optional): "ground" (clamped to the terrain), "ellipsoidal" (the trajectory's heights are
            ellipsoidal heights, e.g. GNSS) or "geoid" (the heights are above sea level; converted with the EGM2008
            geoid, which needs an internet connection the first time). Defaults to "ground".

    Raises:
        ValueError: If the trajectory has no known EPSG code, the color, map style or heights are invalid, or
            the geoid model is not available.
    """
    rgba = _rgba(color)
    lat_lon, timestamps, order = _chronological_wgs84(trajectory)
    height_values = _heights(trajectory, order, heights)
    classes = np.zeros(len(timestamps), dtype=int)
    czml, trail = _czml(trajectory.name, lat_lon, height_values, timestamps, classes, [rgba], rgba)
    legend = (
        f'<div class="row"><span class="swatch" style="background:{color}"></span>{html.escape(trajectory.name)}</div>'
    )
    _write_page(filename, trajectory.name, czml, trail, legend, map_style, ion_token)


def write_ate_html(
    ate_result: ATEResult,
    filename: str,
    field: str = "position",
    max_std: float = 3.0,
    color_classes: int = DEFAULT_COLOR_CLASSES,
    map_style: str = "osm",
    ion_token: str = "",
    heights: str = "ground",
) -> None:
    """Exports an ATE result as a web page with 3D playback (CesiumJS), colored by one of its deviation fields.

    Args:
        ate_result (ATEResult): Result to export. Its trajectory needs a known EPSG code.
        filename (str): The output filename (e.g., "ate.html").
        field (str, optional): Key of the deviation field that colors the line, see
            :func:`trajectopy.io.kml.ate_kml_fields`. Defaults to "position".
        max_std (float, optional): The color scale is clipped to mean ± ``max_std`` standard deviations
            (0 = minimum to maximum). Defaults to 3.0, like the plots.
        color_classes (int, optional): Number of colors of the scale. Defaults to 11.
        map_style (str, optional): "osm" (OpenStreetMap), "satellite" (Esri World Imagery) or "photorealistic"
            (Google Photorealistic 3D Tiles, needs ``ion_token``). Defaults to "osm".
        ion_token (str, optional): Cesium ion access token for terrain and 3D buildings. Defaults to none.
        heights (str, optional): "ground" (clamped to the terrain), "ellipsoidal" (the trajectory's heights are
            ellipsoidal heights, e.g. GNSS) or "geoid" (the heights are above sea level; converted with the EGM2008
            geoid, which needs an internet connection the first time). Defaults to "ground".

    Raises:
        ValueError: If the trajectory has no known EPSG code, the field, map style or heights are invalid, or
            the geoid model is not available.
    """
    fields = {available.key: available for available in ate_kml_fields(ate_result)}
    if field not in fields:
        raise ValueError(f'Unknown or unavailable field "{field}", choose one of {sorted(fields)}.')
    if color_classes < 1:
        raise ValueError("At least one color class is required.")
    selected = fields[field]

    lat_lon, timestamps, order = _chronological_wgs84(ate_result.trajectory)
    height_values = _heights(ate_result.trajectory, order, heights)
    values = np.asarray(selected.values(ate_result), dtype=float)[order]
    lower, upper = color_range(values, max_std)
    classes = np.clip(((values - lower) / (upper - lower) * color_classes).astype(int), 0, color_classes - 1)
    colors = [value_color((index + 0.5) / color_classes) for index in range(color_classes)]

    czml, trail = _czml(
        ate_result.name, lat_lon, height_values, timestamps, classes, [_rgba(c) for c in colors], [255, 255, 255, 255]
    )
    bounds = np.linspace(lower, upper, color_classes + 1)
    rows = "".join(
        f'<div class="row"><span class="swatch" style="background:{colors[index]}"></span>'
        f"{html.escape(_class_name(bounds, index, selected.unit))}</div>"
        for index in reversed(range(color_classes))
    )
    legend = f"<b>{html.escape(ate_result.name)}</b><div>{html.escape(selected.label)} [{selected.unit}]</div>{rows}"
    _write_page(filename, ate_result.name, czml, trail, legend, map_style, ion_token)


# --------------------------------------------------------------------------------------------- helpers


def _heights(trajectory: Trajectory, order: np.ndarray, mode: str) -> np.ndarray | None:
    """Ellipsoidal heights of the poses ``order`` of ``trajectory``, or None if it is clamped to the ground."""
    if mode not in HEIGHT_MODES:
        raise ValueError(f'Unknown heights "{mode}", choose one of {list(HEIGHT_MODES)}.')
    if mode == "ground":
        return None
    wgs84 = trajectory.positions.to_epsg(4326, inplace=False).xyz[order]
    if mode == "ellipsoidal":
        return wgs84[:, 2]

    network_enabled = pyproj.network.is_network_enabled()
    pyproj.network.set_network_enabled(True)  # the geoid grid is downloaded (and cached) where needed
    try:
        group = TransformerGroup("EPSG:4326+3855", "EPSG:4979", always_xy=True)  # EGM2008 height -> ellipsoidal
        if not group.best_available:
            raise ValueError(
                "The EGM2008 geoid model is not available (it is downloaded from cdn.proj.org on first use). "
                'Check the internet connection, or use heights="ellipsoidal" or "ground".'
            )
        _, _, ellipsoidal = group.transformers[0].transform(wgs84[:, 1], wgs84[:, 0], wgs84[:, 2])
    finally:
        pyproj.network.set_network_enabled(network_enabled)
    return np.asarray(ellipsoidal, dtype=float)


def _czml(
    name: str,
    lat_lon: np.ndarray,
    heights: np.ndarray | None,
    timestamps: np.ndarray,
    classes: np.ndarray,
    class_rgba: list[list[int]],
    point_rgba: list[int],
) -> tuple[list[dict], dict]:
    """CZML of the playback and the data of the trail (see MAX_RUN_POSES)."""
    start, stop = _when(timestamps[0]), _when(timestamps[-1])
    document = {
        "id": "document",
        "name": name,
        "version": "1.0",
        # starts at the end (whole line visible); playing loops back to the start
        "clock": {"interval": f"{start}/{stop}", "currentTime": stop, "multiplier": 1, "range": "LOOP_STOP"},
    }
    clamped = heights is None
    h = np.zeros(len(timestamps)) if heights is None else heights
    samples = np.column_stack([timestamps - timestamps[0], lat_lon[:, 1], lat_lon[:, 0], h])
    point = {
        "id": "position",
        "name": name,
        "availability": f"{start}/{stop}",
        "position": {
            "epoch": start,
            "cartographicDegrees": np.round(samples, 7).ravel().tolist(),
            "interpolationAlgorithm": "LINEAR",
        },
        "point": {
            "pixelSize": 12,
            "color": {"rgba": point_rgba},
            "outlineColor": {"rgba": [40, 40, 40, 255]},
            "outlineWidth": 2,
            "heightReference": "CLAMP_TO_GROUND" if clamped else "NONE",
            "disableDepthTestDistance": 1e9,  # never hidden by terrain or buildings
        },
    }
    lines = []
    runs = _runs(classes)
    for index, (first, last) in enumerate(runs):
        lines.append(
            {
                "id": f"line{index}",
                # appears once the position has passed its end: the line grows with the playback
                "availability": f"{_when(timestamps[last - 1])}/{stop}",
                "polyline": {
                    "positions": {
                        "cartographicDegrees": np.round(
                            np.column_stack([lat_lon[first:last, 1], lat_lon[first:last, 0], h[first:last]]), 7
                        )
                        .ravel()
                        .tolist()
                    },
                    "material": {"solidColor": {"color": {"rgba": class_rgba[classes[first]]}}},
                    "width": 4,
                    # on the ground, not on the roofs of 3D buildings
                    **(
                        {"clampToGround": True, "classificationType": "TERRAIN"}
                        if clamped
                        else {"arcType": "NONE"}  # straight lines between the poses, at their heights
                    ),
                },
            }
        )
    trail = {
        "epoch": start,
        "t": np.round(timestamps - timestamps[0], 3).tolist(),
        "lon": np.round(lat_lon[:, 1], 7).tolist(),
        "lat": np.round(lat_lon[:, 0], 7).tolist(),
        "h": None if heights is None else np.round(heights, 3).tolist(),  # None: clamped to the ground
        "first": [first for first, _ in runs],
        "last": [last for _, last in runs],
        "rgba": [class_rgba[classes[first]] for first, _ in runs],
    }
    return [document, *lines, point], trail


def _runs(classes: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) pose indices (last exclusive) of lines of one class with at most MAX_RUN_POSES poses.
    Consecutive lines share their boundary pose, so that the line has no gaps."""
    runs = []
    changes = np.flatnonzero(np.r_[True, np.diff(classes) != 0])
    ends = np.r_[changes[1:], len(classes)]
    for change, end in zip(changes, ends):
        for first in range(int(change), int(end), MAX_RUN_POSES):
            last = min(first + MAX_RUN_POSES, int(end)) + 1  # connect to the next pose
            if min(last, len(classes)) - first >= 2:
                runs.append((first, min(last, len(classes))))
    return runs


def _rgba(color: str) -> list[int]:
    value = color.strip().removeprefix("#")
    try:
        channels = [int(value[i : i + 2], 16) for i in range(0, len(value), 2)]
    except ValueError:
        channels = []
    if len(value) not in (6, 8) or len(channels) not in (3, 4):
        raise ValueError(f'Invalid color "{color}": use "#RRGGBB" or "#RRGGBBAA", e.g. "#ff0000" for red.')
    return channels + [255] * (4 - len(channels))


def _write_page(
    filename: str, title: str, czml: list[dict], trail: dict, legend: str, map_style: str, ion_token: str
) -> None:
    if map_style not in MAP_STYLES:
        raise ValueError(f'Unknown map style "{map_style}", choose one of {list(MAP_STYLES)}.')
    if map_style == "photorealistic" and not ion_token.strip():
        raise ValueError("Photorealistic 3D tiles need a Cesium ion access token (free at https://ion.cesium.com).")

    def script_json(value) -> str:  # safe inside <script>: no "</script>" can end it early
        return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")

    page = _PAGE.format(
        title=html.escape(title),
        cesium=CESIUM_URL,
        legend=legend,
        czml=script_json(czml),
        trail=script_json(trail),
        token=script_json(ion_token.strip()),
        map_style=script_json(map_style),
    )
    with open(filename, "w", encoding="utf-8") as file:
        file.write(page)


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - trajectopy</title>
<script>window.CESIUM_BASE_URL = "{cesium}/";</script>
<script src="{cesium}/Cesium.js"></script>
<link href="{cesium}/Widgets/widgets.css" rel="stylesheet">
<style>
  html, body, #viewer {{ width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden; }}
  #legend {{ position: absolute; top: 10px; left: 10px; padding: 8px 10px; border-radius: 4px;
            background: rgba(38, 38, 38, 0.85); color: #eee; font: 13px sans-serif; max-width: 320px; }}
  #legend .row {{ display: flex; align-items: center; gap: 6px; margin: 2px 0; }}
  #legend .swatch {{ display: inline-block; width: 16px; height: 10px; border-radius: 2px; }}
  #legend label {{ display: flex; align-items: center; gap: 6px; margin-top: 6px; cursor: pointer; }}
  #legend select {{ font: inherit; }}
  #error {{ position: absolute; top: 50%; width: 100%; text-align: center; color: #c00; font: 15px sans-serif; }}
</style>
</head>
<body>
<div id="viewer"></div>
<div id="legend">{legend}
<label>Map
  <select id="map-style">
    <option value="osm">OpenStreetMap</option>
    <option value="satellite">Satellite</option>
    <option value="photorealistic">Photorealistic 3D (Google)</option>
  </select>
</label>
<label><input type="checkbox" id="follow" checked>Follow the position</label>
</div>
<script>
const czml = {czml};
const trail = {trail};
const ionToken = {token};
const mapStyle = {map_style};

const state = {{ photorealistic: false }};  // the current map; the trail and the camera depend on it

async function main() {{
  const options = {{
    baseLayer: false, baseLayerPicker: false, geocoder: false, infoBox: false, selectionIndicator: false,
    timeline: true, animation: true,
  }};
  if (ionToken) {{
    Cesium.Ion.defaultAccessToken = ionToken;
    options.terrain = Cesium.Terrain.fromWorldTerrain();
  }}
  const viewer = new Cesium.Viewer("viewer", options);
  let osmBuildings, photorealisticTiles;
  if (ionToken) {{
    try {{
      osmBuildings = viewer.scene.primitives.add(await Cesium.createOsmBuildingsAsync());
    }} catch (error) {{
      console.warn("No 3D buildings (check the Cesium ion token):", error);
    }}
  }}

  const source = await Cesium.CzmlDataSource.load(czml);
  await viewer.dataSources.add(source);
  const marker = source.entities.getById("position");
  const trailLine = addTrail(viewer, marker);
  followPosition(viewer, marker);

  // Switches between the maps on the globe (with terrain and 3D buildings if there is a token) and Google's
  // photorealistic 3D tiles, which replace the globe.
  async function setMapStyle(style) {{
    const photorealistic = style === "photorealistic";
    if (photorealistic && !photorealisticTiles) {{
      photorealisticTiles = viewer.scene.primitives.add(await Cesium.createGooglePhotorealistic3DTileset());
    }}
    state.photorealistic = photorealistic;
    if (photorealisticTiles) photorealisticTiles.show = photorealistic;
    if (osmBuildings) osmBuildings.show = !photorealistic;
    viewer.scene.globe.show = !photorealistic;
    viewer.imageryLayers.removeAll();
    if (style === "satellite") {{
      viewer.imageryLayers.add(Cesium.ImageryLayer.fromProviderAsync(Cesium.ArcGisMapServerImageryProvider.fromUrl(
        "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer")));
    }} else if (style === "osm") {{
      viewer.imageryLayers.add(new Cesium.ImageryLayer(
        new Cesium.OpenStreetMapImageryProvider({{ url: "https://tile.openstreetmap.org/" }})));
    }}
    // on the ground, not on roofs; without a globe, on the 3D tiles
    const classification = photorealistic ? Cesium.ClassificationType.CESIUM_3D_TILE : Cesium.ClassificationType.TERRAIN;
    if (trail.h === null) {{  // clamped to the ground
      for (const entity of source.entities.values) {{
        if (entity.polyline) entity.polyline.classificationType = classification;
      }}
    }}
    trailLine.resetHeights();
  }}

  const select = document.getElementById("map-style");
  if (!ionToken) {{
    const option = select.querySelector('option[value="photorealistic"]');
    option.disabled = true;
    option.textContent += " - needs a Cesium ion token";
  }}
  select.value = mapStyle;
  select.addEventListener("change", () => setMapStyle(select.value).catch(showError));
  await setMapStyle(mapStyle);
  await viewer.zoomTo(source);
}}

function showError(error) {{
  console.error(error);
  document.body.insertAdjacentHTML("beforeend", '<div id="error">Could not show the playback: ' + error + "</div>");
}}

function groundHeight(viewer, cartographic, exclude) {{
  return state.photorealistic
    ? viewer.scene.sampleHeight(cartographic, exclude)
    : viewer.scene.globe.getHeight(cartographic);
}}

// The line from the end of the last complete line to the moving position, recomputed every frame. With heights,
// it is at the trajectory's heights. Clamped to the ground, it floats slightly above the ground (heights sampled
// once per pose): a line clamped to the ground would be rebuilt asynchronously every frame and flicker.
function addTrail(viewer, marker) {{
  const epoch = Cesium.JulianDate.fromIso8601(trail.epoch);
  const colors = trail.rgba.map((rgba) => Cesium.Color.fromBytes(...rgba));
  const heights = new Map();
  let color = colors[0];
  let entity;
  const lift = 0.3;  // meters above the ground
  const heightAt = (cartographic) => groundHeight(viewer, cartographic, [marker, entity]);
  const posePosition = (index) => {{
    if (trail.h !== null) return Cesium.Cartesian3.fromDegrees(trail.lon[index], trail.lat[index], trail.h[index]);
    if (!heights.has(index)) {{
      const height = heightAt(Cesium.Cartographic.fromDegrees(trail.lon[index], trail.lat[index]));
      if (height === undefined) return Cesium.Cartesian3.fromDegrees(trail.lon[index], trail.lat[index], lift);
      heights.set(index, height);
    }}
    return Cesium.Cartesian3.fromDegrees(trail.lon[index], trail.lat[index], heights.get(index) + lift);
  }};
  const lastAtOrBefore = (values, value) => {{  // index of the last element <= value (-1 if none)
    let low = 0, high = values.length;
    while (low < high) {{
      const middle = (low + high) >> 1;
      if (values[middle] <= value) low = middle + 1; else high = middle;
    }}
    return low - 1;
  }};
  const positions = new Cesium.CallbackProperty((time) => {{
    const pose = lastAtOrBefore(trail.t, Cesium.JulianDate.secondsDifference(time, epoch));
    const line = lastAtOrBefore(trail.first, pose);
    if (pose < 0 || line < 0 || pose >= trail.last[line] - 1) return [];  // this line is complete (drawn by CZML)
    color = colors[line];
    const result = [];
    for (let index = trail.first[line]; index <= pose; index++) result.push(posePosition(index));
    const current = marker.position.getValue(time);
    if (current && trail.h !== null) {{
      result.push(current);  // at its own height
    }} else if (current) {{
      const cartographic = Cesium.Cartographic.fromCartesian(current);
      cartographic.height = (heightAt(cartographic) ?? 0) + lift;
      result.push(Cesium.Cartographic.toCartesian(cartographic));
    }}
    return result;
  }}, false);
  entity = viewer.entities.add({{
    polyline: {{
      positions: positions,
      width: 4,
      material: new Cesium.ColorMaterialProperty(new Cesium.CallbackProperty(() => color, false)),
    }},
  }});
  return {{ resetHeights: () => heights.clear() }};  // after switching the map: other ground
}}

// The camera follows the position once the playback starts, keeping the heading, pitch and distance chosen by
// the user. It looks at the position on the ground (the data is clamped to the ground; its height is zero) and
// glides after it (exponential smoothing in real time), so that small jumps of the positions or of the ground
// heights (e.g. trees in the 3D tiles) don't shake it. Larger jumps, e.g. on the timeline, are followed directly.
const CAMERA_SMOOTHING_S = 0.6;  // time constant of the smoothing
const CAMERA_JUMP_M = 300;  // the camera jumps instead of gliding if the position moved further at once

function followPosition(viewer, marker) {{
  const checkbox = document.getElementById("follow");
  let following = false;
  let smoothed;  // the point the camera looks at
  let lastHeight;  // last known ground height: used while the ground below the position is not loaded yet
  let lastFrame;
  const release = () => {{
    if (following) viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
    following = false;
    smoothed = undefined;
  }};
  checkbox.addEventListener("change", () => {{ if (!checkbox.checked) release(); }});
  viewer.clock.onTick.addEventListener((clock) => {{
    const now = performance.now();
    const seconds = lastFrame === undefined ? 0 : (now - lastFrame) / 1000;
    lastFrame = now;
    if (!checkbox.checked || (!following && !clock.shouldAnimate)) return;  // starts with the playback
    const position = marker.position.getValue(clock.currentTime);
    if (!position) return;
    let target = position;  // at its own height ...
    if (trail.h === null) {{  // ... or on the ground
      const ground = Cesium.Cartographic.fromCartesian(position);
      const height = groundHeight(viewer, ground, [marker]);
      if (height !== undefined) lastHeight = height;
      ground.height = lastHeight ?? 0;
      target = Cesium.Cartographic.toCartesian(ground);
    }}
    if (smoothed === undefined || Cesium.Cartesian3.distance(smoothed, target) > CAMERA_JUMP_M) {{
      smoothed = target;
    }} else {{
      smoothed = Cesium.Cartesian3.lerp(smoothed, target, 1 - Math.exp(-seconds / CAMERA_SMOOTHING_S), new Cesium.Cartesian3());
    }}
    const view = following
      ? new Cesium.HeadingPitchRange(viewer.camera.heading, viewer.camera.pitch,
          Cesium.Cartesian3.magnitude(viewer.camera.position))
      : new Cesium.HeadingPitchRange(viewer.camera.heading, Cesium.Math.toRadians(-35), 400);
    viewer.camera.lookAt(smoothed, view);
    following = true;
  }});
}}

main().catch(showError);
</script>
</body>
</html>
"""


# ------------------------------------------------------------------------------------------------ viewing

_servers: list[http.server.ThreadingHTTPServer] = []  # keep serving while the application runs


def show_in_browser(filename: str, wait: float = 10.0) -> str:
    """Opens an exported page in the default browser, served by a local web server (127.0.0.1) that only
    serves this file.

    Args:
        filename (str): The exported page.
        wait (float, optional): Seconds to wait until the browser has loaded the page, so that a script may
            end right after this call (the page doesn't need the server once loaded). 0 = don't wait.

    Returns:
        str: The URL of the page.
    """
    path = os.path.abspath(filename)
    loaded = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (name given by BaseHTTPRequestHandler)
            if self.path.split("?")[0] != "/" + os.path.basename(path):
                self.send_error(404)
                return
            with open(path, "rb") as file:
                content = file.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            loaded.set()

        def log_message(self, format: str, *args) -> None:  # noqa: A002 - quiet
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="TrajectopyPlaybackServer").start()
    _servers.append(server)
    url = f"http://127.0.0.1:{server.server_address[1]}/{os.path.basename(path)}"
    webbrowser.open(url)
    if wait > 0:
        loaded.wait(wait)
    return url
