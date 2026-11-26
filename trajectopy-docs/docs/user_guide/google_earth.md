# Google Earth

Georeferenced trajectories (trajectories with an EPSG code other than 0) and ATE results of georeferenced
trajectories can be exported as KML files and opened in Google Earth. It shows them in their real-world context:
on imagery and terrain and, where available, between photorealistic 3D buildings. In the GUI, choose
*Export to Google Earth (KML)…* in the context menu of a trajectory or an ATE result; the dialog offers all
options below except `precision` and remembers them.

The KITTI example data is not georeferenced. For the examples, it is placed in UTM zone 32N (EPSG 25832):

```python
import numpy as np
import trajectopy as tpy


def georeferenced(trajectory: tpy.Trajectory) -> tpy.Trajectory:
    utm = tpy.Positions(trajectory.xyz + np.array([364000.0, 5621000.0, 60.0]), epsg=25832)
    return tpy.Trajectory(utm, trajectory.rotations, trajectory.timestamps, name=trajectory.name)


reference = georeferenced(tpy.Trajectory.from_file("./example_data/KITTI_gt.traj"))
estimate = georeferenced(tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj"))
ate_result = tpy.ate(estimate, reference)

reference.to_kml("track.kml", color="#1f77b4")  # one color: "#RRGGBB" or "#RRGGBBAA"
reference.to_kml("speed.kml", field="speed")
ate_result.to_kml("ate.kml", field="along_track", colormap="viridis", value_range=(-2.0, 2.0))
```

The functions behind these methods are [`trajectopy.io.write_kml`](../api/io.md#trajectopy.io.write_kml) and
[`trajectopy.io.write_ate_kml`](../api/io.md#trajectopy.io.write_ate_kml).

## Playback

The trajectory is exported as a time-stamped track: the time slider of Google Earth plays it back in
chronological order (also for spatially sorted trajectories). Playback needs Google Earth Pro (desktop); Google
Earth on the web shows the trajectory, but has no time slider. During playback, the colored line grows with the
time slider.

## Colors

- Trajectories are drawn in one color (`color`), or colored by a field (`field`): `speed`, `height` or `time`.
- ATE results are colored by one of their deviations (`field`): `position`, `x`, `y`, `z`, `along_track`,
  `horizontal_cross_track`, `vertical_cross_track` (in meters) and, for trajectories with orientations,
  `rotation`, `roll`, `pitch` and `yaw` (in degrees, also for `value_range`).
- `colormap`: `"rdylbu"` (blue = low, red = high, like the plots) or `"viridis"` (colorblind-safe).
- Like in the plots, the color scale is clipped to the mean ± 3 standard deviations (`max_std`; 0 = minimum to
  maximum). With `value_range=(lower, upper)`, it has fixed limits instead, so that the colors of several exports
  are comparable.
- `color_classes`: number of colors (default 11). The Places panel of Google Earth lists them with their value
  ranges (the legend); each class can be switched on and off.
- `line_width`: line width in pixels (default 3 for trajectories, 4 for ATE results).
- `precision`: precision of the coordinates in the file in degrees (default 1e-6, about 0.1 m).
- Without `color` and `field`, trajectories are drawn in red.

## Heights

- `heights="ground"` (default): the trajectory is clamped to the terrain of Google Earth; its own heights are
  ignored. With a `height_offset`, it floats that many meters above the terrain.
- `heights="ellipsoidal"`: the trajectory is drawn at its own heights, which are heights above the WGS84
  ellipsoid (e.g. from GNSS). Google Earth expects heights above sea level, so they are converted with the EGM96
  geoid model, which is downloaded once from cdn.proj.org.
- `heights="sea_level"`: the trajectory is drawn at its own heights, which are already heights above sea level
  (e.g. orthometric or normal heights of a national height system). They are used as they are; the difference
  between national height systems and Google Earth's EGM96 is usually below a meter or two.

With the wrong choice, the trajectory floats above the ground or sinks into it by the geoid height, e.g. about
30 to 50 m in Central Europe (between about −106 and +85 m worldwide). `height_offset`
(meters) is added in all modes, e.g. to correct for the antenna height. `extrude=True` connects the trajectory to
the ground with a semi-transparent vertical curtain, which makes heights easy to read (not for trajectories
clamped to the ground):

<!-- skip-test: downloads the geoid model -->
```python
reference.to_kml("heights.kml", field="height", heights="ellipsoidal", extrude=True)
```
