# Glossary

**Absolute trajectory error (ATE)**
:   The deviations between corresponding poses of a trajectory and a reference after an
    [alignment](alignment.md), and their statistics. It measures the global accuracy of a trajectory. See
    [Evaluation](evaluation.md#absolute-trajectory-error-ate) and [`ate`](../api/evaluation.md#trajectopy.ate).

**Alignment**
:   The transformation between the frames (and clocks) of two trajectories, estimated by least squares from
    corresponding poses: similarity transformation, time shift and leverarm. See [Alignment](alignment.md).

**Along-track, cross-track**
:   Directions relative to the trajectory: along-track is the direction of travel, horizontal cross-track points
    to the right of it, vertical cross-track upwards. See the [frame definition](settings.md#ate-frame-definition).

**Deviation**
:   The difference between a pose of the evaluated trajectory and the corresponding pose of the reference, always
    *evaluated trajectory minus reference*.

**EPSG code**
:   The identifier of a coordinate reference system in the [EPSG registry](https://epsg.io), e.g. 25832 for UTM
    zone 32N or 4326 for latitude and longitude. EPSG code 0 marks a local system without a known datum. See
    [Coordinate Systems](trajectories.md#coordinate-systems).

**Index**
:   The variable a trajectory is a function of: its timestamps, or its path lengths after spatial sorting
    (`Trajectory.index`).

**Leverarm**
:   The constant offset between the reference points of two platforms, e.g. between a GNSS antenna and an IMU,
    given in the body frame.

**Local frame**
:   A tangential east-north-up frame at the mean of georeferenced positions, in which path lengths are computed.

**Matching**
:   Finding the corresponding poses of two trajectories, by interpolation, by the nearest timestamp or by the
    nearest position. See [Matching](alignment.md#matching).

**Path length**
:   The distance travelled since the first pose, in meters (`Trajectory.path_lengths`).

**Pose**
:   A position together with an orientation at one point in time.

**Reference**
:   The trajectory that another one is evaluated against, usually the ground truth. In the API, it is the
    argument `other`.

**Relative pose error (RPE)**
:   The deviations of the relative motion between pairs of poses that are a given distance or time apart,
    normalized by that distance or time. It measures the drift and needs no alignment. See
    [Evaluation](evaluation.md#relative-pose-error-rpe).

**Sensor rotation**
:   A constant rotation between the body frames of two trajectories, e.g. a mounting rotation.

**Similarity transformation**
:   Translation, rotation and scale between two coordinate frames (Helmert transformation); without the scale,
    a rigid body transformation.

**Spatial sorting**
:   Ordering the poses of a trajectory by their position along the path instead of by time, e.g. to compare
    several laps on the same track. See [Time and Path Length](trajectories.md#time-and-path-length).

**Time shift**
:   The offset between the clocks of two trajectories, in seconds.
