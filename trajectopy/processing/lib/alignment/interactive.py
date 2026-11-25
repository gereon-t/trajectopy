"""Building blocks of the interactive alignment (no GUI dependencies).

- :class:`AlignmentState` is one placement of the trajectory: a similarity transformation, a time shift and
  a leverarm. It is immutable; every manipulation (move, rotate about a point, ...) returns a new state.
- :class:`AlignmentPreview` holds both trajectories as plain arrays and evaluates a state quickly: aligned
  positions, the distance to the reference path and the deviation at equal timestamps.
- The functions propose states: from the first pose, the centers, poses matched in time, point pairs picked
  by the user, or the speed profiles (time shift). :func:`best_start` picks the closest of them.

A state converts to :class:`~trajectopy.core.parameters.AlignmentParameters`, e.g. to start the least squares
alignment from it (``estimate_alignment(..., initial_parameters=state.to_parameters())``).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.signal import correlate
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from trajectopy.core.parameters import AlignmentParameters
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.lib.alignment.direct import direct_helmert_transformation

logger = logging.getLogger(__name__)

PREVIEW_POINTS = 5000
MIN_SPEED_CORRELATION_OVERLAP = 0.3  # of the shorter speed profile
SPEED_WINDOW = 1.0  # s, over which the speed is averaged
STANDSTILL_FRACTION = 0.05  # of the typical (95th percentile) speed
STANDSTILL_MIN_SPEED = 0.02  # m/s
STANDSTILL_MAX_SPEED = 0.3  # m/s
STANDSTILL_MIN_DURATION = 1.0  # s
TIME_SYNC_MIN_IMPROVEMENT = 0.5  # factor by which a time shift from the speed profiles must reduce the deviation
LEVERARM_KEYS = ("lever_x", "lever_y", "lever_z")


def _read_only(values, shape: tuple[int, ...]) -> np.ndarray:
    array = np.array(values, dtype=float).reshape(shape)
    array.setflags(write=False)
    return array


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix of a rotation by ``angle`` (rad) about ``axis``."""
    axis = np.asarray(axis, dtype=float)
    return Rotation.from_rotvec(angle * axis / np.linalg.norm(axis)).as_matrix()


@dataclass(frozen=True, eq=False)
class AlignmentState:
    """A placement of the trajectory: ``x' = translation + scale * rotation @ (x + R_body @ leverarm)`` and
    ``t' = t - time_shift`` (the conventions of ``apply_alignment``).

    Attributes:
        rotation: 3x3 rotation matrix.
        translation: Translation [m].
        scale: Scale factor.
        time_shift: Time shift [s], subtracted from the timestamps.
        leverarm: Leverarm in the body frame [m]; only effective for trajectories with orientations.
    """

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    time_shift: float = 0.0
    leverarm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", _read_only(self.rotation, (3, 3)))
        object.__setattr__(self, "translation", _read_only(self.translation, (3,)))
        object.__setattr__(self, "leverarm", _read_only(self.leverarm, (3,)))
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "time_shift", float(self.time_shift))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlignmentState):
            return NotImplemented
        return (
            np.array_equal(self.rotation, other.rotation)
            and np.array_equal(self.translation, other.translation)
            and np.array_equal(self.leverarm, other.leverarm)
            and self.scale == other.scale
            and self.time_shift == other.time_shift
        )

    __hash__ = None  # type: ignore[assignment]

    # ------------------------------------------------------------------------------------------- evaluation

    def apply(self, xyz: np.ndarray, body_rotations: np.ndarray | None = None) -> np.ndarray:
        """Transforms positions (n x 3).

        Args:
            xyz (np.ndarray): Positions of the unaligned trajectory.
            body_rotations (np.ndarray, optional): Orientations of the poses as rotation matrices (n x 3 x 3),
                needed to apply the leverarm. Defaults to None (no leverarm).
        """
        xyz = np.asarray(xyz, dtype=float)
        if body_rotations is not None and np.any(self.leverarm):
            xyz = xyz + body_rotations @ self.leverarm
        return self.translation + self.scale * xyz @ self.rotation.T

    def apply_time(self, timestamps: np.ndarray) -> np.ndarray:
        return np.asarray(timestamps, dtype=float) - self.time_shift

    @property
    def euler(self) -> np.ndarray:
        """Roll, pitch, yaw (rad) of the rotation, in the convention of the alignment parameters."""
        return Rotation.from_matrix(self.rotation).as_euler("xyz")

    # ----------------------------------------------------------------------------------------- manipulation

    def composed(
        self, rotation: np.ndarray | None = None, translation: np.ndarray | None = None, scale: float = 1.0
    ) -> "AlignmentState":
        """Applies ``x'' = translation + scale * rotation @ x'`` on top of this placement."""
        rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
        return replace(
            self,
            rotation=rotation @ self.rotation,
            translation=translation + scale * rotation @ self.translation,
            scale=scale * self.scale,
        )

    def moved(self, delta: np.ndarray) -> "AlignmentState":
        return self.composed(translation=delta)

    def rotated_about(self, point: np.ndarray, rotation: np.ndarray) -> "AlignmentState":
        """Rotates the placed trajectory by ``rotation`` about ``point`` (given where it is shown)."""
        point = np.asarray(point, dtype=float)
        return self.composed(rotation=rotation, translation=point - rotation @ point)

    def scaled_about(self, point: np.ndarray, factor: float) -> "AlignmentState":
        """Scales the placed trajectory by ``factor`` about ``point`` (given where it is shown)."""
        point = np.asarray(point, dtype=float)
        return self.composed(translation=(1.0 - factor) * point, scale=factor)

    def with_rotation(self, rotation: np.ndarray, fixed_point: np.ndarray) -> "AlignmentState":
        """Sets the rotation, keeping ``fixed_point`` (given where it is shown) in place."""
        return self.rotated_about(fixed_point, np.asarray(rotation, dtype=float) @ self.rotation.T)

    def with_scale(self, scale: float, fixed_point: np.ndarray) -> "AlignmentState":
        """Sets the scale, keeping ``fixed_point`` (given where it is shown) in place."""
        return self.scaled_about(fixed_point, scale / self.scale)

    def with_time_shift(self, time_shift: float) -> "AlignmentState":
        return replace(self, time_shift=time_shift)

    def with_leverarm(self, leverarm: np.ndarray) -> "AlignmentState":
        return replace(self, leverarm=leverarm)

    # ------------------------------------------------------------------------------------------- conversion

    def to_parameters(self) -> AlignmentParameters:
        """Alignment parameters of this placement, all enabled (so that they are applied)."""
        roll, pitch, yaw = self.euler
        values = {
            "sim_trans_x": self.translation[0],
            "sim_trans_y": self.translation[1],
            "sim_trans_z": self.translation[2],
            "sim_rot_x": roll,
            "sim_rot_y": pitch,
            "sim_rot_z": yaw,
            "sim_scale": self.scale,
            "time_shift": self.time_shift,
        } | dict(zip(LEVERARM_KEYS, self.leverarm, strict=True))
        parameters = AlignmentParameters()
        for name, value in values.items():
            parameter = getattr(parameters, name)
            parameter.value = float(value)
            parameter.enabled = True
        return parameters

    @classmethod
    def from_parameters(cls, parameters: AlignmentParameters) -> "AlignmentState":
        helmert = parameters.helmert
        return cls(
            rotation=helmert.rotation_matrix,
            translation=helmert.translation,
            scale=helmert.scale.value,
            time_shift=parameters.time_shift.value,
            leverarm=np.array([getattr(parameters, key).value for key in LEVERARM_KEYS]),
        )


def _interpolate(epochs: np.ndarray, timestamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(epochs, timestamps, values[:, i]) for i in range(values.shape[1])])


def velocity(timestamps: np.ndarray, xyz: np.ndarray, window: float = SPEED_WINDOW) -> np.ndarray:
    """Velocity [m/s] at each pose, from the positions ``window / 2`` seconds before and after it.

    Averaging over a window keeps the noise of the positions (e.g. of GNSS at a standstill) from appearing as
    motion; between poses that are further apart than the window, it is the velocity of the neighbours.
    """
    if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
        return np.zeros((len(timestamps), 3))
    half = max(0.5 * window, 0.5 * float(np.median(np.diff(timestamps))))
    before = np.clip(timestamps - half, timestamps[0], timestamps[-1])
    after = np.clip(timestamps + half, timestamps[0], timestamps[-1])
    duration = np.maximum(after - before, 1e-12)
    return (_interpolate(after, timestamps, xyz) - _interpolate(before, timestamps, xyz)) / duration[:, None]


def travel_directions(velocities: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Horizontal unit direction of travel at each pose. Where the platform does not move, the direction of the
    last (or, at the start, the first) movement is used."""
    directions = np.zeros((len(velocities), 3))
    if not np.any(moving):
        directions[:, 0] = 1.0
        return directions
    horizontal = velocities[:, :2] / np.maximum(np.linalg.norm(velocities[:, :2], axis=1), 1e-12)[:, None]
    index = np.where(moving, np.arange(len(moving)), -1)
    index = np.maximum.accumulate(index)
    index[index < 0] = np.flatnonzero(moving)[0]
    directions[:, :2] = horizontal[index]
    return directions


def standstill_threshold(speed: np.ndarray) -> float:
    """Speed [m/s] below which the platform is regarded as standing still."""
    if len(speed) == 0:
        return STANDSTILL_MIN_SPEED
    typical = float(np.percentile(speed, 95))
    return float(np.clip(STANDSTILL_FRACTION * typical, STANDSTILL_MIN_SPEED, STANDSTILL_MAX_SPEED))


def standstills(
    timestamps: np.ndarray, speed: np.ndarray, min_duration: float = STANDSTILL_MIN_DURATION
) -> np.ndarray:
    """Poses where the platform stands still for at least ``min_duration`` seconds."""
    still = speed < standstill_threshold(speed)
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(int), 0]))
    result = np.zeros(len(still), dtype=bool)
    for start, stop in zip(edges[0::2], edges[1::2], strict=True):
        if timestamps[stop - 1] - timestamps[start] >= min_duration:
            result[start:stop] = True
    return result


class AlignmentPreview:
    """Both trajectories as arrays (sorted by time, thinned out to ``max_points`` poses) and the evaluation of
    alignment states on them. Everything here is fast enough to be recomputed while the user drags.
    """

    def __init__(self, trajectory: Trajectory, reference: Trajectory, max_points: int = PREVIEW_POINTS) -> None:
        self.t, self.xyz, self.body_rotations = self._arrays(trajectory, max_points)
        self.reference_t, self.reference_xyz, _ = self._arrays(reference, max_points)
        if len(self.t) == 0 or len(self.reference_t) == 0:
            raise ValueError("Cannot align empty trajectories.")
        self.speed = np.linalg.norm(velocity(self.t, self.xyz), axis=1)
        reference_velocity = velocity(self.reference_t, self.reference_xyz)
        self.reference_speed = np.linalg.norm(reference_velocity, axis=1)
        self.reference_standstill = standstills(self.reference_t, self.reference_speed)
        moving = self.reference_speed >= standstill_threshold(self.reference_speed)
        self.reference_direction = travel_directions(reference_velocity, moving)
        self.has_orientation = self.body_rotations is not None
        self.center = np.mean(self.xyz, axis=0)
        self.extent = max(
            float(np.max(np.ptp(self.xyz, axis=0))), float(np.max(np.ptp(self.reference_xyz, axis=0))), 1.0
        )
        self._reference_tree = KDTree(self.reference_xyz)

    @staticmethod
    def _arrays(trajectory: Trajectory, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        timestamps = np.asarray(trajectory.timestamps, dtype=float)
        order = np.argsort(timestamps, kind="stable")
        order = order[:: max(int(np.ceil(len(order) / max_points)), 1)]
        rotations = trajectory.rotations if trajectory.has_orientation else None
        return (
            timestamps[order],
            np.asarray(trajectory.positions.xyz, dtype=float)[order],
            None if rotations is None else np.asarray(rotations.as_matrix())[order],
        )

    def positions(self, state: AlignmentState) -> np.ndarray:
        """Positions of the trajectory placed with ``state``."""
        return state.apply(self.xyz, self.body_rotations)

    def source_points(self, state: AlignmentState) -> np.ndarray:
        """Positions before the similarity transformation (i.e. with the leverarm only)."""
        return AlignmentState(leverarm=state.leverarm).apply(self.xyz, self.body_rotations)

    def timestamps(self, state: AlignmentState) -> np.ndarray:
        return state.apply_time(self.t)

    def nearest_distances(self, state: AlignmentState) -> np.ndarray:
        """Distance of each pose to the nearest reference position. Does not depend on the timestamps, so it
        also judges placements while the time offset is still unknown."""
        distances, _ = self._reference_tree.query(self.positions(state))
        return distances

    def nearest_rms(self, state: AlignmentState) -> float:
        return float(np.sqrt(np.mean(self.nearest_distances(state) ** 2)))

    def nearest_reference_point(self, point: np.ndarray) -> np.ndarray:
        _, index = self._reference_tree.query(point)
        return self.reference_xyz[index]

    def time_matches(self, state: AlignmentState) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Poses at equal timestamps: the reference epochs within the (shifted) time span of the trajectory,
        the trajectory interpolated at them and the reference positions. None without temporal overlap."""
        t = self.timestamps(state)
        valid = (self.reference_t >= t[0]) & (self.reference_t <= t[-1])
        if not np.any(valid) or len(t) < 2:
            return None
        epochs = self.reference_t[valid]
        return epochs, self.positions_at(state, epochs), self.reference_xyz[valid]

    def overlap(self, state: AlignmentState) -> np.ndarray:
        """Indices of the reference poses within the (shifted) time span of the trajectory."""
        t = self.timestamps(state)
        return np.flatnonzero((self.reference_t >= t[0]) & (self.reference_t <= t[-1]))

    def positions_at(self, state: AlignmentState, epochs: np.ndarray) -> np.ndarray:
        """Positions of the placed trajectory at ``epochs`` (in the time of the reference), interpolated."""
        return _interpolate(epochs, self.timestamps(state), self.positions(state))

    def body_rotations_at(self, state: AlignmentState, epochs: np.ndarray) -> np.ndarray | None:
        """Orientations (n x 3 x 3) of the placed trajectory's body frame at ``epochs`` (nearest pose)."""
        if self.body_rotations is None:
            return None
        t = self.timestamps(state)
        index = np.clip(np.searchsorted(t, epochs), 1, len(t) - 1)
        index -= (epochs - t[index - 1]) < (t[index] - epochs)
        return state.rotation @ self.body_rotations[index]

    def deviations(self, state: AlignmentState) -> tuple[np.ndarray, np.ndarray] | None:
        """Reference epochs and the distance to the reference at these epochs (None without overlap)."""
        if (matches := self.time_matches(state)) is None:
            return None
        epochs, xyz, reference_xyz = matches
        return epochs, np.linalg.norm(xyz - reference_xyz, axis=1)


# ---------------------------------------------------------------------------------------------- proposals


def _heading(xyz: np.ndarray, distance: float) -> float | None:
    """Direction of travel (rad, from the x axis) over the first ``distance`` meters."""
    travelled = np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1)
    beyond = np.flatnonzero(travelled >= distance)
    if len(beyond) == 0:
        return None
    delta = xyz[beyond[0], :2] - xyz[0, :2]
    return float(np.arctan2(delta[1], delta[0]))


def match_first_pose(preview: AlignmentPreview, state: AlignmentState) -> AlignmentState:
    """Puts the first position onto the reference's first position and aligns the initial directions of
    travel. Does not use the timestamps."""
    xyz = preview.positions(state)
    distance = max(0.02 * preview.extent, 2.0)
    heading, reference_heading = _heading(xyz, distance), _heading(preview.reference_xyz, distance)
    yaw = 0.0 if heading is None or reference_heading is None else reference_heading - heading
    rotation = axis_rotation(np.array([0.0, 0.0, 1.0]), yaw)
    return state.composed(rotation=rotation, translation=preview.reference_xyz[0] - rotation @ xyz[0])


def match_centers(preview: AlignmentPreview, state: AlignmentState) -> AlignmentState:
    """Moves the trajectory so that its center coincides with the reference's center."""
    return state.moved(np.mean(preview.reference_xyz, axis=0) - np.mean(preview.positions(state), axis=0))


def fit_point_pairs(
    source: np.ndarray, target: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Similarity transformation ``target ≈ translation + scale * rotation @ source`` from corresponding points.

    What is determined depends on the number of pairs: one pair gives the translation, two pairs (or more that
    lie on a line) add the heading (rotation about the vertical axis), three or more give the full rotation.

    Args:
        source (np.ndarray): Points to transform (n x 3).
        target (np.ndarray): Corresponding points (n x 3).
        with_scale (bool, optional): Also determine the scale (needs two pairs). Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Rotation matrix, translation and scale.
    """
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    if len(source) == 0 or len(source) != len(target):
        raise ValueError("Corresponding points are required.")

    center_source, center_target = np.mean(source, axis=0), np.mean(target, axis=0)
    centered_source, centered_target = source - center_source, target - center_target
    rotation, scale = np.eye(3), 1.0
    spread = np.linalg.svd(centered_source, compute_uv=False)
    if len(source) >= 2 and spread[0] > 0.0:
        if len(source) >= 3 and spread[1] > 1e-6 * spread[0]:
            rotation = direct_helmert_transformation(source, target).rotation_matrix
        else:  # the points lie on a line, which leaves the rotation about it open: only turn the heading
            a, b = centered_source[:, :2], centered_target[:, :2]
            yaw = np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b))
            rotation = axis_rotation(np.array([0.0, 0.0, 1.0]), float(yaw))
        if with_scale:
            fitted = np.sum((centered_source @ rotation.T) * centered_target) / np.sum(centered_source**2)
            scale = float(fitted) if fitted > 0.0 else 1.0
    return rotation, center_target - scale * rotation @ center_source, scale


def fit_time_matches(
    preview: AlignmentPreview, state: AlignmentState, with_scale: bool = False
) -> AlignmentState | None:
    """Closed-form fit of the placement to the reference using the poses at equal timestamps.

    Returns:
        AlignmentState | None: None if fewer than three poses overlap in time.
    """
    matches = preview.time_matches(state)
    if matches is None or len(matches[0]) < 3:
        return None
    try:
        rotation, translation, scale = fit_point_pairs(matches[1], matches[2], with_scale)
    except ValueError as error:
        logger.debug("Fit on matched poses not possible: %s", error)
        return None
    return state.composed(rotation=rotation, translation=translation, scale=scale)


def _overlap_correlation(moving: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pearson correlation of the overlapping parts of two signals for every lag (``moving[n + lag]`` against
    ``reference[n]``), computed with the FFT and cumulative sums.

    Each overlap is normalized on its own: normalizing by the whole signals favours short overlaps of
    above-average values.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Correlation per lag (-2 where unusable), whether the overlap
            is long enough, and the lags.
    """
    n_m, n_r = len(moving), len(reference)
    lags = np.arange(-(n_r - 1), n_m)
    # overlap of lag L: reference[a:b] with moving[a + L:b + L]
    start = np.maximum(0, -lags)
    end = np.minimum(n_r, n_m - lags)
    count = end - start
    usable = count >= max(10.0, MIN_SPEED_CORRELATION_OVERLAP * min(n_m, n_r))

    def window_sums(values: np.ndarray, begin: np.ndarray, stop: np.ndarray) -> np.ndarray:
        cumulative = np.r_[0.0, np.cumsum(values)]
        return cumulative[stop] - cumulative[begin]

    sum_r, sum_rr = window_sums(reference, start, end), window_sums(reference**2, start, end)
    sum_m, sum_mm = window_sums(moving, start + lags, end + lags), window_sums(moving**2, start + lags, end + lags)
    cross = correlate(moving, reference, mode="full", method="fft")
    safe_count = np.maximum(count, 1)
    covariance = cross - sum_m * sum_r / safe_count
    variance = (sum_mm - sum_m**2 / safe_count) * (sum_rr - sum_r**2 / safe_count)
    valid = usable & (variance > 1e-12 * safe_count**2)
    score = np.where(valid, covariance / np.sqrt(np.where(valid, variance, 1.0)), -2.0)
    return score, valid, lags


@dataclass(frozen=True)
class TimeShiftEstimate:
    """Time shift for which the speed profiles agree best, and their correlation there (1 = identical)."""

    time_shift: float
    correlation: float


def estimate_time_shift(preview: AlignmentPreview) -> TimeShiftEstimate | None:
    """Estimates the time shift by correlating the speed profiles of both trajectories.

    The speed does not depend on the coordinate frame, so this works before the trajectories are placed on
    each other (and regardless of their scale). It needs a speed that varies: stops, turns, accelerations.

    Returns:
        TimeShiftEstimate | None: None if a speed profile is constant or too short.
    """
    steps = [np.median(np.diff(t)) for t in (preview.t, preview.reference_t) if len(t) > 1]
    if len(steps) < 2 or min(steps) <= 0:
        return None
    span = (preview.t[-1] - preview.t[0]) + (preview.reference_t[-1] - preview.reference_t[0])
    step = max(float(min(steps)), span / 2**17)

    def profile(t: np.ndarray, speed: np.ndarray) -> np.ndarray | None:
        grid = np.arange(t[0], t[-1], step)
        if len(grid) < 10:
            return None
        values = np.interp(grid, t, speed)
        deviation = np.std(values)
        return None if deviation < 1e-9 else (values - np.mean(values)) / deviation

    moving, reference = profile(preview.t, preview.speed), profile(preview.reference_t, preview.reference_speed)
    if moving is None or reference is None:
        return None

    score, usable, lags = _overlap_correlation(moving, reference)
    if not np.any(usable):
        return None
    best = int(np.argmax(score))
    lag = float(lags[best])
    if 0 < best < len(score) - 1 and usable[best - 1] and usable[best + 1]:  # parabola through the peak
        left, peak, right = score[best - 1 : best + 2]
        if (curvature := left - 2 * peak + right) < 0:
            lag += 0.5 * (left - right) / curvature
    # moving[n + lag] matches reference[n]
    time_shift = (preview.t[0] - preview.reference_t[0]) + lag * step
    return TimeShiftEstimate(time_shift=float(time_shift), correlation=float(np.clip(score[best], -1.0, 1.0)))


def best_start(
    preview: AlignmentPreview, state: AlignmentState, with_scale: bool = False
) -> tuple[str, AlignmentState]:
    """Tries several rough placements and returns the one closest to the reference (by the distance to the
    nearest reference positions, which does not depend on the timestamps), with its description."""

    def fitted(candidate: AlignmentState) -> AlignmentState:
        return fit_time_matches(preview, candidate, with_scale) or candidate

    def time_synced() -> AlignmentState:
        estimate = estimate_time_shift(preview)
        return state if estimate is None else fitted(state.with_time_shift(estimate.time_shift))

    strategies: dict[str, Callable[[], AlignmentState]] = {
        "current placement": lambda: state,
        "first pose": lambda: match_first_pose(preview, state),
        "fit on poses matched in time": lambda: fitted(state),
        "first pose, then fit": lambda: fitted(match_first_pose(preview, state)),
        "centers, then fit": lambda: fitted(match_centers(preview, state)),
        "time sync, then fit": time_synced,
    }
    placements = [(name, strategy()) for name, strategy in strategies.items()]
    candidates = [(name, candidate, preview.nearest_rms(candidate)) for name, candidate in placements]
    # among (practically) equally good starts, prefer the simplest one (they are ordered so)
    tolerance = max(1e-9, 1e-6 * preview.extent)
    best = min(distance for _, _, distance in candidates)
    name, chosen, _ = next(candidate for candidate in candidates if candidate[2] <= best + tolerance)

    # the distance to the path says nothing about the time: adopt the time shift of the speed profiles if the
    # poses at equal timestamps are much closer to the reference with it (or only overlap in time with it).
    # Not for small gains: a time shift that is not estimated afterwards would bias the other parameters.
    def deviation(candidate: AlignmentState) -> float:
        deviations = preview.deviations(candidate)
        return float("inf") if deviations is None else float(np.sqrt(np.mean(deviations[1] ** 2)))

    if (estimate := estimate_time_shift(preview)) is not None:
        synced = chosen.with_time_shift(estimate.time_shift)
        if deviation(synced) < TIME_SYNC_MIN_IMPROVEMENT * deviation(chosen):
            name, chosen = f"{name}, time from the speed profiles", synced
    return name, chosen
