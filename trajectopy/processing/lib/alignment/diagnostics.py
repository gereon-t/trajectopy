"""Diagnosis of the deviations that remain after placing a trajectory on a reference.

Every alignment parameter leaves its own signature in the deviation (trajectory minus reference at equal time):

=================  ======================================================================================
Parameter          Signature
=================  ======================================================================================
time shift         along-track deviation proportional to the speed (zero at standstills)
leverarm           constant in the body frame, independent of the speed
shift              constant in the world frame (in the body frame: varies with the heading)
heading (yaw)      tangential deviation (about the pivot) proportional to the distance from the pivot
scale              radial deviation proportional to the distance from the pivot
roll, pitch        vertical deviation proportional to the horizontal offset from the pivot
=================  ======================================================================================

:func:`residuals` splits the deviation into these components; :func:`diagnose` determines for every
parameter how much of the deviation it explains, and :func:`fit` corrects parameters (Gauss-Newton on the
preview). Rotations and the scale act about the pivot, the shift moves it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from trajectopy.processing.lib.alignment.interactive import AlignmentPreview, AlignmentState, axis_rotation

MIN_EPOCHS = 10
INSEPARABLE_CORRELATION = 0.999  # parameters whose effects correlate more cannot be told apart
MIN_SIGNIFICANCE = 3.0  # t value: a correction must explain this many times what noise would
MAX_AUTOCORRELATION = 0.999
CONVERGED_MOVEMENT = 1e-6  # m: the fit has converged when a step moves the trajectory less than this
WEAK_INDEPENDENCE = 0.1  # parameters whose effect is less independent of the others are weakly determined
_UNIT = np.eye(3)
_UP = np.array([0.0, 0.0, 1.0])

VIEW_TIME = "time"
VIEW_STANDSTILL = "standstill"
VIEW_GEOMETRY = "geometry"


@dataclass(frozen=True)
class ParameterInfo:
    key: str
    name: str
    view: str  # diagnostic view that shows the signature of the parameter
    step: float  # for the numerical derivative (internal unit)
    needs_orientation: bool = False


PARAMETERS: dict[str, ParameterInfo] = {
    info.key: info
    for info in (
        ParameterInfo("shift_x", "Shift x", VIEW_STANDSTILL, 1.0),
        ParameterInfo("shift_y", "Shift y", VIEW_STANDSTILL, 1.0),
        ParameterInfo("shift_z", "Shift z", VIEW_STANDSTILL, 1.0),
        ParameterInfo("rot_x", "Roll (about x)", VIEW_GEOMETRY, 1e-5),
        ParameterInfo("rot_y", "Pitch (about y)", VIEW_GEOMETRY, 1e-5),
        ParameterInfo("rot_z", "Heading (about z)", VIEW_GEOMETRY, 1e-5),
        ParameterInfo("scale", "Scale", VIEW_GEOMETRY, 1e-5),
        ParameterInfo("time_shift", "Time shift", VIEW_TIME, 1e-3),
        ParameterInfo("lever_x", "Leverarm x (body)", VIEW_STANDSTILL, 1.0, needs_orientation=True),
        ParameterInfo("lever_y", "Leverarm y (body)", VIEW_STANDSTILL, 1.0, needs_orientation=True),
        ParameterInfo("lever_z", "Leverarm z (body)", VIEW_STANDSTILL, 1.0, needs_orientation=True),
    )
}


def available_parameters(preview: AlignmentPreview) -> tuple[str, ...]:
    return tuple(key for key, info in PARAMETERS.items() if preview.has_orientation or not info.needs_orientation)


def perturbed(state: AlignmentState, key: str, amount: float, pivot: np.ndarray) -> AlignmentState:
    """Changes one parameter by ``amount``: shifts [m], rotations about the pivot [rad], the scale about the pivot
    (relative: 1 + amount), the time shift [s] or the leverarm [m]."""
    axis = "xyz".index(key[-1]) if key[-2] == "_" and key[-1] in "xyz" else -1
    if key.startswith("shift_"):
        return state.moved(amount * _UNIT[axis])
    if key.startswith("rot_"):
        return state.rotated_about(pivot, axis_rotation(_UNIT[axis], amount))
    if key == "scale":
        return state.scaled_about(pivot, 1.0 + amount)
    if key == "time_shift":
        return state.with_time_shift(state.time_shift + amount)
    if key.startswith("lever_"):
        return state.with_leverarm(state.leverarm + amount * _UNIT[axis])
    raise KeyError(key)


# ---------------------------------------------------------------------------------------------- residuals


@dataclass(frozen=True)
class Residuals:
    """Deviations of the placed trajectory from the reference at the reference epochs it overlaps in time.

    All arrays have one row per epoch. Directions refer to the reference: along-track is its (horizontal)
    direction of travel, cross-track points to the left of it.
    """

    t: np.ndarray
    deviation: np.ndarray  # trajectory minus reference, world frame (n x 3)
    reference_xyz: np.ndarray
    speed: np.ndarray  # of the reference
    standstill: np.ndarray  # bool
    along: np.ndarray
    cross: np.ndarray
    vertical: np.ndarray
    pivot: np.ndarray
    radius: np.ndarray  # horizontal distance of the reference position from the pivot
    radial: np.ndarray  # deviation away from the pivot (horizontal)
    tangential: np.ndarray  # deviation counter-clockwise about the pivot (horizontal)
    body_deviation: np.ndarray | None  # in the body frame of the placed trajectory, divided by its scale
    body_heading: np.ndarray | None  # heading [rad] of the placed trajectory's body x axis

    @property
    def count(self) -> int:
        return len(self.t)

    @property
    def distance(self) -> np.ndarray:
        return np.linalg.norm(self.deviation, axis=1)

    @staticmethod
    def rms_of(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0

    @property
    def rms(self) -> float:
        return self.rms_of(self.distance)


def residuals(
    preview: AlignmentPreview, state: AlignmentState, pivot: np.ndarray, deviation: np.ndarray | None = None
) -> Residuals | None:
    """The deviations of ``state`` split into their components; None if fewer than a few epochs overlap.

    Args:
        deviation (np.ndarray, optional): Deviations to use instead of the actual ones, e.g. partial residuals
            (see :func:`partial_residuals`). One row per overlapping reference epoch.
    """
    index = preview.overlap(state)
    if len(index) < MIN_EPOCHS:
        return None
    t, reference_xyz = preview.reference_t[index], preview.reference_xyz[index]
    if deviation is None:
        deviation = preview.positions_at(state, t) - reference_xyz

    along_direction = preview.reference_direction[index]
    cross_direction = np.cross(_UP, along_direction)

    offset = reference_xyz[:, :2] - np.asarray(pivot)[:2]
    radius = np.linalg.norm(offset, axis=1)
    radial_direction = offset / np.maximum(radius, 1e-12)[:, None]
    tangential_direction = np.column_stack((-radial_direction[:, 1], radial_direction[:, 0]))

    body_deviation = body_heading = None
    if (body := preview.body_rotations_at(state, t)) is not None:
        body_deviation = np.einsum("nji,nj->ni", body, deviation) / state.scale
        body_heading = np.arctan2(body[:, 1, 0], body[:, 0, 0])

    return Residuals(
        t=t,
        deviation=deviation,
        reference_xyz=reference_xyz,
        speed=preview.reference_speed[index],
        standstill=preview.reference_standstill[index],
        along=np.sum(deviation * along_direction, axis=1),
        cross=np.sum(deviation * cross_direction, axis=1),
        vertical=deviation[:, 2],
        pivot=np.asarray(pivot, dtype=float),
        radius=radius,
        radial=np.sum(deviation[:, :2] * radial_direction, axis=1),
        tangential=np.sum(deviation[:, :2] * tangential_direction, axis=1),
        body_deviation=body_deviation,
        body_heading=body_heading,
    )


# ------------------------------------------------------------------------------------------------ diagnosis


def _jacobian(
    preview: AlignmentPreview, state: AlignmentState, pivot: np.ndarray, epochs: np.ndarray, keys: Sequence[str]
) -> np.ndarray:
    """Derivatives of the deviation (3n, flattened) with respect to the parameters (central differences)."""
    columns = []
    for key in keys:
        step = PARAMETERS[key].step
        ahead = preview.positions_at(perturbed(state, key, step, pivot), epochs)
        behind = preview.positions_at(perturbed(state, key, -step, pivot), epochs)
        columns.append(((ahead - behind) / (2 * step)).ravel())
    return np.column_stack(columns)


@dataclass(frozen=True)
class Correction:
    """One parameter of the joint diagnosis.

    Attributes:
        key: Parameter.
        value: Its correction, in the unit of :func:`perturbed` (m, rad, relative scale, s), determined together
            with all other parameters, so that it is not distorted by effects that only look similar.
        explained: RMS [m] of the part of the deviation that only this parameter explains (how much larger the
            remaining deviation would be if it was left out of the joint correction).
        rms_before: RMS deviation [m] now.
        rms_after: RMS deviation [m] after correcting this parameter alone (by ``value``).
        independence: Fraction (0 to 1) of the parameter's effect that differs from the effects of the other
            parameters. Small values mean that the correction is weakly determined: a different combination of
            parameters fits almost as well (e.g. a leverarm and a shift without standstills at different headings).
        significance: How clearly the parameter explains more than noise would by chance: ``explained`` in units
            of what pure noise of the remaining deviation would explain (a t value). The remaining deviation of
            trajectories is correlated from pose to pose, which is taken into account.
    """

    key: str
    value: float
    explained: float
    rms_before: float
    rms_after: float
    independence: float = 1.0
    significance: float = float("inf")

    @property
    def significant(self) -> bool:
        """Whether the parameter explains clearly more of the deviation than noise would."""
        return self.significance >= MIN_SIGNIFICANCE

    @property
    def weakly_determined(self) -> bool:
        return self.independence < WEAK_INDEPENDENCE

    @property
    def name(self) -> str:
        return PARAMETERS[self.key].name

    @property
    def view(self) -> str:
        return PARAMETERS[self.key].view

    @property
    def share(self) -> float:
        """``explained`` relative to the current RMS deviation."""
        return self.explained / self.rms_before if self.rms_before > 0 else 0.0


@dataclass(frozen=True)
class Potential:
    """A parameter that is not part of the diagnosis, and the RMS deviation if it was added to it."""

    key: str
    rms_after: float

    @property
    def name(self) -> str:
        return PARAMETERS[self.key].name


@dataclass(frozen=True)
class Diagnosis:
    """Which parameters explain the deviation (determined jointly), most important first."""

    corrections: tuple[Correction, ...]
    inseparable: tuple[tuple[str, str], ...]  # (left out, cannot be told apart from)
    rms: float
    rms_after_all: float  # after correcting all parameters jointly (linearized)
    # the part of the deviation (n x 3, at the overlapping reference epochs) caused by each parameter's error
    effects: dict[str, np.ndarray] = field(default_factory=dict)
    # parameters that were not considered, by how much they would reduce the deviation further (best first)
    potential: tuple[Potential, ...] = ()

    def correction(self, key: str) -> Correction | None:
        return next((correction for correction in self.corrections if correction.key == key), None)


def _remaining_rms(scaled: np.ndarray, residual: np.ndarray, count: int) -> float:
    """RMS deviation after the least squares correction of the (normalized) columns."""
    solution = np.linalg.lstsq(scaled, -residual, rcond=1e-12)[0]
    remaining = residual + scaled @ solution
    return float(np.sqrt(remaining @ remaining / count))


def diagnose(
    preview: AlignmentPreview,
    state: AlignmentState,
    pivot: np.ndarray,
    keys: Sequence[str] | None = None,
    others: Sequence[str] | None = None,
) -> Diagnosis | None:
    """Determines the corrections of the parameters ``keys`` jointly (linearized) and how much each one explains
    compared to the others in ``keys``.

    Parameters whose effect cannot be told apart from that of an earlier one in ``keys`` are left out.

    Args:
        keys (Sequence[str], optional): Parameters to consider, e.g. those that are estimated. Defaults to all
            available ones.
        others (Sequence[str], optional): Parameters that are not considered, but for which the diagnosis tells
            how much they would reduce the deviation in addition (:attr:`Diagnosis.potential`). Defaults to all
            available parameters not in ``keys``.

    Returns:
        Diagnosis | None: None without enough poses at equal time.
    """
    available = available_parameters(preview)
    keys = [key for key in (available if keys is None else keys) if key in available]
    others = [key for key in (available if others is None else others) if key in available and key not in keys]
    index = preview.overlap(state)
    if len(index) < MIN_EPOCHS:
        return None
    epochs = preview.reference_t[index]
    count = len(epochs)
    residual = (preview.positions_at(state, epochs) - preview.reference_xyz[index]).ravel()
    squared = float(residual @ residual)
    rms = float(np.sqrt(squared / count))
    if not keys and not others:
        return Diagnosis((), (), rms, rms, {})
    full_jacobian = _jacobian(preview, state, pivot, epochs, [*keys, *others])
    jacobian, other_jacobian = full_jacobian[:, : len(keys)], full_jacobian[:, len(keys) :]
    columns, inseparable = _separable(jacobian, keys)
    keys, jacobian = [keys[i] for i in columns], jacobian[:, columns]
    potential = _potential(jacobian, other_jacobian, others, residual, count)
    if not keys:
        return Diagnosis((), tuple(inseparable), rms, rms, {}, potential)

    # scaled columns keep the normal matrix well conditioned (the units differ by orders of magnitude)
    norms = np.linalg.norm(jacobian, axis=0)
    scaled = jacobian / norms
    normal = scaled.T @ scaled
    inverse = np.linalg.pinv(normal)
    solution = -inverse @ (scaled.T @ residual)
    remaining = residual + scaled @ solution
    rms_after_all = float(np.sqrt(remaining @ remaining / count))
    noise_variance = _noise_variance(remaining.reshape(-1, 3), len(keys))

    corrections = []
    effects = {key: -(scaled[:, i] * solution[i]).reshape(-1, 3) for i, key in enumerate(keys)}
    for i, key in enumerate(keys):
        # increase of the remaining sum of squares if the parameter was left out: solution_i^2 / (N^-1)_ii
        increase = max(solution[i] ** 2 / max(inverse[i, i], 1e-300), 0.0)
        explained = float(np.sqrt(increase / count))
        # without an effect, the increase follows noise_variance * chi-square(1): its square root is a t value
        significance = float(np.sqrt(increase / noise_variance)) if noise_variance > 0 else float("inf")
        alone = residual + scaled[:, i] * solution[i]
        corrections.append(
            Correction(
                key=key,
                value=float(solution[i] / norms[i]),
                explained=explained,
                rms_before=rms,
                rms_after=float(np.sqrt(alone @ alone / count)),
                # the columns are normalized: 1 / (N^-1)_ii = 1 - R^2 of the column regressed on all others
                independence=float(np.sqrt(1.0 / max(inverse[i, i], 1.0))),
                significance=significance,
            )
        )
    corrections.sort(key=lambda correction: -correction.explained)
    return Diagnosis(tuple(corrections), tuple(inseparable), rms, rms_after_all, effects, potential)


def _noise_variance(remaining: np.ndarray, parameters: int) -> float:
    """Variance of the remaining deviation (n x 3, in time order) per coordinate, inflated for its correlation
    from pose to pose: correlated noise fits a parameter by chance more easily than independent noise would.

    The inflation is the ratio of the number of observations to the effective number of independent ones,
    (1 + rho) / (1 - rho) for a lag-one autocorrelation rho.
    """
    count = remaining.size
    variance = float(np.sum(remaining**2)) / max(count - parameters, 1)
    if len(remaining) < 3 or variance <= 0:
        return variance
    centered = remaining - np.mean(remaining, axis=0)
    lagged = float(np.sum(centered[1:] * centered[:-1]))
    rho = float(np.clip(lagged / max(float(np.sum(centered**2)), 1e-300), 0.0, MAX_AUTOCORRELATION))
    return variance * (1.0 + rho) / (1.0 - rho)


def _potential(
    jacobian: np.ndarray, other_jacobian: np.ndarray, others: Sequence[str], residual: np.ndarray, count: int
) -> tuple[Potential, ...]:
    """For each parameter in ``others``: the RMS deviation if it was corrected together with the considered
    parameters (``jacobian``). Parameters that do not reduce it are left out."""
    if not others:
        return ()

    def normalized(columns: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(columns, axis=0)
        return columns / np.where(norms > 0, norms, 1.0)

    base = normalized(jacobian) if jacobian.shape[1] else np.zeros((len(residual), 0))
    rms_base = _remaining_rms(base, residual, count) if base.shape[1] else float(np.sqrt(residual @ residual / count))
    found = []
    for key, column in zip(others, other_jacobian.T, strict=True):
        if np.linalg.norm(column) <= 1e-12:
            continue
        rms_with = _remaining_rms(np.column_stack((base, normalized(column[:, None]))), residual, count)
        if rms_with < rms_base:
            found.append(Potential(key, rms_with))
    return tuple(sorted(found, key=lambda potential: potential.rms_after))


def partial_residuals(
    preview: AlignmentPreview, state: AlignmentState, pivot: np.ndarray, diagnosis: Diagnosis, keep: Sequence[str]
) -> Residuals | None:
    """The deviations without the (jointly estimated) effects of all parameters but ``keep``: the signature of
    these parameters plus what no parameter explains (noise, unmodelled errors)."""
    if (actual := residuals(preview, state, pivot)) is None:
        return None
    deviation = actual.deviation.copy()
    for key, effect in diagnosis.effects.items():
        if key not in keep and len(effect) == len(deviation):
            deviation -= effect
    return residuals(preview, state, pivot, deviation)


@dataclass(frozen=True)
class FitResult:
    state: AlignmentState
    # parameters left unchanged because their effect cannot be told apart from that of another one in the data
    # used, as pairs (left unchanged, corrected instead); e.g. leverarm z and shift z of a level platform
    inseparable: tuple[tuple[str, str], ...] = ()


def epoch_index(preview: AlignmentPreview, state: AlignmentState, epoch_mask: str | None) -> np.ndarray:
    """Reference poses overlapping in time; with ``epoch_mask`` "standstill" or "moving" only those."""
    index = preview.overlap(state)
    if epoch_mask is not None:
        still = preview.reference_standstill[index]
        index = index[still if epoch_mask == "standstill" else ~still]
    return index


def _separable(jacobian: np.ndarray, keys: Sequence[str]) -> tuple[list[int], list[tuple[str, str]]]:
    """Columns whose effect differs from that of all earlier ones (by correlation), and the others."""
    norms = np.linalg.norm(jacobian, axis=0)
    kept: list[int] = []
    inseparable = []
    for i in range(jacobian.shape[1]):
        if norms[i] <= 1e-12:
            continue
        correlations = [abs(jacobian[:, i] @ jacobian[:, j]) / (norms[i] * norms[j]) for j in kept]
        twins = [j for j, correlation in zip(kept, correlations, strict=True) if correlation > INSEPARABLE_CORRELATION]
        if twins:
            inseparable.append((keys[i], keys[twins[0]]))
        else:
            kept.append(i)
    return kept, inseparable


def fit(
    preview: AlignmentPreview,
    state: AlignmentState,
    pivot_source: np.ndarray,
    keys: Sequence[str],
    epoch_mask: str | None = None,
    nuisance: Sequence[str] = (),
    iterations: int = 20,
) -> FitResult:
    """Corrects the parameters ``keys`` so that the deviation is minimal (Gauss-Newton on the preview).

    The ``nuisance`` parameters are estimated together with them but not changed: their effects are then not
    mistaken for those of ``keys`` (e.g. a leverarm along the direction of travel for a time shift).

    A parameter whose effect cannot be told apart from that of an earlier one (``keys`` before ``nuisance``) is
    left out (see :attr:`FitResult.inseparable`); the order therefore sets the priority.

    Args:
        preview (AlignmentPreview): Trajectories.
        state (AlignmentState): Start.
        pivot_source (np.ndarray): Pivot in the frame of the unaligned trajectory (it moves with the placement).
        keys (Sequence[str]): Parameters to correct.
        epoch_mask (str, optional): "standstill" to use only the epochs at which the reference stands still,
            "moving" for the others. Defaults to None (all epochs).
        nuisance (Sequence[str], optional): Parameters estimated along, but not corrected.
        iterations (int, optional): Maximum number of iterations.
    """
    estimated = list(dict.fromkeys([*keys, *nuisance]))
    applied = set(keys)
    inseparable: list[tuple[str, str]] = []
    for iteration in range(iterations):
        index = epoch_index(preview, state, epoch_mask)
        if len(index) < MIN_EPOCHS:
            break
        epochs = preview.reference_t[index]
        pivot = state.apply(pivot_source)
        residual = (preview.positions_at(state, epochs) - preview.reference_xyz[index]).ravel()
        jacobian = _jacobian(preview, state, pivot, epochs, estimated)
        if iteration == 0:
            columns, inseparable = _separable(jacobian, estimated)
            estimated = [estimated[i] for i in columns]
            jacobian = jacobian[:, columns]
            inseparable = [pair for pair in inseparable if pair[0] in applied]
            if not applied & set(estimated):
                break
        norms = np.linalg.norm(jacobian, axis=0)
        delta = np.linalg.lstsq(jacobian / norms, -residual, rcond=1e-12)[0] / norms
        moved = 0.0  # how far the step moves the trajectory (RMS over the epochs) [m]
        for key, amount, norm in zip(estimated, delta, norms, strict=True):
            if key in applied:
                state = perturbed(state, key, float(amount), pivot)
                moved = max(moved, abs(float(amount)) * float(norm) / np.sqrt(len(epochs)))
        if moved < CONVERGED_MOVEMENT:
            break
    return FitResult(state, tuple(inseparable))


def line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Slope and intercept of the straight line through the points; None if they do not determine it."""
    if len(x) < 2 or np.ptp(x) <= 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)
