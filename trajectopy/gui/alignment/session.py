"""State of one interactive alignment: the current placement, its history and the estimation.

The session is the single owner of the :class:`AlignmentState`. The widgets of the window read it and ask the
session to change it; every change is a step that can be undone.
"""

import copy
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore

from trajectopy.core.settings import (
    AlignmentEstimationSettings,
    AlignmentSettings,
    MatchingSettings,
    ProcessingSettings,
)
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing import alignment
from trajectopy.processing.lib.alignment import diagnostics, interactive
from trajectopy.processing.lib.alignment.interactive import AlignmentPreview, AlignmentState
from trajectopy.results.alignment_result import AlignmentResult

logger = logging.getLogger(__name__)

LOW_CORRELATION = 0.5


@dataclass
class _Step:
    label: str
    state: AlignmentState
    # result of the least squares alignment that produced this state (applied with its statistics)
    result: AlignmentResult | None = None
    # a drag in progress: further changes with the same label replace this step instead of adding one
    open: bool = False


@dataclass(frozen=True)
class PointPair:
    """A pose of the trajectory (index into the preview) and the reference position it belongs to."""

    index: int
    target: np.ndarray


def _estimate_alignment(**arguments) -> AlignmentResult | Exception:
    """Least squares alignment (see alignment.estimate_alignment), with the error returned instead of raised."""
    try:
        return alignment.estimate_alignment(**arguments)
    except Exception as error:
        logger.exception("Least squares alignment failed")
        return error


class _EstimationSignals(QtCore.QObject):
    finished = QtCore.Signal(object, int, str)  # AlignmentResult or exception, run id, label


class AlignmentSession(QtCore.QObject):
    state_changed = QtCore.Signal()  # the placement, the pivot or the point pairs changed
    history_changed = QtCore.Signal()
    jumped = QtCore.Signal()  # the placement changed by a larger amount at once (views should show everything)
    busy_changed = QtCore.Signal(bool)
    estimation_changed = QtCore.Signal()  # the parameters to estimate (``estimated_parameters``) changed
    message_changed = QtCore.Signal(str)

    def __init__(
        self,
        trajectory: Trajectory,
        reference: Trajectory,
        matching_settings: MatchingSettings | None = None,
        alignment_settings: AlignmentSettings | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        # copies: operations on the trajectories in the background must not change them while the window is open
        self.trajectory = trajectory.copy()
        self.reference = reference.copy()
        self.matching_settings = matching_settings or MatchingSettings()
        self.alignment_settings = alignment_settings or AlignmentSettings()
        self.preview = AlignmentPreview(self.trajectory, self.reference)

        self._steps = [_Step("Start", AlignmentState())]
        self._index = 0
        self._pivot_source = self.preview.center.copy()  # in the frame of the unaligned trajectory
        # the parameters that are estimated (keys of diagnostics.PARAMETERS); the diagnosis considers only these
        self._estimated: tuple[str, ...] = diagnostics.available_parameters(self.preview)
        self.pairs: list[PointPair] = []
        self.pairs_with_scale = False
        self.message = ""
        self._busy = False
        self._run_id = 0
        self._signals = _EstimationSignals(self)  # owned by the session, so that the worker never deletes it
        self._signals.finished.connect(self._on_estimated)

    # ---------------------------------------------------------------------------------------------- state

    @property
    def state(self) -> AlignmentState:
        return self._steps[self._index].state

    @property
    def has_orientation(self) -> bool:
        return self.preview.has_orientation

    def set_state(
        self,
        state: AlignmentState,
        label: str,
        *,
        coalesce: bool = False,
        result: AlignmentResult | None = None,
        jump: bool = False,
        message: str = "",
    ) -> None:
        """Makes ``state`` the current placement as an undoable step.

        Args:
            state (AlignmentState): New placement.
            label (str): Describes the step, e.g. for the undo action.
            coalesce (bool, optional): Part of a continuous change (drag): merged with the previous step if that
                was part of it, too. :meth:`commit` ends the change. Defaults to False.
            result (AlignmentResult, optional): Estimation result that led to the state.
            jump (bool, optional): The placement changed by a larger amount at once.
            message (str, optional): Shown to the user.
        """
        current = self._steps[self._index]
        if state == current.state and result is None:
            self._set_message(message)
            return
        if coalesce and current.open and current.label == label:
            current.state = state
        else:
            current.open = False
            del self._steps[self._index + 1 :]
            self._steps.append(_Step(label, state, result, open=coalesce))
            self._index += 1
            self.history_changed.emit()
        self._set_message(message)
        self.state_changed.emit()
        if jump:
            self.jumped.emit()

    def commit(self) -> None:
        """Ends a continuous change: the next change is a step of its own."""
        self._steps[self._index].open = False

    def _set_message(self, message: str) -> None:
        self.message = message
        self.message_changed.emit(message)

    # -------------------------------------------------------------------------------------------- history

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index < len(self._steps) - 1

    @property
    def undo_label(self) -> str:
        return self._steps[self._index].label if self.can_undo else ""

    @property
    def redo_label(self) -> str:
        return self._steps[self._index + 1].label if self.can_redo else ""

    def undo(self) -> None:
        if self.can_undo and not self._busy:
            self._go_to(self._index - 1, f"Undone: {self.undo_label}")

    def redo(self) -> None:
        if self.can_redo and not self._busy:
            self._go_to(self._index + 1, f"Redone: {self.redo_label}")

    def _go_to(self, index: int, message: str) -> None:
        self.commit()
        self._index = index
        self.history_changed.emit()
        self._set_message(message)
        self.state_changed.emit()

    def reset(self) -> None:
        """Back to the unaligned trajectory (as a step, so it can be undone)."""
        self.clear_pairs()
        self.set_state(AlignmentState(), "Reset", jump=True, message="Reset to the unaligned trajectory.")

    # ------------------------------------------------------------------------------------- estimated

    @property
    def estimated_parameters(self) -> tuple[str, ...]:
        """The parameters that are estimated: the diagnosis and the corrections consider only these."""
        return self._estimated

    def set_estimated_parameters(self, keys: Sequence[str]) -> None:
        available = diagnostics.available_parameters(self.preview)
        estimated = tuple(key for key in available if key in keys)
        if estimated != self._estimated:
            self._estimated = estimated
            self.estimation_changed.emit()

    # ---------------------------------------------------------------------------------------------- pivot

    @property
    def pivot(self) -> np.ndarray:
        """The point rotations and scalings act about, where it is shown. It moves with the trajectory."""
        return self.state.apply(self._pivot_source)

    def set_pivot(self, point: np.ndarray) -> None:
        state = self.state
        self._pivot_source = state.rotation.T @ (np.asarray(point, dtype=float) - state.translation) / state.scale
        self.state_changed.emit()

    def center_pivot(self) -> None:
        self._pivot_source = self.preview.center.copy()
        self.state_changed.emit()

    @property
    def pivot_source(self) -> np.ndarray:
        """The pivot in the frame of the unaligned trajectory."""
        return self._pivot_source.copy()

    @property
    def pivot_shift(self) -> np.ndarray:
        """How far the placement moved the pivot."""
        return self.pivot - self._pivot_source

    # ---------------------------------------------------------------------------------------- placements

    def match_first_pose(self) -> None:
        self.set_state(
            interactive.match_first_pose(self.preview, self.state),
            "Match first pose",
            jump=True,
            message="Matched the first pose.",
        )

    def sync_time(self) -> None:
        """Sets the time shift for which the speed profiles of both trajectories agree best."""
        estimate = interactive.estimate_time_shift(self.preview)
        if estimate is None:
            self._set_message(
                "Sync time needs a speed that varies along both trajectories. Drag in the speed plot instead."
            )
            return
        confidence = (
            "The speed profiles agree only weakly: check the speed plot."
            if estimate.correlation < LOW_CORRELATION
            else "Check the speed plot."
        )
        self.set_state(
            self.state.with_time_shift(estimate.time_shift),
            "Sync time",
            jump=True,
            message=f"Time shift from the speed profiles: {estimate.time_shift:.3f} s "
            f"(correlation {estimate.correlation:.2f}). {confidence}",
        )

    def correct(self, keys: Sequence[str], label: str, epoch_mask: str | None = None) -> None:
        """Corrects the parameters ``keys`` so that the deviation at equal time is minimal (on the preview).

        The other estimated parameters (:attr:`estimated_parameters`) are estimated along, but not changed, so
        the corrections are those of the joint diagnosis.

        Args:
            keys (Sequence[str]): Parameters (see ``diagnostics.PARAMETERS``); earlier ones have priority if two
                cannot be told apart.
            label (str): Name of the step.
            epoch_mask (str, optional): "standstill" or "moving" to use only these epochs.
        """
        keys = [key for key in keys if key in diagnostics.available_parameters(self.preview)]
        before = diagnostics.residuals(self.preview, self.state, self.pivot)
        if before is None:
            self._set_message("No poses at equal time: place the trajectory and synchronize the time first.")
            return
        if not keys:
            self._set_message("Nothing to correct: check “estimate” for the parameters that may change.")
            return
        if epoch_mask is not None and len(diagnostics.epoch_index(self.preview, self.state, epoch_mask)) < (
            diagnostics.MIN_EPOCHS
        ):
            self._set_message(f"Too few {'standstill' if epoch_mask == 'standstill' else 'moving'} epochs.")
            return
        # the other estimated parameters are estimated along, so that their effects are not attributed to ``keys``
        result = diagnostics.fit(
            self.preview, self.state, self._pivot_source, keys, epoch_mask, nuisance=self._estimated
        )
        after = diagnostics.residuals(self.preview, result.state, result.state.apply(self._pivot_source))
        rms_after = after.rms if after is not None else float("nan")
        notes = [
            f"{diagnostics.PARAMETERS[kept].name} cannot be told apart from {diagnostics.PARAMETERS[used].name} "
            "here and was left unchanged."
            for kept, used in result.inseparable
        ]
        self.set_state(
            result.state,
            label,
            message=" ".join([f"{label}: RMS deviation {before.rms:.3f} m → {rms_after:.3f} m.", *notes]),
        )

    # ---------------------------------------------------------------------------------------- point pairs

    def add_pair(self, index: int, target: np.ndarray) -> None:
        self.pairs.append(PointPair(int(index), np.asarray(target, dtype=float)))
        self.fit_pairs()

    def remove_pair(self, row: int) -> None:
        if 0 <= row < len(self.pairs):
            del self.pairs[row]
            self.fit_pairs()

    def clear_pairs(self) -> None:
        if self.pairs:
            self.pairs.clear()
            self.state_changed.emit()

    def fit_pairs(self) -> None:
        """Places the trajectory so that the picked poses meet their reference positions."""
        if not self.pairs:
            self.state_changed.emit()
            return
        source = self.preview.positions(self.state)[[pair.index for pair in self.pairs]]
        target = np.array([pair.target for pair in self.pairs])
        rotation, translation, scale = interactive.fit_point_pairs(source, target, self.pairs_with_scale)
        determined = {1: "position", 2: "position and heading"}.get(len(self.pairs), "position and rotation")
        if self.pairs_with_scale and len(self.pairs) > 1:
            determined += ", scale"
        count = len(self.pairs)
        state = self.state.composed(rotation=rotation, translation=translation, scale=scale)
        self.set_state(
            state,
            "Fit point pairs",
            message=f"{count} point pair{'s' if count != 1 else ''}: {determined} fitted.",
        )
        self.state_changed.emit()  # the pairs changed, even if the placement did not

    # ----------------------------------------------------------------------------------------- estimation

    @property
    def busy(self) -> bool:
        return self._busy

    def auto_align(self, estimation: AlignmentEstimationSettings) -> None:
        """Rough placement chosen automatically, followed by the least squares alignment of the parameters
        selected in ``estimation`` (in the background)."""
        name, start = interactive.best_start(self.preview, self.state, with_scale=estimation.scale)
        self.set_state(start, "Auto align", jump=True, message=f"Placed roughly ({name}).")
        if any(estimation.to_dict().values()):
            self.refine(estimation, label="Auto align", start=name)

    def refine(self, estimation: AlignmentEstimationSettings, label: str = "Refine", start: str = "") -> None:
        """Least squares alignment of the parameters selected in ``estimation``, starting from the current
        placement. Runs in the background; the result becomes a new step."""
        if self._busy:
            return
        if not any(estimation.to_dict().values()):
            self._set_message("Nothing to estimate: check “estimate” for at least one parameter.")
            return
        self.commit()
        self._set_busy(True)
        self._set_message("Estimating …")
        self._run_id += 1
        # the worker gets plain data and a bound signal only: no Qt object is created or released on its thread
        arguments = self.estimation_arguments(estimation)
        emit, run_id = self._signals.finished.emit, self._run_id
        description = f"{label} (start: {start})" if start else label
        # a daemon thread: it only computes, so a fit still running must not keep the application alive
        threading.Thread(
            target=lambda: emit(_estimate_alignment(**arguments), run_id, description),
            name="TrajectopyAlignment",
            daemon=True,
        ).start()

    def cancel_estimation(self) -> None:
        """Stops waiting for the estimation running in the background; its result is discarded."""
        if self._busy:
            self._run_id += 1
            self._set_busy(False)
            self._set_message("Estimation cancelled.")

    def estimation_arguments(self, estimation: AlignmentEstimationSettings) -> dict:
        settings = copy.deepcopy(self.alignment_settings)
        settings.estimation_settings = estimation
        return {
            "trajectory": self.trajectory,
            "other": self.reference,
            "settings": ProcessingSettings(alignment=settings, matching=self.matching_settings),
            "initial_parameters": self.state.to_parameters(),
        }

    def estimate_now(self, estimation: AlignmentEstimationSettings, label: str = "Refine") -> None:
        """Like :meth:`refine`, but blocks until the result is available."""
        self._run_id += 1
        self._set_busy(True)
        self._on_estimated(_estimate_alignment(**self.estimation_arguments(estimation)), self._run_id, label)

    def _on_estimated(self, outcome: AlignmentResult | Exception, run_id: int, label: str) -> None:
        if run_id != self._run_id:  # cancelled
            return
        self._set_busy(False)
        if isinstance(outcome, Exception):
            self._set_message(f"The least squares alignment failed ({outcome}). The placement is unchanged.")
        elif not outcome.converged:
            self._set_message("The least squares alignment did not converge. The placement is unchanged.")
        else:
            state = AlignmentState.from_parameters(outcome.position_parameters)
            if not self.has_orientation:
                state = state.with_leverarm(np.zeros(3))
            self.set_state(
                state,
                label.split(" (")[0],
                result=outcome,
                jump=True,
                message=f"{label}: estimated by least squares. Press “Apply” to keep the result.",
            )

    def _set_busy(self, busy: bool) -> None:
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    # --------------------------------------------------------------------------------------------- result

    def alignment_result(self) -> AlignmentResult:
        """The alignment of the current placement: the least squares result if the placement comes unchanged
        from it (keeping its statistics), otherwise a manual alignment with the values of the placement."""
        step = self._steps[self._index]
        if step.result is not None:
            return step.result
        return alignment.manual_alignment(self.trajectory, self.reference, step.state.to_parameters())
