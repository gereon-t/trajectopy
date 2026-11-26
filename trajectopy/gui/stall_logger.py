"""Logs where the main thread is when the window stops responding.

A timer in the main (GUI) thread records a heartbeat. A background thread checks it: if the heartbeat is older
than a threshold, the main thread is blocked and the window does not respond. The background thread then logs
the stack of the main thread, once when the stall is noticed and once more when it is over, together with its
duration. The log panel thus shows which code blocks the window.
"""

import logging
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

from PySide6 import QtCore

logger = logging.getLogger(__name__)

STALL_THRESHOLD_S = 0.5
HEARTBEAT_INTERVAL_MS = 50
CHECK_INTERVAL_S = 0.1
MAX_STACK_FRAMES = 25

_pauses = 0  # number of active paused() blocks


@contextmanager
def paused() -> Iterator[None]:
    """Does not report stalls inside the block.

    For native dialogs (e.g. the Windows file dialog): they run an event loop of the operating system, in which
    Qt timers do not fire, so the heartbeat stops although the dialog responds.
    """
    global _pauses  # pylint: disable=global-statement
    _pauses += 1
    try:
        yield
    finally:
        _pauses -= 1


class StallLogger(QtCore.QObject):
    """Watches the main thread and logs its stack while it is blocked (start with :meth:`start`)."""

    def __init__(self, threshold: float = STALL_THRESHOLD_S, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.threshold = threshold
        self._main_thread_id = threading.get_ident()  # created in the main thread
        self._heartbeat = time.monotonic()
        self._stop = threading.Event()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(HEARTBEAT_INTERVAL_MS)
        self._timer.timeout.connect(self._beat)
        self._thread = threading.Thread(target=self._watch, name="TrajectopyStallLogger", daemon=True)

    def start(self) -> None:
        """Starts watching. Call it from the main thread."""
        self._heartbeat = time.monotonic()
        self._timer.start()
        self._thread.start()

    def stop(self) -> None:
        self._timer.stop()
        self._stop.set()

    def _beat(self) -> None:
        self._heartbeat = time.monotonic()

    def _watch(self) -> None:
        stall_start: float | None = None  # last heartbeat before the current stall
        last_stack = ""
        while not self._stop.wait(CHECK_INTERVAL_S):
            if _pauses:
                self._heartbeat = time.monotonic()
                stall_start = None
                continue
            heartbeat = self._heartbeat
            if stall_start is None:
                if (blocked := time.monotonic() - heartbeat) > self.threshold:
                    stall_start = heartbeat
                    last_stack = self._main_stack()
                    logger.warning(
                        "The window has not responded for %.1f s. The main thread is in:\n%s", blocked, last_stack
                    )
            elif heartbeat > stall_start:  # the heartbeat ran again: the stall is over
                logger.warning(
                    "The window did not respond for %.1f s. The main thread was last in:\n%s",
                    heartbeat - stall_start,
                    last_stack,
                )
                stall_start = None
            else:
                last_stack = self._main_stack()

    def _main_stack(self) -> str:
        frame: FrameType | None = sys._current_frames().get(self._main_thread_id)  # noqa: SLF001
        if frame is None:
            return "  (unknown)"
        lines = traceback.format_stack(frame)[-MAX_STACK_FRAMES:]
        return "".join(lines).rstrip()
