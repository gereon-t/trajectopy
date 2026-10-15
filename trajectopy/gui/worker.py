"""Execution of long-running GUI operations off the UI thread.

All managers live on the Qt main thread. Long-running work is handed to a single
:class:`BackgroundWorker`, which runs it sequentially on one plain Python thread. Signals emitted
from that thread are delivered to the receivers on the main thread through Qt's queued
connections, so they arrive in the order they were emitted.

A plain Python thread (instead of a ``QThread``) is used because Python 3.14 requires
Python-created threads for some of the numerical libraries used during processing.
"""

import concurrent.futures
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class BackgroundWorker:
    """Runs submitted functions one after another on a single background thread.

    Args:
        inline: Run submitted functions synchronously on the calling thread instead
            (used for ``--single-thread`` mode and when debugging).
    """

    def __init__(self, inline: bool = False) -> None:
        self._executor = (
            None if inline else concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="Trajectopy")
        )
        self._futures: set[concurrent.futures.Future] = set()

    def submit(self, function: Callable[..., Any], *args: Any) -> None:
        """Schedules ``function(*args)``. Exceptions are logged; handlers report errors to the user themselves."""
        if self._executor is None:
            self._run(function, *args)
            return

        future = self._executor.submit(self._run, function, *args)
        self._futures.add(future)
        future.add_done_callback(self._futures.discard)

    @property
    def is_busy(self) -> bool:
        """Whether an operation is running or waiting to run."""
        return any(not future.done() for future in self._futures)

    @staticmethod
    def _run(function: Callable[..., Any], *args: Any) -> None:
        try:
            function(*args)
        except Exception:
            logger.exception("Unhandled error in background operation %s", getattr(function, "__name__", function))

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Cancels pending work and waits for the running operation to finish.

        Returns:
            bool: True if no operation is running anymore, False if the timeout expired.
        """
        if self._executor is None:
            return True

        for future in list(self._futures):
            future.cancel()

        _, still_running = concurrent.futures.wait(list(self._futures), timeout=timeout)
        if still_running:
            return False

        self._executor.shutdown(wait=False)
        return True
