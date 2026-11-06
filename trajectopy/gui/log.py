"""Logging of the GUI: messages go to the log panel of the main window (and to the console, if there is one).

The installed app runs without a console window, so the log panel is where messages and uncaught exceptions
can be read. Records are buffered from start-up on, so that the panel also shows what was logged before it was
opened. Kept free of heavy imports: it is set up before the splash screen is shown.
"""

import logging
import sys
import threading
from collections import deque
from types import TracebackType

from PySide6.QtCore import QObject, Signal

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BUFFERED_RECORDS = 5000

logger = logging.getLogger(__name__)


class _Emitter(QObject):
    # formatted message, level; emitted in the logging thread and delivered queued to the main thread
    record_emitted = Signal(str, int)


class QtLogHandler(logging.Handler):
    """Keeps the most recent formatted records and announces new ones with a Qt signal (thread-safe)."""

    def __init__(self, capacity: int = MAX_BUFFERED_RECORDS) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        self._records: deque[tuple[str, int]] = deque(maxlen=capacity)
        self._records_lock = threading.Lock()
        self._emitter = _Emitter()
        self.record_emitted = self._emitter.record_emitted

    @property
    def records(self) -> list[tuple[str, int]]:
        """The buffered (message, level) pairs, oldest first."""
        with self._records_lock:
            return list(self._records)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            with self._records_lock:
                self._records.append((message, record.levelno))
            self.record_emitted.emit(message, record.levelno)
        except RuntimeError:  # the emitter was deleted while the application exits
            pass
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)


_handler: QtLogHandler | None = None


def log_handler() -> QtLogHandler:
    """The handler that feeds the log panel; installed on the root logger on first use."""
    global _handler  # pylint: disable=global-statement
    if _handler is None:
        _handler = QtLogHandler()
        logging.getLogger().addHandler(_handler)
    return _handler


def setup_logging() -> None:
    """Logs INFO and above to the log panel, to the console if there is one, and logs uncaught exceptions."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    log_handler()
    # sys.stderr is None in the app built without a console
    if sys.stderr is not None and not any(type(h) is logging.StreamHandler for h in root.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root.addHandler(console)
    sys.excepthook = _log_uncaught_exception
    threading.excepthook = _log_uncaught_thread_exception


def _log_uncaught_exception(
    exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType | None
) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def _log_uncaught_thread_exception(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit or args.exc_value is None:
        return
    thread = args.thread.name if args.thread is not None else "unknown"
    logger.critical(
        "Uncaught exception in thread %s", thread, exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )
