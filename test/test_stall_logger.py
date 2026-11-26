import time
import unittest

from PySide6 import QtCore

from trajectopy.gui.stall_logger import StallLogger, paused
from trajectopy.visualization.figure import ensure_app


def _blocking_function_for_the_test() -> None:
    time.sleep(0.6)


class TestStallLogger(unittest.TestCase):
    def setUp(self) -> None:
        self.app = ensure_app()
        self.stall_logger = StallLogger(threshold=0.2)
        self.stall_logger.start()

    def tearDown(self) -> None:
        self.stall_logger.stop()

    def _process_events(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
            time.sleep(0.01)

    def test_blocked_main_thread_is_logged_with_its_stack(self):
        self._process_events(0.2)
        with self.assertLogs("trajectopy.gui.stall_logger", level="WARNING") as logs:
            _blocking_function_for_the_test()
            self._process_events(0.4)
        output = "\n".join(logs.output)
        self.assertIn("has not responded", output)
        self.assertIn("did not respond", output)
        self.assertIn("_blocking_function_for_the_test", output)

    def test_paused_blocks_are_not_logged(self):
        self._process_events(0.2)
        with self.assertNoLogs("trajectopy.gui.stall_logger", level="WARNING"):
            with paused():
                _blocking_function_for_the_test()
            self._process_events(0.4)

    def test_responsive_main_thread_is_not_logged(self):
        with self.assertNoLogs("trajectopy.gui.stall_logger", level="WARNING"):
            self._process_events(0.5)


if __name__ == "__main__":
    unittest.main()
