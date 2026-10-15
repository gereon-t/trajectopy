"""Update check: version comparison, parsing of GitHub's response and the checker's signals (no network)."""

import io
import json
import os
import time
import unittest
import urllib.error
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets

    from trajectopy.gui import update_check
except ImportError:  # pragma: no cover - GUI extras not installed
    update_check = None


@unittest.skipIf(update_check is None, "GUI dependencies not installed")
class TestVersions(unittest.TestCase):
    def test_parse_version(self):
        self.assertEqual(update_check.parse_version("v5.0.1"), (5, 0, 1))
        self.assertEqual(update_check.parse_version("5.1"), (5, 1))
        self.assertIsNone(update_check.parse_version("v5.1.0rc1"))
        self.assertIsNone(update_check.parse_version("latest"))

    def test_is_newer(self):
        self.assertTrue(update_check.is_newer("5.0.2", current="5.0.1"))
        self.assertTrue(update_check.is_newer("5.10.0", current="5.9.3"))  # numeric, not alphabetical
        self.assertTrue(update_check.is_newer("6", current="5.9.9"))
        self.assertFalse(update_check.is_newer("5.0.1", current="5.0.1"))
        self.assertFalse(update_check.is_newer("5.1", current="5.1.0"))
        self.assertFalse(update_check.is_newer("5.0.0", current="5.0.1"))
        self.assertFalse(update_check.is_newer("garbage", current="5.0.1"))

    def test_parse_latest_release(self):
        payload = json.dumps({"tag_name": "v5.2.0", "html_url": "https://example.org/v5.2.0"}).encode()
        self.assertEqual(
            update_check.parse_latest_release(payload), update_check.Release("5.2.0", "https://example.org/v5.2.0")
        )
        without_url = update_check.parse_latest_release(json.dumps({"tag_name": "v5.2.0"}).encode())
        self.assertEqual(without_url.url, update_check.RELEASES_PAGE_URL)
        self.assertIsNone(update_check.parse_latest_release(b"not json"))
        self.assertIsNone(update_check.parse_latest_release(json.dumps({"message": "Not Found"}).encode()))


@unittest.skipIf(update_check is None, "GUI dependencies not installed")
class TestUpdateChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _check(self, urlopen) -> list[tuple[str, object]]:
        """Runs a check with the given urlopen replacement and returns the emitted signals."""
        checker = update_check.UpdateChecker()
        emitted: list[tuple[str, object]] = []
        checker.update_available.connect(lambda release: emitted.append(("update_available", release)))
        checker.up_to_date.connect(lambda: emitted.append(("up_to_date", None)))
        checker.failed.connect(lambda reason: emitted.append(("failed", reason)))
        with mock.patch.object(update_check.urllib.request, "urlopen", urlopen):
            checker.check()
            self.assertTrue(checker.is_checking)
            checker.check()  # ignored while a check is running
            deadline = time.monotonic() + 10
            while not emitted and time.monotonic() < deadline:
                self.app.processEvents()
        self.assertFalse(checker.is_checking)
        self.assertEqual(len(emitted), 1, emitted)
        return emitted

    @staticmethod
    def _respond(tag: str):
        def urlopen(request, timeout):
            assert request.full_url == update_check.LATEST_RELEASE_URL
            return io.BytesIO(json.dumps({"tag_name": tag, "html_url": f"https://example.org/{tag}"}).encode())

        return urlopen

    def test_newer_release(self):
        [(signal, release)] = self._check(self._respond("v999.0.0"))
        self.assertEqual(signal, "update_available")
        self.assertEqual(release.version, "999.0.0")

    def test_current_release(self):
        [(signal, _)] = self._check(self._respond(f"v{update_check.__version__}"))
        self.assertEqual(signal, "up_to_date")

    def test_network_error(self):
        def offline(request, timeout):
            raise urllib.error.URLError("no network")

        [(signal, reason)] = self._check(offline)
        self.assertEqual(signal, "failed")
        self.assertIn("no network", reason)


if __name__ == "__main__":
    unittest.main()
