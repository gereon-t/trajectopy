"""Checks GitHub for a newer release of trajectopy.

The latest release is requested on a short-lived background thread with Python's urllib. (Qt's network
module would not need a thread, but its OpenSSL backend crashes once Python's ssl module, which loads
another OpenSSL, has been imported, as pandas and pyproj do.) Only published, non-prerelease releases count,
as returned by the GitHub API's "latest release" endpoint.
"""

import json
import logging
import re
import threading
import urllib.request
from dataclasses import dataclass

from PySide6 import QtCore

from trajectopy.__version__ import __version__

logger = logging.getLogger(__name__)

LATEST_RELEASE_URL = "https://api.github.com/repos/gereon-t/trajectopy/releases/latest"
RELEASES_PAGE_URL = "https://github.com/gereon-t/trajectopy/releases"
TIMEOUT_S = 10


@dataclass(frozen=True)
class Release:
    version: str
    url: str


def parse_version(text: str) -> tuple[int, ...] | None:
    """``"v5.0.1"`` → ``(5, 0, 1)``; None for anything else (e.g. pre-release suffixes)."""
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", text.strip())
    return tuple(int(part) for part in match[1].split(".")) if match else None


def is_newer(version: str, current: str = __version__) -> bool:
    """Whether ``version`` is a newer release than ``current`` (missing parts count as 0: 5.1 == 5.1.0)."""
    new, old = parse_version(version), parse_version(current)
    if new is None or old is None:
        return False
    length = max(len(new), len(old))
    return new + (0,) * (length - len(new)) > old + (0,) * (length - len(old))


def parse_latest_release(payload: bytes) -> Release | None:
    """The release described by a GitHub "latest release" API response."""
    try:
        data = json.loads(payload)
        version = str(data["tag_name"]).lstrip("v")
    except (ValueError, KeyError, TypeError):
        return None
    if parse_version(version) is None:
        return None
    return Release(version=version, url=str(data.get("html_url") or RELEASES_PAGE_URL))


class UpdateChecker(QtCore.QObject):
    """Requests the latest release and reports the outcome with one of its signals (on the GUI thread)."""

    update_available = QtCore.Signal(object)  # Release
    up_to_date = QtCore.Signal()
    failed = QtCore.Signal(str)
    _response = QtCore.Signal(object)  # bytes, or the error message as str; emitted by the request thread

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._checking = False
        self._response.connect(self._on_response)  # queued: the request thread emits it

    @property
    def is_checking(self) -> bool:
        return self._checking

    def check(self) -> None:
        if self._checking:
            return
        self._checking = True
        # a daemon thread: an unanswered request must not keep the application from exiting
        threading.Thread(target=self._request, name="TrajectopyUpdateCheck", daemon=True).start()

    def _request(self) -> None:
        request = urllib.request.Request(
            LATEST_RELEASE_URL,
            headers={"Accept": "application/vnd.github+json", "User-Agent": f"trajectopy/{__version__}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                outcome: bytes | str = response.read()
        except Exception as error:  # offline, timeout, HTTP error (e.g. no release yet), ...
            outcome = str(error)
        try:
            self._response.emit(outcome)
        except RuntimeError:  # the checker was deleted in the meantime (window closed)
            pass

    def _on_response(self, outcome: bytes | str) -> None:
        self._checking = False
        if isinstance(outcome, str):
            logger.info("Update check failed: %s", outcome)
            self.failed.emit(outcome)
        else:
            self.handle_response(outcome)

    def handle_response(self, payload: bytes) -> None:
        if (release := parse_latest_release(payload)) is None:
            self.failed.emit("Unexpected response from GitHub.")
        elif is_newer(release.version):
            logger.info("Trajectopy %s is available: %s", release.version, release.url)
            self.update_available.emit(release)
        else:
            self.up_to_date.emit()
