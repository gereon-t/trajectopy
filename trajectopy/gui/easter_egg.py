"""A hidden jingle: clicking the logo in the About window often enough plays the first two bars of the Trajectopy song.

The clip (with a fade-out of 1.5 seconds) is stored as MP3 (mono, 64 kbit/s, about 54 KB) in
``resources/jingle.mp3``. It is played with what the operating system offers instead of Qt Multimedia, which would
add many megabytes to the app: the Media Control Interface on Windows, ``afplay`` on macOS and ``mpg123``,
``ffplay``, ``pw-play`` or ``paplay`` on Linux (the last two need libsndfile 1.1 or newer for MP3). Without a
player, the jingle stays silent.
"""

import logging
import shutil
import subprocess
import sys
import time

from trajectopy.utils.resources import resource_path

logger = logging.getLogger(__name__)

JINGLE_FILE_PATH = resource_path("gui/resources/jingle.mp3")
CLICKS_TO_PLAY = 7
MAX_CLICK_INTERVAL_S = 1.0

_MCI_ALIAS = "trajectopy_jingle"
_LINUX_PLAYERS = (
    ("mpg123", "-q"),
    ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"),
    ("pw-play",),
    ("paplay",),
)


class ClickCounter:
    """Counts clicks in quick succession and reports when enough came together."""

    def __init__(self, clicks: int = CLICKS_TO_PLAY, max_interval: float = MAX_CLICK_INTERVAL_S) -> None:
        self.clicks = clicks
        self.max_interval = max_interval
        self._count = 0
        self._last = -float("inf")

    def click(self, now: float | None = None) -> bool:
        """Registers a click; True on the click that completes the series (the count starts over then)."""
        now = time.monotonic() if now is None else now
        self._count = self._count + 1 if now - self._last <= self.max_interval else 1
        self._last = now
        if self._count < self.clicks:
            return False
        self._count = 0
        return True


def mci(command: str) -> str:
    """Sends a command to the Windows Media Control Interface and returns its answer."""
    import ctypes

    answer = ctypes.create_unicode_buffer(256)
    error = ctypes.windll.winmm.mciSendStringW(command, answer, len(answer), None)  # type: ignore[attr-defined,unused-ignore]  # Windows only
    if error:
        raise OSError(f"MCI error {error} for '{command}'")
    return answer.value


def _play_windows(path: str) -> None:
    try:
        mci(f"close {_MCI_ALIAS}")  # a jingle that is still playing starts over
    except OSError:
        pass  # none was open
    mci(f'open "{path}" type mpegvideo alias {_MCI_ALIAS}')
    mci(f"play {_MCI_ALIAS}")  # returns at once, the jingle plays in the background


def _play_with_player(path: str) -> None:
    players = [("afplay",)] if sys.platform == "darwin" else list(_LINUX_PLAYERS)
    for executable, *options in players:
        if found := shutil.which(executable):
            subprocess.Popen([found, *options, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603
            return
    logger.debug("No audio player found for the jingle.")


def play_jingle(path: str = JINGLE_FILE_PATH) -> None:
    """Plays the jingle without blocking; an error is only logged, an easter egg must never disturb."""
    try:
        if sys.platform == "win32":
            _play_windows(path)
        else:
            _play_with_player(path)
    except Exception as error:
        logger.debug("Could not play the jingle: %s", error)
