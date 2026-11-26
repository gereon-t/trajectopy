"""A hidden jingle: clicking the logo in the About window often enough plays the start of the Trajectopy song.

The clip is stored as 4-bit IMA ADPCM (mono, 11025 Hz, a few KB) in ``resources/jingle.adpcm`` and played with
a reverb that is computed when it is played, so that the reverb tail takes no space. It is played with what the
operating system offers (``winsound`` on Windows, ``afplay`` on macOS, ``pw-play``, ``paplay`` or ``aplay`` on
Linux) instead of Qt Multimedia, which would add many megabytes to the app.
"""

import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np

from trajectopy.utils.resources import resource_path

logger = logging.getLogger(__name__)

JINGLE_FILE_PATH = resource_path("gui/resources/jingle.adpcm")
CLICKS_TO_PLAY = 7
MAX_CLICK_INTERVAL_S = 1.0

_MAGIC = b"TPYJ"
_HEADER = struct.Struct("<4sII")  # magic, sample rate, number of samples

# IMA ADPCM step sizes and index changes
STEP_SIZES = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97,
    107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871,
    5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623,
    27086, 29794, 32767,
)  # fmt: skip
INDEX_CHANGES = (-1, -1, -1, -1, 2, 4, 6, 8)


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


def decode_adpcm(data: bytes, num_samples: int) -> np.ndarray:
    """Decodes 4-bit IMA ADPCM (low nibble first, starting at predictor 0 and step index 0) to 16-bit samples."""
    samples = np.empty(num_samples, dtype=np.int16)
    predictor, index = 0, 0
    for n in range(num_samples):
        code = (data[n >> 1] >> ((n & 1) << 2)) & 0xF
        step = STEP_SIZES[index]
        difference = step >> 3
        if code & 1:
            difference += step >> 2
        if code & 2:
            difference += step >> 1
        if code & 4:
            difference += step
        predictor = max(-32768, min(32767, predictor - difference if code & 8 else predictor + difference))
        index = max(0, min(88, index + INDEX_CHANGES[code & 7]))
        samples[n] = predictor
    return samples


def read_jingle(path: str = JINGLE_FILE_PATH) -> tuple[np.ndarray, int]:
    """The clip as floats in [-1, 1] and its sample rate."""
    with open(path, "rb") as file:
        content = file.read()
    magic, sample_rate, num_samples = _HEADER.unpack_from(content)
    if magic != _MAGIC:
        raise ValueError(f"{path} is not a jingle file")
    return decode_adpcm(content[_HEADER.size :], num_samples) / 32768.0, sample_rate


def add_reverb(signal: np.ndarray, sample_rate: int, decay: float = 2.0, wet: float = 0.45) -> np.ndarray:
    """Adds a hall-like reverb; returns stereo samples (n x 2) with the reverb tail, normalized to a peak of 0.9.

    The impulse response of each channel is decaying noise (60 dB in ``decay`` seconds) that becomes darker
    over time, after a pre-delay of 25 ms; different noise in both channels makes the reverb wide.
    """
    from scipy.signal import fftconvolve, lfilter

    rng = np.random.default_rng(7)
    length = int(decay * sample_rate)
    t = np.arange(length) / sample_rate
    pre_delay = int(0.025 * sample_rate)
    channels = []
    for _ in range(2):
        noise = rng.standard_normal(length) * np.exp(-6.9 * t / decay)
        darker = lfilter([0.35], [1.0, -0.65], noise)  # high frequencies decay faster in a real hall
        tail = np.where(t < 0.04, noise, darker)  # bright early reflections, then a darker tail
        response = np.concatenate([np.zeros(pre_delay), tail])
        response /= np.sqrt(np.sum(response**2))
        channels.append(fftconvolve(signal, response))
    reverb = np.column_stack(channels)
    dry = np.zeros_like(reverb)
    dry[: len(signal)] = signal[:, None]
    mixed = (1.0 - wet) * dry + wet * reverb
    return 0.9 * mixed / max(float(np.max(np.abs(mixed))), 1e-9)


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """Writes stereo float samples (n x 2) as 16-bit WAV file."""
    with wave.open(path, "wb") as file:
        file.setnchannels(samples.shape[1])
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def _play_file(path: str) -> None:
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        return
    players = ["afplay"] if sys.platform == "darwin" else ["pw-play", "paplay", "aplay"]
    for player in players:
        if executable := shutil.which(player):
            subprocess.Popen([executable, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603
            return
    logger.debug("No audio player found for the jingle.")


def _play() -> None:
    try:
        signal, sample_rate = read_jingle()
        path = os.path.join(tempfile.gettempdir(), "trajectopy-jingle.wav")
        write_wav(path, add_reverb(signal, sample_rate), sample_rate)
        _play_file(path)
    except Exception as error:  # an easter egg must never disturb the application
        logger.debug("Could not play the jingle: %s", error)


def play_jingle() -> None:
    """Plays the jingle without blocking (decoding and the reverb run on a background thread)."""
    threading.Thread(target=_play, name="TrajectopyJingle", daemon=True).start()
