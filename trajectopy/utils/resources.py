"""Paths to bundled resources (works for source checkouts and PyInstaller builds)."""

import datetime
import logging
import os
import sys

logger = logging.getLogger(__name__)


def resource_path(relative_path: str) -> str:
    """Get absolute path to a resource inside the trajectopy package; works for dev and for PyInstaller."""
    bundle_dir = getattr(sys, "_MEIPASS", None)
    base_path = os.path.join(bundle_dir, "trajectopy") if bundle_dir else os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)


FULL_ICON_FILE_PATH = resource_path("gui/resources/full-icon-poppins.png")
ICON_FILE_PATH = resource_path("gui/resources/icon.png")
ICON_BG_FILE_PATH = resource_path("gui/resources/icon-bg.png")
YEAR = str(datetime.datetime.now().year)
