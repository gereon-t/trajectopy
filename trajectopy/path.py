"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import datetime
import logging
import os
import sys

logger = logging.getLogger("root")


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = os.path.join(sys._MEIPASS, "trajectopy")
    except Exception:
        base_path = os.path.dirname(__file__)

    return os.path.join(base_path, relative_path)


VERSION_FILE_PATH = resource_path("version")
FULL_ICON_FILE_PATH = resource_path("gui/resources/full-icon-poppins.png")
ICON_FILE_PATH = resource_path("gui/resources/icon.png")
ICON_BG_FILE_PATH = resource_path("gui/resources/icon-bg.png")


VERSION = open(VERSION_FILE_PATH, "r", encoding="utf-8").read()
YEAR = str(datetime.datetime.now().year)
