"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
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


FULL_ICON_FILE_PATH = resource_path("gui/resources/full-icon-poppins.png")
ICON_FILE_PATH = resource_path("gui/resources/icon.png")
ICON_BG_FILE_PATH = resource_path("gui/resources/icon-bg.png")
YEAR = str(datetime.datetime.now().year)
