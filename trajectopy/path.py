"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging
import os
import sys

logger = logging.getLogger("root")


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(__file__)

    return os.path.join(base_path, relative_path)


VERSION_FILE_PATH = resource_path("version")
FULL_ICON_FILE_PATH = resource_path("resources/full-icon-poppins.png")
ICON_FILE_PATH = resource_path("resources/icon.png")
ICON_BG_FILE_PATH = resource_path("resources/icon-bg.png")


def mplstyle_file_path() -> str:
    custom_path = os.path.join("./custom.mplstyle")
    if os.path.isfile(custom_path):
        logger.info("Using custom matplotlib style from %s", custom_path)
        return custom_path

    logger.info(
        "Using default settings for matplotlib style. You can use custom styles by creating a 'custom.mplstyle' file in the current directory."
    )
    return resource_path("default.mplstyle")
