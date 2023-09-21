"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger("root")

CURRENT_DIR = Path(__file__).parents[1]
VERSION_FILE_PATH = os.path.join(CURRENT_DIR, "version")
FULL_ICON_FILE_PATH = os.path.join(CURRENT_DIR, "gui", "resources", "full-icon-poppins.png")
ICON_FILE_PATH = os.path.join(CURRENT_DIR, "gui", "resources", "icon.png")


def mplstyle_file_path() -> str:
    custom_path = os.path.join("./custom.mplstyle")
    if os.path.isfile(custom_path):
        logger.info("Using custom matplotlib style from %s", custom_path)
        return custom_path

    logger.info(
        "Using default settings for matplotlib style. You can use custom styles by creating a 'custom.mplstyle' file in the current directory."
    )
    return os.path.join(CURRENT_DIR, "default.mplstyle")
