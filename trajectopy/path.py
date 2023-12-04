"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging
import os
import sys
import dotenv

dotenv.load_dotenv()
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
REPORT_PATH = os.path.abspath(os.getenv("TRAJECTOPY_REPORT_PATH", "reports"))
