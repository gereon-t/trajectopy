"""GUI entry point for trajectopy."""

import ctypes
import logging
import os
import sys

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication
from rich.logging import RichHandler

from trajectopy.gui.themes import DARK_STYLESHEET

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[RichHandler(omit_repeated_times=False, log_time_format="%Y-%m-%d %H:%M:%S")],
)

from trajectopy.__version__ import __version__ as VERSION
from trajectopy.gui.views.main_window import TrajectopyGUI
from trajectopy.utils.common import ICON_BG_FILE_PATH

if os.name == "nt":
    myappid = f"gereont.trajectopy.main.{VERSION}"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


def main(
    single_thread: bool = False,
    report_output_path: str = "",
    report_settings_path: str = "",
    mpl_plot_settings_path: str = "",
    mapbox_token: str = "",
):
    """Launch the GUI application.

    Args:
        single_thread: Disable multithreading
        report_output_path: Output directory for reports
        report_settings_path: Path to JSON report settings file
        mpl_plot_settings_path: Path to JSON matplotlib plot settings file
        mapbox_token: Mapbox token for map styles
    """
    # On Windows with fractional DPI (125%, 150%) Qt rounds the scale factor
    # by default, causing blurry UI. PassThrough keeps the exact scale factor.
    if os.name == "nt":
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication([])

    if sys.platform == "darwin":
        font = QFont(".AppleSystemUIFont", 13)
    elif os.name == "nt":
        font = QFont("Segoe UI", 10)
    else:
        # Don't assume a specific family on Linux — let Qt use whatever
        # the desktop environment provides (Cantarell, Noto Sans, etc.)
        font = app.font()
        font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(DARK_STYLESHEET)

    _ = TrajectopyGUI(
        single_thread=single_thread,
        report_output_path=report_output_path,
        report_settings_path=report_settings_path,
        mpl_plot_settings_path=mpl_plot_settings_path,
        mapbox_token=mapbox_token,
    )
    app.setWindowIcon(QtGui.QIcon(ICON_BG_FILE_PATH))
    app.exec()


if __name__ == "__main__":
    main()
