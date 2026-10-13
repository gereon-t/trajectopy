"""GUI entry point for trajectopy."""

import ctypes
import logging
import os
import sys

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from trajectopy.gui.themes import DARK_STYLESHEET, LIGHT_STYLESHEET

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

from trajectopy.__version__ import __version__ as VERSION
from trajectopy.gui.views.main_window import TrajectopyGUI
from trajectopy.utils.resources import ICON_BG_FILE_PATH

if os.name == "nt":
    myappid = f"gereont.trajectopy.main.{VERSION}"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


def main(
    single_thread: bool = False,
    plot_settings_path: str = "",
):
    """Launch the GUI application.

    Args:
        single_thread: Disable multithreading
        plot_settings_path: Path to a JSON plot settings file
    """
    # On Windows with fractional DPI (125%, 150%) Qt rounds the scale factor
    # by default, causing blurry UI. PassThrough keeps the exact scale factor.
    if os.name == "nt":
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    # pyqtgraph caches compiled OpenGL shader programs per item class; sharing all OpenGL contexts keeps
    # them valid when a 3D playback window is closed and another one is opened.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    app = QApplication([])
    app.setOrganizationName("trajectopy")
    app.setApplicationName("trajectopy")  # where QSettings stores the window layout

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

    is_dark = app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    app.setStyleSheet(DARK_STYLESHEET if is_dark else LIGHT_STYLESHEET)

    _ = TrajectopyGUI(
        single_thread=single_thread,
        plot_settings_path=plot_settings_path,
    )
    app.setWindowIcon(QtGui.QIcon(ICON_BG_FILE_PATH))
    app.exec()


if __name__ == "__main__":
    main()
