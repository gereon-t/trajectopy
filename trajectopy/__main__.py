"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import argparse
import ctypes
import logging
import os
import sys

from PyQt6 import QtGui
from PyQt6.QtWidgets import QApplication
from rich.logging import RichHandler

from trajectopy.path import ICON_BG_FILE_PATH
from trajectopy.views.main_window import VERSION, TrajectopyGUI

if os.name == "nt":
    myappid = f"gereont.trajectopy.main.{VERSION}"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG,
    handlers=[RichHandler(omit_repeated_times=False, log_time_format="%Y-%m-%d %H:%M:%S")],
)


def main():
    parser = argparse.ArgumentParser(description="Trajectopy - Trajectory Evaluation in Python")
    parser.add_argument("--version", "-v", action="store_true")
    parser.add_argument(
        "--single_thread",
        action="store_true",
        help="Disable multithreading",
        default=getattr(sys, "gettrace", None)(),
    )
    args = parser.parse_args()

    if args.version:
        print(f"Trajectopy {VERSION}")
        return

    app = QApplication([])
    _ = TrajectopyGUI(single_thread=args.single_thread)
    app.setWindowIcon(QtGui.QIcon(ICON_BG_FILE_PATH))
    app.exec()


if __name__ == "__main__":
    main()
