import os
from functools import wraps

from PySide6 import QtWidgets


def set_dark_titlebar(window: QtWidgets.QWidget, use_dark: bool) -> None:
    """Set the Windows title bar to dark or light mode using DWM."""
    if os.name == "nt":
        try:
            import ctypes
            import sys

            # https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute
            build = sys.getwindowsversion().build
            if build < 17763:
                return

            attribute = 20 if build >= 18985 else 19
            value = ctypes.c_int(1 if use_dark else 0)
            hwnd = int(window.winId())
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value))
        except Exception:
            pass


def center_window(window: QtWidgets.QWidget) -> None:
    """Center a window on its screen."""
    qr = window.frameGeometry()
    cp = window.screen().availableGeometry().center()
    qr.moveCenter(cp)
    window.move(qr.topLeft())


def show_progress(func):
    """
    Decorator for synchronous manager slots: emits operation_started before and
    operation_finished after the call (also if it raises).

    Managers that run work on the :class:`~trajectopy.gui.worker.BackgroundWorker` emit these signals themselves.
    """

    @wraps(func)
    def wrapper(manager, *args, **kwargs):
        manager.operation_started.emit()
        try:
            return func(manager, *args, **kwargs)
        finally:
            manager.operation_finished.emit()

    return wrapper


def show_msg_box(message: str):
    message_box = QtWidgets.QMessageBox()
    message_box.setText(message)
    message_box.exec()


def read_file_dialog(
    parent,
    file_filter: str = "All (*.*)",
    mode: QtWidgets.QFileDialog.FileMode = QtWidgets.QFileDialog.FileMode.ExistingFiles,
):
    file_dialog = QtWidgets.QFileDialog(parent=parent)
    file_dialog.setFileMode(mode)
    fileName, _ = file_dialog.getOpenFileNames(caption="Open File", dir="", filter=file_filter)
    return fileName


def save_file_dialog(parent, file_filter: str = "All (*.*)"):
    file_dialog = QtWidgets.QFileDialog()
    file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
    fileName, _ = file_dialog.getSaveFileName(parent, caption="Select Output File", dir="", filter=file_filter)
    return fileName


def browse_dir_dialog(parent) -> str:
    file_dialog = QtWidgets.QFileDialog()
    file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
    return file_dialog.getExistingDirectory(parent, caption="Select Directory", dir="")
