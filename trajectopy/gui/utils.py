import sys
from functools import wraps

from PySide6 import QtCore, QtWidgets


def app_settings() -> QtCore.QSettings:
    """Persistent user settings of the GUI (window layout, preferences)."""
    return QtCore.QSettings("trajectopy", "trajectopy")


def set_dark_titlebar(window: QtWidgets.QWidget, use_dark: bool) -> None:
    """Set the Windows title bar to dark or light mode using DWM."""
    if sys.platform == "win32":
        try:
            import ctypes

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


def show_msg_box(message: str, parent: QtWidgets.QWidget | None = None, error: bool = False) -> None:
    """Shows a message in a dialog that is modal to (and centered on) ``parent``."""
    icon = QtWidgets.QMessageBox.Icon.Warning if error else QtWidgets.QMessageBox.Icon.Information
    message_box = QtWidgets.QMessageBox(icon, "Error" if error else "Trajectopy", message, parent=parent)
    message_box.exec()


def read_file_dialog(parent: QtWidgets.QWidget | None, file_filter: str = "All (*.*)") -> list[str]:
    file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(parent, "Open File", "", file_filter)
    return file_names


def save_file_dialog(parent: QtWidgets.QWidget | None, file_filter: str = "All (*.*)") -> str:
    file_name, _ = QtWidgets.QFileDialog.getSaveFileName(parent, "Select Output File", "", file_filter)
    return file_name


def browse_dir_dialog(parent: QtWidgets.QWidget | None) -> str:
    return QtWidgets.QFileDialog.getExistingDirectory(parent, "Select Directory", "")
