import sys
from collections.abc import Callable, Hashable
from functools import wraps

from PySide6 import QtCore, QtGui, QtWidgets


class SingleWindows:
    """Windows that are opened at most once per key (e.g. the settings of a trajectory): opening one again
    brings the open window to the front instead of creating a second one.

    The windows must delete themselves on close (``WA_DeleteOnClose``).
    """

    def __init__(self) -> None:
        self._windows: dict[Hashable, QtWidgets.QWidget] = {}

    def show(self, key: Hashable, create: Callable[[], QtWidgets.QWidget], replace: bool = False) -> QtWidgets.QWidget:
        """Shows the window of ``key``, created with ``create`` if none is open.

        Args:
            key (Hashable): Identifies the window, e.g. the entry whose settings it shows.
            create (Callable[[], QtWidgets.QWidget]): Creates the window.
            replace (bool, optional): Close an open window and create a new one, e.g. if the window works on the
                current selection. Defaults to False (the open window is brought to the front).

        Returns:
            QtWidgets.QWidget: The shown window.
        """
        window = self._open_window(key)
        if window is not None and replace:
            window.close()
            window = None
        if window is None:
            window = create()
            self._windows[key] = window
            window.destroyed.connect(lambda _=None, key=key, window=window: self._forget(key, window))
            center_window(window)
            window.show()
        if window.isMinimized():
            window.showNormal()
        window.raise_()
        window.activateWindow()
        return window

    def close(self, key: Hashable) -> None:
        """Closes the window of ``key``, if it is open."""
        if (window := self._open_window(key)) is not None:
            window.close()

    def _open_window(self, key: Hashable) -> QtWidgets.QWidget | None:
        window = self._windows.get(key)
        try:
            # a closed window may still exist until Qt deletes it
            return window if window is not None and window.isVisible() else None
        except RuntimeError:  # already deleted
            return None

    def _forget(self, key: Hashable, window: QtWidgets.QWidget) -> None:
        if self._windows.get(key) is window:
            del self._windows[key]


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


def center_window(window: QtWidgets.QWidget, anchor: QtWidgets.QWidget | None = None) -> None:
    """Centers a window over the Trajectopy window it belongs to, so that it opens on the screen where Trajectopy
    is (not on the primary screen); without one, on the screen of the window. It is kept within that screen.

    Args:
        window (QtWidgets.QWidget): Window to place, with its final size.
        anchor (QtWidgets.QWidget | None, optional): Window to center on. Defaults to the window of the parent,
            or the active window.
    """
    if anchor is None:
        parent = window.parentWidget()
        anchor = parent.window() if parent is not None else QtWidgets.QApplication.activeWindow()
    if anchor is not None and anchor is not window and anchor.isVisible():
        center = anchor.frameGeometry().center()
        screen = QtGui.QGuiApplication.screenAt(center) or anchor.screen()
    else:
        screen = window.screen()
        center = screen.availableGeometry().center()
    available = screen.availableGeometry()
    frame = window.frameGeometry()
    frame.moveCenter(center)
    left = max(available.left(), min(frame.left(), available.right() + 1 - frame.width()))
    top = max(available.top(), min(frame.top(), available.bottom() + 1 - frame.height()))
    window.move(left, top)


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
