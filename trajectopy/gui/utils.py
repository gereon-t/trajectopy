import os
from functools import wraps

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QMetaObject, Qt


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


def handle_drag_enter(event: QtGui.QDragEnterEvent | None) -> None:
    """Shared drag enter handler for table views accepting file drops."""
    if event is None:
        return
    if (mime_data := event.mimeData()) is None:
        return
    if mime_data.hasUrls():
        event.accept()
    else:
        event.ignore()


def handle_drag_move(event: QtGui.QDragMoveEvent | None) -> None:
    """Shared drag move handler for table views accepting file drops."""
    if event is None:
        return
    if (mime_data := event.mimeData()) is None:
        return
    if mime_data.hasUrls():
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
    else:
        event.ignore()


def show_progress(func):
    """
    Decorator to show progress bar while executing a function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        manager = args[0]
        progress_window = getattr(manager, "_progress_window", None)

        if progress_window is not None:
            QMetaObject.invokeMethod(
                progress_window,
                "handle_show_request",
                Qt.ConnectionType.BlockingQueuedConnection,
            )

        manager.operation_started.emit()

        try:
            result = func(*args, **kwargs)
            # Check if the result is a concurrent.futures.Future
            import concurrent.futures

            if isinstance(result, concurrent.futures.Future):

                def on_done(future):
                    # We must emit signals from the manager's thread
                    # QMetaObject.invokeMethod handles this nicely if we call a slot,
                    # but we can also just emit the signal directly since Qt queued connections will route it.
                    manager.operation_finished.emit()
                    if progress_window is not None:
                        QMetaObject.invokeMethod(
                            progress_window,
                            "handle_close_request",
                            Qt.ConnectionType.BlockingQueuedConnection,
                        )
                    # If there's an exception, we might want to log it
                    try:
                        future.result()
                    except Exception as e:
                        import logging

                        logging.getLogger(__name__).exception("Error in background task: %s", e)

                result.add_done_callback(on_done)
                return result
            else:
                return result
        finally:
            # Only emit finished immediately if it's NOT a Future
            if not isinstance(result, concurrent.futures.Future):
                manager.operation_finished.emit()
                if progress_window is not None:
                    QMetaObject.invokeMethod(
                        progress_window,
                        "handle_close_request",
                        Qt.ConnectionType.BlockingQueuedConnection,
                    )

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
