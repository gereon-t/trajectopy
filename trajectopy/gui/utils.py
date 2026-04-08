from functools import wraps

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QMetaObject, Qt


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

    This should be used for functions that take a long time to execute.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        manager = args[0]
        progress_window = getattr(manager, "_progress_window", None)

        if progress_window is not None:
            # Cross-thread: invoke the progress window slot directly via
            # QMetaObject.invokeMethod with BlockingQueuedConnection.
            # This guarantees the progress window is shown before work
            # begins, even in PyInstaller-packaged executables where
            # PySide6 signal dispatch across threads can be unreliable
            # on the first call.
            QMetaObject.invokeMethod(
                progress_window,
                "handle_show_request",
                Qt.ConnectionType.BlockingQueuedConnection,
            )

        manager.operation_started.emit()

        try:
            return func(*args, **kwargs)
        finally:
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
