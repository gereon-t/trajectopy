"""The file dialog of the GUI.

On Windows, Qt's own dialog is used instead of the native one (which takes seconds to return after a file was
chosen, see :func:`file_dialog_options`). It is adapted to trajectopy: it opens in the folder that was used last
for the same kind of file, remembers its size, lists the usual and the recently used folders in its sidebar, and
shows the columns and buttons in the style of the rest of the GUI.
"""

import os
import re
import sys
from collections.abc import Iterable
from enum import Enum

from PySide6 import QtCore, QtWidgets

from trajectopy.gui import stall_logger

SETTINGS_GROUP = "fileDialog"
GEOMETRY_KEY = f"{SETTINGS_GROUP}/geometry"
FOLDER_KEY = f"{SETTINGS_GROUP}/folders"
MAX_RECENT_FOLDERS = 5
DEFAULT_SIZE = QtCore.QSize(900, 560)
SIDEBAR_MIN_WIDTH = 170

# columns of the detail view of QFileDialog
_NAME_COLUMN, _SIZE_COLUMN, _TYPE_COLUMN, _DATE_COLUMN = range(4)


class Mode(Enum):
    OPEN = "open"
    OPEN_MANY = "open_many"
    SAVE = "save"
    DIRECTORY = "directory"


def file_dialog_options() -> QtWidgets.QFileDialog.Option:
    """Options of all file dialogs of the GUI.

    On Windows, Qt's own dialog is used: the native Windows dialog took about five seconds to return after a
    file was chosen (measured with Qt 6.11 and with tkinter, so it is a delay of the Windows shell), during which
    the window does not respond. Qt's dialog returns at once.
    """
    if sys.platform == "win32":
        return QtWidgets.QFileDialog.Option.DontUseNativeDialog
    return QtWidgets.QFileDialog.Option(0)


def run_file_dialog(
    parent: QtWidgets.QWidget | None,
    mode: Mode,
    caption: str,
    file_filter: str = "",
    directory: str = "",
) -> list[str]:
    """Shows a file dialog and returns the chosen paths (empty if it was cancelled).

    Args:
        parent: Parent window.
        mode: Open one or several existing files, save a file or select a directory.
        caption: Window title.
        file_filter: Qt name filters, e.g. ``"Trajectory File (*.traj);;All (*.*)"``.
        directory: Folder (or, when saving, file) to start in. Defaults to the folder used last for the same
            kind of file.
    """
    settings = _settings()
    kind = _kind(mode, file_filter)
    folders = _remembered_folders(settings)
    start = directory or folders.get(kind, "")

    dialog = QtWidgets.QFileDialog(parent, caption, start if _exists(start) else "", file_filter)
    options = file_dialog_options()
    if mode == Mode.DIRECTORY:
        options |= QtWidgets.QFileDialog.Option.ShowDirsOnly
    dialog.setOptions(options)
    dialog.setFileMode(
        {
            Mode.OPEN: QtWidgets.QFileDialog.FileMode.ExistingFile,
            Mode.OPEN_MANY: QtWidgets.QFileDialog.FileMode.ExistingFiles,
            Mode.SAVE: QtWidgets.QFileDialog.FileMode.AnyFile,
            Mode.DIRECTORY: QtWidgets.QFileDialog.FileMode.Directory,
        }[mode]
    )
    if mode == Mode.SAVE:
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        if suffix := _first_suffix(file_filter):
            dialog.setDefaultSuffix(suffix)

    if dialog.testOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog):
        _customize(dialog, mode, settings, folders.values())

    with stall_logger.paused():  # a native dialog stops the Qt timers, see stall_logger.paused
        accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    if dialog.testOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog):
        settings.setValue(GEOMETRY_KEY, dialog.saveGeometry())

    paths = dialog.selectedFiles() if accepted else []
    if paths:
        folder = paths[0] if mode == Mode.DIRECTORY else os.path.dirname(paths[0])
        _remember_folder(settings, kind, folder)
    return paths


def _customize(
    dialog: QtWidgets.QFileDialog, mode: Mode, settings: QtCore.QSettings, recent_folders: Iterable[str]
) -> None:
    """Adapts Qt's dialog to the GUI: labels, sidebar, columns, buttons and size."""
    labels = QtWidgets.QFileDialog.DialogLabel
    dialog.setLabelText(labels.LookIn, "Folder:")
    dialog.setLabelText(labels.FileName, "Folder name:" if mode == Mode.DIRECTORY else "File name:")
    dialog.setLabelText(labels.FileType, "Type:")
    dialog.setLabelText(labels.Accept, {Mode.SAVE: "Save", Mode.DIRECTORY: "Select"}.get(mode, "Open"))
    dialog.setLabelText(labels.Reject, "Cancel")
    dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
    dialog.setSidebarUrls(_sidebar_urls(recent_folders))
    if (sidebar := dialog.findChild(QtWidgets.QListView, "sidebar")) is not None:
        sidebar.setMinimumWidth(SIDEBAR_MIN_WIDTH)  # Qt may restore a collapsed sidebar from an earlier session
    if (splitter := dialog.findChild(QtWidgets.QSplitter, "splitter")) is not None:
        splitter.setStretchFactor(0, 0)  # the file list, not the sidebar, takes up additional space
        splitter.setStretchFactor(1, 1)

    if (view := dialog.findChild(QtWidgets.QTreeView, "treeView")) is not None:
        view.setColumnHidden(_TYPE_COLUMN, True)  # only says "text/plain" for most trajectory files
        header = view.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(_NAME_COLUMN, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_SIZE_COLUMN, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_DATE_COLUMN, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

    if (buttons := dialog.findChild(QtWidgets.QDialogButtonBox, "buttonBox")) is not None:
        for button in buttons.buttons():
            if buttons.buttonRole(button) == QtWidgets.QDialogButtonBox.ButtonRole.RejectRole:
                button.setObjectName("secondaryButton")  # styled as a secondary action, see themes.py
                button.style().unpolish(button)  # apply the style sheet rule of the new name
                button.style().polish(button)
            elif buttons.buttonRole(button) == QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole and isinstance(
                button, QtWidgets.QPushButton
            ):
                button.setDefault(True)

    geometry = settings.value(GEOMETRY_KEY)
    if not (isinstance(geometry, QtCore.QByteArray) and dialog.restoreGeometry(geometry)):
        parent = dialog.parentWidget()
        size = DEFAULT_SIZE if parent is None else parent.window().size() * 0.75
        dialog.resize(size.expandedTo(DEFAULT_SIZE))


def _sidebar_urls(recent_folders: Iterable[str]) -> list[QtCore.QUrl]:
    """The computer, the usual folders of the user, the recently used folders and the drives."""
    locations = QtCore.QStandardPaths.StandardLocation
    usual = [
        QtCore.QStandardPaths.writableLocation(location)
        for location in (locations.HomeLocation, locations.DesktopLocation, locations.DocumentsLocation)
    ] + [QtCore.QStandardPaths.writableLocation(locations.DownloadLocation)]
    drives = [drive.absolutePath() for drive in QtCore.QDir.drives()]
    paths: list[str] = []
    for path in [*usual, *list(recent_folders)[:MAX_RECENT_FOLDERS], *drives]:
        normalized = os.path.normcase(os.path.normpath(path)) if path else ""
        if path and os.path.isdir(path) and normalized not in {os.path.normcase(os.path.normpath(p)) for p in paths}:
            paths.append(path)
    return [QtCore.QUrl("file:"), *(QtCore.QUrl.fromLocalFile(path) for path in paths)]


def _kind(mode: Mode, file_filter: str) -> str:
    """Kind of file the dialog is for, e.g. "traj"; each kind remembers its own folder."""
    if mode == Mode.DIRECTORY:
        return "directory"
    return _first_suffix(file_filter) or "any"


def _first_suffix(file_filter: str) -> str:
    match = re.search(r"\*\.(\w+)", file_filter)
    return match.group(1).lower() if match else ""


def _exists(path: str) -> bool:
    return bool(path) and (os.path.isdir(path) or os.path.isdir(os.path.dirname(path)))


def _settings() -> QtCore.QSettings:
    return QtCore.QSettings("trajectopy", "trajectopy")


def _remembered_folders(settings: QtCore.QSettings) -> dict[str, str]:
    """Kind of file to the folder used last for it, most recently used first."""
    value = settings.value(FOLDER_KEY)
    # stored as "kind<TAB>folder" strings: a list keeps the order, a map would be sorted by the registry
    entries = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    folders: dict[str, str] = {}
    for entry in entries:
        kind, _, folder = str(entry).partition("\t")
        if kind and kind not in folders and os.path.isdir(folder):
            folders[kind] = folder
    return folders


def _remember_folder(settings: QtCore.QSettings, kind: str, folder: str) -> None:
    folders = {kind: folder, **{k: v for k, v in _remembered_folders(settings).items() if k != kind}}
    settings.setValue(FOLDER_KEY, [f"{k}\t{v}" for k, v in folders.items()])
