import logging

from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.gui.log import MAX_BUFFERED_RECORDS, QtLogHandler

# readable on the dark and on the light theme
_LEVEL_COLORS = {
    logging.WARNING: "#d18616",
    logging.ERROR: "#e51400",
}


def _level_color(level: int) -> str | None:
    for threshold in (logging.ERROR, logging.WARNING):
        if level >= threshold:
            return _LEVEL_COLORS[threshold]
    return None


class LogPanel(QtWidgets.QDockWidget):
    """Log messages of the application; docked at the bottom of the main window."""

    def __init__(self, handler: QtLogHandler, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Log", parent)
        self.setObjectName("logPanel")
        self.setAllowedAreas(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(MAX_BUFFERED_RECORDS)
        self._text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._text.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._text.customContextMenuRequested.connect(self._show_context_menu)
        self.setWidget(self._text)

        for message, level in handler.records:
            self._append(message, level)
        handler.record_emitted.connect(self._append)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(super().sizeHint().width(), 180)

    @QtCore.Slot(str, int)
    def _append(self, message: str, level: int) -> None:
        scrollbar = self._text.verticalScrollBar()
        at_end = scrollbar.value() >= scrollbar.maximum() - 4
        text_format = QtGui.QTextCharFormat()  # default: the text color of the theme
        color = _level_color(level)
        if color is not None:
            text_format.setForeground(QtGui.QColor(color))
        cursor = QtGui.QTextCursor(self._text.document())
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        if not self._text.document().isEmpty():
            cursor.insertBlock()
        cursor.insertText(message, text_format)
        if at_end:  # keeps the position while the user reads older messages
            scrollbar.setValue(scrollbar.maximum())

    @QtCore.Slot(QtCore.QPoint)
    def _show_context_menu(self, position: QtCore.QPoint) -> None:
        menu = self._text.createStandardContextMenu()
        menu.addSeparator()
        clear_action = QtGui.QAction("Clear", menu)
        clear_action.triggered.connect(self._text.clear)
        menu.addAction(clear_action)
        menu.exec(self._text.mapToGlobal(position))
        menu.deleteLater()
