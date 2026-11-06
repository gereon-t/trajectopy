"""Splash screen shown while the application starts (kept free of heavy imports)."""

from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.__version__ import __version__ as VERSION
from trajectopy.utils.resources import FULL_ICON_FILE_PATH

WIDTH = 440
HEIGHT = 300
LOGO_WIDTH = 300


class SplashScreen(QtWidgets.QSplashScreen):
    """The logo, the version and a status message (see `show_status`) on the colors of the theme."""

    def __init__(self, dark: bool) -> None:
        self._background = QtGui.QColor("#252526" if dark else "#ffffff")
        self._border = QtGui.QColor("#3c3c3c" if dark else "#cccccc")
        self._text_color = QtGui.QColor("#bbbbbb" if dark else "#555555")
        super().__init__(self._pixmap())

    def show_status(self, message: str) -> None:
        self.showMessage(message)
        QtWidgets.QApplication.processEvents()

    def drawContents(self, painter: QtGui.QPainter) -> None:
        # with a margin: QSplashScreen would draw the message at the very edge
        painter.setPen(self._text_color)
        painter.drawText(
            self.rect().adjusted(16, 0, -16, -14),
            QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignLeft,
            self.message(),
        )

    def _pixmap(self) -> QtGui.QPixmap:
        screen = QtGui.QGuiApplication.primaryScreen()
        ratio = screen.devicePixelRatio() if screen is not None else 1.0
        pixmap = QtGui.QPixmap(int(WIDTH * ratio), int(HEIGHT * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(self._background)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.setPen(self._border)
        painter.drawRect(QtCore.QRectF(0.5, 0.5, WIDTH - 1, HEIGHT - 1))

        logo = QtGui.QPixmap(FULL_ICON_FILE_PATH)
        if not logo.isNull():
            logo_height = LOGO_WIDTH * logo.height() / logo.width()
            painter.drawPixmap(QtCore.QRectF((WIDTH - LOGO_WIDTH) / 2, 36, LOGO_WIDTH, logo_height), logo, logo.rect())
            text_top = 36 + logo_height + 12
        else:
            text_top = 120

        painter.setPen(self._text_color)
        painter.drawText(
            QtCore.QRectF(0, text_top, WIDTH, 24), QtCore.Qt.AlignmentFlag.AlignHCenter, f"Version {VERSION}"
        )
        painter.end()
        return pixmap
