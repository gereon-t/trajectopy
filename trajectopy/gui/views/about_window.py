from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.gui.easter_egg import ClickCounter, play_jingle
from trajectopy.gui.utils import center_window
from trajectopy.utils.resources import FULL_ICON_FILE_PATH


class _Logo(QtWidgets.QLabel):
    """The logo, which reports left clicks. (A label of its own instead of an event filter of the window: a
    Python event filter can still be called while Qt destroys the window, after its Python object is gone.)"""

    clicked = QtCore.Signal()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class AboutGUI(QtWidgets.QMainWindow):
    def __init__(self, parent, version_str: str, year_str: str) -> None:
        super().__init__(parent=parent)
        self.setupUi()
        self.version: QtWidgets.QLabel
        self.version.setText(f"Version {version_str}")

        self.year: QtWidgets.QLabel
        self.year.setText(year_str)

        self.link_label.linkActivated.connect(self.link)
        self.link_label.setText('<a href="https://github.com/gereon-t/trajectopy">GitHub</a>')

        self._logo_clicks = ClickCounter()
        self.icon.clicked.connect(self._on_logo_clicked)

    def _on_logo_clicked(self) -> None:
        if self._logo_clicks.click():
            play_jingle()

    def link(self, linkStr):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(320, 300)
        center_window(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(320, 300))
        self.setMaximumSize(QtCore.QSize(320, 300))
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 320, 300))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.icon_image = QtGui.QPixmap(FULL_ICON_FILE_PATH)
        self.icon = _Logo(self.verticalLayoutWidget)
        self.icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.icon.setObjectName("icon")
        self.icon.setPixmap(self.icon_image)
        self.icon.setScaledContents(True)
        self.verticalLayout.addWidget(self.icon)
        self.version = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.version.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.version.setObjectName("version")
        self.verticalLayout.addWidget(self.version)
        spacerItem = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.verticalLayout.addItem(spacerItem)
        self.author = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.author.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.author.setObjectName("author")
        self.verticalLayout.addWidget(self.author)
        self.organization = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.organization.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.organization.setObjectName("organization")
        self.verticalLayout.addWidget(self.organization)
        self.link_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.link_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.link_label.setObjectName("uni")
        self.verticalLayout.addWidget(self.link_label)
        self.year = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.year.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.year.setObjectName("year")
        self.verticalLayout.addWidget(self.year)
        self.setCentralWidget(self.centralwidget)
        self.retranslateUi()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "About Trajectopy"))
        self.version.setText(_translate("MainWindow", "Version Number"))
        self.author.setText(_translate("MainWindow", "Gereon Tombrink"))
        self.organization.setText(_translate("MainWindow", "University of Bonn"))
        self.year.setText(_translate("MainWindow", "2024"))
