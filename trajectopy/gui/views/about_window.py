"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
from PyQt6 import QtCore, QtGui, QtWidgets

from trajectopy.gui.path import FULL_ICON_FILE_PATH


class AboutGUI(QtWidgets.QMainWindow):
    def __init__(self, parent, version_str: str, year_str: str) -> None:
        super().__init__(parent=parent)
        self.setupUi()
        self.version: QtWidgets.QLabel
        self.version.setText(version_str)

        self.year: QtWidgets.QLabel
        self.year.setText(year_str)

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(320, 270)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(320, 270))
        self.setMaximumSize(QtCore.QSize(320, 270))
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 320, 270))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.icon_image = QtGui.QPixmap(FULL_ICON_FILE_PATH)
        self.icon = QtWidgets.QLabel(self.verticalLayoutWidget)
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
        self.uni = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.uni.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.uni.setObjectName("uni")
        self.verticalLayout.addWidget(self.uni)
        self.year = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.year.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.year.setObjectName("year")
        self.verticalLayout.addWidget(self.year)
        self.setCentralWidget(self.centralwidget)
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "About Trajectopy"))
        # self.name.setText(_translate("MainWindow", "Trajectopy"))
        self.version.setText(_translate("MainWindow", "Version Number"))
        self.author.setText(_translate("MainWindow", "Gereon Tombrink"))
        self.uni.setText(_translate("MainWindow", "University of Bonn"))
        self.year.setText(_translate("MainWindow", "2023"))
