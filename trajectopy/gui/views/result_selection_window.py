"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
from typing import List

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QGuiApplication
from trajectopy_core.util.entries import AlignmentEntry


class AlignmentSelector(QtWidgets.QMainWindow):
    selection_made = QtCore.pyqtSignal(AlignmentEntry)

    def __init__(self, alignments: List[AlignmentEntry], parent=None) -> None:
        super().__init__(parent=parent)
        self.setupUi()
        self.alignments = alignments
        self.set_options(
            [f"{alignment.name}, {alignment.type}, {alignment.time}, {alignment.entry_id}" for alignment in alignments]
        )
        self.centralwidget.setLayout(self.verticalLayout)
        self.okButton.clicked.connect(self.handle_ok)
        self.cancelButton.clicked.connect(self.close)

    def setupUi(self):
        self.setObjectName("Form")
        self.resize(400, 300)

        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        desired_pos = QtCore.QPoint(screen_geometry.center().x() - 200, screen_geometry.center().y() - 150)
        self.move(desired_pos)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 381, 281))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.centralwidget.setLayout(self.verticalLayout)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.comboBox = QtWidgets.QComboBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout.addWidget(self.comboBox)
        spacerItem1 = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(-1, -1, -1, 10)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem2 = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.horizontalLayout_2.addItem(spacerItem2)
        self.cancelButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.cancelButton.setObjectName("cancelButton")
        self.horizontalLayout_2.addWidget(self.cancelButton)
        self.okButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.okButton.setObjectName("okButton")
        self.horizontalLayout_2.addWidget(self.okButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.setCentralWidget(self.centralwidget)
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Form", "Select Alignment"))
        self.label.setText(_translate("Form", "Select Alignment:"))
        self.cancelButton.setText(_translate("Form", "Cancel"))
        self.okButton.setText(_translate("Form", "OK"))

    def handle_ok(self) -> None:
        self.selection_made.emit(self.alignments[self.comboBox.currentIndex()])
        self.close()

    def set_options(self, options: List[str]) -> None:
        self.comboBox.clear()
        self.comboBox.addItems(options)
