from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QGuiApplication

from trajectopy.gui.models.selection import TrajectorySelection


class DOFOrganizer(QtWidgets.QMainWindow):
    selection_made = QtCore.pyqtSignal(dict)

    def __init__(self, parent, selection: TrajectorySelection) -> None:
        super().__init__(parent=parent)
        self.setupUi()
        self.selection = selection
        self.pushButton.clicked.connect(self.handle_ok)

    def handle_ok(self):
        self.selection_made.emit(
            {
                "x": {
                    "sign": self.x_sign.currentText(),
                    "target": self.x_combo.currentText(),
                    "bias": self.x_bias.value(),
                },
                "y": {
                    "sign": self.y_sign.currentText(),
                    "target": self.y_combo.currentText(),
                    "bias": self.y_bias.value(),
                },
                "z": {
                    "sign": self.z_sign.currentText(),
                    "target": self.z_combo.currentText(),
                    "bias": self.z_bias.value(),
                },
                "roll": {
                    "sign": self.roll_sign.currentText(),
                    "target": self.roll_combo.currentText(),
                    "bias": self.roll_bias.value(),
                },
                "pitch": {
                    "sign": self.pitch_sign.currentText(),
                    "target": self.pitch_combo.currentText(),
                    "bias": self.pitch_bias.value(),
                },
                "yaw": {
                    "sign": self.yaw_sign.currentText(),
                    "target": self.yaw_combo.currentText(),
                    "bias": self.yaw_bias.value(),
                },
                "time": {
                    "sign": self.time_sign.currentText(),
                    "target": self.time_combo.currentText(),
                    "bias": self.time_bias.value(),
                },
            },
        )

        self.close()

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(600, 280)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(600, 280))
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        desired_pos = QtCore.QPoint(screen_geometry.center().x() - 300, screen_geometry.center().y() - 140)
        self.move(desired_pos)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        self.label = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label, 0, 0)
        self.x_sign = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.x_sign, 0, 1)
        self.x_combo = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.x_combo, 0, 2)
        self.label_2 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_2, 0, 3)
        self.x_bias = QtWidgets.QDoubleSpinBox()
        self.x_bias.setMinimum(-10000.0)
        self.x_bias.setMaximum(10000.0)
        self.x_bias.setDecimals(4)
        self.x_bias.setSingleStep(0.0001)
        self.gridLayout.addWidget(self.x_bias, 0, 4)

        self.label_3 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_3, 1, 0)
        self.y_sign = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.y_sign, 1, 1)
        self.y_combo = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.y_combo, 1, 2)
        self.label_4 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_4, 1, 3)
        self.y_bias = QtWidgets.QDoubleSpinBox()
        self.y_bias.setMinimum(-10000.0)
        self.y_bias.setMaximum(10000.0)
        self.y_bias.setDecimals(4)
        self.y_bias.setSingleStep(0.0001)
        self.gridLayout.addWidget(self.y_bias, 1, 4)

        self.label_5 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_5, 2, 0)
        self.z_sign = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.z_sign, 2, 1)
        self.z_combo = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.z_combo, 2, 2)
        self.label_6 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_6, 2, 3)
        self.z_bias = QtWidgets.QDoubleSpinBox()
        self.z_bias.setMinimum(-10000.0)
        self.z_bias.setMaximum(10000.0)
        self.z_bias.setDecimals(4)
        self.z_bias.setSingleStep(0.0001)
        self.gridLayout.addWidget(self.z_bias, 2, 4)

        self.label_7 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_7, 3, 0)
        self.roll_sign = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.roll_sign, 3, 1)
        self.roll_combo = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.roll_combo, 3, 2)
        self.label_8 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_8, 3, 3)
        self.roll_bias = QtWidgets.QDoubleSpinBox()
        self.roll_bias.setMinimum(-360.0)
        self.roll_bias.setMaximum(360.0)
        self.roll_bias.setDecimals(4)
        self.roll_bias.setSingleStep(0.0001)
        self.gridLayout.addWidget(self.roll_bias, 3, 4)

        self.label_9 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_9, 4, 0)
        self.pitch_sign = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.pitch_sign, 4, 1)
        self.pitch_combo = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.pitch_combo, 4, 2)
        self.label_10 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_10, 4, 3)
        self.pitch_bias = QtWidgets.QDoubleSpinBox()
        self.pitch_bias.setMinimum(-360.0)
        self.pitch_bias.setMaximum(360.0)
        self.pitch_bias.setDecimals(4)
        self.pitch_bias.setSingleStep(0.0001)
        self.gridLayout.addWidget(self.pitch_bias, 4, 4)

        self.label_11 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_11, 5, 0)
        self.yaw_sign = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.yaw_sign, 5, 1)
        self.yaw_combo = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.yaw_combo, 5, 2)
        self.label_12 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_12, 5, 3)
        self.yaw_bias = QtWidgets.QDoubleSpinBox()
        self.yaw_bias.setMinimum(-360.0)
        self.yaw_bias.setMaximum(360.0)
        self.yaw_bias.setDecimals(4)
        self.yaw_bias.setSingleStep(0.0001)
        self.gridLayout.addWidget(self.yaw_bias, 5, 4)

        self.label_13 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_13, 6, 0)
        self.time_sign = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.time_sign, 6, 1)
        self.time_combo = QtWidgets.QComboBox()
        self.gridLayout.addWidget(self.time_combo, 6, 2)
        self.label_14 = QtWidgets.QLabel()
        self.gridLayout.addWidget(self.label_14, 6, 3)
        self.time_bias = QtWidgets.QDoubleSpinBox()
        self.time_bias.setMinimum(-86400)
        self.time_bias.setMaximum(86400)
        self.time_bias.setDecimals(4)
        self.time_bias.setSingleStep(0.0001)
        self.gridLayout.addWidget(self.time_bias, 6, 4)

        dof_options = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Time"]
        sign_options = ["+", "-"]
        all_combos = [
            self.x_combo,
            self.y_combo,
            self.z_combo,
            self.roll_combo,
            self.pitch_combo,
            self.yaw_combo,
            self.time_combo,
        ]
        all_signs = [
            self.x_sign,
            self.y_sign,
            self.z_sign,
            self.roll_sign,
            self.pitch_sign,
            self.yaw_sign,
            self.time_sign,
        ]

        for combo in all_combos:
            combo.addItems(dof_options)

        for sign_combo in all_signs:
            sign_combo.addItems(sign_options)

        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.gridLayout.addItem(spacerItem, 7, 0, 1, 5)
        self.pushButton = QtWidgets.QPushButton()
        self.gridLayout.addWidget(self.pushButton, 8, 0, 1, 5)

        self.retranslateUi()

        # Set default selections
        self.x_combo.setCurrentIndex(0)
        self.y_combo.setCurrentIndex(1)
        self.z_combo.setCurrentIndex(2)
        self.roll_combo.setCurrentIndex(3)
        self.pitch_combo.setCurrentIndex(4)
        self.yaw_combo.setCurrentIndex(5)
        self.time_combo.setCurrentIndex(6)

        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "DOF Organizer"))

        self.label.setText(_translate("MainWindow", "X"))
        self.label_2.setText(_translate("MainWindow", "+"))
        self.label_3.setText(_translate("MainWindow", "Y"))
        self.label_4.setText(_translate("MainWindow", "+"))
        self.label_5.setText(_translate("MainWindow", "Z"))
        self.label_6.setText(_translate("MainWindow", "+"))
        self.label_7.setText(_translate("MainWindow", "Roll"))
        self.label_8.setText(_translate("MainWindow", "+ (deg)"))
        self.label_9.setText(_translate("MainWindow", "Pitch"))
        self.label_10.setText(_translate("MainWindow", "+ (deg)"))
        self.label_11.setText(_translate("MainWindow", "Yaw"))
        self.label_12.setText(_translate("MainWindow", "+ (deg)"))
        self.label_13.setText(_translate("MainWindow", "Time"))
        self.label_14.setText(_translate("MainWindow", "+ (s)"))

        self.pushButton.setText(_translate("MainWindow", "OK"))
