"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging

from PyQt6 import QtCore, QtGui, QtWidgets
from trajectopy_core.util.entries import AlignmentEntry

logger = logging.getLogger("root")


class AlignmentEditWindow(QtWidgets.QMainWindow):
    update_signal = QtCore.pyqtSignal()

    def __init__(self, parent, alignment_entry: AlignmentEntry) -> None:
        super().__init__(parent=parent)
        self.setupUi()
        self.alignment_entry = alignment_entry
        self.setWindowTitle(f"Alignment Parameter Settings for: {alignment_entry.name}")

        self.update_view()

        self.okButton.clicked.connect(self.handle_ok)
        self.cancelButton.clicked.connect(self.close)

    def show(self) -> None:
        self.update_view()
        super().show()

    def setupUi(self):
        self.setObjectName("Form")
        self.resize(250, 400)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(250, 400))
        self.setMaximumSize(QtCore.QSize(250, 400))
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 250, 400))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.trans_x = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.trans_x.setObjectName("checkBox_2")
        self.verticalLayout.addWidget(self.trans_x)
        self.trans_y = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.trans_y.setObjectName("checkBox_3")
        self.verticalLayout.addWidget(self.trans_y)
        self.trans_z = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.trans_z.setObjectName("checkBox_4")
        self.verticalLayout.addWidget(self.trans_z)
        self.rot_x = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.rot_x.setObjectName("checkBox_5")
        self.verticalLayout.addWidget(self.rot_x)
        self.rot_y = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.rot_y.setObjectName("checkBox_6")
        self.verticalLayout.addWidget(self.rot_y)
        self.rot_z = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.rot_z.setObjectName("checkBox_7")
        self.verticalLayout.addWidget(self.rot_z)
        self.scale = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.scale.setObjectName("checkBox_8")
        self.verticalLayout.addWidget(self.scale)
        spacerItem = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.verticalLayout.addItem(spacerItem)
        self.time_shift = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.time_shift.setObjectName("checkBox_9")
        self.verticalLayout.addWidget(self.time_shift)
        spacerItem1 = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.verticalLayout.addItem(spacerItem1)
        self.lever_x = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.lever_x.setObjectName("checkBox_11")
        self.verticalLayout.addWidget(self.lever_x)
        self.lever_y = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.lever_y.setObjectName("checkBox_12")
        self.verticalLayout.addWidget(self.lever_y)
        self.lever_z = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.lever_z.setObjectName("checkBox_13")
        self.verticalLayout.addWidget(self.lever_z)

        spacerItem2 = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.verticalLayout.addItem(spacerItem2)
        self.sensorRot = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.sensorRot.setObjectName("checkBox_14")
        self.verticalLayout.addWidget(self.sensorRot)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cancelButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.cancelButton.setObjectName("cancelButton")
        self.horizontalLayout.addWidget(self.cancelButton)
        self.okButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.okButton.setObjectName("okButton")
        self.horizontalLayout.addWidget(self.okButton)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Enabled Parameters:"))
        self.trans_x.setText(_translate("Form", "x translation"))
        self.trans_y.setText(_translate("Form", "y translation"))
        self.trans_z.setText(_translate("Form", "z translation"))
        self.rot_x.setText(_translate("Form", "x rotation"))
        self.rot_y.setText(_translate("Form", "y rotation"))
        self.rot_z.setText(_translate("Form", "z rotation"))
        self.scale.setText(_translate("Form", "scale"))
        self.time_shift.setText(_translate("Form", "Time shift"))
        self.lever_x.setText(_translate("Form", "x lever"))
        self.lever_y.setText(_translate("Form", "y lever"))
        self.lever_z.setText(_translate("Form", "z lever"))
        self.sensorRot.setText(_translate("Form", "Sensor rotation"))
        self.cancelButton.setText(_translate("Form", "Cancel"))
        self.okButton.setText(_translate("Form", "OK"))

    def update_view(self) -> None:
        helmert_enabled = self.alignment_entry.estimation_of.helmert_enabled
        leverarm_enabled = self.alignment_entry.estimation_of.leverarm_enabled
        self.trans_x.setChecked(self.alignment_entry.estimated_parameters.sim_trans_x.enabled)
        self.trans_x.setEnabled(self.alignment_entry.estimation_of.trans_x and helmert_enabled)

        self.trans_y.setChecked(self.alignment_entry.estimated_parameters.sim_trans_y.enabled)
        self.trans_y.setEnabled(self.alignment_entry.estimation_of.trans_y and helmert_enabled)

        self.trans_z.setChecked(self.alignment_entry.estimated_parameters.sim_trans_z.enabled)
        self.trans_z.setEnabled(self.alignment_entry.estimation_of.trans_z and helmert_enabled)

        self.rot_x.setChecked(self.alignment_entry.estimated_parameters.sim_rot_x.enabled)
        self.rot_x.setEnabled(self.alignment_entry.estimation_of.rot_x and helmert_enabled)

        self.rot_y.setChecked(self.alignment_entry.estimated_parameters.sim_rot_y.enabled)
        self.rot_y.setEnabled(self.alignment_entry.estimation_of.rot_y and helmert_enabled)

        self.rot_z.setChecked(self.alignment_entry.estimated_parameters.sim_rot_z.enabled)
        self.rot_z.setEnabled(self.alignment_entry.estimation_of.rot_z and helmert_enabled)

        self.scale.setChecked(self.alignment_entry.estimated_parameters.sim_scale.enabled)
        self.scale.setEnabled(self.alignment_entry.estimation_of.scale and helmert_enabled)

        self.time_shift.setChecked(self.alignment_entry.estimated_parameters.time_shift.enabled)
        self.time_shift.setEnabled(self.alignment_entry.estimation_of.time_shift_enabled)

        self.lever_x.setChecked(self.alignment_entry.estimated_parameters.lever_x.enabled)
        self.lever_x.setEnabled(self.alignment_entry.estimation_of.lever_x and leverarm_enabled)

        self.lever_y.setChecked(self.alignment_entry.estimated_parameters.lever_y.enabled)
        self.lever_y.setEnabled(self.alignment_entry.estimation_of.lever_y and leverarm_enabled)

        self.lever_z.setChecked(self.alignment_entry.estimated_parameters.lever_z.enabled)
        self.lever_z.setEnabled(self.alignment_entry.estimation_of.lever_z and leverarm_enabled)

        self.sensorRot.setChecked(self.alignment_entry.alignment_result.rotation_parameters.any_enabled)
        self.sensorRot.setEnabled(self.alignment_entry.estimation_of.sensor_rotation)

    def update_model(self) -> None:
        self.alignment_entry.estimated_parameters.sim_trans_x.enabled = self.trans_x.isChecked()
        self.alignment_entry.estimated_parameters.sim_trans_y.enabled = self.trans_y.isChecked()
        self.alignment_entry.estimated_parameters.sim_trans_z.enabled = self.trans_z.isChecked()
        self.alignment_entry.estimated_parameters.sim_scale.enabled = self.scale.isChecked()
        self.alignment_entry.estimated_parameters.sim_rot_x.enabled = self.rot_x.isChecked()
        self.alignment_entry.estimated_parameters.sim_rot_y.enabled = self.rot_y.isChecked()
        self.alignment_entry.estimated_parameters.sim_rot_z.enabled = self.rot_z.isChecked()

        self.alignment_entry.estimated_parameters.lever_x.enabled = self.lever_x.isChecked()
        self.alignment_entry.estimated_parameters.lever_y.enabled = self.lever_y.isChecked()
        self.alignment_entry.estimated_parameters.lever_z.enabled = self.lever_z.isChecked()

        self.alignment_entry.estimated_parameters.time_shift.enabled = self.time_shift.isChecked()

        self.alignment_entry.alignment_result.rotation_parameters.enabled_bool_list = [self.sensorRot.isChecked()] * 3

    def handle_ok(self) -> None:
        self.update_model()
        self.update_signal.emit()
        self.close()
