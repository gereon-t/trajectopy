import logging

from PyQt6 import QtCore, QtWidgets

from trajectopy.gui.models.entries import AlignmentEntry
from trajectopy.gui.utils import center_window

logger = logging.getLogger(__name__)


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
        self.resize(280, 480)
        center_window(self)
        self.setMinimumSize(QtCore.QSize(280, 480))
        self.setMaximumSize(QtCore.QSize(280, 480))

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # --- Helmert group ---
        helmert_group = QtWidgets.QGroupBox("Helmert Parameters")
        helmert_layout = QtWidgets.QVBoxLayout(helmert_group)

        self.trans_x = QtWidgets.QCheckBox("x translation")
        self.trans_y = QtWidgets.QCheckBox("y translation")
        self.trans_z = QtWidgets.QCheckBox("z translation")
        self.rot_x = QtWidgets.QCheckBox("x rotation")
        self.rot_y = QtWidgets.QCheckBox("y rotation")
        self.rot_z = QtWidgets.QCheckBox("z rotation")
        self.scale = QtWidgets.QCheckBox("scale")

        for cb in (self.trans_x, self.trans_y, self.trans_z, self.rot_x, self.rot_y, self.rot_z, self.scale):
            helmert_layout.addWidget(cb)

        main_layout.addWidget(helmert_group)

        # --- Time shift group ---
        time_group = QtWidgets.QGroupBox("Temporal")
        time_layout = QtWidgets.QVBoxLayout(time_group)
        self.time_shift = QtWidgets.QCheckBox("Time shift")
        time_layout.addWidget(self.time_shift)
        main_layout.addWidget(time_group)

        # --- Lever arm group ---
        lever_group = QtWidgets.QGroupBox("Lever Arm")
        lever_layout = QtWidgets.QVBoxLayout(lever_group)
        self.lever_x = QtWidgets.QCheckBox("x lever")
        self.lever_y = QtWidgets.QCheckBox("y lever")
        self.lever_z = QtWidgets.QCheckBox("z lever")
        for cb in (self.lever_x, self.lever_y, self.lever_z):
            lever_layout.addWidget(cb)
        main_layout.addWidget(lever_group)

        # --- Sensor rotation group ---
        sensor_group = QtWidgets.QGroupBox("Sensor")
        sensor_layout = QtWidgets.QVBoxLayout(sensor_group)
        self.sensorRot = QtWidgets.QCheckBox("Sensor rotation")
        sensor_layout.addWidget(self.sensorRot)
        main_layout.addWidget(sensor_group)

        # --- Buttons ---
        main_layout.addStretch()
        button_layout = QtWidgets.QHBoxLayout()
        self.cancelButton = QtWidgets.QPushButton("Cancel")
        self.okButton = QtWidgets.QPushButton("OK")
        button_layout.addWidget(self.cancelButton)
        button_layout.addWidget(self.okButton)
        main_layout.addLayout(button_layout)

    def retranslateUi(self):
        pass

    def update_view(self) -> None:
        helmert_enabled = self.alignment_entry.estimation_of.helmert_enabled
        leverarm_enabled = self.alignment_entry.estimation_of.leverarm_enabled
        self.trans_x.setChecked(self.alignment_entry.estimated_parameters.sim_trans_x.enabled)
        self.trans_x.setEnabled(self.alignment_entry.estimation_of.translation_x and helmert_enabled)

        self.trans_y.setChecked(self.alignment_entry.estimated_parameters.sim_trans_y.enabled)
        self.trans_y.setEnabled(self.alignment_entry.estimation_of.translation_y and helmert_enabled)

        self.trans_z.setChecked(self.alignment_entry.estimated_parameters.sim_trans_z.enabled)
        self.trans_z.setEnabled(self.alignment_entry.estimation_of.translation_z and helmert_enabled)

        self.rot_x.setChecked(self.alignment_entry.estimated_parameters.sim_rot_x.enabled)
        self.rot_x.setEnabled(self.alignment_entry.estimation_of.rotation_x and helmert_enabled)

        self.rot_y.setChecked(self.alignment_entry.estimated_parameters.sim_rot_y.enabled)
        self.rot_y.setEnabled(self.alignment_entry.estimation_of.rotation_y and helmert_enabled)

        self.rot_z.setChecked(self.alignment_entry.estimated_parameters.sim_rot_z.enabled)
        self.rot_z.setEnabled(self.alignment_entry.estimation_of.rotation_z and helmert_enabled)

        self.scale.setChecked(self.alignment_entry.estimated_parameters.sim_scale.enabled)
        self.scale.setEnabled(self.alignment_entry.estimation_of.scale and helmert_enabled)

        self.time_shift.setChecked(self.alignment_entry.estimated_parameters.time_shift.enabled)
        self.time_shift.setEnabled(self.alignment_entry.estimation_of.time_shift_enabled)

        self.lever_x.setChecked(self.alignment_entry.estimated_parameters.lever_x.enabled)
        self.lever_x.setEnabled(self.alignment_entry.estimation_of.leverarm_x and leverarm_enabled)

        self.lever_y.setChecked(self.alignment_entry.estimated_parameters.lever_y.enabled)
        self.lever_y.setEnabled(self.alignment_entry.estimation_of.leverarm_y and leverarm_enabled)

        self.lever_z.setChecked(self.alignment_entry.estimated_parameters.lever_z.enabled)
        self.lever_z.setEnabled(self.alignment_entry.estimation_of.leverarm_z and leverarm_enabled)

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
