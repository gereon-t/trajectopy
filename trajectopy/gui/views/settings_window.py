"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import QCoreApplication, QMetaObject, QPoint, QRect, QSize, Qt, pyqtSlot
from PyQt6.QtGui import QFont, QGuiApplication
from trajectopy_core.settings.processing_settings import ProcessingSettings
from trajectopy_core.util.definitions import RotApprox, Unit
from trajectopy_core.util.entries import TrajectoryEntry

from trajectopy.gui.util import read_file_dialog, save_file_dialog, show_msg_box

logger = logging.getLogger("root")


ROT_APPROX_DICT = {0: RotApprox.WINDOW, 1: RotApprox.INTERP}
UNIT_DICT = {0: Unit.METER, 1: Unit.SECOND}


class SettingsGUI(QtWidgets.QMainWindow):
    def __init__(self, parent, trajectory_entry: TrajectoryEntry) -> None:
        super().__init__(parent=parent)
        self.trajectory_entry = trajectory_entry
        self.setupUi()

        self.setWindowTitle(f"Trajectory Settings for: {self.trajectory_entry.name}")

        self.alignSimilarity.stateChanged.connect(self.handle_similarity_changed)
        self.alignTimeShift.stateChanged.connect(self.handle_timeshift_changed)
        self.alignLeverarm.stateChanged.connect(self.handle_leverarm_changed)

        self.okButton.clicked.connect(self.handle_ok)
        self.loadButton.clicked.connect(self.handle_load)
        self.saveButton.clicked.connect(self.handle_save)
        self.cancelButton.clicked.connect(self.close)
        self.update_view()

    def setupUi(self) -> None:
        self.setObjectName("MainWindow")
        self.resize(600, 540)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QSize(600, 540))
        self.setMaximumSize(QSize(600, 540))

        if (primary_screen := QGuiApplication.primaryScreen()) is not None:
            screen_geometry = primary_screen.availableGeometry()
            self.resize(screen_geometry.width(), screen_geometry.height() - 50)

        desired_pos = QPoint(screen_geometry.center().x() - 300, screen_geometry.center().y() - 320)
        self.move(desired_pos)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setMinimumSize(QSize(600, 540))
        self.centralwidget.setMaximumSize(QSize(600, 540))
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QRect(0, 0, 600, 540))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QSize(600, 540))
        self.tabWidget.setMaximumSize(QSize(600, 540))
        self.tabWidget.setObjectName("tabWidget")

        self.alignmentTab = QtWidgets.QWidget()
        self.alignmentTab.setObjectName("alignmentTab")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.alignmentTab)
        self.verticalLayoutWidget.setGeometry(QRect(0, 0, 600, 540))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.alignmentVerticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.alignmentVerticalLayout.setContentsMargins(10, 10, 10, 10)
        self.alignmentVerticalLayout.setObjectName("verticalLayout")
        self.preprocessing_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QFont()
        font.setBold(True)

        self.preprocessing_label.setFont(font)
        self.preprocessing_label.setObjectName("label_15")
        self.alignmentVerticalLayout.addWidget(self.preprocessing_label)
        self.preprocessingHorizontalLayout = QtWidgets.QHBoxLayout()
        self.preprocessingHorizontalLayout.setObjectName("horizontalLayout")
        self.minimum_speed_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.minimum_speed_label.setObjectName("label_18")
        self.preprocessingHorizontalLayout.addWidget(self.minimum_speed_label)
        self.alignMinSpeed = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.alignMinSpeed.setMinimum(0.0)
        self.alignMinSpeed.setSingleStep(0.1)
        self.alignMinSpeed.setObjectName("alignMinSpeed")
        self.preprocessingHorizontalLayout.addWidget(self.alignMinSpeed)
        self.time_start_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.time_start_label.setObjectName("label_19")
        self.preprocessingHorizontalLayout.addWidget(self.time_start_label)
        self.alignTimeStart = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.alignTimeStart.setDecimals(3)
        self.alignTimeStart.setSingleStep(0.01)
        self.alignTimeStart.setObjectName("alignTimeStart")
        self.preprocessingHorizontalLayout.addWidget(self.alignTimeStart)
        self.time_end_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.time_end_label.setObjectName("label_20")
        self.preprocessingHorizontalLayout.addWidget(self.time_end_label)
        self.alignTimeEnd = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.alignTimeEnd.setDecimals(3)
        self.alignTimeEnd.setMinimum(0.0)
        self.alignTimeEnd.setMaximum(85400.0)
        self.alignTimeEnd.setSingleStep(0.01)
        self.alignTimeEnd.setValue(0.0)
        self.alignTimeEnd.setObjectName("alignTimeEnd")
        self.preprocessingHorizontalLayout.addWidget(self.alignTimeEnd)
        spacerItemPreprocessing = QtWidgets.QSpacerItem(
            0,
            0,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.preprocessingHorizontalLayout.addItem(spacerItemPreprocessing)
        self.alignmentVerticalLayout.addLayout(self.preprocessingHorizontalLayout)

        self.stochastics_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.stochastics_label.setFont(font)
        self.stochastics_label.setObjectName("label_16")
        self.alignmentVerticalLayout.addWidget(self.stochastics_label)

        self.errorProbHorizontalLayout = QtWidgets.QHBoxLayout()
        self.errorProbHorizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.errorProbHorizontalLayout.setObjectName("horizontalLayout_9")
        self.error_probability_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.error_probability_label.setObjectName("label_5")
        self.errorProbHorizontalLayout.addWidget(self.error_probability_label)
        self.alignErrorProb = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.alignErrorProb.setProperty("value", 5.0)
        self.alignErrorProb.setObjectName("alignErrorProb")
        self.errorProbHorizontalLayout.addWidget(self.alignErrorProb)
        spacerItem2 = QtWidgets.QSpacerItem(
            0,
            0,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.errorProbHorizontalLayout.addItem(spacerItem2)
        self.alignmentVerticalLayout.addLayout(self.errorProbHorizontalLayout)

        self.stochasticsHorizontalLayout = QtWidgets.QHBoxLayout()
        self.stochasticsHorizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.stochasticsHorizontalLayout.setObjectName("horizontalLayout_12")
        self.stochasticsFormLayout = QtWidgets.QFormLayout()
        self.stochasticsFormLayout.setContentsMargins(0, 0, 0, 0)
        self.stochasticsFormLayout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.stochasticsFormLayout.setFormAlignment(Qt.AlignmentFlag.AlignRight)
        self.stochasticsFormLayout.setObjectName("formLayout_5")
        self.s_dev_xy_source_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.s_dev_xy_source_label.setObjectName("label_33")
        self.stochasticsFormLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.s_dev_xy_source_label)
        self.alignStdXYFrom = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.alignStdXYFrom.setDecimals(4)
        self.alignStdXYFrom.setMinimum(0.0001)
        self.alignStdXYFrom.setSingleStep(0.001)
        self.alignStdXYFrom.setProperty("value", 0.02)
        self.alignStdXYFrom.setObjectName("alignStdXYFrom")
        self.stochasticsFormLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.alignStdXYFrom)
        self.s_dev_z_source_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.s_dev_z_source_label.setObjectName("label_37")
        self.stochasticsFormLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.s_dev_z_source_label)
        self.alignStdZFrom = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.alignStdZFrom.setDecimals(4)
        self.alignStdZFrom.setMinimum(0.0001)
        self.alignStdZFrom.setSingleStep(0.001)
        self.alignStdZFrom.setProperty("value", 0.05)
        self.alignStdZFrom.setObjectName("alignStdZFrom")
        self.stochasticsFormLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.alignStdZFrom)
        self.s_dev_rp_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.s_dev_rp_label.setObjectName("label_38")
        self.stochasticsFormLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.s_dev_rp_label)
        self.alignStdRP = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.alignStdRP.setDecimals(4)
        self.alignStdRP.setMinimum(0.0001)
        self.alignStdRP.setSingleStep(0.001)
        self.alignStdRP.setProperty("value", 0.03)
        self.alignStdRP.setObjectName("alignStdRP")
        self.stochasticsFormLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.alignStdRP)

        self.stochasticsFormLayout2 = QtWidgets.QFormLayout()
        self.stochasticsFormLayout2.setContentsMargins(0, 0, 0, 0)
        self.stochasticsFormLayout2.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.stochasticsFormLayout2.setFormAlignment(Qt.AlignmentFlag.AlignRight)
        self.stochasticsFormLayout2.setObjectName("formLayout_6")
        self.s_dev_xy_target_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.s_dev_xy_target_label.setObjectName("label_39")
        self.stochasticsFormLayout2.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.s_dev_xy_target_label)
        self.s_dev_z_target_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.s_dev_z_target_label.setObjectName("label_40")
        self.stochasticsFormLayout2.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.s_dev_z_target_label)
        self.s_dev_yaw_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.s_dev_yaw_label.setObjectName("label_41")
        self.stochasticsFormLayout2.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.s_dev_yaw_label)
        self.alignStdXYTo = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.alignStdXYTo.setDecimals(4)
        self.alignStdXYTo.setMinimum(0.0001)
        self.alignStdXYTo.setSingleStep(0.001)
        self.alignStdXYTo.setProperty("value", 0.004)
        self.alignStdXYTo.setObjectName("alignStdXYTo")
        self.stochasticsFormLayout2.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.alignStdXYTo)
        self.alignStdZTo = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.alignStdZTo.setDecimals(4)
        self.alignStdZTo.setMinimum(0.0001)
        self.alignStdZTo.setSingleStep(0.001)
        self.alignStdZTo.setProperty("value", 0.004)
        self.alignStdZTo.setObjectName("alignStdZTo")
        self.stochasticsFormLayout2.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.alignStdZTo)
        self.alignStdYaw = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.alignStdYaw.setDecimals(4)
        self.alignStdYaw.setMinimum(0.0001)
        self.alignStdYaw.setSingleStep(0.001)
        self.alignStdYaw.setProperty("value", 0.12)
        self.alignStdYaw.setObjectName("alignStdYaw")
        self.stochasticsFormLayout2.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.alignStdYaw)

        self.alignStdSpeed = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.alignStdSpeed.setDecimals(4)
        self.alignStdSpeed.setMinimum(0.0001)
        self.alignStdSpeed.setSingleStep(0.001)
        self.alignStdSpeed.setProperty("value", 0.1)
        self.alignStdSpeed.setObjectName("alignStdSpeed")
        self.stochasticsFormLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.FieldRole, self.alignStdSpeed)
        self.s_dev_speed_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.s_dev_speed_label.setObjectName("label_42")
        self.stochasticsFormLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.s_dev_speed_label)

        self.stochasticsHorizontalLayout.addLayout(self.stochasticsFormLayout)
        self.stochasticsHorizontalLayout.addLayout(self.stochasticsFormLayout2)
        spacerItem3 = QtWidgets.QSpacerItem(
            0,
            0,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.stochasticsHorizontalLayout.addItem(spacerItem3)
        self.alignmentVerticalLayout.addLayout(self.stochasticsHorizontalLayout)

        self.estimation_of_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.estimation_of_label.setFont(font)
        self.estimation_of_label.setObjectName("label_17")
        self.alignmentVerticalLayout.addWidget(self.estimation_of_label)
        self.estimationSettingsHorizontalLayout = QtWidgets.QHBoxLayout()
        self.estimationSettingsHorizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.estimationSettingsHorizontalLayout.setObjectName("horizontalLayout_3")
        self.similarityEstimationVerticallayout = QtWidgets.QVBoxLayout()
        self.similarityEstimationVerticallayout.setObjectName("verticalLayout_4")
        self.alignSimilarity = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignSimilarity.setObjectName("alignSimilarity")

        self.similarityEstimationVerticallayout.addWidget(self.alignSimilarity)
        self.alignXTrans = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignXTrans.setEnabled(False)
        self.alignXTrans.setCheckable(True)
        self.alignXTrans.setChecked(True)
        self.alignXTrans.setObjectName("alignXTrans")
        self.similarityEstimationVerticallayout.addWidget(self.alignXTrans)
        self.alignYTrans = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignYTrans.setEnabled(False)
        self.alignYTrans.setChecked(True)
        self.alignYTrans.setObjectName("alignYTrans")
        self.similarityEstimationVerticallayout.addWidget(self.alignYTrans)
        self.alignZTrans = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignZTrans.setEnabled(False)
        self.alignZTrans.setChecked(True)
        self.alignZTrans.setObjectName("alignZTrans")
        self.similarityEstimationVerticallayout.addWidget(self.alignZTrans)
        self.alignXRot = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignXRot.setEnabled(False)
        self.alignXRot.setChecked(True)
        self.alignXRot.setObjectName("alignXRot")
        self.similarityEstimationVerticallayout.addWidget(self.alignXRot)
        self.alignYRot = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignYRot.setEnabled(False)
        self.alignYRot.setChecked(True)
        self.alignYRot.setObjectName("alignYRot")
        self.similarityEstimationVerticallayout.addWidget(self.alignYRot)
        self.alignZRot = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignZRot.setEnabled(False)
        self.alignZRot.setChecked(True)
        self.alignZRot.setObjectName("alignZRot")
        self.similarityEstimationVerticallayout.addWidget(self.alignZRot)
        self.alignScale = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignScale.setEnabled(False)
        self.alignScale.setObjectName("alignScale")
        self.similarityEstimationVerticallayout.addWidget(self.alignScale)
        spacerItem4 = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.similarityEstimationVerticallayout.addItem(spacerItem4)
        self.estimationSettingsHorizontalLayout.addLayout(self.similarityEstimationVerticallayout)
        self.timeshiftEstimationVerticalLayout = QtWidgets.QVBoxLayout()
        self.timeshiftEstimationVerticalLayout.setObjectName("verticalLayout_5")
        self.alignTimeShift = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignTimeShift.setObjectName("alignTimeShift")

        self.timeshiftEstimationVerticalLayout.addWidget(self.alignTimeShift)
        self.alignUseXSpeed = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignUseXSpeed.setEnabled(False)
        self.alignUseXSpeed.setCheckable(True)
        self.alignUseXSpeed.setChecked(True)
        self.alignUseXSpeed.setObjectName("alignUseXSpeed")
        self.timeshiftEstimationVerticalLayout.addWidget(self.alignUseXSpeed)
        self.alignUseYSpeed = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignUseYSpeed.setEnabled(False)
        self.alignUseYSpeed.setCheckable(True)
        self.alignUseYSpeed.setChecked(True)
        self.alignUseYSpeed.setObjectName("alignUseYSpeed")
        self.timeshiftEstimationVerticalLayout.addWidget(self.alignUseYSpeed)
        self.alignUseZSpeed = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignUseZSpeed.setEnabled(False)
        self.alignUseZSpeed.setCheckable(True)
        self.alignUseZSpeed.setChecked(True)
        self.alignUseZSpeed.setObjectName("alignUseZSpeed")
        self.timeshiftEstimationVerticalLayout.addWidget(self.alignUseZSpeed)
        spacerItem5 = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.timeshiftEstimationVerticalLayout.addItem(spacerItem5)
        self.estimationSettingsHorizontalLayout.addLayout(self.timeshiftEstimationVerticalLayout)
        self.leverarmEstimationVerticalLayout = QtWidgets.QVBoxLayout()
        self.leverarmEstimationVerticalLayout.setObjectName("verticalLayout_6")
        self.alignLeverarm = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignLeverarm.setObjectName("alignLeverarm")

        self.leverarmEstimationVerticalLayout.addWidget(self.alignLeverarm)
        self.alignXLever = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignXLever.setEnabled(False)
        self.alignXLever.setCheckable(True)
        self.alignXLever.setChecked(True)
        self.alignXLever.setObjectName("alignXLever")
        self.leverarmEstimationVerticalLayout.addWidget(self.alignXLever)
        self.alignYLever = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignYLever.setEnabled(False)
        self.alignYLever.setCheckable(True)
        self.alignYLever.setChecked(True)
        self.alignYLever.setObjectName("alignYLever")
        self.leverarmEstimationVerticalLayout.addWidget(self.alignYLever)
        self.alignZLever = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.alignZLever.setEnabled(False)
        self.alignZLever.setCheckable(True)
        self.alignZLever.setChecked(True)
        self.alignZLever.setObjectName("alignZLever")
        self.leverarmEstimationVerticalLayout.addWidget(self.alignZLever)

        spacerItem6 = QtWidgets.QSpacerItem(
            20,
            20,
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self.leverarmEstimationVerticalLayout.addItem(spacerItem6)

        self.sensorRot = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.sensorRot.setCheckable(True)
        self.sensorRot.setChecked(False)
        self.sensorRot.setObjectName("sensorRot")
        self.leverarmEstimationVerticalLayout.addWidget(self.sensorRot)

        spacerItem7 = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.leverarmEstimationVerticalLayout.addItem(spacerItem7)
        self.estimationSettingsHorizontalLayout.addLayout(self.leverarmEstimationVerticalLayout)
        self.alignmentVerticalLayout.addLayout(self.estimationSettingsHorizontalLayout)
        self.tabWidget.addTab(self.alignmentTab, "")

        self.sortingTab = QtWidgets.QWidget()
        self.sortingTab.setObjectName("sortingTab")
        self.sortingVerticalLayoutWidget = QtWidgets.QWidget(self.sortingTab)
        self.sortingVerticalLayoutWidget.setGeometry(QRect(0, 0, 600, 540))
        self.sortingVerticalLayout = QtWidgets.QVBoxLayout(self.sortingVerticalLayoutWidget)
        self.sortingVerticalLayout.setContentsMargins(10, 10, 10, 10)
        self.sortingHorizontalLayout = QtWidgets.QHBoxLayout()
        self.sortingHorizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.sortingFormLayout = QtWidgets.QFormLayout()
        self.sortingFormLayout.setContentsMargins(0, 0, 0, 0)
        self.sortingFormLayout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sortingFormLayout.setFormAlignment(Qt.AlignmentFlag.AlignRight)
        self.sortingFormLayout.setObjectName("formLayout")
        self.voxel_size_label = QtWidgets.QLabel(self.sortingVerticalLayoutWidget)
        self.voxel_size_label.setObjectName("label_22")
        self.sortingFormLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.voxel_size_label)
        self.k_nearest_label = QtWidgets.QLabel(self.sortingVerticalLayoutWidget)
        self.k_nearest_label.setObjectName("label_23")
        self.sortingFormLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.k_nearest_label)
        self.movement_threshold_label = QtWidgets.QLabel(self.sortingVerticalLayoutWidget)
        self.movement_threshold_label.setObjectName("label_24")
        self.sortingFormLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.movement_threshold_label)
        self.discard_missing_label = QtWidgets.QLabel(self.sortingVerticalLayoutWidget)
        self.discard_missing_label.setObjectName("label_25")
        self.sortingFormLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.discard_missing_label)
        self.sortingDiscardMissing = QtWidgets.QCheckBox(self.sortingVerticalLayoutWidget)
        self.sortingDiscardMissing.setText("")
        self.sortingDiscardMissing.setChecked(True)
        self.sortingDiscardMissing.setObjectName("sortingDiscardMissing")
        self.sortingFormLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.FieldRole, self.sortingDiscardMissing)
        self.sortingKNearest = QtWidgets.QSpinBox(self.sortingVerticalLayoutWidget)
        self.sortingKNearest.setMinimum(1)
        self.sortingKNearest.setProperty("value", 8)
        self.sortingKNearest.setObjectName("sortingKNearest")
        self.sortingFormLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.sortingKNearest)
        self.sortingVoxel = QtWidgets.QDoubleSpinBox(self.sortingVerticalLayoutWidget)
        self.sortingVoxel.setSingleStep(0.01)
        self.sortingVoxel.setProperty("value", 0.05)
        self.sortingVoxel.setObjectName("sortingVoxel")
        self.sortingFormLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.sortingVoxel)
        self.sortingMovThresh = QtWidgets.QDoubleSpinBox(self.sortingVerticalLayoutWidget)
        self.sortingMovThresh.setDecimals(4)
        self.sortingMovThresh.setMinimum(0.0001)
        self.sortingMovThresh.setSingleStep(0.001)
        self.sortingMovThresh.setProperty("value", 0.005)
        self.sortingMovThresh.setObjectName("sortingMovThresh")
        self.sortingFormLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.sortingMovThresh)

        self.sortingHorizontalLayout.addLayout(self.sortingFormLayout)
        spacerItem8 = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.sortingHorizontalLayout.addItem(spacerItem8)
        self.sortingVerticalLayout.addLayout(self.sortingHorizontalLayout)

        self.tabWidget.addTab(self.sortingTab, "")

        self.approximationTab = QtWidgets.QWidget()
        self.approximationTab.setObjectName("approximationTab")
        self.approximationVerticalLayoutWidget = QtWidgets.QWidget(self.approximationTab)
        self.approximationVerticalLayoutWidget.setGeometry(QRect(0, 0, 600, 540))
        self.approximationVerticalLayout = QtWidgets.QVBoxLayout(self.approximationVerticalLayoutWidget)
        self.approximationVerticalLayout.setContentsMargins(10, 10, 10, 10)
        self.approximationHorizontalLayout = QtWidgets.QHBoxLayout()
        self.approximationHorizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.approximationFormLayout = QtWidgets.QFormLayout()
        self.approximationFormLayout.setContentsMargins(0, 0, 0, 0)
        self.approximationFormLayout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.approximationFormLayout.setFormAlignment(Qt.AlignmentFlag.AlignRight)
        self.approximationFormLayout.setObjectName("formLayout_2")

        self.pos_window_size_label = QtWidgets.QLabel(self.approximationVerticalLayoutWidget)
        self.pos_window_size_label.setObjectName("label_26")
        self.approximationFormLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.pos_window_size_label)
        self.min_obs_per_int_label = QtWidgets.QLabel(self.approximationVerticalLayoutWidget)
        self.min_obs_per_int_label.setObjectName("label_27")
        self.approximationFormLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.min_obs_per_int_label)
        self.rot_approx_technique_label = QtWidgets.QLabel(self.approximationVerticalLayoutWidget)
        self.rot_approx_technique_label.setObjectName("label_28")
        self.approximationFormLayout.setWidget(
            2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.rot_approx_technique_label
        )
        self.rot_windows_size_label = QtWidgets.QLabel(self.approximationVerticalLayoutWidget)
        self.rot_windows_size_label.setObjectName("label_29")
        self.approximationFormLayout.setWidget(
            3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.rot_windows_size_label
        )
        self.approxObsInt = QtWidgets.QSpinBox(self.approximationVerticalLayoutWidget)
        self.approxObsInt.setMinimum(3)
        self.approxObsInt.setProperty("value", 25)
        self.approxObsInt.setObjectName("approxObsInt")
        self.approximationFormLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.approxObsInt)
        self.approxPosWinSize = QtWidgets.QDoubleSpinBox(self.approximationVerticalLayoutWidget)
        self.approxPosWinSize.setDecimals(4)
        self.approxPosWinSize.setMinimum(0.005)
        self.approxPosWinSize.setSingleStep(0.005)
        self.approxPosWinSize.setProperty("value", 0.15)
        self.approxPosWinSize.setObjectName("approxPosWinSize")
        self.approximationFormLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.approxPosWinSize)
        self.approxTechnique = QtWidgets.QComboBox(self.approximationVerticalLayoutWidget)
        self.approxTechnique.setObjectName("approxTechnique")
        self.approxTechnique.addItem("")
        self.approxTechnique.addItem("")
        self.approximationFormLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.approxTechnique)
        self.approxRotWinSize = QtWidgets.QDoubleSpinBox(self.approximationVerticalLayoutWidget)
        self.approxRotWinSize.setDecimals(4)
        self.approxRotWinSize.setMinimum(0.005)
        self.approxRotWinSize.setSingleStep(0.005)
        self.approxRotWinSize.setProperty("value", 0.15)
        self.approxRotWinSize.setObjectName("approxRotWinSize")
        self.approximationFormLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.FieldRole, self.approxRotWinSize)

        self.approximationHorizontalLayout.addLayout(self.approximationFormLayout)
        spacerItem9 = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.approximationHorizontalLayout.addItem(spacerItem9)
        self.approximationVerticalLayout.addLayout(self.approximationHorizontalLayout)

        self.tabWidget.addTab(self.approximationTab, "")

        self.comparisonTab = QtWidgets.QWidget()
        self.comparisonTab.setObjectName("comparisonTab")
        self.comparisonVerticalLayoutWidget = QtWidgets.QWidget(self.comparisonTab)
        self.comparisonVerticalLayoutWidget.setGeometry(QRect(0, 0, 600, 540))
        self.comparisonVerticalLayout = QtWidgets.QVBoxLayout(self.comparisonVerticalLayoutWidget)
        self.comparisonVerticalLayout.setContentsMargins(10, 10, 10, 10)
        self.comparisonHorizontalLayout = QtWidgets.QHBoxLayout()
        self.comparisonHorizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.comparisonFormLayout = QtWidgets.QFormLayout()
        self.comparisonFormLayout.setContentsMargins(0, 0, 0, 0)
        self.comparisonFormLayout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.comparisonFormLayout.setFormAlignment(Qt.AlignmentFlag.AlignRight)

        self.matching_label = QtWidgets.QLabel(self.comparisonVerticalLayoutWidget)
        self.matching_label.setObjectName("voxel_label")
        self.matching_label.setFont(font)
        self.comparisonFormLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.matching_label)

        self.max_time_diff_label = QtWidgets.QLabel(self.comparisonVerticalLayoutWidget)
        self.max_time_diff_label.setObjectName("max_time_diff_label")
        self.comparisonFormLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.max_time_diff_label)
        self.matchingMaxTimeDiff = QtWidgets.QDoubleSpinBox(self.comparisonVerticalLayoutWidget)
        self.matchingMaxTimeDiff.setDecimals(4)
        self.matchingMaxTimeDiff.setMinimum(0.0)
        self.matchingMaxTimeDiff.setSingleStep(0.0001)
        self.matchingMaxTimeDiff.setProperty("value", 0.01)
        self.matchingMaxTimeDiff.setObjectName("matchingMaxTimeDiff")
        self.comparisonFormLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.matchingMaxTimeDiff)

        self.max_dist_diff_label = QtWidgets.QLabel(self.comparisonVerticalLayoutWidget)
        self.max_dist_diff_label.setObjectName("max_dist_diff_label")
        self.comparisonFormLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.max_dist_diff_label)
        self.matchingMaxDistDiff = QtWidgets.QDoubleSpinBox(self.comparisonVerticalLayoutWidget)
        self.matchingMaxDistDiff.setDecimals(4)
        self.matchingMaxDistDiff.setMinimum(0.0)
        self.matchingMaxDistDiff.setSingleStep(0.001)
        self.matchingMaxDistDiff.setProperty("value", 0.0)
        self.matchingMaxDistDiff.setObjectName("matchingMaxDistDiff")
        self.comparisonFormLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.matchingMaxDistDiff)

        self.relative_label = QtWidgets.QLabel(self.comparisonVerticalLayoutWidget)
        self.relative_label.setObjectName("relative_label")
        self.relative_label.setFont(font)
        self.comparisonFormLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.relative_label)

        self.relative_min_dist_label = QtWidgets.QLabel(self.comparisonVerticalLayoutWidget)
        self.relative_min_dist_label.setObjectName("relative_min_dist_label")
        self.comparisonFormLayout.setWidget(4, QtWidgets.QFormLayout.ItemRole.LabelRole, self.relative_min_dist_label)
        self.relativeMinDist = QtWidgets.QDoubleSpinBox(self.comparisonVerticalLayoutWidget)
        self.relativeMinDist.setDecimals(2)
        self.relativeMinDist.setMinimum(0.01)
        self.relativeMinDist.setMaximum(999999)
        self.relativeMinDist.setSingleStep(0.01)
        self.relativeMinDist.setProperty("value", 100)
        self.relativeMinDist.setObjectName("relativeMinDist")
        self.comparisonFormLayout.setWidget(4, QtWidgets.QFormLayout.ItemRole.FieldRole, self.relativeMinDist)

        self.relative_max_dist_label = QtWidgets.QLabel(self.comparisonVerticalLayoutWidget)
        self.relative_max_dist_label.setObjectName("relative_min_dist_label")
        self.comparisonFormLayout.setWidget(5, QtWidgets.QFormLayout.ItemRole.LabelRole, self.relative_max_dist_label)
        self.relativeMaxDist = QtWidgets.QDoubleSpinBox(self.comparisonVerticalLayoutWidget)
        self.relativeMaxDist.setDecimals(2)
        self.relativeMaxDist.setMinimum(0.01)
        self.relativeMaxDist.setMaximum(999999)
        self.relativeMaxDist.setSingleStep(0.01)
        self.relativeMaxDist.setProperty("value", 800)
        self.relativeMaxDist.setObjectName("relativeMaxDist")
        self.comparisonFormLayout.setWidget(5, QtWidgets.QFormLayout.ItemRole.FieldRole, self.relativeMaxDist)

        self.relative_dist_step_label = QtWidgets.QLabel(self.comparisonVerticalLayoutWidget)
        self.relative_dist_step_label.setObjectName("relative_dist_step_label")
        self.comparisonFormLayout.setWidget(6, QtWidgets.QFormLayout.ItemRole.LabelRole, self.relative_dist_step_label)
        self.relativeDistStep = QtWidgets.QDoubleSpinBox(self.comparisonVerticalLayoutWidget)
        self.relativeDistStep.setDecimals(2)
        self.relativeDistStep.setMinimum(0.01)
        self.relativeDistStep.setMaximum(999999)
        self.relativeDistStep.setSingleStep(0.01)
        self.relativeDistStep.setProperty("value", 100)
        self.relativeDistStep.setObjectName("relativeDistStep")
        self.comparisonFormLayout.setWidget(6, QtWidgets.QFormLayout.ItemRole.FieldRole, self.relativeDistStep)

        self.relative_dist_unit_label = QtWidgets.QLabel(self.comparisonVerticalLayoutWidget)
        self.relative_dist_unit_label.setObjectName("relative_dist_unit_label")
        self.comparisonFormLayout.setWidget(7, QtWidgets.QFormLayout.ItemRole.LabelRole, self.relative_dist_unit_label)
        self.relativeDistUnit = QtWidgets.QComboBox(self.approximationVerticalLayoutWidget)
        self.relativeDistUnit.setObjectName("relativeDistUnit")
        self.relativeDistUnit.addItem("")
        self.relativeDistUnit.addItem("")
        self.comparisonFormLayout.setWidget(7, QtWidgets.QFormLayout.ItemRole.FieldRole, self.relativeDistUnit)

        self.relative_all_pairs_label = QtWidgets.QLabel(self.comparisonVerticalLayoutWidget)
        self.relative_all_pairs_label.setObjectName("relative_all_pairs_label")
        self.comparisonFormLayout.setWidget(8, QtWidgets.QFormLayout.ItemRole.LabelRole, self.relative_all_pairs_label)
        self.relativeAllPairs = QtWidgets.QCheckBox(self.comparisonVerticalLayoutWidget)
        self.relativeAllPairs.setText("")
        self.relativeAllPairs.setChecked(True)
        self.relativeAllPairs.setObjectName("relativeAllPairs")
        self.comparisonFormLayout.setWidget(8, QtWidgets.QFormLayout.ItemRole.FieldRole, self.relativeAllPairs)

        self.comparisonHorizontalLayout.addLayout(self.comparisonFormLayout)
        spacerItem10 = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.comparisonHorizontalLayout.addItem(spacerItem10)
        self.comparisonVerticalLayout.addLayout(self.comparisonHorizontalLayout)

        self.tabWidget.addTab(self.comparisonTab, "")

        self.okButton = QtWidgets.QPushButton(self.centralwidget)
        self.okButton.setGeometry(QRect(510, 490, 75, 23))
        self.okButton.setObjectName("okButton")

        self.cancelButton = QtWidgets.QPushButton(self.centralwidget)
        self.cancelButton.setGeometry(QRect(420, 490, 75, 23))
        self.cancelButton.setObjectName("cancelButton")
        self.loadButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadButton.setGeometry(QRect(105, 490, 75, 23))
        self.loadButton.setObjectName("loadButton")

        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setGeometry(QRect(15, 490, 75, 23))
        self.saveButton.setObjectName("saveButton")

        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QRect(0, 0, 600, 21))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        self.tabWidget.setCurrentIndex(0)
        QMetaObject.connectSlotsByName(self)

    def retranslateUi(self) -> None:
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Trajectory Processing Settings"))
        self.preprocessing_label.setText(_translate("MainWindow", "Preprocessing:"))
        self.minimum_speed_label.setText(_translate("MainWindow", "Minimum speed [m/s]:"))
        self.time_start_label.setText(_translate("MainWindow", "Time start [s]:"))
        self.time_end_label.setText(_translate("MainWindow", "Time end [s]:"))
        self.stochastics_label.setText(_translate("MainWindow", "Stochastics:"))
        self.error_probability_label.setText(_translate("MainWindow", "Error probability [%]:"))
        self.s_dev_xy_source_label.setText(_translate("MainWindow", "S.Dev. XY (Source) [m]:"))
        self.s_dev_z_source_label.setText(_translate("MainWindow", "S.Dev. Z (Source) [m]:"))
        self.s_dev_rp_label.setText(_translate("MainWindow", "S.Dev. roll / pitch [°]:"))
        self.s_dev_xy_target_label.setText(_translate("MainWindow", "S.Dev. XY (Target) [m]:"))
        self.s_dev_z_target_label.setText(_translate("MainWindow", "S.Dev. Z (Target) [m]:"))
        self.s_dev_yaw_label.setText(_translate("MainWindow", "S.Dev. yaw [°]:"))
        self.s_dev_speed_label.setText(_translate("MainWindow", "S.Dev. speed [m/s]:"))
        self.estimation_of_label.setText(_translate("MainWindow", "Estimation of:"))
        self.alignSimilarity.setText(_translate("MainWindow", "Similarity transformation"))
        self.alignXTrans.setText(_translate("MainWindow", "x translation"))
        self.alignYTrans.setText(_translate("MainWindow", "y translation"))
        self.alignZTrans.setText(_translate("MainWindow", "z translation"))
        self.alignXRot.setText(_translate("MainWindow", "x rotation"))
        self.alignYRot.setText(_translate("MainWindow", "y rotation"))
        self.alignZRot.setText(_translate("MainWindow", "z rotation"))
        self.alignScale.setText(_translate("MainWindow", "scale"))
        self.alignTimeShift.setText(_translate("MainWindow", "Time shift"))
        self.alignUseXSpeed.setText(_translate("MainWindow", "use x speed"))
        self.alignUseYSpeed.setText(_translate("MainWindow", "use y speed"))
        self.alignUseZSpeed.setText(_translate("MainWindow", "use z speed"))
        self.alignLeverarm.setText(_translate("MainWindow", "Leverarm"))
        self.alignXLever.setText(_translate("MainWindow", "x lever"))
        self.alignYLever.setText(_translate("MainWindow", "y lever"))
        self.alignZLever.setText(_translate("MainWindow", "z lever"))
        self.sensorRot.setText(_translate("MainWindow", "Sensor rotation"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.alignmentTab),
            _translate("MainWindow", "Alignment"),
        )
        self.voxel_size_label.setText(_translate("MainWindow", "Voxel size [m]:"))
        self.k_nearest_label.setText(_translate("MainWindow", "k-nearest:"))
        self.movement_threshold_label.setText(_translate("MainWindow", "Movement threshold [m]:"))
        self.discard_missing_label.setText(_translate("MainWindow", "Discard missing points:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.sortingTab), _translate("MainWindow", "Sorting"))
        self.pos_window_size_label.setText(_translate("MainWindow", "Window size (Positions) [m]:"))
        self.min_obs_per_int_label.setText(_translate("MainWindow", "Minimum observations per interval:"))
        self.rot_approx_technique_label.setText(_translate("MainWindow", "Rotation Approximation Technique:"))
        self.rot_windows_size_label.setText(_translate("MainWindow", "Window size (Rotations) [m]:"))
        self.approxTechnique.setItemText(0, _translate("MainWindow", "Window"))
        self.approxTechnique.setItemText(1, _translate("MainWindow", "Lap interpolation"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.approximationTab),
            _translate("MainWindow", "Approximation"),
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.comparisonTab),
            _translate("MainWindow", "Comparison"),
        )
        self.matching_label.setText(_translate("MainWindow", "Matching:"))
        self.max_time_diff_label.setText(_translate("MainWindow", "Maximum Time Difference (Temporal Matching) [s]:"))
        self.max_dist_diff_label.setText(_translate("MainWindow", "Maximum Distance (Spatial Matching) [m]:"))

        self.relative_label.setText(_translate("MainWindow", "Relative Comparison:"))
        self.relative_min_dist_label.setText(_translate("MainWindow", "Minimum pose distance:"))
        self.relative_max_dist_label.setText(_translate("MainWindow", "Maximum pose distance:"))
        self.relative_dist_step_label.setText(_translate("MainWindow", "Distance step:"))
        self.relative_dist_unit_label.setText(_translate("MainWindow", "Distance unit:"))
        self.relativeDistUnit.setItemText(0, _translate("MainWindow", "Meter"))
        self.relativeDistUnit.setItemText(1, _translate("MainWindow", "Second"))
        self.relative_all_pairs_label.setText(_translate("MainWindow", "Use all pairs:"))

        self.okButton.setText(_translate("MainWindow", "OK"))
        self.cancelButton.setText(_translate("MainWindow", "Cancel"))
        self.loadButton.setText(_translate("MainWindow", "Load"))
        self.saveButton.setText(_translate("MainWindow", "Save"))

    def update_view(self) -> None:
        try:
            self.update_alignment_view()
            self.update_sorting_view()
            self.update_approximation_view()
            self.update_comparison_view()
        except ValueError:
            self.trajectory_entry.settings = ProcessingSettings()
            show_msg_box("Error updating settings view! Probably the settings were corrupted! Reset settings.")

    def update_comparison_view(self) -> None:
        self.matchingMaxTimeDiff.setValue(self.trajectory_entry.settings.matching.max_time_diff)
        self.matchingMaxDistDiff.setValue(self.trajectory_entry.settings.matching.max_distance)
        self.relativeMinDist.setValue(self.trajectory_entry.settings.rel_comparison.pair_min_distance)
        self.relativeMaxDist.setValue(self.trajectory_entry.settings.rel_comparison.pair_max_distance)
        self.relativeDistStep.setValue(self.trajectory_entry.settings.rel_comparison.pair_distance_step)
        self.relativeDistUnit.setCurrentIndex(
            0 if self.trajectory_entry.settings.rel_comparison.pair_distance_unit == Unit.METER else 1
        )
        self.relativeAllPairs.setChecked(self.trajectory_entry.settings.rel_comparison.use_all_pose_pairs)

    def update_approximation_view(self) -> None:
        self.approxPosWinSize.setValue(self.trajectory_entry.settings.approximation.fe_int_size)
        self.approxObsInt.setValue(self.trajectory_entry.settings.approximation.fe_min_obs)

        if self.trajectory_entry.settings.approximation.rot_approx_technique == RotApprox.WINDOW:
            self.approxTechnique.setCurrentIndex(0)
        elif self.trajectory_entry.settings.approximation.rot_approx_technique == RotApprox.INTERP:
            self.approxTechnique.setCurrentIndex(1)
        else:
            raise ValueError(f"Invalid technique {self.trajectory_entry.settings.approximation.rot_approx_technique}")

        self.approxRotWinSize.setValue(self.trajectory_entry.settings.approximation.rot_approx_win_size)

    def update_sorting_view(self) -> None:
        self.sortingVoxel.setValue(self.trajectory_entry.settings.sorting.voxel_size)
        self.sortingKNearest.setValue(self.trajectory_entry.settings.sorting.k_nearest)
        self.sortingMovThresh.setValue(self.trajectory_entry.settings.sorting.movement_threshold)
        self.sortingDiscardMissing.setChecked(self.trajectory_entry.settings.sorting.discard_missing)

    def update_alignment_view(self) -> None:
        self.alignMinSpeed.setValue(self.trajectory_entry.settings.alignment.preprocessing.min_speed)
        self.alignTimeStart.setValue(self.trajectory_entry.settings.alignment.preprocessing.time_start)
        self.alignTimeEnd.setValue(self.trajectory_entry.settings.alignment.preprocessing.time_end)

        self.alignErrorProb.setValue(self.trajectory_entry.settings.alignment.stochastics.error_probability * 100)

        self.alignStdXYFrom.setValue(float(np.sqrt(self.trajectory_entry.settings.alignment.stochastics.var_xy_from)))
        self.alignStdZFrom.setValue(float(np.sqrt(self.trajectory_entry.settings.alignment.stochastics.var_z_from)))

        self.alignStdXYTo.setValue(float(np.sqrt(self.trajectory_entry.settings.alignment.stochastics.var_xy_to)))
        self.alignStdZTo.setValue(float(np.sqrt(self.trajectory_entry.settings.alignment.stochastics.var_z_to)))

        self.alignStdRP.setValue(
            float(np.sqrt(self.trajectory_entry.settings.alignment.stochastics.var_roll_pitch)) * 180 / np.pi
        )
        self.alignStdYaw.setValue(
            float(np.sqrt(self.trajectory_entry.settings.alignment.stochastics.var_yaw)) * 180 / np.pi
        )
        self.alignStdSpeed.setValue(float(np.sqrt(self.trajectory_entry.settings.alignment.stochastics.var_speed_to)))

        self.alignSimilarity.setChecked(self.trajectory_entry.settings.alignment.estimation_of.helmert)
        self.alignXTrans.setChecked(self.trajectory_entry.settings.alignment.estimation_of.trans_x)
        self.alignYTrans.setChecked(self.trajectory_entry.settings.alignment.estimation_of.trans_y)
        self.alignZTrans.setChecked(self.trajectory_entry.settings.alignment.estimation_of.trans_z)
        self.alignXRot.setChecked(self.trajectory_entry.settings.alignment.estimation_of.rot_x)
        self.alignYRot.setChecked(self.trajectory_entry.settings.alignment.estimation_of.rot_y)
        self.alignZRot.setChecked(self.trajectory_entry.settings.alignment.estimation_of.rot_z)
        self.alignScale.setChecked(self.trajectory_entry.settings.alignment.estimation_of.scale)

        self.alignTimeShift.setChecked(self.trajectory_entry.settings.alignment.estimation_of.time_shift)
        self.alignUseXSpeed.setChecked(self.trajectory_entry.settings.alignment.estimation_of.use_x_speed)
        self.alignUseYSpeed.setChecked(self.trajectory_entry.settings.alignment.estimation_of.use_y_speed)
        self.alignUseZSpeed.setChecked(self.trajectory_entry.settings.alignment.estimation_of.use_z_speed)

        self.alignLeverarm.setChecked(self.trajectory_entry.settings.alignment.estimation_of.leverarm)
        self.alignXLever.setChecked(self.trajectory_entry.settings.alignment.estimation_of.lever_x)
        self.alignYLever.setChecked(self.trajectory_entry.settings.alignment.estimation_of.lever_y)
        self.alignZLever.setChecked(self.trajectory_entry.settings.alignment.estimation_of.lever_z)

        self.sensorRot.setChecked(self.trajectory_entry.settings.alignment.estimation_of.sensor_rotation)

    def update_model(self):
        self.update_alignment_model()
        self.update_sorting_model()
        self.update_approximation_model()
        self.update_comparison_model()

    def update_comparison_model(self) -> None:
        self.trajectory_entry.settings.matching.max_time_diff = self.matchingMaxTimeDiff.value()
        self.trajectory_entry.settings.matching.max_distance = self.matchingMaxDistDiff.value()
        self.trajectory_entry.settings.rel_comparison.pair_min_distance = self.relativeMinDist.value()
        self.trajectory_entry.settings.rel_comparison.pair_max_distance = self.relativeMaxDist.value()
        self.trajectory_entry.settings.rel_comparison.pair_distance_step = self.relativeDistStep.value()
        self.trajectory_entry.settings.rel_comparison.pair_distance_unit = UNIT_DICT.get(
            self.relativeDistUnit.currentIndex(), Unit.METER
        )
        self.trajectory_entry.settings.rel_comparison.use_all_pose_pairs = self.relativeAllPairs.isChecked()

    def update_approximation_model(self):
        self.trajectory_entry.settings.approximation.fe_int_size = self.approxPosWinSize.value()
        self.trajectory_entry.settings.approximation.fe_min_obs = int(self.approxObsInt.value())
        self.trajectory_entry.settings.approximation.rot_approx_technique = ROT_APPROX_DICT.get(
            self.approxTechnique.currentIndex(), RotApprox.WINDOW
        )
        self.trajectory_entry.settings.approximation.rot_approx_win_size = self.approxRotWinSize.value()

    def update_sorting_model(self):
        self.trajectory_entry.settings.sorting.voxel_size = self.sortingVoxel.value()
        self.trajectory_entry.settings.sorting.k_nearest = int(self.sortingKNearest.value())
        self.trajectory_entry.settings.sorting.movement_threshold = self.sortingMovThresh.value()
        self.trajectory_entry.settings.sorting.discard_missing = self.sortingDiscardMissing.isChecked()

    def update_alignment_model(self):
        self.trajectory_entry.settings.alignment.preprocessing.min_speed = self.alignMinSpeed.value()
        self.trajectory_entry.settings.alignment.preprocessing.time_start = self.alignTimeStart.value()
        self.trajectory_entry.settings.alignment.preprocessing.time_end = self.alignTimeEnd.value()

        self.trajectory_entry.settings.alignment.stochastics.error_probability = self.alignErrorProb.value() / 100
        self.trajectory_entry.settings.alignment.stochastics.var_xy_from = self.alignStdXYFrom.value() ** 2
        self.trajectory_entry.settings.alignment.stochastics.var_z_from = self.alignStdZFrom.value() ** 2
        self.trajectory_entry.settings.alignment.stochastics.var_xy_to = self.alignStdXYTo.value() ** 2
        self.trajectory_entry.settings.alignment.stochastics.var_z_to = self.alignStdZTo.value() ** 2
        self.trajectory_entry.settings.alignment.stochastics.var_roll_pitch = (
            self.alignStdRP.value() * np.pi / 180
        ) ** 2
        self.trajectory_entry.settings.alignment.stochastics.var_yaw = (self.alignStdYaw.value() * np.pi / 180) ** 2
        self.trajectory_entry.settings.alignment.stochastics.var_speed_to = self.alignStdSpeed.value() ** 2

        self.trajectory_entry.settings.alignment.estimation_of.helmert = self.alignSimilarity.isChecked()
        self.trajectory_entry.settings.alignment.estimation_of.trans_x = (
            self.alignXTrans.isChecked() and self.alignSimilarity.isChecked()
        )
        self.trajectory_entry.settings.alignment.estimation_of.trans_y = (
            self.alignYTrans.isChecked() and self.alignSimilarity.isChecked()
        )
        self.trajectory_entry.settings.alignment.estimation_of.trans_z = (
            self.alignZTrans.isChecked() and self.alignSimilarity.isChecked()
        )
        self.trajectory_entry.settings.alignment.estimation_of.rot_x = (
            self.alignXRot.isChecked() and self.alignSimilarity.isChecked()
        )
        self.trajectory_entry.settings.alignment.estimation_of.rot_y = (
            self.alignYRot.isChecked() and self.alignSimilarity.isChecked()
        )
        self.trajectory_entry.settings.alignment.estimation_of.rot_z = (
            self.alignZRot.isChecked() and self.alignSimilarity.isChecked()
        )
        self.trajectory_entry.settings.alignment.estimation_of.scale = (
            self.alignScale.isChecked() and self.alignSimilarity.isChecked()
        )

        self.trajectory_entry.settings.alignment.estimation_of.time_shift = self.alignTimeShift.isChecked()
        self.trajectory_entry.settings.alignment.estimation_of.use_x_speed = (
            self.alignUseXSpeed.isChecked() and self.alignTimeShift.isChecked()
        )
        self.trajectory_entry.settings.alignment.estimation_of.use_y_speed = (
            self.alignUseYSpeed.isChecked() and self.alignTimeShift.isChecked()
        )
        self.trajectory_entry.settings.alignment.estimation_of.use_z_speed = (
            self.alignUseZSpeed.isChecked() and self.alignTimeShift.isChecked()
        )

        self.trajectory_entry.settings.alignment.estimation_of.leverarm = self.alignLeverarm.isChecked()
        self.trajectory_entry.settings.alignment.estimation_of.lever_x = (
            self.alignXLever.isChecked() and self.alignLeverarm.isChecked()
        )
        self.trajectory_entry.settings.alignment.estimation_of.lever_y = (
            self.alignYLever.isChecked() and self.alignLeverarm.isChecked()
        )
        self.trajectory_entry.settings.alignment.estimation_of.lever_z = (
            self.alignZLever.isChecked() and self.alignLeverarm.isChecked()
        )
        self.trajectory_entry.settings.alignment.estimation_of.sensor_rotation = self.sensorRot.isChecked()

    @pyqtSlot(int)
    def handle_similarity_changed(self, state: int) -> None:
        self.set_helmert_state(state == 2)

    @pyqtSlot(int)
    def handle_timeshift_changed(self, state: int) -> None:
        self.set_timeshift_state(state == 2)

    @pyqtSlot(int)
    def handle_leverarm_changed(self, state: int) -> None:
        self.set_leverarm_state(state == 2)

    def set_helmert_state(self, state: bool):
        self.alignXTrans.setEnabled(state)
        self.alignYTrans.setEnabled(state)
        self.alignZTrans.setEnabled(state)
        self.alignXRot.setEnabled(state)
        self.alignYRot.setEnabled(state)
        self.alignZRot.setEnabled(state)
        self.alignScale.setEnabled(state)

    def set_timeshift_state(self, state: bool) -> None:
        self.alignUseXSpeed.setEnabled(state)
        self.alignUseYSpeed.setEnabled(state)
        self.alignUseZSpeed.setEnabled(state)

    def set_leverarm_state(self, state: bool) -> None:
        self.alignXLever.setEnabled(state)
        self.alignYLever.setEnabled(state)
        self.alignZLever.setEnabled(state)

    @pyqtSlot()
    def handle_ok(self) -> None:
        self.update_model()
        self.close()

    @pyqtSlot()
    def handle_save(self) -> None:
        if not (file := save_file_dialog(parent=self, file_filter="Settings File (*.yaml)")):
            return

        self.update_model()
        self.trajectory_entry.settings.to_file(file)

    @pyqtSlot()
    def handle_load(self) -> None:
        if not (
            file := read_file_dialog(
                parent=self,
                file_filter="Settings File (*.yaml)",
                mode=QtWidgets.QFileDialog.FileMode.ExistingFile,
            )
        ):
            return

        try:
            logger.info("Reading file: %s", file)
            self.trajectory_entry.settings = ProcessingSettings.from_file(file[0])
            self.update_view()

        except Exception as e:
            show_msg_box(f"Error reading {file=} ({e})")
