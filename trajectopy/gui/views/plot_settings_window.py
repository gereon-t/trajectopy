"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import logging
import os

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from trajectopy_core.settings.plot_settings import PlotSettings

from trajectopy.gui.managers.requests import (
    PlotRequest,
    PlotRequestType,
    PlotSettingsRequest,
    PlotSettingsRequestType,
    UIRequest,
    generic_request_handler,
)

logger = logging.getLogger("root")


class PlotSettingsGUI(QtWidgets.QMainWindow):
    plot_request = QtCore.pyqtSignal(PlotRequest)
    ui_request = QtCore.pyqtSignal(UIRequest)
    operation_finished = QtCore.pyqtSignal()

    def __init__(self, parent) -> None:
        super().__init__(parent=parent)
        self.setupUi()
        self.settings: PlotSettings = PlotSettings()
        self.update_view()

        self.okButton.clicked.connect(self.handle_ok)
        self.cancelButton.clicked.connect(self.close)

        self.REQUEST_MAPPING = {
            PlotSettingsRequestType.RESET: self.clear_settings,
            PlotSettingsRequestType.EXPORT_TO_SESSION: self.export_settings_to_session,
            PlotSettingsRequestType.IMPORT_FROM_SESSION: self.import_settings_from_session,
        }
        self.request: PlotSettingsRequest

    def show(self) -> None:
        self.update_view()
        super().show()

    def set_settings(self, settings: PlotSettings) -> None:
        self.settings = settings

    @QtCore.pyqtSlot(PlotSettingsRequest)
    def handle_request(self, request: PlotSettingsRequest) -> None:
        generic_request_handler(self, request, passthrough_request=True)

    def clear_settings(self, _: PlotSettingsRequest) -> None:
        self.settings.reset()
        self.update_view()

    def import_settings_from_session(self, request: PlotSettingsRequest):
        filename = os.path.join(request.file_path, "plot_settings.yaml")

        if not os.path.isfile(filename):
            return

        self.settings = PlotSettings.from_file(filename)
        print(f"Reading {os.path.join(request.file_path, 'plot_settings.yaml')}")
        self.update_view()
        self.plot_request.emit(PlotRequest(type=PlotRequestType.UPDATE_SETTINGS, plot_settings=self.settings))

    def export_settings_to_session(self, request: PlotSettingsRequest):
        self.settings.to_file(os.path.join(request.file_path, "plot_settings.yaml"))

    def setupUi(self):
        font = QtGui.QFont()
        font.setBold(True)

        self.setObjectName("MainWindow")
        self.resize(640, 320)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 640, 300))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.centralwidget.setLayout(self.verticalLayout)
        self.horizontalLayout_forms = QtWidgets.QHBoxLayout()
        self.horizontalLayout_forms.setContentsMargins(-1, -1, 0, -1)
        self.horizontalLayout_forms.setObjectName("horizontalLayout_forms")
        self.verticalLayout_general = QtWidgets.QVBoxLayout()
        self.verticalLayout_general.setContentsMargins(-1, -1, 0, 0)
        self.verticalLayout_general.setObjectName("verticalLayout_general")
        self.formLayout_general = QtWidgets.QFormLayout()
        self.formLayout_general.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.formLayout_general.setFormAlignment(Qt.AlignmentFlag.AlignRight)
        self.formLayout_general.setVerticalSpacing(10)
        self.formLayout_general.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.formLayout_general.setFormAlignment(Qt.AlignmentFlag.AlignRight)
        self.formLayout_general.setObjectName("formLayout_general")
        self.general_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.general_label.setFont(font)
        self.general_label.setObjectName("label")
        self.formLayout_general.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.general_label)
        self.positionUnitLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.positionUnitLabel.setObjectName("positionUnitLabel")
        self.formLayout_general.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.positionUnitLabel)
        self.positionUnitComboBox = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.positionUnitComboBox.setObjectName("positionUnitComboBox")
        self.positionUnitComboBox.addItem("")
        self.positionUnitComboBox.addItem("")
        self.formLayout_general.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.positionUnitComboBox)
        self.stairHistogramLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.stairHistogramLabel.setObjectName("stairHistogramLabel")
        self.formLayout_general.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.stairHistogramLabel)
        self.stairHistogramCheckBox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.stairHistogramCheckBox.setObjectName("stairHistogramCheckBox")
        self.formLayout_general.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.stairHistogramCheckBox)
        self.smoothingWindowWidthMLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.smoothingWindowWidthMLabel.setObjectName("smoothingWindowWidthMLabel")
        self.formLayout_general.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.smoothingWindowWidthMLabel)
        self.smoothingWindowWidthMDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.smoothingWindowWidthMDoubleSpinBox.setMaximum(1000.0)
        self.smoothingWindowWidthMDoubleSpinBox.setSingleStep(0.1)
        self.smoothingWindowWidthMDoubleSpinBox.setProperty("value", 1.0)
        self.smoothingWindowWidthMDoubleSpinBox.setObjectName("smoothingWindowWidthMDoubleSpinBox")
        self.formLayout_general.setWidget(
            3,
            QtWidgets.QFormLayout.ItemRole.FieldRole,
            self.smoothingWindowWidthMDoubleSpinBox,
        )
        self.showMeanLineLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.showMeanLineLabel.setObjectName("showMeanLineLabel")
        self.formLayout_general.setWidget(4, QtWidgets.QFormLayout.ItemRole.LabelRole, self.showMeanLineLabel)
        self.showMeanLineCheckBox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.showMeanLineCheckBox.setObjectName("showMeanLineCheckBox")
        self.formLayout_general.setWidget(4, QtWidgets.QFormLayout.ItemRole.FieldRole, self.showMeanLineCheckBox)
        self.showDirectedDeviationsAlongCrossVerticalLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.showDirectedDeviationsAlongCrossVerticalLabel.setObjectName(
            "showDirectedDeviationsAlongCrossVerticalLabel"
        )
        self.formLayout_general.setWidget(
            5,
            QtWidgets.QFormLayout.ItemRole.LabelRole,
            self.showDirectedDeviationsAlongCrossVerticalLabel,
        )
        self.showDirectedDeviationsAlongCrossVerticalCheckBox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.showDirectedDeviationsAlongCrossVerticalCheckBox.setObjectName(
            "showDirectedDeviationsAlongCrossVerticalCheckBox"
        )
        self.formLayout_general.setWidget(
            5,
            QtWidgets.QFormLayout.ItemRole.FieldRole,
            self.showDirectedDeviationsAlongCrossVerticalCheckBox,
        )
        self.verticalLayout_general.addLayout(self.formLayout_general)
        spacerItem = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.verticalLayout_general.addItem(spacerItem)
        self.horizontalLayout_forms.addLayout(self.verticalLayout_general)
        spacerItem1 = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.horizontalLayout_forms.addItem(spacerItem1)
        self.verticalLayout_scatter = QtWidgets.QVBoxLayout()
        self.verticalLayout_scatter.setContentsMargins(-1, -1, 0, 0)
        self.verticalLayout_scatter.setObjectName("verticalLayout_scatter")
        self.formLayout_scatter = QtWidgets.QFormLayout()
        self.formLayout_scatter.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.formLayout_scatter.setFormAlignment(Qt.AlignmentFlag.AlignRight)
        self.formLayout_scatter.setContentsMargins(-1, -1, 10, 0)
        self.formLayout_scatter.setVerticalSpacing(10)
        self.formLayout_scatter.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.formLayout_scatter.setFormAlignment(Qt.AlignmentFlag.AlignRight)
        self.formLayout_scatter.setObjectName("formLayout_scatter")
        self.scatter_plot_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.scatter_plot_label.setFont(font)
        self.scatter_plot_label.setObjectName("label_2")
        self.formLayout_scatter.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.scatter_plot_label)
        self.gridMegapixelsLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.gridMegapixelsLabel.setObjectName("gridMegapixelsLabel")
        self.formLayout_scatter.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.gridMegapixelsLabel)
        self.gridMegapixelsDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.gridMegapixelsDoubleSpinBox.setDecimals(4)
        self.gridMegapixelsDoubleSpinBox.setMinimum(0.001)
        self.gridMegapixelsDoubleSpinBox.setMaximum(24.0)
        self.gridMegapixelsDoubleSpinBox.setSingleStep(0.1)
        self.gridMegapixelsDoubleSpinBox.setObjectName("gridMegapixelsDoubleSpinBox")
        self.formLayout_scatter.setWidget(
            1,
            QtWidgets.QFormLayout.ItemRole.FieldRole,
            self.gridMegapixelsDoubleSpinBox,
        )
        self.rMSWindowWidthMLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.rMSWindowWidthMLabel.setObjectName("rMSWindowWidthMLabel")
        self.formLayout_scatter.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.rMSWindowWidthMLabel)
        self.rMSWindowWidthMDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.rMSWindowWidthMDoubleSpinBox.setMaximum(1000.0)
        self.rMSWindowWidthMDoubleSpinBox.setSingleStep(0.1)
        self.rMSWindowWidthMDoubleSpinBox.setProperty("value", 1.0)
        self.rMSWindowWidthMDoubleSpinBox.setObjectName("rMSWindowWidthMDoubleSpinBox")
        self.formLayout_scatter.setWidget(
            2,
            QtWidgets.QFormLayout.ItemRole.FieldRole,
            self.rMSWindowWidthMDoubleSpinBox,
        )
        self.showZeroCrossingInColorbarsLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.showZeroCrossingInColorbarsLabel.setObjectName("showZeroCrossingInColorbarsLabel")
        self.formLayout_scatter.setWidget(
            3,
            QtWidgets.QFormLayout.ItemRole.LabelRole,
            self.showZeroCrossingInColorbarsLabel,
        )
        self.showZeroCrossingInColorbarsCheckBox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.showZeroCrossingInColorbarsCheckBox.setObjectName("showZeroCrossingInColorbarsCheckBox")
        self.formLayout_scatter.setWidget(
            3,
            QtWidgets.QFormLayout.ItemRole.FieldRole,
            self.showZeroCrossingInColorbarsCheckBox,
        )
        self.colorbarStepDivisorLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.colorbarStepDivisorLabel.setObjectName("colorbarStepDivisorLabel")
        self.formLayout_scatter.setWidget(4, QtWidgets.QFormLayout.ItemRole.LabelRole, self.colorbarStepDivisorLabel)
        self.colorbarStepDivisorSpinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.colorbarStepDivisorSpinBox.setMinimum(1)
        self.colorbarStepDivisorSpinBox.setProperty("value", 4)
        self.colorbarStepDivisorSpinBox.setObjectName("colorbarStepDivisorSpinBox")
        self.formLayout_scatter.setWidget(4, QtWidgets.QFormLayout.ItemRole.FieldRole, self.colorbarStepDivisorSpinBox)
        self.clipColorbarAtXSigmaLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.clipColorbarAtXSigmaLabel.setObjectName("clipColorbarAtXSigmaLabel")
        self.formLayout_scatter.setWidget(5, QtWidgets.QFormLayout.ItemRole.LabelRole, self.clipColorbarAtXSigmaLabel)
        self.clipColorbarAtXSigmaDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget)
        self.clipColorbarAtXSigmaDoubleSpinBox.setMaximum(10.0)
        self.clipColorbarAtXSigmaDoubleSpinBox.setSingleStep(0.1)
        self.clipColorbarAtXSigmaDoubleSpinBox.setProperty("value", 3.0)
        self.clipColorbarAtXSigmaDoubleSpinBox.setObjectName("clipColorbarAtXSigmaDoubleSpinBox")
        self.formLayout_scatter.setWidget(
            5,
            QtWidgets.QFormLayout.ItemRole.FieldRole,
            self.clipColorbarAtXSigmaDoubleSpinBox,
        )
        self.doNotPlotAxisLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.doNotPlotAxisLabel.setObjectName("doNotPlotAxisLabel")
        self.formLayout_scatter.setWidget(6, QtWidgets.QFormLayout.ItemRole.LabelRole, self.doNotPlotAxisLabel)
        self.doNotPlotAxisCheckBox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.doNotPlotAxisCheckBox.setObjectName("doNotPlotAxisCheckBox")
        self.formLayout_scatter.setWidget(6, QtWidgets.QFormLayout.ItemRole.FieldRole, self.doNotPlotAxisCheckBox)
        self.rotateScatterPlotsToMainAxisLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.rotateScatterPlotsToMainAxisLabel.setObjectName("rotateScatterPlotsToMainAxisLabel")
        self.formLayout_scatter.setWidget(
            7,
            QtWidgets.QFormLayout.ItemRole.LabelRole,
            self.rotateScatterPlotsToMainAxisLabel,
        )
        self.rotateScatterPlotsToMainAxisCheckBox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.rotateScatterPlotsToMainAxisCheckBox.setObjectName("rotateScatterPlotsToMainAxisCheckBox")
        self.formLayout_scatter.setWidget(
            7,
            QtWidgets.QFormLayout.ItemRole.FieldRole,
            self.rotateScatterPlotsToMainAxisCheckBox,
        )
        self.verticalLayout_scatter.addLayout(self.formLayout_scatter)
        spacerItem2 = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.verticalLayout_scatter.addItem(spacerItem2)
        self.horizontalLayout_forms.addLayout(self.verticalLayout_scatter)
        self.verticalLayout.addLayout(self.horizontalLayout_forms)
        self.horizontalLayout_buttons = QtWidgets.QHBoxLayout()
        self.horizontalLayout_buttons.setObjectName("horizontalLayout_buttons")
        spacerItem3 = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.horizontalLayout_buttons.addItem(spacerItem3)
        self.cancelButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.cancelButton.setObjectName("cancelButton")
        self.horizontalLayout_buttons.addWidget(self.cancelButton)
        self.okButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.okButton.setObjectName("okButton")
        self.horizontalLayout_buttons.addWidget(self.okButton)
        self.verticalLayout.addLayout(self.horizontalLayout_buttons)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 905, 21))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Plot Settings"))
        self.general_label.setText(_translate("MainWindow", "General Settings"))
        self.positionUnitLabel.setText(_translate("MainWindow", "Position Unit:"))
        self.positionUnitComboBox.setItemText(0, _translate("MainWindow", "mm"))
        self.positionUnitComboBox.setItemText(1, _translate("MainWindow", "m"))
        self.stairHistogramLabel.setText(_translate("MainWindow", "Stair Histogram:"))
        self.smoothingWindowWidthMLabel.setText(_translate("MainWindow", "Smoothing Window Width [m] or [s]:"))
        self.showMeanLineLabel.setText(_translate("MainWindow", "Show Mean Line:"))
        self.showDirectedDeviationsAlongCrossVerticalLabel.setText(
            _translate("MainWindow", "Show Directed Deviations (Along / Cross / Vertical):")
        )
        self.scatter_plot_label.setText(_translate("MainWindow", "Scatter Plot Settings"))
        self.gridMegapixelsLabel.setText(_translate("MainWindow", "Grid Megapixels:"))
        self.rMSWindowWidthMLabel.setText(_translate("MainWindow", "RMS Window Width [m] or [s]:"))
        self.showZeroCrossingInColorbarsLabel.setText(_translate("MainWindow", "Show Zero-Crossing in Colorbars:"))
        self.colorbarStepDivisorLabel.setText(_translate("MainWindow", "Colorbar Step Divisor:"))
        self.clipColorbarAtXSigmaLabel.setText(_translate("MainWindow", "Clip Colorbar at X Sigma:"))
        self.doNotPlotAxisLabel.setText(_translate("MainWindow", "Do Not Plot Axis:"))
        self.rotateScatterPlotsToMainAxisLabel.setText(_translate("MainWindow", "Rotate Scatter Plots to Main Axis:"))
        self.cancelButton.setText(_translate("MainWindow", "Cancel"))
        self.okButton.setText(_translate("MainWindow", "OK"))

    def update_view(self) -> None:
        self.gridMegapixelsDoubleSpinBox.setValue(self.settings.grid_mp)
        self.rMSWindowWidthMDoubleSpinBox.setValue(self.settings.rms_window_width)
        self.smoothingWindowWidthMDoubleSpinBox.setValue(self.settings.smoothing_window_size)
        self.positionUnitComboBox.setCurrentIndex(0 if self.settings.unit_is_mm else 1)
        self.showZeroCrossingInColorbarsCheckBox.setChecked(self.settings.always_show_zero)
        self.colorbarStepDivisorSpinBox.setValue(self.settings.c_bar_step_divisor)
        self.clipColorbarAtXSigmaDoubleSpinBox.setValue(self.settings.scatter_sigma_factor)
        self.doNotPlotAxisCheckBox.setChecked(self.settings.scatter_no_axis)
        self.rotateScatterPlotsToMainAxisCheckBox.setChecked(self.settings.scatter_rotate)
        self.stairHistogramCheckBox.setChecked(self.settings.hist_as_stairs)
        self.showMeanLineCheckBox.setChecked(self.settings.show_mean_line)
        self.showDirectedDeviationsAlongCrossVerticalCheckBox.setChecked(self.settings.show_directed_devs)

    def update_model(self) -> None:
        self.settings.grid_mp = self.gridMegapixelsDoubleSpinBox.value()
        self.settings.rms_window_width = self.rMSWindowWidthMDoubleSpinBox.value()
        self.settings.smoothing_window_size = self.smoothingWindowWidthMDoubleSpinBox.value()
        self.settings.unit_is_mm = self.positionUnitComboBox.currentIndex() == 0
        self.settings.always_show_zero = self.showZeroCrossingInColorbarsCheckBox.isChecked()
        self.settings.c_bar_step_divisor = self.colorbarStepDivisorSpinBox.value()
        self.settings.scatter_sigma_factor = self.clipColorbarAtXSigmaDoubleSpinBox.value()
        self.settings.scatter_no_axis = self.doNotPlotAxisCheckBox.isChecked()
        self.settings.scatter_rotate = self.rotateScatterPlotsToMainAxisCheckBox.isChecked()
        self.settings.hist_as_stairs = self.stairHistogramCheckBox.isChecked()
        self.settings.show_mean_line = self.showMeanLineCheckBox.isChecked()
        self.settings.show_directed_devs = self.showDirectedDeviationsAlongCrossVerticalCheckBox.isChecked()

    def handle_ok(self) -> None:
        self.update_model()
        self.plot_request.emit(PlotRequest(type=PlotRequestType.UPDATE_SETTINGS, plot_settings=self.settings))
        self.close()
