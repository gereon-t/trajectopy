"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
import os
import shutil
from tempfile import mkdtemp
from typing import Callable, Dict, Union

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QCloseEvent

from trajectopy.api import ReportSettings
from trajectopy.gui.managers.file_manager import FileManager
from trajectopy.gui.managers.plot_manager import PlotManager
from trajectopy.gui.managers.requests import (
    PlotRequest,
    ReportSettingsRequest,
    ReportSettingsRequestType,
    UIRequest,
    UIRequestType,
    generic_request_handler,
)
from trajectopy.gui.managers.session_manager import SessionManager
from trajectopy.gui.managers.trajectory_manager import TrajectoryManager
from trajectopy.gui.managers.ui_manager import UIManager
from trajectopy.gui.models.result_model import ResultTableModel
from trajectopy.gui.models.trajectory_model import TrajectoryTableModel
from trajectopy.gui.views.about_window import AboutGUI
from trajectopy.gui.views.json_settings_view import JSONViewer
from trajectopy.gui.views.progress_window import ProgressWindow
from trajectopy.gui.views.result_table_view import ResultTableView
from trajectopy.gui.views.trajectory_table_view import TrajectoryTableView
from trajectopy.path import VERSION, YEAR

logger = logging.getLogger("root")


class TrajectopyGUI(QtWidgets.QMainWindow):
    """
    Class representing the main window of the GUI

    This class is the main window of the GUI.
    It contains the table views for trajectories
    and results as well as the menu bar.
    """

    operation_started = pyqtSignal()
    operation_finished = pyqtSignal()
    ui_request = pyqtSignal(UIRequest)

    def __init__(
        self,
        single_thread: bool = False,
        report_settings_path: str = "",
        report_output_path: str = "",
        mapbox_token: str = "",
    ) -> None:
        QtWidgets.QMainWindow.__init__(self)

        self.REQUEST_MAPPING: Dict[ReportSettingsRequestType, Callable[[ReportSettingsRequest], None]] = {
            ReportSettingsRequestType.EXPORT: self.handle_report_settings_export,
            ReportSettingsRequestType.IMPORT: self.handle_report_settings_import,
            ReportSettingsRequestType.RESET: self.handle_report_settings_reset,
        }

        self.trajectory_table_model = TrajectoryTableModel()
        self.result_table_model = ResultTableModel()
        self.setupUi()

        self.computation_thread = QtCore.QThread(parent=self)

        self.trajectory_manager = TrajectoryManager()
        self.ui_manager = UIManager(parent=self)
        self.file_manager = FileManager()
        self.session_manager = SessionManager()

        self.report_settings = self.get_report_settings(report_settings_path)
        self.report_settings.scatter_mapbox_token = mapbox_token or self.get_mapbox_token()
        self.report_output_path = self.get_report_directory(report_output_path)
        self.plot_manager = PlotManager(report_dir=self.report_output_path)

        if not single_thread:
            self.trajectory_manager.moveToThread(self.computation_thread)
            self.file_manager.moveToThread(self.computation_thread)
            self.plot_manager.moveToThread(self.computation_thread)
            logger.info("Multithreading enabled")
        else:
            logger.info("Multithreading disabled")

        self.about_window = AboutGUI(parent=self, version_str=VERSION, year_str=YEAR)
        self.progress_window = ProgressWindow(parent=self)

        self.ui_request.connect(self.ui_manager.handle_request)
        self.setup_io_connections()
        self.setup_worker_connections()
        self.setup_progress_connections()
        self.setup_menu_bar()

        self.computation_thread.start()
        self.show()

    def get_report_directory(self, report_output_path) -> str:
        if not report_output_path:
            self.temp_dir = True
            return mkdtemp(prefix="trajectopy_reports_")
        else:
            self.temp_dir = False
            return report_output_path

    def get_report_settings(self, report_settings_path: str = "") -> ReportSettings:
        if report_settings_path:
            try:
                logger.info("Loaded report settings from %s", report_settings_path)
                return ReportSettings.from_file(report_settings_path)
            except Exception as e:
                logger.error("Could not load report settings file: %s", e)
                return ReportSettings()
        else:
            return ReportSettings()

    def get_mapbox_token(self) -> str:
        mapbox_token = os.environ.get("MAPBOX_TOKEN")

        if mapbox_token:
            logger.info("Using mapbox token from environment variable")
            return mapbox_token

        mapbox_token_file = os.path.join(os.getcwd(), ".mapbox_token")
        if os.path.isfile(mapbox_token_file):
            with open(mapbox_token_file, "r", encoding="utf-8") as f:
                mapbox_token = f.read()
                logger.info("Using mapbox token from file")
                return mapbox_token

        return ""

    def setup_menu_bar(self):
        """
        This method sets up the menu bar containing
        the session, plot settings and about menu
        """
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        session_menu = QtWidgets.QMenu("Session", parent=self)
        new_session_action = QAction("New", parent=self)
        new_session_action.triggered.connect(self.handle_new_session)
        session_menu.addAction(new_session_action)

        load_session_action = QAction("Load", parent=self)
        load_session_action.triggered.connect(self.handle_import_session)
        session_menu.addAction(load_session_action)

        save_session_action = QAction("Save", parent=self)
        save_session_action.triggered.connect(self.handle_export_session)
        session_menu.addAction(save_session_action)
        menubar.addMenu(session_menu)

        report_settings_action = QAction("Report Settings", parent=self)
        report_settings_action.triggered.connect(self.handle_show_report_settings)
        menubar.addAction(report_settings_action)

        about_action = QAction("About", parent=self)
        about_action.triggered.connect(self.about_window.show)
        menubar.addAction(about_action)

    def closeEvent(self, a0: Union[QCloseEvent, None]) -> None:
        self.computation_thread.quit()
        self.computation_thread.wait()

        if self.temp_dir:
            shutil.rmtree(self.report_output_path)
            logger.info("Removed temporary report directory %s", self.report_output_path)

        return super().closeEvent(a0)

    def setup_worker_connections(self):
        """This method sets up the signal connections between the worker thread and the main thread"""
        self.setup_trajectory_table_view_connections()
        self.setup_result_table_view_connections()
        self.setup_ui_manager_connections()
        self.setup_session_manager_connections()
        self.setup_file_manager_connections()
        self.setup_plot_manager_connections()
        self.setup_trajectory_manager_connections()
        self.setup_result_table_connections()
        self.setup_trajectory_table_connections()
        self.setup_plottings_connections()

    def setup_plot_manager_connections(self):
        self.plot_manager.ui_request.connect(self.ui_manager.handle_request)

    def setup_trajectory_table_connections(self):
        self.trajectory_table_model.file_request.connect(self.file_manager.handle_request)

    def setup_result_table_connections(self):
        self.result_table_model.ui_request.connect(self.ui_manager.handle_request)
        self.result_table_model.file_request.connect(self.file_manager.handle_request)

    def setup_trajectory_manager_connections(self):
        self.trajectory_manager.ui_request.connect(self.ui_manager.handle_request)
        self.trajectory_manager.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.trajectory_manager.result_model_request.connect(self.result_table_model.handle_request)

    def setup_file_manager_connections(self):
        self.file_manager.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.file_manager.result_model_request.connect(self.result_table_model.handle_request)
        self.file_manager.ui_request.connect(self.ui_manager.handle_request)

    def setup_session_manager_connections(self):
        self.session_manager.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.session_manager.result_model_request.connect(self.result_table_model.handle_request)
        self.session_manager.file_request.connect(self.file_manager.handle_request)
        self.session_manager.ui_request.connect(self.ui_manager.handle_request)
        self.session_manager.report_settings_request.connect(self.handle_report_settings_request)

    def setup_ui_manager_connections(self):
        self.ui_manager.trajectory_manager_request.connect(self.trajectory_manager.handle_request)
        self.ui_manager.file_request.connect(self.file_manager.handle_request)
        self.ui_manager.session_manager_request.connect(self.session_manager.handle_request)
        self.ui_manager.ui_request.connect(self.ui_manager.handle_request)
        self.ui_manager.result_model_request.connect(self.result_table_model.handle_request)

    def setup_result_table_view_connections(self):
        self.resultTableView.result_model_request.connect(self.result_table_model.handle_request)
        self.resultTableView.ui_request.connect(self.ui_manager.handle_request)
        self.resultTableView.file_request.connect(self.file_manager.handle_request)

    def setup_trajectory_table_view_connections(self):
        self.trajectoryTableView.trajectory_model_request.connect(self.trajectory_table_model.handle_request)
        self.trajectoryTableView.trajectory_manager_request.connect(self.trajectory_manager.handle_request)
        self.trajectoryTableView.ui_request.connect(self.ui_manager.handle_request)
        self.trajectoryTableView.file_request.connect(self.file_manager.handle_request)
        self.trajectoryTableView.result_model_request.connect(self.result_table_model.handle_request)

    def setup_progress_connections(self):
        self.trajectory_manager.operation_started.connect(self.progress_window.handle_show_request)
        self.trajectory_manager.operation_finished.connect(self.progress_window.handle_close_request)

        self.file_manager.operation_started.connect(self.progress_window.handle_show_request)
        self.file_manager.operation_finished.connect(self.progress_window.handle_close_request)

        self.plot_manager.operation_started.connect(self.progress_window.handle_show_request)
        self.plot_manager.operation_finished.connect(self.progress_window.handle_close_request)

    def setup_plottings_connections(self):
        self.resultTableView.plot_request.connect(self.inject_report_settings)
        self.trajectoryTableView.plot_request.connect(self.inject_report_settings)

    def setup_io_connections(self):
        self.trajectory_manager.update_view.connect(self.refresh)
        self.addTrajectoryButton.clicked.connect(
            lambda: self.ui_manager.handle_request(UIRequest(type=UIRequestType.IMPORT_TRAJ))
        )
        self.addResultButton.clicked.connect(
            lambda: self.ui_manager.handle_request(UIRequest(type=UIRequestType.IMPORT_RES))
        )

    @QtCore.pyqtSlot()
    def refresh(self) -> None:
        self.trajectory_table_model.layoutChanged.emit()
        self.result_table_model.layoutChanged.emit()

    @pyqtSlot(ReportSettingsRequest)
    def handle_report_settings_request(self, request: ReportSettingsRequest) -> None:
        """Logic for handling a request."""
        generic_request_handler(self, request, passthrough_request=True)

    def handle_show_report_settings(self) -> None:
        viewer = JSONViewer(settings=self.report_settings, parent=self)
        viewer.show()

    def handle_report_settings_export(self, request: ReportSettingsRequest) -> None:
        self.report_settings.to_file(os.path.join(request.file_path, "report_settings.json"))

    def handle_report_settings_import(self, request: ReportSettingsRequest) -> None:
        self.report_settings = ReportSettings.from_file(os.path.join(request.file_path, "report_settings.json"))

    def handle_report_settings_reset(self, _: ReportSettingsRequest) -> None:
        self.report_settings = ReportSettings()

    @QtCore.pyqtSlot(PlotRequest)
    def inject_report_settings(self, plot_request: PlotRequest) -> None:
        logger.debug("Injecting report settings into plot request of type %s", plot_request.type)
        plot_request.report_settings = self.report_settings
        self.plot_manager.handle_request(plot_request)

    @QtCore.pyqtSlot()
    def handle_import_session(self) -> None:
        self.ui_manager.handle_request(UIRequest(type=UIRequestType.IMPORT_SESSION))

    @QtCore.pyqtSlot()
    def handle_export_session(self) -> None:
        self.ui_manager.handle_request(UIRequest(type=UIRequestType.EXPORT_SESSION))

    @QtCore.pyqtSlot()
    def handle_new_session(self) -> None:
        self.ui_manager.handle_request(UIRequest(type=UIRequestType.CONFIRM_RESET))

    def center(self):
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()

        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def setupUi(self):
        """This method sets up the GUI"""
        self.setObjectName("MainWindow")
        self.resize(640, 480)
        self.center()

        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(640, 480))
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.centralwidget.setLayout(self.verticalLayout)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.trajectoryTableView = TrajectoryTableView(
            parent=self.verticalLayoutWidget,
            trajectory_table_model=self.trajectory_table_model,
        )
        # span over whole table width
        header = self.trajectoryTableView.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.verticalLayout.addWidget(self.trajectoryTableView)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem1 = QtWidgets.QSpacerItem(
            0,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.horizontalLayout_3.addItem(spacerItem1)
        self.addTrajectoryButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.addTrajectoryButton.setObjectName("addTrajectoryButton")
        self.horizontalLayout_3.addWidget(self.addTrajectoryButton)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(-1, -1, -1, 10)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        spacerItem2 = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.horizontalLayout_4.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.resultTableView = ResultTableView(
            parent=self.verticalLayoutWidget, result_table_model=self.result_table_model
        )
        self.verticalLayout.addWidget(self.resultTableView)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setContentsMargins(-1, -1, -1, 10)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem3 = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.horizontalLayout_5.addItem(spacerItem3)
        self.addResultButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.addResultButton.setObjectName("addResultButton")
        self.horizontalLayout_5.addWidget(self.addResultButton)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.setCentralWidget(self.centralwidget)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Trajectopy - Trajectory Evaluation in Python"))
        self.label.setText(_translate("MainWindow", "Trajectories:"))
        self.addTrajectoryButton.setText(_translate("MainWindow", "Add"))
        self.label_2.setText(_translate("MainWindow", "Results:"))
        self.addResultButton.setText(_translate("MainWindow", "Add"))
