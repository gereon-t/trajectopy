"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
from typing import Dict

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QGuiApplication
from trajectopy_core.util.entries import PropertyEntry

from trajectopy.gui.managers.requests import PropertyModelRequest, PropertyModelRequestType
from trajectopy.gui.models.property_model import PropertyTableModel
from trajectopy.gui.util import save_file_dialog


class PropertiesGUI(QtWidgets.QMainWindow):
    """
    Window for displaying properties of trajectories and results.
    """

    def __init__(self, parent, num_cols: int = 2) -> None:
        super().__init__(parent=parent)
        self.setupUi()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tableView)
        layout.addWidget(self.exportButton)
        self.centralwidget.setLayout(layout)
        self.setCentralWidget(self.centralwidget)

        self.property_table_model = PropertyTableModel(num_cols=num_cols)
        self.tableView.setModel(self.property_table_model)
        header = self.tableView.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.exportButton.clicked.connect(self.prepare_export)

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(600, 280)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(600, 280))
        self.setMaximumSize(QtCore.QSize(16777215, 16777215))
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        desired_pos = QtCore.QPoint(screen_geometry.center().x() - 300, screen_geometry.center().y() - 140)
        self.move(desired_pos)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(10, 10, 580, 220))
        self.tableView.setObjectName("tableView")

        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 20))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.exportButton = QtWidgets.QPushButton(self.centralwidget)
        self.exportButton.setObjectName("okButton")

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Property Viewer"))
        self.exportButton.setText(_translate("Form", "Export"))

    def prepare_export(self) -> None:
        if selected_file := save_file_dialog(None, file_filter="CSV file (*.csv);;All Files (*)"):
            self.property_table_model.handle_request(
                PropertyModelRequest(type=PropertyModelRequestType.EXPORT, file_path=selected_file)
            )
        else:
            return

    def refresh(self) -> None:
        self.property_table_model.layoutChanged.emit()

    def reset(self) -> None:
        self.property_table_model.items = []
        self.refresh()

    def add_entry(self, entry: PropertyEntry) -> None:
        self.property_table_model.items.append(entry)
        self.refresh()

    def add_from_dict(self, input_dict: Dict[str, str]) -> None:
        for key, values in input_dict.items():
            self.property_table_model.items.append(PropertyEntry(name=key, values=tuple(values)))
        self.refresh()
