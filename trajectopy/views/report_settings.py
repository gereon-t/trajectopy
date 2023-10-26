from PyQt6 import QtCore, QtWidgets


class ReportSettingsGUI(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Report Settings")
        self.resize(200, 200)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(200, 200))
        self.setMaximumSize(QtCore.QSize(200, 200))

        layout = QtWidgets.QVBoxLayout()

        # Label for the unit selection
        unit_label = QtWidgets.QLabel("Select unit:")
        layout.addWidget(unit_label)

        # Dropdown for selecting the unit
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItem("m")
        self.unit_combo.addItem("mm")
        layout.addWidget(self.unit_combo)

        # Label for the size input
        size_label = QtWidgets.QLabel("Downsample to:")
        layout.addWidget(size_label)

        # Numeric field for specifying the size
        self.size_spinbox = QtWidgets.QSpinBox()
        self.size_spinbox.setMinimum(-1)
        self.size_spinbox.setMaximum(100000)
        self.size_spinbox.setValue(2000)
        layout.addWidget(self.size_spinbox)

        # OK and Cancel buttons
        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout = QtWidgets.QVBoxLayout()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)
