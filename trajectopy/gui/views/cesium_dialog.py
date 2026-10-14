from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.gui.utils import app_settings

MAP_STYLE_KEY = "playback/map_style"
ION_TOKEN_KEY = "playback/ion_token"
HEIGHTS_KEY = "playback/heights"
HEIGHT_MODES = {
    "ground": "On the ground (terrain / 3D tiles)",
    "ellipsoidal": "Trajectory heights: ellipsoidal (e.g. GNSS)",
    "geoid": "Trajectory heights: above sea level (EGM2008)",
}
MAP_STYLES = {
    "osm": "OpenStreetMap",
    "satellite": "Satellite (Esri World Imagery)",
    "photorealistic": "Photorealistic 3D (Google, needs a token)",
}


class CesiumPlaybackDialog(QtWidgets.QDialog):
    """Options of the 3D playback in the browser: color (trajectories) or deviation field (ATE results), map
    and the optional Cesium ion token. The map and the token are remembered."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        fields: list[tuple[str, str]] | None = None,
        field: str = "position",
        color: QtGui.QColor | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Playback in the Browser")
        self._settings = app_settings()
        self.color = color or QtGui.QColor("#ff0000")
        form = QtWidgets.QFormLayout()

        self.field_box: QtWidgets.QComboBox | None = None
        if fields is not None:
            self.field_box = QtWidgets.QComboBox()
            for key, label in fields:
                self.field_box.addItem(label, key)
            self.field_box.setCurrentIndex(max(0, self.field_box.findData(field)))
            form.addRow("Color by:", self.field_box)
        else:
            self.color_button = QtWidgets.QPushButton()
            self.color_button.clicked.connect(self._choose_color)
            self._show_color()
            form.addRow("Color:", self.color_button)

        self.map_box = QtWidgets.QComboBox()
        for key, label in MAP_STYLES.items():
            self.map_box.addItem(label, key)
        self.map_box.setCurrentIndex(max(0, self.map_box.findData(self._settings.value(MAP_STYLE_KEY, "osm"))))
        self.map_box.setToolTip("The map shown first; it can be switched on the page.")
        form.addRow("Map:", self.map_box)

        self.heights_box = QtWidgets.QComboBox()
        for key, label in HEIGHT_MODES.items():
            self.heights_box.addItem(label, key)
        self.heights_box.setCurrentIndex(
            max(0, self.heights_box.findData(self._settings.value(HEIGHTS_KEY, "ground")))
        )
        self.heights_box.setToolTip(
            "Where the trajectory is drawn vertically. Trajectory heights above sea level are converted with the "
            "EGM2008 geoid model, which is downloaded (once) from cdn.proj.org."
        )
        form.addRow("Heights:", self.heights_box)

        self.token_edit = QtWidgets.QLineEdit(str(self._settings.value(ION_TOKEN_KEY, "")))
        self.token_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.token_edit.setPlaceholderText("optional")
        form.addRow("Cesium ion token:", self.token_edit)
        hint = QtWidgets.QLabel(
            'With a free <a href="https://ion.cesium.com/tokens">Cesium ion access token</a>, the globe also shows '
            "terrain and 3D buildings, or photorealistic 3D tiles. The token is stored on this computer and written "
            "into the exported page."
        )
        hint.setWordWrap(True)
        hint.setOpenExternalLinks(True)
        hint.setEnabled(False)
        form.addRow(hint)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Export and Open…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setMinimumWidth(460)

    @property
    def field(self) -> str:
        return str(self.field_box.currentData()) if self.field_box is not None else ""

    @property
    def map_style(self) -> str:
        return str(self.map_box.currentData())

    @property
    def heights(self) -> str:
        return str(self.heights_box.currentData())

    @property
    def ion_token(self) -> str:
        return self.token_edit.text().strip()

    def accept(self) -> None:
        self._settings.setValue(MAP_STYLE_KEY, self.map_style)
        self._settings.setValue(ION_TOKEN_KEY, self.ion_token)
        self._settings.setValue(HEIGHTS_KEY, self.heights)
        super().accept()

    def _choose_color(self) -> None:
        color = QtWidgets.QColorDialog.getColor(self.color, self, "Trajectory Color")
        if color.isValid():
            self.color = color
            self._show_color()

    def _show_color(self) -> None:
        self.color_button.setText(self.color.name())
        icon = QtGui.QPixmap(QtCore.QSize(14, 14))
        icon.fill(self.color)
        self.color_button.setIcon(QtGui.QIcon(icon))
