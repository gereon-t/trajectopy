"""The workspace: views (tables, the 3D view, plots) that the user arranges and saves as named layouts.

Each view is a dock widget inside a nested main window, so it can be dragged next to or onto another view
or into its own window. A layout is the list of views (id and type) plus the dock arrangement as saved by Qt.
"""

import base64
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.core.settings import PlotSettings
from trajectopy.gui import themes
from trajectopy.gui.models.entries import (
    AbsoluteDeviationEntry,
    AlignmentEntry,
    RelativeDeviationEntry,
    TrajectoryEntry,
)
from trajectopy.gui.views.scene_view import SceneTrajectory, SceneView
from trajectopy.visualization import Figure, alignment_figures, deviation_figures, style, trajectory_figures

logger = logging.getLogger(__name__)

LAYOUT_FORMAT = 1


# --------------------------------------------------------------------------------------------------- data


@dataclass(frozen=True)
class WorkspaceTrajectory:
    """A checked trajectory and its most recent results."""

    entry: TrajectoryEntry
    color: str
    ate: AbsoluteDeviationEntry | None = None
    rpe: RelativeDeviationEntry | None = None
    alignment: AlignmentEntry | None = None


@dataclass(frozen=True)
class WorkspaceData:
    """Everything the views show; rebuilt by the main window whenever the data or the selection changes."""

    trajectories: Sequence[WorkspaceTrajectory] = ()
    reference: TrajectoryEntry | None = None
    selected_ids: frozenset[str] = frozenset()
    scene: Sequence[SceneTrajectory] = ()
    plot_settings: PlotSettings = field(default_factory=PlotSettings)
    settings_version: int = 0  # increased when the plot settings were edited in place

    def content_key(self, with_selection: bool = False) -> tuple:
        """Identifies what plots show: they are only redrawn when it changes (not on every selection change)."""
        trajectories = tuple(
            (
                t.entry.entry_id,
                id(t.entry.trajectory),
                t.color,
                *(r.entry_id if r is not None else None for r in (t.ate, t.rpe, t.alignment)),
            )
            for t in self.trajectories
        )
        selection = tuple(sorted(self.selected_ids)) if with_selection else ()
        return trajectories, id(self.plot_settings), self.settings_version, selection

    @property
    def evaluated(self) -> list[WorkspaceTrajectory]:
        return [t for t in self.trajectories if t.ate is not None or t.rpe is not None]


# -------------------------------------------------------------------------------------------- view contents


class ViewContent(QtWidgets.QWidget):
    """Base of what a view shows."""

    trajectory_activated = QtCore.Signal(str)  # entry id, e.g. a row clicked in the comparison table

    def set_data(self, data: WorkspaceData) -> None:
        raise NotImplementedError

    def apply_theme(self) -> None:
        """Called when the light/dark theme changes."""


class ComparisonContent(ViewContent):
    """One row per evaluated trajectory with its key metrics; the best value of each column is highlighted."""

    COLUMNS = ("Trajectory", "ATE RMS [m]", "ATE mean [m]", "ATE max [m]", "ATE rot. RMS [°]", "RPE")

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._stack = QtWidgets.QStackedWidget()
        layout.addWidget(self._stack)

        self._table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._table.cellClicked.connect(self._on_clicked)
        self._stack.addWidget(self._table)

        self._message = _message_label("")
        self._stack.addWidget(self._message)
        self._ids: list[str] = []
        self._data: WorkspaceData | None = None

    def set_data(self, data: WorkspaceData) -> None:
        self._data = data
        rows = sorted(data.evaluated, key=lambda t: t.ate.deviations.pos_dev_rms if t.ate else np.inf)
        if not rows:
            self._message.setText(
                "No evaluated trajectory is checked.\nUse “Evaluate all” to compare every trajectory with the reference."
            )
            self._stack.setCurrentWidget(self._message)
            return
        self._stack.setCurrentWidget(self._table)

        values = [self._row_values(t) for t in rows]
        best = [
            min((row[column] for row in values if row[column] is not None), default=None)
            for column in range(len(self.COLUMNS) - 1)
        ]
        self._ids = [t.entry.entry_id for t in rows]
        self._table.setRowCount(len(rows))
        for row, (trajectory, numbers) in enumerate(zip(rows, values)):
            name = QtWidgets.QTableWidgetItem(trajectory.entry.name)
            name.setIcon(_swatch(trajectory.color))
            if len(rows) > 1 and row == 0 and numbers[0] is not None:
                name.setText(f"{trajectory.entry.name}  ·  best")
                name.setToolTip("Lowest ATE RMS")
            self._table.setItem(row, 0, name)
            for column, value in enumerate(numbers):
                text = "–" if value is None else self._format(column, value, trajectory)
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                font = _monospace()
                if value is not None and len(rows) > 1 and value == best[column]:
                    font.setBold(True)
                    item.setForeground(QtGui.QColor(themes.color("success")))
                item.setFont(font)
                self._table.setItem(row, column + 1, item)
            if trajectory.entry.entry_id in data.selected_ids:
                self._table.selectRow(row)

    @staticmethod
    def _row_values(t: WorkspaceTrajectory) -> list[float | None]:
        ate = t.ate.deviations if t.ate else None
        rpe = t.rpe.deviations if t.rpe else None
        return [
            ate.pos_dev_rms if ate else None,
            ate.pos_ate if ate else None,
            ate.pos_dev_max if ate else None,
            float(np.rad2deg(ate.rot_dev_rms)) if ate and ate.abs_dev.rot_dev is not None else None,
            rpe.pos_rpe if rpe else None,
        ]

    @staticmethod
    def _format(column: int, value: float, t: WorkspaceTrajectory) -> str:
        if column == 4 and t.rpe is not None:
            return f"{value:.3f} {t.rpe.deviations.pos_drift_unit}"
        return f"{value:.3f}"

    def apply_theme(self) -> None:
        if self._data is not None:
            self.set_data(self._data)  # the highlight of the best values uses a theme color

    def _on_clicked(self, row: int, _: int) -> None:
        if row < len(self._ids):
            self.trajectory_activated.emit(self._ids[row])


class SceneContent(ViewContent):
    """The trajectories in 3D, each in its color or colored by a scalar field."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        toolbar = QtWidgets.QToolBar()
        toolbar.setIconSize(QtCore.QSize(14, 14))
        self.scene = SceneView()
        for text, tip, slot in (
            ("Fit", "Fit the view to all trajectories", lambda: self.scene.fit()),
            ("Top", "View from above", lambda: self.scene.view_from("top")),
            ("Front", "View from the front", lambda: self.scene.view_from("front")),
            ("Side", "View from the side", lambda: self.scene.view_from("side")),
            ("3D", "Oblique view", lambda: self.scene.view_from("iso")),
        ):
            action = toolbar.addAction(text, slot)
            action.setToolTip(tip)
        toolbar.addSeparator()
        frames = toolbar.addAction("Frames")
        frames.setCheckable(True)
        frames.setToolTip("Show the orientation (body frame axes) along the trajectories")
        frames.toggled.connect(self.scene.set_frames_visible)
        layout.addWidget(toolbar)
        layout.addWidget(self.scene, stretch=1)

    def set_data(self, data: WorkspaceData) -> None:
        if not data.scene:
            self.scene.clear_data()
            return
        self.scene.set_content(data.scene, selected=data.selected_ids)

    def apply_theme(self) -> None:
        self.scene.apply_theme()


class FigureContent(ViewContent):
    """One of the figures of trajectopy.visualization, rebuilt when the data changes."""

    def __init__(
        self, build: Callable[[WorkspaceData], tuple[Figure | None, list[str], str]], uses_selection: bool = False
    ) -> None:
        super().__init__()
        self._build = build
        self._uses_selection = uses_selection
        self._key: tuple | None = None
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._widget: QtWidgets.QWidget | None = None
        self._variant = ""
        self._data: WorkspaceData | None = None

    def set_data(self, data: WorkspaceData, force: bool = False) -> None:
        self._data = data
        key = data.content_key(with_selection=self._uses_selection)
        if key == self._key and not force:
            return
        self._key = key
        if self._widget is not None:
            if (selector := getattr(self._widget, "selector", None)) is not None:
                self._variant = selector.currentText()
            self._layout.removeWidget(self._widget)
            self._widget.deleteLater()
            self._widget = None

        try:
            figure, colors, message = self._build(data)
        except Exception as error:
            logger.exception("Could not create the plot")
            figure, colors, message = None, [], f"The plot could not be created:\n{error}"

        if figure is None:
            self._widget = _message_label(message)
        else:
            self._widget = _colored(figure, colors).create_widget()
            if (graphics := getattr(self._widget, "graphics", None)) is not None:
                graphics.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)  # the card draws the border
            selector = getattr(self._widget, "selector", None)
            if selector is not None and self._variant and selector.findText(self._variant) >= 0:
                selector.setCurrentText(self._variant)
        self._layout.addWidget(self._widget)

    def apply_theme(self) -> None:
        if self._data is not None:
            self.set_data(self._data, force=True)  # plots take their colors from the theme when they are drawn


def _colored(figure: Figure, colors: list[str]) -> Figure:
    """The figure, drawing its data series in ``colors`` (the colors of the trajectories in the list)."""

    def wrap(render):
        def colored_render(layout) -> None:
            token = style.series_colors.set(colors)
            try:
                render(layout)
            finally:
                style.series_colors.reset(token)

        return colored_render

    return Figure(figure.title, variants={name: wrap(render) for name, render in figure.variants.items()})


def _message_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setObjectName("emptyMessage")
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    label.setMinimumSize(160, 80)
    return label


def _monospace() -> QtGui.QFont:
    """Numbers in tables line up in a fixed-width font."""
    return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)


def _swatch(color: str) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(12, 12)
    pixmap.fill(QtGui.QColor(color))
    return QtGui.QIcon(pixmap)


# ----------------------------------------------------------------------------------------------- catalog

CHECK_TRAJECTORIES = "Check at least one trajectory in the list."
EVALUATE_FIRST = "No checked trajectory has been evaluated yet.\nUse “Evaluate all” first."


def _trajectory_figure(title: str):
    def build(data: WorkspaceData):
        if not data.trajectories:
            return None, [], CHECK_TRAJECTORIES
        trajectories = [t.entry.trajectory for t in data.trajectories]
        figure = _pick(trajectory_figures(trajectories, data.plot_settings), title)
        if figure is None:
            return None, [], f"“{title}” is not available for the checked trajectories (e.g. no orientations)."
        return figure, [t.color for t in data.trajectories], ""

    return build


def _ate_figure(title: str):
    def build(data: WorkspaceData):
        with_ate = [t for t in data.trajectories if t.ate is not None]
        if not with_ate:
            return None, [], EVALUATE_FIRST
        figure = _pick(deviation_figures([t.ate.deviations for t in with_ate], (), data.plot_settings), title)
        if figure is None:
            return None, [], f"“{title}” is not available for these results."
        return figure, [t.color for t in with_ate], ""

    return build


def _rpe_figure(title: str):
    def build(data: WorkspaceData):
        with_rpe = [t for t in data.trajectories if t.rpe is not None]
        if not with_rpe:
            return None, [], EVALUATE_FIRST
        figure = _pick(deviation_figures((), [t.rpe.deviations for t in with_rpe], data.plot_settings), title)
        if figure is None:
            return None, [], f"No “{title}” results: the RPE was computed with another pair distance unit."
        return figure, [t.color for t in with_rpe], ""

    return build


def _alignment_figure(title: str):
    def build(data: WorkspaceData):
        aligned = [t for t in data.trajectories if t.alignment is not None]
        chosen = next((t for t in aligned if t.entry.entry_id in data.selected_ids), aligned[0] if aligned else None)
        if chosen is None:
            return None, [], "No checked trajectory has an alignment.\nIt is estimated by “Evaluate all” (aligned)."
        return _pick(alignment_figures(chosen.alignment.estimated_parameters, data.plot_settings), title), [], ""

    return build


def _pick(figures: Sequence[Figure], title: str) -> Figure | None:
    return next((figure for figure in figures if figure.title == title), None)


@dataclass(frozen=True)
class ViewType:
    key: str
    label: str
    create: Callable[[], ViewContent]


VIEW_TYPES: tuple[ViewType, ...] = (
    ViewType("comparison", "Comparison table", ComparisonContent),
    ViewType("scene", "3D view", SceneContent),
    ViewType("map", "Trajectory · Map", lambda: FigureContent(_trajectory_figure("Map"))),
    ViewType("positions", "Trajectory · Positions", lambda: FigureContent(_trajectory_figure("Positions"))),
    ViewType("xyz", "Trajectory · XYZ over time", lambda: FigureContent(_trajectory_figure("XYZ"))),
    ViewType("rpy", "Trajectory · Orientation (RPY)", lambda: FigureContent(_trajectory_figure("RPY"))),
    ViewType("ate", "ATE · Over time", lambda: FigureContent(_ate_figure("ATE"))),
    ViewType("ate_components", "ATE · Components", lambda: FigureContent(_ate_figure("ATE Components"))),
    ViewType("ate_histograms", "ATE · Histograms", lambda: FigureContent(_ate_figure("ATE Histograms"))),
    ViewType("ate_statistics", "ATE · Statistics", lambda: FigureContent(_ate_figure("ATE Statistics"))),
    ViewType("ate_distribution", "ATE · Distribution", lambda: FigureContent(_ate_figure("ATE Distribution"))),
    ViewType("ate_map", "ATE · Map", lambda: FigureContent(_ate_figure("ATE Map"))),
    ViewType("rpe_meter", "RPE · Per meter", lambda: FigureContent(_rpe_figure("RPE (per meter)"))),
    ViewType("rpe_second", "RPE · Per second", lambda: FigureContent(_rpe_figure("RPE (per second)"))),
    ViewType(
        "alignment_covariance",
        "Alignment · Parameter covariance",
        lambda: FigureContent(_alignment_figure("Parameter Covariance"), uses_selection=True),
    ),
    ViewType(
        "alignment_correlation",
        "Alignment · Parameter correlation",
        lambda: FigureContent(_alignment_figure("Parameter Correlation"), uses_selection=True),
    ),
)
VIEW_TYPE = {view_type.key: view_type for view_type in VIEW_TYPES}

# Built-in layouts: rows of views, top to bottom, with their relative heights; the views of a row are side by side.
BUILT_IN_LAYOUTS: dict[str, list[tuple[int, list[str]]]] = {
    "Evaluation": [(1, ["comparison"]), (3, ["scene", "ate"])],
    "Error analysis": [(1, ["ate", "ate_statistics"]), (1, ["ate_histograms", "rpe_meter"])],
    "Trajectory check": [(1, ["scene", "map"]), (1, ["xyz", "rpy"])],
}
DEFAULT_LAYOUT = "Evaluation"


# ----------------------------------------------------------------------------------------------- the docks


class ViewDock(QtWidgets.QDockWidget):
    """A view: a title bar with the content type, "own window" and "remove" buttons, and the content."""

    type_changed = QtCore.Signal(object)  # the dock
    remove_requested = QtCore.Signal(object)  # the dock

    def __init__(self, view_id: str, view_type: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName(view_id)  # identifies the view when a layout is restored
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.view_type = ""
        self.content: ViewContent | None = None

        bar = QtWidgets.QWidget()
        bar.setObjectName("viewTitleBar")
        bar.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
        bar_layout = QtWidgets.QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 6, 6, 6)
        bar_layout.setSpacing(4)
        self._type_box = QtWidgets.QComboBox()
        self._type_box.setToolTip("What this view shows")
        for view_type_option in VIEW_TYPES:
            self._type_box.addItem(view_type_option.label, view_type_option.key)
        self._type_box.activated.connect(lambda _: self._set_type(self._type_box.currentData()))
        bar_layout.addWidget(self._type_box)
        grip = QtWidgets.QLabel("")  # empty title bar space: drag here to move the view
        grip.setToolTip("Drag to move this view next to or onto another one")
        bar_layout.addWidget(grip, stretch=1)
        bar_layout.addWidget(
            self._tool_button("⧉", "Open this view in its own window (again to dock it)", self._toggle_float)
        )
        bar_layout.addWidget(self._tool_button("✕", "Remove this view", lambda: self.remove_requested.emit(self)))
        self.setTitleBarWidget(bar)

        self._set_type(view_type, notify=False)

    @staticmethod
    def _tool_button(text: str, tip: str, slot: Callable[[], None]) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setAutoRaise(True)
        button.clicked.connect(slot)
        return button

    def _toggle_float(self) -> None:
        self.setFloating(not self.isFloating())

    def _set_type(self, view_type: str, notify: bool = True) -> None:
        if view_type == self.view_type or view_type not in VIEW_TYPE:
            return
        self.view_type = view_type
        self._type_box.setCurrentIndex(self._type_box.findData(view_type))
        self.setWindowTitle(VIEW_TYPE[view_type].label)
        if self.content is not None:
            self.content.deleteLater()
        self.content = VIEW_TYPE[view_type].create()
        self.content.setObjectName("viewContent")  # drawn as the body of a card below the title bar
        self.content.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
        self.content.setContentsMargins(1, 0, 1, 1)
        self.setWidget(self.content)
        if notify:
            self.type_changed.emit(self)


class Workspace(QtWidgets.QMainWindow):
    """Arranges the views; knows how to save and restore their arrangement."""

    changed = QtCore.Signal()  # the arrangement or the views were changed by the user
    trajectory_activated = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AllowNestedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
            | QtWidgets.QMainWindow.DockOption.AnimatedDocks
        )
        self.setCentralWidget(None)
        self._docks: list[ViewDock] = []
        self._next_id = 1
        self._data = WorkspaceData()
        self._loading = False

    @property
    def docks(self) -> list[ViewDock]:
        return list(self._docks)

    def set_data(self, data: WorkspaceData) -> None:
        self._data = data
        for dock in self._docks:
            dock.content.set_data(data)

    def apply_theme(self) -> None:
        for dock in self._docks:
            dock.content.apply_theme()

    # -------------------------------------------------------------------------------------- views

    def add_view(self, view_type: str, view_id: str | None = None) -> ViewDock:
        """Adds a view to the right of the existing ones."""
        dock = self._create_dock(view_type, view_id)
        visible = [d for d in self._docks if d is not dock and not d.isFloating()]
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        if visible:
            self.splitDockWidget(visible[-1], dock, QtCore.Qt.Orientation.Horizontal)
        dock.content.set_data(self._data)
        self._notify_changed()
        return dock

    def _create_dock(self, view_type: str, view_id: str | None) -> ViewDock:
        if view_id is None:
            view_id = self._new_id()
        dock = ViewDock(view_id, view_type, self)
        dock.type_changed.connect(self._on_type_changed)
        dock.remove_requested.connect(self.remove_view)
        dock.dockLocationChanged.connect(lambda _: self._notify_changed())
        dock.topLevelChanged.connect(lambda _: self._notify_changed())
        self._connect_content(dock)
        self._docks.append(dock)
        return dock

    def _new_id(self) -> str:
        used = {dock.objectName() for dock in self._docks}
        while f"view{self._next_id}" in used:
            self._next_id += 1
        self._next_id += 1
        return f"view{self._next_id - 1}"

    def _connect_content(self, dock: ViewDock) -> None:
        dock.content.trajectory_activated.connect(self.trajectory_activated)

    def _on_type_changed(self, dock: ViewDock) -> None:
        self._connect_content(dock)
        dock.content.set_data(self._data)
        self._notify_changed()

    def remove_view(self, dock: ViewDock) -> None:
        if dock not in self._docks:
            return
        self._docks.remove(dock)
        self.removeDockWidget(dock)
        dock.deleteLater()
        self._notify_changed()

    def _clear(self) -> None:
        for dock in self._docks:
            self.removeDockWidget(dock)
            dock.deleteLater()
        self._docks = []

    def _notify_changed(self) -> None:
        if not self._loading:
            self.changed.emit()

    # ------------------------------------------------------------------------------------ layouts

    def to_dict(self) -> dict:
        return {
            "format": LAYOUT_FORMAT,
            "views": [{"id": dock.objectName(), "type": dock.view_type} for dock in self._docks],
            "state": base64.b64encode(bytes(self.saveState(LAYOUT_FORMAT))).decode("ascii"),
        }

    def load(self, layout: dict) -> bool:
        """Restores a layout saved by :meth:`to_dict`; returns False if it is not readable."""
        views = [view for view in layout.get("views", []) if view.get("type") in VIEW_TYPE]
        if layout.get("format") != LAYOUT_FORMAT or not isinstance(layout.get("views"), list):
            return False
        self._loading = True
        try:
            self._clear()
            for view in views:
                dock = self._create_dock(view["type"], str(view["id"]))
                self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
            restored = self.restoreState(QtCore.QByteArray(base64.b64decode(layout.get("state", ""))), LAYOUT_FORMAT)
            if not restored:
                self._arrange_rows([[dock] for dock in self._docks])
            for dock in self._docks:
                dock.content.set_data(self._data)
        finally:
            self._loading = False
        return True

    def load_built_in(self, name: str) -> None:
        self._loading = True
        try:
            self._clear()
            spec = BUILT_IN_LAYOUTS[name]
            rows = [[self._create_dock(view_type, None) for view_type in row] for _, row in spec]
            self._arrange_rows(rows)
            # applied once the workspace has its size; until then Qt would distribute the space evenly
            starts, weights = [row[0] for row in rows], [100 * weight for weight, _ in spec]
            QtCore.QTimer.singleShot(0, lambda: self._resize_rows(starts, weights))
            for dock in self._docks:
                dock.content.set_data(self._data)
        finally:
            self._loading = False

    def _resize_rows(self, starts: list[ViewDock], weights: list[int]) -> None:
        if all(dock in self._docks for dock in starts):
            self.resizeDocks(starts, weights, QtCore.Qt.Orientation.Vertical)

    def _arrange_rows(self, rows: list[list[ViewDock]]) -> None:
        rows = [row for row in rows if row]
        if not rows:
            return
        # first stack the first view of every row, so that each row spans the full width, then fill the rows
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, rows[0][0])
        for above, below in zip(rows, rows[1:]):
            self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, below[0])
            self.splitDockWidget(above[0], below[0], QtCore.Qt.Orientation.Vertical)
        for row in rows:
            for left, right in zip(row, row[1:]):
                self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, right)
                self.splitDockWidget(left, right, QtCore.Qt.Orientation.Horizontal)
