"""Interactive alignment of a trajectory to a reference.

The window guides through three phases:

1. **Place roughly**: get the trajectory onto the reference, in space and time. "Auto align" tries it on its own;
   otherwise drag the trajectory in the view, pick point pairs, match the first pose or synchronize the time.
2. **Diagnose and correct**: once the trajectories overlap, the deviation at equal time (trajectory minus
   reference) remains. Every parameter leaves its own signature in it (e.g. a time shift an along-track
   deviation proportional to the speed); the diagnosis ranks the parameters by how much of the deviation they
   explain, and the views below the map show their signatures, each with the matching correction.
3. **Least squares**: the rigorous estimation of the checked parameters, with statistics, then "Apply".

Every step can be undone. The logic lives in :mod:`trajectopy.processing._lib.alignment.interactive` (placement),
:mod:`trajectopy.processing._lib.alignment.diagnostics` (signatures, corrections) and in the
:class:`~trajectopy.gui.alignment.session.AlignmentSession`.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.core.settings import AlignmentSettings, MatchingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.gui.alignment.canvas import HINT_DETAILS, PROJECTIONS, TOOL_MOVE, TOOL_PAIRS, AlignmentCanvas
from trajectopy.gui.alignment.diagnosis import DiagnosisPanel, DiagnosticsModel
from trajectopy.gui.alignment.parameter_panel import ParameterPanel
from trajectopy.gui.alignment.plots import (
    CoordinatesView,
    DiagnosticView,
    GeometryView,
    OverTimeView,
    StandstillView,
    TimeShiftView,
)
from trajectopy.gui.alignment.session import AlignmentSession
from trajectopy.gui.utils import center_window
from trajectopy.processing._lib.alignment import diagnostics

SIDE_WIDTH = 410
ARROW_MAGNIFICATIONS = (
    ("Auto", -1.0),
    ("Off", 0.0),
    ("True size", 1.0),
    ("× 10", 10.0),
    ("× 100", 100.0),
    ("× 1000", 1000.0),
)


class InteractiveAlignmentWindow(QtWidgets.QMainWindow):
    """Align a trajectory to a reference by hand and/or by least squares, guided by a diagnosis of the deviation."""

    alignment_accepted = QtCore.Signal(object)  # AlignmentResult

    def __init__(
        self,
        trajectory: Trajectory,
        reference: Trajectory,
        matching_settings: MatchingSettings | None = None,
        alignment_settings: AlignmentSettings | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        # a child of the main window: without this, closing only hides it and it is never freed
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Interactive Alignment: {trajectory.name} to {reference.name}")
        self.session = AlignmentSession(trajectory, reference, matching_settings, alignment_settings, parent=self)
        self._override_cursor = False

        self.model = DiagnosticsModel(self.session, parent=self)
        self.canvas = AlignmentCanvas(self.session)
        self.parameters = ParameterPanel(self.session)
        self.diagnosis = DiagnosisPanel(self.model)
        self.over_time = OverTimeView(self.session)
        self.views: dict[str, DiagnosticView] = {
            "over_time": self.over_time,
            diagnostics.VIEW_TIME: TimeShiftView(self.session),
            diagnostics.VIEW_STANDSTILL: StandstillView(self.session),
            diagnostics.VIEW_GEOMETRY: GeometryView(self.session),
            "coordinates": CoordinatesView(self.session),
        }

        self._setup_ui()
        self._setup_toolbar()

        session = self.session
        session.state_changed.connect(self._update_pairs)
        session.history_changed.connect(self._update_history_actions)
        session.jumped.connect(self._follow_jump)
        session.busy_changed.connect(self._set_busy)
        session.message_changed.connect(self._show_message)
        self.model.residuals_updated.connect(self._distribute_residuals)
        self.model.updated.connect(self._distribute_residuals)
        self.canvas.magnification_changed.connect(self._show_auto_magnification)
        self.diagnosis.view_requested.connect(self.show_view)
        self.canvas.hint_changed.connect(self.hint_label.setText)
        for view in self.views.values():
            view.time_hovered.connect(self.canvas.set_cursor_time)

        self._update_history_actions()
        self.model.update()
        self.fit_view()
        self.resize(1550, 950)
        center_window(self)

    # ------------------------------------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        side = QtWidgets.QWidget()
        side.setFixedWidth(SIDE_WIDTH)
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)

        self.controls = QtWidgets.QWidget()  # everything that is unavailable while estimating
        controls_layout = QtWidgets.QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 6, 0)
        controls_layout.addWidget(self._placement_box())
        controls_layout.addWidget(self._diagnosis_box())
        controls_layout.addWidget(self._least_squares_box())
        controls_layout.addStretch()
        scroll = QtWidgets.QScrollArea()
        # a fixed width (also with a scroll bar), so that no text can widen the panel and move everything
        scrollbar_width = self.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent)
        self.controls.setFixedWidth(SIDE_WIDTH - scrollbar_width - 2)
        scroll.setWidget(self.controls)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_layout.addWidget(scroll, 1)
        side_layout.addLayout(self._accept_buttons())

        layout.addWidget(side)
        layout.addWidget(self._view_panel(), 1)
        self._setup_status_bar()

    def _setup_status_bar(self) -> None:
        """What the last step did and the progress of the estimation, in a fixed place."""
        status = self.statusBar()
        status.setSizeGripEnabled(False)
        self.message_label = QtWidgets.QLabel()
        # one line that is cut off rather than wrapped: long messages must not resize anything
        self.message_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        status.addWidget(self.message_label, 1)
        status.addPermanentWidget(self._busy_row())
        status.setFixedHeight(status.sizeHint().height() + 8)

    def _placement_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("1 · Place roughly")
        layout = QtWidgets.QGridLayout(box)
        self.auto_button = QtWidgets.QPushButton("Auto align")
        self.auto_button.setDefault(True)
        self.auto_button.setToolTip(
            "Tries several rough placements (the current one, the first pose, a fit on poses at equal time, the "
            "centers, the time from the speed profiles), keeps the closest and refines it by least squares "
            "(the parameters checked with “estimate”)."
        )
        first_pose_button = QtWidgets.QPushButton("Match first pose")
        first_pose_button.setToolTip(
            "Puts the first position onto the reference's first position and aligns the initial direction "
            "of travel. Does not use the timestamps, so it also works with an unknown time offset."
        )
        sync_button = QtWidgets.QPushButton("Sync time")
        sync_button.setToolTip(
            "Sets the time shift for which the speed profiles of both trajectories agree best. The speed does "
            "not depend on the coordinate frame, so this works before the trajectories are placed on each other."
        )
        self.pick_button = QtWidgets.QPushButton("Pick point pairs")
        self.pick_button.setCheckable(True)
        self.pick_button.setToolTip(
            "While active, click a point on the trajectory and the reference position it belongs to. One pair "
            "fixes the position, two add the heading, three or more the full rotation."
        )
        self.pick_button.toggled.connect(lambda active: self.canvas.set_tool(TOOL_PAIRS if active else TOOL_MOVE))
        self.pair_scale_check = QtWidgets.QCheckBox("fit scale")
        self.pair_scale_check.setToolTip("Also determine the scale from the point pairs (needs two pairs).")
        self.pair_scale_check.toggled.connect(self._on_pair_scale_toggled)

        pairs_row = QtWidgets.QHBoxLayout()  # always there, so that nothing moves when pairs are picked
        self.pairs_label = QtWidgets.QLabel()
        self.pairs_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.pairs_label.setToolTip("The pairs are numbered in the view.")
        self.remove_pair_button = QtWidgets.QPushButton("Remove last")
        self.remove_pair_button.setToolTip("Removes the last point pair")
        self.clear_pairs_button = QtWidgets.QPushButton("Clear")
        self.clear_pairs_button.setToolTip("Removes all point pairs")
        self.remove_pair_button.clicked.connect(lambda: self.session.remove_pair(len(self.session.pairs) - 1))
        self.clear_pairs_button.clicked.connect(self.session.clear_pairs)
        pairs_row.addWidget(self.pairs_label, 1)
        pairs_row.addWidget(self.remove_pair_button)
        pairs_row.addWidget(self.clear_pairs_button)

        self.auto_button.clicked.connect(lambda: self.session.auto_align(self.parameters.estimation_settings))
        first_pose_button.clicked.connect(self.session.match_first_pose)
        sync_button.clicked.connect(self.session.sync_time)
        layout.addWidget(self.auto_button, 0, 0, 1, 2)
        layout.addWidget(first_pose_button, 1, 0)
        layout.addWidget(sync_button, 1, 1)
        layout.addWidget(self.pick_button, 2, 0)
        layout.addWidget(self.pair_scale_check, 2, 1)
        layout.addLayout(pairs_row, 3, 0, 1, 2)
        self._update_pairs()
        return box

    def _diagnosis_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("2 · Diagnose and correct")
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(self.diagnosis)
        return box

    def _least_squares_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("3 · Least squares")
        layout = QtWidgets.QVBoxLayout(box)
        self.values_button = QtWidgets.QToolButton()
        self.values_button.setText("Values and estimated parameters")
        self.values_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.values_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.values_button.setAutoRaise(True)
        self.values_button.setToolTip("Type in values and choose the parameters that the least squares estimate")
        # not checkable: a checked tool button is filled with the accent color, which looks like a pressed
        # action rather than an expanded section
        self.values_button.clicked.connect(lambda: self._show_values(self.parameters.isHidden()))
        layout.addWidget(self.values_button)
        layout.addWidget(self.parameters)
        self.parameters.setVisible(False)
        self.refine_button = QtWidgets.QPushButton("Refine by least squares")
        self.refine_button.setToolTip(
            "Rigorous spatio-temporal least squares alignment of the parameters checked with “estimate”, starting "
            "from the placement shown. Its statistics (e.g. the covariance) are saved with the alignment."
        )
        self.refine_button.clicked.connect(lambda: self.session.refine(self.parameters.estimation_settings))
        layout.addWidget(self.refine_button)
        return box

    def _busy_row(self) -> QtWidgets.QWidget:
        self.busy_row = QtWidgets.QWidget()
        # transparent: the global style paints every plain widget, which would show as a box in the status bar
        self.busy_row.setObjectName("busyRow")
        self.busy_row.setStyleSheet("#busyRow { background: transparent; }")
        layout = QtWidgets.QHBoxLayout(self.busy_row)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)
        label = QtWidgets.QLabel("Estimating …")
        progress = QtWidgets.QProgressBar()
        progress.setRange(0, 0)  # indeterminate
        progress.setTextVisible(False)
        progress.setFixedSize(120, 4)
        cancel_button = QtWidgets.QPushButton("Cancel estimation")
        cancel_button.setToolTip("Stops waiting for the least squares alignment and discards its result.")
        cancel_button.clicked.connect(self.session.cancel_estimation)
        layout.addWidget(label)
        layout.addWidget(progress, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(cancel_button)
        self.busy_row.setVisible(False)
        return self.busy_row

    def _accept_buttons(self) -> QtWidgets.QHBoxLayout:
        layout = QtWidgets.QHBoxLayout()
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.setToolTip(
            "Aligns the trajectory as shown and saves the alignment. If the placement comes unchanged from the "
            "least squares alignment, its statistics are saved, too."
        )
        close_button = QtWidgets.QPushButton("Cancel")
        self.apply_button.clicked.connect(self.accept)
        close_button.clicked.connect(self.close)
        layout.addWidget(self.apply_button)
        layout.addWidget(close_button)
        return layout

    def _view_panel(self) -> QtWidgets.QWidget:
        upper = QtWidgets.QWidget()
        upper_layout = QtWidgets.QVBoxLayout(upper)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.addWidget(self.canvas, 1)
        self.hint_label = QtWidgets.QLabel(self.canvas.hint())
        self.hint_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.hint_label.setToolTip(HINT_DETAILS)
        upper_layout.addWidget(self.hint_label)

        self.tabs = QtWidgets.QTabWidget()
        for key, title, tooltip in (
            ("over_time", "Over time", "Deviation components and speed over time; drag the speed to shift in time"),
            (diagnostics.VIEW_TIME, "Time shift", "Along-track deviation against the speed"),
            (diagnostics.VIEW_STANDSTILL, "Shift and leverarm", "Deviation at standstills, in the body frame"),
            (diagnostics.VIEW_GEOMETRY, "Rotation and scale", "Deviation against the position relative to the pivot"),
            ("coordinates", "Coordinates", "Coordinates of both trajectories over time"),
        ):
            index = self.tabs.addTab(self.views[key], title)
            self.tabs.setTabToolTip(index, f"{tooltip}. Hover over the data to see where it is in the view above.")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(upper)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 420])
        return splitter

    def _setup_toolbar(self) -> None:
        toolbar = self.addToolBar("Interactive alignment")
        toolbar.setMovable(False)
        self.toolbar = toolbar

        self.undo_action = toolbar.addAction("Undo", self.session.undo)
        self.undo_action.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        self.redo_action = toolbar.addAction("Redo", self.session.redo)
        self.redo_action.setShortcuts([QtGui.QKeySequence("Ctrl+Y"), QtGui.QKeySequence("Ctrl+Shift+Z")])
        toolbar.addAction("Reset", self.session.reset).setToolTip("Back to the unaligned trajectory")
        toolbar.addSeparator()

        toolbar.addWidget(QtWidgets.QLabel(" View: "))
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems([projection.name for projection in PROJECTIONS])
        self.view_combo.setToolTip(
            "Direction the scene is looked at. Rotations by dragging turn about this direction."
        )
        self.view_combo.currentIndexChanged.connect(self.canvas.set_projection)
        toolbar.addWidget(self.view_combo)
        toolbar.addAction("Fit view", self.fit_view).setToolTip("Show both trajectories completely")
        toolbar.addAction("Pivot to center", self.session.center_pivot).setToolTip(
            "Puts the pivot (+) back to the center of the trajectory. Double-click in the view to place it."
        )
        toolbar.addSeparator()

        toolbar.addWidget(QtWidgets.QLabel(" Deviation arrows: "))
        self.arrows_combo = QtWidgets.QComboBox()
        for label, magnification in ARROW_MAGNIFICATIONS:
            self.arrows_combo.addItem(label, magnification)
        self.arrows_combo.setToolTip(
            "Draws the deviation at equal time as arrows from the reference, magnified. Their pattern shows the "
            "cause: parallel arrows a shift, arrows along the path a time shift, a swirl about the pivot a heading "
            "error, arrows away from (or towards) the pivot a scale error. Auto: the typical arrow is a few percent "
            "of the view."
        )
        self.arrows_combo.currentIndexChanged.connect(
            lambda index: self.canvas.set_arrow_magnification(float(self.arrows_combo.itemData(index)))
        )
        # wide enough for "Auto (× 1000)", so that the toolbar does not change when the magnification does
        self.arrows_combo.setMinimumContentsLength(len("Auto (× 1000000)"))
        self.arrows_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        toolbar.addWidget(self.arrows_combo)

    # -------------------------------------------------------------------------------------------- updates

    def show_view(self, key: str) -> None:
        self.tabs.setCurrentWidget(self.views[key])

    def fit_view(self) -> None:
        self.canvas.fit_view()
        self.over_time.fit_view()

    def _follow_jump(self) -> None:
        """After a large change, the views follow only if the trajectory left them (the zoom is kept)."""
        self.canvas.fit_view_if_hidden()
        self.over_time.fit_view(only_if_hidden=True)

    def _show_values(self, shown: bool) -> None:
        self.parameters.setVisible(shown)
        self.values_button.setArrowType(QtCore.Qt.ArrowType.DownArrow if shown else QtCore.Qt.ArrowType.RightArrow)

    def _show_message(self, message: str) -> None:
        self.message_label.setText(message)
        self.message_label.setToolTip(message)

    def _show_auto_magnification(self, magnification: float) -> None:
        text = f"Auto (× {magnification:g})" if magnification >= 1 else f"Auto (× {magnification:.2g})"
        self.arrows_combo.setItemText(0, text)

    def _distribute_residuals(self) -> None:
        for view in self.views.values():
            view.set_data(self.model.residuals, self.model.diagnosis)

    def _update_history_actions(self) -> None:
        session = self.session
        for action, enabled, label, name in (
            (self.undo_action, session.can_undo, session.undo_label, "Undo"),
            (self.redo_action, session.can_redo, session.redo_label, "Redo"),
        ):
            action.setEnabled(enabled)
            action.setToolTip(f"{name}: {label}" if enabled else f"Nothing to {name.lower()}")

    def _update_pairs(self) -> None:
        count = len(self.session.pairs)
        self.pairs_label.setText(f"{count} point pair{'' if count == 1 else 's'}" if count else "No point pairs")
        self.remove_pair_button.setEnabled(bool(count))
        self.clear_pairs_button.setEnabled(bool(count))

    def _on_pair_scale_toggled(self, checked: bool) -> None:
        self.session.pairs_with_scale = checked
        if self.session.pairs:
            self.session.fit_pairs()

    def _set_busy(self, busy: bool) -> None:
        self.controls.setEnabled(not busy)
        self.toolbar.setEnabled(not busy)
        self.apply_button.setEnabled(not busy)
        for view in self.views.values():
            view.set_actions_enabled(not busy)
        self.busy_row.setVisible(busy)
        if busy != self._override_cursor:
            self._override_cursor = busy
            if busy:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.BusyCursor)
            else:
                QtWidgets.QApplication.restoreOverrideCursor()

    # --------------------------------------------------------------------------------------------- result

    def accept(self) -> None:
        self.alignment_accepted.emit(self.session.alignment_result())
        self.close()

    def closeEvent(self, event) -> None:
        self.session.cancel_estimation()  # also restores the cursor
        super().closeEvent(event)
