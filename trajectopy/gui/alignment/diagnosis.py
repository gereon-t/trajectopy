"""What explains the remaining deviation: the diagnosis of the current placement and its panel.

The panel keeps its size: the table has a fixed number of rows and the texts a fixed number of lines, so that
nothing moves while the placement changes. The deviation is updated live; the ranking of the parameters only
once the input pauses, so that the rows do not reorder under the mouse while dragging.
"""

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.gui.alignment.session import AlignmentSession
from trajectopy.processing._lib.alignment import diagnostics
from trajectopy.processing._lib.alignment.diagnostics import Correction, Diagnosis, Residuals

RESIDUALS_DELAY_MS = 60  # the deviation follows the placement closely
DIAGNOSIS_DELAY_MS = 350  # the ranking waits until the input pauses
MIN_EXPLAINED = 1e-4  # m: parameters that explain less are not listed (besides those not significant)
RELEVANT_SHARE = 0.1  # nor those that explain less than this share of what remains after correcting everything
ROWS = 6  # visible rows; more parameters are listed, too (the table scrolls)
MAX_ROWS = len(diagnostics.PARAMETERS)
SUMMARY_LINES = 2
NOTE_LINES = 3
UNEXPLAINED_WARNING = 0.3  # below this explained fraction, the rest is not an alignment error
FAR_FRACTION = 0.05  # of the extent: beyond, the trajectory is not yet placed roughly
POTENTIAL_GAIN = 0.2  # a parameter that is not estimated is pointed out if it reduces the RMS by more than this


def format_correction(key: str, value: float) -> str:
    """A correction (in the unit of ``diagnostics.perturbed``) for display."""
    if key.startswith("rot_"):
        return f"{np.rad2deg(value):+.4f}°"
    if key == "scale":
        return f"{value * 1e6:+.0f} ppm" if abs(value) < 1e-2 else f"×{1 + value:.5f}"
    if key == "time_shift":
        return f"{value * 1000:+.1f} ms" if abs(value) < 1 else f"{value:+.3f} s"
    return f"{value * 1000:+.1f} mm" if abs(value) < 0.1 else f"{value:+.3f} m"


class DiagnosticsModel(QtCore.QObject):
    """The residuals (soon after every change) and the diagnosis (once the changes pause) of the placement."""

    residuals_updated = QtCore.Signal()
    updated = QtCore.Signal()  # the diagnosis, too

    def __init__(self, session: AlignmentSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.residuals: Residuals | None = None
        self.diagnosis: Diagnosis | None = None
        self._residuals_timer = self._timer(RESIDUALS_DELAY_MS, self.update_residuals)
        self._diagnosis_timer = self._timer(DIAGNOSIS_DELAY_MS, self.update)
        session.state_changed.connect(self._residuals_timer.start)
        session.state_changed.connect(self._diagnosis_timer.start)
        session.estimation_changed.connect(self.update)

    def _timer(self, delay: int, slot) -> QtCore.QTimer:
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(delay)
        timer.timeout.connect(slot)
        return timer

    def update_residuals(self) -> None:
        session = self.session
        self.residuals = diagnostics.residuals(session.preview, session.state, session.pivot)
        self.residuals_updated.emit()

    def update(self) -> None:
        """Recomputes everything now."""
        self._residuals_timer.stop()
        self._diagnosis_timer.stop()
        self.update_residuals()
        session = self.session
        self.diagnosis = (
            diagnostics.diagnose(session.preview, session.state, session.pivot, session.estimated_parameters)
            if self.residuals is not None
            else None
        )
        self.updated.emit()


class _FixedLabel(QtWidgets.QLabel):
    """A label of a fixed number of lines; longer texts are cut off and shown completely as tooltip."""

    def __init__(self, lines: int) -> None:
        super().__init__()
        self.setWordWrap(True)
        self.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.setFixedHeight(lines * self.fontMetrics().lineSpacing() + 4)

    def setText(self, text: str) -> None:
        super().setText(text)
        self.setToolTip(text)


class DiagnosisPanel(QtWidgets.QWidget):
    """Summary of the deviation and the parameters that explain it, each with its correction.

    The corrections are determined jointly, so that effects that look alike (e.g. a leverarm along the direction
    of travel and a time shift, both along-track) are attributed correctly.
    """

    view_requested = QtCore.Signal(str)  # diagnostics.VIEW_*

    def __init__(self, model: DiagnosticsModel, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = _FixedLabel(SUMMARY_LINES)
        layout.addWidget(self.summary)

        self.table = QtWidgets.QTableWidget(MAX_ROWS, 4)
        for column, (title, tooltip) in enumerate(
            (
                ("Parameter", "Parameters checked with “estimate” that explain the deviation clearly"),
                (
                    "Effect",
                    "RMS of the part of the deviation that only this parameter explains (how much larger the "
                    "deviation would remain if it was not corrected)",
                ),
                ("Correction", "How much the parameter has to change, determined together with the others"),
                ("", ""),
            )
        ):
            item = QtWidgets.QTableWidgetItem(title)
            item.setToolTip(tooltip)
            self.table.setHorizontalHeaderItem(column, item)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        # compact buttons inside the rows (the global padding is meant for buttons of their own)
        self.table.setStyleSheet("QPushButton { padding: 1px 8px; }")
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        # always there, so that its appearance does not change the column widths
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setToolTip(
            "Only the parameters checked with “estimate” are considered, and only those that explain clearly more "
            "than noise would. Effect: the part of the deviation (RMS) "
            "that only this parameter accounts for, compared to the other estimated ones. Correction: its error, "
            "determined together with them. Select a row to see the pattern the parameter leaves in the "
            "deviation, below the view. Italic: weakly determined (another combination of parameters fits almost "
            "as well)."
        )
        self._buttons = []
        for row in range(MAX_ROWS):
            for column in range(3):
                self.table.setItem(row, column, QtWidgets.QTableWidgetItem())
            # the table shows its cell widgets itself: the button is hidden inside a container that stays
            cell = QtWidgets.QWidget()
            # transparent, so that the row's background (alternating, selected) shows around the button
            cell.setObjectName("buttonCell")
            cell.setStyleSheet("#buttonCell { background: transparent; }")
            cell_layout = QtWidgets.QHBoxLayout(cell)
            cell_layout.setContentsMargins(4, 3, 4, 3)
            button = QtWidgets.QPushButton("Correct")
            button.clicked.connect(lambda _=False, row=row: self._correct_row(row))
            cell_layout.addWidget(button)
            self.table.setCellWidget(row, 3, cell)
            self._buttons.append(button)
        # fixed widths (from the font, not in pixels), so that changing values never move the columns
        metrics = self.table.fontMetrics()
        value_width = max(
            metrics.horizontalAdvance(sample) for sample in ("9999.999 m", "+999.999 m", "+0.0000°", "+9999 ppm")
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, width in (
            (1, value_width + 14),
            (2, value_width + 14),
            (3, self._buttons[0].sizeHint().width() + 8),
        ):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, width)
        self.table.verticalHeader().setDefaultSectionSize(self._buttons[0].sizeHint().height() + 6)
        rows_height = ROWS * self.table.rowHeight(0)
        self.table.setFixedHeight(header.sizeHint().height() + rows_height + 2 * self.table.frameWidth())
        self.table.itemSelectionChanged.connect(self._on_selected)
        layout.addWidget(self.table)

        self.correct_all_button = QtWidgets.QPushButton("Correct all")
        self.correct_all_button.setToolTip(
            "Corrects all estimated parameters together (fast, on the preview), also those that are not listed: "
            "correcting one parameter can change the best value of a correlated one. Afterwards, nothing is left "
            "that the estimated parameters explain. For the final result with statistics, use “Refine by least "
            "squares”."
        )
        self.correct_all_button.clicked.connect(self._correct_all)
        layout.addWidget(self.correct_all_button)
        self.note = _FixedLabel(NOTE_LINES)
        layout.addWidget(self.note)

        self._listed: list[Correction] = []
        model.residuals_updated.connect(self.refresh_summary)
        model.updated.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------------------------------------ texts

    def refresh_summary(self) -> None:
        residuals = self.model.residuals
        if residuals is None:
            distance = self.model.session.preview.nearest_rms(self.model.session.state)
            self.summary.setText(
                "<b>No poses at equal time</b>: use <i>Auto align</i> or <i>Sync time</i>.<br>"
                f"Distance between the paths (regardless of time): {distance:.3f} m RMS"
            )
            return
        rms = Residuals.rms_of
        self.summary.setText(
            f"<b>Deviation at equal time</b>: RMS {residuals.rms:.3f} m, max {float(np.max(residuals.distance)):.3f} m<br>"
            f"along {rms(residuals.along):.3f} · cross {rms(residuals.cross):.3f} · "
            f"vertical {rms(residuals.vertical):.3f} m"
        )

    def refresh(self) -> None:
        self.refresh_summary()
        residuals, diagnosis = self.model.residuals, self.model.diagnosis
        if residuals is None or diagnosis is None:
            self._fill([])
            self.note.setText("")
            return

        # only what clearly exceeds noise and is relevant compared to what remains after correcting everything
        # (that level hardly changes while corrections are applied, so the list does not fill up as the deviation
        # gets smaller)
        relevant = max(MIN_EXPLAINED, RELEVANT_SHARE * diagnosis.rms_after_all)
        listed = [c for c in diagnosis.corrections if c.significant and c.explained >= relevant]
        self._fill(listed)
        explained = 1.0 - diagnosis.rms_after_all / diagnosis.rms if diagnosis.rms > 0 else 1.0
        notes = []
        if residuals.rms > FAR_FRACTION * self.model.session.preview.extent:
            notes.append("<i>Still far from the reference: place it roughly first.</i>")
        if not self.model.session.estimated_parameters:
            notes.append("No parameter is estimated: check “estimate” under <i>Values and estimated parameters</i>.")
        elif not listed:
            notes.append(
                "Nothing left that the estimated parameters explain clearly: the rest is noise or not modelled."
            )
        elif explained < UNEXPLAINED_WARNING:
            notes.append(
                f"<b>The parameters explain only {explained:.0%}</b> ({diagnosis.rms:.3f} → "
                f"{diagnosis.rms_after_all:.3f} m): the rest is not an alignment error, e.g. drift."
            )
        else:
            notes.append(f"All together: RMS {diagnosis.rms:.3f} → {diagnosis.rms_after_all:.3f} m.")
        if diagnosis.potential and diagnosis.potential[0].rms_after < (1.0 - POTENTIAL_GAIN) * diagnosis.rms_after_all:
            best = diagnosis.potential[0]
            notes.append(
                f"Not estimated: {best.name.lower()} would reduce the RMS to {best.rms_after:.3f} m "
                "(check “estimate”)."
            )
        names = diagnostics.PARAMETERS
        notes += [
            f"{names[left].name} cannot be told apart from {names[used].name.lower()} here."
            for left, used in diagnosis.inseparable
        ]
        self.note.setText(" ".join(notes))

    # ------------------------------------------------------------------------------------------------ table

    def _fill(self, corrections: list[Correction]) -> None:
        self._listed = corrections
        self.correct_all_button.setEnabled(bool(corrections))
        self.table.blockSignals(True)
        self.table.clearSelection()
        muted = self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText)
        normal = self.palette().color(QtGui.QPalette.ColorRole.Text)
        # the rows (and their buttons) always exist; those beyond the listed ones and the visible minimum are
        # hidden (changing the row count would delete the buttons of removed rows)
        for row in range(MAX_ROWS):
            self.table.setRowHidden(row, row >= max(len(corrections), ROWS))
            correction = corrections[row] if row < len(corrections) else None
            texts = (
                ("", "", "")
                if correction is None
                else (
                    correction.name,
                    f"{correction.explained:.3f} m",
                    format_correction(correction.key, correction.value),
                )
            )
            weak = correction is not None and correction.weakly_determined
            for column, text in enumerate(texts):
                item = self.table.item(row, column)
                if item is None:
                    continue
                item.setText(text)
                font = item.font()
                font.setItalic(weak)
                item.setFont(font)
                item.setForeground(muted if weak else normal)
                item.setToolTip(
                    ""
                    if correction is None
                    else f"{correction.share:.0%} of the RMS deviation"
                    + (
                        f". Weakly determined: only {correction.independence:.0%} of its effect differs from that "
                        "of other parameters."
                        if weak
                        else ""
                    )
                )
            button = self._buttons[row]
            button.setVisible(correction is not None)
            if correction is not None:
                button.setToolTip(
                    f"Corrects the {correction.name.lower()} alone: RMS {correction.rms_before:.3f} m → "
                    f"{correction.rms_after:.3f} m"
                )
        self.table.blockSignals(False)

    def row_names(self) -> list[str]:
        """Names of the listed parameters, in the order shown."""
        return [correction.name for correction in self._listed]

    def _correct_row(self, row: int) -> None:
        if row < len(self._listed):
            correction = self._listed[row]
            self.model.session.correct([correction.key], f"Correct {correction.name.lower()}")

    def _correct_all(self) -> None:
        # the listed ones first: they have priority if two parameters cannot be told apart
        listed = [correction.key for correction in self._listed]
        estimated = [key for key in self.model.session.estimated_parameters if key not in listed]
        self.model.session.correct([*listed, *estimated], "Correct all")

    def _on_selected(self) -> None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        if rows and (row := min(rows)) < len(self._listed):
            self.view_requested.emit(self._listed[row].view)
