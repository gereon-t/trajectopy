"""Compact timeline strip showing trajectory timespans."""

from __future__ import annotations

from datetime import datetime, timezone

from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.gui.models.entries import TrajectoryEntry

_BAR_COLORS = [
    "#4e8fd4",  # blue
    "#e07b54",  # orange
    "#57a773",  # green
    "#b05cc2",  # purple
    "#c2a22a",  # yellow-gold
    "#c25c7a",  # rose
    "#3aabb0",  # teal
    "#8c7a6b",  # warm grey
    "#6e9dc2",  # steel blue
    "#a0c25c",  # lime green
    "#c27b3a",  # amber
    "#7a6ec2",  # indigo
]

_PAD = 4  # horizontal padding inside the bar area


class TimelineWidget(QtWidgets.QWidget):
    """Compact horizontal timeline strip: one thin bar per trajectory."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[TrajectoryEntry] = []
        self._dark = True
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self._update_height()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_entries(self, entries: list[TrajectoryEntry]) -> None:
        self._entries = entries
        self._update_height()
        self.update()

    def apply_theme(self, dark: bool) -> None:
        self._dark = dark
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_metrics(self) -> tuple[int, int, int, int]:
        """Return (row_h, row_gap, tick_h, label_w) derived from the app font."""
        fm = QtGui.QFontMetrics(QtWidgets.QApplication.font())
        line_h = fm.height()
        row_h = int(line_h * 0.85)
        row_gap = max(4, line_h // 4)
        tick_h = line_h + 4
        label_w = fm.horizontalAdvance("W" * 12)  # ~12 chars wide
        return row_h, row_gap, tick_h, label_w

    def _update_height(self) -> None:
        n = max(len(self._entries), 0)
        row_h, row_gap, tick_h, _ = self._row_metrics()
        h = n * (row_h + row_gap) + tick_h if n > 0 else 0
        self.setFixedHeight(h)

    def _text_color(self) -> QtGui.QColor:
        return QtGui.QColor("#aaaaaa" if self._dark else "#555555")

    def _overlap_color(self) -> QtGui.QColor:
        c = QtGui.QColor("#ffffff" if self._dark else "#000000")
        c.setAlphaF(0.20)
        return c

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        entries = self._entries
        if not entries:
            return

        row_h, row_gap, tick_h, label_w = self._row_metrics()
        fm = QtGui.QFontMetrics(QtWidgets.QApplication.font())

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setFont(QtWidgets.QApplication.font())

        w = self.width()
        spans = [(float(e.trajectory.timestamps.min()), float(e.trajectory.timestamps.max())) for e in entries]
        t_min = min(s[0] for s in spans)
        t_max = max(s[1] for s in spans)
        t_range = t_max - t_min or 1.0

        bar_x0 = label_w
        bar_w = w - label_w - _PAD

        use_dates = t_min > 1_000_000

        # --- Draw bars and name labels ---
        for i, (entry, (ts, te)) in enumerate(zip(entries, spans)):
            y = i * (row_h + row_gap)
            color = QtGui.QColor(_BAR_COLORS[i % len(_BAR_COLORS)])

            x0 = bar_x0 + _PAD + int((ts - t_min) / t_range * (bar_w - 2 * _PAD))
            x1 = bar_x0 + _PAD + int((te - t_min) / t_range * (bar_w - 2 * _PAD))
            bar_rect = QtCore.QRect(x0, y, max(x1 - x0, 2), row_h)

            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(bar_rect, 2, 2)

            # Name label
            name = fm.elidedText(entry.trajectory.name, QtCore.Qt.TextElideMode.ElideRight, label_w - 6)
            painter.setPen(self._text_color())
            painter.drawText(
                QtCore.QRect(0, y, label_w - 6, row_h),
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                name,
            )

        # --- Overlap highlights ---
        # Merge all pairwise overlaps into disjoint intervals so each
        # pixel is painted at most once (avoids alpha accumulation).
        n = len(spans)
        raw_overlaps: list[tuple[float, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                ol_s = max(spans[i][0], spans[j][0])
                ol_e = min(spans[i][1], spans[j][1])
                if ol_e > ol_s:
                    raw_overlaps.append((ol_s, ol_e))

        if raw_overlaps:
            raw_overlaps.sort()
            merged: list[tuple[float, float]] = [raw_overlaps[0]]
            for s, e in raw_overlaps[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))

            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(self._overlap_color())
            total_bar_h = n * (row_h + row_gap)
            for ol_s, ol_e in merged:
                x0 = bar_x0 + _PAD + int((ol_s - t_min) / t_range * (bar_w - 2 * _PAD))
                x1 = bar_x0 + _PAD + int((ol_e - t_min) / t_range * (bar_w - 2 * _PAD))
                painter.drawRect(QtCore.QRect(x0, 0, max(x1 - x0, 2), total_bar_h))

        # --- X-axis tick labels (start / end of total range) ---
        tick_y = n * (row_h + row_gap)
        painter.setPen(self._text_color())
        if use_dates:
            fmt = "%H:%M:%S" if t_range < 3600 else "%H:%M"
            label_start = datetime.fromtimestamp(t_min, tz=timezone.utc).strftime(fmt)
            label_end = datetime.fromtimestamp(t_max, tz=timezone.utc).strftime(fmt)
        else:
            label_start = f"{t_min:.1f} s"
            label_end = f"{t_max:.1f} s"

        tick_label_w = fm.horizontalAdvance(label_end) + 8
        painter.drawText(
            QtCore.QRect(bar_x0 + _PAD, tick_y, tick_label_w, tick_h),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            label_start,
        )
        painter.drawText(
            QtCore.QRect(w - tick_label_w, tick_y, tick_label_w, tick_h),
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
            label_end,
        )

        painter.end()


class TimelineDialog(QtWidgets.QDialog):
    """Floating dialog that hosts the TimelineWidget."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Trajectory Timeline")
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.WindowCloseButtonHint
            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(600, 160)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.timeline = TimelineWidget(parent=self)
        layout.addWidget(self.timeline)

    def set_entries(self, entries: list[TrajectoryEntry]) -> None:
        self.timeline.set_entries(entries)
        # Resize height to fit content, using font-derived metrics
        if entries:
            content_h = self.timeline.height() + 32
            self.resize(self.width(), max(content_h, 80))

    def apply_theme(self, dark: bool) -> None:
        self.timeline.apply_theme(dark)
