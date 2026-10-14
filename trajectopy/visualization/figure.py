"""Figures, the plot window and export.

A :class:`Figure` is a titled page of plots. It only stores how to draw itself, so lists of figures are
cheap to create; the plots are rendered when a page is first shown or exported. A figure may offer
several *variants* (for example the quantity used to color a trajectory), which the plot window offers
in a drop-down and :func:`export_figures` writes to separate files.

The same :class:`PlotWindow` is used by the GUI and by :func:`show` in scripts.
"""

import logging
import os
import re
from collections.abc import Callable, Sequence

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from trajectopy.visualization import style
from trajectopy.visualization._svg_export import export_svg

logger = logging.getLogger(__name__)

RenderFunction = Callable[[pg.GraphicsLayout], None]

# Plots are laid out at this logical size when exported and then scaled to the requested width,
# so that fonts and line widths keep their proportions independent of the export resolution.
EXPORT_LOGICAL_SIZE = QtCore.QSizeF(1000.0, 625.0)


def ensure_app() -> QtWidgets.QApplication:
    """Returns the running Qt application, creating one if necessary (required before creating plots)."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = pg.mkQApp("Trajectopy")
    return app  # type: ignore[return-value]


class Figure:
    """A titled page of plots.

    Args:
        title: Title of the page (tab title in the plot window and file name stem on export).
        render: Draws the plots into the given layout.
        variants: Alternatively, several named render functions; the user picks one in a drop-down.
    """

    def __init__(
        self,
        title: str,
        render: RenderFunction | None = None,
        variants: dict[str, RenderFunction] | None = None,
    ) -> None:
        if (render is None) == (variants is None):
            raise ValueError("Provide either render or variants.")
        self.title = title
        self.variants: dict[str, RenderFunction] = variants if variants is not None else {"": render}  # type: ignore[dict-item]
        if not self.variants:
            raise ValueError("A figure needs at least one variant.")

    @property
    def has_variants(self) -> bool:
        return len(self.variants) > 1 or "" not in self.variants

    def create_widget(self) -> QtWidgets.QWidget:
        """Creates an interactive widget showing the figure (with a variant selector if needed)."""
        return _FigureWidget(self)

    def export(self, path: str, variant: str | None = None, width: int = 1600) -> None:
        """Exports the figure (or one of its variants) to a PNG, JPG or SVG file.

        Args:
            path: Output file; the format is derived from the extension.
            variant: Name of the variant to export. Defaults to the first one.
            width: Width in pixels of raster images.
        """
        app = ensure_app()
        render = self.variants[variant if variant is not None else next(iter(self.variants))]
        graphics = pg.GraphicsLayoutWidget()
        render(graphics.ci)
        _settle_layout(app, graphics)
        _render_to_file(graphics, path, width)


def _settle_layout(app: QtWidgets.QApplication, graphics: pg.GraphicsLayoutWidget) -> None:
    """Brings an unshown layout into its final state.

    Axes only learn how much space their tick labels need while they are painted, and the resulting
    layout requests are processed by the event loop. Since export widgets are never shown, paint twice
    into a scratch image and process the requests in between.
    """
    scratch = QtGui.QImage(16, 16, QtGui.QImage.Format.Format_ARGB32)
    for _ in range(2):
        app.processEvents()
        graphics.ci.setGeometry(QtCore.QRectF(QtCore.QPointF(0, 0), EXPORT_LOGICAL_SIZE))
        graphics.ci.layout.activate()
        painter = QtGui.QPainter(scratch)
        graphics.scene().render(painter, QtCore.QRectF(scratch.rect()), graphics.ci.sceneBoundingRect())
        painter.end()
    app.processEvents()
    graphics.ci.layout.activate()


def _render_to_file(graphics: pg.GraphicsLayoutWidget, path: str, width: int) -> None:
    if path.lower().endswith(".svg"):
        export_svg(graphics, path)
        logger.info("Exported plot to %s", path)
        return

    source = graphics.ci.sceneBoundingRect()
    height = round(width * source.height() / source.width())
    image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtGui.QColor(style.theme().background))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    graphics.scene().render(painter, QtCore.QRectF(0, 0, width, height), source)
    painter.end()

    if not image.save(path):
        raise OSError(f"Could not write image to {path}")
    logger.info("Exported plot to %s", path)


class _FigureWidget(QtWidgets.QWidget):
    """Interactive view of a figure: the plots and, if the figure has variants, a selector."""

    def __init__(self, figure: Figure, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = figure
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Parent it right away: a parentless widget made visible becomes a top-level window of its own
        self.selector = QtWidgets.QComboBox(self)
        self.selector.addItems(list(figure.variants))
        self.selector.setVisible(figure.has_variants)
        self.selector.currentTextChanged.connect(self._render)
        layout.addWidget(self.selector)

        self.graphics = pg.GraphicsLayoutWidget(self)
        layout.addWidget(self.graphics)
        self._render(self.selector.currentText())

    @property
    def current_variant(self) -> str:
        return self.selector.currentText()

    def _render(self, variant: str) -> None:
        self.graphics.clear()
        self.figure.variants[variant](self.graphics.ci)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name).strip("_") or "plot"


class PlotWindow(QtWidgets.QMainWindow):
    """Window showing figures as tabs. Tabs are rendered when they are first opened."""

    def __init__(
        self,
        figures: Sequence[Figure],
        title: str = "Trajectopy",
        export_format: str = "png",
        export_width: int = 1600,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.figures = list(figures)
        self.export_format = export_format
        self.export_width = export_width

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        for figure in self.figures:
            placeholder = QtWidgets.QWidget()
            QtWidgets.QVBoxLayout(placeholder).setContentsMargins(0, 0, 0, 0)
            self.tabs.addTab(placeholder, figure.title)
        self.tabs.currentChanged.connect(self._ensure_rendered)
        self.setCentralWidget(self.tabs)

        toolbar = self.addToolBar("Export")
        toolbar.setMovable(False)
        toolbar.addAction("Export Plot…", self.export_current)
        toolbar.addAction("Export All…", self.export_all)

        if (screen := QtGui.QGuiApplication.primaryScreen()) is not None:
            available = screen.availableGeometry()
            self.resize(int(available.width() * 0.7), int(available.height() * 0.75))

        self._ensure_rendered(0)

    def figure_widget(self, index: int) -> _FigureWidget | None:
        container = self.tabs.widget(index)
        if container is None or (layout := container.layout()) is None or layout.count() == 0:
            return None
        widget = layout.itemAt(0).widget()  # type: ignore[union-attr]
        return widget if isinstance(widget, _FigureWidget) else None

    def _ensure_rendered(self, index: int) -> None:
        if index < 0 or index >= len(self.figures) or self.figure_widget(index) is not None:
            return
        container = self.tabs.widget(index)
        container.layout().addWidget(self.figures[index].create_widget())  # type: ignore[union-attr]

    def export_current(self) -> None:
        index = self.tabs.currentIndex()
        if index < 0:
            return
        figure = self.figures[index]
        widget = self.figure_widget(index)
        variant = widget.current_variant if widget is not None else None
        stem = _safe_filename(f"{figure.title} {variant}" if variant else figure.title)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Plot", f"{stem}.{self.export_format}", "Images (*.png *.jpg *.svg)"
        )
        if path:
            figure.export(path, variant=variant, width=self.export_width)

    def export_all(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Export All Plots")
        if directory:
            export_figures(self.figures, directory, export_format=self.export_format, width=self.export_width)


def export_figures(
    figures: Sequence[Figure], directory: str, export_format: str = "png", width: int = 1600
) -> list[str]:
    """Exports all figures (and all their variants) to image files in ``directory``.

    Returns:
        list[str]: Paths of the written files.
    """
    os.makedirs(directory, exist_ok=True)
    paths = []
    for index, figure in enumerate(figures, start=1):
        for variant in figure.variants:
            stem = _safe_filename(f"{index:02d}_{figure.title}" + (f"_{variant}" if variant else ""))
            path = os.path.join(directory, f"{stem}.{export_format}")
            figure.export(path, variant=variant, width=width)
            paths.append(path)
    return paths


def show(figures: Figure | Sequence[Figure], title: str = "Trajectopy", block: bool = True) -> PlotWindow:
    """Shows figures in a plot window.

    Args:
        figures: Figure(s) to show, one tab each.
        title: Window title.
        block: Run the Qt event loop until the window is closed (for scripts). Use ``False`` if an
            event loop is already running, e.g. inside a Qt application.

    Returns:
        PlotWindow: The window.
    """
    app = ensure_app()
    window = PlotWindow([figures] if isinstance(figures, Figure) else figures, title=title)
    window.show()
    if block:
        app.exec()
    return window
