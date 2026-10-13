"""SVG export of pyqtgraph layouts.

Qt's ``QSvgGenerator`` cannot be used directly: pyqtgraph draws some items in device coordinates and
relies on item clipping, both of which the generator does not reproduce. pyqtgraph's own
``SVGExporter`` handles this, but pyqtgraph <= 0.14.0 fails on the path data written by current Qt
versions, which closes paths with a separate ``Z`` command. This is fixed upstream (pyqtgraph master),
but not released yet; until then :func:`_tolerate_close_path_commands` makes the exporter skip and
restore these commands. The workaround is a no-op for fixed pyqtgraph versions.
"""

import functools
import importlib
from xml.dom import minidom

import pyqtgraph as pg
import pyqtgraph.exporters

# ``pyqtgraph.exporters.SVGExporter`` is shadowed by the class of the same name.
svg_exporter_module = importlib.import_module("pyqtgraph.exporters.SVGExporter")


def _close_path_positions(node: minidom.Element) -> list[tuple[minidom.Element, list[int]]]:
    """Removes standalone ``Z`` tokens from all paths and returns where they were."""
    removed = []
    for path in node.getElementsByTagName("path"):
        tokens = path.getAttribute("d").strip().split(" ")
        positions = [i for i, token in enumerate(tokens) if token == "Z"]
        if positions:
            path.setAttribute("d", " ".join(token for token in tokens if token != "Z"))
            removed.append((path, positions))
    return removed


def _restore_close_path_commands(removed: list[tuple[minidom.Element, list[int]]]) -> None:
    for path, positions in removed:
        tokens = path.getAttribute("d").strip().split(" ")
        for position in positions:
            tokens.insert(position, "Z")
        path.setAttribute("d", " ".join(tokens))


def _tolerate_close_path_commands() -> None:
    original = svg_exporter_module.correctCoordinates
    if getattr(original, "_trajectopy_patched", False):
        return

    @functools.wraps(original)
    def correct_coordinates(node, defs, item, options):
        removed = _close_path_positions(node)
        try:
            return original(node, defs, item, options)
        finally:
            _restore_close_path_commands(removed)

    correct_coordinates._trajectopy_patched = True  # type: ignore[attr-defined]
    setattr(svg_exporter_module, "correctCoordinates", correct_coordinates)  # noqa: B010


def export_svg(graphics: pg.GraphicsLayoutWidget, path: str) -> None:
    """Writes the layout of ``graphics`` to an SVG file."""
    _tolerate_close_path_commands()
    pyqtgraph.exporters.SVGExporter(graphics.ci).export(path)
