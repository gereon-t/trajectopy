"""Properties dock content: the properties of the selected trajectories or results."""

import logging
from collections.abc import Sequence

from PySide6 import QtWidgets

from trajectopy.gui.models.entries import Entry

logger = logging.getLogger(__name__)


class PropertiesPanel(QtWidgets.QTreeWidget):
    """Property / value table; with several selected entries, one collapsible group per entry."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["Property", "Value"])
        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        if (header := self.header()) is not None:
            header.setStretchLastSection(True)
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

    def show_entries(self, entries: Sequence[Entry]) -> None:
        self.clear()
        grouped = len(entries) > 1
        self.setRootIsDecorated(grouped)
        for entry in entries:
            parent = QtWidgets.QTreeWidgetItem(self, [_name(entry), ""]) if grouped else self.invisibleRootItem()
            for key, value in self._properties(entry).items():
                item = QtWidgets.QTreeWidgetItem(parent, [key.rstrip(":"), str(value)])
                item.setToolTip(1, str(value))
            if grouped:
                parent.setFirstColumnSpanned(True)
                parent.setExpanded(True)

    @staticmethod
    def _properties(entry: Entry) -> dict[str, str]:
        try:
            return entry.property_dict
        except Exception as error:
            logger.exception("Could not compute the properties of '%s'", _name(entry))
            return {"Name": _name(entry), "Error": str(error)}


def _name(entry: Entry) -> str:
    return getattr(entry, "name", entry.type)
