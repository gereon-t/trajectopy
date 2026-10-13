"""Helpers for self-explanatory context menus.

Every entry gets a hint (shown on hover and in the status bar) and, if it is not available for the current
selection, stays visible but disabled with the reason as hint. Keyboard shortcuts are attached to the menu
actions, so the menu shows them and the table views trigger the very same actions (see ``trigger_shortcut``).
"""

from collections.abc import Callable, Iterator
from typing import Any

from PySide6 import QtGui, QtWidgets
from PySide6.QtGui import QAction, QKeySequence


def create_menu(title: str, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
    menu = QtWidgets.QMenu(title, parent)
    menu.setToolTipsVisible(True)
    return menu


def add_action(
    menu: QtWidgets.QMenu,
    text: str,
    slot: Callable[[], object],
    *,
    hint: str = "",
    shortcuts: tuple[str, ...] = (),
    unavailable: str = "",
) -> QAction:
    """Adds an action to the menu.

    Args:
        menu: The menu.
        text: Label of the entry.
        slot: Called when the entry is triggered.
        hint: What the entry does.
        shortcuts: Key sequences; the first one is shown in the menu.
        unavailable: If not empty, the entry is disabled and this reason is shown as hint.
    """
    action = QAction(text, menu)
    action.triggered.connect(lambda: slot())
    if shortcuts:
        action.setShortcuts([QKeySequence(shortcut) for shortcut in shortcuts])
    tip = f"{unavailable}\n\n{hint}" if unavailable and hint else unavailable or hint
    action.setToolTip(tip)
    action.setStatusTip(tip.replace("\n\n", " "))
    action.setEnabled(not unavailable)
    menu.addAction(action)
    return action


def add_submenu(menu: QtWidgets.QMenu, title: str, *, hint: str = "") -> QtWidgets.QMenu:
    submenu = create_menu(title, menu)
    submenu.menuAction().setToolTip(hint)
    submenu.menuAction().setStatusTip(hint)
    menu.addMenu(submenu)
    return submenu


def add_note(menu: QtWidgets.QMenu, text: str) -> None:
    """Adds a disabled, informative line (e.g. why the following entries are not available)."""
    note = QAction(f"ⓘ {text}", menu)
    note.setEnabled(False)
    menu.addAction(note)


def iter_actions(menu: QtWidgets.QMenu) -> Iterator[QAction]:
    """All entries of the menu and its submenus (without separators)."""
    for action in menu.actions():
        if (submenu := action.menu()) is not None:
            yield from iter_actions(submenu)  # type: ignore[arg-type]
        elif not action.isSeparator():
            yield action


def trigger_shortcut(menu: QtWidgets.QMenu, event: QtGui.QKeyEvent) -> bool:
    """Triggers the enabled menu action whose shortcut matches the key event.

    Returns:
        bool: True if an action was triggered.
    """
    pressed = QKeySequence(event.keyCombination())
    for action in iter_actions(menu):
        if not action.isEnabled():
            continue
        if any(shortcut.matches(pressed) == QKeySequence.SequenceMatch.ExactMatch for shortcut in action.shortcuts()):
            action.trigger()
            return True
    return False


class SelectionMenu(QtWidgets.QMenu):
    """Context menu that is (re)built for the current selection of a table."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTipsVisible(True)
        self._selection: Any = None

    def get_selection(self) -> Any:
        return self._selection

    def populate(self, selection: Any) -> None:
        """(Re)builds the menu for the given selection."""
        self.clear()
        self._selection = selection
        self._add_entries()

    def _add_entries(self) -> None:
        raise NotImplementedError

    def show_context_menu(self, selection: Any) -> None:
        if not selection:
            return
        self.populate(selection)
        self.exec(QtGui.QCursor.pos())

    def handle_key_press(self, selection: Any, event: QtGui.QKeyEvent) -> bool:
        """Triggers the menu entry with the pressed shortcut. Returns True if the key was handled."""
        if not selection:
            return False
        self.populate(selection)
        return trigger_shortcut(self, event)
