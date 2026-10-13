"""Light and dark stylesheets of the Trajectopy GUI.

Both are filled in from one template, so the themes only differ in their colors: quiet grey backgrounds,
white (or dark) cards with thin borders and rounded corners, one blue accent for the primary action and the
selection. Widgets opt into the card and label styles with their object name (e.g. ``sectionLabel``).
"""

from trajectopy.utils.resources import resource_path

LIGHT = {
    "window": "#f6f7f9",  # background behind cards
    "surface": "#ffffff",  # cards, lists, toolbars
    "surface_alt": "#f6f8fa",  # alternating rows, headers, pressed buttons
    "border": "#d0d7de",
    "border_soft": "#eaeef2",
    "text": "#1f2328",
    "muted": "#57606a",
    "disabled": "#8c959f",
    "accent": "#1f5fbf",
    "accent_hover": "#174a94",
    "accent_text": "#ffffff",
    "selection": "#eaf1fb",
    "selection_border": "#9dbbe6",
    "hover": "#f0f3f6",
    "success": "#1a7f37",
    "warning": "#9a6700",
    "scrollbar": "#c4ccd4",
}

DARK = {
    "window": "#16191d",
    "surface": "#1f2328",
    "surface_alt": "#262b31",
    "border": "#3a4048",
    "border_soft": "#2d3238",
    "text": "#e6e8eb",
    "muted": "#9aa4ae",
    "disabled": "#5f6873",
    "accent": "#3b78d8",
    "accent_hover": "#5590ea",
    "accent_text": "#ffffff",
    "selection": "#1f3352",
    "selection_border": "#3c5f94",
    "hover": "#2a3038",
    "success": "#3fb950",
    "warning": "#d29922",
    "scrollbar": "#4a525c",
}

_TEMPLATE = """
QWidget {
    background-color: %(window)s;
    color: %(text)s;
    selection-background-color: %(selection)s;
    selection-color: %(text)s;
}
QWidget:disabled { color: %(disabled)s; }
QLabel { background: transparent; }
QToolTip {
    background-color: %(surface)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    padding: 4px 6px;
}

/* ------------------------------------------------------------------ menus, toolbar, status bar */
QMenuBar {
    background-color: %(surface)s;
    border-bottom: 1px solid %(border_soft)s;
    padding: 2px 6px;
}
QMenuBar::item { padding: 4px 10px; background: transparent; border-radius: 6px; }
QMenuBar::item:selected, QMenuBar::item:pressed { background-color: %(hover)s; }
QMenu {
    background-color: %(surface)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item { padding: 6px 24px 6px 12px; border-radius: 6px; background: transparent; }
QMenu::item:selected { background-color: %(selection)s; }
QMenu::item:disabled { color: %(disabled)s; }
QMenu::separator { height: 1px; background: %(border_soft)s; margin: 4px 6px; }
QMenu::indicator { width: 14px; height: 14px; left: 4px; }

QToolBar#mainToolbar {
    background-color: %(surface)s;
    border: none;
    border-bottom: 1px solid %(border)s;
    padding: 8px 12px;
    spacing: 8px;
}
QToolBar#mainToolbar QLabel { background: transparent; color: %(muted)s; }
QToolBar#mainToolbar QLabel#appTitle { color: %(text)s; font-weight: 600; font-size: 15px; padding-right: 10px; }
QToolBar::separator { width: 1px; background: %(border_soft)s; margin: 4px 6px; }

QStatusBar {
    background-color: %(surface)s;
    color: %(muted)s;
    border-top: 1px solid %(border)s;
}
QStatusBar::item { border: none; }
QStatusBar QLabel { background: transparent; color: %(muted)s; }

/* ------------------------------------------------------------------------------------- buttons */
QPushButton, QToolButton#outlineButton {
    background-color: %(surface)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 20px;
}
QPushButton:hover, QToolButton#outlineButton:hover { background-color: %(hover)s; }
QPushButton:pressed, QToolButton#outlineButton:pressed { background-color: %(surface_alt)s; }
QPushButton:disabled, QToolButton#outlineButton:disabled { color: %(disabled)s; border-color: %(border_soft)s; }
QPushButton:default, QPushButton#primaryButton, QToolButton#primaryButton {
    background-color: %(accent)s;
    color: %(accent_text)s;
    border: 1px solid %(accent)s;
    font-weight: 600;
}
QPushButton#primaryButton:hover, QToolButton#primaryButton:hover { background-color: %(accent_hover)s; }
QToolButton#primaryButton {
    border-radius: 6px;
    padding: 6px 30px 6px 16px;
    min-height: 20px;
}
QToolButton#primaryButton::menu-button {
    border: none;
    border-left: 1px solid rgba(255, 255, 255, 0.35);
    width: 22px;
}
QToolButton#primaryButton::menu-arrow { width: 9px; height: 9px; }
QToolButton {
    background: transparent;
    color: %(text)s;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 8px;
}
QToolButton:hover { background-color: %(hover)s; }
QToolButton:pressed { background-color: %(surface_alt)s; }
QToolButton:checked { background-color: %(selection)s; border-color: %(selection_border)s; }

/* --------------------------------------------------------------------------------------- inputs */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QDateTimeEdit, QPlainTextEdit, QTextEdit {
    background-color: %(surface)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 5px 8px;
}
QComboBox { padding-right: 24px; min-height: 20px; }
QComboBox:hover, QLineEdit:hover { border-color: %(muted)s; }
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: %(accent)s; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow { image: url(%(chevron_icon)s); width: 12px; height: 12px; }
QToolButton#outlineButton::menu-indicator { image: url(%(chevron_icon)s); subcontrol-position: right center; right: 8px; }
QToolButton#outlineButton[popupMode="2"] { padding-right: 26px; }
QComboBox QAbstractItemView {
    background-color: %(surface)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 4px;
    outline: none;
}
QCheckBox, QRadioButton { background: transparent; spacing: 8px; }
QCheckBox::indicator, QTableView::indicator, QTreeView::indicator, QMenu::indicator:non-exclusive {
    width: 14px;
    height: 14px;
    border: 1px solid %(muted)s;
    border-radius: 3px;
    background: %(surface)s;
}
QCheckBox::indicator:checked, QTableView::indicator:checked, QTreeView::indicator:checked,
QMenu::indicator:non-exclusive:checked {
    background-color: %(accent)s;
    border-color: %(accent)s;
    image: url(%(check_icon)s);
}
QRadioButton::indicator { width: 14px; height: 14px; border: 1px solid %(muted)s; border-radius: 8px; }
QRadioButton::indicator:checked { background-color: %(accent)s; border-color: %(accent)s; }
QSlider::groove:horizontal { height: 4px; background: %(border)s; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::handle:horizontal {
    background: %(surface)s;
    border: 2px solid %(accent)s;
    width: 12px;
    height: 12px;
    margin: -6px 0;
    border-radius: 8px;
}
QProgressBar {
    background-color: %(surface_alt)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    text-align: center;
}
QProgressBar::chunk { background-color: %(accent)s; border-radius: 5px; }

/* -------------------------------------------------------------------------------- lists, tables */
QTableView, QTreeView, QListView {
    background-color: %(surface)s;
    alternate-background-color: %(surface_alt)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    gridline-color: %(border_soft)s;
    outline: none;
}
QTableView::item, QTreeView::item, QListView::item { padding: 6px 8px; border: none; }
QTableView::item:hover, QTreeView::item:hover { background-color: %(hover)s; }
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected {
    background-color: %(selection)s;
}
QHeaderView { background-color: %(surface)s; border: none; }
QHeaderView::section {
    background-color: %(surface)s;
    color: %(muted)s;
    font-weight: 500;
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid %(border)s;
}
QTableCornerButton::section { background-color: %(surface)s; border: none; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; border: none; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; border: none; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: %(scrollbar)s; border-radius: 3px; }
QScrollBar::handle:vertical { min-height: 24px; }
QScrollBar::handle:horizontal { min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ------------------------------------------------------------------------------------ structure */
QSplitter::handle { background-color: %(window)s; }
QSplitter::handle:hover { background-color: %(border)s; }
QMainWindow::separator { background-color: %(window)s; width: 12px; height: 12px; }
QMainWindow::separator:hover { background-color: %(border_soft)s; }

QWidget#sidebar { background-color: %(surface)s; border-right: 1px solid %(border)s; }
QWidget#sidebar QWidget { background-color: %(surface)s; }
QWidget#sidebar QTableView, QWidget#sidebar QTreeView { border: none; }
QLabel#sectionLabel {
    background: transparent;
    color: %(muted)s;
    font-size: 11px;
    font-weight: 600;
}
QLabel#selectionTitle { background: transparent; font-size: 15px; font-weight: 600; }
QLabel#mutedLabel { background: transparent; color: %(muted)s; }
QLabel#warningLabel { background: transparent; color: %(warning)s; }
QLabel#emptyMessage { background: transparent; color: %(muted)s; }

QWidget#workspacePanel { background-color: %(window)s; }
QFrame#layoutBar {
    background-color: %(surface)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
}
QFrame#layoutBar QLabel { background: transparent; }

QDockWidget { color: %(text)s; }
QWidget#viewTitleBar {
    background-color: %(surface)s;
    border: 1px solid %(border)s;
    border-bottom: 1px solid %(border_soft)s;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
QWidget#viewTitleBar QLabel { background: transparent; }
QWidget#viewContent {
    background-color: %(surface)s;
    border: 1px solid %(border)s;
    border-top: none;
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
}
QWidget#viewContent QWidget { background-color: %(surface)s; }
QWidget#viewContent QTableView { border: none; border-radius: 0; }
QWidget#viewContent QToolBar { background-color: %(surface)s; border: none; padding: 2px 6px; spacing: 2px; }
QWidget#viewContent QComboBox { margin: 6px 8px 0 8px; }

/* ------------------------------------------------------------------ other windows (plots, dialogs) */
QTabWidget::pane { border: 1px solid %(border)s; border-radius: 6px; top: -1px; background: %(surface)s; }
QTabBar { background: transparent; }
QTabBar::tab {
    background: transparent;
    color: %(muted)s;
    padding: 7px 14px;
    border: none;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: %(text)s; border-bottom: 2px solid %(accent)s; }
QTabBar::tab:hover:!selected { color: %(text)s; }
QGroupBox {
    background-color: %(surface)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: %(muted)s;
    font-weight: 600;
}
"""


def _resource(name: str) -> str:
    return resource_path(f"gui/resources/{name}").replace("\\", "/")


def _stylesheet(colors: dict[str, str], variant: str) -> str:
    return _TEMPLATE % {
        **colors,
        "check_icon": _resource("check.svg"),
        "chevron_icon": _resource(f"chevron-{variant}.svg"),
    }


LIGHT_STYLESHEET = _stylesheet(LIGHT, "light")
DARK_STYLESHEET = _stylesheet(DARK, "dark")

_dark = False


def set_dark(dark: bool) -> None:
    """Remembers the theme for :func:`color` (the stylesheet itself is applied by the caller)."""
    global _dark
    _dark = dark


def color(name: str) -> str:
    """A color of the current theme, for widgets that paint or color items themselves."""
    return (DARK if _dark else LIGHT)[name]
