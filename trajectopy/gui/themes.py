"""VS Code-inspired stylesheets for the Trajectopy GUI."""

_BASE = """
        QToolBar {
            background-color: %(chrome)s;
            border-bottom: 1px solid %(border)s;
            padding: 2px 4px;
            spacing: 4px;
        }
        QToolBar::separator {
            width: 1px;
            background: %(border)s;
            margin: 4px 2px;
        }
        QToolButton {
            background: transparent;
            color: %(text)s;
            border: none;
            padding: 4px 8px;
            border-radius: 3px;
        }
        QToolButton:hover {
            background-color: %(hover)s;
        }
        QToolButton:pressed {
            background-color: %(selection)s;
        }
        QGroupBox {
            border: 1px solid %(border)s;
            border-radius: 3px;
            margin-top: 10px;
            padding-top: 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 4px;
            left: 8px;
            color: %(muted)s;
        }
        QHeaderView {
            background-color: %(chrome)s;
            border: none;
        }
        QHeaderView::section {
            background-color: %(chrome)s;
            color: %(muted)s;
            padding: 4px 6px;
            border: none;
            border-right: 1px solid %(border)s;
            border-bottom: 1px solid %(border)s;
        }
        QHeaderView::section:hover {
            background-color: %(header_hover)s;
        }
        QScrollBar:vertical {
            background: %(bg)s;
            width: 10px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: %(scrollbar)s;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: %(scrollbar_hover)s;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        QScrollBar:horizontal {
            background: %(bg)s;
            height: 10px;
            margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: %(scrollbar)s;
            min-width: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal:hover {
            background: %(scrollbar_hover)s;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0;
        }
        QSplitter::handle {
            background-color: %(border)s;
        }
        QSplitter::handle:horizontal {
            width: 1px;
        }
        QSplitter::handle:vertical {
            height: 1px;
        }
        QCheckBox {
            color: %(text)s;
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 13px;
            height: 13px;
            border: 1px solid %(muted)s;
            border-radius: 2px;
            background: %(input_bg)s;
        }
        QCheckBox::indicator:checked {
            background-color: %(accent)s;
            border-color: %(accent)s;
        }
        QComboBox {
            background-color: %(input_bg)s;
            color: %(text)s;
            border: 1px solid %(input_border)s;
            padding: 4px 8px;
            border-radius: 2px;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox QAbstractItemView {
            background-color: %(menu_bg)s;
            color: %(text)s;
            border: 1px solid %(border)s;
            selection-background-color: %(selection)s;
            selection-color: #ffffff;
        }
        QLineEdit {
            background-color: %(input_bg)s;
            color: %(text)s;
            border: 1px solid %(input_border)s;
            padding: 4px 6px;
            border-radius: 2px;
        }
        QLineEdit:focus {
            border-color: %(accent)s;
        }
        QSizeGrip {
            width: 0;
            height: 0;
        }
        QTabBar {
            background-color: %(chrome)s;
        }
        QTabBar::tab {
            background-color: %(chrome)s;
            color: %(muted)s;
            padding: 6px 12px;
            border: none;
            border-right: 1px solid %(border)s;
        }
        QTabBar::tab:selected {
            background-color: %(bg)s;
            color: %(text)s;
            border-top: 2px solid %(accent)s;
        }
        QTabBar::tab:hover:!selected {
            background-color: %(hover)s;
            color: %(text)s;
        }
        QTabWidget::pane {
            border: 1px solid %(border)s;
        }
"""

DARK_STYLESHEET = """
        QMainWindow, QDialog, QWidget {
            background-color: #1e1e1e;
            color: #cccccc;
        }
        QMenuBar {
            background-color: #2d2d30;
            color: #cccccc;
            border-bottom: 1px solid #3c3c3c;
            padding: 2px 0;
        }
        QMenuBar::item {
            padding: 4px 10px;
            background: transparent;
        }
        QMenuBar::item:selected, QMenuBar::item:pressed {
            background-color: #094771;
        }
        QMenu {
            background-color: #252526;
            color: #cccccc;
            border: 1px solid #3c3c3c;
        }
        QMenu::item {
            padding: 5px 20px 5px 20px;
        }
        QMenu::item:selected {
            background-color: #094771;
        }
        QMenu::separator {
            height: 1px;
            background: #3c3c3c;
            margin: 3px 0;
        }
        QStatusBar {
            background-color: #2d2d30;
            color: #bbbbbb;
            border-top: 1px solid #3c3c3c;
        }
        QStatusBar::item {
            border: none;
        }
        QPushButton {
            background-color: #0e639c;
            color: #ffffff;
            border: none;
            padding: 3px 16px;
            border-radius: 2px;
        }
        QPushButton:hover {
            background-color: #1177bb;
        }
        QPushButton:pressed {
            background-color: #094771;
        }
        QPushButton:disabled {
            background-color: #3c3c3c;
            color: #666666;
        }
        QTableView {
            background-color: #1e1e1e;
            alternate-background-color: #252526;
            color: #cccccc;
            gridline-color: #2d2d30;
            border: 1px solid #3c3c3c;
            selection-background-color: #094771;
            selection-color: #ffffff;
            outline: none;
        }
        QTableView::item:selected {
            background-color: #094771;
            color: #ffffff;
        }
        QTextEdit, QPlainTextEdit {
            background-color: #1e1e1e;
            color: #cccccc;
            border: 1px solid #3c3c3c;
            selection-background-color: #264f78;
            font-family: "Consolas", "Menlo", "DejaVu Sans Mono", "Liberation Mono", "Courier New", monospace;
        }
        QLabel {
            color: #cccccc;
            background: transparent;
        }
        QToolTip {
            background-color: #252526;
            color: #cccccc;
            border: 1px solid #454545;
            padding: 4px;
        }
        QDialog QLabel {
            color: #cccccc;
        }
    """ + _BASE % dict(
    bg="#1e1e1e",
    chrome="#2d2d30",
    border="#3c3c3c",
    text="#cccccc",
    muted="#888888",
    accent="#007acc",
    selection="#094771",
    hover="#3e3e42",
    header_hover="#3e3e42",
    menu_bg="#252526",
    input_bg="#3c3c3c",
    input_border="#555555",
    scrollbar="#424242",
    scrollbar_hover="#686868",
)

LIGHT_STYLESHEET = """
        QMainWindow, QDialog, QWidget {
            background-color: #f3f3f3;
            color: #333333;
        }
        QMenuBar {
            background-color: #dddddd;
            color: #333333;
            border-bottom: 1px solid #cccccc;
            padding: 2px 0;
        }
        QMenuBar::item {
            padding: 4px 10px;
            background: transparent;
        }
        QMenuBar::item:selected, QMenuBar::item:pressed {
            background-color: #0078d4;
            color: #ffffff;
        }
        QMenu {
            background-color: #f3f3f3;
            color: #333333;
            border: 1px solid #cccccc;
        }
        QMenu::item {
            padding: 5px 20px 5px 20px;
        }
        QMenu::item:selected {
            background-color: #0078d4;
            color: #ffffff;
        }
        QMenu::separator {
            height: 1px;
            background: #e0e0e0;
            margin: 3px 0;
        }
        QStatusBar {
            background-color: #f3f3f3;
            color: #666666;
            border-top: 1px solid #e0e0e0;
        }
        QStatusBar::item {
            border: none;
        }
        QPushButton {
            background-color: #0078d4;
            color: #ffffff;
            border: none;
            padding: 3px 16px;
            border-radius: 2px;
        }
        QPushButton:hover {
            background-color: #106ebe;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #888888;
        }
        QTableView {
            background-color: #ffffff;
            alternate-background-color: #f8f8f8;
            color: #333333;
            gridline-color: #f0f0f0;
            border: 1px solid #e0e0e0;
            selection-background-color: #0078d4;
            selection-color: #ffffff;
            outline: none;
        }
        QTableView::item:selected {
            background-color: #0078d4;
            color: #ffffff;
        }
        QTextEdit, QPlainTextEdit {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #e0e0e0;
            selection-background-color: #add6ff;
            font-family: "Consolas", "Menlo", "DejaVu Sans Mono", "Liberation Mono", "Courier New", monospace;
        }
        QLabel {
            color: #333333;
            background: transparent;
        }
        QToolTip {
            background-color: #fffce8;
            color: #333333;
            border: 1px solid #cccccc;
            padding: 4px;
        }
        QDialog QLabel {
            color: #333333;
        }
    """ + _BASE % dict(
    bg="#f3f3f3",
    chrome="#e8e8e8",
    border="#e0e0e0",
    text="#333333",
    muted="#666666",
    accent="#0078d4",
    selection="#0078d4",
    hover="#dce9f5",
    header_hover="#e0e0e0",
    menu_bg="#f3f3f3",
    input_bg="#ffffff",
    input_border="#cccccc",
    scrollbar="#c1c1c1",
    scrollbar_hover="#a8a8a8",
)

# Matplotlib preview colours keyed by theme
MPL_COLORS = {
    "dark": dict(bg="#1e1e1e", spine="#3c3c3c", tick="#888888"),
    "light": dict(bg="#ffffff", spine="#e0e0e0", tick="#888888"),
}
