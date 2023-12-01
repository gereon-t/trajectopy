import json
import sys

from PyQt6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QTextDocument
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# hex codes
dark_blue = "#1E1E2C"
dark_green = "#1E2C1E"
dark_red = "#2C1E1E"
dark_magenta = "#2C1E2C"
dark_cyan = "#1E2C2C"


class JsonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent):
        super().__init__(parent)
        self.format = QTextCharFormat()
        self.format.setForeground(QColor(dark_blue))
        self.format.setFontWeight(QFont.Weight.Bold)

        self.json_format = QTextCharFormat()
        self.json_format.setForeground(QColor(dark_green))
        self.json_format.setFontWeight(QFont.Weight.Bold)

        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor(dark_red))
        self.string_format.setFontWeight(QFont.Weight.Bold)

        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor(dark_magenta))
        self.number_format.setFontWeight(QFont.Weight.Bold)

        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor(dark_cyan))
        self.keyword_format.setFontWeight(QFont.Weight.Bold)

        self.value_formats = {
            str: self.string_format,
            int: self.number_format,
            float: self.number_format,
            bool: self.keyword_format,
            type(None): self.keyword_format,
        }

    def highlightBlock(self, text):
        stack = []
        for i, char in enumerate(text):
            if char == "{":
                stack.append(i)
            elif char == "}":
                if stack:
                    start = stack.pop()
                    length = i - start + 1
                    self.setFormat(start, length, self.json_format)
            elif '"' in char:
                self.setFormat(i, 1, self.string_format)
                if stack and text[stack[-1] : i] != '\\"':
                    stack.pop()
            elif str(char).isdigit():
                self.setFormat(i, 1, self.number_format)
            elif text[i : i + 4] in ["true", "false", "null"]:
                self.setFormat(i, 4, self.keyword_format)

        for start, length in zip(stack, [len(text) - s for s in stack]):
            self.setFormat(start, length, self.json_format)


class JsonViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("JSON Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.hlayout = QHBoxLayout(self.central_widget)
        self.layout.addLayout(self.hlayout)

        self.json_text_edit = QTextEdit(self)
        self.json_text_edit.setTabStopDistance(20)
        self.layout.addWidget(self.json_text_edit)

        self.json_highlighter = JsonHighlighter(self.json_text_edit.document())

        self.apply_button = QPushButton("Apply", self)
        self.apply_button.clicked.connect(self.apply_changes)
        self.hlayout.addWidget(self.apply_button)

    def load_json_file(self, filename):
        try:
            with open(filename, "r", encoding="utf-8") as file:
                json_content = file.read()
                self.json_text_edit.setPlainText(json_content)
        except Exception as e:
            print(f"Error loading JSON file: {e}")

    def apply_changes(self):
        try:
            json_content = self.json_text_edit.toPlainText()
            parsed_json = json.loads(json_content)
            print(parsed_json)
            # You can use the parsed JSON for further processing or saving to a file.
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = JsonViewer()

    file_dialog = QFileDialog(viewer)
    file_dialog.setNameFilter("JSON Files (*.json)")

    def open_file_dialog():
        filename, _ = file_dialog.getOpenFileName()
        if filename:
            viewer.load_json_file(filename)

    open_button = QPushButton("Open JSON File", viewer)
    open_button.clicked.connect(open_file_dialog)
    viewer.layout.addWidget(open_button)

    viewer.show()
    sys.exit(app.exec())
