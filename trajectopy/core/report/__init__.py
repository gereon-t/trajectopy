import os
import uuid
import webbrowser
import logging

logger = logging.getLogger("root")


def write_report(*, output_file: str, report_text: str) -> None:
    """
    Writes a report to the given output file.

    Args:

        output_file (str): The output file path

    """
    logger.info("Writing report to %s", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)


def show_report(report_text: str, filepath: str = "") -> None:
    """
    Shows a report in the browser.

    Args:

        report_text (str): The report string

    """
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    random_string = uuid.uuid4().hex

    file = filepath or os.path.join(dirname, f"{random_string}.html")

    with open(file, "w", encoding="utf-8") as f:
        f.write(report_text)
        url = "file://" + os.path.realpath(f.name)
        webbrowser.open(url)
