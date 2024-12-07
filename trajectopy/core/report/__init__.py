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
        report_text (str): The report text

    """
    logger.info("Writing report to %s", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)


def show_report(report_text: str, filepath: str = "") -> None:
    """
    This function writes a report to a file and opens it in the default web browser.

    Args:
        report_text (str): The report text
        filepath (str, optional): The file path to save the report. If not given, a random file name will be generated.

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
