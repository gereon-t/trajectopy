"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import logging
import os
import uuid
import webbrowser
from typing import List, Union

import jinja2

from trajectopy.core.alignment.parameters import AlignmentParameters
from trajectopy.core.evaluation.ate_result import ATEResult
from trajectopy.core.evaluation.rpe_result import RPEResult
from trajectopy.core.report.alignment import render_one_line_alignment_plots
from trajectopy.core.report.multi import render_multi_deviations_report
from trajectopy.core.report.single import render_single_deviations_report
from trajectopy.core.report.trajectory import render_one_line_trajectory_plots
from trajectopy.core.report.utils import TEMPLATES_PATH, convert_icon_to_base64
from trajectopy.settings import ReportSettings
from trajectopy.trajectory import Trajectory

logger = logging.getLogger("root")


def create_alignment_report(
    *,
    alignment_parameters: AlignmentParameters,
    name: str = "Alignment",
    report_settings: ReportSettings = ReportSettings(),
) -> str:
    """
    Render trajectory alignment heatmaps.

    Heatmaps are covariance matrix and correlation matrix.

    Args:
        alignment_parameters (AlignmentParameters): Estimated alignment parameters.
        name (str, optional): Name of the report. Defaults to "Alignment".
        report_settings (ReportSettings, optional): Report settings. Defaults to ReportSettings()

    Returns:
        HTML string of the rendered report including the heatmap plots.
    """

    template = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_PATH)).get_template("generic.html")
    icon = convert_icon_to_base64()

    one_line_plots = render_one_line_alignment_plots(alignment_parameters, report_settings)

    context = {
        "title": name,
        "one_line_plots": one_line_plots,
        "icon": icon,
    }

    return template.render(context)


def create_trajectory_report(
    *, trajectories: List[Trajectory], report_settings: ReportSettings = ReportSettings()
) -> str:
    """
    Render a HTML report containing trajectory plots.

    Args:
        trajectories: List of trajectories to render.
        report_settings: Report settings.

    Returns:
        HTML string of the rendered report including the trajectory plots.
    """

    template = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_PATH)).get_template("generic.html")
    icon = convert_icon_to_base64()

    one_line_plots = render_one_line_trajectory_plots(trajectories, report_settings)

    context = {
        "title": trajectories[0].name if len(trajectories) == 1 else "Trajectory Plot",
        "one_line_plots": one_line_plots,
        "icon": icon,
    }

    return template.render(context)


def create_deviation_report(
    ate_result: Union[ATEResult, List[ATEResult]],
    rpe_result: Union[RPEResult, List[RPEResult], None],
    report_settings: ReportSettings = ReportSettings(),
):
    """
    Create a HTML deviation report containing the absolute trajectory error and relative pose error results.

    Args:
        ate_result (Union[ATEResult, list[ATEResult]]): The absolute trajectory error results (one or multiple)
        rpe_result (Union[RPEResult, list[RPEResult]]): The relative pose error results (one or multiple)
        report_settings (ReportSettings): The report settings, defaults to ReportSettings()

    Returns:
        str: The deviation report
    """
    if (ate_result is not None and isinstance(ate_result, list)) or (
        rpe_result is not None and isinstance(rpe_result, list)
    ):
        return render_multi_deviations_report(
            ate_results=ate_result, rpe_results=rpe_result, report_settings=report_settings
        )

    return render_single_deviations_report(
        ate_result=ate_result, rpe_result=rpe_result, report_settings=report_settings
    )


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
