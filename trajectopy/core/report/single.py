"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import List, Optional

import jinja2
import numpy as np

from trajectopy.core.evaluation.ate_result import ATEResult
from trajectopy.core.evaluation.rpe_result import RPEResult
from trajectopy.core.plotting.plotly import bar_plots, histograms, line_plots, scatter_plots
from trajectopy.core.report.data import ATEReportData, RPEReportData
from trajectopy.core.report.utils import TEMPLATES_PATH, convert_icon_to_base64, number_to_string
from trajectopy.core.settings.report import ReportSettings

logger = logging.getLogger("root")


def render_side_by_side_plots(ate_report_data: ATEReportData) -> List[str]:
    side_by_side_plots = [scatter_plots.render_pos_devs(ate_report_data)]

    if not ate_report_data.has_ate_rot:
        return side_by_side_plots

    side_by_side_plots.append(scatter_plots.render_rot_devs(ate_report_data))

    return side_by_side_plots


def render_one_line_plots(
    ate_report_data: Optional[ATEReportData] = None, rpe_report_data: Optional[RPEReportData] = None
) -> List[str]:
    one_line_plots = []

    if rpe_report_data is not None:
        one_line_plots.append(line_plots.render_rpe(rpe_report_data))

    if ate_report_data is None:
        return one_line_plots

    one_line_plots.extend(
        (
            histograms.render_pos_devs(ate_report_data),
            bar_plots.render_pos_bar_plot(ate_report_data),
            line_plots.render_dev_edf(ate_report_data),
            line_plots.render_dev_comb_plot(ate_report_data),
            line_plots.render_dev_pos_plot(ate_report_data),
        )
    )

    if ate_report_data.settings.scatter_detailed:
        one_line_plots.extend(
            (
                scatter_plots.render_pos_x_devs(ate_report_data),
                scatter_plots.render_pos_y_devs(ate_report_data),
                scatter_plots.render_pos_z_devs(ate_report_data),
            )
        )
    if not ate_report_data.has_ate_rot:
        return one_line_plots

    one_line_plots.insert(2, histograms.render_rot_devs(ate_report_data))
    one_line_plots.insert(4, bar_plots.render_rot_bar_plot(ate_report_data))
    one_line_plots.append(line_plots.render_dev_rot_plot(ate_report_data))

    if ate_report_data.settings.scatter_detailed:
        one_line_plots.extend(
            (
                scatter_plots.render_rot_x_devs(ate_report_data),
                scatter_plots.render_rot_y_devs(ate_report_data),
                scatter_plots.render_rot_z_devs(ate_report_data),
            )
        )
    return one_line_plots


def render_single_report(
    *,
    ate_result: Optional[ATEResult] = None,
    rpe_result: Optional[RPEResult] = None,
    report_settings: ReportSettings = ReportSettings(),
) -> str:
    """
    Renders a html report string of a single trajectory comparison.

    Args:
        ate_result (ATEResult): The absolute trajectory error result
        rpe_result (Optional[RPEResult]): The relative pose error result
        max_std (float): The upper bound of scatter plot colorbars is set to max_std * std of the data
        report_settings (ReportSettings): The report settings

    Returns:
        str: The html report string

    """
    if ate_result is None and rpe_result is None:
        raise ValueError("Either ate_result or rpe_result must be provided.")

    template = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_PATH)).get_template("single_template.html")
    icon = convert_icon_to_base64()

    ate_report_data = (
        ATEReportData(ate_result=ate_result, settings=report_settings) if ate_result is not None else None
    )
    rpe_report_data = (
        RPEReportData(rpe_result=rpe_result, settings=report_settings) if rpe_result is not None else None
    )

    side_by_side_plots = render_side_by_side_plots(ate_report_data) if ate_report_data is not None else []
    one_line_plots = render_one_line_plots(ate_report_data, rpe_report_data)

    if len(side_by_side_plots) == 1:
        one_line_plots = side_by_side_plots + one_line_plots
        side_by_side_plots = []

    context = {
        "title": ate_result.name if ate_result is not None else rpe_result.name,
        "ate_pos": number_to_string(ate_result.pos_ate) if ate_result is not None else "-",
        "ate_rot": number_to_string(np.rad2deg(ate_result.rot_ate)) if ate_result is not None else "-",
        "rpe_pos": number_to_string(rpe_result.pos_rpe) if rpe_result is not None else "-",
        "rpe_rot": number_to_string(np.rad2deg(rpe_result.rot_rpe)) if rpe_result is not None else "-",
        "rpe_pos_drift_unit": rpe_result.pos_drift_unit if rpe_result is not None else "-",
        "rpe_rot_drift_unit": rpe_result.rot_drift_unit if rpe_result is not None else "-",
        "ate_pos_unit": "mm" if report_settings.ate_unit_is_mm else "m",
        "rpe_available": rpe_result is not None,
        "side_by_side_plots": side_by_side_plots,
        "one_line_plots": one_line_plots,
        "icon": icon,
        "rot_unit": report_settings.rot_unit,
    }

    return template.render(context)
