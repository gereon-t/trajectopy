"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import List, Optional

import jinja2

from trajectopy.core.evaluation.ate_result import ATEResult
from trajectopy.core.evaluation.rpe_result import RPEResult
from trajectopy.core.plotting.plotly import bar_plots, multi_line_plots
from trajectopy.core.report.data import ATEReportData, ATEReportDataCollection, RPEReportData, RPEReportDataCollection
from trajectopy.core.report.utils import TEMPLATES_PATH, convert_icon_to_base64
from trajectopy.core.settings.report import ReportSettings

logger = logging.getLogger("root")


def render_one_line_plots(
    ate_report_data_collection: Optional[ATEReportDataCollection] = None,
    rpe_report_data_collection: Optional[RPEReportDataCollection] = None,
) -> List[str]:
    if ate_report_data_collection is None and rpe_report_data_collection is not None:
        return [multi_line_plots.render_rpe(rpe_report_data_collection)]

    # only for type checking (this shouldnt be necessary but mypy is complaining otherwise)
    if ate_report_data_collection is None:
        return []

    one_line_plots = [
        multi_line_plots.render_dev_comb_plot(ate_report_data_collection),
        multi_line_plots.render_dev_edf(ate_report_data_collection),
        bar_plots.render_multi_pos_bar_plot(ate_report_data_collection),
        multi_line_plots.render_dev_pos_plot(ate_report_data_collection),
    ]

    if ate_report_data_collection.has_ate_rot:
        one_line_plots.append(bar_plots.render_multi_rot_bar_plot(ate_report_data_collection))
        one_line_plots.append(multi_line_plots.render_dev_rot_plot(ate_report_data_collection))

    if rpe_report_data_collection is not None:
        one_line_plots.insert(1, multi_line_plots.render_rpe(rpe_report_data_collection))

    return one_line_plots


def render_multi_report(
    *,
    ate_results: Optional[List[ATEResult]] = None,
    rpe_results: Optional[List[RPEResult]] = None,
    report_settings: ReportSettings = ReportSettings(),
) -> str:
    """
    Renders a html report string of multiple trajectory comparisons

    Args:
        ate_results (list[ATEResult]): A list of absolute trajectory error results
        rpe_results (Optional[list[RPEResult]]): A list of relative pose error results
        report_settings (ReportSettings): The report settings

    Returns:
        str: The html report string

    """
    if ate_results is None and rpe_results is None:
        raise ValueError("Either ate_results or rpe_results must be provided")

    template = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_PATH)).get_template("multi_template.html")
    icon = convert_icon_to_base64()

    ate_report_data_collection = (
        ATEReportDataCollection(
            [ATEReportData(ate_result=ate_result, settings=report_settings) for ate_result in ate_results]
        )
        if ate_results is not None
        else None
    )
    rpe_report_data_collection = (
        RPEReportDataCollection(
            [RPEReportData(rpe_result=rpe_result, settings=report_settings) for rpe_result in rpe_results]
        )
        if rpe_results is not None
        else None
    )

    one_line_plots = render_one_line_plots(ate_report_data_collection, rpe_report_data_collection)

    context = {
        "title": "Trajectory Comparison",
        "rpe_available": rpe_results is not None,
        "one_line_plots": one_line_plots,
        "icon": icon,
    }

    return template.render(context)
