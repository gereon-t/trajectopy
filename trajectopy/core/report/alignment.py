"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import List

import jinja2
import numpy as np
import pandas as pd

from trajectopy.core.alignment.parameters import AlignmentParameters
from trajectopy.core.plotting.plotly import heatmaps, tables
from trajectopy.core.report.utils import TEMPLATES_PATH, convert_icon_to_base64
from trajectopy.core.settings.report import ReportSettings

logger = logging.getLogger("root")


def build_correlation_dataframe(estimated_parameters: AlignmentParameters, enabled_only: bool = True) -> pd.DataFrame:
    covariance_matrix = estimated_parameters.get_covariance_matrix(enabled_only=enabled_only)
    std_devs = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
    np.fill_diagonal(correlation_matrix, np.nan)
    labels = estimated_parameters.params_labels(enabled_only=enabled_only, lower_case=True)
    return pd.DataFrame(
        correlation_matrix,
        index=labels,
        columns=labels,
    )


def render_one_line_plots(
    alignment_parameters: AlignmentParameters, report_settings: ReportSettings = ReportSettings()
) -> List[str]:
    dataframe = build_correlation_dataframe(alignment_parameters, enabled_only=True)

    one_line_plots = [tables.render_alignment_table(alignment_parameters, report_settings)]
    one_line_plots.append(heatmaps.render_heatmap(dataframe, report_settings))

    return one_line_plots


def create_alignment_report(
    *,
    alignment_parameters: AlignmentParameters,
    name: str = "Alignment",
    report_settings: ReportSettings = ReportSettings()
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

    one_line_plots = render_one_line_plots(alignment_parameters, report_settings)

    context = {
        "title": name,
        "one_line_plots": one_line_plots,
        "icon": icon,
    }

    return template.render(context)
