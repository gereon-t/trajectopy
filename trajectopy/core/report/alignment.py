"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from trajectopy.core.alignment.parameters import AlignmentParameters
from trajectopy.core.plotting.plotly import heatmaps, tables
from trajectopy.settings import ReportSettings

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


def render_one_line_alignment_plots(
    alignment_parameters: AlignmentParameters, report_settings: ReportSettings = ReportSettings()
) -> List[str]:
    dataframe = build_correlation_dataframe(alignment_parameters, enabled_only=True)

    one_line_plots = [tables.render_alignment_table(alignment_parameters, report_settings)]
    one_line_plots.append(heatmaps.render_heatmap(dataframe, report_settings))

    return one_line_plots
