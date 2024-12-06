"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import pandas as pd
import plotly.express as px
from plotly.offline import plot

from trajectopy.core.settings.report import ReportSettings


def render_heatmap(dataframe: pd.DataFrame, report_settings: ReportSettings = ReportSettings()) -> str:
    """
    Render a heatmap plot.

    Args:
        dataframe: The data to plot.
        report_settings: Report settings.

    Returns:
        HTML string of the rendered report including the heatmap plot.
    """
    fig = px.imshow(
        dataframe,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        title="Correlation Matrix",
        aspect="auto",
    )

    fig.update_layout(height=report_settings.single_plot_height)

    return plot(fig, output_type="div", config=report_settings.single_plot_export.to_config())
