"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import plotly.graph_objects as go
from plotly.offline import plot

from trajectopy.core.report.data import ATEReportData


def render_pos_devs(report_data: ATEReportData) -> str:
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=report_data.pos_dev_x, name=report_data.pos_dev_x_name, opacity=report_data.settings.histogram_opacity
        )
    )
    fig.add_trace(
        go.Histogram(
            x=report_data.pos_dev_y, name=report_data.pos_dev_y_name, opacity=report_data.settings.histogram_opacity
        )
    )
    fig.add_trace(
        go.Histogram(
            x=report_data.pos_dev_z, name=report_data.pos_dev_z_name, opacity=report_data.settings.histogram_opacity
        )
    )

    fig.update_layout(
        title="Position Deviations",
        xaxis=dict(title=f"Absolute Position Error [{report_data.ate_unit}]"),
        yaxis=dict(title=report_data.settings.histogram_yaxis_title),
        barmode=report_data.settings.histogram_barmode,
        bargap=report_data.settings.histogram_bargap,
        height=report_data.settings.single_plot_height,
    )
    return plot(fig, output_type="div", config=report_data.settings.single_plot_export.to_config())


def render_rot_devs(report_data: ATEReportData) -> str:
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=report_data.rot_dev_x,
            name=report_data.settings.rot_x_name,
            opacity=report_data.settings.histogram_opacity,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=report_data.rot_dev_y,
            name=report_data.settings.rot_y_name,
            opacity=report_data.settings.histogram_opacity,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=report_data.rot_dev_z,
            name=report_data.settings.rot_z_name,
            opacity=report_data.settings.histogram_opacity,
        )
    )

    fig.update_layout(
        title="Rotation Deviations",
        xaxis=dict(title=f"Absolute Rotation Error [{report_data.settings.rot_unit}]"),
        yaxis=dict(title=report_data.settings.histogram_yaxis_title),
        barmode=report_data.settings.histogram_barmode,
        bargap=report_data.settings.histogram_bargap,
        height=report_data.settings.single_plot_height,
    )

    return plot(fig, output_type="div", config=report_data.settings.single_plot_export.to_config())
