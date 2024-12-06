"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot

from trajectopy.core.report.data import ATEReportData, ATEReportDataCollection


def add_to_dict(metrics: dict, field_name: str, content: list):
    current_content = metrics.get(field_name) or []
    current_content.extend(content)
    metrics[field_name] = current_content


def to_pos_metrics_df(report_data_collection: ATEReportDataCollection) -> pd.DataFrame:
    metrics = {}

    for data in report_data_collection.items:
        add_to_dict(metrics, "Trajectory", [data.short_name] * 6)
        add_to_dict(metrics, "Metric", ["ATE", "ATE Min", "ATE Max", "ATE Median", "ATE RMS", "ATE Std"])
        add_to_dict(
            metrics,
            "Value",
            [
                data.ate_result.pos_ate,
                data.ate_result.pos_dev_min,
                data.ate_result.pos_dev_max,
                data.ate_result.pos_dev_median,
                data.ate_result.pos_dev_rms,
                data.ate_result.pos_dev_std,
            ],
        )

    return pd.DataFrame(metrics)


def to_rot_metrics_df(report_data_collection: ATEReportDataCollection) -> pd.DataFrame:
    metrics = {}

    if not report_data_collection.has_ate_rot:
        return pd.DataFrame(metrics)

    for data in report_data_collection.items:
        if not data.has_ate_rot:
            continue

        add_to_dict(metrics, "Trajectory", [data.short_name] * 6)
        add_to_dict(metrics, "Metric", ["ATE", "ATE Min", "ATE Max", "ATE Median", "ATE RMS", "ATE Std"])
        add_to_dict(
            metrics,
            "Value",
            [
                np.rad2deg(data.ate_result.rot_ate),
                np.rad2deg(data.ate_result.rot_dev_min),
                np.rad2deg(data.ate_result.rot_dev_max),
                np.rad2deg(data.ate_result.rot_dev_median),
                np.rad2deg(data.ate_result.rot_dev_rms),
                np.rad2deg(data.ate_result.rot_dev_std),
            ],
        )

    return pd.DataFrame(metrics)


def render_multi_pos_bar_plot(report_data_collection: ATEReportDataCollection) -> str:
    metrics_df = to_pos_metrics_df(report_data_collection)

    fig = px.bar(metrics_df, barmode="group", x="Metric", y="Value", color="Trajectory")

    fig.update_layout(
        title_text="Absolute Trajectory Error (ATE) - Position",
        height=report_data_collection.items[0].settings.single_plot_height,
    )
    fig.update_yaxes(title_text=f"Value [{report_data_collection.items[0].ate_unit}]")
    return plot(fig, output_type="div", config=report_data_collection.items[0].settings.single_plot_export.to_config())


def render_pos_bar_plot(report_data: ATEReportData) -> str:
    return render_multi_pos_bar_plot(ATEReportDataCollection([report_data]))


def render_multi_rot_bar_plot(report_data_collection: ATEReportDataCollection) -> str:
    metrics_df = to_rot_metrics_df(report_data_collection)

    fig = px.bar(metrics_df, barmode="group", x="Metric", y="Value", color="Trajectory")

    fig.update_layout(
        title_text="Absolute Trajectory Error (ATE) - Rotation",
        height=report_data_collection.items[0].settings.single_plot_height,
    )
    fig.update_yaxes(title_text=f"Value [{report_data_collection.items[0].settings.rot_unit}]")
    return plot(fig, output_type="div", config=report_data_collection.items[0].settings.single_plot_export.to_config())


def render_rot_bar_plot(report_data: ATEReportData) -> str:
    return render_multi_rot_bar_plot(ATEReportDataCollection([report_data]))
