"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from trajectopy.core.report.data import ATEReportData, RPEReportData


def render_dev_edf(report_data: ATEReportData) -> str:
    if report_data.has_ate_rot:
        fig = make_subplots(rows=2, cols=1)
    else:
        fig = make_subplots(rows=1, cols=1)

    sorted_comb_pos_dev = np.sort(report_data.comb_dev_pos)
    pos_norm_cdf = np.arange(len(sorted_comb_pos_dev)) / float(len(sorted_comb_pos_dev))
    fig.add_trace(
        go.Scattergl(x=sorted_comb_pos_dev, y=pos_norm_cdf, mode=report_data.settings.plot_mode, name="position"),
        row=1,
        col=1,
    )

    if report_data.has_ate_rot:
        sorted_comb_rot_dev = np.sort(report_data.comb_dev_rot)
        rot_norm_cdf = np.arange(len(sorted_comb_rot_dev)) / float(len(sorted_comb_rot_dev))
        fig.add_trace(
            go.Scattergl(x=sorted_comb_rot_dev, y=rot_norm_cdf, mode=report_data.settings.plot_mode, name="rotation"),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text=f"[{report_data.settings.rot_unit}]", row=2, col=1)
        fig.update_yaxes(title_text="CDF", row=2, col=1)

    if report_data.has_ate_rot:
        height = report_data.settings.two_subplots_height
        config = report_data.settings.two_subplots_export.to_config()
    else:
        height = report_data.settings.single_plot_height
        config = report_data.settings.single_plot_export.to_config()

    fig.update_layout(title="Cummulative Probability", height=height)
    fig.update_xaxes(title_text=f"[{report_data.ate_unit}]", row=1, col=1)
    fig.update_yaxes(title_text="CDF", row=1, col=1)

    return plot(fig, output_type="div", config=config)


def render_pos_plot(report_data: ATEReportData) -> str:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.pos_x,
            mode=report_data.settings.plot_mode,
            name=report_data.settings.pos_x_name,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.pos_y,
            mode=report_data.settings.plot_mode,
            name=report_data.settings.pos_y_name,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.pos_z,
            mode=report_data.settings.plot_mode,
            name=report_data.settings.pos_z_name,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(title="Position Components", height=report_data.settings.three_subplots_height)

    fig.update_xaxes(title_text=report_data.function_of_label, row=3, col=1)
    fig.update_yaxes(title_text=f"[{report_data.settings.pos_x_unit}]", row=1, col=1)
    fig.update_yaxes(title_text=f"[{report_data.settings.pos_y_unit}]", row=2, col=1)
    fig.update_yaxes(title_text=f"[{report_data.settings.pos_z_unit}]", row=3, col=1)

    return plot(fig, output_type="div", config=report_data.settings.three_subplots_export.to_config())


def render_rot_plot(report_data: ATEReportData) -> str:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.roll,
            mode=report_data.settings.plot_mode,
            name=report_data.settings.rot_x_name,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.pitch,
            mode=report_data.settings.plot_mode,
            name=report_data.settings.rot_y_name,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.yaw,
            mode=report_data.settings.plot_mode,
            name=report_data.settings.rot_z_name,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(title="Rotation Components", height=report_data.settings.three_subplots_height)

    fig.update_xaxes(title_text=report_data.function_of_label, row=3, col=1)
    fig.update_yaxes(title_text=f"[{report_data.settings.rot_unit}]", row=1, col=1)
    fig.update_yaxes(title_text=f"[{report_data.settings.rot_unit}]", row=2, col=1)
    fig.update_yaxes(title_text=f"[{report_data.settings.rot_unit}]", row=3, col=1)

    return plot(fig, output_type="div", config=report_data.settings.three_subplots_export.to_config())


def render_dev_pos_plot(report_data: ATEReportData) -> str:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.pos_dev_x,
            mode=report_data.settings.plot_mode,
            name=report_data.pos_dev_x_name,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.pos_dev_y,
            mode=report_data.settings.plot_mode,
            name=report_data.pos_dev_y_name,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.pos_dev_z,
            mode=report_data.settings.plot_mode,
            name=report_data.pos_dev_z_name,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(title="Position Deviations per Direction", height=report_data.settings.three_subplots_height)

    fig.update_xaxes(title_text=report_data.function_of_label, row=3, col=1)
    fig.update_yaxes(title_text=f"[{report_data.ate_unit}]", row=1, col=1)
    fig.update_yaxes(title_text=f"[{report_data.ate_unit}]", row=2, col=1)
    fig.update_yaxes(title_text=f"[{report_data.ate_unit}]", row=3, col=1)

    return plot(fig, output_type="div", config=report_data.settings.three_subplots_export.to_config())


def render_dev_rot_plot(report_data: ATEReportData) -> str:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.rot_dev_x,
            mode=report_data.settings.plot_mode,
            name=report_data.settings.rot_x_name,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.rot_dev_y,
            mode=report_data.settings.plot_mode,
            name=report_data.settings.rot_y_name,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.rot_dev_z,
            mode=report_data.settings.plot_mode,
            name=report_data.settings.rot_z_name,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(title="Rotation Deviations per Axis", height=report_data.settings.three_subplots_height)
    fig.update_xaxes(title_text=report_data.function_of_label, row=3, col=1)
    fig.update_yaxes(title_text=f"[{report_data.settings.rot_unit}]", row=1, col=1)
    fig.update_yaxes(title_text=f"[{report_data.settings.rot_unit}]", row=2, col=1)
    fig.update_yaxes(title_text=f"[{report_data.settings.rot_unit}]", row=3, col=1)

    return plot(fig, output_type="div", config=report_data.settings.three_subplots_export.to_config())


def render_dev_comb_plot(report_data: ATEReportData) -> str:
    if report_data.has_ate_rot:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        config = report_data.settings.two_subplots_export.to_config()
        height = report_data.settings.two_subplots_height
    else:
        fig = make_subplots(rows=1, cols=1)
        config = report_data.settings.single_plot_export.to_config()
        height = report_data.settings.single_plot_height

    fig.add_trace(
        go.Scattergl(
            x=report_data.function_of,
            y=report_data.comb_dev_pos,
            mode=report_data.settings.plot_mode,
            name="position",
        ),
        row=1,
        col=1,
    )

    if report_data.has_ate_rot:
        fig.add_trace(
            go.Scattergl(
                x=report_data.function_of,
                y=report_data.comb_dev_rot,
                mode=report_data.settings.plot_mode,
                name="rotation",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text=f"[{report_data.settings.rot_unit}]", row=2, col=1)

    fig.update_layout(title="Trajectory Deviations", height=height)

    fig.update_xaxes(title_text=report_data.function_of_label, row=2 if report_data.has_ate_rot else 1, col=1)
    fig.update_yaxes(title_text=f"[{report_data.ate_unit}]", row=1, col=1)

    return plot(fig, output_type="div", config=config)


def render_rpe(report_data: RPEReportData) -> str:
    rpe_result = report_data.rpe_result
    if rpe_result is None:
        return ""

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scattergl(
            x=rpe_result.mean_pair_distances,
            y=rpe_result.pos_dev_mean,
            mode=report_data.settings.plot_mode,
            name="position",
            error_y=dict(
                type="data",
                array=rpe_result.pos_std,
                visible=True,
            ),
        ),
        row=1,
        col=1,
    )

    if rpe_result.has_rot_dev:
        fig.add_trace(
            go.Scattergl(
                x=rpe_result.mean_pair_distances,
                y=np.rad2deg(rpe_result.rot_dev_mean),
                mode=report_data.settings.plot_mode,
                name="rotation",
                error_y=dict(
                    type="data",
                    array=np.rad2deg(rpe_result.rot_std),
                    visible=True,
                ),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text=f"[{rpe_result.rot_drift_unit}]", row=2, col=1)

        height = report_data.settings.two_subplots_height
        config = report_data.settings.two_subplots_export.to_config()
    else:
        height = report_data.settings.single_plot_height
        config = report_data.settings.single_plot_export.to_config()

    fig.update_layout(title="Relative Pose Error", height=height)
    fig.update_yaxes(title_text=f"[{rpe_result.pos_drift_unit}]", row=1, col=1)
    fig.update_xaxes(
        title_text=f"Pose Distance [{rpe_result.pair_distance_unit}]", row=2 if rpe_result.has_rot_dev else 1, col=1
    )

    return plot(fig, output_type="div", config=config)
