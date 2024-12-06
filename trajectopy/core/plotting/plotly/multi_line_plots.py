"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import itertools
from typing import List, Tuple, Union

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from trajectopy.core.plotting.utils import derive_xlabel_from_sortings, get_axis_label
from trajectopy.core.report.data import ATEReportDataCollection, RPEReportDataCollection
from trajectopy.core.settings.report import ReportSettings
from trajectopy.core.trajectory import Trajectory


def setup_edf_axis(report_data_collection: ATEReportDataCollection) -> Tuple[go.Figure, dict]:
    report_data_item = report_data_collection.items[0]
    if report_data_collection.has_ate_rot:
        fig = make_subplots(rows=2, cols=1)
        height = report_data_item.settings.two_subplots_height
        config = report_data_item.settings.two_subplots_export.to_config()

        fig.update_xaxes(title_text=f"[{report_data_item.settings.rot_unit}]", row=2, col=1)
        fig.update_yaxes(title_text="CDF", row=2, col=1)
    else:
        fig = make_subplots(rows=1, cols=1)
        height = report_data_item.settings.single_plot_height
        config = report_data_item.settings.single_plot_export.to_config()

    fig.update_layout(title="Cummulative Probability", height=height)
    fig.update_xaxes(title_text=f"[{report_data_item.ate_unit}]", row=1, col=1)
    fig.update_yaxes(title_text="CDF", row=1, col=1)

    return fig, config


def render_dev_edf(report_data_collection: ATEReportDataCollection) -> str:
    fig, config = setup_edf_axis(report_data_collection)

    for data, color in zip(report_data_collection.items, itertools.cycle(px.colors.qualitative.Plotly)):
        sorted_comb_pos_dev = np.sort(data.comb_dev_pos)
        pos_norm_cdf = np.arange(len(sorted_comb_pos_dev)) / float(len(sorted_comb_pos_dev))
        fig.add_trace(
            go.Scattergl(
                x=sorted_comb_pos_dev,
                y=pos_norm_cdf,
                mode=data.settings.plot_mode,
                name=f"{data.short_name}",
                marker=dict(color=color),
            ),
            row=1,
            col=1,
        )

        if data.has_ate_rot:
            sorted_comb_rot_dev = np.sort(data.comb_dev_rot)
            rot_norm_cdf = np.arange(len(sorted_comb_rot_dev)) / float(len(sorted_comb_rot_dev))
            fig.add_trace(
                go.Scattergl(
                    x=sorted_comb_rot_dev,
                    y=rot_norm_cdf,
                    mode=data.settings.plot_mode,
                    name=f"{data.short_name}",
                    marker=dict(color=color),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
    return plot(fig, output_type="div", config=config)


def setup_dev_comb_axis(report_data_collection: ATEReportDataCollection) -> Tuple[go.Figure, dict]:
    report_data_item = report_data_collection.items[0]
    if report_data_collection.has_ate_rot:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.update_xaxes(title_text=report_data_item.function_of_label, row=2, col=1)
        fig.update_yaxes(title_text=f"[{report_data_item.settings.rot_unit}]", row=2, col=1)

        config = report_data_item.settings.two_subplots_export.to_config()
        height = report_data_item.settings.two_subplots_height
    else:
        fig = make_subplots(rows=1, cols=1)
        fig.update_xaxes(title_text=report_data_item.function_of_label, row=1, col=1)

        config = report_data_item.settings.single_plot_export.to_config()
        height = report_data_item.settings.single_plot_height

    fig.update_layout(title="Trajectory Deviations", height=height)
    fig.update_yaxes(title_text=f"[{report_data_item.ate_unit}]", row=1, col=1)

    return fig, config


def render_dev_comb_plot(report_data_collection: ATEReportDataCollection) -> str:
    report_data = report_data_collection.items[0]

    any_rot_available = any(data.has_ate_rot for data in report_data_collection.items)

    y_data = (
        [[data.comb_dev_pos, data.comb_dev_rot if data.has_ate_rot else None] for data in report_data_collection.items]
        if any_rot_available
        else [[data.comb_dev_pos] for data in report_data_collection.items]
    )

    y_labels = (
        [f"[{report_data.ate_unit}]", f"[{report_data.settings.rot_unit}]"]
        if any_rot_available
        else [f"[{report_data.ate_unit}]"]
    )

    return render_shared_x_plot(
        x_data=[data.function_of for data in report_data_collection.items],
        y_data=y_data,
        names=[data.short_name for data in report_data_collection.items],
        x_label=report_data.function_of_label,
        y_labels=y_labels,
        title="Trajectory Deviations",
        report_settings=report_data.settings,
    )


def render_dev_pos_plot(report_data_collection: ATEReportDataCollection) -> str:
    report_data = report_data_collection.items[0]

    return render_shared_x_plot(
        x_data=[data.function_of for data in report_data_collection.items],
        y_data=[[data.pos_dev_x, data.pos_dev_y, data.pos_dev_z] for data in report_data_collection.items],
        names=[data.short_name for data in report_data_collection.items],
        x_label=report_data.function_of_label,
        y_labels=[
            f"{report_data.pos_dev_x_name} [{report_data.ate_unit}]",
            f"{report_data.pos_dev_y_name} [{report_data.ate_unit}]",
            f"{report_data.pos_dev_z_name} [{report_data.ate_unit}]",
        ],
        title="Position Deviations per Direction",
        report_settings=report_data.settings,
    )


def render_dev_rot_plot(report_data_collection: ATEReportDataCollection) -> str:
    report_data = report_data_collection.items[0]

    return render_shared_x_plot(
        x_data=[data.function_of for data in report_data_collection.items],
        y_data=[
            [data.rot_dev_x, data.rot_dev_y, data.rot_dev_z]
            for data in report_data_collection.items
            if data.has_ate_rot
        ],
        names=[data.short_name for data in report_data_collection.items],
        x_label=report_data.function_of_label,
        y_labels=[
            f"{report_data.settings.rot_x_name} [{report_data.settings.rot_unit}]",
            f"{report_data.settings.rot_y_name} [{report_data.settings.rot_unit}]",
            f"{report_data.settings.rot_z_name} [{report_data.settings.rot_unit}]",
        ],
        title="Rotation Deviations per Direction",
        report_settings=report_data.settings,
    )


def setup_rpe_axis(report_data_collection: RPEReportDataCollection) -> Tuple[go.Figure, dict]:
    report_data_item = report_data_collection.items[0]
    if report_data_collection.has_rpe_rot:
        rpe_rot_item = report_data_collection.get_rpe_results(rot_required=True)[0]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        height = report_data_item.settings.two_subplots_height
        config = report_data_item.settings.two_subplots_export.to_config()
        fig.update_xaxes(title_text=f"Pose Distance [{rpe_rot_item.pair_distance_unit}]", row=2, col=1)
        fig.update_yaxes(title_text=f"[{rpe_rot_item.rot_drift_unit}]", row=2, col=1)

    else:
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        height = report_data_item.settings.single_plot_height
        config = report_data_item.settings.single_plot_export.to_config()
        fig.update_xaxes(title_text=f"Pose Distance [{report_data_item.rpe_result.pair_distance_unit}]", row=1, col=1)

    fig.update_layout(title="Relative Pose Error", height=height)
    fig.update_yaxes(title_text=f"[{report_data_item.rpe_result.pos_drift_unit}]", row=1, col=1)

    return fig, config


def render_rpe(report_data_collection: RPEReportDataCollection) -> str:
    fig, config = setup_rpe_axis(report_data_collection)

    for data, color in zip(report_data_collection.items, itertools.cycle(px.colors.qualitative.Plotly)):
        rpe_result = data.rpe_result

        fig.add_trace(
            go.Scattergl(
                x=rpe_result.mean_pair_distances,
                y=rpe_result.pos_dev_mean,
                mode=data.settings.plot_mode,
                name=f"{data.short_name}",
                error_y=dict(
                    type="data",
                    array=rpe_result.pos_std,
                    visible=True,
                ),
                marker=dict(color=color),
            ),
            row=1,
            col=1,
        )

        if rpe_result.has_rot_dev:
            fig.add_trace(
                go.Scattergl(
                    x=rpe_result.mean_pair_distances,
                    y=np.rad2deg(rpe_result.rot_dev_mean),
                    mode=data.settings.plot_mode,
                    name=f"{data.short_name}",
                    error_y=dict(
                        type="data",
                        array=np.rad2deg(rpe_result.rot_std),
                        visible=True,
                    ),
                    marker=dict(color=color),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    return plot(fig, output_type="div", config=config)


def render_pos_plot(trajectories: List[Trajectory], report_settings: ReportSettings = ReportSettings()) -> str:
    pos_x_label, pos_y_label, pos_z_label = get_axis_label(trajectories=trajectories)
    x_label = derive_xlabel_from_sortings([traj.sorting for traj in trajectories])

    return render_shared_x_plot(
        x_data=[traj.function_of for traj in trajectories],
        y_data=[[traj.xyz[:, 0], traj.xyz[:, 1], traj.xyz[:, 2]] for traj in trajectories],
        names=[traj.name for traj in trajectories],
        x_label=x_label,
        y_labels=[pos_x_label, pos_y_label, pos_z_label],
        title="Position Components",
        report_settings=report_settings,
    )


def render_rot_plot(trajectories: List[Trajectory], report_settings: ReportSettings = ReportSettings()) -> str:
    rot_x_label, rot_y_label, rot_z_label = ("roll [°]", "pitch [°]", "yaw [°]")
    x_label = derive_xlabel_from_sortings([traj.sorting for traj in trajectories])

    traj_rpy = [np.rad2deg(traj.rpy) for traj in trajectories]
    return render_shared_x_plot(
        x_data=[traj.function_of for traj in trajectories],
        y_data=[[rpy[:, 0], rpy[:, 1], rpy[:, 2]] for rpy in traj_rpy],
        names=[traj.name for traj in trajectories],
        x_label=x_label,
        y_labels=[rot_x_label, rot_y_label, rot_z_label],
        title="Position Components",
        report_settings=report_settings,
    )


def render_shared_x_plot(
    x_data: List[np.ndarray],
    y_data: List[List[Union[np.ndarray, None]]],
    names: List[str],
    x_label: str,
    y_labels: List[str],
    title: str,
    report_settings: ReportSettings = ReportSettings(),
) -> str:
    height_dict = {
        1: report_settings.single_plot_height,
        2: report_settings.two_subplots_height,
        3: report_settings.three_subplots_height,
    }
    fig = make_subplots(rows=len(y_labels), cols=1, shared_xaxes=True)
    fig.update_layout(title=title, height=height_dict[len(y_labels)])

    fig.update_xaxes(title_text=x_label, row=len(y_labels), col=1)

    for i, y_label in enumerate(y_labels):
        fig.update_yaxes(title_text=y_label, row=i + 1, col=1)

    for x_data_item, y_data_item, name, color in zip(
        x_data, y_data, names, itertools.cycle(px.colors.qualitative.Plotly)
    ):
        for i, y_data_subitem in enumerate(y_data_item):
            if y_data_subitem is None:
                continue

            fig.add_trace(
                go.Scattergl(
                    x=x_data_item,
                    y=y_data_subitem,
                    mode=report_settings.plot_mode,
                    name=name,
                    marker=dict(color=color),
                    showlegend=(i == 0) and len(y_data) > 1,
                ),
                row=i + 1,
                col=1,
            )

    config_dicts = {
        1: report_settings.single_plot_export.to_config(),
        2: report_settings.two_subplots_export.to_config(),
        3: report_settings.three_subplots_export.to_config(),
    }

    return plot(fig, output_type="div", config=config_dicts[len(y_labels)])
