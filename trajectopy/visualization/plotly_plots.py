import itertools
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from trajectopy.core.settings import ReportSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.lib.alignment.parameters import AlignmentParameters
from trajectopy.utils.common import (
    derive_xlabel_from_sortings,
    get_axis_label,
    get_sorting,
)
from trajectopy.utils.definitions import TrajectoriesSorting

logger = logging.getLogger(__name__)


def _setup_pos_axis_indices(report_settings: ReportSettings) -> tuple[list[str], list[int]]:
    pos_axis_labels = [
        f"{report_settings.pos_x_name} [{report_settings.pos_x_unit}]",
        f"{report_settings.pos_y_name} [{report_settings.pos_y_unit}]",
        f"{report_settings.pos_z_name} [{report_settings.pos_z_unit}]",
    ]
    axes = ["x", "y", "z"]
    axes_indices = [axes.index(ax_char) for ax_char in report_settings.scatter_axis_order]
    return pos_axis_labels, axes_indices


def _get_marker_dict(
    report_settings: ReportSettings, colorbar_title: str = "", colors: np.ndarray | None = None
) -> dict:
    if colors is None:
        return dict(size=report_settings.scatter_marker_size)

    if report_settings.scatter_smooth:
        colors = np.convolve(
            colors, np.ones(report_settings.scatter_smooth_window) / report_settings.scatter_smooth_window, mode="same"
        )

    cbar_min = min(colors)
    cbar_max = min(np.max(colors), cbar_min + np.std(colors) * report_settings.scatter_max_std)
    return dict(
        color=colors,
        colorscale=report_settings.scatter_colorscale,
        colorbar=dict(title=colorbar_title),
        cmin=cbar_min,
        cmax=cbar_max,
        size=report_settings.scatter_marker_size,
    )


def scatter(
    pos: np.ndarray,
    report_settings: ReportSettings,
    figure_title: str,
    colorbar_title: str,
    colors: np.ndarray | None = None,
) -> str:
    marker_dict = _get_marker_dict(report_settings, colorbar_title, colors)
    pos_axis_labels, axes_indices = _setup_pos_axis_indices(report_settings)

    fig = go.Figure()

    plotting_dim = len(report_settings.scatter_axis_order)
    mean_pos = np.mean(pos, axis=0)

    if plotting_dim == 2:
        if report_settings.scatter_plot_on_map:
            fig.add_trace(
                go.Scattermapbox(
                    lat=pos[:, axes_indices[0]],
                    lon=pos[:, axes_indices[1]],
                    mode=report_settings.scatter_mode,
                    marker=marker_dict,
                ),
            )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=pos[:, axes_indices[0]],
                    y=pos[:, axes_indices[1]],
                    mode=report_settings.scatter_mode,
                    marker=marker_dict,
                )
            )
    elif plotting_dim == 3:
        if report_settings.scatter_plot_on_map:
            logger.warning("Mapbox is not supported for 3D scatter plots.")

        fig.add_trace(
            go.Scatter3d(
                x=pos[:, axes_indices[0]],
                y=pos[:, axes_indices[1]],
                z=pos[:, axes_indices[2]],
                mode=report_settings.scatter_mode,
                marker=marker_dict,
            )
        )
    else:
        raise ValueError(f"Invalid dimension {plotting_dim}.")

    if report_settings.scatter_mapbox_token:
        mapbox_dict = dict(
            accesstoken=report_settings.scatter_mapbox_token,
            bearing=0,
            center=dict(lat=mean_pos[0], lon=mean_pos[1]),
            zoom=report_settings.scatter_mapbox_zoom,
        )
    else:
        mapbox_dict = dict(
            bearing=0,
            center=dict(lat=mean_pos[0], lon=mean_pos[1]),
            zoom=report_settings.scatter_mapbox_zoom,
        )

    fig.update_layout(
        xaxis=dict(title=pos_axis_labels[axes_indices[0]]),
        yaxis=dict(title=pos_axis_labels[axes_indices[1]]),
        title=figure_title,
        height=report_settings.single_plot_height,
        mapbox_style=report_settings.scatter_mapbox_style,
        mapbox=mapbox_dict,
        hovermode="closest",
        autosize=True,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )

    if plotting_dim == 3:
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis=dict(title=pos_axis_labels[axes_indices[0]]),
                yaxis=dict(title=pos_axis_labels[axes_indices[1]]),
                zaxis=dict(title=pos_axis_labels[axes_indices[2]]),
            ),
        )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return plot(fig, output_type="div", config=report_settings.single_plot_export.to_config())


def plot_heatmap(dataframe: pd.DataFrame, report_settings: ReportSettings = ReportSettings()) -> str:
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


def plot_trajectories_xyz(trajectories: list[Trajectory], report_settings: ReportSettings = ReportSettings()) -> str:
    pos_x_label, pos_y_label, pos_z_label = get_axis_label(trajectories=trajectories)
    trajectories_sorting = get_sorting([traj.sorting for traj in trajectories])
    all_unix = all(traj.is_unix_time for traj in trajectories)
    x_label = derive_xlabel_from_sortings(trajectories_sorting, all_unix)
    return plot_subplots_with_shared_x_axis(
        x_data=[
            traj.datetimes if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME else traj.index
            for traj in trajectories
        ],
        y_data=[[traj.xyz[:, 0], traj.xyz[:, 1], traj.xyz[:, 2]] for traj in trajectories],
        names=[traj.name for traj in trajectories],
        x_label=x_label,
        y_labels=[pos_x_label, pos_y_label, pos_z_label],
        title="Position Components",
        report_settings=report_settings,
    )


def plot_trajectories_rpy(trajectories: list[Trajectory], report_settings: ReportSettings = ReportSettings()) -> str:
    rot_x_label, rot_y_label, rot_z_label = ("roll [°]", "pitch [°]", "yaw [°]")
    trajectories_sorting = get_sorting([traj.sorting for traj in trajectories])
    all_unix = all(traj.is_unix_time for traj in trajectories)
    x_label = derive_xlabel_from_sortings(trajectories_sorting, all_unix)
    traj_rpy = [np.rad2deg(traj.rpy) for traj in trajectories]
    return plot_subplots_with_shared_x_axis(
        x_data=[
            traj.datetimes if all_unix and trajectories_sorting == TrajectoriesSorting.ALL_TIME else traj.index
            for traj in trajectories
        ],
        y_data=[[rpy[:, 0], rpy[:, 1], rpy[:, 2]] for rpy in traj_rpy],
        names=[traj.name for traj in trajectories],
        x_label=x_label,
        y_labels=[rot_x_label, rot_y_label, rot_z_label],
        title="Rotation Components",
        report_settings=report_settings,
    )


def plot_subplots_with_shared_x_axis(
    x_data: list[np.ndarray],
    y_data: list[list[np.ndarray | None]],
    names: list[str],
    x_label: str,
    y_labels: list[str],
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
                    name=f"{name} {y_labels[i]}",
                    marker=dict(color=color),
                    showlegend=len(y_data) > 1,
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


def scatter_trajectories_mapbox(trajectories: list[Trajectory], report_settings: ReportSettings) -> str:
    plotable_trajectories = [
        trajectory for trajectory in trajectories if trajectory.positions.local_transformer is not None
    ]

    if len(plotable_trajectories) != len(trajectories):
        logger.warning("Some trajectories could not be plotted because they are in a local coordinate system.")
        return ""

    for trajectory in plotable_trajectories:
        trajectory.positions.to_epsg(4326)  # WGS84

    marker_dict = _get_marker_dict(report_settings)
    _, axes_indices = _setup_pos_axis_indices(report_settings)

    fig = go.Figure()

    mean_pos = np.mean(np.vstack([trajectory.xyz for trajectory in plotable_trajectories]), axis=0)
    for trajectory in plotable_trajectories:
        pos = trajectory.xyz
        fig.add_trace(
            go.Scattermapbox(
                lat=pos[:, axes_indices[0]],
                lon=pos[:, axes_indices[1]],
                mode=report_settings.scatter_mode,
                marker=marker_dict,
                name=trajectory.name,
            )
        )

    if report_settings.scatter_mapbox_token:
        mapbox_dict = dict(
            accesstoken=report_settings.scatter_mapbox_token,
            bearing=0,
            center=dict(lat=mean_pos[0], lon=mean_pos[1]),
            zoom=report_settings.scatter_mapbox_zoom,
        )
    else:
        mapbox_dict = dict(
            bearing=0,
            center=dict(lat=mean_pos[0], lon=mean_pos[1]),
            zoom=report_settings.scatter_mapbox_zoom,
        )

    fig.update_layout(
        title="Trajectory Plot",
        mapbox_style=report_settings.scatter_mapbox_style,
        autosize=True,
        hovermode="closest",
        mapbox=mapbox_dict,
        height=report_settings.single_plot_height,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )

    return plot(fig, output_type="div", config=report_settings.single_plot_export.to_config())


def scatter_trajectories(trajectories: list[Trajectory], report_settings: ReportSettings) -> str:
    x_label, y_label, z_label = get_axis_label(trajectories=trajectories)
    pos_axis_labels = [x_label, y_label, z_label]

    marker_dict = _get_marker_dict(report_settings)
    _, axes_indices = _setup_pos_axis_indices(report_settings)

    fig = go.Figure()

    plotting_dim = len(report_settings.scatter_axis_order)

    for trajectory in trajectories:
        pos = trajectory.xyz
        if plotting_dim == 2:
            fig.add_trace(
                go.Scattergl(
                    x=pos[:, axes_indices[0]],
                    y=pos[:, axes_indices[1]],
                    mode=report_settings.scatter_mode,
                    marker=marker_dict,
                    name=trajectory.name,
                )
            )
        elif plotting_dim == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=pos[:, axes_indices[0]],
                    y=pos[:, axes_indices[1]],
                    z=pos[:, axes_indices[2]],
                    mode=report_settings.scatter_mode,
                    marker=marker_dict,
                    name=trajectory.name,
                )
            )
        else:
            raise ValueError(f"Invalid dimension {plotting_dim}.")

    fig.update_layout(
        xaxis=dict(title=pos_axis_labels[axes_indices[0]]),
        yaxis=dict(title=pos_axis_labels[axes_indices[1]]),
        title="Trajectory Plot",
        height=report_settings.single_plot_height,
    )

    if plotting_dim == 3:
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis=dict(title=pos_axis_labels[axes_indices[0]]),
                yaxis=dict(title=pos_axis_labels[axes_indices[1]]),
                zaxis=dict(title=pos_axis_labels[axes_indices[2]]),
            ),
        )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return plot(fig, output_type="div", config=report_settings.single_plot_export.to_config())


def plot_alignment_table(
    alignment_parameters: AlignmentParameters, report_settings: ReportSettings = ReportSettings()
) -> str:
    """
    Render a heatmap plot.

    Args:
        alignment_parameters (AlignmentParameters): Alignment parameters.
        report_settings (ReportSettings, optional): Report settings. Defaults to ReportSettings().

    Returns:
        HTML string of the rendered report including the heatmap plot.
    """

    def extract_value(param_string: str) -> str:
        return param_string.split("=")[1].split("s-dev")[0].strip()

    def extract_std(param_string: str) -> str:
        return param_string.split(":")[-1].strip()

    labels = alignment_parameters.params_labels(enabled_only=True, lower_case=False)
    alignment_data = [
        go.Table(
            header=dict(values=["Parameter", "Value", "Standard Deviation"]),
            cells=dict(
                values=[
                    labels,
                    [extract_value(param_string) for param_string in alignment_parameters.to_string_list()],
                    [extract_std(param_string) for param_string in alignment_parameters.to_string_list()],
                ],
            ),
            name="Alignment Parameters",
        )
    ]
    fig = go.Figure(data=alignment_data)
    return plot(fig, output_type="div", config=report_settings.single_plot_export.to_config())
