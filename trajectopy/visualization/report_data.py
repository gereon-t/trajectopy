import itertools
import logging
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from trajectopy.core.settings import ReportSettings
from trajectopy.results.ate_result import ATEResult
from trajectopy.results.rpe_result import RPEResult
from trajectopy.visualization.plotly_plots import (
    plot_subplots_with_shared_x_axis,
    scatter,
)

logger = logging.getLogger(__name__)


def _add_to_dict(metrics: dict, field_name: str, content: list):
    current_content = metrics.get(field_name) or []
    current_content.extend(content)
    metrics[field_name] = current_content


@dataclass
class ATEReportData:
    """
    Class to store all ATE data needed to render the report.

    Args:
        ate_result: The ATE result to be rendered.
        settings: The report settings.

    """

    ate_result: ATEResult
    settings: ReportSettings = field(default_factory=ReportSettings)

    def __post_init__(self) -> None:
        if self.settings.ate_unit_is_mm:
            self.ate_result.abs_dev.pos_dev *= 1000.0
            self.ate_result.abs_dev.directed_pos_dev *= 1000.0

        if self.settings.scatter_plot_on_map:
            self.ate_result.trajectory.positions.to_epsg(4326, inplace=True)

    @property
    def short_name(self) -> str:
        return self.ate_result.name.split("vs")[0]

    @property
    def ate_unit(self) -> str:
        return "mm" if self.settings.ate_unit_is_mm else "m"

    @property
    def has_ate_rot(self) -> bool:
        return self.ate_result.has_orientation

    @property
    def function_of_label(self) -> str:
        return self.ate_result.trajectory.index_label

    @cached_property
    def pos_x(self) -> np.ndarray:
        return self.ate_result.trajectory.xyz[:, 0]

    @cached_property
    def pos_y(self) -> np.ndarray:
        return self.ate_result.trajectory.xyz[:, 1]

    @cached_property
    def pos_z(self) -> np.ndarray:
        return self.ate_result.trajectory.xyz[:, 2]

    @cached_property
    def index(self) -> np.ndarray:
        return (
            self.ate_result.trajectory.datetimes if self.ate_result.trajectory.is_unix_time else self.ate_result.index
        )

    @cached_property
    def comb_dev_pos(self) -> np.ndarray:
        return self.ate_result.pos_dev_comb

    @cached_property
    def pos_dev_x(self) -> np.ndarray:
        return self.ate_result.pos_dev_along if self.settings.directed_ate else self.ate_result.pos_dev_x

    @cached_property
    def pos_dev_y(self) -> np.ndarray:
        return self.ate_result.pos_dev_cross_h if self.settings.directed_ate else self.ate_result.pos_dev_y

    @cached_property
    def pos_dev_z(self) -> np.ndarray:
        return self.ate_result.pos_dev_cross_v if self.settings.directed_ate else self.ate_result.pos_dev_z

    @property
    def pos_dev_x_name(self) -> str:
        return self.settings.directed_pos_dev_x_name if self.settings.directed_ate else self.settings.pos_x_name

    @property
    def pos_dev_y_name(self) -> str:
        return self.settings.directed_pos_dev_y_name if self.settings.directed_ate else self.settings.pos_y_name

    @property
    def pos_dev_z_name(self) -> str:
        return self.settings.directed_pos_dev_z_name if self.settings.directed_ate else self.settings.pos_z_name

    @cached_property
    def roll(self) -> np.ndarray:
        if not self.ate_result.has_orientation:
            raise ValueError("ATE result has no orientation.")

        return np.rad2deg(self.ate_result.trajectory.rpy[:, 0])

    @cached_property
    def pitch(self) -> np.ndarray:
        if not self.ate_result.has_orientation:
            raise ValueError("ATE result has no orientation.")

        return np.rad2deg(self.ate_result.trajectory.rpy[:, 1])

    @cached_property
    def yaw(self) -> np.ndarray:
        if not self.ate_result.has_orientation:
            raise ValueError("ATE result has no orientation.")

        return np.rad2deg(self.ate_result.trajectory.rpy[:, 2])

    @cached_property
    def comb_dev_rot(self) -> np.ndarray:
        if not self.ate_result.has_orientation:
            raise ValueError("ATE result has no orientation.")

        return np.rad2deg(self.ate_result.rot_dev_comb)

    @cached_property
    def rot_dev_x(self) -> np.ndarray:
        if not self.ate_result.has_orientation:
            raise ValueError("ATE result has no orientation.")

        return np.rad2deg(self.ate_result.rot_dev_x)

    @cached_property
    def rot_dev_y(self) -> np.ndarray:
        if not self.ate_result.has_orientation:
            raise ValueError("ATE result has no orientation.")

        return np.rad2deg(self.ate_result.rot_dev_y)

    @cached_property
    def rot_dev_z(self) -> np.ndarray:
        if not self.ate_result.has_orientation:
            raise ValueError("ATE result has no orientation.")

        return np.rad2deg(self.ate_result.rot_dev_z)

    def plot_pos_dev_bar(self) -> str:
        return ATEReportDataCollection([self]).plot_pos_dev_bar()

    def plot_rot_dev_bar(self) -> str:
        return ATEReportDataCollection([self]).plot_rot_dev_bar()

    def plot_pos_dev_hist(self) -> str:
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=self.pos_dev_x,
                name=self.pos_dev_x_name,
                opacity=self.settings.histogram_opacity,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=self.pos_dev_y,
                name=self.pos_dev_y_name,
                opacity=self.settings.histogram_opacity,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=self.pos_dev_z,
                name=self.pos_dev_z_name,
                opacity=self.settings.histogram_opacity,
            )
        )

        fig.update_layout(
            title="Position Deviations",
            xaxis=dict(title=f"Absolute Position Error [{self.ate_unit}]"),
            yaxis=dict(title=self.settings.histogram_yaxis_title),
            barmode=self.settings.histogram_barmode,
            bargap=self.settings.histogram_bargap,
            height=self.settings.single_plot_height,
        )
        return plot(fig, output_type="div", config=self.settings.single_plot_export.to_config())

    def plot_rot_dev_hist(self) -> str:
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=self.rot_dev_x,
                name=self.settings.rot_x_name,
                opacity=self.settings.histogram_opacity,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=self.rot_dev_y,
                name=self.settings.rot_y_name,
                opacity=self.settings.histogram_opacity,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=self.rot_dev_z,
                name=self.settings.rot_z_name,
                opacity=self.settings.histogram_opacity,
            )
        )

        fig.update_layout(
            title="Rotation Deviations",
            xaxis=dict(title=f"Absolute Rotation Error [{self.settings.rot_unit}]"),
            yaxis=dict(title=self.settings.histogram_yaxis_title),
            barmode=self.settings.histogram_barmode,
            bargap=self.settings.histogram_bargap,
            height=self.settings.single_plot_height,
        )

        return plot(fig, output_type="div", config=self.settings.single_plot_export.to_config())

    def plot_edf(self) -> str:
        if self.has_ate_rot:
            fig = make_subplots(rows=2, cols=1)
        else:
            fig = make_subplots(rows=1, cols=1)

        sorted_comb_pos_dev = np.sort(self.comb_dev_pos)
        pos_norm_cdf = np.arange(len(sorted_comb_pos_dev)) / float(len(sorted_comb_pos_dev))
        fig.add_trace(
            go.Scattergl(
                x=sorted_comb_pos_dev,
                y=pos_norm_cdf,
                mode=self.settings.plot_mode,
                name=f"{self.ate_result.name} position",
            ),
            row=1,
            col=1,
        )

        if self.has_ate_rot:
            sorted_comb_rot_dev = np.sort(self.comb_dev_rot)
            rot_norm_cdf = np.arange(len(sorted_comb_rot_dev)) / float(len(sorted_comb_rot_dev))
            fig.add_trace(
                go.Scattergl(
                    x=sorted_comb_rot_dev,
                    y=rot_norm_cdf,
                    mode=self.settings.plot_mode,
                    name=f"{self.ate_result.name} rotation",
                ),
                row=2,
                col=1,
            )
            fig.update_xaxes(title_text=f"[{self.settings.rot_unit}]", row=2, col=1)
            fig.update_yaxes(title_text="CDF", row=2, col=1)

        if self.has_ate_rot:
            height = self.settings.two_subplots_height
            config = self.settings.two_subplots_export.to_config()
        else:
            height = self.settings.single_plot_height
            config = self.settings.single_plot_export.to_config()

        fig.update_layout(title="Cumulative Probability", height=height)
        fig.update_xaxes(title_text=f"[{self.ate_unit}]", row=1, col=1)
        fig.update_yaxes(title_text="CDF", row=1, col=1)

        return plot(fig, output_type="div", config=config)

    def plot_pos(self) -> str:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.pos_x,
                mode=self.settings.plot_mode,
                name=self.settings.pos_x_name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.pos_y,
                mode=self.settings.plot_mode,
                name=self.settings.pos_y_name,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.pos_z,
                mode=self.settings.plot_mode,
                name=self.settings.pos_z_name,
            ),
            row=3,
            col=1,
        )

        fig.update_layout(title="Position Components", height=self.settings.three_subplots_height)

        fig.update_xaxes(title_text=self.function_of_label, row=3, col=1)
        fig.update_yaxes(title_text=f"[{self.settings.pos_x_unit}]", row=1, col=1)
        fig.update_yaxes(title_text=f"[{self.settings.pos_y_unit}]", row=2, col=1)
        fig.update_yaxes(title_text=f"[{self.settings.pos_z_unit}]", row=3, col=1)

        return plot(fig, output_type="div", config=self.settings.three_subplots_export.to_config())

    def plot_rot(self) -> str:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.roll,
                mode=self.settings.plot_mode,
                name=self.settings.rot_x_name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.pitch,
                mode=self.settings.plot_mode,
                name=self.settings.rot_y_name,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.yaw,
                mode=self.settings.plot_mode,
                name=self.settings.rot_z_name,
            ),
            row=3,
            col=1,
        )

        fig.update_layout(title="Rotation Components", height=self.settings.three_subplots_height)

        fig.update_xaxes(title_text=self.function_of_label, row=3, col=1)
        fig.update_yaxes(title_text=f"[{self.settings.rot_unit}]", row=1, col=1)
        fig.update_yaxes(title_text=f"[{self.settings.rot_unit}]", row=2, col=1)
        fig.update_yaxes(title_text=f"[{self.settings.rot_unit}]", row=3, col=1)

        return plot(fig, output_type="div", config=self.settings.three_subplots_export.to_config())

    def plot_pos_dev(self) -> str:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.pos_dev_x,
                mode=self.settings.plot_mode,
                name=self.pos_dev_x_name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.pos_dev_y,
                mode=self.settings.plot_mode,
                name=self.pos_dev_y_name,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.pos_dev_z,
                mode=self.settings.plot_mode,
                name=self.pos_dev_z_name,
            ),
            row=3,
            col=1,
        )

        fig.update_layout(title="Position Deviations per Direction", height=self.settings.three_subplots_height)

        fig.update_xaxes(title_text=self.function_of_label, row=3, col=1)
        fig.update_yaxes(title_text=f"[{self.ate_unit}]", row=1, col=1)
        fig.update_yaxes(title_text=f"[{self.ate_unit}]", row=2, col=1)
        fig.update_yaxes(title_text=f"[{self.ate_unit}]", row=3, col=1)

        return plot(fig, output_type="div", config=self.settings.three_subplots_export.to_config())

    def plot_rot_dev(self) -> str:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.rot_dev_x,
                mode=self.settings.plot_mode,
                name=self.settings.rot_x_name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.rot_dev_y,
                mode=self.settings.plot_mode,
                name=self.settings.rot_y_name,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.rot_dev_z,
                mode=self.settings.plot_mode,
                name=self.settings.rot_z_name,
            ),
            row=3,
            col=1,
        )

        fig.update_layout(title="Rotation Deviations per Axis", height=self.settings.three_subplots_height)
        fig.update_xaxes(title_text=self.function_of_label, row=3, col=1)
        fig.update_yaxes(title_text=f"[{self.settings.rot_unit}]", row=1, col=1)
        fig.update_yaxes(title_text=f"[{self.settings.rot_unit}]", row=2, col=1)
        fig.update_yaxes(title_text=f"[{self.settings.rot_unit}]", row=3, col=1)

        return plot(fig, output_type="div", config=self.settings.three_subplots_export.to_config())

    def plot_comb_dev(self) -> str:
        if self.has_ate_rot:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            config = self.settings.two_subplots_export.to_config()
            height = self.settings.two_subplots_height
        else:
            fig = make_subplots(rows=1, cols=1)
            config = self.settings.single_plot_export.to_config()
            height = self.settings.single_plot_height

        fig.add_trace(
            go.Scattergl(
                x=self.index,
                y=self.comb_dev_pos,
                mode=self.settings.plot_mode,
                name="position",
            ),
            row=1,
            col=1,
        )

        if self.has_ate_rot:
            fig.add_trace(
                go.Scattergl(
                    x=self.index,
                    y=self.comb_dev_rot,
                    mode=self.settings.plot_mode,
                    name="rotation",
                ),
                row=2,
                col=1,
            )
            fig.update_yaxes(title_text=f"[{self.settings.rot_unit}]", row=2, col=1)

        fig.update_layout(title="Trajectory Deviations", height=height)

        fig.update_xaxes(title_text=self.function_of_label, row=2 if self.has_ate_rot else 1, col=1)
        fig.update_yaxes(title_text=f"[{self.ate_unit}]", row=1, col=1)

        return plot(fig, output_type="div", config=config)

    def scatter_pos_dev_3d(self) -> str:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=self.ate_result.pos_dev_x,
                y=self.ate_result.pos_dev_y,
                z=self.ate_result.pos_dev_z,
                mode=self.settings.scatter_mode,
                marker=dict(size=self.settings.scatter_marker_size),
            )
        )

        fig.update_layout(
            title="ATE 3D Plot",
            height=self.settings.single_plot_height,
            hovermode="closest",
            autosize=True,
            scene=dict(
                aspectmode="data",
                xaxis=dict(title=f"{self.pos_dev_x_name} [{self.ate_unit}]"),
                yaxis=dict(title=f"{self.pos_dev_y_name} [{self.ate_unit}]"),
                zaxis=dict(title=f"{self.pos_dev_z_name} [{self.ate_unit}]"),
            ),
        )

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

        return plot(fig, output_type="div", config=self.settings.single_plot_export.to_config())

    def scatter_comb_pos_dev(self) -> str:
        comb_pos_devs = self.comb_dev_pos

        return scatter(
            pos=np.c_[self.pos_x, self.pos_y, self.pos_z],
            colors=comb_pos_devs,
            report_settings=self.settings,
            figure_title="Position Deviations",
            colorbar_title=f"[{self.ate_unit}]",
        )

    def scatter_pos_x_dev(self) -> str:
        return scatter(
            pos=np.c_[self.pos_x, self.pos_y, self.pos_z],
            colors=self.pos_dev_x,
            report_settings=self.settings,
            figure_title=f"{self.pos_dev_x_name.capitalize()} Deviations",
            colorbar_title=f"{self.settings.pos_x_name} [{self.ate_unit}]",
        )

    def scatter_pos_y_dev(self) -> str:
        return scatter(
            pos=np.c_[self.pos_x, self.pos_y, self.pos_z],
            colors=self.pos_dev_y,
            report_settings=self.settings,
            figure_title=f"{self.pos_dev_y_name.capitalize()} Deviations",
            colorbar_title=f"{self.settings.pos_y_name} [{self.ate_unit}]",
        )

    def scatter_pos_z_dev(self) -> str:
        return scatter(
            pos=np.c_[self.pos_x, self.pos_y, self.pos_z],
            colors=self.pos_dev_z,
            report_settings=self.settings,
            figure_title=f"{self.pos_dev_z_name.capitalize()} Deviations",
            colorbar_title=f"{self.settings.pos_z_name} [{self.ate_unit}]",
        )

    def scatter_rot_x_dev(self) -> str:
        return scatter(
            pos=np.c_[self.pos_x, self.pos_y, self.pos_z],
            colors=self.rot_dev_x,
            report_settings=self.settings,
            figure_title=f"{self.settings.rot_x_name.capitalize()} Deviations",
            colorbar_title=f"{self.settings.rot_x_name} [{self.settings.rot_unit}]",
        )

    def scatter_rot_y_dev(self) -> str:
        return scatter(
            pos=np.c_[self.pos_x, self.pos_y, self.pos_z],
            colors=self.rot_dev_y,
            report_settings=self.settings,
            figure_title=f"{self.settings.rot_y_name.capitalize()} Deviations",
            colorbar_title=f"{self.settings.rot_y_name} [{self.settings.rot_unit}]",
        )

    def scatter_rot_z_dev(self) -> str:
        return scatter(
            pos=np.c_[self.pos_x, self.pos_y, self.pos_z],
            colors=self.rot_dev_z,
            report_settings=self.settings,
            figure_title=f"{self.settings.rot_z_name.capitalize()} Deviations",
            colorbar_title=f"{self.settings.rot_z_name} [{self.settings.rot_unit}]",
        )

    def scatter_comb_rot_dev(self) -> str:
        comb_rot_devs = self.comb_dev_rot

        return scatter(
            pos=np.c_[self.pos_x, self.pos_y, self.pos_z],
            colors=comb_rot_devs,
            report_settings=self.settings,
            figure_title="Rotation Deviations",
            colorbar_title=f"[{self.settings.rot_unit}]",
        )


@dataclass
class RPEReportData:
    """
    Class to store all RPE data needed to render the report.

    Args:
        rpe_result: The RPE result to be rendered.
        settings: The report settings.

    """

    rpe_result: RPEResult
    settings: ReportSettings = field(default_factory=ReportSettings)

    @property
    def short_name(self) -> str:
        return self.rpe_result.name.split("vs")[0]

    def plot(self) -> str:
        rpe_result = self.rpe_result
        if rpe_result is None:
            return ""

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_trace(
            go.Scattergl(
                x=rpe_result.mean_pair_distances,
                y=rpe_result.pos_dev_mean,
                mode=self.settings.plot_mode,
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
                    mode=self.settings.plot_mode,
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

            height = self.settings.two_subplots_height
            config = self.settings.two_subplots_export.to_config()
        else:
            height = self.settings.single_plot_height
            config = self.settings.single_plot_export.to_config()

        fig.update_layout(title="Relative Pose Error", height=height)
        fig.update_yaxes(title_text=f"[{rpe_result.pos_drift_unit}]", row=1, col=1)
        fig.update_xaxes(
            title_text=f"Pose Distance [{rpe_result.pair_distance_unit}]",
            row=2 if rpe_result.has_rot_dev else 1,
            col=1,
        )

        return plot(fig, output_type="div", config=config)


@dataclass
class ATEReportDataCollection:
    """
    Class to store multiple ReportData objects in a list
    """

    items: list[ATEReportData]

    @property
    def has_ate_rot(self) -> bool:
        return any(item.has_ate_rot for item in self.items)

    def get_ate_results(self, rot_required: bool = False) -> list[ATEResult]:
        return [item.ate_result for item in self.items if not rot_required or item.has_ate_rot]

    def _to_pos_metrics_df(self) -> pd.DataFrame:
        metrics = {}

        for data in self.items:
            _add_to_dict(metrics, "Trajectory", [data.short_name] * 6)
            _add_to_dict(metrics, "Metric", ["ATE", "ATE Min", "ATE Max", "ATE Median", "ATE RMS", "ATE Std"])
            _add_to_dict(
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

    def _to_rot_metrics_df(self) -> pd.DataFrame:
        metrics = {}

        if not self.has_ate_rot:
            return pd.DataFrame(metrics)

        for data in self.items:
            if not data.has_ate_rot:
                continue

            _add_to_dict(metrics, "Trajectory", [data.short_name] * 6)
            _add_to_dict(metrics, "Metric", ["ATE", "ATE Min", "ATE Max", "ATE Median", "ATE RMS", "ATE Std"])
            _add_to_dict(
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

    def _setup_edf_axis(self) -> tuple[go.Figure, dict]:
        report_data_item = self.items[0]
        if self.has_ate_rot:
            fig = make_subplots(rows=2, cols=1)
            height = report_data_item.settings.two_subplots_height
            config = report_data_item.settings.two_subplots_export.to_config()

            fig.update_xaxes(title_text=f"[{report_data_item.settings.rot_unit}]", row=2, col=1)
            fig.update_yaxes(title_text="CDF", row=2, col=1)
        else:
            fig = make_subplots(rows=1, cols=1)
            height = report_data_item.settings.single_plot_height
            config = report_data_item.settings.single_plot_export.to_config()

        fig.update_layout(title="Cumulative Probability", height=height)
        fig.update_xaxes(title_text=f"[{report_data_item.ate_unit}]", row=1, col=1)
        fig.update_yaxes(title_text="CDF", row=1, col=1)

        return fig, config

    def _setup_dev_comb_axis(self) -> tuple[go.Figure, dict]:
        report_data_item = self.items[0]
        if self.has_ate_rot:
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

    def plot_pos_dev_bar(self) -> str:
        metrics_df = self._to_pos_metrics_df()

        fig = px.bar(metrics_df, barmode="group", x="Metric", y="Value", color="Trajectory")

        fig.update_layout(
            title_text="Absolute Trajectory Error (ATE) - Position",
            height=self.items[0].settings.single_plot_height,
        )
        fig.update_yaxes(title_text=f"Value [{self.items[0].ate_unit}]")
        return plot(fig, output_type="div", config=self.items[0].settings.single_plot_export.to_config())

    def plot_rot_dev_bar(self) -> str:
        metrics_df = self._to_rot_metrics_df()

        fig = px.bar(metrics_df, barmode="group", x="Metric", y="Value", color="Trajectory")

        fig.update_layout(
            title_text="Absolute Trajectory Error (ATE) - Rotation",
            height=self.items[0].settings.single_plot_height,
        )
        fig.update_yaxes(title_text=f"Value [{self.items[0].settings.rot_unit}]")
        return plot(fig, output_type="div", config=self.items[0].settings.single_plot_export.to_config())

    def plot_edf(self) -> str:
        fig, config = self._setup_edf_axis()

        for data, color in zip(self.items, itertools.cycle(px.colors.qualitative.Plotly)):
            sorted_comb_pos_dev = np.sort(data.comb_dev_pos)
            pos_norm_cdf = np.arange(len(sorted_comb_pos_dev)) / float(len(sorted_comb_pos_dev))
            fig.add_trace(
                go.Scattergl(
                    x=sorted_comb_pos_dev,
                    y=pos_norm_cdf,
                    mode=data.settings.plot_mode,
                    name=f"{data.short_name} Position",
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
                        name=f"{data.short_name} Rotation",
                        marker=dict(color=color),
                    ),
                    row=2,
                    col=1,
                )
        return plot(fig, output_type="div", config=config)

    def plot_comb_dev(self) -> str:
        report_data = self.items[0]

        any_rot_available = any(data.has_ate_rot for data in self.items)

        y_data = (
            [[data.comb_dev_pos, data.comb_dev_rot if data.has_ate_rot else None] for data in self.items]
            if any_rot_available
            else [[data.comb_dev_pos] for data in self.items]
        )

        y_labels = (
            [f"[{report_data.ate_unit}]", f"[{report_data.settings.rot_unit}]"]
            if any_rot_available
            else [f"[{report_data.ate_unit}]"]
        )

        return plot_subplots_with_shared_x_axis(
            x_data=[data.index for data in self.items],
            y_data=y_data,
            names=[data.short_name for data in self.items],
            x_label=report_data.function_of_label,
            y_labels=y_labels,
            title="Trajectory Deviations",
            report_settings=report_data.settings,
        )

    def plot_pos_dev_xyz(self) -> str:
        report_data = self.items[0]

        return plot_subplots_with_shared_x_axis(
            x_data=[data.index for data in self.items],
            y_data=[[data.pos_dev_x, data.pos_dev_y, data.pos_dev_z] for data in self.items],
            names=[data.short_name for data in self.items],
            x_label=report_data.function_of_label,
            y_labels=[
                f"{report_data.pos_dev_x_name} [{report_data.ate_unit}]",
                f"{report_data.pos_dev_y_name} [{report_data.ate_unit}]",
                f"{report_data.pos_dev_z_name} [{report_data.ate_unit}]",
            ],
            title="Position Deviations per Direction",
            report_settings=report_data.settings,
        )

    def plot_rot_dev_xyz(self) -> str:
        report_data = self.items[0]

        return plot_subplots_with_shared_x_axis(
            x_data=[data.index for data in self.items],
            y_data=[[data.rot_dev_x, data.rot_dev_y, data.rot_dev_z] for data in self.items if data.has_ate_rot],
            names=[data.short_name for data in self.items],
            x_label=report_data.function_of_label,
            y_labels=[
                f"{report_data.settings.rot_x_name} [{report_data.settings.rot_unit}]",
                f"{report_data.settings.rot_y_name} [{report_data.settings.rot_unit}]",
                f"{report_data.settings.rot_z_name} [{report_data.settings.rot_unit}]",
            ],
            title="Rotation Deviations per Direction",
            report_settings=report_data.settings,
        )


@dataclass
class RPEReportDataCollection:
    """
    Class to store multiple ReportData objects in a list
    """

    items: list[RPEReportData]

    @property
    def has_rpe(self) -> bool:
        return any(item is not None for item in self.items)

    @property
    def has_rpe_rot(self) -> bool:
        if not self.has_rpe:
            return False

        rpe_results = [item.rpe_result for item in self.items]
        return any(result.has_rot_dev for result in rpe_results)

    def get_rpe_results(self, rot_required: bool = False) -> list[RPEResult]:
        return [item.rpe_result for item in self.items if not rot_required or item.rpe_result.has_rot_dev]

    def _setup_rpe_axis(self) -> tuple[go.Figure, dict]:
        report_data_item = self.items[0]
        if self.has_rpe_rot:
            rpe_rot_item = self.get_rpe_results(rot_required=True)[0]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            height = report_data_item.settings.two_subplots_height
            config = report_data_item.settings.two_subplots_export.to_config()
            fig.update_xaxes(title_text=f"Pose Distance [{rpe_rot_item.pair_distance_unit}]", row=2, col=1)
            fig.update_yaxes(title_text=f"[{rpe_rot_item.rot_drift_unit}]", row=2, col=1)

        else:
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
            height = report_data_item.settings.single_plot_height
            config = report_data_item.settings.single_plot_export.to_config()
            fig.update_xaxes(
                title_text=f"Pose Distance [{report_data_item.rpe_result.pair_distance_unit}]", row=1, col=1
            )

        fig.update_layout(title="Relative Pose Error", height=height)
        fig.update_yaxes(title_text=f"[{report_data_item.rpe_result.pos_drift_unit}]", row=1, col=1)

        return fig, config

    def plot(self) -> str:
        fig, config = self._setup_rpe_axis()

        for data, color in zip(self.items, itertools.cycle(px.colors.qualitative.Plotly)):
            rpe_result = data.rpe_result

            fig.add_trace(
                go.Scattergl(
                    x=rpe_result.mean_pair_distances,
                    y=rpe_result.pos_dev_mean,
                    mode=data.settings.plot_mode,
                    name=f"{data.short_name} [{data.rpe_result.pos_drift_unit}]",
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
                        name=f"{data.short_name} [{data.rpe_result.rot_drift_unit}]",
                        error_y=dict(
                            type="data",
                            array=np.rad2deg(rpe_result.rot_std),
                            visible=True,
                        ),
                        marker=dict(color=color),
                    ),
                    row=2,
                    col=1,
                )

        return plot(fig, output_type="div", config=config)
