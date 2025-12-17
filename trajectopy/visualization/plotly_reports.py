import base64
import logging
import os
import sys
import uuid
import webbrowser

import jinja2
import numpy as np
import pandas as pd

import trajectopy.visualization.plotly_plots as plotly_plots
from trajectopy.core.settings import ReportSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.lib.alignment.parameters import AlignmentParameters
from trajectopy.results.ate_result import ATEResult
from trajectopy.results.rpe_result import RPEResult
from trajectopy.utils.common import number_to_string
from trajectopy.visualization.report_data import (
    ATEReportData,
    ATEReportDataCollection,
    RPEReportData,
    RPEReportDataCollection,
)

logger = logging.getLogger(__name__)

try:
    BASE_PATH = os.path.join(sys._MEIPASS, "trajectopy/visualization")
except Exception:
    BASE_PATH = os.path.dirname(__file__)
TEMPLATES_PATH = os.path.join(BASE_PATH, "templates")


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def convert_icon_to_base64() -> str:
    icon_path = os.path.join(os.path.join(BASE_PATH), "assets", "icon.png")
    return image_to_base64(icon_path)


def _build_correlation_dataframe(estimated_parameters: AlignmentParameters, enabled_only: bool = True) -> pd.DataFrame:
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


def _render_one_line_alignment_plots(
    alignment_parameters: AlignmentParameters, report_settings: ReportSettings = ReportSettings()
) -> list[str]:
    dataframe = _build_correlation_dataframe(alignment_parameters, enabled_only=True)

    one_line_plots = [plotly_plots.plot_alignment_table(alignment_parameters, report_settings)]
    one_line_plots.append(plotly_plots.plot_heatmap(dataframe, report_settings))

    return one_line_plots


def _render_one_line_deviation_collection_plots(
    ate_report_data_collection: ATEReportDataCollection | None = None,
    rpe_report_data_collection: RPEReportDataCollection | None = None,
) -> list[str]:
    if ate_report_data_collection is None and rpe_report_data_collection is not None:
        return [rpe_report_data_collection.plot()]

    # only for type checking (this shouldnt be necessary but mypy is complaining otherwise)
    if ate_report_data_collection is None:
        return []

    one_line_plots = [
        ate_report_data_collection.plot_comb_dev(),
        ate_report_data_collection.plot_edf(),
        ate_report_data_collection.plot_pos_dev_bar(),
        ate_report_data_collection.plot_pos_dev_xyz(),
    ]

    if ate_report_data_collection.has_ate_rot:
        one_line_plots.append(ate_report_data_collection.plot_rot_dev_bar())
        one_line_plots.append(ate_report_data_collection.plot_rot_dev_xyz())

    if rpe_report_data_collection is not None:
        one_line_plots.insert(1, rpe_report_data_collection.plot())

    return one_line_plots


def _render_multi_deviations_report(
    *,
    ate_results: list[ATEResult] | None = None,
    rpe_results: list[RPEResult] | None = None,
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

    one_line_plots = _render_one_line_deviation_collection_plots(
        ate_report_data_collection, rpe_report_data_collection
    )

    context = {
        "title": "Trajectory Comparison",
        "rpe_available": rpe_results is not None,
        "one_line_plots": one_line_plots,
        "icon": icon,
    }

    return template.render(context)


def _render_side_by_side_deviations_plots(ate_report_data: ATEReportData) -> list[str]:
    side_by_side_plots = [ate_report_data.scatter_comb_pos_dev()]

    if not ate_report_data.has_ate_rot:
        return side_by_side_plots

    side_by_side_plots.append(ate_report_data.scatter_comb_rot_dev())

    return side_by_side_plots


def _render_one_line_deviations_plots(
    ate_report_data: ATEReportData | None = None, rpe_report_data: RPEReportData | None = None
) -> list[str]:
    one_line_plots = []

    if rpe_report_data is not None:
        one_line_plots.append(rpe_report_data.plot())

    if ate_report_data is None:
        return one_line_plots

    one_line_plots.extend(
        (
            ate_report_data.plot_pos_dev_hist(),
            ate_report_data.plot_pos_dev_bar(),
            ate_report_data.scatter_pos_dev_3d(),
            ate_report_data.plot_edf(),
            ate_report_data.plot_comb_dev(),
            ate_report_data.plot_pos_dev(),
        )
    )

    if ate_report_data.settings.scatter_show_individual_dofs:
        one_line_plots.extend(
            (
                ate_report_data.scatter_pos_x_dev(),
                ate_report_data.scatter_pos_y_dev(),
                ate_report_data.scatter_pos_z_dev(),
            )
        )
    if not ate_report_data.has_ate_rot:
        return one_line_plots

    one_line_plots.insert(2, ate_report_data.plot_rot_dev_hist())
    one_line_plots.insert(4, ate_report_data.plot_rot_dev_bar())
    one_line_plots.append(ate_report_data.plot_rot_dev())

    if ate_report_data.settings.scatter_show_individual_dofs:
        one_line_plots.extend(
            (
                ate_report_data.scatter_rot_x_dev(),
                ate_report_data.scatter_rot_y_dev(),
                ate_report_data.scatter_rot_z_dev(),
            )
        )
    return one_line_plots


def _render_single_deviations_report(
    *,
    ate_result: ATEResult | None = None,
    rpe_result: RPEResult | None = None,
    report_settings: ReportSettings = ReportSettings(),
) -> str:
    """
    Renders a html report string of a single trajectory comparison.

    Args:
        ate_result (ATEResult): The absolute trajectory error result
        rpe_result (Optional[RPEResult]): The relative pose error result
        max_std (float): The upper bound of scatter plot colorbars is set to max_std * std of the data
        report_settings (ReportSettings): The report settings

    Returns:
        str: The html report string

    """
    if ate_result is None and rpe_result is None:
        raise ValueError("Either ate_result or rpe_result must be provided.")

    template = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_PATH)).get_template("single_template.html")
    icon = convert_icon_to_base64()

    ate_report_data = (
        ATEReportData(ate_result=ate_result, settings=report_settings) if ate_result is not None else None
    )
    rpe_report_data = (
        RPEReportData(rpe_result=rpe_result, settings=report_settings) if rpe_result is not None else None
    )

    side_by_side_plots = _render_side_by_side_deviations_plots(ate_report_data) if ate_report_data is not None else []
    one_line_plots = _render_one_line_deviations_plots(ate_report_data, rpe_report_data)

    if len(side_by_side_plots) == 1:
        one_line_plots = side_by_side_plots + one_line_plots
        side_by_side_plots = []

    context = {
        "title": ate_result.name if ate_result is not None else rpe_result.name,
        "ate_pos": number_to_string(ate_result.pos_ate) if ate_result is not None else "-",
        "ate_rot": number_to_string(np.rad2deg(ate_result.rot_ate)) if ate_result is not None else "-",
        "rpe_pos": number_to_string(rpe_result.pos_rpe) if rpe_result is not None else "-",
        "rpe_rot": number_to_string(np.rad2deg(rpe_result.rot_rpe)) if rpe_result is not None else "-",
        "rpe_pos_drift_unit": rpe_result.pos_drift_unit if rpe_result is not None else "-",
        "rpe_rot_drift_unit": rpe_result.rot_drift_unit if rpe_result is not None else "-",
        "ate_pos_unit": "mm" if report_settings.ate_unit_is_mm else "m",
        "rpe_available": rpe_result is not None,
        "side_by_side_plots": side_by_side_plots,
        "one_line_plots": one_line_plots,
        "icon": icon,
        "rot_unit": report_settings.rot_unit,
    }

    return template.render(context)


def _render_one_line_trajectory_plots(
    trajectories: list[Trajectory], report_settings: ReportSettings = ReportSettings()
) -> list[str]:
    one_line_plots = [
        (
            plotly_plots.scatter_trajectories_mapbox(trajectories, report_settings)
            if report_settings.scatter_plot_on_map
            else plotly_plots.scatter_trajectories(trajectories, report_settings)
        )
    ]
    one_line_plots.append(plotly_plots.plot_trajectories_xyz(trajectories, report_settings))

    if rot_trajectories := [traj for traj in trajectories if traj.has_orientation]:
        one_line_plots.append(plotly_plots.plot_trajectories_rpy(rot_trajectories, report_settings))

    return one_line_plots


def create_alignment_report(
    *,
    alignment_parameters: AlignmentParameters,
    name: str = "Alignment",
    report_settings: ReportSettings = ReportSettings(),
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

    one_line_plots = _render_one_line_alignment_plots(alignment_parameters, report_settings)

    context = {
        "title": name,
        "one_line_plots": one_line_plots,
        "icon": icon,
    }

    return template.render(context)


def create_trajectory_report(
    *, trajectories: list[Trajectory], report_settings: ReportSettings = ReportSettings()
) -> str:
    """
    Render a HTML report containing trajectory plots.

    Args:
        trajectories: List of trajectories to render.
        report_settings: Report settings.

    Returns:
        HTML string of the rendered report including the trajectory plots.
    """

    template = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_PATH)).get_template("generic.html")
    icon = convert_icon_to_base64()

    one_line_plots = _render_one_line_trajectory_plots(trajectories, report_settings)

    context = {
        "title": trajectories[0].name if len(trajectories) == 1 else "Trajectory Plot",
        "one_line_plots": one_line_plots,
        "icon": icon,
    }

    return template.render(context)


def create_deviation_report(
    ate_result: ATEResult | list[ATEResult],
    rpe_result: RPEResult | list[RPEResult] | None,
    report_settings: ReportSettings = ReportSettings(),
) -> str:
    """Create a HTML deviation report containing the absolute trajectory error and relative pose error results.

    Args:
        ate_result (Union[ATEResult, list[ATEResult]]): The absolute trajectory error results (one or multiple).
        rpe_result (Union[RPEResult, list[RPEResult]]): The relative pose error results (one or multiple).
        report_settings (ReportSettings): The report settings, defaults to ReportSettings().

    Returns:
        str: The deviation report.
    """
    if (ate_result is not None and isinstance(ate_result, list)) or (
        rpe_result is not None and isinstance(rpe_result, list)
    ):
        return _render_multi_deviations_report(
            ate_results=ate_result, rpe_results=rpe_result, report_settings=report_settings
        )

    return _render_single_deviations_report(
        ate_result=ate_result, rpe_result=rpe_result, report_settings=report_settings
    )


def write_report(*, output_file: str, report_text: str) -> None:
    """
    Writes a report to the given output file.

    Args:
        output_file (str): The output file path
        report_text (str): The report text

    """
    logger.info("Writing report to %s", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)


def show_report(report_text: str, filepath: str = "") -> None:
    """
    This function writes a report to a file and opens it in the default web browser.

    Args:
        report_text (str): The report text
        filepath (str, optional): The file path to save the report. If not given, a random file name will be generated.

    """
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    random_string = uuid.uuid4().hex

    file = filepath or os.path.join(dirname, f"{random_string}.html")

    with open(file, "w", encoding="utf-8") as f:
        f.write(report_text)
        url = f"file://{os.path.realpath(f.name)}"
        webbrowser.open(url)
