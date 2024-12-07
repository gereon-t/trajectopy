"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import List

import jinja2

from trajectopy.core.plotting.plotly import multi_line_plots, scatter_plots
from trajectopy.core.report.utils import TEMPLATES_PATH, convert_icon_to_base64
from trajectopy.core.settings.report import ReportSettings
from trajectopy.core.trajectory import Trajectory

logger = logging.getLogger("root")


def render_one_line_plots(
    trajectories: List[Trajectory], report_settings: ReportSettings = ReportSettings()
) -> List[str]:
    one_line_plots = [
        (
            scatter_plots.render_trajectories_mapbox(trajectories, report_settings)
            if report_settings.scatter_mapbox
            else scatter_plots.render_trajectories(trajectories, report_settings)
        )
    ]
    one_line_plots.append(multi_line_plots.render_pos_plot(trajectories, report_settings))

    if rot_trajectories := [traj for traj in trajectories if traj.has_orientation]:
        one_line_plots.append(multi_line_plots.render_rot_plot(rot_trajectories, report_settings))

    return one_line_plots


def create_trajectory_report(
    *, trajectories: List[Trajectory], report_settings: ReportSettings = ReportSettings()
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

    one_line_plots = render_one_line_plots(trajectories, report_settings)

    context = {
        "title": trajectories[0].name if len(trajectories) == 1 else "Trajectory Plot",
        "one_line_plots": one_line_plots,
        "icon": icon,
    }

    return template.render(context)
