"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import logging
from typing import List

from trajectopy.core.plotting.plotly import multi_line_plots, scatter_plots
from trajectopy.settings import ReportSettings
from trajectopy.trajectory import Trajectory

logger = logging.getLogger("root")


def render_one_line_trajectory_plots(
    trajectories: List[Trajectory], report_settings: ReportSettings = ReportSettings()
) -> List[str]:
    one_line_plots = [
        (
            scatter_plots.render_trajectories_mapbox(trajectories, report_settings)
            if report_settings.scatter_plot_on_map
            else scatter_plots.render_trajectories(trajectories, report_settings)
        )
    ]
    one_line_plots.append(multi_line_plots.render_pos_plot(trajectories, report_settings))

    if rot_trajectories := [traj for traj in trajectories if traj.has_orientation]:
        one_line_plots.append(multi_line_plots.render_rot_plot(rot_trajectories, report_settings))

    return one_line_plots
