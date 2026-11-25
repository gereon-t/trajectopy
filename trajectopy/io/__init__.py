"""Reading and writing of trajectories, evaluation results and alignment parameters.

This package is the only place that knows about file formats. Domain classes such as
:class:`~trajectopy.core.trajectory.Trajectory` expose ``from_file``/``to_file`` as thin
convenience wrappers around the functions defined here.
"""

from typing import TYPE_CHECKING

from trajectopy.io.formats import (
    TrajectoryFormat,
    detect_format,
    read_euroc,
    read_kitti,
    read_trajectory,
    read_tum,
    write_trajectory,
)
from trajectopy.io.header import HeaderData
from trajectopy.io.kml import write_ate_kml, write_kml
from trajectopy.io.results_io import (
    read_alignment_parameters,
    read_alignment_result,
    read_ate_result,
    read_rpe_result,
    read_sensor_rotation_parameters,
    write_alignment_parameters,
    write_alignment_result,
    write_ate_result,
    write_rpe_result,
    write_sensor_rotation_parameters,
)
from trajectopy.io.trajectory_io import read_trajectory_string, trajectory_to_string

if TYPE_CHECKING:
    from trajectopy.io.rosbag import list_topics, trajectories_from_rosbag

__all__ = [
    "HeaderData",
    "TrajectoryFormat",
    "detect_format",
    "read_trajectory",
    "read_tum",
    "read_kitti",
    "read_euroc",
    "list_topics",
    "trajectories_from_rosbag",
    "read_trajectory_string",
    "trajectory_to_string",
    "write_trajectory",
    "write_kml",
    "write_ate_kml",
    "read_ate_result",
    "write_ate_result",
    "read_rpe_result",
    "write_rpe_result",
    "read_alignment_result",
    "write_alignment_result",
    "read_alignment_parameters",
    "write_alignment_parameters",
    "read_sensor_rotation_parameters",
    "write_sensor_rotation_parameters",
]


def __getattr__(name: str):
    # ROS bag support is imported on first use, since the rosbags package takes a while to import
    if name in ("list_topics", "trajectories_from_rosbag"):
        from trajectopy.io import rosbag

        return getattr(rosbag, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
