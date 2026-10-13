"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

from typing import Any, Protocol


class Stamp(Protocol):
    sec: int
    nanosec: int


class Point(Protocol):
    x: float
    y: float
    z: float


class Quaternion(Protocol):
    x: float
    y: float
    z: float
    w: float


class Header(Protocol):
    stamp: Stamp
    frame_id: str


class Pose(Protocol):
    position: Point
    orientation: Quaternion


class PoseMessage(Protocol):
    header: Header
    pose: Pose


def geometry_pose_stamped_handler(msg: PoseMessage) -> dict[str, Any]:
    """Handler for geometry_msgs/msg/PoseStamped messages

    Args:
        msg (Any): Message to handle

    Returns:
        dict[str, Any]: Dictionary containing the data
    """
    return {
        "tstamps": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        "x": msg.pose.position.x,
        "y": msg.pose.position.y,
        "z": msg.pose.position.z,
        "qx": msg.pose.orientation.x,
        "qy": msg.pose.orientation.y,
        "qz": msg.pose.orientation.z,
        "qw": msg.pose.orientation.w,
    }
