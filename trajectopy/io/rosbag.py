import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.io.rosmsg import Header, Point, geometry_pose_stamped_handler

typestore = get_typestore(Stores.LATEST)


logger = logging.getLogger(__name__)


class PointMessage(Protocol):
    header: Header
    point: Point


def geometry_point_stamped_handler(msg: PointMessage) -> dict[str, Any]:
    """Handler for geometry_msgs/msg/PointStamped messages

    Args:
        msg (Any): Message to handle

    Returns:
        dict[str, Any]: Dictionary containing the data
    """
    return {
        "tstamps": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        "x": msg.point.x,
        "y": msg.point.y,
        "z": msg.point.z,
    }


def trajectories_from_rosbag(filename: str) -> list[Trajectory]:
    """Creates a trajectory from a ROS bag file

    Args:
        filename (str): File to read

    Returns:
        Trajectory: Trajectory created from the ROS bag file
    """
    data = read_ros_bag(filename)

    trajectories = []
    for traj_key, traj_data in data.items():
        pos = Positions(xyz=np.c_[traj_data["x"], traj_data["y"], traj_data["z"]])

        if any(key not in traj_data for key in ["qx", "qy", "qz", "qw"]):
            rot = None
            logger.info("No rotation data found for trajectory %s", traj_key)
        else:
            rot = Rotations.from_quat(quat=np.c_[traj_data["qx"], traj_data["qy"], traj_data["qz"], traj_data["qw"]])

        tstamps = np.array(traj_data.get("tstamps", np.arange(len(pos))), dtype=float)

        trajectories.append(Trajectory(name=traj_key, positions=pos, rotations=rot, timestamps=tstamps))
        logger.info("Found trajectory: %s", traj_key)

    logger.info("Created %d trajectories from ROS bag file", len(trajectories))
    return trajectories


def read_ros_bag(filename: str) -> dict[str, dict[str, list[Any]]]:
    """Reads a ROS bag file and returns the header and the data

    Args:
        filename (str): File to read

    Returns:
        Tuple[HeaderData, np.ndarray]: Header data and data
    """
    ROS_MESSAGE_HANDLERS: dict[str, Callable[[Any], dict[str, Any]]] = {
        "geometry_msgs/msg/PoseStamped": geometry_pose_stamped_handler,
        "geometry_msgs/msg/PointStamped": geometry_point_stamped_handler,
    }
    data: dict[str, dict[str, list[Any]]] = {}
    with AnyReader([Path(filename)], default_typestore=typestore) as reader:
        for connection, _, rawdata in reader.messages(connections=reader.connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            msg_dict = ROS_MESSAGE_HANDLERS.get(connection.msgtype)
            if msg_dict is None:
                logger.warning("Message type %s not supported", connection.msgtype)
                continue

            for key, value in msg_dict(msg).items():
                add_to_dict(data, connection.topic, key, value)

    return data


def add_to_dict(data: dict[str, dict[str, list[Any]]], key: str, sub_key: str, value: Any) -> None:
    """Adds a value to a dictionary

    Args:
        data (dict): Dictionary to add the value to
        key (str): Key to add the value to
        value (Any): Value to add
    """
    if key not in data:
        data[key] = {}

    if sub_key not in data[key]:
        data[key][sub_key] = []

    data[key][sub_key].append(value)
