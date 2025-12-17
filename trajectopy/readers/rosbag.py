import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory

typestore = get_typestore(Stores.LATEST)


logger = logging.getLogger(__name__)


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
    ROS_MESSAGE_HANDLERS = {"geometry_msgs/msg/PoseStamped": geometry_pose_stamped_handler}
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
