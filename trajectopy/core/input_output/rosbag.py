"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pointset import PointSet
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from trajectopy.core.input_output.rosmsg import geometry_pose_stamped_handler
from trajectopy.core.rotationset import RotationSet
from trajectopy.core.trajectory import Trajectory

typestore = get_typestore(Stores.LATEST)
ROS_MESSAGE_HANDLERS = {"geometry_msgs/msg/PoseStamped": geometry_pose_stamped_handler}

logger = logging.getLogger("root")


def trajectories_from_rosbag(filename: str) -> List[Trajectory]:
    """Creates a trajectory from a ROS bag file

    Args:
        filename (str): File to read

    Returns:
        Trajectory: Trajectory created from the ROS bag file
    """
    data = read_ros_bag(filename)

    trajectories = []
    for traj_key, traj_data in data.items():
        pos = PointSet(xyz=np.c_[traj_data["x"], traj_data["y"], traj_data["z"]])

        if any(key not in traj_data for key in ["qx", "qy", "qz", "qw"]):
            rot = None
            logger.info("No rotation data found for trajectory %s", traj_key)
        else:
            rot = RotationSet.from_quat(quat=np.c_[traj_data["qx"], traj_data["qy"], traj_data["qz"], traj_data["qw"]])

        tstamps = np.array(traj_data.get("tstamps", np.arange(len(pos))), dtype=float)

        trajectories.append(Trajectory(name=traj_key, pos=pos, rot=rot, tstamps=tstamps))
        logger.info("Found trajectory: %s", traj_key)

    logger.info("Created %d trajectories from ROS bag file", len(trajectories))
    return trajectories


def read_ros_bag(filename: str) -> Dict[str, Dict[str, List[Any]]]:
    """Reads a ROS bag file and returns the header and the data

    Args:
        filename (str): File to read

    Returns:
        Tuple[HeaderData, np.ndarray]: Header data and data
    """
    data: Dict[str, Dict[str, List[Any]]] = {}
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


def add_to_dict(data: Dict[str, Dict[str, List[Any]]], key: str, sub_key: str, value: Any) -> None:
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
