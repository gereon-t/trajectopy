"""Reading of trajectories from ROS1 bags (``.bag``) and ROS2 bags (``.mcap``, ``.db3`` or a bag directory).

Supported message types (each topic becomes one trajectory, named after the topic):

- ``geometry_msgs/PoseStamped``, ``geometry_msgs/PoseWithCovarianceStamped``, ``nav_msgs/Odometry``:
  positions and orientations (with position standard deviations from the covariance, if given)
- ``geometry_msgs/PointStamped``: positions
- ``geometry_msgs/TransformStamped`` and ``tf2_msgs/TFMessage`` (``/tf``): one trajectory per
  ``parent -> child`` frame pair
- ``nav_msgs/Path``: the poses of the last message on the topic (paths are usually republished as a whole)
- ``sensor_msgs/NavSatFix``: latitude, longitude and ellipsoidal height (EPSG 4326) with standard deviations
"""

import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from trajectopy.core.positions import Positions
from trajectopy.core.trajectory import Trajectory
from trajectopy.io.rosmsg import geometry_pose_stamped_handler, stamp_to_seconds

typestore = get_typestore(Stores.LATEST)

logger = logging.getLogger(__name__)

ROS_BAG_SUFFIXES = (".bag", ".mcap", ".db3")

# geographic coordinates (latitude, longitude, ellipsoidal height) of NavSatFix messages
NAVSAT_EPSG = 4326

Samples = list[tuple[str, dict[str, Any]]]


def is_ros_bag(path: str | Path) -> bool:
    """Whether the path is a ROS bag: a file with a bag suffix or a ROS2 bag directory."""
    path = Path(path)
    return path.suffix.lower() in ROS_BAG_SUFFIXES or (path.is_dir() and (path / "metadata.yaml").exists())


def _pose(position: Any, orientation: Any) -> dict[str, Any]:
    return {
        "x": position.x,
        "y": position.y,
        "z": position.z,
        "qx": orientation.x,
        "qy": orientation.y,
        "qz": orientation.z,
        "qw": orientation.w,
    }


def _std_from_covariance(covariance: Any, size: int) -> dict[str, float]:
    """Position standard deviations from a row-major covariance matrix of the given size (6 for poses, 3 for
    NavSatFix); none if the covariance is unknown (all zeros)."""
    values = np.asarray(covariance, dtype=float)
    diagonal = values[[0, size + 1, 2 * size + 2]]
    if not np.any(diagonal > 0):
        return {}
    sx, sy, sz = np.sqrt(np.clip(diagonal, 0, None))
    return {"sx": sx, "sy": sy, "sz": sz}


def point_stamped_handler(topic: str, msg: Any) -> Samples:
    return [
        (topic, {"tstamps": stamp_to_seconds(msg.header.stamp), "x": msg.point.x, "y": msg.point.y, "z": msg.point.z})
    ]


def pose_stamped_handler(topic: str, msg: Any) -> Samples:
    return [(topic, geometry_pose_stamped_handler(msg))]


def pose_with_covariance_handler(topic: str, msg: Any) -> Samples:
    """``geometry_msgs/PoseWithCovarianceStamped`` and ``nav_msgs/Odometry`` (both have ``pose.pose`` and
    ``pose.covariance``)."""
    sample = {
        "tstamps": stamp_to_seconds(msg.header.stamp),
        **_pose(msg.pose.pose.position, msg.pose.pose.orientation),
        **_std_from_covariance(msg.pose.covariance, 6),
    }
    return [(topic, sample)]


def _transform_sample(topic: str, transform: Any) -> tuple[str, dict[str, Any]]:
    key = f"{topic}: {transform.header.frame_id} -> {transform.child_frame_id}"
    sample = {
        "tstamps": stamp_to_seconds(transform.header.stamp),
        **_pose(transform.transform.translation, transform.transform.rotation),
    }
    return key, sample


def transform_stamped_handler(topic: str, msg: Any) -> Samples:
    return [_transform_sample(topic, msg)]


def tf_message_handler(topic: str, msg: Any) -> Samples:
    return [_transform_sample(topic, transform) for transform in msg.transforms]


def navsatfix_handler(topic: str, msg: Any) -> Samples:
    """``sensor_msgs/NavSatFix``: fixes without a position (status -1) are skipped. The covariance is given
    in east, north, up; the standard deviations are stored in the order of the coordinates (latitude/north,
    longitude/east, height)."""
    if msg.status.status < 0:
        return []
    std = _std_from_covariance(msg.position_covariance, 3) if msg.position_covariance_type > 0 else {}
    if std:
        std = {"sx": std["sy"], "sy": std["sx"], "sz": std["sz"]}
    sample = {
        "tstamps": stamp_to_seconds(msg.header.stamp),
        "x": msg.latitude,
        "y": msg.longitude,
        "z": msg.altitude,
        "epsg": NAVSAT_EPSG,
        **std,
    }
    return [(topic, sample)]


ROS_MESSAGE_HANDLERS: dict[str, Callable[[str, Any], Samples]] = {
    "geometry_msgs/msg/PoseStamped": pose_stamped_handler,
    "geometry_msgs/msg/PointStamped": point_stamped_handler,
    "geometry_msgs/msg/PoseWithCovarianceStamped": pose_with_covariance_handler,
    "nav_msgs/msg/Odometry": pose_with_covariance_handler,
    "geometry_msgs/msg/TransformStamped": transform_stamped_handler,
    "tf2_msgs/msg/TFMessage": tf_message_handler,
    "sensor_msgs/msg/NavSatFix": navsatfix_handler,
}

PATH_MSGTYPE = "nav_msgs/msg/Path"


def supported_message_types() -> list[str]:
    return sorted([*ROS_MESSAGE_HANDLERS, PATH_MSGTYPE])


def list_topics(filename: str | os.PathLike) -> dict[str, str]:
    """The topics of a bag with a supported message type, mapped to their message type.

    Args:
        filename (str): Path to the bag (file or ROS2 bag directory).

    Returns:
        dict[str, str]: Topic name to message type.
    """
    filename = os.fspath(filename)
    with AnyReader([Path(filename)], default_typestore=typestore) as reader:
        return {
            connection.topic: connection.msgtype
            for connection in reader.connections
            if connection.msgtype in ROS_MESSAGE_HANDLERS or connection.msgtype == PATH_MSGTYPE
        }


def trajectories_from_rosbag(filename: str | os.PathLike, topics: Iterable[str] | None = None) -> list[Trajectory]:
    """Creates trajectories from a ROS1 or ROS2 bag.

    Args:
        filename (str): Path to the bag: a ``.bag`` (ROS1), ``.mcap`` or ``.db3`` file, or a ROS2 bag directory.
        topics (Iterable[str] | None, optional): Topics to read. Defaults to all topics with a supported
            message type.

    Returns:
        list[Trajectory]: One trajectory per topic (and per frame pair of transform topics).
    """
    filename = os.fspath(filename)
    data = read_ros_bag(filename, topics=topics)

    trajectories = []
    for traj_key, traj_data in data.items():
        if not traj_data.get("x"):
            continue
        epsg = int(traj_data["epsg"][0]) if "epsg" in traj_data else 0
        pos = Positions(xyz=np.c_[traj_data["x"], traj_data["y"], traj_data["z"]], epsg=epsg)

        if any(len(traj_data.get(key, [])) != len(pos) for key in ["qx", "qy", "qz", "qw"]):
            rot = None
            logger.info("No rotation data found for trajectory %s", traj_key)
        else:
            rot = Rotation.from_quat(quat=np.c_[traj_data["qx"], traj_data["qy"], traj_data["qz"], traj_data["qw"]])

        position_std = None
        if all(len(traj_data.get(key, [])) == len(pos) for key in ["sx", "sy", "sz"]):
            position_std = np.c_[traj_data["sx"], traj_data["sy"], traj_data["sz"]]

        tstamps = np.array(traj_data.get("tstamps", np.arange(len(pos))), dtype=float)

        trajectories.append(
            Trajectory(name=traj_key, positions=pos, rotations=rot, timestamps=tstamps, position_std=position_std)
        )
        logger.info("Found trajectory: %s", traj_key)

    logger.info("Created %d trajectories from ROS bag %s", len(trajectories), filename)
    return trajectories


def read_ros_bag(filename: str | os.PathLike, topics: Iterable[str] | None = None) -> dict[str, dict[str, list[Any]]]:
    """Reads the messages of supported types from a ROS1 or ROS2 bag.

    Args:
        filename (str): Path to the bag (file or ROS2 bag directory).
        topics (Iterable[str] | None, optional): Topics to read. Defaults to all.

    Returns:
        dict[str, dict[str, list[Any]]]: For each trajectory (topic), the values of each field.
    """
    filename = os.fspath(filename)
    wanted = set(topics) if topics is not None else None
    data: dict[str, dict[str, list[Any]]] = {}
    unsupported: set[str] = set()
    with AnyReader([Path(filename)], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if wanted is None or c.topic in wanted]
        if wanted is not None and (missing := wanted - {c.topic for c in connections}):
            logger.warning("Topics not found in %s: %s", filename, ", ".join(sorted(missing)))

        for connection, _, rawdata in reader.messages(connections=connections):
            if connection.msgtype == PATH_MSGTYPE:
                path_msg: Any = reader.deserialize(rawdata, connection.msgtype)
                data.pop(connection.topic, None)  # the last path contains the whole trajectory
                for pose in path_msg.poses:
                    for key, value in geometry_pose_stamped_handler(pose).items():
                        add_to_dict(data, connection.topic, key, value)
                continue

            handler = ROS_MESSAGE_HANDLERS.get(connection.msgtype)
            if handler is None:
                unsupported.add(connection.msgtype)
                continue

            msg = reader.deserialize(rawdata, connection.msgtype)
            for traj_key, sample in handler(connection.topic, msg):
                for key, value in sample.items():
                    add_to_dict(data, traj_key, key, value)

    for msgtype in sorted(unsupported):
        logger.info("Skipped messages of the unsupported type %s", msgtype)
    return data


def add_to_dict(data: dict[str, dict[str, list[Any]]], key: str, sub_key: str, value: Any) -> None:
    """Adds a value to a dictionary

    Args:
        data (dict): Dictionary to add the value to
        key (str): Key to add the value to
        value (Any): Value to add
    """
    data.setdefault(key, {}).setdefault(sub_key, []).append(value)


def geometry_point_stamped_handler(msg: Any) -> dict[str, Any]:
    """Deprecated: handler for ``geometry_msgs/PointStamped`` messages of earlier versions."""
    return point_stamped_handler("", msg)[0][1]
