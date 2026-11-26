"""Reading trajectories from ROS2 bags with the supported message types."""

import numpy as np
import pytest
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

from trajectopy.io.rosbag import is_ros_bag, list_topics, trajectories_from_rosbag

typestore = get_typestore(Stores.LATEST)
NUM_MESSAGES = 20


def _types():
    names = [
        "builtin_interfaces/msg/Time",
        "std_msgs/msg/Header",
        "geometry_msgs/msg/Point",
        "geometry_msgs/msg/Quaternion",
        "geometry_msgs/msg/Pose",
        "geometry_msgs/msg/PoseStamped",
        "geometry_msgs/msg/PoseWithCovariance",
        "geometry_msgs/msg/Twist",
        "geometry_msgs/msg/TwistWithCovariance",
        "geometry_msgs/msg/Vector3",
        "geometry_msgs/msg/Transform",
        "geometry_msgs/msg/TransformStamped",
        "nav_msgs/msg/Odometry",
        "nav_msgs/msg/Path",
        "sensor_msgs/msg/NavSatStatus",
        "sensor_msgs/msg/NavSatFix",
        "tf2_msgs/msg/TFMessage",
    ]
    return {name.rsplit("/", 1)[1]: typestore.types[name] for name in names}


def _header(t, i: int, frame: str = "map"):
    return t["Header"](stamp=t["Time"](sec=100 + i, nanosec=500_000_000), frame_id=frame)


def _pose(t, i: int):
    return t["Pose"](
        position=t["Point"](x=float(i), y=2.0 * i, z=0.5),
        orientation=t["Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
    )


@pytest.fixture(scope="module")
def bag_path(tmp_path_factory):
    return write_bag(tmp_path_factory.mktemp("bags") / "recording")


def write_bag(path):
    """Writes a ROS2 bag (directory ``path`` with an MCAP file) with /odom, /tf, /fix and /path messages."""
    t = _types()
    covariance = np.zeros(36)
    covariance[[0, 7, 14]] = [0.04, 0.09, 0.16]
    with Writer(path, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        odom = writer.add_connection("/odom", "nav_msgs/msg/Odometry", typestore=typestore)
        tf = writer.add_connection("/tf", "tf2_msgs/msg/TFMessage", typestore=typestore)
        fix = writer.add_connection("/fix", "sensor_msgs/msg/NavSatFix", typestore=typestore)
        path_conn = writer.add_connection("/path", "nav_msgs/msg/Path", typestore=typestore)
        for i in range(NUM_MESSAGES):
            stamp_ns = (100 + i) * 1_000_000_000 + 500_000_000
            twist = t["TwistWithCovariance"](
                twist=t["Twist"](linear=t["Vector3"](x=0.0, y=0.0, z=0.0), angular=t["Vector3"](x=0.0, y=0.0, z=0.0)),
                covariance=np.zeros(36),
            )
            msg = t["Odometry"](
                header=_header(t, i),
                child_frame_id="base_link",
                pose=t["PoseWithCovariance"](pose=_pose(t, i), covariance=covariance),
                twist=twist,
            )
            writer.write(odom, stamp_ns, typestore.serialize_cdr(msg, "nav_msgs/msg/Odometry"))

            transforms = [
                t["TransformStamped"](
                    header=_header(t, i, parent),
                    child_frame_id=child,
                    transform=t["Transform"](
                        translation=t["Vector3"](x=float(i), y=0.0, z=0.0),
                        rotation=t["Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
                    ),
                )
                for parent, child in (("map", "odom"), ("odom", "base_link"))
            ]
            writer.write(
                tf, stamp_ns, typestore.serialize_cdr(t["TFMessage"](transforms=transforms), "tf2_msgs/msg/TFMessage")
            )

            navsat = t["NavSatFix"](
                header=_header(t, i),
                status=t["NavSatStatus"](status=0, service=1),
                latitude=50.7 + i * 1e-5,
                longitude=7.1,
                altitude=100.0,
                position_covariance=np.array([0.01, 0, 0, 0, 0.04, 0, 0, 0, 0.09]),
                position_covariance_type=2,
            )
            writer.write(fix, stamp_ns, typestore.serialize_cdr(navsat, "sensor_msgs/msg/NavSatFix"))

            poses = [t["PoseStamped"](header=_header(t, j), pose=_pose(t, j)) for j in range(i + 1)]
            writer.write(
                path_conn,
                stamp_ns,
                typestore.serialize_cdr(t["Path"](header=_header(t, i), poses=poses), "nav_msgs/msg/Path"),
            )
    return path


def test_is_ros_bag(bag_path, tmp_path):
    assert is_ros_bag(bag_path)
    assert is_ros_bag("recording.mcap") and is_ros_bag("recording.bag") and is_ros_bag("recording.db3")
    assert not is_ros_bag(tmp_path / "trajectory.traj")


def test_list_topics(bag_path):
    assert list_topics(str(bag_path)) == {
        "/odom": "nav_msgs/msg/Odometry",
        "/tf": "tf2_msgs/msg/TFMessage",
        "/fix": "sensor_msgs/msg/NavSatFix",
        "/path": "nav_msgs/msg/Path",
    }


def test_all_message_types(bag_path):
    trajectories = {trajectory.name: trajectory for trajectory in trajectories_from_rosbag(str(bag_path))}
    assert set(trajectories) == {"/odom", "/tf: map -> odom", "/tf: odom -> base_link", "/fix", "/path"}

    odom = trajectories["/odom"]
    assert len(odom) == NUM_MESSAGES and odom.has_orientation
    np.testing.assert_allclose(odom.timestamps, 100.5 + np.arange(NUM_MESSAGES))
    np.testing.assert_allclose(odom.xyz[:, 1], 2.0 * np.arange(NUM_MESSAGES))
    np.testing.assert_allclose(odom.position_std, np.tile([0.2, 0.3, 0.4], (NUM_MESSAGES, 1)))

    assert len(trajectories["/tf: odom -> base_link"]) == NUM_MESSAGES

    fix = trajectories["/fix"]
    assert fix.positions.epsg == 4326 and not fix.has_orientation
    # covariance east, north, up -> latitude (north), longitude (east), height
    np.testing.assert_allclose(fix.position_std[0], [0.2, 0.1, 0.3])

    # the last path message holds the whole trajectory
    assert len(trajectories["/path"]) == NUM_MESSAGES


def test_topic_selection(bag_path):
    trajectories = trajectories_from_rosbag(str(bag_path), topics=["/odom"])
    assert [trajectory.name for trajectory in trajectories] == ["/odom"]
