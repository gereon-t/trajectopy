"""Third-party trajectory formats (TUM, KITTI, EuRoC) and their detection."""

import numpy as np
import pytest

from test.testdata import generated_trajectory
from trajectopy.core.trajectory import Trajectory
from trajectopy.io.formats import (
    TrajectoryFormat,
    detect_format,
    read_euroc,
    read_kitti,
    read_trajectory_file,
    write_trajectory_file,
)


@pytest.fixture
def trajectory() -> Trajectory:
    return generated_trajectory.copy()


@pytest.mark.parametrize("file_format", ["tum", "euroc"])
def test_round_trip_with_timestamps(tmp_path, trajectory, file_format):
    path = tmp_path / f"trajectory.{file_format}"
    write_trajectory_file(trajectory, str(path), format=file_format)

    assert detect_format(str(path)) == (
        TrajectoryFormat.EUROC if file_format == "euroc" else TrajectoryFormat.TRAJECTOPY
    )
    loaded = Trajectory.from_file(str(path))
    np.testing.assert_allclose(loaded.timestamps, trajectory.timestamps, atol=1e-6)
    np.testing.assert_allclose(loaded.xyz, trajectory.xyz, atol=1e-6)
    np.testing.assert_allclose(np.abs(np.sum(loaded.quat * trajectory.quat, axis=1)), 1.0, atol=1e-6)
    assert loaded.name == "trajectory"


def test_kitti_round_trip_and_detection(tmp_path, trajectory):
    path = tmp_path / "poses.txt"
    write_trajectory_file(trajectory, str(path), format="kitti")

    assert detect_format(str(path)) == TrajectoryFormat.KITTI
    loaded = Trajectory.from_file(str(path))
    assert len(loaded) == len(trajectory)
    np.testing.assert_allclose(loaded.timestamps, np.arange(len(trajectory)))
    np.testing.assert_allclose(loaded.xyz, trajectory.xyz, atol=1e-6)
    np.testing.assert_allclose(loaded.rotations.as_matrix(), trajectory.rotations.as_matrix(), atol=1e-6)


def test_kitti_with_timestamps_file(tmp_path, trajectory):
    path, times = tmp_path / "poses.txt", tmp_path / "times.txt"
    write_trajectory_file(trajectory, str(path), format="kitti")
    np.savetxt(times, trajectory.timestamps)

    loaded = read_kitti(str(path), timestamps=str(times))
    np.testing.assert_allclose(loaded.timestamps, trajectory.timestamps)

    with pytest.raises(ValueError, match="timestamps"):
        read_kitti(str(path), timestamps=np.arange(3))


def test_euroc_quaternion_order_and_nanoseconds(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n"
        "1403636579758555392,4.688,-1.786,0.783,0.534,-0.153,-0.827,-0.082\n"
        "1403636579763555584,4.688,-1.786,0.787,0.534,-0.153,-0.827,-0.082\n"
    )
    trajectory = read_euroc(str(path))
    np.testing.assert_allclose(trajectory.timestamps[0], 1403636579.758555392)
    quat = np.array([-0.153, -0.827, -0.082, 0.534])
    np.testing.assert_allclose(trajectory.quat[0], quat / np.linalg.norm(quat), atol=1e-9)


def test_trajectopy_files_are_not_misdetected(tmp_path, trajectory):
    path = tmp_path / "trajectory.traj"
    trajectory.to_file(str(path))
    assert detect_format(str(path)) == TrajectoryFormat.TRAJECTOPY
    assert read_trajectory_file(str(path)) == trajectory


def test_unknown_format():
    with pytest.raises(ValueError, match="Unknown trajectory format"):
        TrajectoryFormat.from_str("bogus")
