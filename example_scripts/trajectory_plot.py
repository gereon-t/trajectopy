import logging

from trajectopy import Trajectory
from trajectopy import visualization as vis

logging.basicConfig(level=logging.INFO)


def main():
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

    # positions, coordinates and orientations over time, and maps colored by velocity / height
    vis.show(vis.trajectory_figures([gt_traj, est_traj]))


if __name__ == "__main__":
    main()
