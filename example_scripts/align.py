from trajectopy import Trajectory, align
from trajectopy import visualization as vis


def main():
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

    est_traj_aligned = align(trajectory=est_traj, other=gt_traj, inplace=False)
    est_traj_aligned.name += " (aligned)"

    vis.show(vis.positions_figure([gt_traj, est_traj, est_traj_aligned]))


if __name__ == "__main__":
    main()
