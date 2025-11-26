from matplotlib import pyplot as plt

from trajectopy import Trajectory, align
from trajectopy.visualization.mpl_plots import plot_positions


def main():
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

    est_traj_aligned = align(trajectory=est_traj, other=gt_traj, inplace=False)
    est_traj_aligned.name += " (Aligned)"

    plot_positions(trajectories=[gt_traj, est_traj, est_traj_aligned])
    plt.show()


if __name__ == "__main__":
    main()
