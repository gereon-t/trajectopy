import logging

import matplotlib.pyplot as plt

from trajectopy import Trajectory, ate, estimate_alignment, rpe
from trajectopy.visualization import mpl_plots

logging.basicConfig(level=logging.INFO)


def main():
    gt = Trajectory.from_file("./example_data/KITTI_gt.traj")
    orb = Trajectory.from_file("./example_data/KITTI_ORB.traj")
    sptam = Trajectory.from_file("./example_data/KITTI_SPTAM.traj")
    mpl_plots.plot_trajectories([gt, orb])

    alignment_result = estimate_alignment(gt, orb)
    ate_result_orb = ate(gt, orb)
    ate_result_sptam = ate(gt, sptam)
    rpe_result = rpe(gt, orb)

    # All available plotting functions
    mpl_plots.plot_covariance_heatmap(alignment_result.position_parameters)
    mpl_plots.plot_correlation_heatmap(alignment_result.position_parameters)

    mpl_plots.plot_ate_3d([ate_result_orb, ate_result_sptam])
    mpl_plots.plot_compact_ate_hist(ate_result_orb)
    mpl_plots.plot_ate([ate_result_orb, ate_result_sptam])
    mpl_plots.plot_ate_bars([ate_result_orb, ate_result_sptam], mode="positions")
    mpl_plots.plot_ate_bars([ate_result_orb, ate_result_sptam], mode="rotations")

    mpl_plots.plot_ate_edf(ate_result_orb)

    mpl_plots.scatter_ate(ate_result_orb)

    mpl_plots.plot_rpe(rpe_result)

    plt.show()


if __name__ == "__main__":
    main()
