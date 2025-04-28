import matplotlib.pyplot as plt

import trajectopy as tpy


def main():
    gt = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
    orb = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
    sptam = tpy.Trajectory.from_file("./example_data/KITTI_SPTAM.traj")
    tpy.plot_trajectories([gt, orb])

    alignment = tpy.estimate_alignment(gt, orb)
    ate_result_orb = tpy.ate(gt, orb)
    ate_result_sptam = tpy.ate(gt, sptam)
    rpe_result = tpy.rpe(gt, orb)

    # All available plotting functions
    tpy.plot_covariance_heatmap(alignment.position_parameters)
    tpy.plot_correlation_heatmap(alignment.position_parameters)

    tpy.plot_ate_3d([ate_result_orb, ate_result_sptam])
    tpy.plot_compact_ate_hist(ate_result_orb)
    tpy.plot_ate([ate_result_orb, ate_result_sptam])
    tpy.plot_ate_bars([ate_result_orb, ate_result_sptam], mode="positions")
    tpy.plot_ate_bars([ate_result_orb, ate_result_sptam], mode="rotations")

    tpy.plot_ate_edf(ate_result_orb)

    tpy.scatter_ate(ate_result_orb)

    tpy.plot_rpe(rpe_result)

    plt.show()


if __name__ == "__main__":
    main()
