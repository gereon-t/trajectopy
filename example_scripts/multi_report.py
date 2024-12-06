import trajectopy as tpy


def main():
    # Import
    gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj_1 = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
    est_traj_2 = tpy.Trajectory.from_file("./example_data/KITTI_SPTAM.traj")

    # default settings
    settings = tpy.ProcessingSettings()

    ate_result_1 = tpy.ate(trajectory_gt=gt_traj, trajectory_est=est_traj_1, settings=settings)
    rpe_result_1 = tpy.rpe(trajectory_gt=gt_traj, trajectory_est=est_traj_1, settings=settings)

    ate_result_2 = tpy.ate(trajectory_gt=gt_traj, trajectory_est=est_traj_2, settings=settings)
    rpe_result_2 = tpy.rpe(trajectory_gt=gt_traj, trajectory_est=est_traj_2, settings=settings)

    multi_report = tpy.create_deviation_report(
        ate_result=[ate_result_1, ate_result_2], rpe_result=[rpe_result_1, rpe_result_2]
    )

    tpy.show_report(report_text=multi_report, filepath="reports/report.html")


if __name__ == "__main__":
    main()
