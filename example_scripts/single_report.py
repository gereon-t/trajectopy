import trajectopy as tpy


def main():
    # Import
    gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

    # default settings
    settings = tpy.ProcessingSettings()

    ate_result = tpy.ate(trajectory_gt=gt_traj, trajectory_est=est_traj, settings=settings)
    rpe_result = tpy.rpe(trajectory_gt=gt_traj, trajectory_est=est_traj, settings=settings)

    report_settings = tpy.ReportSettings(ate_unit_is_mm=False, scatter_marker_size=8)
    report = tpy.create_deviation_report(ate_result=ate_result, rpe_result=rpe_result, report_settings=report_settings)
    tpy.show_report(report_text=report, filepath="reports/report.html")


if __name__ == "__main__":
    main()
