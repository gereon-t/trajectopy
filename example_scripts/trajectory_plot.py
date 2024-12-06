import trajectopy as tpy


def main():
    # Import
    gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

    report_settings = tpy.ReportSettings(scatter_axis_order="xy", ate_unit_is_mm=False)
    traj_report = tpy.create_trajectory_report(trajectories=[gt_traj, est_traj], report_settings=report_settings)
    tpy.show_report(traj_report, filepath="reports/report.html")


if __name__ == "__main__":
    main()
