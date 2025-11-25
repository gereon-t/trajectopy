import logging

from trajectopy import ReportSettings, Trajectory
from trajectopy.visualization import plotly_reports

logging.basicConfig(level=logging.INFO)


def main():
    # Import
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

    report_settings = ReportSettings(scatter_axis_order="xy", ate_unit_is_mm=False)
    traj_report = plotly_reports.create_trajectory_report(
        trajectories=[gt_traj, est_traj], report_settings=report_settings
    )
    plotly_reports.show_report(traj_report, filepath="reports/report.html")


if __name__ == "__main__":
    main()
