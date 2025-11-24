import logging

from trajectopy import settings
from trajectopy.tools import evaluation
from trajectopy.trajectory import Trajectory
from trajectopy.visualization import plotly_reports

logging.basicConfig(level=logging.INFO)


def main():
    # Import
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

    # default settings
    processing_settings = settings.ProcessingSettings()

    ate_result = evaluation.ate(other=gt_traj, trajectory=est_traj, processing_settings=processing_settings)
    rpe_result = evaluation.rpe(other=gt_traj, trajectory=est_traj, processing_settings=processing_settings)

    report_settings = settings.ReportSettings(ate_unit_is_mm=False, scatter_marker_size=8)
    report = plotly_reports.create_deviation_report(
        ate_result=ate_result, rpe_result=rpe_result, report_settings=report_settings
    )
    plotly_reports.show_report(report_text=report, filepath="reports/report.html")


if __name__ == "__main__":
    main()
