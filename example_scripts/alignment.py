import logging

from trajectopy import ProcessingSettings, Trajectory, estimate_alignment
from trajectopy.visualization import plotly_reports

logging.basicConfig(level=logging.INFO)


def main():
    # Import
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

    # default settings
    processing_settings = ProcessingSettings()

    alignment_result = estimate_alignment(
        trajectory=est_traj,
        other=gt_traj,
        alignment_settings=processing_settings.alignment,
        matching_settings=processing_settings.matching,
    )

    report = plotly_reports.create_alignment_report(
        alignment_parameters=alignment_result.position_parameters,
        name=alignment_result.name,
    )
    plotly_reports.show_report(report_text=report, filepath="reports/report.html")


if __name__ == "__main__":
    main()
