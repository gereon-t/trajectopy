import logging

from trajectopy import ProcessingSettings, Trajectory, ate, rpe
from trajectopy.visualization import plotly_reports

logging.basicConfig(level=logging.INFO)


def main():
    # Import
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj_1 = Trajectory.from_file("./example_data/KITTI_ORB.traj")
    est_traj_2 = Trajectory.from_file("./example_data/KITTI_SPTAM.traj")

    # default settings
    processing_settings = ProcessingSettings()

    ate_result_1 = ate(other=gt_traj, trajectory=est_traj_1, processing_settings=processing_settings)
    rpe_result_1 = rpe(other=gt_traj, trajectory=est_traj_1, processing_settings=processing_settings)

    ate_result_2 = ate(other=gt_traj, trajectory=est_traj_2, processing_settings=processing_settings)
    rpe_result_2 = rpe(other=gt_traj, trajectory=est_traj_2, processing_settings=processing_settings)

    multi_report = plotly_reports.create_deviation_report(
        ate_result=[ate_result_1, ate_result_2], rpe_result=[rpe_result_1, rpe_result_2]
    )

    plotly_reports.show_report(report_text=multi_report, filepath="reports/report.html")


if __name__ == "__main__":
    main()
