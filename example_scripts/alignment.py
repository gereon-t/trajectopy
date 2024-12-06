import trajectopy as tpy


def main():
    # Import
    gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

    # default settings
    settings = tpy.ProcessingSettings()

    alignment_result = tpy.estimate_alignment(
        traj_from=est_traj,
        traj_to=gt_traj,
        alignment_settings=settings.alignment,
        matching_settings=settings.matching,
    )

    report = tpy.create_alignment_report(
        alignment_parameters=alignment_result.position_parameters,
        name=alignment_result.name,
    )
    tpy.show_report(report_text=report, filepath="reports/report.html")


if __name__ == "__main__":
    main()
