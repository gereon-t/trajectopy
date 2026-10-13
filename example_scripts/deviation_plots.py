import logging

from trajectopy import PlotSettings, ProcessingSettings, Trajectory, ate, rpe
from trajectopy import visualization as vis

logging.basicConfig(level=logging.INFO)


def main():
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    orb_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")
    sptam_traj = Trajectory.from_file("./example_data/KITTI_SPTAM.traj")

    processing_settings = ProcessingSettings()
    ate_results = [
        ate(trajectory=orb_traj, other=gt_traj, processing_settings=processing_settings),
        ate(trajectory=sptam_traj, other=gt_traj, processing_settings=processing_settings),
    ]
    rpe_results = [
        rpe(trajectory=orb_traj, other=gt_traj, processing_settings=processing_settings),
        rpe(trajectory=sptam_traj, other=gt_traj, processing_settings=processing_settings),
    ]

    # The same figures work for one or several results.
    plot_settings = PlotSettings(ate_unit_is_mm=False, directed_ate=True)
    figures = vis.deviation_figures(ate_results, rpe_results, settings=plot_settings)

    # Write every figure (and every variant, e.g. each color-coded quantity) to image files ...
    vis.export_figures(figures, "plots", export_format="png")

    # ... and/or explore them interactively.
    vis.show(figures, title="KITTI Evaluation")


if __name__ == "__main__":
    main()
