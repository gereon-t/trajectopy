import logging

from trajectopy import ProcessingSettings, Trajectory, estimate_alignment
from trajectopy import visualization as vis

logging.basicConfig(level=logging.INFO)


def main():
    # Import
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

    # default settings
    processing_settings = ProcessingSettings()

    alignment_result = estimate_alignment(est_traj, gt_traj, processing_settings)

    # covariance and correlation of the estimated parameters
    vis.show(vis.alignment_figures(alignment_result.position_parameters), title=alignment_result.name)


if __name__ == "__main__":
    main()
