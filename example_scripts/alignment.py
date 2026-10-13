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

    alignment_result = estimate_alignment(
        trajectory=est_traj,
        other=gt_traj,
        alignment_settings=processing_settings.alignment,
        matching_settings=processing_settings.matching,
    )

    # covariance and correlation of the estimated parameters
    vis.show(vis.alignment_figures(alignment_result.position_parameters), title=alignment_result.name)


if __name__ == "__main__":
    main()
