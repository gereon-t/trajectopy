import logging

from trajectopy import ProcessingSettings, Trajectory, ate

logging.basicConfig(level=logging.INFO)


def main():
    # Import
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

    # default settings
    processing_settings = ProcessingSettings()

    # align = True is the default, but can be disabled if you want to evaluate the unaligned trajectories
    ate_result = ate(other=gt_traj, trajectory=est_traj, processing_settings=processing_settings, align=True)

    width = max(len(key) for key in ate_result.property_dict)
    for key, value in ate_result.property_dict.items():
        print(f"{key:<{width}}  {value}")


if __name__ == "__main__":
    main()
