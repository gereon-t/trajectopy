from trajectopy import ProcessingSettings, Trajectory, rpe


def main():
    # Import
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

    # default settings
    processing_settings = ProcessingSettings()

    rpe_result = rpe(other=gt_traj, trajectory=est_traj, processing_settings=processing_settings)

    width = max(len(key) for key in rpe_result.property_dict)
    for key, value in rpe_result.property_dict.items():
        print(f"{key:<{width}}  {value}")


if __name__ == "__main__":
    main()
