from rich.console import Console
from rich.table import Table

import trajectopy as tpy


def dict_to_table(data: dict):
    """Converts a dictionary to a rich table."""
    table_data = Table(title="RPE Results")
    table_data.add_column("Property")
    table_data.add_column("Value")
    for key, value in data.items():
        table_data.add_row(key, str(value))
    return table_data


def main():
    console = Console()

    # Import
    gt_traj = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")

    # default settings
    settings = tpy.ProcessingSettings()

    rpe_result = tpy.rpe(trajectory_gt=gt_traj, trajectory_est=est_traj, settings=settings)
    console.print(dict_to_table(rpe_result.property_dict))


if __name__ == "__main__":
    main()
