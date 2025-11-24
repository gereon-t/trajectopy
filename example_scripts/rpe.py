from rich.console import Console
from rich.table import Table

from trajectopy import settings
from trajectopy.tools import evaluation
from trajectopy.trajectory import Trajectory


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
    gt_traj = Trajectory.from_file("./example_data/KITTI_gt.traj")
    est_traj = Trajectory.from_file("./example_data/KITTI_ORB.traj")

    # default settings
    processing_settings = settings.ProcessingSettings()

    rpe_result = evaluation.rpe(other=gt_traj, trajectory=est_traj, processing_settings=processing_settings)
    console.print(dict_to_table(rpe_result.property_dict))


if __name__ == "__main__":
    main()
