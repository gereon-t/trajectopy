"""Reading and writing of trajectories (trajectopy ASCII format and KML)."""

import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from trajectopy.core.trajectory import Trajectory
from trajectopy.io import ascii
from trajectopy.io.header import HeaderData
from trajectopy.utils.definitions import Sorting


def _trajectory_from_data(header_data: HeaderData, trajectory_data: np.ndarray) -> Trajectory:
    return Trajectory(
        timestamps=ascii.extract_trajectory_timestamps(header_data=header_data, trajectory_data=trajectory_data),
        positions=ascii.extract_trajectory_positions(header_data=header_data, trajectory_data=trajectory_data),
        rotations=ascii.extract_trajectory_rotations(header_data=header_data, trajectory_data=trajectory_data),
        name=header_data.name,
        path_lengths=ascii.extract_trajectory_path_lengths(header_data=header_data, trajectory_data=trajectory_data),
        velocity_xyz=ascii.extract_trajectory_velocity_xyz(header_data=header_data, trajectory_data=trajectory_data),
        sorting=Sorting.from_str(header_data.sorting),
    )


def read_trajectory(filename: str) -> Trajectory:
    """Reads a trajectory from a trajectopy ASCII file.

    Args:
        filename (str): Path to the file.

    Returns:
        Trajectory: The loaded trajectory.
    """
    return _trajectory_from_data(*ascii.read_data(filename, dtype=object))


def read_trajectory_string(content: str) -> Trajectory:
    """Reads a trajectory from the string content of a trajectopy ASCII file.

    Args:
        content (str): File content including the header.

    Returns:
        Trajectory: The loaded trajectory.
    """
    return _trajectory_from_data(*ascii.read_string(content, dtype=object))


def trajectory_to_string(trajectory: Trajectory) -> str:
    """Serializes a trajectory to a CSV-formatted string with metadata headers.

    Headers included: #epsg, #name, #nframe, #sorting, #fields.
    """
    fields = "t,l,px,py,pz,vx,vy,vz" if trajectory.rotations is None else "t,l,px,py,pz,qx,qy,qz,qw,vx,vy,vz"
    header = [
        f"#epsg {trajectory.positions.epsg}",
        f"#name {trajectory.name}",
        "#nframe enu",
        f"#sorting {trajectory.sorting.value}",
        f"#fields {fields}",
    ]

    if trajectory.rotations is None:
        trajectory_data = np.c_[
            trajectory.timestamps, trajectory.path_lengths, trajectory.positions.xyz, trajectory.velocity_xyz
        ]
    else:
        trajectory_data = np.c_[
            trajectory.timestamps,
            trajectory.path_lengths,
            trajectory.positions.xyz,
            trajectory.rotations.as_quat(),
            trajectory.velocity_xyz,
        ]

    output = io.StringIO()
    output.write("\n".join(header) + "\n")
    pd.DataFrame(trajectory_data).to_csv(output, header=False, index=False, float_format="%.9f")
    return output.getvalue()


def write_trajectory(trajectory: Trajectory, filename: str, mode: str = "w") -> None:
    """Writes a trajectory to a trajectopy ASCII file.

    Args:
        trajectory (Trajectory): Trajectory to write.
        filename (str): The output file path.
        mode (str, optional): File open mode. Defaults to "w".
    """
    with open(filename, mode=mode, newline="\n", encoding="utf-8") as file:
        file.write(trajectory_to_string(trajectory))


def write_kml(trajectory: Trajectory, filename: str, precision: float = 1e-6) -> None:
    """Exports a trajectory to a Google Earth KML file.

    Requires the trajectory to have a valid EPSG code so it can be converted to WGS84 (EPSG:4326).

    Args:
        trajectory (Trajectory): Trajectory to export.
        filename (str): The output filename (e.g., "track.kml").
        precision (float, optional): Coordinate precision in degrees for rounding/simplification. Defaults to 1e-6.

    Raises:
        ValueError: If the trajectory does not have a known EPSG code.
    """
    traj = trajectory.copy()
    if traj.positions.local_transformer is None:
        raise ValueError(
            "Trajectory must be defined in a well-known coordinate system (EPSG code) to be exported to KML. "
        )
    traj.positions.to_epsg(4326)

    traj.positions = traj.positions.round_to(precision)
    _, indices = np.unique(traj.positions.xyz[:, 0:2], return_index=True, axis=0)
    traj.mask(np.sort(indices))

    kml_file = ET.Element("kml", xmlns="http://earth.google.com/kml/2.1")
    document = ET.SubElement(kml_file, "Document")

    placemark = ET.SubElement(document, "Placemark")
    name = ET.SubElement(placemark, "name")
    name.text = traj.name

    style = ET.SubElement(placemark, "Style")
    line_style = ET.SubElement(style, "LineStyle")
    color = ET.SubElement(line_style, "color")
    color.text = "ff0000ff"
    width = ET.SubElement(line_style, "width")
    width.text = "2"

    line_string = ET.SubElement(placemark, "LineString")
    coordinates = ET.SubElement(line_string, "coordinates")

    coordinates.text = "\n".join(f"  {pos[1]:.9f},{pos[0]:.9f},{0.00:.3f}" for pos in traj.positions.xyz)

    tree = ET.ElementTree(kml_file)
    ET.indent(tree, space="", level=0)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
