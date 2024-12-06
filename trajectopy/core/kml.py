"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import xml.etree.ElementTree as ET

import numpy as np

from trajectopy.core.trajectory import Trajectory


def create_kml(trajectory: Trajectory, filename: str, precision: float = 1e-6) -> str:
    """
    Create a KML file from a trajectory.

    Args:
        trajectory (Trajectory): Trajectory to be exported.
        filename (str): Filename of the KML file.
        precision (float, optional): Precision of the exported positions in degree. Defaults to 1e-6.
    """
    if trajectory.pos.local_transformer is None:
        raise ValueError(
            "Trajectory must be defined in a well-known coordinate system (EPSG code) to be exported to KML. "
        )
    trajectory.pos.to_epsg(4326)

    trajectory.pos = trajectory.pos.round_to(precision)
    _, indices = np.unique(trajectory.pos.xyz[:, 0:2], return_index=True, axis=0)
    trajectory.apply_index(np.sort(indices))

    kml_file = ET.Element("kml", xmlns="http://earth.google.com/kml/2.1")
    document = ET.SubElement(kml_file, "Document")

    placemark = ET.SubElement(document, "Placemark")
    name = ET.SubElement(placemark, "name")
    name.text = trajectory.name

    style = ET.SubElement(placemark, "Style")
    line_style = ET.SubElement(style, "LineStyle")
    color = ET.SubElement(line_style, "color")
    color.text = "ff0000ff"
    width = ET.SubElement(line_style, "width")
    width.text = "2"

    line_string = ET.SubElement(placemark, "LineString")
    coordinates = ET.SubElement(line_string, "coordinates")

    coordinates.text = "\n".join(f"  {pos[1]:.9f},{pos[0]:.9f},{0.00:.3f}" for pos in trajectory.pos.xyz)

    tree = ET.ElementTree(kml_file)
    ET.indent(tree, space="", level=0)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
