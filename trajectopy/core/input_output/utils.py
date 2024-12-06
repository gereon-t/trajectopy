"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging

import numpy as np

# logger configuration
logger = logging.getLogger("root")


def get_rot_matrix(nframe: str) -> np.ndarray:
    """Returns rotation matrix to transform rotations to ENU

    Args:
        nframe (str): String defining the current nframe definition.
                      e.g. "ned". This string must contain "e", "n",
                      and either "d" OR "u"

    Returns:
        np.ndarray: Rotation Matrix
    """
    nframe = nframe.lower()

    if "e" not in nframe or "n" not in nframe or "u" not in nframe and "d" not in nframe:
        raise ValueError("Invalid input: The input string must contain at least 'e', 'n', and either 'u' or 'd'")

    sign_flipped = "d" in nframe
    nframe = nframe.replace("d", "u")
    rot_matrix = np.zeros((3, 3))

    for i, char in enumerate(nframe):
        index = "enu".find(char)
        value = -1 if char == "u" and sign_flipped else 1
        rot_matrix[i, index] = value

    return rot_matrix
