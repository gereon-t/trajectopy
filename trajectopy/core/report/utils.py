"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import base64
import logging
import os
import sys

import numpy as np

# import webbrowser


# logger configuration
logger = logging.getLogger("root")

try:
    BASE_PATH = os.path.join(sys._MEIPASS, "trajectopy")
except Exception:
    BASE_PATH = os.path.dirname(__file__)
TEMPLATES_PATH = os.path.join(BASE_PATH, "templates")


def number_to_string(number: float) -> str:
    return f"{number:.3f}"


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def convert_icon_to_base64() -> str:
    icon_path = os.path.join(os.path.join(BASE_PATH), "assets", "icon.png")
    return image_to_base64(icon_path)


def shrink_data(data: np.ndarray, max_size: int = 1000) -> np.ndarray:
    """
    Shrink the given data to the given max_size by taking
    the moving average of the data.

    Args:
        data (np.ndarray): The data to shrink
        max_size (int): The maximum size of the data

    Returns:
        np.ndarray: The shrunk data
    """
    if max_size < 0:
        return data

    if len(data) <= max_size:
        return data

    data = data.astype(float)
    array_size = (-len(data) % max_size) + len(data)
    padded_data = np.pad(data, (0, array_size - len(data)))
    padded_data[padded_data == 0] = np.nan

    reshaped_data = np.array(
        [row for row in padded_data.reshape(-1, len(padded_data) // max_size) if not np.all(np.isnan(row))]
    )

    return np.nanmean(reshaped_data, 1)
