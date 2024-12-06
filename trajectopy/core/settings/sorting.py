"""
Gereon Tombrink, 2024
mail@gtombrink.de
"""

from dataclasses import dataclass

from trajectopy.core.settings.base import Settings


@dataclass
class SortingSettings(Settings):
    """This class stores all sorting settings"""

    discard_missing: bool = True
    voxel_size: float = 0.05
    movement_threshold: float = 0.005
    k_nearest: int = 4
