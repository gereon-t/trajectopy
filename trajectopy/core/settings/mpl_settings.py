"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

import logging
from dataclasses import dataclass

from trajectopy.core.settings.base import Settings

logger = logging.getLogger("root")


@dataclass
class MPLPlotSettings(Settings):
    """Dataclass defining plot configuration"""

    scatter_cbar_show_zero: bool = True
    scatter_cbar_steps: int = 4
    scatter_no_axis: bool = False
    scatter_max_std: float = 3.0
    ate_unit_is_mm: bool = False
    hist_as_stairs: bool = False
    directed_ate: bool = False
    scatter_pos_dim: int = 2

    @property
    def unit_multiplier(self) -> float:
        return 1000 if self.ate_unit_is_mm else 1

    @property
    def unit_str(self) -> str:
        return "[mm]" if self.ate_unit_is_mm else "[m]"
