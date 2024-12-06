from dataclasses import dataclass

from trajectopy.core.settings.base import Settings


@dataclass
class ApproximationSettings(Settings):
    """Dataclass defining approximation configuration"""

    fe_int_size: float = 0.15
    fe_min_obs: int = 25
    rot_approx_win_size: float = 0.15
