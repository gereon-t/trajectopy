"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2025
tombrink@igg.uni-bonn.de
"""

from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

from trajectopy.rotationset import RotationSet
from trajectopy.settings import PairDistanceUnit


@dataclass
class RelativeTrajectoryDeviations:
    pos_dev: Dict[float, List[float]]
    rot_dev: Dict[float, List[float]]
    pair_distance: Dict[float, List[float]]
    pair_distance_unit: PairDistanceUnit = PairDistanceUnit.METER

    @property
    def num_pairs(self) -> int:
        return sum(len(values) for values in self.pair_distance.values())


@dataclass
class AbsoluteTrajectoryDeviations:
    pos_dev: np.ndarray
    directed_pos_dev: np.ndarray
    rot_dev: Union[RotationSet, None] = None
    rotations_used: bool = False
