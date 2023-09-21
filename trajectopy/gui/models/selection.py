"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
from dataclasses import dataclass, field
from typing import List, Union

from trajectopy_core.util.entries import ResultEntry, TrajectoryEntry


@dataclass
class TrajectorySelection:
    """Dataclass for storing the selected trajectories."""

    entries: List[TrajectoryEntry] = field(default_factory=list)
    reference_entry: Union[TrajectoryEntry, None] = None

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self):
        return bool(self.entries)

    @property
    def reference_is_set(self) -> bool:
        return self.reference_entry is not None


@dataclass
class ResultSelection:
    """Dataclass for storing the selected results."""

    entries: List[ResultEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self):
        return bool(self.entries)
