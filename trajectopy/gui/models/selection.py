from dataclasses import dataclass, field

from trajectopy.gui.models.entries import ResultEntry, TrajectoryEntry


@dataclass
class TrajectorySelection:
    """Dataclass for storing the selected trajectories."""

    entries: list[TrajectoryEntry] = field(default_factory=list)
    reference_entry: TrajectoryEntry | None = None

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

    entries: list[ResultEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self):
        return bool(self.entries)
