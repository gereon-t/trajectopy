"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
from dataclasses import dataclass, field

from spatialsorter.settings import SortingSettings
from trajectopy_approx.settings import ApproximationSettings
from trajectopy_core.alignment.settings import AlignmentSettings
from trajectopy_core.evaluation.comparison import RelativeComparisonSettings
from trajectopy_core.evaluation.settings import MatchingSettings
from yaml_dataclass import Settings, dataclass


@dataclass
class ProcessingSettings(Settings):
    """This class stores all processing settings

    It can be initialized by either providing separate
    configurationsets for

    - sorting
    - approximation
    - alignment

    or by using the 'from_file' method together with a yaml configuration file
    """

    sorting: SortingSettings = field(default_factory=SortingSettings)
    approximation: ApproximationSettings = field(default_factory=ApproximationSettings)
    alignment: AlignmentSettings = field(default_factory=AlignmentSettings)
    rel_comparison: RelativeComparisonSettings = field(default_factory=RelativeComparisonSettings)
    matching: MatchingSettings = field(default_factory=MatchingSettings)

    def __str__(self) -> str:
        return str(self.sorting) + str(self.approximation) + str(self.alignment)
