from dataclasses import dataclass, field

import numpy as np

from trajectopy.core.parameters import (
    AlignmentParameters,
    SensorRotationParameters,
)
from trajectopy.core.settings import AlignmentEstimationSettings


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    return np.shape(a) == np.shape(b) and bool(np.allclose(a, b))


@dataclass
class AlignmentResult:
    """
    Class to store the result of an alignment.

    Attributes:
        name (str): Name of the result.
        position_parameters (AlignmentParameters): Estimated position alignment parameters.
        rotation_parameters (SensorRotationParameters): Estimated rotation alignment parameters.
        estimation_of (AlignmentEstimationSettings): Settings defining which parameters were estimated.
        converged (bool): Whether the estimation converged.
    """

    name: str = "Alignment Result"
    position_parameters: AlignmentParameters = field(default_factory=AlignmentParameters)
    rotation_parameters: SensorRotationParameters = field(default_factory=SensorRotationParameters)
    estimation_of: AlignmentEstimationSettings = field(default_factory=AlignmentEstimationSettings)
    converged: bool = True

    def __eq__(self, other) -> bool:
        if not isinstance(other, AlignmentResult):
            return NotImplemented

        return (
            _allclose(self.position_parameters.values, other.position_parameters.values)
            and _allclose(self.position_parameters.covariance_matrix, other.position_parameters.covariance_matrix)
            and _allclose(self.rotation_parameters.values, other.rotation_parameters.values)
            and self.estimation_of == other.estimation_of
        )

    def to_file(self, filename: str) -> None:
        """Appends the result to a file."""
        from trajectopy.io import results_io

        results_io.write_alignment_result(self, filename)

    @classmethod
    def from_file(cls, filename: str) -> "AlignmentResult":
        """Loads the result from a file."""
        from trajectopy.io import results_io

        return results_io.read_alignment_result(filename)
