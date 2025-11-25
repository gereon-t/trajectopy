from dataclasses import dataclass, field

import numpy as np

from trajectopy.core.settings import AlignmentEstimationSettings
from trajectopy.processing.lib.alignment.parameters import (
    AlignmentParameters,
    SensorRotationParameters,
)
from trajectopy.readers.header import HeaderData


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
            return False

        assert np.allclose(self.position_parameters.values, other.position_parameters.values)
        assert np.allclose(
            self.position_parameters.covariance_matrix,
            other.position_parameters.covariance_matrix,
        )
        assert np.allclose(self.rotation_parameters.values, other.rotation_parameters.values)
        assert self.estimation_of == other.estimation_of

        return True

    def to_file(self, filename: str) -> None:
        """
        Save the result to a file.

        Args:
            filename (str): Path to the file.
        """
        if self.position_parameters is None:
            raise ValueError("No estimated parameters available!")

        with open(filename, "a", newline="\n", encoding="utf-8") as file:
            file.write(f"#name {self.name}\n")

        self.position_parameters.to_dataframe().to_csv(
            filename, header=False, index=False, mode="a", float_format="%.15f"
        )
        self.rotation_parameters.to_file(filename=filename)

    @classmethod
    def from_file(cls, filename: str) -> "AlignmentResult":
        """
        Load the result from a file.

        Args:
            filename (str): Path to the file.

        Returns:
            AlignmentResult: The loaded result.
        """
        header_data = HeaderData.from_file(filename)
        estimated_parameters = AlignmentParameters.from_file(filename)
        sensor_rot_parameters = SensorRotationParameters.from_file(filename)
        return cls(
            name=str(header_data.data.get("name", "Alignment")),
            position_parameters=estimated_parameters,
            estimation_of=AlignmentEstimationSettings.from_bool_list(
                estimated_parameters.enabled_bool_list + sensor_rot_parameters.enabled_bool_list
            ),
            rotation_parameters=sensor_rot_parameters,
        )
