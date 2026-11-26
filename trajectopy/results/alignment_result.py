import os
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
    """Class to store the result of an alignment.

    Attributes:
        name (str): Name of the result.
        position_parameters (AlignmentParameters): Estimated position alignment parameters.
        rotation_parameters (SensorRotationParameters): Estimated rotation alignment parameters.
        estimation_of (AlignmentEstimationSettings): Settings defining which parameters were estimated.
        converged (bool): Whether the estimation converged.

    See Also:
        [`trajectopy.estimate_alignment`][trajectopy.estimate_alignment]: computes the result.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> alignment = tpy.estimate_alignment(estimate, reference)
        >>> alignment
        <AlignmentResult 'KITTI ORB to KITTI Ground Truth': Helmert estimated, converged>
    """

    name: str = "Alignment Result"
    position_parameters: AlignmentParameters = field(default_factory=AlignmentParameters)
    rotation_parameters: SensorRotationParameters = field(default_factory=SensorRotationParameters)
    estimation_of: AlignmentEstimationSettings = field(default_factory=AlignmentEstimationSettings)
    converged: bool = True

    def __repr__(self) -> str:
        state = "converged" if self.converged else "not converged"
        return f"<AlignmentResult {self.name!r}: {self.estimation_of.short_mode_str.strip() or 'nothing'} estimated, {state}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, AlignmentResult):
            return NotImplemented

        return (
            _allclose(self.position_parameters.values, other.position_parameters.values)
            and _allclose(self.position_parameters.covariance_matrix, other.position_parameters.covariance_matrix)
            and _allclose(self.rotation_parameters.values, other.rotation_parameters.values)
            and self.estimation_of == other.estimation_of
            and self.converged == other.converged
        )

    def to_file(self, filename: str | os.PathLike, *, mode: str = "w") -> None:
        """Writes the result to a file that :meth:`from_file` and the GUI can read.

        Args:
            filename (str | os.PathLike): Path of the file.
            mode (str, optional): ``"w"`` (default) overwrites the file, ``"a"`` appends to it.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
            >>> alignment = tpy.estimate_alignment(estimate, reference)
            >>> alignment.to_file("alignment.result")
            >>> tpy.AlignmentResult.from_file("alignment.result") == alignment
            True
        """
        from trajectopy.io import results_io

        filename = os.fspath(filename)
        results_io.write_alignment_result(self, filename, mode=mode)

    @classmethod
    def from_file(cls, filename: str | os.PathLike) -> "AlignmentResult":
        """Reads a result written by :meth:`to_file`.

        Args:
            filename (str | os.PathLike): Path of the file.

        Returns:
            AlignmentResult: The result.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
            >>> alignment = tpy.estimate_alignment(estimate, reference)
            >>> alignment.to_file("alignment.result")
            >>> tpy.AlignmentResult.from_file("alignment.result").name
            'KITTI ORB to KITTI Ground Truth'
        """
        from trajectopy.io import results_io

        filename = os.fspath(filename)
        return results_io.read_alignment_result(filename)
