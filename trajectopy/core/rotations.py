import copy

import numpy as np
from scipy.spatial.transform import Rotation


class Rotations(Rotation):
    """Class representing rotations.

    This class is essentially just a wrapper around the parent scipy.spatial.transform.Rotation
    class. It extends its functionality by introducing + / - operators as well as the ability
    to create a deepcopy and to output rotation angles. Furthermore, it ensures that the naming
    is consistent with the pointset class.
    """

    def __len__(self) -> int:
        return self.as_quat().shape[0]

    @classmethod
    def from_euler(cls, seq: str, angles: np.ndarray, degrees: bool = False) -> "Rotations":
        rot = Rotation.from_euler(seq, angles, degrees)
        return cls.from_quat(rot.as_quat())

    @classmethod
    def from_quat(cls, quat: np.ndarray) -> "Rotations":
        # Use object.__new__ to create instance, then initialize via Rotation
        instance = super().from_quat(quat)
        instance.__class__ = cls
        return instance

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "Rotations":
        instance = super().from_matrix(matrix)
        instance.__class__ = cls
        return instance

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> "Rotations":
        instance = super().from_rotvec(rotvec)
        instance.__class__ = cls
        return instance

    @classmethod
    def identity(cls, num: int = 1) -> "Rotations":
        instance = super().identity(num)
        instance.__class__ = cls
        return instance

    def __mul__(self, other: "Rotation") -> "Rotations":
        result = super().__mul__(other)
        return Rotations.from_quat(result.as_quat())

    def __sub__(self, other: "Rotations") -> "Rotations":
        result = self * other.inv()
        return Rotations.from_quat(result.as_quat())

    def __add__(self, other: "Rotations") -> "Rotations":
        result = self * other
        return Rotations.from_quat(result.as_quat())

    def inv(self) -> "Rotations":
        result = super().inv()
        return Rotations.from_quat(result.as_quat())

    def mean(self, weights: np.ndarray | None = None) -> "Rotations":
        result = super().mean(weights)
        return Rotations.from_quat(result.as_quat())

    def copy(self) -> "Rotations":
        return copy.deepcopy(self)

    @property
    def rotangle(self) -> np.ndarray:
        """Returns minimum rotation angle(s).

        Returns:
            np.ndarray: Minimum rotation angle(s) in radians.
        """
        rotvec = self.as_rotvec()

        if rotvec.ndim == 1:
            return np.array(np.linalg.norm(rotvec))

        return np.linalg.norm(rotvec, axis=1)

    # def mean(self) -> "Rotations":
    #     """
    #     Function that computes the chordal l2 mean of multiple
    #     rotations given in quaternion representation
    #     """
    #     a = np.zeros((4, 4))
    #     quat = self.as_quat()
    #     for i in range(len(quat)):
    #         # outer product
    #         a += np.outer(quat[i, :], quat[i, :])
    #     return Rotations.from_quat(np.linalg.eigh(a)[1][:, -1])
