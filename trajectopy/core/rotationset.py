"""
Gereon Tombrink, 2024
mail@gtombrink.de
"""

import copy

import numpy as np
from scipy.spatial.transform import Rotation


class RotationSet(Rotation):
    """Class representing rotations

    This class is essentially just a wrapper around the parent
    scipy.spatial.transform.Rotation class. It extends its
    functionality by introducing + / - operators as well as
    the ability to create a deepcopy and to output rotation angles.

    Furthermore, it ensures that the naming is consistent with the
    pointset class.
    """

    @classmethod
    def from_euler(cls, seq: str, angles: np.ndarray, degrees: bool = False) -> "RotationSet":
        return super().from_euler(seq, angles, degrees)

    @classmethod
    def from_quat(cls, quat: np.ndarray) -> "RotationSet":
        return super().from_quat(quat)

    def __sub__(self, other: "RotationSet") -> "RotationSet":
        return self * other.inv()

    def __add__(self, other: "RotationSet") -> "RotationSet":
        return self * other

    def copy(self) -> "RotationSet":
        return copy.deepcopy(self)

    @property
    def rotangle(self) -> np.ndarray:
        """Returns minimum rotation angle(s)"""
        rotvec = self.as_rotvec()

        if rotvec.ndim == 1:
            return np.array(np.linalg.norm(rotvec))

        return np.linalg.norm(rotvec, axis=1)

    # def mean(self) -> "RotationSet":
    #     """
    #     Function that computes the chordal l2 mean of multiple
    #     rotations given in quaternion representation
    #     """
    #     a = np.zeros((4, 4))
    #     quat = self.as_quat()
    #     for i in range(len(quat)):
    #         # outer product
    #         a += np.outer(quat[i, :], quat[i, :])
    #     return RotationSet.from_quat(np.linalg.eigh(a)[1][:, -1])
