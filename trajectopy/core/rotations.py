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
        if (quat := _euler_to_quat(seq, angles, degrees)) is None:
            quat = Rotation.from_euler(seq, angles, degrees).as_quat()
        return cls.from_quat(quat)

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


_AXES = {"x": 0, "y": 1, "z": 2}


def _euler_to_quat(seq: str, angles: np.ndarray, degrees: bool) -> np.ndarray | None:
    """Quaternions (scalar last) for many Euler angle triples, composed like scipy's ``from_euler`` but
    vectorized with numpy, which is several times faster for millions of rotations.

    Returns None for input this fast path does not handle (scipy is used then).
    """
    angles = np.asarray(angles, dtype=float)
    lowered = seq.lower()
    if (
        not 1 <= len(seq) <= 3
        or not (seq.islower() or seq.isupper())
        or any(axis not in _AXES for axis in lowered)
        or any(a == b for a, b in zip(lowered, lowered[1:]))  # scipy rejects consecutive identical axes
        or angles.ndim != 2
        or angles.shape[1] != len(seq)
    ):
        return None

    half = (np.deg2rad(angles) if degrees else angles) / 2
    zeros = np.zeros(len(angles))
    quat: tuple[np.ndarray, ...] | None = None
    for column, axis in enumerate(lowered):
        elementary = [zeros, zeros, zeros, np.cos(half[:, column])]
        elementary[_AXES[axis]] = np.sin(half[:, column])
        if quat is None:
            quat = tuple(elementary)
        elif seq.isupper():  # intrinsic: rotate about the axes of the rotated frame
            quat = _quat_multiply(quat, elementary)
        else:  # extrinsic: rotate about the fixed axes
            quat = _quat_multiply(elementary, quat)
    return np.column_stack(quat)  # type: ignore[arg-type]


def _quat_multiply(p, q) -> tuple[np.ndarray, ...]:
    """Hamilton product of scalar-last quaternions given as (x, y, z, w) component arrays
    (rotation ``q`` followed by ``p``)."""
    px, py, pz, pw = p
    qx, qy, qz, qw = q
    return (
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
        pw * qw - px * qx - py * qy - pz * qz,
    )
