"""Orientations of a trajectory."""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation


class Rotations:
    """A sequence of 3D rotations, e.g. the orientations of the poses of a trajectory.

    The rotations are stored as a :class:`scipy.spatial.transform.Rotation`, which they wrap instead of
    extending it, so that trajectopy does not depend on scipy's internals. Use :meth:`to_scipy` and
    :meth:`from_scipy` to convert between both.

    Rotations are composed with ``*`` (``a * b`` applies ``b`` first, then ``a``), ``a + b`` is the same as
    ``a * b`` and ``a - b`` is ``a * b.inv()``. Indexing returns rotations as well, e.g. ``rotations[0]``
    or ``rotations[mask]``.
    """

    __slots__ = ("_rotation",)
    _rotation: Rotation

    def __init__(self, rotation: Rotation) -> None:
        """Wraps a scipy rotation. Prefer the constructors ``from_quat``, ``from_euler`` etc.

        Args:
            rotation (scipy.spatial.transform.Rotation): Rotation(s) to wrap.
        """
        if isinstance(rotation, Rotations):
            rotation = rotation._rotation
        if not isinstance(rotation, Rotation):
            raise TypeError(f"Expected a scipy Rotation, got {type(rotation).__name__}.")
        self._rotation = rotation

    # constructors

    @classmethod
    def from_scipy(cls, rotation: Rotation) -> "Rotations":
        """Creates rotations from a :class:`scipy.spatial.transform.Rotation`."""
        return cls(rotation)

    @classmethod
    def from_quat(cls, quat: ArrayLike) -> "Rotations":
        """Creates rotations from quaternions (x, y, z, w) of shape (4,) or (N, 4). They are normalized."""
        return cls(Rotation.from_quat(np.asarray(quat, dtype=float)))

    @classmethod
    def from_euler(cls, seq: str, angles: ArrayLike, degrees: bool = False) -> "Rotations":
        """Creates rotations from Euler angles, see :meth:`scipy.spatial.transform.Rotation.from_euler`."""
        if (quat := _euler_to_quat(seq, np.asarray(angles, dtype=float), degrees)) is not None:
            return cls.from_quat(quat)
        return cls(Rotation.from_euler(seq, angles, degrees))

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> "Rotations":
        """Creates rotations from rotation matrices of shape (3, 3) or (N, 3, 3)."""
        return cls(Rotation.from_matrix(np.asarray(matrix, dtype=float)))

    @classmethod
    def from_rotvec(cls, rotvec: ArrayLike, degrees: bool = False) -> "Rotations":
        """Creates rotations from rotation vectors (axis times angle) of shape (3,) or (N, 3)."""
        return cls(Rotation.from_rotvec(np.asarray(rotvec, dtype=float), degrees=degrees))

    @classmethod
    def identity(cls, num: int | None = None) -> "Rotations":
        """Identity rotation(s); a single one if ``num`` is None."""
        return cls(Rotation.identity(num))

    @classmethod
    def random(cls, num: int | None = None, rng: np.random.Generator | int | None = None) -> "Rotations":
        """Uniformly distributed random rotation(s); a single one if ``num`` is None."""
        quat = np.random.default_rng(rng).normal(size=(1 if num is None else num, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        return cls.from_quat(quat[0] if num is None else quat)

    # conversions

    def to_scipy(self) -> Rotation:
        """Returns the rotations as :class:`scipy.spatial.transform.Rotation`."""
        return self._rotation

    def as_quat(self) -> np.ndarray:
        """Quaternions (x, y, z, w), shape (4,) or (N, 4)."""
        return self._rotation.as_quat()

    def as_euler(self, seq: str, degrees: bool = False) -> np.ndarray:
        """Euler angles, shape (3,) or (N, 3), see :meth:`scipy.spatial.transform.Rotation.as_euler`."""
        return self._rotation.as_euler(seq, degrees=degrees)

    def as_matrix(self) -> np.ndarray:
        """Rotation matrices, shape (3, 3) or (N, 3, 3)."""
        return self._rotation.as_matrix()

    def as_rotvec(self, degrees: bool = False) -> np.ndarray:
        """Rotation vectors (axis times angle), shape (3,) or (N, 3)."""
        return self._rotation.as_rotvec(degrees=degrees)

    # operations

    def apply(self, vectors: ArrayLike, inverse: bool = False) -> np.ndarray:
        """Rotates vectors of shape (3,) or (N, 3)."""
        return self._rotation.apply(np.asarray(vectors, dtype=float), inverse=inverse)

    def inv(self) -> "Rotations":
        """The inverse rotations."""
        return Rotations(self._rotation.inv())

    def mean(self, weights: ArrayLike | None = None) -> "Rotations":
        """The (weighted) mean rotation."""
        return Rotations(self._rotation.mean(weights))

    def magnitude(self) -> np.ndarray | float:
        """Rotation angles in radians."""
        return self._rotation.magnitude()

    @property
    def rotangle(self) -> np.ndarray:
        """Rotation angles in radians, as array (also for a single rotation)."""
        return np.asarray(self._rotation.magnitude())

    @property
    def single(self) -> bool:
        """True if this is a single rotation rather than a sequence."""
        return bool(self._rotation.single)

    def copy(self) -> "Rotations":
        """Returns a copy."""
        return Rotations.from_quat(self.as_quat())

    def __mul__(self, other: "Rotations | Rotation") -> "Rotations":
        return Rotations(self._rotation * _scipy(other))

    def __add__(self, other: "Rotations | Rotation") -> "Rotations":
        return self * other

    def __sub__(self, other: "Rotations | Rotation") -> "Rotations":
        return Rotations(self._rotation * _scipy(other).inv())

    def __len__(self) -> int:
        # a single rotation counts as one, unlike scipy, which raises
        return 1 if self._rotation.single else len(self._rotation)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, key: Any) -> "Rotations":
        return Rotations(self._rotation[key])

    def __iter__(self) -> Iterator["Rotations"]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return f"<Rotations: {'single rotation' if self.single else f'{len(self)} rotations'}>"

    def __getstate__(self) -> np.ndarray:
        return self.as_quat()

    def __setstate__(self, quat: np.ndarray) -> None:
        self._rotation = Rotation.from_quat(quat)


def _scipy(rotation: "Rotations | Rotation") -> Rotation:
    return rotation.to_scipy() if isinstance(rotation, Rotations) else rotation


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
