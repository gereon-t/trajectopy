"""Fast construction of scipy rotations from Euler angles."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation


def rotation_from_euler(seq: str, angles: ArrayLike, degrees: bool = False) -> Rotation:
    """Same as :meth:`scipy.spatial.transform.Rotation.from_euler`, but several times faster for many angles.

    Trajectory files can hold millions of orientations as roll, pitch and yaw angles; the quaternions of such
    sequences are composed with vectorized numpy operations. Other input is passed on to scipy.
    """
    angles = np.asarray(angles, dtype=float)
    if (quat := _euler_to_quat(seq, angles, degrees)) is not None:
        return Rotation.from_quat(quat)
    return Rotation.from_euler(seq, angles, degrees=degrees)


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
