import numpy as np
from pointset import PointSet

from trajectopy_core.trajectory import Trajectory
from trajectopy_core.util.rotationset import RotationSet


def random_number(min: float, max: float) -> float:
    return np.random.rand() * (max - min) + min


def generate_noisy_trajectory(num: int) -> Trajectory:
    xyz = np.c_[
        np.sin(np.linspace(0, 2 * np.pi, num)) * 30 + np.random.randn(num) * 0.25,
        np.cos(np.linspace(0, 2 * np.pi, num)) * 30 + np.random.randn(num) * 0.25,
        np.cos(np.linspace(0, 2 * np.pi, num)) * 10 + np.random.randn(num) * 0.01,
    ]
    rpy = np.c_[
        np.linspace(0, 2 * np.pi, num),
        np.linspace(0, 2 * np.pi, num),
        np.linspace(0, 2 * np.pi, num),
    ] + np.random.randn(num, 3) * np.deg2rad(0.1)

    tstamps = np.arange(0, num, dtype=np.float64)
    all = np.c_[tstamps, xyz, rpy]
    indices = np.unique(all, axis=0, return_index=True)[1]

    tstamps = tstamps[indices]
    xyz = xyz[indices]
    rpy = rpy[indices]

    pointset = PointSet(xyz=xyz, epsg=0)
    rotationset = RotationSet.from_euler(seq="xyz", angles=rpy, degrees=False)

    return Trajectory(pos=pointset, rot=rotationset, tstamps=tstamps)


def generate_trajectory(num: int) -> Trajectory:
    xyz = np.c_[
        np.sin(np.linspace(0, 2 * np.pi, num)) * 1000,
        np.cos(np.linspace(0, 2 * np.pi, num)) * 100,
        np.cos(np.linspace(0, 2 * np.pi, num)) * 10,
    ]
    rpy = np.c_[
        np.linspace(0, 2 * np.pi, num),
        np.linspace(0, 2 * np.pi, num),
        np.linspace(0, 2 * np.pi, num),
    ]

    tstamps = np.arange(0, num, dtype=np.float64)
    all = np.c_[tstamps, xyz, rpy]
    indices = np.unique(all, axis=0, return_index=True)[1]

    tstamps = tstamps[indices]
    xyz = xyz[indices]
    rpy = rpy[indices]

    pointset = PointSet(xyz=xyz, epsg=0)
    rotationset = RotationSet.from_euler(seq="xyz", angles=rpy, degrees=False)

    return Trajectory(pos=pointset, rot=rotationset, tstamps=tstamps)
