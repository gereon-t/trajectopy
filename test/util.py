from typing import Tuple

import numpy as np

from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.alignment import apply_alignment
from trajectopy.processing.lib.alignment.parameters import (
    AlignmentParameters,
    Parameter,
)
from trajectopy.results.alignment_result import AlignmentResult
from trajectopy.utils.definitions import Unit


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

    pointset = Positions(xyz=xyz, epsg=0)
    rotationset = Rotations.from_euler(seq="xyz", angles=rpy, degrees=False)

    return Trajectory(positions=pointset, rotations=rotationset, timestamps=tstamps)


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

    pointset = Positions(xyz=xyz, epsg=0)
    rotationset = Rotations.from_euler(seq="xyz", angles=rpy, degrees=False)

    return Trajectory(positions=pointset, rotations=rotationset, timestamps=tstamps)


def random_number(lower_bound: float, upper_bound: float) -> float:
    return np.random.rand() * (upper_bound - lower_bound) + lower_bound


def generate_transformation(
    similarity_enabled: bool = True,
    time_shift_enabled: bool = True,
    lever_enabled: bool = True,
):
    if similarity_enabled:
        rand_rot = Rotations.random().as_euler("xyz", degrees=False) * 0.2
        sim_trans_x = Parameter(value=random_number(-100, upper_bound=100), unit=Unit.METER)
        sim_trans_y = Parameter(value=random_number(-100, upper_bound=100), unit=Unit.METER)
        sim_trans_z = Parameter(value=random_number(-100, upper_bound=100), unit=Unit.METER)
        sim_rot_x = Parameter(value=rand_rot[0], unit=Unit.RADIAN)
        sim_rot_y = Parameter(value=rand_rot[1], unit=Unit.RADIAN)
        sim_rot_z = Parameter(value=rand_rot[2], unit=Unit.RADIAN)
        sim_scale = Parameter(value=random_number(0.1, upper_bound=10), unit=Unit.SCALE)
    else:
        sim_trans_x = Parameter(value=0, unit=Unit.METER)
        sim_trans_y = Parameter(value=0, unit=Unit.METER)
        sim_trans_z = Parameter(value=0, unit=Unit.METER)
        sim_rot_x = Parameter(value=0, unit=Unit.RADIAN)
        sim_rot_y = Parameter(value=0, unit=Unit.RADIAN)
        sim_rot_z = Parameter(value=0, unit=Unit.RADIAN)
        sim_scale = Parameter(value=1, unit=Unit.SCALE)

    if time_shift_enabled:
        time_shift = Parameter(value=random_number(lower_bound=-10, upper_bound=10), unit=Unit.SECOND)
    else:
        time_shift = Parameter(value=0, unit=Unit.SECOND)

    if lever_enabled:
        lever_x = Parameter(value=random_number(-5, upper_bound=5), unit=Unit.METER)
        lever_y = Parameter(value=random_number(-5, upper_bound=5), unit=Unit.METER)
        lever_z = Parameter(value=random_number(-5, upper_bound=5), unit=Unit.METER)
    else:
        lever_x = Parameter(value=0, unit=Unit.METER)
        lever_y = Parameter(value=0, unit=Unit.METER)
        lever_z = Parameter(value=0, unit=Unit.METER)

    return AlignmentParameters(
        sim_trans_x=sim_trans_x,
        sim_trans_y=sim_trans_y,
        sim_trans_z=sim_trans_z,
        sim_rot_x=sim_rot_x,
        sim_rot_y=sim_rot_y,
        sim_rot_z=sim_rot_z,
        sim_scale=sim_scale,
        time_shift=time_shift,
        lever_x=lever_x,
        lever_y=lever_y,
        lever_z=lever_z,
    )


def transform_randomly(
    trajectory: Trajectory,
    similarity_enabled: bool = True,
    time_shift_enabled: bool = True,
    lever_enabled: bool = True,
) -> Tuple[Trajectory, AlignmentParameters]:
    parameters = generate_transformation(
        similarity_enabled=similarity_enabled, time_shift_enabled=time_shift_enabled, lever_enabled=lever_enabled
    )
    transformed = apply_alignment(
        trajectory, alignment_result=AlignmentResult(position_parameters=parameters), inplace=False
    )
    return transformed, parameters
