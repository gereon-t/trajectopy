"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

from typing import Tuple, Union

import autograd.numpy as np


def functional_relationship_x(
    x_from: Union[float, np.ndarray] = 0.0,
    y_from: Union[float, np.ndarray] = 0.0,
    z_from: Union[float, np.ndarray] = 0.0,
    x_to: Union[float, np.ndarray] = 0.0,
    euler_x: Union[float, np.ndarray] = 0.0,
    euler_y: Union[float, np.ndarray] = 0.0,
    euler_z: Union[float, np.ndarray] = 0.0,
    speed_x: Union[float, np.ndarray] = 0.0,
    speed_y: Union[float, np.ndarray] = 0.0,
    speed_z: Union[float, np.ndarray] = 0.0,
    sim_trans_x: Union[float, np.ndarray] = 0.0,
    sim_rot_x: Union[float, np.ndarray] = 0.0,
    sim_rot_y: Union[float, np.ndarray] = 0.0,
    sim_rot_z: Union[float, np.ndarray] = 0.0,
    sim_scale: Union[float, np.ndarray] = 1.0,
    time_shift: Union[float, np.ndarray] = 0.0,
    lever_x: Union[float, np.ndarray] = 0.0,
    lever_y: Union[float, np.ndarray] = 0.0,
    lever_z: Union[float, np.ndarray] = 0.0,
) -> Union[float, np.ndarray]:
    """Evaluates the observation equations of the Gauß-Helmert-Model

    This method puts together the individual building blocks of the
    Helmert transform, lever arm estimation, and synchronization to
    obtain the corresponding functional relationship.


    Returns:
        np.ndarray: Result of the functional relationship
    """
    return -x_to + transformed_x_from(
        x_from=x_from,
        y_from=y_from,
        z_from=z_from,
        euler_x=euler_x,
        euler_y=euler_y,
        euler_z=euler_z,
        speed_x=speed_x,
        speed_y=speed_y,
        speed_z=speed_z,
        sim_trans_x=sim_trans_x,
        sim_rot_x=sim_rot_x,
        sim_rot_y=sim_rot_y,
        sim_rot_z=sim_rot_z,
        sim_scale=sim_scale,
        time_shift=time_shift,
        lever_x=lever_x,
        lever_y=lever_y,
        lever_z=lever_z,
    )


def functional_relationship_y(
    x_from: Union[float, np.ndarray] = 0.0,
    y_from: Union[float, np.ndarray] = 0.0,
    z_from: Union[float, np.ndarray] = 0.0,
    y_to: Union[float, np.ndarray] = 0.0,
    euler_x: Union[float, np.ndarray] = 0.0,
    euler_y: Union[float, np.ndarray] = 0.0,
    euler_z: Union[float, np.ndarray] = 0.0,
    speed_x: Union[float, np.ndarray] = 0.0,
    speed_y: Union[float, np.ndarray] = 0.0,
    speed_z: Union[float, np.ndarray] = 0.0,
    sim_trans_y: Union[float, np.ndarray] = 0.0,
    sim_rot_x: Union[float, np.ndarray] = 0.0,
    sim_rot_y: Union[float, np.ndarray] = 0.0,
    sim_rot_z: Union[float, np.ndarray] = 0.0,
    sim_scale: Union[float, np.ndarray] = 1.0,
    time_shift: Union[float, np.ndarray] = 0.0,
    lever_x: Union[float, np.ndarray] = 0.0,
    lever_y: Union[float, np.ndarray] = 0.0,
    lever_z: Union[float, np.ndarray] = 0.0,
) -> Union[float, np.ndarray]:
    """Evaluates the observation equations of the Gauß-Helmert-Model

    Returns:
        np.ndarray: Result of the functional relationship
    """
    return -y_to + transformed_y_from(
        x_from=x_from,
        y_from=y_from,
        z_from=z_from,
        euler_x=euler_x,
        euler_y=euler_y,
        euler_z=euler_z,
        speed_x=speed_x,
        speed_y=speed_y,
        speed_z=speed_z,
        sim_trans_y=sim_trans_y,
        sim_rot_x=sim_rot_x,
        sim_rot_y=sim_rot_y,
        sim_rot_z=sim_rot_z,
        sim_scale=sim_scale,
        time_shift=time_shift,
        lever_x=lever_x,
        lever_y=lever_y,
        lever_z=lever_z,
    )


def functional_relationship_z(
    x_from: Union[float, np.ndarray] = 0.0,
    y_from: Union[float, np.ndarray] = 0.0,
    z_from: Union[float, np.ndarray] = 0.0,
    z_to: Union[float, np.ndarray] = 0.0,
    euler_x: Union[float, np.ndarray] = 0.0,
    euler_y: Union[float, np.ndarray] = 0.0,
    euler_z: Union[float, np.ndarray] = 0.0,
    speed_x: Union[float, np.ndarray] = 0.0,
    speed_y: Union[float, np.ndarray] = 0.0,
    speed_z: Union[float, np.ndarray] = 0.0,
    sim_trans_z: Union[float, np.ndarray] = 0.0,
    sim_rot_x: Union[float, np.ndarray] = 0.0,
    sim_rot_y: Union[float, np.ndarray] = 0.0,
    sim_scale: Union[float, np.ndarray] = 1.0,
    time_shift: Union[float, np.ndarray] = 0.0,
    lever_x: Union[float, np.ndarray] = 0.0,
    lever_y: Union[float, np.ndarray] = 0.0,
    lever_z: Union[float, np.ndarray] = 0.0,
) -> Union[float, np.ndarray]:
    """Evaluates the observation equations of the Gauß-Helmert-Model

    Returns:
        np.ndarray: Result of the functional relationship
    """
    return -z_to + transformed_z_from(
        x_from=x_from,
        y_from=y_from,
        z_from=z_from,
        euler_x=euler_x,
        euler_y=euler_y,
        euler_z=euler_z,
        speed_x=speed_x,
        speed_y=speed_y,
        speed_z=speed_z,
        sim_trans_z=sim_trans_z,
        sim_rot_x=sim_rot_x,
        sim_rot_y=sim_rot_y,
        sim_scale=sim_scale,
        time_shift=time_shift,
        lever_x=lever_x,
        lever_y=lever_y,
        lever_z=lever_z,
    )


def leverarm_time_component(
    euler_x: Union[float, np.ndarray] = 0.0,
    euler_y: Union[float, np.ndarray] = 0.0,
    euler_z: Union[float, np.ndarray] = 0.0,
    speed_x: Union[float, np.ndarray] = 0.0,
    speed_y: Union[float, np.ndarray] = 0.0,
    speed_z: Union[float, np.ndarray] = 0.0,
    time_shift: Union[float, np.ndarray] = 0.0,
    lever_x: Union[float, np.ndarray] = 0.0,
    lever_y: Union[float, np.ndarray] = 0.0,
    lever_z: Union[float, np.ndarray] = 0.0,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    return (
        functional_leverarm_time_component_x(
            euler_x=euler_x,
            euler_y=euler_y,
            euler_z=euler_z,
            lever_x=lever_x,
            lever_y=lever_y,
            lever_z=lever_z,
            time_shift=time_shift,
            speed_x=speed_x,
        ),
        functional_leverarm_time_component_y(
            euler_x=euler_x,
            euler_y=euler_y,
            euler_z=euler_z,
            lever_x=lever_x,
            lever_y=lever_y,
            lever_z=lever_z,
            time_shift=time_shift,
            speed_y=speed_y,
        ),
        functional_leverarm_time_component_z(
            euler_x=euler_x,
            euler_y=euler_y,
            lever_x=lever_x,
            lever_y=lever_y,
            lever_z=lever_z,
            time_shift=time_shift,
            speed_z=speed_z,
        ),
    )


def transformed_x_from(
    x_from: Union[float, np.ndarray] = 0.0,
    y_from: Union[float, np.ndarray] = 0.0,
    z_from: Union[float, np.ndarray] = 0.0,
    euler_x: Union[float, np.ndarray] = 0.0,
    euler_y: Union[float, np.ndarray] = 0.0,
    euler_z: Union[float, np.ndarray] = 0.0,
    speed_x: Union[float, np.ndarray] = 0.0,
    speed_y: Union[float, np.ndarray] = 0.0,
    speed_z: Union[float, np.ndarray] = 0.0,
    sim_trans_x: Union[float, np.ndarray] = 0.0,
    sim_rot_x: Union[float, np.ndarray] = 0.0,
    sim_rot_y: Union[float, np.ndarray] = 0.0,
    sim_rot_z: Union[float, np.ndarray] = 0.0,
    sim_scale: Union[float, np.ndarray] = 1.0,
    time_shift: Union[float, np.ndarray] = 0.0,
    lever_x: Union[float, np.ndarray] = 0.0,
    lever_y: Union[float, np.ndarray] = 0.0,
    lever_z: Union[float, np.ndarray] = 0.0,
) -> Union[float, np.ndarray]:
    x_lt, y_lt, z_lt = leverarm_time_component(
        euler_x=euler_x,
        euler_y=euler_y,
        euler_z=euler_z,
        lever_x=lever_x,
        lever_y=lever_y,
        lever_z=lever_z,
        time_shift=time_shift,
        speed_x=speed_x,
        speed_y=speed_y,
        speed_z=speed_z,
    )
    return functional_helmert_component_x(
        x_from=x_from + x_lt,
        y_from=y_from + y_lt,
        z_from=z_from + z_lt,
        sim_trans_x=sim_trans_x,
        sim_rot_x=sim_rot_x,
        sim_rot_y=sim_rot_y,
        sim_rot_z=sim_rot_z,
        sim_scale=sim_scale,
    )


def transformed_y_from(
    x_from: Union[float, np.ndarray] = 0.0,
    y_from: Union[float, np.ndarray] = 0.0,
    z_from: Union[float, np.ndarray] = 0.0,
    euler_x: Union[float, np.ndarray] = 0.0,
    euler_y: Union[float, np.ndarray] = 0.0,
    euler_z: Union[float, np.ndarray] = 0.0,
    speed_x: Union[float, np.ndarray] = 0.0,
    speed_y: Union[float, np.ndarray] = 0.0,
    speed_z: Union[float, np.ndarray] = 0.0,
    sim_trans_y: Union[float, np.ndarray] = 0.0,
    sim_rot_x: Union[float, np.ndarray] = 0.0,
    sim_rot_y: Union[float, np.ndarray] = 0.0,
    sim_rot_z: Union[float, np.ndarray] = 0.0,
    sim_scale: Union[float, np.ndarray] = 1.0,
    time_shift: Union[float, np.ndarray] = 0.0,
    lever_x: Union[float, np.ndarray] = 0.0,
    lever_y: Union[float, np.ndarray] = 0.0,
    lever_z: Union[float, np.ndarray] = 0.0,
) -> Union[float, np.ndarray]:
    x_lt, y_lt, z_lt = leverarm_time_component(
        euler_x=euler_x,
        euler_y=euler_y,
        euler_z=euler_z,
        lever_x=lever_x,
        lever_y=lever_y,
        lever_z=lever_z,
        time_shift=time_shift,
        speed_x=speed_x,
        speed_y=speed_y,
        speed_z=speed_z,
    )
    return functional_helmert_component_y(
        x_from=x_from + x_lt,
        y_from=y_from + y_lt,
        z_from=z_from + z_lt,
        sim_trans_y=sim_trans_y,
        sim_rot_x=sim_rot_x,
        sim_rot_y=sim_rot_y,
        sim_rot_z=sim_rot_z,
        sim_scale=sim_scale,
    )


def transformed_z_from(
    x_from: Union[float, np.ndarray] = 0.0,
    y_from: Union[float, np.ndarray] = 0.0,
    z_from: Union[float, np.ndarray] = 0.0,
    euler_x: Union[float, np.ndarray] = 0.0,
    euler_y: Union[float, np.ndarray] = 0.0,
    euler_z: Union[float, np.ndarray] = 0.0,
    speed_x: Union[float, np.ndarray] = 0.0,
    speed_y: Union[float, np.ndarray] = 0.0,
    speed_z: Union[float, np.ndarray] = 0.0,
    sim_trans_z: Union[float, np.ndarray] = 0.0,
    sim_rot_x: Union[float, np.ndarray] = 0.0,
    sim_rot_y: Union[float, np.ndarray] = 0.0,
    sim_scale: Union[float, np.ndarray] = 1.0,
    time_shift: Union[float, np.ndarray] = 0.0,
    lever_x: Union[float, np.ndarray] = 0.0,
    lever_y: Union[float, np.ndarray] = 0.0,
    lever_z: Union[float, np.ndarray] = 0.0,
) -> Union[float, np.ndarray]:
    x_lt, y_lt, z_lt = leverarm_time_component(
        euler_x=euler_x,
        euler_y=euler_y,
        euler_z=euler_z,
        lever_x=lever_x,
        lever_y=lever_y,
        lever_z=lever_z,
        time_shift=time_shift,
        speed_x=speed_x,
        speed_y=speed_y,
        speed_z=speed_z,
    )
    return functional_helmert_component_z(
        x_from=x_from + x_lt,
        y_from=y_from + y_lt,
        z_from=z_from + z_lt,
        sim_trans_z=sim_trans_z,
        sim_rot_x=sim_rot_x,
        sim_rot_y=sim_rot_y,
        sim_scale=sim_scale,
    )


def functional_helmert_component_x(
    x_from: Union[float, np.ndarray],
    y_from: Union[float, np.ndarray],
    z_from: Union[float, np.ndarray],
    sim_trans_x: Union[float, np.ndarray],
    sim_rot_x: Union[float, np.ndarray],
    sim_rot_y: Union[float, np.ndarray],
    sim_rot_z: Union[float, np.ndarray],
    sim_scale: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Helper function returning the helmert component of the functional relationship

    Returns:
        np.ndarray: helmert component
    """

    a = np.cos(sim_rot_y) * np.cos(sim_rot_z)
    b = np.sin(sim_rot_x) * np.sin(sim_rot_y) * np.cos(sim_rot_z) - np.cos(sim_rot_x) * np.sin(sim_rot_z)
    c = np.cos(sim_rot_x) * np.sin(sim_rot_y) * np.cos(sim_rot_z) + np.sin(sim_rot_x) * np.sin(sim_rot_z)

    return sim_trans_x + sim_scale * (a * x_from + b * y_from + c * z_from)


def functional_helmert_component_y(
    x_from: Union[float, np.ndarray],
    y_from: Union[float, np.ndarray],
    z_from: Union[float, np.ndarray],
    sim_trans_y: Union[float, np.ndarray],
    sim_rot_x: Union[float, np.ndarray],
    sim_rot_y: Union[float, np.ndarray],
    sim_rot_z: Union[float, np.ndarray],
    sim_scale: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Helper function returning the helmert component of the functional relationship

    Returns:
        np.ndarray: helmert component
    """
    d = np.cos(sim_rot_y) * np.sin(sim_rot_z)
    e = np.sin(sim_rot_x) * np.sin(sim_rot_y) * np.sin(sim_rot_z) + np.cos(sim_rot_x) * np.cos(sim_rot_z)
    f = np.cos(sim_rot_x) * np.sin(sim_rot_y) * np.sin(sim_rot_z) - np.sin(sim_rot_x) * np.cos(sim_rot_z)

    return sim_trans_y + sim_scale * (d * x_from + e * y_from + f * z_from)


def functional_helmert_component_z(
    x_from: Union[float, np.ndarray],
    y_from: Union[float, np.ndarray],
    z_from: Union[float, np.ndarray],
    sim_trans_z: Union[float, np.ndarray],
    sim_rot_x: Union[float, np.ndarray],
    sim_rot_y: Union[float, np.ndarray],
    sim_scale: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Helper function returning the helmert component of the functional relationship

    Returns:
        np.ndarray: helmert component
    """
    g = -np.sin(sim_rot_y)
    h = np.sin(sim_rot_x) * np.cos(sim_rot_y)
    i = np.cos(sim_rot_x) * np.cos(sim_rot_y)

    return sim_trans_z + sim_scale * (g * x_from + h * y_from + i * z_from)


def functional_leverarm_time_component_x(
    euler_x: Union[float, np.ndarray],
    euler_y: Union[float, np.ndarray],
    euler_z: Union[float, np.ndarray],
    lever_x: Union[float, np.ndarray],
    lever_y: Union[float, np.ndarray],
    lever_z: Union[float, np.ndarray],
    time_shift: Union[float, np.ndarray],
    speed_x: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Helper function returning the helmert component of the functional relationship

    Returns:
        np.ndarray: helmert component
    """
    a = np.cos(euler_y) * np.cos(euler_z)
    b = np.sin(euler_x) * np.sin(euler_y) * np.cos(euler_z) - np.cos(euler_x) * np.sin(euler_z)
    c = np.cos(euler_x) * np.sin(euler_y) * np.cos(euler_z) + np.sin(euler_x) * np.sin(euler_z)

    return a * lever_x + b * lever_y + c * lever_z + time_shift * speed_x


def functional_leverarm_time_component_y(
    euler_x: Union[float, np.ndarray],
    euler_y: Union[float, np.ndarray],
    euler_z: Union[float, np.ndarray],
    lever_x: Union[float, np.ndarray],
    lever_y: Union[float, np.ndarray],
    lever_z: Union[float, np.ndarray],
    time_shift: Union[float, np.ndarray],
    speed_y: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Helper function returning the helmert component of the functional relationship

    Returns:
        np.ndarray: helmert component
    """
    d = np.cos(euler_y) * np.sin(euler_z)
    e = np.sin(euler_x) * np.sin(euler_y) * np.sin(euler_z) + np.cos(euler_x) * np.cos(euler_z)
    f = np.cos(euler_x) * np.sin(euler_y) * np.sin(euler_z) - np.sin(euler_x) * np.cos(euler_z)

    return d * lever_x + e * lever_y + f * lever_z + time_shift * speed_y


def functional_leverarm_time_component_z(
    euler_x: Union[float, np.ndarray],
    euler_y: Union[float, np.ndarray],
    lever_x: Union[float, np.ndarray],
    lever_y: Union[float, np.ndarray],
    lever_z: Union[float, np.ndarray],
    time_shift: Union[float, np.ndarray],
    speed_z: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Helper function returning the helmert component of the functional relationship

    Returns:
        np.ndarray: helmert component
    """
    g = -np.sin(euler_y)
    h = np.sin(euler_x) * np.cos(euler_y)
    i = np.cos(euler_x) * np.cos(euler_y)

    return g * lever_x + h * lever_y + i * lever_z + time_shift * speed_z
