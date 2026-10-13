"""Contract of the ``inplace`` flag shared by all trajectory-transforming functions.

* ``inplace=False`` must never modify the inputs and must return a new object.
* ``inplace=True`` must return the (modified) input object itself.
* Functions without an ``inplace`` flag (``ate``, ``rpe``, ``estimate_alignment``) never modify their inputs.
"""

import unittest
from collections.abc import Callable

import numpy as np

from test.testdata import generated_trajectory, noisy_trajectory
from trajectopy.core.settings import ApproximationSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.alignment import (
    adopt_first_orientation,
    adopt_first_pose,
    adopt_first_position,
    align,
    apply_alignment,
    estimate_alignment,
)
from trajectopy.processing.approximation import approximate
from trajectopy.processing.evaluation import ate, rpe
from trajectopy.processing.interpolation import interpolate
from trajectopy.processing.matching import match_stamps, match_trajectories


def _snapshot(trajectory: Trajectory) -> dict[str, np.ndarray]:
    return {
        "timestamps": trajectory.timestamps.copy(),
        "xyz": trajectory.xyz.copy(),
        "quat": trajectory.quat.copy(),
        "path_lengths": trajectory.path_lengths.copy(),
        "epsg": np.array(trajectory.positions.epsg),
    }


class TestInplaceContract(unittest.TestCase):
    def setUp(self) -> None:
        self.test = noisy_trajectory.copy()
        self.ref = generated_trajectory.copy()

    def assertUnchanged(self, trajectory: Trajectory, snapshot: dict[str, np.ndarray]) -> None:
        current = _snapshot(trajectory)
        for key, value in snapshot.items():
            np.testing.assert_array_equal(current[key], value, err_msg=f"'{key}' was modified")

    def _check_single(self, func: Callable[..., Trajectory]) -> None:
        before_test, before_ref = _snapshot(self.test), _snapshot(self.ref)
        result = func(inplace=False)
        self.assertIsNot(result, self.test)
        self.assertUnchanged(self.test, before_test)
        self.assertUnchanged(self.ref, before_ref)

        result = func(inplace=True)
        self.assertIs(result, self.test)

    def test_align(self):
        self._check_single(lambda inplace: align(self.test, self.ref, inplace=inplace))

    def test_apply_alignment(self):
        alignment = estimate_alignment(self.test, self.ref)
        self._check_single(lambda inplace: apply_alignment(self.test, alignment, inplace=inplace))

    def test_adopt_first(self):
        for func in (adopt_first_pose, adopt_first_position, adopt_first_orientation):
            with self.subTest(func=func.__name__):
                self._check_single(lambda inplace, func=func: func(self.test, self.ref, inplace=inplace))

    def test_approximate(self):
        self._check_single(lambda inplace: approximate(self.test, ApproximationSettings(), inplace=inplace))

    def test_interpolate(self):
        index = self.test.timestamps[::2] + 0.5
        self._check_single(lambda inplace: interpolate(self.test, index, inplace=inplace))

    def test_match_stamps(self):
        self._check_single(lambda inplace: match_stamps(self.test, self.ref.timestamps[::2], inplace=inplace))

    def test_match_trajectories(self):
        before_test, before_ref = _snapshot(self.test), _snapshot(self.ref)
        matched_test, matched_ref = match_trajectories(self.test, self.ref, inplace=False)
        self.assertIsNot(matched_test, self.test)
        self.assertIsNot(matched_ref, self.ref)
        self.assertUnchanged(self.test, before_test)
        self.assertUnchanged(self.ref, before_ref)

        matched_test, matched_ref = match_trajectories(self.test, self.ref, inplace=True)
        self.assertIs(matched_test, self.test)
        self.assertIs(matched_ref, self.ref)

    def test_trajectory_methods(self):
        start, end = self.test.timestamps[10], self.test.timestamps[-10]
        mask = np.arange(len(self.test)) % 2 == 0
        transformation = np.eye(4)
        transformation[:3, 3] = [1.0, 2.0, 3.0]
        for name, func in {
            "crop": lambda inplace: self.test.crop(start, end, inplace=inplace),
            "mask": lambda inplace: self.test.mask(mask[: len(self.test)], inplace=inplace),
            "transform": lambda inplace: self.test.transform(transformation, inplace=inplace),
            "intersect": lambda inplace: self.test.intersect(self.ref.timestamps[5:-5], inplace=inplace),
        }.items():
            with self.subTest(method=name):
                self.test = noisy_trajectory.copy()
                self._check_single(func)

    def test_evaluation_does_not_modify_inputs(self):
        before_test, before_ref = _snapshot(self.test), _snapshot(self.ref)
        estimate_alignment(self.test, self.ref)
        ate(self.test, self.ref)
        rpe(self.test, self.ref)
        self.assertUnchanged(self.test, before_test)
        self.assertUnchanged(self.ref, before_ref)


if __name__ == "__main__":
    unittest.main()
