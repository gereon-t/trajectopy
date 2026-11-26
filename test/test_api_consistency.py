"""The conventions of the API policy hold for every function, also for the less common ones."""

import dataclasses
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

import trajectopy as tpy
from trajectopy import io


def _trajectory() -> tpy.Trajectory:
    angle = np.linspace(0.0, 2 * np.pi, 100, endpoint=False)
    return tpy.Trajectory(np.column_stack([np.cos(angle), np.sin(angle), 0 * angle]), name="circle")


class TestOptionsAsStrings(unittest.TestCase):
    def test_sorting_accepts_strings(self):
        trajectory = tpy.Trajectory(_trajectory().xyz, sorting="Path_Length")
        self.assertIs(trajectory.sorting, tpy.Sorting.PATH_LENGTH)
        self.assertIn("sorted by path_length", repr(trajectory))

        trajectory.set_sorting("time")
        self.assertIs(trajectory.sorting, tpy.Sorting.TIME)

    def test_ate_result_sorting_accepts_strings(self):
        trajectory = _trajectory()
        result = tpy.ate(trajectory, trajectory, align=False)
        result.set_sorting("path_length")
        self.assertIs(result.trajectory.sorting, tpy.Sorting.PATH_LENGTH)


class TestInplace(unittest.TestCase):
    def test_positions_mean_returns_the_input(self):
        positions = tpy.Positions(np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]))
        self.assertIs(positions.mean(inplace=True), positions)
        np.testing.assert_allclose(positions.xyz, [[1.0, 2.0, 3.0]])


class TestKeywordOnlyOptions(unittest.TestCase):
    def test_boolean_options_are_keyword_only(self):
        """Flags are the options that are most easily passed by position by mistake."""
        namespaces = [tpy, io]
        offenders = []
        for namespace in namespaces:
            for name in namespace.__all__:
                obj = getattr(namespace, name)
                functions = [(name, obj)] if inspect.isfunction(obj) else []
                if inspect.isclass(obj):
                    functions += [
                        (f"{name}.{member}", function)
                        for member, function in inspect.getmembers(obj, inspect.isfunction)
                        # the constructors of dataclasses take their fields, e.g. the settings
                        if not member.startswith("_") or (member == "__init__" and not dataclasses.is_dataclass(obj))
                    ]
                for qualname, function in functions:
                    for parameter in inspect.signature(function).parameters.values():
                        if parameter.kind is parameter.POSITIONAL_OR_KEYWORD and isinstance(parameter.default, bool):
                            offenders.append(f"{qualname}({parameter.name})")
        self.assertEqual(offenders, [])


class TestPoses(unittest.TestCase):
    def test_iterated_poses_equal_indexed_poses(self):
        trajectory = _trajectory()
        poses = list(trajectory)
        self.assertIsInstance(poses[3].velocity, tuple)
        self.assertEqual(poses[3], trajectory[3])


class TestAlignmentResults(unittest.TestCase):
    def test_ate_without_alignment_has_none(self):
        trajectory = _trajectory()
        self.assertIsNone(tpy.ate(trajectory, trajectory, align=False).alignment)

    def test_converged_flag_round_trip(self):
        reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        alignment = tpy.estimate_alignment(estimate, reference)
        with tempfile.TemporaryDirectory() as directory:
            for converged in (True, False):
                filename = Path(directory) / f"{converged}.result"
                result = dataclasses.replace(alignment, converged=converged)
                result.to_file(filename)
                self.assertEqual(tpy.AlignmentResult.from_file(filename), result)
                self.assertEqual(tpy.AlignmentResult.from_file(filename).converged, converged)


if __name__ == "__main__":
    unittest.main()
