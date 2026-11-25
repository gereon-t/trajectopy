"""Tests of the GUI-independent building blocks of the interactive alignment."""

import unittest

import numpy as np

from test.testdata import generated_trajectory
from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing import alignment
from trajectopy.processing._lib.alignment import interactive
from trajectopy.processing._lib.alignment.interactive import AlignmentPreview, AlignmentState

UTM_OFFSET = np.array([364800.0, 5621500.0, 60.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


def transformed(trajectory: Trajectory, yaw_deg: float, translation, scale: float = 1.0) -> Trajectory:
    """``trajectory`` moved by a known similarity transformation (about the coordinate origin)."""
    rotation = Rotations.from_euler("z", np.deg2rad(yaw_deg))
    moved = trajectory.copy()
    moved.positions = Positions(xyz=scale * rotation.apply(trajectory.xyz) + np.asarray(translation), epsg=0)
    moved.rotations = rotation * trajectory.rotations
    moved.name = "moved"
    return moved


def with_varying_speed(trajectory: Trajectory) -> Trajectory:
    """The same path, travelled with a speed that varies (the generated trajectory has a constant speed)."""
    result = trajectory.copy()
    t = result.timestamps - result.timestamps[0]
    duration = t[-1]
    warped = (
        t + 0.12 * duration * np.sin(2 * np.pi * t / duration) + 0.05 * duration * np.sin(7 * np.pi * t / duration)
    )
    result.timestamps = result.timestamps[0] + np.sort(warped)
    return result


def rms(preview: AlignmentPreview, state: AlignmentState) -> float:
    """RMS distance of corresponding poses (the test trajectories have the same poses as the reference)."""
    return float(np.sqrt(np.mean(np.sum((preview.positions(state) - preview.reference_xyz) ** 2, axis=1))))


class TestAlignmentState(unittest.TestCase):
    def test_apply_equals_apply_alignment(self):
        """The fast preview and the alignment that is finally applied place the trajectory identically."""
        state = AlignmentState(
            rotation=Rotations.from_euler("xyz", np.array([0.1, -0.2, 1.3])).as_matrix(),
            translation=[10.0, -20.0, 3.0],
            scale=1.1,
            time_shift=2.5,
            leverarm=[0.5, -0.3, 0.2],
        )
        trajectory = generated_trajectory.copy()
        preview = AlignmentPreview(trajectory, trajectory, max_points=len(trajectory))
        result = alignment.manual_alignment(trajectory, trajectory, state.to_parameters())
        aligned = alignment.apply_alignment(trajectory, result, inplace=False)
        np.testing.assert_allclose(preview.positions(state), aligned.xyz, atol=1e-9)
        np.testing.assert_allclose(preview.timestamps(state), aligned.timestamps, atol=1e-9)

    def test_parameter_round_trip(self):
        state = AlignmentState(
            rotation=Rotations.from_euler("xyz", np.array([0.3, 0.2, -2.0])).as_matrix(),
            translation=[1.0, 2.0, 3.0],
            scale=0.9,
            time_shift=-1.0,
            leverarm=[0.1, 0.2, 0.3],
        )
        restored = AlignmentState.from_parameters(state.to_parameters())
        np.testing.assert_allclose(restored.rotation, state.rotation, atol=1e-12)
        np.testing.assert_allclose(restored.translation, state.translation)
        np.testing.assert_allclose(restored.leverarm, state.leverarm)
        self.assertAlmostEqual(restored.scale, 0.9)
        self.assertAlmostEqual(restored.time_shift, -1.0)
        self.assertTrue(all(parameter.enabled for parameter in state.to_parameters()))

    def test_rotation_and_scale_keep_the_pivot_in_place(self):
        """Also far from the origin (georeferenced coordinates)."""
        pivot = UTM_OFFSET + np.array([5.0, 3.0, 1.0])
        state = AlignmentState().moved([1.0, 2.0, 3.0])
        shown_pivot = state.apply(pivot)
        changed = state.rotated_about(shown_pivot, interactive.axis_rotation(Z_AXIS, 0.7)).scaled_about(
            shown_pivot, 1.5
        )
        np.testing.assert_allclose(changed.apply(pivot), shown_pivot, atol=1e-6)
        self.assertAlmostEqual(changed.scale, 1.5)
        self.assertAlmostEqual(changed.euler[2], 0.7)

        absolute = changed.with_rotation(np.eye(3), shown_pivot).with_scale(1.0, shown_pivot)
        np.testing.assert_allclose(absolute.apply(pivot), shown_pivot, atol=1e-6)
        np.testing.assert_allclose(absolute.rotation, np.eye(3), atol=1e-12)

    def test_states_are_immutable_values(self):
        state = AlignmentState()
        self.assertEqual(state, AlignmentState())
        self.assertNotEqual(state, state.moved([1.0, 0.0, 0.0]))
        with self.assertRaises(ValueError):
            state.translation[0] = 1.0


class TestProposals(unittest.TestCase):
    def test_fit_on_time_matches_recovers_a_known_transformation(self):
        reference = generated_trajectory.copy()
        preview = AlignmentPreview(transformed(reference, 30.0, [1000.0, -500.0, 20.0]), reference)
        self.assertGreater(rms(preview, AlignmentState()), 100)
        state = interactive.fit_time_matches(preview, AlignmentState())
        self.assertLess(rms(preview, state), 1e-6)

    def test_fit_with_scale(self):
        reference = generated_trajectory.copy()
        preview = AlignmentPreview(transformed(reference, -45.0, [10.0, 5.0, 0.0], scale=0.7), reference)
        rigid = interactive.fit_time_matches(preview, AlignmentState())
        scaled = interactive.fit_time_matches(preview, AlignmentState(), with_scale=True)
        self.assertGreater(rms(preview, rigid), 1.0)
        self.assertLess(rms(preview, scaled), 1e-6)
        self.assertAlmostEqual(scaled.scale, 1 / 0.7, places=6)

    def test_first_pose_and_centers(self):
        reference = generated_trajectory.copy()
        reference.positions = Positions(xyz=reference.xyz + UTM_OFFSET, epsg=0)
        preview = AlignmentPreview(transformed(generated_trajectory, 60.0, [0.0, 0.0, 0.0]), reference)
        centered = interactive.match_centers(preview, AlignmentState())
        np.testing.assert_allclose(
            np.mean(preview.positions(centered), axis=0), np.mean(preview.reference_xyz, axis=0), atol=1e-6
        )
        self.assertLess(rms(preview, interactive.match_first_pose(preview, AlignmentState())), 5)

    def test_point_pairs(self):
        rng = np.random.default_rng(1)
        source = rng.normal(size=(5, 3)) * 20
        rotation = Rotations.from_euler("xyz", np.array([0.2, -0.1, 2.0])).as_matrix()
        target = np.array([7.0, -3.0, 2.0]) + 1.3 * source @ rotation.T

        # one pair: translation only
        r, t, s = interactive.fit_point_pairs(source[:1], target[:1], with_scale=True)
        np.testing.assert_allclose(r, np.eye(3))
        np.testing.assert_allclose(t + source[0], target[0])
        self.assertEqual(s, 1.0)

        # three or more pairs: full similarity transformation
        r, t, s = interactive.fit_point_pairs(source, target, with_scale=True)
        np.testing.assert_allclose(t + s * source @ r.T, target, atol=1e-9)
        self.assertAlmostEqual(s, 1.3)
        r, _, s = interactive.fit_point_pairs(source, target)
        np.testing.assert_allclose(r, rotation, atol=1e-9)
        self.assertEqual(s, 1.0)

    def test_two_point_pairs_give_translation_and_heading(self):
        source = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]])
        yaw = np.deg2rad(40.0)
        rotation = interactive.axis_rotation(Z_AXIS, yaw)
        target = np.array([100.0, 50.0, 4.0]) + 2.0 * source @ rotation.T
        r, t, s = interactive.fit_point_pairs(source, target, with_scale=True)
        np.testing.assert_allclose(r, rotation, atol=1e-12)
        self.assertAlmostEqual(s, 2.0)
        np.testing.assert_allclose(t + s * source @ r.T, target, atol=1e-9)
        # without scale, the centers of the pairs coincide
        r, t, s = interactive.fit_point_pairs(source, target)
        np.testing.assert_allclose(np.mean(t + source @ r.T, axis=0), np.mean(target, axis=0), atol=1e-9)
        # the same for more points on a line (the rotation about the line is not determined)
        line = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        r, _, _ = interactive.fit_point_pairs(line, line @ rotation.T)
        np.testing.assert_allclose(r, rotation, atol=1e-12)

    def test_time_shift_from_speed_profiles(self):
        reference = with_varying_speed(generated_trajectory)
        moved = transformed(reference, 75.0, [300.0, -200.0, 10.0], scale=1.4)
        moved.timestamps = moved.timestamps + 1234.5
        preview = AlignmentPreview(moved, reference)
        estimate = interactive.estimate_time_shift(preview)
        step = float(np.median(np.diff(preview.t)))
        self.assertAlmostEqual(estimate.time_shift, 1234.5, delta=step)
        self.assertGreater(estimate.correlation, 0.9)

    def test_time_shift_needs_a_varying_speed(self):
        t = np.arange(100.0)
        xyz = np.column_stack((t, np.zeros(100), np.zeros(100)))
        constant = Trajectory(positions=Positions(xyz=xyz, epsg=0), timestamps=t)
        self.assertIsNone(interactive.estimate_time_shift(AlignmentPreview(constant, constant)))

    def test_best_start(self):
        reference = generated_trajectory.copy()
        preview = AlignmentPreview(transformed(reference, -120.0, [2500.0, 800.0, 30.0]), reference)
        name, state = interactive.best_start(preview, AlignmentState())
        self.assertIn(name, {"first pose", "fit on poses matched in time"})
        self.assertLess(rms(preview, state), 1e-6)

    def test_best_start_without_usable_timestamps(self):
        """A large time offset prevents matching in time: the first pose or the speed profiles are used."""
        reference = with_varying_speed(generated_trajectory)
        moved = transformed(reference, 45.0, [300.0, -200.0, 0.0])
        moved.timestamps = moved.timestamps + 1000.0
        preview = AlignmentPreview(moved, reference)
        self.assertIsNone(preview.deviations(AlignmentState()))
        name, state = interactive.best_start(preview, AlignmentState())
        self.assertIn("time", name)
        self.assertLess(preview.nearest_rms(state), 1.0)
        # the time offset is found as well, so that the least squares alignment can match the poses in time
        self.assertAlmostEqual(state.time_shift, 1000.0, delta=float(np.median(np.diff(preview.t))))

    def test_least_squares_with_a_large_prior_time_shift(self):
        """The prior time shift is applied to the timestamps; only the remaining shift is estimated."""
        from trajectopy.core.settings import AlignmentEstimationSettings, AlignmentSettings

        reference = with_varying_speed(generated_trajectory)
        moved = transformed(reference, 20.0, [30.0, -20.0, 1.0])
        moved.timestamps = moved.timestamps + 1000.0
        # without applying it to the timestamps first, the linearized time shift model diverges for this prior
        prior = AlignmentState(time_shift=1000.0).to_parameters()
        for estimate_time_shift in (True, False):
            settings = AlignmentSettings(
                estimation_settings=AlignmentEstimationSettings(time_shift=estimate_time_shift)
            )
            result = alignment.estimate_alignment(moved, reference, settings, initial_parameters=prior)
            self.assertTrue(result.converged)
            self.assertAlmostEqual(result.position_parameters.time_shift.value, 1000.0, places=3)
            aligned = alignment.apply_alignment(moved, result, inplace=False)
            np.testing.assert_allclose(aligned.timestamps, reference.timestamps, atol=1e-3)
            np.testing.assert_allclose(aligned.xyz, reference.xyz, atol=1e-3)
        self.assertEqual(prior.time_shift.value, 1000.0)  # the prior is not modified

    def test_time_shift_of_identical_profiles_is_zero(self):
        """Short overlaps of above-average speeds must not win over the full overlap."""
        trajectory = generated_trajectory.copy()
        estimate = interactive.estimate_time_shift(AlignmentPreview(trajectory, trajectory))
        self.assertAlmostEqual(estimate.time_shift, 0.0, places=6)
        self.assertAlmostEqual(estimate.correlation, 1.0, places=6)

    def test_deviation_follows_the_time_shift(self):
        reference = with_varying_speed(generated_trajectory)
        moved = reference.copy()
        moved.timestamps = moved.timestamps + 500.0
        preview = AlignmentPreview(moved, reference)
        self.assertIsNone(preview.deviations(AlignmentState()))
        self.assertLess(preview.nearest_rms(AlignmentState()), 1e-9)  # the paths coincide nevertheless
        _, deviations = preview.deviations(AlignmentState(time_shift=500.0))
        self.assertLess(np.max(deviations), 1e-6)

    def test_preview_is_sorted_by_time_and_thinned_out(self):
        trajectory = generated_trajectory.copy()
        preview = AlignmentPreview(trajectory, trajectory, max_points=100)
        self.assertLessEqual(len(preview.t), 100)
        self.assertTrue(np.all(np.diff(preview.t) >= 0))


if __name__ == "__main__":
    unittest.main()
