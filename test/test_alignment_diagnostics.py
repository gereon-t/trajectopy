"""Each alignment parameter leaves its own signature in the deviations; the diagnosis finds and corrects it."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from trajectopy.core.positions import Positions
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing._lib.alignment import diagnostics, interactive
from trajectopy.processing._lib.alignment.interactive import AlignmentPreview, AlignmentState


def drive(duration: float = 400.0, rate: float = 10.0, noise: float = 0.0, seed: int = 0) -> Trajectory:
    """A vehicle drive: accelerates, turns and stops several times (at different headings), at 10 Hz."""
    t = np.arange(0.0, duration, 1.0 / rate)
    speed = 8.0 * np.clip(1.5 * np.sin(2 * np.pi * t / 50.0), 0.0, 1.0)  # stands still about half of the time
    heading = np.cumsum(0.25 * np.sin(2 * np.pi * t / 37.0) * (speed > 0)) / rate
    xyz = np.zeros((len(t), 3))
    xyz[:, 0] = np.cumsum(speed * np.cos(heading)) / rate
    xyz[:, 1] = np.cumsum(speed * np.sin(heading)) / rate
    xyz[:, 2] = 3.0 * np.sin(xyz[:, 0] / 150.0)
    xyz += np.array([364800.0, 5621500.0, 60.0])
    if noise:
        xyz += np.random.default_rng(seed).normal(scale=noise, size=xyz.shape)
    rotations = Rotation.from_euler("xyz", np.column_stack((np.zeros_like(t), np.zeros_like(t), heading)))
    return Trajectory(positions=Positions(xyz=xyz, epsg=0), timestamps=t, rotations=rotations, name="drive")


def with_positions(trajectory: Trajectory, xyz: np.ndarray) -> Trajectory:
    result = trajectory.copy()
    result.positions = Positions(xyz=xyz, epsg=0)
    return result


class TestSignatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = drive()

    def _residuals(self, moved: Trajectory, state: AlignmentState | None = None):
        preview = AlignmentPreview(moved, self.reference)
        state = state or AlignmentState()
        return preview, diagnostics.residuals(preview, state, state.apply(preview.center))

    def test_standstills_are_detected_despite_noise(self):
        noisy = drive(noise=0.02)
        preview = AlignmentPreview(noisy, noisy)
        expected = drive().copy()
        true_speed = np.linalg.norm(np.gradient(expected.xyz, expected.timestamps, axis=0), axis=1)
        still = preview.reference_standstill
        self.assertGreater(np.mean(still[true_speed < 1e-6]), 0.8)
        self.assertLess(np.mean(still[true_speed > 1.0]), 0.01)

    def test_time_shift_shows_as_along_track_deviation_proportional_to_speed(self):
        moved = self.reference.copy()
        moved.timestamps = moved.timestamps + 0.2
        preview, residuals = self._residuals(moved)
        moving = ~residuals.standstill
        slope, intercept = diagnostics.line_fit(residuals.speed[moving], residuals.along[moving])
        self.assertAlmostEqual(abs(slope), 0.2, delta=0.02)
        self.assertAlmostEqual(intercept, 0.0, delta=0.1)
        self.assertLess(residuals.rms_of(residuals.cross), 0.2 * residuals.rms_of(residuals.along))
        self.assertLess(np.max(np.abs(residuals.deviation[residuals.standstill])), 0.05)

        best = diagnostics.diagnose(preview, AlignmentState(), preview.center).corrections[0]
        self.assertEqual(best.key, "time_shift")
        self.assertEqual(best.view, diagnostics.VIEW_TIME)
        fixed = diagnostics.fit(preview, AlignmentState(), preview.center, ["time_shift"]).state
        self.assertAlmostEqual(fixed.time_shift, 0.2, places=3)

    def test_leverarm_is_constant_in_the_body_frame(self):
        lever = np.array([1.2, -0.4, 0.8])
        moved = with_positions(self.reference, self.reference.xyz - self.reference.rotations.apply(lever))
        preview, residuals = self._residuals(moved)
        np.testing.assert_allclose(residuals.body_deviation, np.tile(-lever, (residuals.count, 1)), atol=1e-6)

        diagnosis = diagnostics.diagnose(preview, AlignmentState(), preview.center)
        corrections = {correction.key: correction.value for correction in diagnosis.corrections}
        np.testing.assert_allclose([corrections["lever_x"], corrections["lever_y"]], lever[:2], atol=1e-4)
        # level platform: the vertical leverarm is attributed to the (earlier) vertical shift
        self.assertIn(("lever_z", "shift_z"), diagnosis.inseparable)
        self.assertAlmostEqual(corrections["shift_z"], lever[2], places=4)
        self.assertLess(diagnosis.rms_after_all, 1e-4)
        keys = ["lever_x", "lever_y", "lever_z", "shift_x", "shift_y", "shift_z"]
        result = diagnostics.fit(preview, AlignmentState(), preview.center, keys, epoch_mask="standstill")
        np.testing.assert_allclose(result.state.leverarm, lever, atol=1e-4)
        np.testing.assert_allclose(result.state.translation, 0.0, atol=1e-4)
        # the platform is level: the vertical leverarm and the vertical shift cannot be told apart
        self.assertEqual(result.inseparable, (("shift_z", "lever_z"),))

    def test_shift_varies_with_the_heading_in_the_body_frame(self):
        """A shift in the world frame and a leverarm are told apart at standstills with different headings."""
        shift = np.array([0.5, -0.3, 0.2])
        moved = with_positions(self.reference, self.reference.xyz + shift)
        preview, residuals = self._residuals(moved)
        still = residuals.standstill
        self.assertGreater(np.ptp(np.rad2deg(residuals.body_heading[still])), 90)
        self.assertGreater(np.ptp(residuals.body_deviation[still, 0]), 0.3)  # not constant
        keys = ["shift_x", "shift_y", "shift_z", "lever_x", "lever_y", "lever_z"]
        result = diagnostics.fit(preview, AlignmentState(), preview.center, keys, epoch_mask="standstill")
        np.testing.assert_allclose(result.state.translation, -shift, atol=1e-4)
        np.testing.assert_allclose(result.state.leverarm, 0.0, atol=1e-4)
        self.assertEqual(result.inseparable, (("lever_z", "shift_z"),))

    def test_heading_error_shows_as_tangential_deviation_proportional_to_the_distance(self):
        preview = AlignmentPreview(self.reference, self.reference)
        pivot = preview.center
        state = AlignmentState().rotated_about(pivot, interactive.axis_rotation(np.array([0.0, 0.0, 1.0]), 0.01))
        residuals = diagnostics.residuals(preview, state, pivot)
        slope, _ = diagnostics.line_fit(residuals.radius, residuals.tangential)
        self.assertAlmostEqual(slope, 0.01, delta=1e-4)
        self.assertLess(residuals.rms_of(residuals.radial), 0.05 * residuals.rms_of(residuals.tangential))
        best = diagnostics.diagnose(preview, state, pivot).corrections[0]
        self.assertEqual(best.key, "rot_z")
        self.assertAlmostEqual(best.value, -0.01, delta=1e-4)
        self.assertLess(best.rms_after, 0.01 * best.rms_before)

    def test_scale_error_shows_as_radial_deviation_proportional_to_the_distance(self):
        preview = AlignmentPreview(self.reference, self.reference)
        pivot = preview.center
        state = AlignmentState().scaled_about(pivot, 1.001)
        residuals = diagnostics.residuals(preview, state, pivot)
        slope, _ = diagnostics.line_fit(residuals.radius, residuals.radial)
        self.assertAlmostEqual(slope, 0.001, delta=1e-5)
        best = diagnostics.diagnose(preview, state, pivot).corrections[0]
        self.assertEqual(best.key, "scale")
        self.assertAlmostEqual(best.value, 1 / 1.001 - 1, delta=1e-6)

    def test_fit_of_all_parameters(self):
        lever = np.array([0.3, 0.1, -0.2])
        moved = with_positions(self.reference, self.reference.xyz - self.reference.rotations.apply(lever))
        moved.timestamps = moved.timestamps + 0.15
        preview = AlignmentPreview(moved, self.reference)
        pivot_source = preview.center
        start = AlignmentState().rotated_about(
            pivot_source, interactive.axis_rotation(np.array([0.2, 0.1, 1.0]), 0.004)
        )
        start = start.moved([0.4, -0.2, 0.1]).scaled_about(start.apply(pivot_source), 1.0005)
        fixed = diagnostics.fit(preview, start, pivot_source, diagnostics.available_parameters(preview)).state
        residuals = diagnostics.residuals(preview, fixed, fixed.apply(pivot_source))
        self.assertLess(residuals.rms, 1e-3)
        np.testing.assert_allclose(fixed.leverarm[:2], lever[:2], atol=1e-3)  # z: inseparable from the shift
        self.assertAlmostEqual(fixed.time_shift, 0.15, places=3)

    def test_only_the_given_parameters_are_considered(self):
        """How much a parameter explains depends on the other parameters considered; those not considered are
        reported with how much they would reduce the deviation further."""
        lever = np.array([1.2, 0.0, 0.0])
        moved = with_positions(self.reference, self.reference.xyz - self.reference.rotations.apply(lever))
        moved.timestamps = moved.timestamps + 0.15
        preview = AlignmentPreview(moved, self.reference)
        keys = ("shift_x", "shift_y", "shift_z", "rot_x", "rot_y", "rot_z", "time_shift")
        diagnosis = diagnostics.diagnose(preview, AlignmentState(), preview.center, keys)
        self.assertTrue({correction.key for correction in diagnosis.corrections} <= set(keys))
        # without the leverarm, the time shift has to account for its along-track part, too
        self.assertGreater(abs(diagnosis.correction("time_shift").value - 0.15), 0.01)
        self.assertEqual(diagnosis.potential[0].key, "lever_x")
        self.assertLess(diagnosis.potential[0].rms_after, 0.1 * diagnosis.rms_after_all)
        self.assertNotIn("time_shift", {potential.key for potential in diagnosis.potential})

    def test_noise_is_not_significant(self):
        """Without an alignment error, no parameter explains clearly more than noise, also for correlated noise
        (a random walk), which fits parameters by chance more easily."""
        rng = np.random.default_rng(5)
        white = rng.normal(scale=0.02, size=self.reference.xyz.shape)
        walk = np.cumsum(rng.normal(scale=0.002, size=self.reference.xyz.shape), axis=0)
        for name, noise in (("white", white), ("random walk", walk - walk.mean(axis=0))):
            preview = AlignmentPreview(with_positions(self.reference, self.reference.xyz + noise), self.reference)
            diagnosis = diagnostics.diagnose(preview, AlignmentState(), preview.center)
            significant = [
                c.key for c in diagnosis.corrections if c.significant and c.explained > 0.1 * diagnosis.rms_after_all
            ]
            self.assertEqual(significant, [], name)

        # a real error of the size of the noise is found
        moved = with_positions(self.reference, self.reference.xyz + white)
        moved.timestamps = moved.timestamps + 0.005  # 5 ms: up to 4 cm along-track at 8 m/s
        preview = AlignmentPreview(moved, self.reference)
        diagnosis = diagnostics.diagnose(preview, AlignmentState(), preview.center)
        self.assertTrue(diagnosis.correction("time_shift").significant)

    def test_fit_converges_completely(self):
        """The fit does not stop while a step still moves the trajectory noticeably (e.g. 1 mm of leverarm)."""
        lever = np.array([1.2, -0.4, 0.0])
        moved = with_positions(self.reference, self.reference.xyz - self.reference.rotations.apply(lever))
        preview = AlignmentPreview(moved, self.reference)
        keys = diagnostics.available_parameters(preview)
        fixed = diagnostics.fit(preview, AlignmentState(), preview.center, keys).state
        np.testing.assert_allclose(fixed.leverarm[:2], lever[:2], atol=1e-6)

    def test_leverarm_is_not_offered_without_orientations(self):
        without = Trajectory(positions=self.reference.positions.copy(), timestamps=self.reference.timestamps.copy())
        preview = AlignmentPreview(without, self.reference)
        self.assertNotIn("lever_x", diagnostics.available_parameters(preview))
        residuals = diagnostics.residuals(preview, AlignmentState(), preview.center)
        self.assertIsNone(residuals.body_deviation)

    def test_no_diagnosis_without_temporal_overlap(self):
        moved = self.reference.copy()
        moved.timestamps = moved.timestamps + 10_000.0
        preview = AlignmentPreview(moved, self.reference)
        self.assertIsNone(diagnostics.residuals(preview, AlignmentState(), preview.center))
        self.assertIsNone(diagnostics.diagnose(preview, AlignmentState(), preview.center))

    def test_leverarm_along_the_direction_of_travel_and_time_shift_are_told_apart(self):
        """For a car, body x is the direction of travel: both errors deviate along-track. Only jointly (a leverarm
        is constant, a time shift proportional to the speed) are they determined correctly."""
        lever = np.array([1.2, 0.0, 0.0])
        moved = with_positions(self.reference, self.reference.xyz - self.reference.rotations.apply(lever))
        moved.timestamps = moved.timestamps + 0.15
        preview = AlignmentPreview(moved, self.reference)
        diagnosis = diagnostics.diagnose(preview, AlignmentState(), preview.center)
        corrections = {correction.key: correction.value for correction in diagnosis.corrections}
        self.assertAlmostEqual(corrections["lever_x"], 1.2, delta=0.01)
        self.assertAlmostEqual(corrections["time_shift"], 0.15, delta=0.002)
        self.assertEqual({c.key for c in diagnosis.corrections[:2]}, {"lever_x", "time_shift"})

    def test_partial_residuals_show_a_single_signature(self):
        """With a leverarm and a time shift present, the rotation signature is only clean without their effects."""
        lever = np.array([1.2, -0.4, 0.0])
        moved = with_positions(self.reference, self.reference.xyz - self.reference.rotations.apply(lever))
        moved.timestamps = moved.timestamps + 0.15
        preview = AlignmentPreview(moved, self.reference)
        pivot = preview.center
        state = AlignmentState().rotated_about(pivot, interactive.axis_rotation(np.array([0.0, 0.0, 1.0]), 0.005))
        diagnosis = diagnostics.diagnose(preview, state, pivot)

        actual = diagnostics.residuals(preview, state, pivot)
        partial = diagnostics.partial_residuals(preview, state, pivot, diagnosis, ("rot_x", "rot_y", "rot_z", "scale"))
        raw_slope, _ = diagnostics.line_fit(actual.radius, actual.tangential)
        slope, _ = diagnostics.line_fit(partial.radius, partial.tangential)
        self.assertGreater(abs(raw_slope - 0.005), 1e-3)  # distorted by the other effects
        self.assertAlmostEqual(slope, 0.005, delta=1e-4)
        radial_slope, _ = diagnostics.line_fit(partial.radius, partial.radial)
        self.assertAlmostEqual(radial_slope, 0.0, delta=5e-4)  # linearized: the time shift is not quite linear

        timing = diagnostics.partial_residuals(preview, state, pivot, diagnosis, ("time_shift",))
        moving = ~timing.standstill
        slope, intercept = diagnostics.line_fit(timing.speed[moving], timing.along[moving])
        self.assertAlmostEqual(slope, -0.15, delta=0.002)
        self.assertAlmostEqual(intercept, 0.0, delta=0.01)  # the leverarm is removed


if __name__ == "__main__":
    unittest.main()
