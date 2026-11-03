import unittest

import numpy as np

from test.testdata import open_loop_trajectory
from test.util import transform_randomly
from trajectopy.core.parameters import AlignmentParameters
from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import (
    AlignmentEstimationSettings,
    AlignmentPreprocessing,
    AlignmentSettings,
    AlignmentStochastics,
    MatchingSettings,
)
from trajectopy.processing.alignment import (
    apply_alignment,
    estimate_alignment,
    manual_alignment,
)
from trajectopy.processing.lib.alignment.data import AlignmentData
from trajectopy.processing.lib.alignment.direct import align_rotations


class TestAlignment(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def _alignment_test(
        self, similarity_enabled: bool, time_shift_enabled: bool, lever_enabled: bool, lazy: bool = False
    ) -> None:
        transformed, groundtruth = transform_randomly(
            open_loop_trajectory,
            similarity_enabled=similarity_enabled,
            time_shift_enabled=time_shift_enabled,
            lever_enabled=lever_enabled,
        )

        alignment_result = estimate_alignment(
            trajectory=open_loop_trajectory.copy(),
            other=transformed,
            alignment_settings=AlignmentSettings(
                estimation_settings=AlignmentEstimationSettings.from_components(
                    similarity=similarity_enabled,
                    time_shift=time_shift_enabled,
                    leverarm=lever_enabled,
                    sensor_rotation=False,
                ),
                stochastics=AlignmentStochastics(),
            ),
            matching_settings=MatchingSettings(),
        )

        self._verify_alignment(target=groundtruth, estimation=alignment_result.position_parameters, lazy=lazy)

    def test_alignment_combinations(self):
        """Test various combinations of alignment parameters (similarity, time shift, lever arm)."""
        cases = [
            (True, False, False),
            (False, False, True),
            (False, True, False),
            (True, False, True),
            (True, True, False),
            (False, True, True),
            (True, True, True),
        ]
        for similarity, time_shift, lever in cases:
            with self.subTest(similarity=similarity, time_shift=time_shift, lever=lever):
                self._alignment_test(similarity_enabled=similarity, time_shift_enabled=time_shift, lever_enabled=lever)

    def test_convergence_thresholds_belong_to_estimated_parameters(self):
        """The time shift update is compared with the time threshold, also if parameters before it are fixed."""
        transformed, _ = transform_randomly(
            open_loop_trajectory, similarity_enabled=False, time_shift_enabled=True, lever_enabled=False
        )
        settings = AlignmentSettings(
            estimation_settings=AlignmentEstimationSettings.from_components(time_shift=True),
            time_threshold=1e3,  # any update is small enough
        )
        prior = AlignmentParameters()  # starts at zero, so that the first update is large
        with self.assertLogs("trajectopy.processing.lib.alignment.estimation", level="INFO") as logs:
            estimate_alignment(
                trajectory=open_loop_trajectory.copy(),
                other=transformed,
                alignment_settings=settings,
                matching_settings=MatchingSettings(),
                initial_parameters=prior,
            )
        self.assertIn("Adjustment did converge after 1 iterations", "\n".join(logs.output))

    def test_alignment_with_prior(self):
        """A prior is used as start value; non-estimated parameters stay fixed at the prior."""
        transformed, groundtruth = transform_randomly(
            open_loop_trajectory, similarity_enabled=True, time_shift_enabled=False, lever_enabled=False
        )
        prior = AlignmentParameters()
        for name in ("sim_trans_x", "sim_trans_y", "sim_trans_z", "sim_rot_x", "sim_rot_y", "sim_rot_z"):
            getattr(prior, name).enabled = True
        prior.values = groundtruth.values + 1e-3

        settings = AlignmentSettings(
            estimation_settings=AlignmentEstimationSettings.from_components(similarity=True),
            stochastics=AlignmentStochastics(),
        )
        result = estimate_alignment(
            trajectory=open_loop_trajectory.copy(),
            other=transformed,
            alignment_settings=settings,
            matching_settings=MatchingSettings(),
            initial_parameters=prior,
        )
        self._verify_alignment(target=groundtruth, estimation=result.position_parameters)

        fixed_settings = AlignmentSettings(
            estimation_settings=AlignmentEstimationSettings.from_components(similarity=False),
        )
        fixed_result = estimate_alignment(
            trajectory=open_loop_trajectory.copy(),
            other=transformed,
            alignment_settings=fixed_settings,
            matching_settings=MatchingSettings(),
            initial_parameters=prior,
        )
        np.testing.assert_allclose(fixed_result.position_parameters.values, prior.values)

    def test_manual_alignment(self):
        """Manual parameters are applied exactly as given."""
        params = AlignmentParameters()
        params.sim_trans_x.value = 2.0
        params.time_shift.value = 0.5
        result = manual_alignment(open_loop_trajectory, open_loop_trajectory, params)
        aligned = apply_alignment(open_loop_trajectory, result, inplace=False)

        np.testing.assert_allclose(aligned.positions.xyz[:, 0], open_loop_trajectory.positions.xyz[:, 0] + 2.0)
        np.testing.assert_allclose(aligned.timestamps, open_loop_trajectory.timestamps - 0.5)

    def test_sensor_alignment_loop(self):
        """Test that sensor rotation alignment can be applied and reversed correctly."""
        trajectory = open_loop_trajectory.copy()
        trajectory_rotated = trajectory.copy()
        random_rot = Rotations.from_euler(
            seq="xyz",
            angles=[np.random.randint(-360, 360), np.random.randint(-360, 360), np.random.randint(-360, 360)],
            degrees=True,
        )
        trajectory_rotated.rotations = trajectory_rotated.rotations * random_rot

        sensor_alignment = align_rotations(trajectory_rotated.rotations, trajectory.rotations)

        trajectory_rotated_back = trajectory_rotated.copy()

        # apply as in apply_alignment()
        trajectory_rotated_back.rotations = trajectory_rotated_back.rotations * sensor_alignment.rotation_set
        alignment_between_back_and_original = align_rotations(trajectory_rotated_back.rotations, trajectory.rotations)

        np.testing.assert_allclose(
            alignment_between_back_and_original.rotation_set.rotangle,
            0,
            atol=1e-8,
            rtol=1e-8,
        )

    def test_rotation_alignment(self):
        """The sensor rotation is a constant rotation in the body frame, also with changing orientations."""
        rotations_1 = Rotations.from_euler(seq="xyz", angles=np.random.rand(100, 3) * 360, degrees=True)
        sensor_rotation = Rotations.from_euler(seq="xyz", angles=[2.0, -3.0, 5.0], degrees=True)
        rotations_2 = rotations_1 * sensor_rotation

        sensor_rot_params = align_rotations(rot_from=rotations_1, rot_to=rotations_2)

        np.testing.assert_allclose(np.rad2deg(sensor_rot_params.rotation), [2.0, -3.0, 5.0], atol=1e-8)
        rotations_1_to_2 = rotations_1 * sensor_rot_params.rotation_set
        np.testing.assert_allclose((rotations_2 - rotations_1_to_2).rotangle, 0, atol=1e-8)

    def test_sensor_rotation_with_similarity_transformation(self):
        """A mounting rotation is recovered after the similarity transformation and removed when applied."""
        transformed, _ = transform_randomly(
            open_loop_trajectory, similarity_enabled=True, time_shift_enabled=False, lever_enabled=False
        )
        sensor_rotation = Rotations.from_euler(seq="xyz", angles=[2.0, -3.0, 5.0], degrees=True)
        transformed.rotations = transformed.rotations * sensor_rotation

        settings = AlignmentSettings(
            estimation_settings=AlignmentEstimationSettings.from_components(similarity=True, sensor_rotation=True)
        )
        result = estimate_alignment(open_loop_trajectory.copy(), transformed, settings, MatchingSettings())
        np.testing.assert_allclose(np.rad2deg(result.rotation_parameters.rotation), [2.0, -3.0, 5.0], atol=1e-3)

        aligned = apply_alignment(open_loop_trajectory, result, inplace=False)
        np.testing.assert_allclose((transformed.rotations - aligned.rotations).rotangle, 0, atol=1e-5)

    def test_robust_estimation_against_outliers(self):
        """Outliers are down-weighted, so that they hardly affect the estimated parameters."""
        transformed, groundtruth = transform_randomly(
            open_loop_trajectory, similarity_enabled=True, time_shift_enabled=False, lever_enabled=False
        )
        rng = np.random.default_rng(1)
        transformed.positions.xyz += rng.normal(0, 0.02, transformed.positions.xyz.shape)
        outliers = rng.choice(len(transformed), len(transformed) // 50, replace=False)
        transformed.positions.xyz[outliers] += rng.normal(0, 3.0, (len(outliers), 3))

        errors = {}
        for robust in (False, True):
            settings = AlignmentSettings(
                estimation_settings=AlignmentEstimationSettings.from_components(similarity=True),
                stochastics=AlignmentStochastics(robust_estimation=robust),
                preprocessing=AlignmentPreprocessing(max_samples=0),
            )
            result = estimate_alignment(open_loop_trajectory.copy(), transformed, settings, MatchingSettings())
            self.assertTrue(result.converged)
            errors[robust] = np.max(
                np.abs(result.position_parameters.helmert.translation - groundtruth.helmert.translation)
            )
        self.assertLess(errors[True], 0.005)
        self.assertLess(errors[True], errors[False] / 5)

    def test_time_window_relative_to_first_common_timestamp(self):
        """time_start/time_end refer to the first common timestamp, not to the start of the aligned trajectory."""
        reference = open_loop_trajectory.copy()
        first_common = reference.timestamps[0] + 50.0
        reference.mask(reference.timestamps >= first_common)
        settings = AlignmentSettings(preprocessing=AlignmentPreprocessing(time_start=10.0, time_end=20.0))

        data = AlignmentData(open_loop_trajectory.copy(), reference, settings, MatchingSettings())

        self.assertGreaterEqual(data.tstamps.min(), first_common + 10.0)
        self.assertLessEqual(data.tstamps.max(), first_common + 20.0)

    def test_nothing_to_estimate_counts_as_converged(self):
        """Without least squares parameters (e.g. only the sensor rotation), the result is not flagged."""
        settings = AlignmentSettings(
            estimation_settings=AlignmentEstimationSettings.from_components(similarity=False, sensor_rotation=True)
        )
        result = estimate_alignment(
            open_loop_trajectory.copy(), open_loop_trajectory.copy(), settings, MatchingSettings()
        )
        self.assertTrue(result.converged)

    def _verify_alignment(
        self, target: AlignmentParameters, estimation: AlignmentParameters, lazy: bool = False
    ) -> None:
        tolerance = 1e-1 if lazy else 1e-3
        np.testing.assert_allclose(estimation.helmert.translation, target.helmert.translation, atol=tolerance)
        np.testing.assert_allclose(target.helmert.rotation_set.rotangle, estimation.helmert.rotation_set.rotangle)
        np.testing.assert_allclose(estimation.helmert.scale.value, target.helmert.scale.value, atol=tolerance)
        np.testing.assert_allclose(estimation.time_shift.value, target.time_shift.value, atol=tolerance)
        np.testing.assert_allclose(estimation.leverarm.values, target.leverarm.values, atol=tolerance)


if __name__ == "__main__":
    unittest.main()
