import unittest

import numpy as np

from trajectopy.core.settings import AlignmentEstimationSettings


class TestSettingsEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(19)

    def test_alignment_estimation_from_bool_list_correct_length(self):
        bool_list = [True] * 14
        settings = AlignmentEstimationSettings.from_bool_list(bool_list)
        self.assertIsInstance(settings, AlignmentEstimationSettings)
        self.assertTrue(settings.translation_x)
        self.assertTrue(settings.sensor_rotation)

    def test_alignment_estimation_from_bool_list_invalid_length_raises(self):
        bool_list = [True] * 10  # Wrong length, should be 14
        with self.assertRaises(ValueError):
            AlignmentEstimationSettings.from_bool_list(bool_list)

    def test_alignment_estimation_all_creates_all_enabled(self):
        settings = AlignmentEstimationSettings.all(sensor_rotation=True)
        self.assertTrue(settings.translation_x)
        self.assertTrue(settings.translation_y)
        self.assertTrue(settings.translation_z)
        self.assertTrue(settings.rotation_x)
        self.assertTrue(settings.rotation_y)
        self.assertTrue(settings.rotation_z)
        self.assertTrue(settings.scale)
        self.assertTrue(settings.time_shift)
        self.assertTrue(settings.leverarm_x)
        self.assertTrue(settings.leverarm_y)
        self.assertTrue(settings.leverarm_z)
        self.assertTrue(settings.sensor_rotation)

    def test_alignment_estimation_all_disabled(self):
        settings = AlignmentEstimationSettings.from_components(
            similarity=False, time_shift=False, leverarm=False, sensor_rotation=False
        )
        self.assertTrue(settings.all_disabled)
        self.assertTrue(settings.all_lq_disabled)

    def test_alignment_estimation_helmert_enabled_check(self):
        settings = AlignmentEstimationSettings.from_components(
            similarity=True, time_shift=False, leverarm=False, sensor_rotation=False
        )
        self.assertTrue(settings.helmert_enabled)
        self.assertFalse(settings.leverarm_enabled)
        self.assertFalse(settings.time_shift_enabled)

    def test_alignment_estimation_leverarm_enabled_check(self):
        settings = AlignmentEstimationSettings.from_components(
            similarity=False, time_shift=False, leverarm=True, sensor_rotation=False
        )
        self.assertFalse(settings.helmert_enabled)
        self.assertTrue(settings.leverarm_enabled)
        self.assertFalse(settings.time_shift_enabled)

    def test_alignment_estimation_time_shift_enabled_check(self):
        settings = AlignmentEstimationSettings.from_components(
            similarity=False, time_shift=True, leverarm=False, sensor_rotation=False
        )
        self.assertFalse(settings.helmert_enabled)
        self.assertFalse(settings.leverarm_enabled)
        self.assertTrue(settings.time_shift_enabled)

    def test_alignment_estimation_short_mode_str(self):
        settings = AlignmentEstimationSettings.from_components(
            similarity=True, time_shift=True, leverarm=True, sensor_rotation=True
        )
        mode_str = settings.short_mode_str
        self.assertIn("Helmert", mode_str)
        self.assertIn("Time-Shift", mode_str)
        self.assertIn("Leverarm", mode_str)
        self.assertIn("Sensor-Rotation", mode_str)

    def test_alignment_estimation_filters(self):
        settings = AlignmentEstimationSettings.from_components(
            similarity=True, time_shift=True, leverarm=True, sensor_rotation=False
        )
        helmert_filter = settings.helmert_filter
        self.assertEqual(len(helmert_filter), 7)
        self.assertTrue(all(helmert_filter))

        leverarm_filter = settings.leverarm_filter
        self.assertEqual(len(leverarm_filter), 3)
        self.assertTrue(all(leverarm_filter))

        time_shift_filter = settings.time_shift_filter
        self.assertEqual(len(time_shift_filter), 3)
        self.assertTrue(all(time_shift_filter))


if __name__ == "__main__":
    unittest.main()
