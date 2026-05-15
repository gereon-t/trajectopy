import unittest

from trajectopy.core.settings import (
    AlignmentEstimationSettings,
    AlignmentSettings,
    AlignmentStochastics,
    MatchingSettings,
)


class TestSettingsValidation(unittest.TestCase):
    def test_alignment_estimation_from_components_defaults(self):
        s = AlignmentEstimationSettings.from_components(
            similarity=False, time_shift=False, leverarm=False, sensor_rotation=False
        )
        self.assertIsInstance(s, AlignmentEstimationSettings)

    def test_alignment_settings_construct(self):
        settings = AlignmentSettings(
            estimation_settings=AlignmentEstimationSettings.from_components(
                similarity=True, time_shift=True, leverarm=True, sensor_rotation=False
            ),
            stochastics=AlignmentStochastics(),
        )
        self.assertIsInstance(settings, AlignmentSettings)

    def test_matching_settings_defaults(self):
        ms = MatchingSettings()
        self.assertIsInstance(ms, MatchingSettings)

    def test_alignment_settings_from_dict_missing_keys_raises(self):
        with self.assertRaises(ValueError):
            AlignmentSettings.from_dict({"estimation_settings": {}, "metric_threshold": 1e-4})


if __name__ == "__main__":
    unittest.main()
