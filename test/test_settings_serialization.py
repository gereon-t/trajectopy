import json
import unittest

from trajectopy.core.settings import (
    AlignmentEstimationSettings,
    AlignmentSettings,
    ApproximationSettings,
    MatchingMethod,
    MatchingSettings,
    PlotSettings,
    ProcessingSettings,
    RelativeComparisonSettings,
    SortingSettings,
)

ALL_SETTINGS = [
    AlignmentSettings,
    AlignmentEstimationSettings,
    ApproximationSettings,
    MatchingSettings,
    PlotSettings,
    ProcessingSettings,
    RelativeComparisonSettings,
    SortingSettings,
]


class TestSettingsSerialization(unittest.TestCase):
    def test_json_roundtrip(self):
        for cls in ALL_SETTINGS:
            with self.subTest(settings=cls.__name__):
                settings = cls()
                self.assertEqual(cls.from_dict(json.loads(settings.to_json())), settings)

    def test_missing_keys_keep_defaults(self):
        settings = MatchingSettings.from_dict({"max_time_diff": 0.5})
        self.assertEqual(settings.max_time_diff, 0.5)
        self.assertEqual(settings.method, MatchingSettings().method)

    def test_nested_missing_keys_keep_defaults(self):
        settings = ProcessingSettings.from_dict({"matching": {"method": MatchingMethod.NEAREST_SPATIAL.value}})
        self.assertEqual(settings.matching.method, MatchingMethod.NEAREST_SPATIAL)
        self.assertEqual(settings.alignment, AlignmentSettings())

    def test_unknown_keys_are_ignored(self):
        with self.assertLogs("trajectopy.core.settings", level="WARNING"):
            settings = SortingSettings.from_dict({"not_a_setting": 1})
        self.assertEqual(settings, SortingSettings())

    def test_int_accepted_for_float(self):
        settings = MatchingSettings.from_dict({"max_time_diff": 1})
        self.assertIsInstance(settings.max_time_diff, float)

    def test_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            MatchingSettings.from_dict({"max_time_diff": "fast"})
        with self.assertRaises(TypeError):
            PlotSettings.from_dict({"font_size": True})

    def test_update_is_atomic(self):
        settings = ProcessingSettings()
        with self.assertRaises(TypeError):
            settings.update_from_dict({"matching": {"max_time_diff": 2.0}, "alignment": {"preprocessing": "bad"}})
        self.assertEqual(settings, ProcessingSettings())

    def test_update_keeps_nested_settings_objects(self):
        settings = ProcessingSettings()
        held = settings.alignment.estimation_settings  # e.g. referenced by an open window
        settings.update_from_dict({"alignment": {"estimation_settings": {"leverarm_x": True}}})
        self.assertIs(settings.alignment.estimation_settings, held)
        self.assertTrue(held.leverarm_x)


if __name__ == "__main__":
    unittest.main()
