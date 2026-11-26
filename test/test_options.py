import unittest

import numpy as np

import trajectopy as tpy


class TestOptions(unittest.TestCase):
    def test_lookup_ignores_case_and_accepts_aliases(self):
        self.assertIs(tpy.MatchingMethod("Nearest_Temporal"), tpy.MatchingMethod.NEAREST_TEMPORAL)
        self.assertIs(tpy.PairDistanceUnit("m"), tpy.PairDistanceUnit.METER)
        self.assertIs(tpy.Sorting("arc_length"), tpy.Sorting.PATH_LENGTH)
        self.assertEqual(str(tpy.Sorting.TIME), "time")
        self.assertIs(tpy.Sorting.from_str("Time"), tpy.Sorting.TIME)  # kept for compatibility

    def test_unknown_option_raises(self):
        with self.assertRaises(ValueError):
            tpy.MatchingMethod("closest")

    def test_settings_accept_strings(self):
        settings = tpy.MatchingSettings(method="nearest_spatial")
        self.assertIs(settings.method, tpy.MatchingMethod.NEAREST_SPATIAL)
        relative = tpy.RelativeComparisonSettings(pair_distance_unit="s")
        self.assertIs(relative.pair_distance_unit, tpy.PairDistanceUnit.SECOND)

    def test_settings_round_trip_options_as_strings(self):
        settings = tpy.ProcessingSettings()
        settings.matching.method = tpy.MatchingMethod.NEAREST_TEMPORAL
        data = settings.to_dict()
        self.assertEqual(data["matching"]["method"], "nearest_temporal")
        self.assertEqual(tpy.ProcessingSettings.from_dict(data), settings)

    def test_invalid_option_in_dict_raises_type_error(self):
        with self.assertRaises(TypeError):
            tpy.MatchingSettings.from_dict({"method": "closest"})

    def test_interpolate_accepts_string_method(self):
        trajectory = tpy.Trajectory.from_arrays(xyz=np.c_[np.arange(10.0), np.zeros(10), np.zeros(10)])
        result = tpy.interpolate(trajectory, [0.5, 1.5], method="linear")
        np.testing.assert_allclose(result.xyz[:, 0], [0.5, 1.5])


if __name__ == "__main__":
    unittest.main()
