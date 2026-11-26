"""Every processing function accepts its own settings type or the complete ProcessingSettings."""

import unittest

import trajectopy as tpy
from test.testdata import generated_trajectory, noisy_trajectory
from trajectopy.core.settings import resolve_settings


class TestResolveSettings(unittest.TestCase):
    def test_none_gives_defaults(self):
        self.assertEqual(resolve_settings(None, tpy.MatchingSettings), tpy.MatchingSettings())

    def test_specific_settings_are_used(self):
        matching = tpy.MatchingSettings(max_gap_size=1.0)
        self.assertIs(resolve_settings(matching, tpy.MatchingSettings), matching)

    def test_part_of_processing_settings_is_used(self):
        processing = tpy.ProcessingSettings()
        self.assertIs(resolve_settings(processing, tpy.SortingSettings), processing.sorting)

    def test_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            resolve_settings(tpy.SortingSettings(), tpy.MatchingSettings)


class TestSettingsArguments(unittest.TestCase):
    def test_processing_settings_everywhere(self):
        settings = tpy.ProcessingSettings()
        test, reference = noisy_trajectory.copy(), generated_trajectory.copy()
        matched, _ = tpy.match_trajectories(test, reference, settings)
        self.assertGreater(len(matched), 0)
        self.assertIsInstance(tpy.estimate_alignment(test, reference, settings), tpy.AlignmentResult)
        self.assertEqual(len(tpy.align(test, reference, settings)), len(test))
        self.assertIsInstance(tpy.rpe(test, reference, settings), tpy.RPEResult)

    def test_ate_result_holds_alignment(self):
        result = tpy.ate(noisy_trajectory.copy(), generated_trajectory.copy())
        self.assertIsInstance(result.alignment, tpy.AlignmentResult)

    def test_arguments_of_version_6_are_rejected(self):
        test, reference = noisy_trajectory.copy(), generated_trajectory.copy()
        with self.assertRaises(TypeError):
            tpy.match_trajectories(test, reference, matching_settings=tpy.MatchingSettings())  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            tpy.ate(test, reference, return_alignment=True)  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
