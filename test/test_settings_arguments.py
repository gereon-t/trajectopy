"""Every processing function accepts its own settings type or the complete ProcessingSettings."""

import unittest

import trajectopy as tpy
from test.testdata import generated_trajectory, noisy_trajectory
from trajectopy.core.settings import resolve_settings
from trajectopy.utils.deprecation import TrajectopyDeprecationWarning


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

    def test_legacy_arguments_warn(self):
        test, reference = noisy_trajectory.copy(), generated_trajectory.copy()
        with self.assertWarns(TrajectopyDeprecationWarning):
            tpy.match_trajectories(test, reference, matching_settings=tpy.MatchingSettings())
        with self.assertWarns(TrajectopyDeprecationWarning):
            tpy.estimate_alignment(test, reference, alignment_settings=tpy.AlignmentSettings())
        with self.assertWarns(TrajectopyDeprecationWarning):
            result, alignment = tpy.ate(test, reference, processing_settings=None, return_alignment=True)
        self.assertIs(result.alignment, alignment)

    def test_legacy_and_new_argument_raise(self):
        with self.assertRaises(TypeError):
            tpy.estimate_alignment(
                noisy_trajectory, generated_trajectory, settings=tpy.AlignmentSettings(), alignment_settings=None
            )


if __name__ == "__main__":
    unittest.main()
