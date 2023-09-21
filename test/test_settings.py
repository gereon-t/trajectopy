import unittest
from pathlib import Path

from trajectopy_core.settings.alignment_settings import (
    AlignmentEstimationSettings,
    AlignmentPreprocessing,
    AlignmentSettings,
    AlignmentStochastics,
)
from trajectopy_core.settings.approximation_settings import ApproximationSettings
from trajectopy_core.settings.comparison_settings import RelativeComparisonSettings
from trajectopy_core.settings.core import Settings
from trajectopy_core.settings.matching_settings import MatchingSettings
from trajectopy_core.settings.plot_settings import PlotSettings
from trajectopy_core.settings.processing_settings import ProcessingSettings
from trajectopy_core.settings.sorting_settings import SortingSettings


class TestSettings(unittest.TestCase):
    _file = 0

    def setUp(self) -> None:
        super().setUp()
        Path("./test/tmp").mkdir(parents=True, exist_ok=True)

    def settings_io_test(self, Settings: Settings) -> None:
        filename = f"./test/tmp/settings{TestSettings._file}.yaml"
        created_settings = Settings()
        created_settings.to_file(filename)
        imported_settings = Settings.from_file(filename)
        self.assertTrue(imported_settings == created_settings)
        TestSettings._file += 1

    def test_alignment_settings(self) -> None:
        self.settings_io_test(AlignmentSettings)

    def test_alignment_preprocessing_settings(self) -> None:
        self.settings_io_test(AlignmentPreprocessing)

    def test_rel_comparison_settings(self) -> None:
        self.settings_io_test(RelativeComparisonSettings)

    def test_alignment_estimation_settings(self) -> None:
        self.settings_io_test(AlignmentEstimationSettings)

    def test_alignment_stochastics_settings(self) -> None:
        self.settings_io_test(AlignmentStochastics)

    def test_approximation_settings(self) -> None:
        self.settings_io_test(ApproximationSettings)

    def test_matching_settings(self) -> None:
        self.settings_io_test(MatchingSettings)

    def test_plot_settings(self) -> None:
        self.settings_io_test(PlotSettings)

    def test_processing_settings(self) -> None:
        self.settings_io_test(ProcessingSettings)

    def test_sorting_settings(self) -> None:
        self.settings_io_test(SortingSettings)
