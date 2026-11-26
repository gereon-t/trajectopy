"""File functions accept path-like objects and overwrite existing files by default."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import trajectopy as tpy
from test.testdata import generated_trajectory, noisy_trajectory


class TestFileApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ate_result = tpy.ate(noisy_trajectory.copy(), generated_trajectory.copy())
        cls.rpe_result = tpy.rpe(noisy_trajectory.copy(), generated_trajectory.copy())

    def test_results_are_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            for result, cls in (
                (self.ate_result, tpy.ATEResult),
                (self.rpe_result, tpy.RPEResult),
                (self.ate_result.alignment, tpy.AlignmentResult),
            ):
                path = Path(directory) / f"{cls.__name__}.result"
                result.to_file(path)
                result.to_file(path)
                read = cls.from_file(path)
                self.assertIsInstance(read, cls)
                self.assertEqual(path.read_text(encoding="utf-8").count("#type"), 1)

    def test_alignment_parameters_round_trip(self):
        parameters = self.ate_result.alignment.position_parameters
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.csv"
            parameters.to_file(path)
            read = tpy.AlignmentParameters.from_file(path)
        np.testing.assert_allclose(read.values, parameters.values)
        np.testing.assert_allclose(read.covariance_matrix, parameters.covariance_matrix)
        self.assertEqual(read.enabled_bool_list, parameters.enabled_bool_list)

    def test_settings_accept_paths(self):
        settings = tpy.ProcessingSettings()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings.to_file(path)
            self.assertEqual(tpy.ProcessingSettings.from_file(path), settings)


if __name__ == "__main__":
    unittest.main()
