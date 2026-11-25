import unittest

import numpy as np

import trajectopy as tpy
from test.testdata import generated_trajectory, noisy_trajectory


class TestResultsApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ate = tpy.ate(noisy_trajectory.copy(), generated_trajectory.copy())
        cls.rpe = tpy.rpe(noisy_trajectory.copy(), generated_trajectory.copy())

    def test_reprs_are_one_line(self):
        for result in (self.ate, self.rpe, self.ate.alignment):
            self.assertNotIn("\n", repr(result))
            self.assertTrue(repr(result).startswith(f"<{type(result).__name__}"))

    def test_ate_describe(self):
        statistics = self.ate.describe()
        self.assertEqual(statistics.name, self.ate.name)
        self.assertEqual(statistics["poses"], len(self.ate))
        self.assertAlmostEqual(statistics["pos_mean"], self.ate.pos_ate)
        self.assertAlmostEqual(statistics["rot_mean"], self.ate.rot_ate)

    def test_rpe_describe(self):
        table = self.rpe.describe()
        self.assertEqual(int(table["pairs"].sum()), len(self.rpe))
        np.testing.assert_allclose(table["pos_mean"], self.rpe.pos_dev_mean)
        self.assertTrue(table.index.name.startswith("pair_distance"))


if __name__ == "__main__":
    unittest.main()
