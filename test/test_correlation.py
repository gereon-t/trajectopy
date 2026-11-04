"""Temporal correlation of deviations: known correlation times are found, gaps and irregular sampling are handled."""

import unittest

import numpy as np

from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.evaluation import ate
from trajectopy.utils.definitions import Sorting
from trajectopy.utils.statistics import autocorrelation, correlation_time, effective_samples, windowed_std


def _gauss_markov(timestamps: np.ndarray, tau: float, rng: np.random.Generator) -> np.ndarray:
    """First-order Gauss-Markov process (exponential autocorrelation with the correlation time tau)."""
    values = np.empty(len(timestamps))
    values[0] = rng.normal()
    for i in range(1, len(timestamps)):
        factor = np.exp(-(timestamps[i] - timestamps[i - 1]) / tau)
        values[i] = factor * values[i - 1] + rng.normal(0, np.sqrt(1 - factor**2))
    return values


class TestAutocorrelation(unittest.TestCase):
    def test_correlation_time_of_a_gauss_markov_process(self):
        rng = np.random.default_rng(0)
        timestamps = np.arange(0, 3600, 1 / 19)  # one hour at 19 Hz
        for tau in (0.5, 5.0):
            with self.subTest(tau=tau):
                lags, rho, _ = autocorrelation(timestamps, _gauss_markov(timestamps, tau, rng), max_lag=60)
                self.assertAlmostEqual(correlation_time(lags, rho), tau, delta=0.15 * tau)

    def test_gaps_and_jitter(self):
        rng = np.random.default_rng(1)
        timestamps = np.arange(0, 3600, 1 / 19) + rng.normal(0, 0.002, 68400)
        values = _gauss_markov(np.sort(timestamps), 5.0, rng)
        keep = np.ones(len(timestamps), dtype=bool)
        for start in rng.integers(0, len(timestamps), 20):
            keep[start : start + 300] = False  # gaps of about 16 s, e.g. losses of lock

        lags, rho, pairs = autocorrelation(np.sort(timestamps)[keep], values[keep], max_lag=60)

        self.assertAlmostEqual(correlation_time(lags, rho), 5.0, delta=0.75)
        self.assertTrue(np.all(pairs[1:] < pairs[0]))

    def test_no_correlation_across_a_gap(self):
        # 10 s blocks with alternating signs, separated by 20 s gaps: within a block, the values are perfectly
        # correlated; empty grid cells of the gaps must not produce pairs
        timestamps = np.concatenate([np.arange(0, 10, 0.1) + 30 * block for block in range(40)])
        values = np.concatenate([np.full(100, (-1.0) ** block) for block in range(40)])

        lags, rho, pairs = autocorrelation(timestamps, values)

        np.testing.assert_allclose(rho[(lags > 0) & (lags < 9.95)], 1.0)
        self.assertTrue(np.all(pairs[(lags > 10.05) & (lags < 19.95)] == 0))  # only gaps at these lags
        self.assertTrue(np.all(rho[(lags > 20.05) & (lags < 29.95)] < 0))  # neighboring blocks, opposite signs

    def test_lags_up_to_a_tenth_of_the_samples(self):
        timestamps = np.arange(1000) * 0.1
        lags, _, _ = autocorrelation(timestamps, np.random.default_rng(6).normal(size=1000), max_lag=600)
        self.assertAlmostEqual(lags[-1], 10.0)  # 100 sampling intervals

    def test_correlation_time_beyond_the_largest_lag(self):
        timestamps = np.arange(1000) * 0.1
        lags, rho, _ = autocorrelation(timestamps, np.linspace(-1, 1, 1000))  # a trend: correlated throughout
        self.assertEqual(correlation_time(lags, rho), float("inf"))

    def test_white_noise_is_not_correlated(self):
        rng = np.random.default_rng(2)
        timestamps = np.arange(5000) * 0.01
        lags, rho, _ = autocorrelation(timestamps, rng.normal(size=5000))
        self.assertLess(correlation_time(lags, rho), 0.01)

    def test_effective_samples(self):
        self.assertEqual(effective_samples(1000, 100.0, 5.0), 10.0)
        self.assertEqual(effective_samples(1000, 100.0, 0.0), 1000.0)
        self.assertEqual(effective_samples(1000, 100.0, np.inf), 1.0)

    def test_windowed_std(self):
        timestamps = np.arange(0, 20, 0.1)
        values = np.r_[np.random.default_rng(3).normal(0, 1, 100), np.random.default_rng(4).normal(0, 3, 100)]
        centers, stds = windowed_std(timestamps, values, window=10.0)
        np.testing.assert_allclose(centers, [5.0, 15.0])
        self.assertLess(stds[0], stds[1])


class TestDeviationCorrelation(unittest.TestCase):
    def _result(self, sorting: Sorting = Sorting.TIME):
        rng = np.random.default_rng(5)
        t = np.arange(0, 600, 0.1)
        reference = Trajectory.from_arrays(np.c_[t, 0 * t, 0 * t], rpy=np.zeros((len(t), 3)), timestamps=t)
        test = reference.copy()
        # cross-track deviations correlated over 5 s, along-track white noise
        test.positions.xyz += np.c_[rng.normal(0, 0.01, len(t)), _gauss_markov(t, 5.0, rng) * 0.02, 0 * t]
        result = ate(test, reference, align=False)
        if sorting == Sorting.PATH_LENGTH:
            result.trajectory.path_lengths = result.trajectory.path_lengths[::-1].copy()
            result.set_sorting(Sorting.PATH_LENGTH)
        return result

    def test_components_and_properties(self):
        result = self._result()
        correlation = result.correlation

        self.assertAlmostEqual(correlation.components["cross_h"].correlation_time, 5.0, delta=1.5)
        self.assertLess(correlation.components["along"].correlation_time, 0.1)
        self.assertLess(correlation.components["cross_h"].effective_samples, 100)
        self.assertIn("roll", correlation.components)
        self.assertIn("Correlation Time Along / Cross-H / Cross-V [s]", result.property_dict)

    def test_spatially_sorted_results_are_analyzed_in_chronological_order(self):
        by_time, by_path = self._result(), self._result(Sorting.PATH_LENGTH)
        self.assertAlmostEqual(
            by_time.correlation.components["cross_h"].correlation_time,
            by_path.correlation.components["cross_h"].correlation_time,
        )

    def test_removing_deviations_updates_the_correlation(self):
        result = self._result()
        samples = result.correlation.num_samples
        result.remove_ate_above(0.02)
        self.assertLess(result.correlation.num_samples, samples)


if __name__ == "__main__":
    unittest.main()
