"""Position standard deviations, confidence intervals, percentiles, segment statistics and summary tables."""

import numpy as np
import pytest

import trajectopy as tpy
from test.testdata import generated_trajectory
from trajectopy.core.settings import AlignmentSettings, AlignmentStochastics
from trajectopy.processing.lib.alignment.data import AlignmentData
from trajectopy.results.summary import format_table, summary_table
from trajectopy.utils.statistics import block_bootstrap_ci, block_length


@pytest.fixture(scope="module")
def reference() -> tpy.Trajectory:
    return generated_trajectory.copy()


@pytest.fixture(scope="module")
def estimate(reference) -> tpy.Trajectory:
    rng = np.random.default_rng(1)
    trajectory = reference.copy()
    trajectory.positions.xyz = trajectory.xyz + rng.normal(0, 0.05, trajectory.xyz.shape)
    trajectory.name = "noisy"
    return trajectory


@pytest.fixture(scope="module")
def ate_result(estimate, reference) -> tpy.ATEResult:
    return tpy.ate(estimate, reference)


# --------------------------------------------------------------------------------- position standard deviations


def test_position_std_follows_masking_sorting_and_io(tmp_path, reference):
    trajectory = reference.copy()
    trajectory.position_std = np.column_stack([np.arange(len(trajectory)) + 1.0] * 3)

    masked = trajectory.mask(np.arange(0, len(trajectory), 2), inplace=False)
    np.testing.assert_allclose(masked.position_std[:, 0], np.arange(0, len(trajectory), 2) + 1.0)

    path = tmp_path / "with_std.traj"
    trajectory.to_file(str(path))
    loaded = tpy.Trajectory.from_file(str(path))
    np.testing.assert_allclose(loaded.position_std, trajectory.position_std)

    interpolated = tpy.interpolate(
        trajectory, trajectory.timestamps[:-1] + 0.5 * np.diff(trajectory.timestamps), inplace=False
    )
    np.testing.assert_allclose(interpolated.position_std[:, 0], np.arange(len(trajectory) - 1) + 1.5)


def test_invalid_position_std_is_ignored(reference):
    trajectory = tpy.Trajectory(
        positions=reference.positions.copy(),
        timestamps=reference.timestamps,
        position_std=np.zeros((len(reference), 3)),
    )
    assert trajectory.position_std is None


def test_transform_scales_position_std(reference):
    trajectory = reference.copy()
    trajectory.position_std = np.tile([1.0, 2.0, 3.0], (len(trajectory), 1))
    transformation = np.eye(4)
    transformation[:3, :3] = 2.0 * np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90° yaw, scale 2
    trajectory.transform(transformation, inplace=True)
    np.testing.assert_allclose(trajectory.position_std[0], [4.0, 2.0, 6.0])


def test_alignment_uses_position_std(estimate, reference):
    weighted = estimate.copy()
    weighted.position_std = np.tile([0.3, 0.1, 0.5], (len(weighted), 1))
    settings = AlignmentSettings()
    data = AlignmentData(weighted, reference, settings, tpy.MatchingSettings())
    var_x, var_y, var_z = data.get_var_group("XYZ_FROM")
    np.testing.assert_allclose(var_x, 0.05)  # mean of the horizontal variances
    np.testing.assert_allclose(var_y, 0.05)
    np.testing.assert_allclose(var_z, 0.25)
    np.testing.assert_allclose(data.get_var_group("XYZ_TO")[0], settings.stochastics.var_xy_to)

    unweighted = AlignmentSettings(stochastics=AlignmentStochastics(use_position_std=False))
    data = AlignmentData(weighted, reference, unweighted, tpy.MatchingSettings())
    np.testing.assert_allclose(data.get_var_group("XYZ_FROM")[0], unweighted.stochastics.var_xy_from)


# -------------------------------------------------------------------------------------- statistics of results


def test_block_bootstrap_white_noise_matches_standard_error():
    rng = np.random.default_rng(0)
    values = rng.normal(1.0, 1.0, 10_000)
    lower, upper = block_bootstrap_ci(values, block=1)
    half_width = 1.96 / np.sqrt(len(values))
    assert lower < 1.0 < upper
    assert (upper - lower) / 2 == pytest.approx(half_width, rel=0.15)


def test_block_bootstrap_widens_for_correlated_values():
    rng = np.random.default_rng(0)
    timestamps = np.arange(20_000) * 0.1
    noise = rng.normal(0, 1, len(timestamps))
    correlated = np.convolve(noise, np.ones(100) / 10, mode="same")  # correlated over ~10 s
    assert block_length(timestamps, correlated) > 10
    independent = block_bootstrap_ci(correlated, block=1)
    respecting = block_bootstrap_ci(correlated, timestamps)
    assert respecting[1] - respecting[0] > 3 * (independent[1] - independent[0])


def test_ate_percentiles_and_confidence_interval(ate_result):
    assert ate_result.pos_dev_p95 == pytest.approx(np.percentile(ate_result.pos_dev_comb, 95))
    lower, upper = ate_result.confidence_interval("position", "mean")
    assert lower <= ate_result.pos_ate <= upper
    lower, upper = ate_result.confidence_interval("position", "rms")
    assert lower <= ate_result.pos_dev_rms <= upper
    assert "95th Percentile Position [m]" in ate_result.property_dict
    assert "Mean Position 95% Confidence Interval [m]" in ate_result.property_dict
    with pytest.raises(ValueError):
        ate_result.confidence_interval("velocity")


def test_segment_statistics(ate_result):
    duration = np.ptp(ate_result.trajectory.timestamps)
    table = ate_result.segment_statistics(duration / 4)
    assert table["count"].sum() == len(ate_result.pos_dev_comb)
    assert len(table) in (4, 5)  # the last pose may start a window of its own
    assert np.all(table["pos_max"] >= table["pos_p95"])

    labels = np.where(ate_result.trajectory.timestamps < np.median(ate_result.trajectory.timestamps), "a", "b")
    by_label = ate_result.segment_statistics(labels=labels)
    assert list(by_label["segment"]) == ["a", "b"]

    with pytest.raises(ValueError):
        ate_result.segment_statistics()


def test_summary_table_and_formats(ate_result, estimate, reference):
    rpe_result = tpy.rpe(estimate, reference)
    table = summary_table([ate_result, ate_result], [rpe_result])
    assert len(table) == 2  # the repeated name is numbered
    assert table["poses"].dtype.name == "Int64"
    assert "pos_mean_ci_low [m]" in table.columns

    markdown = format_table(table, "markdown")
    assert markdown.splitlines()[1].startswith("| ---")
    latex = format_table(table, "latex")
    assert latex.startswith("\\begin{tabular}") and "pos\\_mean" in latex
    assert format_table(table, "csv").startswith("name,")
    with pytest.raises(ValueError):
        format_table(table, "xlsx")
