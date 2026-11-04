"""Regression tests for bugs found during the architecture cleanup."""

import unittest

import numpy as np

from test.testdata import generated_trajectory
from trajectopy.core.rotations import Rotations
from trajectopy.core.settings import MatchingMethod, MatchingSettings, PairDistanceUnit, ProcessingSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing.alignment import adopt_first_orientation
from trajectopy.processing.evaluation import _along_track_dev, rpe
from trajectopy.processing.matching import match_trajectories
from trajectopy.processing.merging import merge_trajectories
from trajectopy.utils.definitions import Sorting, TrajectoriesSorting
from trajectopy.visualization.axes import get_sorting


class TestRegressions(unittest.TestCase):
    def test_get_sorting_with_generator(self):
        sortings = [Sorting.PATH_LENGTH, Sorting.TIME]
        self.assertEqual(get_sorting(s for s in sortings), TrajectoriesSorting.MIXED)
        self.assertEqual(get_sorting(iter([Sorting.TIME, Sorting.TIME])), TrajectoriesSorting.ALL_TIME)

    def test_spatial_interpolation_skips_poses_without_neighbours(self):
        ref = generated_trajectory.copy()
        test = generated_trajectory.copy()
        test.positions.xyz[::10] += 1000.0  # far away from the reference: no neighbours within max_distance

        settings = MatchingSettings(method=MatchingMethod.NEAREST_SPATIAL_INTERPOLATED, max_distance=1.0)
        matched_test, matched_ref = match_trajectories(test, ref, matching_settings=settings, inplace=False)

        self.assertEqual(len(matched_test), len(matched_ref))
        self.assertLess(len(matched_test), len(test))
        self.assertFalse(np.isnan(matched_ref.xyz).any())

    def test_remove_ate_above_keeps_deviations_consistent(self):
        from test.testdata import noisy_trajectory
        from trajectopy.processing.evaluation import ate

        result = ate(noisy_trajectory.copy(), generated_trajectory.copy())
        _ = result.rot_dev_x  # populate the cached rotation statistics
        result.remove_ate_above(float(np.median(result.pos_dev_comb)))

        self.assertEqual(len(result.pos_dev_comb), len(result.trajectory))
        self.assertEqual(len(result.rot_dev_x), len(result.trajectory))
        self.assertEqual(len(result.abs_dev.rot_dev), len(result.trajectory))

    def test_direct_helmert_on_planar_points_is_a_rotation(self):
        from trajectopy.core.rotations import Rotations
        from trajectopy.processing.lib.alignment.direct import direct_helmert_transformation

        xyz_from = generated_trajectory.xyz  # lies exactly in a plane
        rotation = Rotations.from_euler("z", 0.3).as_matrix()
        xyz_to = 1.4 * (rotation @ xyz_from.T).T + [5.0, -3.0, 1.0]

        helmert = direct_helmert_transformation(xyz_from, xyz_to)

        self.assertAlmostEqual(np.linalg.det(helmert.rotation_matrix), 1.0)
        np.testing.assert_allclose(helmert.rotation_matrix, rotation, atol=1e-9)
        np.testing.assert_allclose(helmert.apply_to(xyz_from), xyz_to, atol=1e-6)


def _line(timestamps: np.ndarray, **kwargs) -> Trajectory:
    """Straight line along x at 1 m/s."""
    return Trajectory.from_arrays(np.c_[timestamps, 0 * timestamps, 0 * timestamps], timestamps=timestamps, **kwargs)


class TestTrajectoryOperationRegressions(unittest.TestCase):
    def test_unordered_input_gets_chronological_path_lengths_and_velocities(self):
        order = np.array([0, 5, 1, 6, 2, 7, 3, 8, 4, 9])
        t = np.arange(10.0)
        velocity = np.c_[t, 0 * t, 0 * t]  # tagged with the timestamp, to follow the reordering

        trajectory = Trajectory.from_arrays(
            np.c_[t, 0 * t, 0 * t][order], timestamps=t[order], velocity_xyz=velocity[order]
        )

        np.testing.assert_allclose(trajectory.timestamps, t)
        np.testing.assert_allclose(trajectory.path_lengths, t)
        np.testing.assert_allclose(trajectory.velocity_xyz[:, 0], t)

    def test_duplicate_epochs_keep_the_given_path_lengths(self):
        t = np.array([0.0, 1.0, 1.0, 2.0])
        path_lengths = np.array([0.0, 10.0, 99.0, 20.0])

        trajectory = Trajectory.from_arrays(np.zeros((4, 3)), timestamps=t, path_lengths=path_lengths)

        np.testing.assert_allclose(trajectory.timestamps, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(trajectory.path_lengths, [0.0, 10.0, 20.0])

    def test_merging_interleaved_trajectories_gives_monotonic_path_lengths(self):
        t = np.arange(10.0)
        merged = merge_trajectories([_line(t[::2]), _line(t[1::2])])

        np.testing.assert_allclose(merged.timestamps, t)
        np.testing.assert_allclose(merged.path_lengths, t)

    def test_transform_rotates_velocities_and_keeps_missing_orientations(self):
        trajectory = _line(np.arange(10.0))
        transformation = np.eye(4)
        transformation[:3, :3] = 2.0 * Rotations.from_euler("z", np.pi / 2).as_matrix()

        trajectory.transform(transformation)

        self.assertFalse(trajectory.has_orientation)
        np.testing.assert_allclose(trajectory.velocity_xyz[0], [0.0, 2.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(trajectory.path_lengths, 2.0 * np.arange(10.0))

    def test_adopt_first_orientation_keeps_relative_rotations(self):
        num = 5
        trajectory = Trajectory.from_arrays(np.zeros((num, 3)), quat=Rotations.random(num, random_state=1).as_quat())
        other = Trajectory.from_arrays(np.zeros((num, 3)), quat=Rotations.random(num, random_state=2).as_quat())

        adopted = adopt_first_orientation(trajectory, other, inplace=False)

        def relative_angles(rotations: Rotations) -> np.ndarray:
            return (Rotations.from_quat(rotations.as_quat()[0]).inv() * rotations).magnitude()

        np.testing.assert_allclose(adopted.quat[0], other.quat[0] * np.sign(adopted.quat[0] @ other.quat[0]))
        np.testing.assert_allclose(relative_angles(adopted.rotations), relative_angles(trajectory.rotations))

    def test_rpe_across_a_gap_has_no_zero_length_pairs(self):
        t = np.r_[np.arange(0, 50.0), np.arange(100, 150.0)]
        settings = ProcessingSettings()
        settings.relative_comparison.pair_distance_unit = PairDistanceUnit.SECOND
        settings.relative_comparison.pair_min_distance = 10.0
        settings.relative_comparison.pair_max_distance = 10.0
        settings.relative_comparison.use_all_pose_pairs = False

        result = rpe(_line(t), _line(t), settings)

        self.assertTrue(np.all(np.array(result.rpe_dev.pair_distance[10.0]) > 0))
        self.assertFalse(np.isnan(result.rpe_dev.pos_dev[10.0]).any())

    def test_spatial_interpolation_keeps_the_epsg(self):
        xyz = np.c_[np.linspace(364000, 364100, 50), np.full(50, 5621000.0), np.full(50, 100.0)]
        reference = Trajectory.from_arrays(xyz, epsg=25832)
        test = Trajectory.from_arrays(xyz + 0.05, epsg=25832)

        settings = MatchingSettings(method=MatchingMethod.NEAREST_SPATIAL_INTERPOLATED)
        matched_test, matched_ref = match_trajectories(test, reference, matching_settings=settings)

        self.assertEqual(matched_test.positions.epsg, 25832)
        self.assertEqual(matched_ref.positions.epsg, 25832)

    def test_along_track_deviation_of_the_last_point_far_behind(self):
        a, b = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        # 3 m behind the last reference point b
        deviation = _along_track_dev(p=np.array([-2.0, 0.0, 0.0]), line_pts=[a, b], is_last=True)
        self.assertAlmostEqual(deviation, -3.0)


class TestRobustnessRegressions(unittest.TestCase):
    def test_shrink_data_averages_zeros(self):
        from trajectopy.utils.numeric import shrink_data

        np.testing.assert_allclose(shrink_data(np.array([0.0, 2.0, 0.0, 2.0]), max_size=2), [1.0, 1.0])

    def test_rpe_in_seconds_of_a_spatially_sorted_trajectory(self):
        t = np.arange(100.0)
        trajectory = _line(t)
        # spatially sorted: the order of the path lengths differs from the chronological one
        trajectory.path_lengths = (t * 37) % 100
        trajectory.set_sorting(Sorting.PATH_LENGTH)
        settings = ProcessingSettings()
        settings.relative_comparison.pair_distance_unit = PairDistanceUnit.SECOND
        settings.relative_comparison.pair_min_distance = 10.0
        settings.relative_comparison.pair_max_distance = 10.0

        result = rpe(trajectory, trajectory.copy(), settings)

        np.testing.assert_allclose(result.rpe_dev.pair_distance[10.0], 10.0)

    def test_rpe_rejects_non_positive_pair_distances(self):
        for attribute in ("pair_min_distance", "pair_distance_step"):
            settings = ProcessingSettings()
            setattr(settings.relative_comparison, attribute, 0.0)
            with self.assertRaises(ValueError):
                rpe(_line(np.arange(100.0)), _line(np.arange(100.0)), settings)

    def test_spatial_interpolation_needs_two_neighbours(self):
        settings = MatchingSettings(method=MatchingMethod.NEAREST_SPATIAL_INTERPOLATED, k_nearest=1)
        with self.assertRaises(ValueError):
            match_trajectories(_line(np.arange(10.0)), _line(np.arange(10.0)), matching_settings=settings)

    def test_interpolating_a_single_pose(self):
        from trajectopy.processing.interpolation import interpolate

        single = Trajectory.from_arrays(np.ones((1, 3)), rpy=np.zeros((1, 3)), timestamps=np.array([5.0]))
        interpolated = interpolate(single, [5.0], inplace=False)

        np.testing.assert_allclose(interpolated.xyz, [[1.0, 1.0, 1.0]])
        self.assertEqual(len(interpolated.rotations), 1)

    def test_rotation_window_is_centered(self):
        from trajectopy.processing.approximation import _average_rotations_in_window

        index = np.arange(20.0)
        yaw = np.deg2rad(index)
        quat = Rotations.from_euler("xyz", np.c_[0 * yaw, 0 * yaw, yaw]).as_quat()

        averaged = Rotations.from_quat(_average_rotations_in_window(index, quat, win_size=4.0))

        # the mean of a symmetric window of linearly increasing angles is the center angle
        np.testing.assert_allclose(averaged.as_euler("xyz")[2:-2, 2], yaw[2:-2], atol=1e-6)

    def test_rotation_averaging_per_path_length_follows_the_trajectory_order(self):
        from trajectopy.processing.approximation import _average_rotations_per_path_length

        # two laps of 50 m, every pose oriented by its position on the lap
        t = np.arange(100.0)
        lap_position = t % 50
        trajectory = Trajectory.from_arrays(
            np.c_[lap_position, 0 * t, 0 * t], rpy=np.c_[0 * t, 0 * t, np.deg2rad(lap_position)], timestamps=t
        )
        trajectory.path_lengths = lap_position.copy()

        yaw = Rotations.from_quat(_average_rotations_per_path_length(trajectory)).as_euler("xyz")[:, 2]

        np.testing.assert_allclose(yaw, np.deg2rad(lap_position), atol=1e-9)

    def test_cubic_approximation_with_a_last_interval_of_exactly_the_window_size(self):
        from trajectopy.processing.lib.approximation.cubic_approximation import CubicApproximation

        index = np.arange(8.0)
        values = np.sin(index) + 5.0
        approximation = CubicApproximation(index, values, min_win_size=3.0, min_obs=3)

        self.assertLess(np.max(np.abs(approximation.est_obs - values)), 0.5)

    def test_cubic_approximation_of_too_few_observations(self):
        from trajectopy.exceptions import ApproximationError
        from trajectopy.processing.lib.approximation.cubic_approximation import CubicApproximation

        with self.assertRaises(ApproximationError):
            CubicApproximation(np.arange(2.0), np.ones(2), min_win_size=3.0, min_obs=3)

    def test_header_tolerates_spaces_and_missing_values(self):
        from trajectopy.io.header import HeaderData

        header = HeaderData.from_string("#epsg\n#fields t, px, py, pz\n1,2,3,4")

        self.assertEqual(header.epsg, 0)
        self.assertEqual(header.fields, ["t", "px", "py", "pz"])

    def test_alignment_of_integer_data(self):
        from trajectopy.processing.alignment import apply_alignment
        from trajectopy.results.alignment_result import AlignmentResult

        trajectory = Trajectory.from_arrays(np.arange(15).reshape(5, 3))
        apply_alignment(trajectory, AlignmentResult())

        np.testing.assert_allclose(trajectory.xyz, np.arange(15).reshape(5, 3))

    def test_timestamps_have_to_match_the_positions(self):
        from trajectopy.exceptions import TrajectoryError

        with self.assertRaises(TrajectoryError):
            Trajectory.from_arrays(np.zeros((5, 3)), timestamps=np.arange(4.0))

    def test_voxels_of_different_cells_stay_separate(self):
        from trajectopy.processing.lib.approximation.voxelizer import Voxelizer

        # the cells (1, 12, 3) and (11, 2, 3) must not share a voxel
        points = np.array([[0, 0, 0], [1, 12, 3], [11, 2, 3]], dtype=float) - 0.5
        voxelizer = Voxelizer(points, voxel_size=1.0)

        self.assertEqual(len(voxelizer.voxels), 3)
        self.assertEqual(len(voxelizer.k_nearest_query(points, k_nearest=10)[0]), 3)

    def test_variances_of_independent_parameters_add_up(self):
        from trajectopy.core.parameters import Parameter

        a, b = Parameter(value=1.0, variance=4.0), Parameter(value=2.0, variance=9.0)

        self.assertAlmostEqual((a + b).variance, 13.0)
        self.assertAlmostEqual((a - b).variance, 13.0)

    def test_read_ate_result_keeps_deviations_with_their_poses(self):
        import os
        import tempfile

        from trajectopy.results.ate_result import AbsoluteTrajectoryDeviations, ATEResult

        t = np.arange(6.0)
        trajectory = _line(t)
        # spatially sorted with ties in path length, which are not in chronological order
        trajectory.timestamps = np.array([3.0, 0.0, 4.0, 1.0, 5.0, 2.0])
        trajectory.path_lengths = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        trajectory._sorting = Sorting.PATH_LENGTH
        # every deviation is tagged with the timestamp of its pose
        deviations = np.c_[trajectory.timestamps, 0 * t, 0 * t]
        result = ATEResult(trajectory, AbsoluteTrajectoryDeviations(pos_dev=deviations, directed_pos_dev=deviations))

        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "ate.result")
            result.to_file(filename)
            read = ATEResult.from_file(filename)

        np.testing.assert_allclose(read.abs_dev.pos_dev[:, 0], read.trajectory.timestamps)

    def test_smoothing_keeps_constant_values_at_the_ends(self):
        from trajectopy.utils.numeric import round_to_precision

        values = np.full(10, 5.0)
        _, smoothed = round_to_precision(np.arange(10.0), values, resolution=0.1, filter_size=5)

        np.testing.assert_allclose(smoothed, 5.0)

    def test_merging_does_not_make_up_orientations(self):
        with_rotations = _line(np.arange(5.0), rpy=np.full((5, 3), 0.1))
        without_rotations = _line(np.arange(5.0, 10.0))

        merged = merge_trajectories([with_rotations, without_rotations])

        self.assertFalse(merged.has_orientation)
        self.assertEqual(len(merged), 10)

    def test_assigned_local_transformer_survives_export_and_import(self):
        import os
        import tempfile

        from trajectopy.core.positions import Positions

        xyz = np.c_[np.linspace(364000, 364100, 10), np.full(10, 5621000.0), np.full(10, 100.0)]
        georeferenced = Positions(xyz, epsg=25832)
        # positions that took over the transformer of other positions, e.g. matched or "reference EPSG"
        local = Positions(georeferenced.to_local(inplace=False).xyz, local_transformer=georeferenced.local_transformer)

        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "local_transformer.txt")
            local.export_local_transformer(filename)
            restored = Positions(local.xyz.copy())
            restored.import_local_transformer(filename)

        np.testing.assert_allclose(restored.to_epsg(25832, inplace=False).xyz, xyz, atol=1e-6)

    def test_interpolating_rotations_at_a_single_index(self):
        from trajectopy.processing.interpolation import interpolate

        t = np.arange(6.0)
        trajectory = _line(t, rpy=np.c_[0 * t, 0 * t, 0.1 * t])

        interpolated = interpolate(trajectory, [2.5], inplace=False)

        self.assertEqual(len(interpolated.rotations), 1)
        self.assertAlmostEqual(interpolated.rpy[0, 2], 0.25)

    def test_singular_adjustment_does_not_converge_instead_of_failing(self):
        import warnings

        from trajectopy.core.settings import AlignmentSettings
        from trajectopy.processing.alignment import estimate_alignment

        settings = AlignmentSettings()
        # without any uncertainty of the positions, the equations of the adjustment are singular
        for name in ("std_xy_from", "std_z_from", "std_xy_to", "std_z_to"):
            setattr(settings.stochastics, name, 0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = estimate_alignment(generated_trajectory, generated_trajectory.copy(), alignment_settings=settings)

        self.assertFalse(result.converged)

    def test_deviations_are_test_minus_reference(self):
        from trajectopy.processing.evaluation import ate

        t = np.arange(20.0)
        reference = _line(t, rpy=np.zeros((20, 3)))
        # 0.5 m ahead, 0.2 m to the left, 0.1 m above and turned 2 degrees to the left
        test = _line(t, rpy=np.c_[0 * t, 0 * t, np.full(20, np.deg2rad(2.0))])
        test.positions.xyz += [0.5, 0.2, 0.1]

        result = ate(test, reference, align=False)

        np.testing.assert_allclose([result.pos_bias_x, result.pos_bias_y, result.pos_bias_z], [0.5, 0.2, 0.1])
        np.testing.assert_allclose(
            # horizontal cross-track is positive to the right, as in the frame figure of the documentation
            [result.pos_bias_along, result.pos_bias_cross_h, result.pos_bias_cross_v],
            [0.5, -0.2, 0.1],
            atol=0.02,
        )
        self.assertAlmostEqual(np.rad2deg(result.rot_bias_z), 2.0)

    def test_ate_files_of_earlier_versions_are_converted_to_test_minus_reference(self):
        import os
        import tempfile

        from trajectopy.processing.evaluation import ate
        from trajectopy.results.ate_result import ATEResult

        t = np.arange(20.0)
        test = _line(t, rpy=np.c_[0 * t, 0 * t, np.full(20, 0.1)])
        test.positions.xyz += [0.5, 0.2, 0.1]
        result = ate(test, _line(t, rpy=np.zeros((20, 3))), align=False)

        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "ate.result")
            result.to_file(filename)
            with open(filename, encoding="utf-8") as file:
                lines = file.readlines()
            # as written by an earlier version: no marker, deviations reference minus test, cross-track to the left
            old = [line for line in lines if not line.startswith("#deviations")]
            header = old[: next(i for i, line in enumerate(old) if not line.startswith("#")) + 1]
            rows = [line.split(",") for line in old[len(header) :]]
            columns = header[-1].strip().split(",")
            for row in rows:
                for name in ("pos_dev_x", "pos_dev_y", "pos_dev_z", "pos_dev_cross_h"):
                    row[columns.index(name)] = str(-float(row[columns.index(name)]))
                quat = [float(row[columns.index(f"rot_dev_{c}")]) for c in "xyzw"]
                for c, value in zip("xyz", quat[:3]):
                    row[columns.index(f"rot_dev_{c}")] = str(-value)  # inverse rotation
            with open(filename, "w", encoding="utf-8") as file:
                file.writelines(header + [",".join(row).rstrip("\n") + "\n" for row in rows])

            read = ATEResult.from_file(filename)

        np.testing.assert_allclose(read.abs_dev.pos_dev, result.abs_dev.pos_dev)
        np.testing.assert_allclose(read.abs_dev.directed_pos_dev, result.abs_dev.directed_pos_dev, atol=1e-9)
        self.assertAlmostEqual(read.rot_bias_z, result.rot_bias_z)

    def test_cli_reports_only_a_missing_gui_as_missing_dependencies(self):
        import importlib
        import sys
        from unittest import mock

        cli = importlib.import_module("trajectopy.cli.main")

        app_module = mock.MagicMock()
        app_module.main.side_effect = ImportError("broken optional package")
        with (
            mock.patch.dict(sys.modules, {"trajectopy.gui.app": app_module}),
            mock.patch.object(sys, "argv", ["trajectopy"]),
        ):
            with self.assertRaisesRegex(ImportError, "broken optional package"):  # not hidden
                cli.main()

        with (
            mock.patch.dict(sys.modules, {"trajectopy.gui.app": None}),
            mock.patch.object(sys, "argv", ["trajectopy"]),
        ):
            with self.assertRaises(SystemExit) as exit_info, mock.patch("builtins.print"):
                cli.main()
        self.assertEqual(exit_info.exception.code, 1)

    def test_dict2table_of_an_empty_dictionary(self):
        from trajectopy.utils.formatting import dict2table

        self.assertIn("Empty", dict2table({}, "Empty"))

    def test_translation_from_a_single_pose(self):
        import warnings

        from trajectopy.core.settings import AlignmentEstimationSettings, AlignmentSettings
        from trajectopy.processing.alignment import estimate_alignment

        trajectory = Trajectory.from_arrays(np.array([[1.0, 2.0, 3.0]]), timestamps=np.array([0.0]))
        reference = Trajectory.from_arrays(np.array([[1.5, 2.0, 3.0]]), timestamps=np.array([0.0]))
        estimation = AlignmentEstimationSettings(
            translation_x=True,
            translation_y=True,
            translation_z=True,
            rotation_x=False,
            rotation_y=False,
            rotation_z=False,
        )
        settings = AlignmentSettings(estimation_settings=estimation)
        settings.stochastics.variance_estimation = True  # without redundancy, there is no variance to estimate

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = estimate_alignment(trajectory, reference, alignment_settings=settings)

        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.position_parameters.sim_trans_x.value, 0.5)

    def test_positions_outside_of_their_coordinate_system_are_rejected(self):
        from trajectopy.core.positions import Positions
        from trajectopy.exceptions import PointSetError

        # metric coordinates declared as latitude and longitude
        positions = Positions(np.array([[364000.0, 5621000.0, 100.0]]), epsg=4326, init_local_transformer=False)
        with self.assertRaisesRegex(PointSetError, "outside of the valid area"):
            positions.to_epsg(25832)

    def test_color_range_of_equal_large_values(self):
        from trajectopy.utils.colors import color_range

        lower, upper = color_range(np.full(5, 1.7e9), max_std=3.0)
        self.assertGreater(upper - lower, 0.0)

    def test_undetermined_parameters_do_not_converge_without_warnings(self):
        import warnings

        from trajectopy.core.settings import AlignmentEstimationSettings, AlignmentSettings
        from trajectopy.processing.alignment import estimate_alignment

        # rotations cannot be determined from a single pose
        trajectory = Trajectory.from_arrays(np.array([[1.0, 2.0, 3.0]]), timestamps=np.array([0.0]))
        reference = Trajectory.from_arrays(np.array([[1.5, 2.0, 3.0]]), timestamps=np.array([0.0]))
        settings = AlignmentSettings(
            estimation_settings=AlignmentEstimationSettings(translation_x=True, translation_y=True, translation_z=True)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = estimate_alignment(trajectory, reference, alignment_settings=settings)

        self.assertFalse(result.converged)

    def test_conversion_of_all_earlier_deviation_conventions(self):
        from trajectopy.io.results_io import DEVIATIONS_CONVENTION, convert_ate_deviations

        pos_dev, directed = np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]])
        rot_dev = Rotations.from_euler("xyz", np.array([[0.0, 0.0, 0.1]]))

        current = convert_ate_deviations(DEVIATIONS_CONVENTION, pos_dev, directed, rot_dev)
        cross_left = convert_ate_deviations("test-reference", pos_dev, directed, rot_dev)
        earliest = convert_ate_deviations("", pos_dev, directed, rot_dev)

        np.testing.assert_array_equal(current[0], pos_dev)
        np.testing.assert_array_equal(current[1], directed)
        np.testing.assert_array_equal(cross_left[0], pos_dev)
        np.testing.assert_array_equal(cross_left[1], [[4.0, -5.0, 6.0]])
        np.testing.assert_array_equal(earliest[0], -pos_dev)
        np.testing.assert_array_equal(earliest[1], [[4.0, -5.0, 6.0]])
        self.assertAlmostEqual(earliest[2].as_euler("xyz")[0, 2], -0.1)

    def test_malformed_trajectory_files_give_clear_errors(self):
        from trajectopy.exceptions import PointSetError

        cases = {
            "#epsg 999999\n#fields t,px,py,pz\n0,1,2,3\n": (PointSetError, "Unknown EPSG code 999999"),
            "#fields t,px,py,pz,qx,qy,qz,qw\n0;1;2;3\n": (ValueError, "header lists 8 fields"),
            "#fields t,px,py,pz\n0,1,2,3\n1e400,1,2,3\n": (ValueError, "Invalid timestamps"),
            "#fields t,px,py,pz\n0,1,2,3\n1,nan,2,3\n": (ValueError, r"Invalid positions .* row\(s\) 2"),
            "#time_format datetime\n#fields t,t,px,py,pz\n2024-01-01,,1,2,3\n": (ValueError, ""),
        }
        for content, (error, message) in cases.items():
            with self.subTest(content=content.splitlines()[0]), self.assertRaisesRegex(error, message):
                Trajectory.from_file(content, io_stream=True)

    def test_georeferenced_file_without_poses(self):
        trajectory = Trajectory.from_file("#epsg 25832\n#fields t,px,py,pz\n", io_stream=True)
        self.assertEqual(len(trajectory), 0)

    def test_invalid_derived_columns_are_derived_anew(self):
        trajectory = Trajectory.from_file("#fields t,l,px,py,pz\n0,nan,0,0,0\n1,nan,3,4,0\n", io_stream=True)
        np.testing.assert_allclose(trajectory.path_lengths, [0.0, 5.0])

    def test_rpe_deviations_are_arrays_for_fast_copies(self):
        import copy
        import time

        from trajectopy.results.rpe_result import RelativeTrajectoryDeviations

        result = rpe(_line(np.arange(2000.0)), _line(np.arange(2000.0)))
        self.assertTrue(all(isinstance(values, np.ndarray) for values in result.rpe_dev.pos_dev.values()))
        start = time.perf_counter()
        copy.deepcopy(result)  # e.g. a project save request on the GUI thread
        self.assertLess(time.perf_counter() - start, 0.5)

        # lists are still accepted
        deviations = RelativeTrajectoryDeviations(
            pos_dev={1.0: [0.1, 0.2]}, rot_dev={1.0: []}, pair_distance={1.0: [1.0, 1.0]}
        )
        self.assertEqual(deviations.num_pairs, 2)

    def test_cached_ate_statistics_follow_changes_of_the_deviations(self):
        from trajectopy.processing.evaluation import ate

        t = np.arange(20.0)
        test = _line(t, rpy=np.c_[0 * t, 0 * t, 0.01 * t])
        test.positions.xyz[:, 1] += np.where(t < 10, 0.1, 5.0)  # large deviations in the second half
        result = ate(test, _line(t, rpy=np.zeros((20, 3))), align=False)
        self.assertAlmostEqual(result.pos_dev_max, 5.0)
        before = result.rot_dev_max

        result.remove_ate_above(1.0)

        self.assertAlmostEqual(result.pos_dev_max, 0.1)
        self.assertLess(result.rot_dev_max, before)
        self.assertEqual(len(result.pos_dev_comb), 10)


if __name__ == "__main__":
    unittest.main()
