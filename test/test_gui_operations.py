"""Trajectory operations of the GUI keep the trajectory data consistent.

Skipped if the GUI extras are not installed.
"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from trajectopy.gui.managers.requests import TrajectoryManagerRequest, TrajectoryManagerRequestType
    from trajectopy.gui.managers.trajectory_manager import TrajectoryEntryPair, TrajectoryManager
    from trajectopy.gui.models.entries import TrajectoryEntry
except ImportError:  # pragma: no cover - GUI extras not installed
    TrajectoryManager = None

from trajectopy.core.trajectory import Trajectory
from trajectopy.utils.definitions import Sorting


def _pair(trajectory: Trajectory, **request) -> "TrajectoryEntryPair":
    return TrajectoryEntryPair(
        entry=TrajectoryEntry(full_filename="", trajectory=trajectory),
        request=TrajectoryManagerRequest(type=TrajectoryManagerRequestType.INTERPOLATE_TO_GRID, **request),
    )


def _line() -> Trajectory:
    t = np.arange(10.0)
    return Trajectory.from_arrays(np.c_[t, 2 * t, 0 * t], timestamps=t)


@unittest.skipIf(TrajectoryManager is None, "GUI dependencies not installed")
class TestTrajectoryOperations(unittest.TestCase):
    def test_grid_of_a_spatially_sorted_trajectory_is_interpolated_over_time(self):
        trajectory = _line()
        trajectory.path_lengths = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], dtype=float)
        trajectory.set_sorting(Sorting.PATH_LENGTH)

        (entry,) = TrajectoryManager.operation_interpolate_to_grid(_pair(trajectory, grid=0.5))
        result = entry.trajectory

        self.assertEqual(result.sorting, Sorting.PATH_LENGTH)
        order = np.argsort(result.timestamps)
        np.testing.assert_allclose(result.timestamps[order], np.arange(0.0, 9.01, 0.5))
        np.testing.assert_allclose(result.xyz[order, 0], result.timestamps[order])

    def test_grid_step_has_to_be_positive(self):
        with self.assertRaises(ValueError):
            TrajectoryManager.operation_interpolate_to_grid(_pair(_line(), grid=0.0))

    def test_rearranging_axes_moves_the_velocities_along(self):
        mapping = {
            "x": {"target": "y", "sign": "+", "bias": 0.0},
            "y": {"target": "x", "sign": "-", "bias": 0.0},
            "z": {"target": "z", "sign": "+", "bias": 0.0},
            "time": {"target": "time", "sign": "+", "bias": 0.0},
        }

        (entry,) = TrajectoryManager.operation_rearange_dof(_pair(_line(), dof_mapping=mapping))

        # positions (t, 2t, 0) become (-2t, t, 0), at velocity (-2, 1, 0)
        np.testing.assert_allclose(entry.trajectory.velocity_xyz[:-1], np.tile([-2.0, 1.0, 0.0], (9, 1)))

    def test_reversed_time_is_ordered_chronologically(self):
        mapping = {
            "x": {"target": "x", "sign": "+", "bias": 0.0},
            "y": {"target": "y", "sign": "+", "bias": 0.0},
            "z": {"target": "z", "sign": "+", "bias": 0.0},
            "time": {"target": "time", "sign": "-", "bias": 0.0},
        }

        (entry,) = TrajectoryManager.operation_rearange_dof(_pair(_line(), dof_mapping=mapping))

        self.assertTrue(np.all(np.diff(entry.trajectory.timestamps) > 0))

    def test_unknown_epsg_has_no_transformation(self):
        xyz = np.c_[np.linspace(364000, 364100, 10), np.full(10, 5621000.0), np.zeros(10)]
        trajectory = Trajectory.from_arrays(xyz, epsg=25832)

        (entry,) = TrajectoryManager.operation_epsg_edit(_pair(trajectory, target_epsg=0))

        self.assertIsNone(entry.trajectory.positions.local_transformer)

    def test_reading_continues_after_a_damaged_file(self):
        import tempfile

        from trajectopy.gui.managers.file_manager import FileManager
        from trajectopy.gui.managers.requests import FileRequest, FileRequestType

        manager = FileManager()
        added = []
        manager.trajectory_model_request.connect(lambda request: added.extend(request.selection.entries))
        with tempfile.TemporaryDirectory() as directory:
            damaged, valid = os.path.join(directory, "damaged.traj"), os.path.join(directory, "valid.traj")
            with open(damaged, "w", encoding="utf-8") as file:
                file.write("#fields t,qx\n1,2\n")  # no positions
            _line().to_file(valid)
            errors = []
            manager.ui_request.connect(lambda request: errors.append(request.message))
            manager.handle_request(FileRequest(type=FileRequestType.READ_TRAJ, file_list=[damaged, valid]))

        self.assertEqual(len(added), 1)
        self.assertIn("damaged.traj", errors[0])

    def test_result_files_without_id_get_distinct_ids(self):
        import tempfile

        from test.testdata import generated_trajectory
        from trajectopy.gui.models.entries import AbsoluteDeviationEntry
        from trajectopy.processing.evaluation import ate

        result = ate(generated_trajectory, generated_trajectory, align=False)
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "ate.result")
            result.to_file(filename)  # without "#id"
            first, second = AbsoluteDeviationEntry.from_file(filename), AbsoluteDeviationEntry.from_file(filename)

        self.assertTrue(first.entry_id)
        self.assertNotEqual(first.entry_id, second.entry_id)

    def test_playback_of_a_spatially_sorted_trajectory_is_chronological(self):
        from trajectopy.gui.views.playback_window import PlaybackWindow

        trajectory = _line()
        trajectory.path_lengths = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], dtype=float)
        trajectory.set_sorting(Sorting.PATH_LENGTH)

        (data,) = PlaybackWindow._build_traj_data([trajectory])

        np.testing.assert_allclose(data.t_rel, np.arange(10.0))
        # the line runs along x at 1 m/s
        np.testing.assert_allclose(np.diff(data.xyz[:, 0]), 1.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
