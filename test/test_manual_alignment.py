import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtCore, QtGui, QtWidgets

    from trajectopy.gui.views import manual_alignment_window as maw
except ImportError:  # pragma: no cover - GUI extras not installed
    maw = None

import numpy as np

from test.testdata import generated_trajectory
from trajectopy.core.positions import Positions
from trajectopy.core.rotations import Rotations
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing import alignment

UTM_OFFSET = np.array([364800.0, 5621500.0, 60.0])


def _transformed(trajectory: Trajectory, yaw_deg: float, translation, scale: float = 1.0) -> Trajectory:
    """``trajectory`` moved by a known similarity transformation (about the coordinate origin)."""
    rotation = Rotations.from_euler("z", np.deg2rad(yaw_deg))
    moved = trajectory.copy()
    moved.positions = Positions(xyz=scale * rotation.apply(trajectory.xyz) + np.asarray(translation), epsg=0)
    moved.rotations = rotation * trajectory.rotations
    moved.name = "moved"
    return moved


@unittest.skipIf(maw is None, "GUI dependencies not installed")
class TestManualAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _window(self, trajectory: Trajectory, reference: Trajectory) -> "maw.ManualAlignmentWindow":
        window = maw.ManualAlignmentWindow(trajectory, reference)
        self.addCleanup(window.close)
        return window

    def _aligned_full(self, window) -> Trajectory:
        result = alignment.manual_alignment(window._full_trajectory, window._full_reference, window.parameters)
        return alignment.apply_alignment(window._full_trajectory, result, inplace=False)

    def _rmse(self, window) -> float:
        aligned = self._aligned_full(window)
        return float(np.sqrt(np.mean(np.sum((aligned.xyz - window._full_reference.xyz) ** 2, axis=1))))

    def test_rotation_and_scale_act_about_the_center(self):
        trajectory = generated_trajectory.copy()
        trajectory.positions = Positions(xyz=trajectory.xyz + UTM_OFFSET, epsg=0)
        window = self._window(trajectory, trajectory.copy())
        window._apply_values({"yaw": np.deg2rad(1.0), "scale": 1.2})
        aligned = self._aligned_full(window)
        # rotating and scaling about the center keeps the center in place, even far from the origin
        np.testing.assert_allclose(np.mean(aligned.xyz, axis=0), window.pivot, atol=1e-6)

    def test_similarity_round_trip(self):
        window = self._window(generated_trajectory.copy(), generated_trajectory.copy())
        window._apply_values({"east": 3.0, "north": -2.0, "up": 1.0, "roll": 0.01, "pitch": -0.02, "yaw": 0.5})
        values = window.values
        window._set_similarity(*window._similarity())
        for key in ("east", "north", "up", "roll", "pitch", "yaw", "scale"):
            self.assertAlmostEqual(window.values[key], values[key], places=9)

    def test_best_fit_recovers_known_transformation(self):
        reference = generated_trajectory.copy()
        moved = _transformed(reference, yaw_deg=30.0, translation=[1000.0, -500.0, 20.0])
        window = self._window(moved, reference)
        self.assertGreater(self._rmse(window), 100)
        self.assertTrue(window.best_fit())
        self.assertLess(self._rmse(window), 1e-6)

    def test_best_fit_with_scale(self):
        reference = generated_trajectory.copy()
        moved = _transformed(reference, yaw_deg=-45.0, translation=[10.0, 5.0, 0.0], scale=0.7)
        window = self._window(moved, reference)
        window.best_fit(with_scale=False)
        rigid_rmse = self._rmse(window)
        window.best_fit(with_scale=True)
        self.assertLess(self._rmse(window), 1e-6)
        self.assertGreater(rigid_rmse, 1.0)
        self.assertAlmostEqual(window.values["scale"], 1 / 0.7, places=6)

    def test_center_and_match_start_bring_distant_trajectories_together(self):
        reference = generated_trajectory.copy()
        reference.positions = Positions(xyz=reference.xyz + UTM_OFFSET, epsg=0)
        moved = _transformed(generated_trajectory, yaw_deg=60.0, translation=[0.0, 0.0, 0.0])
        window = self._window(moved, reference)
        window.center_on_reference()
        aligned = self._aligned_full(window)
        np.testing.assert_allclose(np.mean(aligned.xyz, axis=0), np.mean(reference.xyz, axis=0), atol=1e-6)
        window.match_first_pose()
        self.assertLess(self._rmse(window), 5)
        self.assertIn("RMSE", window.metrics_label.text())

    def test_auto_align(self):
        reference = generated_trajectory.copy()
        moved = _transformed(reference, yaw_deg=-120.0, translation=[2500.0, 800.0, 30.0])
        window = self._window(moved, reference)
        self.assertIn(window.choose_start(), {"first pose", "fit on matched poses"})
        self.assertLess(self._rmse(window), 1e-6)

    def test_auto_align_without_usable_timestamps(self):
        """A large time offset prevents matching in time: the first pose is used instead."""
        reference = generated_trajectory.copy()
        moved = _transformed(reference, yaw_deg=45.0, translation=[300.0, -200.0, 0.0])
        moved.timestamps = moved.timestamps + 1000.0
        window = self._window(moved, reference)
        chosen = window.choose_start()
        self.assertIn("first pose", chosen)
        self.assertLess(window.spatial_distance(), 1.0)

    def test_jog_slider_is_fine_near_the_center_and_fast_at_full_deflection(self):
        window = self._window(generated_trajectory.copy(), generated_trajectory.copy())
        row = window._rows["yaw"]
        rates = {d: maw.JogSlider.steps_per_second(d) * row.spec.step for d in (0.05, 0.5, 1.0)}  # °/s
        self.assertLess(rates[0.05], 0.05)
        self.assertGreater(rates[1.0], 50)
        row.jog.setValue(maw.JOG_RANGE // 20)  # 5 % deflection
        for _ in range(100):
            row.jog._tick()
        expected = 100 * rates[0.05] * maw.JOG_TICK_MS / 1000
        # increments far below the displayed precision accumulate exactly
        self.assertAlmostEqual(np.rad2deg(window.values["yaw"]), expected, places=9)
        row.jog._stop()
        self.assertEqual(row.jog.value(), 0)
        self.assertLess(maw.JogSlider.steps_per_second(-0.5), 0)

    def test_dragging_handles_in_all_views(self):
        window = self._window(generated_trajectory.copy(), generated_trajectory.copy())
        top, side, front = window._views
        pivot = window.pivot
        top.move_handle.setPos(pivot[0] + 5.0, pivot[1] - 3.0)
        self.assertAlmostEqual(window.values["east"], 5.0)
        self.assertAlmostEqual(window.values["north"], -3.0)
        front.move_handle.setPos(pivot[1] - 3.0, pivot[2] + 2.0)
        self.assertAlmostEqual(window.values["up"], 2.0)

        center = window._pivot_position
        top.rotate_handle.setPos(center[0], center[1] + 10.0)  # pointing to +y: 90° yaw
        self.assertAlmostEqual(np.rad2deg(window.values["yaw"]), 90.0)
        side.rotate_handle.setPos(center[0], center[2] - 10.0)  # x axis tilted down in x/z: +90° pitch
        self.assertAlmostEqual(np.rad2deg(window.values["pitch"]), 90.0)
        front.rotate_handle.setPos(center[1], center[2] + 10.0)  # y axis tilted up in y/z: +90° roll
        self.assertAlmostEqual(np.rad2deg(window.values["roll"]), 90.0)

    def test_pitch_handle_matches_the_rotation(self):
        """The side view's rotate handle points where the trajectory's x axis is rotated to."""
        window = self._window(generated_trajectory.copy(), generated_trajectory.copy())
        window._apply_values({"pitch": np.deg2rad(20.0)})
        _, rotation, _ = window._similarity()
        x_axis = rotation @ [1.0, 0.0, 0.0]
        handle = window._views[1].rotate_handle.pos()
        center = window._pivot_position
        direction = np.array([handle.x() - center[0], handle.y() - center[2]])
        np.testing.assert_allclose(
            direction / np.linalg.norm(direction), x_axis[[0, 2]] / np.linalg.norm(x_axis[[0, 2]])
        )

    def test_keyboard_nudging(self):
        window = self._window(generated_trajectory.copy(), generated_trajectory.copy())
        step = window._rows["east"].spec.step
        for modifiers, expected in (
            (QtCore.Qt.KeyboardModifier.NoModifier, step),
            (QtCore.Qt.KeyboardModifier.ShiftModifier, 11 * step),
            (QtCore.Qt.KeyboardModifier.ControlModifier, 11.1 * step),
        ):
            event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Right, modifiers)
            self.assertTrue(window._handle_key(event))
            self.assertAlmostEqual(window.values["east"], expected)

    def test_applied_alignment_reproduces_the_preview(self):
        reference = generated_trajectory.copy()
        moved = _transformed(reference, yaw_deg=10.0, translation=[50.0, 0.0, 0.0])
        window = self._window(moved, reference)
        window.choose_start()
        received = []
        window.alignment_accepted.connect(received.append)
        window._accept()
        aligned = alignment.apply_alignment(moved, received[0], inplace=False)
        np.testing.assert_allclose(aligned.xyz, reference.xyz, atol=1e-6)

    def _refinement_case(self, **estimation):
        from trajectopy.core.settings import AlignmentEstimationSettings, AlignmentSettings

        reference = generated_trajectory.copy()
        moved = _transformed(reference, yaw_deg=3.0, translation=[2.0, -1.0, 0.5])
        settings = AlignmentSettings(estimation_settings=AlignmentEstimationSettings(**estimation))
        window = maw.ManualAlignmentWindow(moved, reference, alignment_settings=settings)
        self.addCleanup(window.close)
        window.match_first_pose()  # rough start values
        return window

    def test_refine_shows_the_result_and_keeps_the_window_open(self):
        window = self._refinement_case()
        rough = self._rmse(window)
        window.show()
        window._on_aligned(window.estimate(window.parameters))
        self.assertTrue(window.isVisible())
        self.assertLess(self._rmse(window), min(rough, 1e-3))
        self.assertIn("Aligned", window._status)

    def test_apply_keeps_the_least_squares_result_unless_changed(self):
        window = self._refinement_case()
        result = window.estimate(window.parameters)
        window._on_aligned(result)
        self.assertIs(window.alignment_result, result)  # statistics of the estimation are kept
        window._apply_values({"east": window.values["east"] + 0.5})
        self.assertIsNot(window.alignment_result, result)
        self.assertIn("manual", window.alignment_result.name)

    def test_refine_keeps_parameters_that_are_not_estimated(self):
        window = self._refinement_case(rotation_x=False, rotation_y=False, rotation_z=False)
        yaw = window.values["yaw"]
        window._on_aligned(window.estimate(window.parameters))
        self.assertAlmostEqual(window.values["yaw"], yaw, places=9)

    def test_refine_estimates_the_leverarm(self):
        """A known leverarm (body frame) is recovered and shown."""
        from trajectopy.core.settings import AlignmentEstimationSettings, AlignmentSettings

        reference = generated_trajectory.copy()
        lever = np.array([1.0, -0.5, 0.3])
        moved = reference.copy()
        moved.positions = Positions(xyz=reference.xyz - reference.rotations.apply(lever), epsg=0)
        estimation = AlignmentEstimationSettings(leverarm_x=True, leverarm_y=True, leverarm_z=True)
        window = maw.ManualAlignmentWindow(
            moved, reference, alignment_settings=AlignmentSettings(estimation_settings=estimation)
        )
        self.addCleanup(window.close)
        self.assertTrue(window._leverarm_box.isChecked())  # estimated parameters are visible
        self.assertTrue(window.estimation_settings.leverarm_x)

        window._on_aligned(window.estimate(window.parameters))
        np.testing.assert_allclose([window.values[key] for key in maw.LEVERARM_KEYS], lever, atol=1e-6)
        self.assertLess(self._rmse(window), 1e-6)
        self.assertIn("Aligned", window._status)

    def test_estimate_checkboxes_select_the_estimated_parameters(self):
        window = self._refinement_case()
        self.assertTrue(window.estimation_settings.rotation_z)
        window._rows["yaw"].estimate.setChecked(False)
        self.assertFalse(window.estimation_settings.rotation_z)
        yaw = window.values["yaw"]
        window._on_aligned(window.estimate(window.parameters))
        self.assertAlmostEqual(window.values["yaw"], yaw, places=9)

        for row in window._rows.values():
            row.estimate.setChecked(False)
        window.auto_align()  # nothing to estimate: only the rough placement
        self.assertIn("Placed roughly", window._status)

    def test_leverarm_cannot_be_estimated_without_orientations(self):
        from trajectopy.core.settings import AlignmentEstimationSettings, AlignmentSettings

        without_rotations = Trajectory(
            positions=generated_trajectory.positions.copy(), timestamps=generated_trajectory.timestamps.copy()
        )
        estimation = AlignmentEstimationSettings(leverarm_x=True)
        window = maw.ManualAlignmentWindow(
            without_rotations, generated_trajectory.copy(), alignment_settings=AlignmentSettings(estimation)
        )
        self.addCleanup(window.close)
        self.assertFalse(window._rows["lever_x"].estimate.isEnabled())
        self.assertFalse(window.estimation_settings.leverarm_x)

    def _wait_for_alignment(self, window) -> None:
        import time

        deadline = time.monotonic() + 30
        while window._alignment_signals is not None and not window._control_panel.isEnabled():
            if time.monotonic() > deadline:
                self.fail("alignment did not finish")
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()

    def test_auto_align_estimates_the_enabled_parameters(self):
        """Leverarm enabled in the trajectory settings + "Auto align" = spatio-temporal alignment incl. leverarm."""
        from trajectopy.core.settings import AlignmentEstimationSettings, AlignmentSettings

        reference = generated_trajectory.copy()
        lever = np.array([1.0, -0.5, 0.3])
        moved = _transformed(reference, yaw_deg=-30.0, translation=[400.0, 100.0, 5.0])
        moved.positions = Positions(xyz=moved.xyz - moved.rotations.apply(lever), epsg=0)
        settings = AlignmentSettings(
            estimation_settings=AlignmentEstimationSettings(leverarm_x=True, leverarm_y=True, leverarm_z=True)
        )
        window = maw.ManualAlignmentWindow(moved, reference, alignment_settings=settings)
        self.addCleanup(window.close)
        window.auto_align()
        self.assertFalse(window._control_panel.isEnabled())  # least squares runs in the background
        self._wait_for_alignment(window)

        self.assertIn("Aligned", window._status)
        np.testing.assert_allclose([window.values[key] for key in maw.LEVERARM_KEYS], lever, atol=1e-6)
        self.assertLess(self._rmse(window), 1e-6)
        self.assertTrue(window._leverarm_box.isChecked())
        self.assertIs(window.alignment_result, window._estimated[1])  # "Apply" keeps the estimation result

    def test_refine_manual_alignment_starts_from_the_values_shown(self):
        window = self._refinement_case()
        for row in window._rows.values():
            row.estimate.setChecked(False)
        window.refine_manual_alignment()
        self.assertIn("Nothing to refine", window._status)

    def test_refine_runs_in_the_background(self):
        window = self._refinement_case()
        window.refine_manual_alignment()
        self.assertFalse(window._control_panel.isEnabled())  # busy while estimating
        import time

        deadline = time.monotonic() + 30
        while "Aligned" not in window._status and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertIn("Aligned (start: manual alignment)", window._status)
        self.assertTrue(window._control_panel.isEnabled())

    def test_gui_applies_the_window_result(self):
        from trajectopy.gui.managers.ui_manager import UIManager
        from trajectopy.gui.models.selection import TrajectorySelection

        window = self._refinement_case()
        result = window.alignment_result
        manager = UIManager()
        requests = []
        manager.trajectory_manager_request.connect(requests.append)
        manager.handle_manual_alignment(result, selection=TrajectorySelection())
        # requests are deep copies (they are processed on the worker thread)
        self.assertEqual(requests[0].manual_alignment, result)

    def test_degree_units(self):
        self.assertEqual(maw.TIME_SERIES_LABELS[3:], ("roll [°]", "pitch [°]", "yaw [°]"))
        window = self._window(generated_trajectory.copy(), generated_trajectory.copy())
        self.assertEqual(window._rows["yaw"].spin.suffix(), " °")
        self.assertTrue(window._rows["yaw"].spin.text().endswith("°"))


if __name__ == "__main__":
    unittest.main()
