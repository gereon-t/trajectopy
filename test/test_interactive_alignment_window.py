import gc
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtCore, QtGui, QtWidgets

    from trajectopy.gui.alignment import canvas as canvas_module
    from trajectopy.gui.alignment.session import AlignmentSession
    from trajectopy.gui.alignment.window import InteractiveAlignmentWindow
except ImportError:  # pragma: no cover - GUI extras not installed
    InteractiveAlignmentWindow = None

import numpy as np

from test.test_alignment_diagnostics import drive, with_positions
from test.test_interactive_alignment import UTM_OFFSET, transformed, with_varying_speed
from test.testdata import generated_trajectory
from trajectopy.core.positions import Positions
from trajectopy.core.settings import AlignmentEstimationSettings, AlignmentSettings
from trajectopy.core.trajectory import Trajectory
from trajectopy.processing import alignment
from trajectopy.processing.lib.alignment import diagnostics
from trajectopy.processing.lib.alignment.interactive import AlignmentState, axis_rotation

Z_AXIS = np.array([0.0, 0.0, 1.0])


class _Drag:
    """A mouse drag event of pyqtgraph, with positions given in view (data) coordinates."""

    def __init__(self, canvas, start, position, first=False, last=False, modifiers=None) -> None:
        self._canvas, self._start, self._position = canvas, start, position
        self._first, self._last = first, last
        self._modifiers = modifiers or QtCore.Qt.KeyboardModifier.NoModifier

    def _item(self, position):
        return self._canvas._view_box.mapFromView(QtCore.QPointF(*position))

    def button(self):
        return QtCore.Qt.MouseButton.LeftButton

    def pos(self):
        return self._item(self._position)

    def buttonDownPos(self):
        return self._item(self._start)

    def isStart(self):
        return self._first

    def isFinish(self):
        return self._last

    def modifiers(self):
        return self._modifiers

    def accept(self):
        pass


def _drag(canvas, start, end, modifiers=None, steps: int = 4) -> bool:
    handled = False
    for i in range(1, steps + 1):
        position = np.asarray(start) + (np.asarray(end) - np.asarray(start)) * i / steps
        handled = canvas.drag(_Drag(canvas, start, position, first=i == 1, last=i == steps, modifiers=modifiers))
    return handled


@unittest.skipIf(InteractiveAlignmentWindow is None, "GUI dependencies not installed")
class TestInteractiveAlignmentWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        # runs last: closed windows are released here, on the GUI thread, instead of whenever the garbage
        # collector happens to run, e.g. on the thread of a background alignment, where deleting widgets crashes
        self.addCleanup(gc.collect)

    def _window(self, trajectory: Trajectory, reference: Trajectory, **estimation) -> "InteractiveAlignmentWindow":
        settings = AlignmentSettings(estimation_settings=AlignmentEstimationSettings(**estimation))
        window = InteractiveAlignmentWindow(trajectory, reference, alignment_settings=settings)
        self.addCleanup(window.close)
        window.show()
        self.app.processEvents()
        return window

    def _identical(self, **estimation) -> "InteractiveAlignmentWindow":
        return self._window(generated_trajectory.copy(), generated_trajectory.copy(), **estimation)

    def _aligned(self, window) -> Trajectory:
        session = window.session
        return alignment.apply_alignment(session.trajectory, session.alignment_result(), inplace=False)

    def _rmse(self, window) -> float:
        aligned = self._aligned(window)
        return float(np.sqrt(np.mean(np.sum((aligned.xyz - window.session.reference.xyz) ** 2, axis=1))))

    def _wait(self, window) -> None:
        deadline = time.monotonic() + 30
        while window.session.busy:
            if time.monotonic() > deadline:
                self.fail("alignment did not finish")
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()

    # -------------------------------------------------------------------------------------------- session

    def test_undo_and_redo(self):
        session = self._identical().session
        self.assertFalse(session.can_undo)
        start = session.state
        session.set_state(start.moved([1.0, 0.0, 0.0]), "Move")
        session.set_state(session.state.moved([0.0, 2.0, 0.0]), "Move")
        session.undo()
        np.testing.assert_allclose(session.state.translation, [1.0, 0.0, 0.0])
        session.redo()
        np.testing.assert_allclose(session.state.translation, [1.0, 2.0, 0.0])
        session.undo()
        session.undo()
        self.assertEqual(session.state, start)
        self.assertFalse(session.can_undo)
        session.set_state(start.moved([5.0, 0.0, 0.0]), "Move")  # a new step discards what could be redone
        self.assertFalse(session.can_redo)
        session.reset()
        self.assertEqual(session.state, start)
        session.undo()  # also the reset
        np.testing.assert_allclose(session.state.translation, [5.0, 0.0, 0.0])

    def test_a_drag_is_a_single_step(self):
        window = self._identical()
        canvas, session = window.canvas, window.session
        start = canvas.projection.project(session.preview.xyz[10])
        self.assertTrue(_drag(canvas, start, start + [5.0, -3.0], steps=6))
        np.testing.assert_allclose(session.state.translation, [5.0, -3.0, 0.0], atol=1e-9)
        self.assertEqual(session.undo_label, "Move")
        session.undo()
        self.assertEqual(session.state, AlignmentState())

    def test_dragging_away_from_the_trajectory_pans(self):
        window = self._identical()
        canvas = window.canvas
        far = canvas.projection.project(window.session.preview.xyz[0]) + 10 * window.session.preview.extent
        self.assertFalse(_drag(canvas, far, far + [5.0, 5.0]))
        self.assertEqual(window.session.state, AlignmentState())

    def test_rotating_by_dragging_turns_about_the_view_direction_and_the_pivot(self):
        """Also after other rotations (Euler angles would turn about a different axis then)."""
        window = self._identical()
        canvas, session = window.canvas, window.session
        session.set_state(session.state.rotated_about(session.pivot, axis_rotation(Z_AXIS, np.pi / 2)), "Rotate")
        pivot = session.pivot.copy()
        before = session.preview.positions(session.state)

        window.view_combo.setCurrentIndex(1)  # side view: x to the right, z up
        center = canvas.projection.project(pivot)
        control = QtCore.Qt.KeyboardModifier.ControlModifier
        self.assertTrue(_drag(canvas, center + [10.0, 0.0], center + [0.0, 10.0], modifiers=control))

        after = session.preview.positions(session.state)
        np.testing.assert_allclose(session.pivot, pivot, atol=1e-9)
        # a quarter turn in the x/z plane: x -> z, z -> -x; y (the view direction) is unchanged
        relative_before, relative_after = before - pivot, after - pivot
        np.testing.assert_allclose(relative_after[:, 1], relative_before[:, 1], atol=1e-9)
        np.testing.assert_allclose(relative_after[:, 2], relative_before[:, 0], atol=1e-9)
        np.testing.assert_allclose(relative_after[:, 0], -relative_before[:, 2], atol=1e-9)

    def test_scaling_by_dragging(self):
        window = self._identical()
        canvas, session = window.canvas, window.session
        center = canvas.projection.project(session.pivot)
        shift = QtCore.Qt.KeyboardModifier.ShiftModifier
        self.assertTrue(_drag(canvas, center + [10.0, 0.0], center + [0.0, 15.0], modifiers=shift))
        self.assertAlmostEqual(session.state.scale, 1.5)
        np.testing.assert_allclose(session.pivot, session.preview.center, atol=1e-9)

    def test_pivot_moves_with_the_trajectory_and_can_be_placed(self):
        trajectory = generated_trajectory.copy()
        trajectory.positions = Positions(xyz=trajectory.xyz + UTM_OFFSET, epsg=0)
        window = self._window(trajectory, trajectory.copy())
        session = window.session
        np.testing.assert_allclose(session.pivot, session.preview.center)
        session.set_state(session.state.moved([3.0, 0.0, 0.0]), "Move")
        np.testing.assert_allclose(session.pivot_shift, [3.0, 0.0, 0.0], atol=1e-6)

        pose = session.preview.positions(session.state)[50]
        window.canvas._place_pivot(window.canvas.projection.project(pose))  # as by a double-click
        np.testing.assert_allclose(session.pivot, pose, atol=1e-6)
        window.parameters.set_value("yaw", np.deg2rad(30.0))
        window.parameters.set_value("scale", 1.2)
        # rotating and scaling keep the pose under the pivot in place, even far from the origin
        np.testing.assert_allclose(session.preview.positions(session.state)[50], pose, atol=1e-6)
        session.center_pivot()
        np.testing.assert_allclose(session.pivot, session.state.apply(session.preview.center), atol=1e-6)

    def test_keyboard_nudging(self):
        window = self._identical()
        canvas, session = window.canvas, window.session
        step = 10.0 * canvas._pixel_size()[0]

        def press(key, modifiers=QtCore.Qt.KeyboardModifier.NoModifier) -> bool:
            return canvas.handle_key(QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key, modifiers))

        self.assertTrue(press(QtCore.Qt.Key.Key_Right))
        self.assertAlmostEqual(session.state.translation[0], step)
        self.assertTrue(press(QtCore.Qt.Key.Key_Right, QtCore.Qt.KeyboardModifier.ShiftModifier))
        self.assertAlmostEqual(session.state.translation[0], 11 * step)
        self.assertTrue(press(QtCore.Qt.Key.Key_Left, QtCore.Qt.KeyboardModifier.ControlModifier))
        self.assertAlmostEqual(session.state.translation[0], 10.9 * step)
        self.assertTrue(press(QtCore.Qt.Key.Key_Q))
        self.assertAlmostEqual(np.rad2deg(session.state.euler[2]), 0.5)
        self.assertFalse(press(QtCore.Qt.Key.Key_A))

    # ---------------------------------------------------------------------------------------- point pairs

    def test_picking_point_pairs_places_the_trajectory(self):
        reference = generated_trajectory.copy()
        window = self._window(transformed(reference, 35.0, [40.0, -25.0, 3.0]), reference)
        canvas, session = window.canvas, window.session
        window.pick_button.setChecked(True)
        self.assertEqual(canvas.tool, canvas_module.TOOL_PAIRS)

        count = len(session.preview.t)
        for number, index in enumerate((0, count // 3, 2 * count // 3), start=1):
            position = session.preview.positions(session.state)[index]
            canvas._pick(canvas.projection.project(position))
            self.assertIn("reference", window.hint_label.text())
            canvas._pick(canvas.projection.project(session.preview.reference_xyz[index]))
            self.assertEqual(len(session.pairs), number)
            self.assertEqual(window.pairs_label.text(), f"{number} point pair{'' if number == 1 else 's'}")

        self.assertLess(self._rmse(window), 1e-6)
        self.assertIn("3 point pairs", window.message_label.text())
        window.remove_pair_button.click()
        self.assertEqual(window.pairs_label.text(), "2 point pairs")
        window.clear_pairs_button.click()
        self.assertEqual(window.pairs_label.text(), "No point pairs")
        self.assertFalse(window.clear_pairs_button.isEnabled())

    def test_a_pick_far_from_both_trajectories_is_ignored(self):
        window = self._identical()
        window.pick_button.setChecked(True)
        far = window.canvas.projection.project(window.session.preview.xyz[0]) + 10 * window.session.preview.extent
        window.canvas._pick(far)
        self.assertIsNone(window.canvas._pending)
        self.assertIn("Click closer", window.hint_label.text())

    # ----------------------------------------------------------------------------------------------- time

    def test_sync_time_and_time_shift_by_dragging(self):
        reference = with_varying_speed(generated_trajectory)
        moved = reference.copy()
        moved.timestamps = moved.timestamps + 750.0
        window = self._window(moved, reference)
        session = window.session
        window.model.update()
        summary = window.diagnosis.summary.text()
        self.assertIn("No poses at equal time", summary)
        self.assertIn("0.000 m RMS", summary)  # the distance between the paths is shown nevertheless

        session.sync_time()
        self.assertAlmostEqual(session.state.time_shift, 750.0, delta=np.median(np.diff(session.preview.t)))
        window.model.update()
        self.assertIn("Deviation at equal time", window.diagnosis.summary.text())

        # dragging 5 s to the right in the speed plot moves the trajectory to later times
        view_box = window.over_time.speed_plot.getViewBox()
        shift = session.state.time_shift

        class Drag(_Drag):
            def _item(self, position):
                return view_box.mapFromView(QtCore.QPointF(*position))

        view_box.mouseDragEvent(Drag(None, (100.0, 0.0), (105.0, 0.0), first=True, last=True))
        self.assertAlmostEqual(session.state.time_shift, shift - 5.0)
        self.assertEqual(session.undo_label, "Time shift")

    def test_sync_time_explains_when_it_is_not_possible(self):
        t = np.arange(200.0)
        xyz = np.column_stack((5.0 * t, np.zeros_like(t), np.zeros_like(t)))  # a constant speed
        straight = Trajectory(positions=Positions(xyz=xyz, epsg=0), timestamps=t)
        window = self._window(straight, straight.copy())
        window.session.sync_time()
        self.assertIn("speed that varies", window.message_label.text())
        self.assertEqual(window.session.state, AlignmentState())

    # ----------------------------------------------------------------------------------------- parameters

    def test_editing_values(self):
        window = self._identical()
        panel, session = window.parameters, window.session
        panel.spins["shift_x"].setValue(5.0)
        panel.spins["yaw"].setValue(90.0)
        panel.spins["time_shift"].setValue(2.5)
        np.testing.assert_allclose(session.pivot_shift, [5.0, 0.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(np.rad2deg(session.state.euler[2]), 90.0)
        self.assertAlmostEqual(session.state.time_shift, 2.5)
        self.assertEqual(session.undo_label, "Set time shift")
        session.undo()
        self.assertAlmostEqual(panel.spins["time_shift"].value(), 0.0)
        self.assertAlmostEqual(panel.spins["yaw"].value(), 90.0)
        self.assertEqual(panel.spins["yaw"].suffix(), " °")

    def test_estimate_checkboxes_select_the_estimated_parameters(self):
        window = self._identical(scale=True, rotation_x=False)
        panel = window.parameters
        self.assertTrue(panel.estimation_settings.scale)
        self.assertFalse(panel.estimation_settings.rotation_x)
        panel.estimate["yaw"].setChecked(False)
        self.assertFalse(panel.estimation_settings.rotation_z)
        self.assertFalse(panel.estimation_settings.sensor_rotation)

        for check in panel.estimate.values():
            check.setChecked(False)
        window.refine_button.click()
        self.assertFalse(window.session.busy)
        self.assertIn("Nothing to estimate", window.message_label.text())

    def test_leverarm_needs_orientations(self):
        without_rotations = Trajectory(
            positions=generated_trajectory.positions.copy(), timestamps=generated_trajectory.timestamps.copy()
        )
        window = self._window(without_rotations, generated_trajectory.copy(), leverarm_x=True)
        panel = window.parameters
        self.assertFalse(panel.estimate["lever_x"].isEnabled())
        self.assertFalse(panel.spins["lever_x"].isEnabled())
        self.assertFalse(panel.estimation_settings.leverarm_x)

    # ----------------------------------------------------------------------------------------- estimation

    def _roughly_placed(self, **estimation) -> "InteractiveAlignmentWindow":
        reference = generated_trajectory.copy()
        window = self._window(transformed(reference, 3.0, [2.0, -1.0, 0.5]), reference, **estimation)
        window.session.match_first_pose()
        return window

    def test_refine_runs_in_the_background(self):
        window = self._roughly_placed()
        rough = self._rmse(window)
        window.refine_button.click()
        self.assertTrue(window.session.busy)
        self.assertFalse(window.controls.isEnabled())
        self.assertFalse(window.apply_button.isEnabled())
        self.assertTrue(window.busy_row.isVisible())
        self._wait(window)
        self.assertTrue(window.controls.isEnabled())
        self.assertFalse(window.busy_row.isVisible())
        self.assertTrue(window.isVisible())
        self.assertLess(self._rmse(window), min(rough, 1e-3))
        self.assertIn("least squares", window.message_label.text())
        self.assertEqual(window.session.undo_label, "Refine")

    def test_cancelled_estimation_is_discarded(self):
        window = self._roughly_placed()
        state = window.session.state
        window.refine_button.click()
        window.session.cancel_estimation()
        self.assertFalse(window.session.busy)
        time.sleep(1.0)
        self.app.processEvents()
        self.assertEqual(window.session.state, state)
        self.assertIn("cancelled", window.message_label.text())

    def test_apply_keeps_the_least_squares_result_unless_changed(self):
        window = self._roughly_placed()
        session = window.session
        session.estimate_now(window.parameters.estimation_settings)
        result = session.alignment_result()
        self.assertNotIn("manual", result.name)  # the estimation result with its statistics
        self.assertIs(session.alignment_result(), result)
        session.set_state(session.state.moved([0.5, 0.0, 0.0]), "Move")
        self.assertIn("manual", session.alignment_result().name)
        session.undo()
        self.assertIs(session.alignment_result(), result)

    def test_refine_keeps_parameters_that_are_not_estimated(self):
        window = self._roughly_placed(rotation_x=False, rotation_y=False, rotation_z=False)
        yaw = window.session.state.euler[2]
        window.session.estimate_now(window.parameters.estimation_settings)
        self.assertAlmostEqual(window.session.state.euler[2], yaw, places=9)
        self.assertIn("least squares", window.message_label.text())

    def test_auto_align_estimates_the_enabled_parameters(self):
        """Leverarm enabled in the trajectory settings + "Auto align" = spatio-temporal alignment incl. leverarm."""
        reference = generated_trajectory.copy()
        lever = np.array([1.0, -0.5, 0.3])
        moved = transformed(reference, -30.0, [400.0, 100.0, 5.0])
        moved.positions = Positions(xyz=moved.xyz - moved.rotations.apply(lever), epsg=0)
        window = self._window(moved, reference, leverarm_x=True, leverarm_y=True, leverarm_z=True)
        window.auto_button.click()
        self.assertTrue(window.session.busy)  # least squares runs in the background
        self._wait(window)

        np.testing.assert_allclose(window.session.state.leverarm, lever, atol=1e-6)
        self.assertAlmostEqual(window.parameters.spins["lever_x"].value(), 1.0, places=3)
        self.assertLess(self._rmse(window), 1e-6)
        self.assertNotIn("manual", window.session.alignment_result().name)
        # one undo returns to the rough placement, a second one to the start
        window.session.undo()
        window.session.undo()
        self.assertEqual(window.session.state, AlignmentState())

    def test_auto_align_with_a_large_time_offset(self):
        reference = with_varying_speed(generated_trajectory)
        moved = transformed(reference, 45.0, [300.0, -200.0, 0.0])
        moved.timestamps = moved.timestamps + 1000.0
        window = self._window(moved, reference, time_shift=True)
        window.auto_button.click()
        self._wait(window)
        self.assertIn("least squares", window.message_label.text())
        self.assertAlmostEqual(window.session.state.time_shift, 1000.0, places=3)
        aligned = self._aligned(window)
        np.testing.assert_allclose(aligned.timestamps, reference.timestamps, atol=1e-3)
        np.testing.assert_allclose(aligned.xyz, reference.xyz, atol=1e-3)

    # --------------------------------------------------------------------------------------------- window

    def test_applied_alignment_reproduces_the_preview(self):
        reference = generated_trajectory.copy()
        moved = transformed(reference, 10.0, [50.0, 0.0, 0.0])
        window = self._window(moved, reference)
        window.session.match_first_pose()
        window.parameters.set_value("shift_z", 1.0)
        received = []
        window.alignment_accepted.connect(received.append)
        preview = window.session.preview.positions(window.session.state)
        window.accept()
        self.assertFalse(window.isVisible())
        aligned = alignment.apply_alignment(moved, received[0], inplace=False)
        np.testing.assert_allclose(aligned.xyz, preview, atol=1e-6)

    def test_views_and_plots_follow_the_state(self):
        window = self._identical()
        for index in range(window.view_combo.count()):
            window.view_combo.setCurrentIndex(index)
        for index in range(window.arrows_combo.count()):
            window.arrows_combo.setCurrentIndex(index)
        window.over_time.time_hovered.emit(float(window.session.preview.reference_t[5]))
        window.session.set_state(window.session.state.moved([2.0, 0.0, 0.0]), "Move")
        for index in range(window.tabs.count()):  # every view draws the new state
            window.tabs.setCurrentIndex(index)
            window.model.update()
            self.app.processEvents()
        self.assertIn("RMS 2.000 m", window.diagnosis.summary.text())
        x, _ = window.views["coordinates"]._curves[0].getData()
        self.assertEqual(len(x), len(window.session.preview.t))
        self.assertTrue(window.undo_action.isEnabled())
        self.assertIn("Move", window.undo_action.toolTip())

    # ------------------------------------------------------------------------------------------ diagnosis

    def _drive_with_errors(self, **estimation) -> "InteractiveAlignmentWindow":
        """A drive with a leverarm, a time shift and a heading error (known values). By default, all of them are
        estimated (besides the shift and the rotation)."""
        reference = drive()
        lever = np.array([1.2, -0.4, 0.0])
        moved = with_positions(reference, reference.xyz - reference.rotations.apply(lever))
        moved.timestamps = moved.timestamps + 0.15
        window = self._window(
            moved,
            reference,
            **(estimation or dict(time_shift=True, leverarm_x=True, leverarm_y=True, leverarm_z=True)),
        )
        session = window.session
        session.set_state(session.state.rotated_about(session.pivot, axis_rotation(Z_AXIS, 0.005)), "Rotate")
        window.model.update()
        return window

    def _listed(self, window) -> list[str]:
        return window.diagnosis.row_names()

    def test_diagnosis_lists_the_parameters_and_links_to_their_signature(self):
        window = self._drive_with_errors()
        names = self._listed(window)
        expected = {"Leverarm x (body)", "Time shift", "Leverarm y (body)", "Heading (about z)"}
        self.assertEqual(set(names[:4]), expected)
        table = window.diagnosis.table
        self.assertIn("ms", table.item(names.index("Time shift"), 2).text())
        table.selectRow(names.index("Time shift"))
        self.assertIs(window.tabs.currentWidget(), window.views[diagnostics.VIEW_TIME])
        table.selectRow(names.index("Leverarm x (body)"))
        self.assertIs(window.tabs.currentWidget(), window.views[diagnostics.VIEW_STANDSTILL])

    def test_only_estimated_parameters_are_diagnosed(self):
        """With the default settings, neither the time shift nor the leverarm is estimated: they are not ranked,
        but the strongest one is pointed out."""
        window = self._drive_with_errors(scale=False)
        names = self._listed(window)
        self.assertFalse({"Time shift", "Leverarm x (body)", "Leverarm y (body)"} & set(names))
        self.assertIn("Not estimated", window.diagnosis.note.text())

        # checking “estimate” takes the parameter into account at once
        window.parameters.estimate["time_shift"].setChecked(True)
        self.assertIn("time_shift", window.session.estimated_parameters)
        self.assertIn("Time shift", self._listed(window))
        window.parameters.estimate["time_shift"].setChecked(False)
        self.assertNotIn("Time shift", self._listed(window))

    def test_corrections_estimate_only_the_estimated_parameters_along(self):
        """A correction uses the estimated parameters as companions only: the value applied is the value shown."""
        window = self._drive_with_errors(scale=False)
        shown = {correction.key: correction.value for correction in window.model.diagnosis.corrections}
        heading = window.session.state.euler[2]
        window.views["geometry"].action_buttons[0].click()  # correct heading
        self.assertAlmostEqual(window.session.state.euler[2] - heading, shown["rot_z"], delta=2e-4)

    def test_no_estimated_parameter(self):
        window = self._drive_with_errors(scale=False)
        for check in window.parameters.estimate.values():
            check.setChecked(False)
        self.assertEqual(self._listed(window), [])
        self.assertIn("No parameter is estimated", window.diagnosis.note.text())

    def test_corrections_one_after_another_find_the_true_values(self):
        """Each correction is estimated together with all other parameters, so the order does not matter."""
        window = self._drive_with_errors()
        session = window.session
        geometry, timing, standstill = (window.views[key] for key in ("geometry", "time", "standstill"))
        geometry.action_buttons[0].click()  # heading
        self.assertEqual(session.undo_label, "Correct heading")
        timing.action_buttons[0].click()
        standstill.action_buttons[0].click()
        self.assertAlmostEqual(session.state.time_shift, 0.15, delta=1e-3)
        np.testing.assert_allclose(session.state.leverarm[:2], [1.2, -0.4], atol=1e-3)
        window.model.update()
        self.assertLess(window.model.residuals.rms, 0.01)

    def _noisy_drive_with_errors(self) -> "InteractiveAlignmentWindow":
        """Like _drive_with_errors, with noise on both trajectories (1 cm) and more errors."""
        reference = drive(noise=0.01, seed=1)
        truth = drive()
        lever = np.array([1.2, -0.4, 0.8])
        noise = np.random.default_rng(2).normal(scale=0.01, size=truth.xyz.shape)
        moved = with_positions(truth, truth.xyz - truth.rotations.apply(lever) + noise)
        moved.timestamps = moved.timestamps + 0.15
        window = self._window(moved, reference, time_shift=True, leverarm_x=True, leverarm_y=True, leverarm_z=True)
        session = window.session
        session.set_state(session.state.rotated_about(session.pivot, axis_rotation(Z_AXIS, 0.005)), "Rotate")
        window.model.update()
        return window

    def test_correcting_a_row_does_not_bring_up_new_rows(self):
        """Corrections remove rows; the noise that remains must not show up as new parameters."""
        window = self._noisy_drive_with_errors()
        names = self._listed(window)
        self.assertGreaterEqual(len(names), 4)
        while names:
            window.diagnosis._buttons[0].click()
            window.model.update()
            remaining = self._listed(window)
            self.assertLess(set(remaining), set(names))  # strictly fewer, nothing new
            names = remaining

    def test_correct_all_leaves_nothing_to_explain(self):
        window = self._noisy_drive_with_errors()
        window.diagnosis.correct_all_button.click()
        window.model.update()
        self.assertEqual(self._listed(window), [])
        self.assertLess(window.model.residuals.rms, 0.03)  # the noise of both trajectories (1 cm each)
        self.assertIn("Nothing left", window.diagnosis.note.text())
        self.assertFalse(window.diagnosis.correct_all_button.isEnabled())

    def test_correct_all_also_corrects_parameters_that_are_not_listed(self):
        """Correcting one parameter can shift the best value of a correlated one that was corrected before."""
        window = self._noisy_drive_with_errors()
        names = self._listed(window)
        window.diagnosis._buttons[names.index("Leverarm x (body)")].click()
        window.diagnosis._buttons[self._listed(window).index("Time shift")].click()
        window.diagnosis.correct_all_button.click()
        diagnosis = diagnostics.diagnose(
            window.session.preview, window.session.state, window.session.pivot, window.session.estimated_parameters
        )
        self.assertLess(abs(diagnosis.correction("lever_x").value), 1e-4)

    def test_correct_all_listed(self):
        window = self._drive_with_errors()
        window.diagnosis.correct_all_button.click()
        window.model.update()
        self.assertLess(window.model.residuals.rms, 0.01)
        self.assertEqual(window.diagnosis.row_names(), [])
        self.assertIn("Nothing left", window.diagnosis.note.text())

    def test_single_correction_from_the_table(self):
        window = self._drive_with_errors()
        names = self._listed(window)
        window.diagnosis._buttons[names.index("Time shift")].click()
        self.assertAlmostEqual(window.session.state.time_shift, 0.15, delta=1e-3)
        self.assertIn("RMS deviation", window.message_label.text())

    def test_views_can_show_the_actual_deviation_or_a_single_signature(self):
        window = self._drive_with_errors()
        view = window.views[diagnostics.VIEW_GEOMETRY]
        window.show_view(diagnostics.VIEW_GEOMETRY)
        self.app.processEvents()
        self.assertIn("rotated by +0.28", view.horizontal_plot.titleLabel.text)
        partial = view.residuals.tangential.copy()
        view.signature_check.setChecked(False)
        self.assertIs(view.residuals, window.model.residuals)
        self.assertGreater(np.max(np.abs(view.residuals.tangential - partial)), 0.1)

    def test_deviation_that_no_parameter_explains_is_named_as_such(self):
        """Drift of the trajectory is not an alignment error: the diagnosis says so instead of suggesting fixes."""
        reference = drive()
        rng = np.random.default_rng(3)
        drift = np.cumsum(rng.normal(scale=0.02, size=reference.xyz.shape), axis=0)
        drift -= np.linspace(0.0, 1.0, len(drift))[:, None] * drift[-1]  # a random walk that returns: no trend
        window = self._window(with_positions(reference, reference.xyz + drift), reference)
        window.model.update()
        self.assertEqual(self._listed(window), [])  # drift is not mistaken for alignment errors
        self.assertIn("noise or not modelled", window.diagnosis.note.text())

    # --------------------------------------------------------------------------------------------- layout

    def _geometry(self, window) -> dict:
        widgets = {
            "table": window.diagnosis.table,
            "correct all": window.diagnosis.correct_all_button,
            "refine": window.refine_button,
            "apply": window.apply_button,
            "pick": window.pick_button,
            "canvas": window.canvas,
            "tabs": window.tabs,
            "arrows": window.arrows_combo,
        }
        return {
            name: (widget.mapTo(window, QtCore.QPoint(0, 0)).toTuple(), widget.size().toTuple())
            for name, widget in widgets.items()
        }

    def _settle(self, window) -> None:
        for _ in range(3):
            self.app.processEvents()
        window.model.update()
        self.app.processEvents()

    def test_nothing_moves_when_the_placement_changes(self):
        window = self._drive_with_errors()
        window.resize(1500, 950)
        self._settle(window)
        before = self._geometry(window)
        session = window.session
        steps = (
            lambda: session.set_state(session.state.moved([0.2, 0.0, 0.0]), "Move"),
            lambda: window.views["time"].action_buttons[0].click(),
            lambda: window.diagnosis.correct_all_button.click(),
            lambda: session.add_pair(10, session.preview.reference_xyz[10]),
            session.clear_pairs,
            lambda: session._set_busy(True),
            lambda: session._set_busy(False),
            lambda: session._set_message("A long message. " * 40),
            session.undo,
        )
        for step in steps:
            step()
            self._settle(window)
            self.assertEqual(self._geometry(window), before)

    def test_the_axes_do_not_jump_while_dragging(self):
        window = self._drive_with_errors()
        session = window.session
        for index in range(4):
            window.tabs.setCurrentIndex(index)
            self._settle(window)
            view = window.tabs.currentWidget()
            ranges = [stable.view_box.viewRange() for stable in view._ranges]
            for _ in range(20):
                session.set_state(session.state.moved([0.01, 0.005, 0.0]), "Move", coalesce=True)
                window.model.update_residuals()
                self.app.processEvents()
            session.commit()
            self.assertEqual(
                [stable.view_box.viewRange() for stable in view._ranges], ranges, window.tabs.tabText(index)
            )

    def test_the_ranking_waits_until_the_input_pauses(self):
        window = self._drive_with_errors()
        names = window.diagnosis.row_names()
        diagnosis = window.model.diagnosis
        window.session.set_state(window.session.state.moved([0.5, 0.0, 0.0]), "Move", coalesce=True)
        window.model.update_residuals()  # what happens while dragging
        self.assertIs(window.model.diagnosis, diagnosis)
        self.assertEqual(window.diagnosis.row_names(), names)
        self.assertIn("1.", window.diagnosis.summary.text())  # the deviation itself follows live

    def test_unused_rows_have_no_buttons(self):
        window = self._drive_with_errors()
        panel = window.diagnosis
        listed = len(self._listed(window))
        self.assertTrue(all(not button.isHidden() for button in panel._buttons[:listed]))
        window.diagnosis.correct_all_button.click()
        self._settle(window)
        self.assertTrue(all(button.isHidden() for button in panel._buttons))
        # the buttons still work when rows come back
        window.session.undo()
        self._settle(window)
        self.assertEqual(len(self._listed(window)), listed)
        panel._buttons[0].click()
        self.assertTrue(window.session.undo_label.startswith("Correct"))

    def test_corrections_need_poses_at_equal_time(self):
        reference = drive()
        moved = reference.copy()
        moved.timestamps = moved.timestamps + 5000.0
        window = self._window(moved, reference)
        window.session.correct(["time_shift"], "Correct time shift")
        self.assertIn("No poses at equal time", window.message_label.text())
        self.assertEqual(window.session.state, AlignmentState())

    def test_session_is_independent_of_later_changes_to_the_trajectories(self):
        trajectory, reference = generated_trajectory.copy(), generated_trajectory.copy()
        session = AlignmentSession(trajectory, reference)
        original = session.trajectory.xyz.copy()
        trajectory.positions.xyz += 100.0  # e.g. a background operation on the trajectory in the table
        np.testing.assert_allclose(session.trajectory.xyz, original)

    def test_gui_applies_the_window_result(self):
        from trajectopy.gui.managers.ui_manager import UIManager
        from trajectopy.gui.models.selection import TrajectorySelection

        result = self._roughly_placed().session.alignment_result()
        manager = UIManager()
        requests = []
        manager.trajectory_manager_request.connect(requests.append)
        manager.handle_interactive_alignment(result, selection=TrajectorySelection())
        # requests are deep copies (they are processed on the worker thread)
        self.assertEqual(requests[0].manual_alignment, result)


if __name__ == "__main__":
    unittest.main()
