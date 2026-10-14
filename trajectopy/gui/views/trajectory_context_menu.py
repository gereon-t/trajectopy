import logging
from collections.abc import Callable

from PySide6 import QtCore

from trajectopy.gui.managers.requests import (
    PlotRequest,
    PlotRequestType,
    ResultModelRequest,
    ResultModelRequestType,
    TrajectoryManagerRequest,
    TrajectoryManagerRequestType,
    TrajectoryModelRequest,
    TrajectoryModelRequestType,
    UIRequest,
    UIRequestType,
)
from trajectopy.gui.models.selection import TrajectorySelection
from trajectopy.gui.views.menu_helpers import SelectionMenu, add_action, add_note, add_submenu

logger = logging.getLogger(__name__)

SINGLE_ONLY = "Select a single trajectory."
MULTIPLE_ONLY = "Select two or more trajectories."
NO_REFERENCE = "Set a reference first: right-click the reference trajectory → Set as Reference (R)."
ONLY_REFERENCE = "Select a trajectory other than the reference."


class TrajectoryContextMenu(SelectionMenu):
    """Context menu of the trajectory table.

    The entries are grouped by task. Entries that are not available for the current selection stay visible
    (disabled) and explain why on hover, so that all features can be discovered.
    """

    trajectory_manager_request = QtCore.Signal(TrajectoryManagerRequest)
    trajectory_model_request = QtCore.Signal(TrajectoryModelRequest)
    result_model_request = QtCore.Signal(ResultModelRequest)
    ui_request = QtCore.Signal(UIRequest)
    plot_request = QtCore.Signal(PlotRequest)
    show_timeline = QtCore.Signal(object)  # emits list[TrajectoryEntry]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._selection = TrajectorySelection(entries=[])
        self.alignments_available: Callable[[], bool] = lambda: True

    def get_selection(self) -> TrajectorySelection:
        return self._selection

    def _add_entries(self) -> None:
        self._view_section()
        self.addSeparator()
        self._reference_section()
        self.addSeparator()
        self._coordinates_menu()
        self._process_menu()
        self.addSeparator()
        self._entry_section()

    # ----------------------------------------------------------------------------------------------- availability
    @property
    def _single(self) -> str:
        return "" if len(self._selection) == 1 else SINGLE_ONLY

    @property
    def _multiple(self) -> str:
        return "" if len(self._selection) > 1 else MULTIPLE_ONLY

    @property
    def _needs_reference(self) -> str:
        if not self._selection.reference_is_set:
            return NO_REFERENCE
        if all(entry.set_as_reference for entry in self._selection.entries):
            return ONLY_REFERENCE
        return ""

    # ---------------------------------------------------------------------------------------------- request helpers
    def _manager(self, request_type: TrajectoryManagerRequestType) -> Callable[[], None]:
        return lambda: self.trajectory_manager_request.emit(
            TrajectoryManagerRequest(type=request_type, selection=self.get_selection())
        )

    def _model(self, request_type: TrajectoryModelRequestType) -> Callable[[], None]:
        return lambda: self.trajectory_model_request.emit(
            TrajectoryModelRequest(type=request_type, selection=self.get_selection())
        )

    def _ui(self, request_type: UIRequestType) -> Callable[[], None]:
        return lambda: self.ui_request.emit(UIRequest(type=request_type, trajectory_selection=self.get_selection()))

    # ------------------------------------------------------------------------------------------------------ sections
    def _view_section(self) -> None:
        add_action(
            self,
            "Plot",
            lambda: self.plot_request.emit(
                PlotRequest(type=PlotRequestType.TRAJECTORIES, trajectory_selection=self.get_selection())
            ),
            hint="Opens the trajectory plots (positions, orientations, velocity, map, …).",
            shortcuts=("V",),
        )
        add_action(
            self,
            "Playback in 3D",
            self._ui(UIRequestType.PLAYBACK),
            hint="Plays the trajectories back in a 3D view, showing the body frame axes at the current pose.",
        )
        add_action(
            self,
            "Timeline",
            lambda: self.show_timeline.emit(self.get_selection().entries),
            hint="Shows the time spans of the trajectories, e.g. to check that they overlap.",
        )

    def _reference_section(self) -> None:
        if len(self._selection) == 1 and self._selection.entries[0].set_as_reference:
            add_action(
                self,
                "Unset Reference",
                self._model(TrajectoryModelRequestType.UNSET_REFERENCE),
                hint="This trajectory is no longer used as the reference.",
                shortcuts=("Shift+R",),
            )
        else:
            add_action(
                self,
                "Set as Reference",
                self._model(TrajectoryModelRequestType.SET_REFERENCE),
                hint="Uses this trajectory as the reference (ground truth) for comparisons and alignments.",
                shortcuts=("R",),
                unavailable=self._single,
            )
        self._compare_menu()
        self._align_menu()

    def _compare_menu(self) -> None:
        menu = add_submenu(
            self,
            "Compare with Reference",
            hint="Computes deviations to the reference. The results appear in the result table.",
        )
        needs_reference = self._needs_reference
        if needs_reference:
            add_note(
                menu,
                "No reference set: right-click it → Set as Reference"
                if needs_reference == NO_REFERENCE
                else needs_reference,
            )
        add_action(
            menu,
            "Absolute Error (ATE), Aligned First",
            self._manager(TrajectoryManagerRequestType.ATE_WITH_ALIGNMENT),
            hint="Aligns the trajectory with the reference (alignment settings), then computes the position and "
            "orientation deviations of matching poses. Also adds the alignment to the results.",
            unavailable=needs_reference,
        )
        add_action(
            menu,
            "Absolute Error (ATE), as Is",
            self._manager(TrajectoryManagerRequestType.ATE_NO_ALIGNMENT),
            hint="Computes the position and orientation deviations of matching poses without aligning first.",
            unavailable=needs_reference,
        )
        add_action(
            menu,
            "Relative Pose Error (RPE)",
            self._manager(TrajectoryManagerRequestType.RPE),
            hint="Computes the drift: deviations of the relative motion over pose pairs separated by the "
            "distances or times defined in the settings (independent of an alignment).",
            unavailable=needs_reference,
        )
        menu.addSeparator()
        add_action(
            menu,
            "Extract Matching Poses",
            self._manager(TrajectoryManagerRequestType.MATCH),
            hint="Creates copies of the trajectory and the reference that only contain the poses that correspond "
            "to each other (matching settings).",
            unavailable=needs_reference,
        )

    def _align_menu(self) -> None:
        menu = add_submenu(
            self,
            "Align with Reference",
            hint="Transforms the trajectory into the frame of the reference. Creates an aligned copy.",
        )
        needs_reference = self._needs_reference
        if needs_reference:
            add_note(
                menu,
                "No reference set: right-click it → Set as Reference"
                if needs_reference == NO_REFERENCE
                else needs_reference,
            )
        add_action(
            menu,
            "Estimate Alignment",
            self._manager(TrajectoryManagerRequestType.ALIGN),
            hint="Estimates the alignment (least squares, alignment settings) and applies it. The estimated "
            "parameters are added to the results.",
            unavailable=needs_reference,
        )
        add_action(
            menu,
            "Interactive Alignment…",
            self._ui(UIRequestType.MANUAL_ALIGNMENT),
            hint="Opens a window to align the trajectory by hand and/or by least squares, with a live preview. "
            "Helpful if the automatic alignment fails, e.g. for unrelated frames or large time offsets.",
            unavailable=needs_reference or self._single,
        )
        add_action(
            menu,
            "Match First Pose",
            self._manager(TrajectoryManagerRequestType.ADAPT_FIRST_POSE),
            hint="Moves and rotates the trajectory so that its first pose coincides with the corresponding pose of "
            "the reference. Only poses that match the reference are kept; the matched reference is added, too.",
            unavailable=needs_reference,
        )
        menu.addSeparator()
        add_action(
            menu,
            "Apply Existing Alignment…",
            lambda: self.result_model_request.emit(
                ResultModelRequest(
                    type=ResultModelRequestType.PASS_ALIGNMENTS_TO_UI,
                    trajectory_selection=self.get_selection(),
                )
            ),
            hint="Applies an alignment from the result table, e.g. one estimated for another trajectory.",
            unavailable="" if self.alignments_available() else "There are no alignments in the result table.",
        )

    def _coordinates_menu(self) -> None:
        menu = add_submenu(self, "Coordinate System", hint="Coordinate reference system (EPSG code) of the positions.")
        add_action(
            menu,
            "Transform to EPSG…",
            self._ui(UIRequestType.EPSG_TRANSFORMATION),
            hint="Transforms the positions into another coordinate system, e.g. 25832 (UTM 32N).",
            shortcuts=("T",),
        )
        reference = self._selection.reference_entry
        if reference is None:
            unavailable = NO_REFERENCE
        elif reference.trajectory.positions.local_transformer is None:
            unavailable = "The coordinate system of the reference is unknown (EPSG 0)."
        else:
            unavailable = ""
        add_action(
            menu,
            "Adapt EPSG from Reference",
            self._manager(TrajectoryManagerRequestType.EPSG_TO_REF),
            hint="Transforms the positions into the coordinate system of the reference.",
            unavailable=unavailable,
        )
        add_action(
            menu,
            "Set EPSG Code (No Transformation)…",
            self._ui(UIRequestType.EPSG_EDIT),
            hint="Declares in which coordinate system the positions are, e.g. if it was missing in the file. "
            "The coordinates themselves are not changed.",
        )

    def _process_menu(self) -> None:
        menu = add_submenu(self, "Process", hint="Operations that create modified copies of the trajectories.")
        add_action(
            menu,
            "Smooth (Approximate)",
            self._manager(TrajectoryManagerRequestType.APPROXIMATE),
            hint="Approximates the trajectory by piecewise cubic polynomials (approximation settings).",
        )
        add_action(
            menu,
            "Resample to Time Grid…",
            self._ui(UIRequestType.GRID_SELECTION),
            hint="Interpolates the trajectory at equally spaced timestamps.",
        )
        add_action(
            menu,
            "Rearrange Axes / Angles…",
            self._ui(UIRequestType.DOF_ORGANIZER),
            hint="Swaps, negates or offsets the coordinate axes and Euler angles, e.g. to fix a different axis "
            "convention.",
        )
        menu.addSeparator()
        add_action(
            menu,
            "Sort Spatially",
            self._manager(TrajectoryManagerRequestType.SORT),
            hint="Orders the poses along the path (sorting settings), e.g. to compare repeated passes of the same "
            "track independent of time.",
        )
        sorting_known = all(entry.state.sorting_known for entry in self._selection.entries)
        add_action(
            menu,
            "Switch Order: Time ↔ Path",
            self._manager(TrajectoryManagerRequestType.SWITCH_SORTING),
            hint="Switches between chronological and spatial order of the poses.",
            unavailable="" if sorting_known else "Sort spatially first.",
        )
        add_action(
            menu,
            "Divide into Laps",
            self._manager(TrajectoryManagerRequestType.DIVIDE_INTO_LAPS),
            hint="Splits a trajectory that repeatedly passes the same closed track into one trajectory per lap.",
        )
        menu.addSeparator()
        add_action(
            menu,
            "Merge",
            self._manager(TrajectoryManagerRequestType.MERGE),
            hint="Combines the selected trajectories into one, ordered by time. They should be in the same "
            "coordinate system.",
            shortcuts=("M",),
            unavailable=self._multiple,
        )
        add_action(
            menu,
            "Average",
            self._manager(TrajectoryManagerRequestType.AVERAGE),
            hint="Averages the selected trajectories pose by pose. They must have the same number of poses, e.g. "
            "after resampling them to the same time grid.",
            unavailable=self._multiple,
        )

    def _entry_section(self) -> None:
        add_action(
            self,
            "Settings…",
            self._ui(UIRequestType.TRAJ_SETTINGS),
            hint="Settings used for processing this trajectory: alignment, matching, comparison, sorting, ….",
            shortcuts=("S",),
            unavailable=self._single,
        )
        add_action(
            self,
            "Rename",
            self._model(TrajectoryModelRequestType.RENAME),
            shortcuts=("F2", "U"),
            unavailable=self._single,
        )
        add_action(
            self,
            "Duplicate",
            self._model(TrajectoryModelRequestType.COPY),
            hint="Adds a copy of the selected trajectories to the table.",
            shortcuts=("Ctrl+D", "Ctrl+C"),
        )
        add_action(
            self,
            "Export…",
            self._ui(UIRequestType.EXPORT_TRAJ),
            hint="Saves the trajectory as a .traj file (readable by trajectopy and as plain text).",
            shortcuts=("E",),
            unavailable=self._single,
        )
        kml_unavailable = self._single
        if not kml_unavailable and self._selection.entries[0].trajectory.positions.epsg == 0:
            kml_unavailable = "The coordinate system is unknown (EPSG 0). Set or transform it under Coordinate System."
        add_action(
            self,
            "Export to Google Earth (KML)…",
            self._ui(UIRequestType.EXPORT_KML),
            hint="Saves the trajectory as a time-stamped track for Google Earth (playback with its time slider), "
            "in one color or colored by speed, height or time, on the terrain or at its heights.",
            unavailable=kml_unavailable,
        )
        self.addSeparator()
        add_action(
            self,
            "Remove",
            self._model(TrajectoryModelRequestType.REMOVE),
            shortcuts=("Del",),
        )
        add_action(
            self,
            "Remove Including Derived",
            self._model(TrajectoryModelRequestType.REMOVE_RELATED),
            hint="Removes the trajectories together with all trajectories derived from the same file (aligned, "
            "matched, sorted, … copies).",
        )
