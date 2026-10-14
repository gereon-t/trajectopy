from collections.abc import Callable

from PySide6.QtCore import Signal

from trajectopy.gui.managers.requests import (
    FileRequest,
    PlotRequest,
    PlotRequestType,
    ResultModelRequest,
    ResultModelRequestType,
    UIRequest,
    UIRequestType,
)
from trajectopy.gui.models.entries import (
    AbsoluteDeviationEntry,
    AlignmentEntry,
    RelativeDeviationEntry,
)
from trajectopy.gui.models.selection import ResultSelection
from trajectopy.gui.views.menu_helpers import SelectionMenu, add_action

SINGLE_ONLY = "Select a single result."


class ResultContextMenu(SelectionMenu):
    """Context menu of the result table. Entries that are not available for the current selection stay visible
    (disabled) and explain why on hover."""

    result_model_request = Signal(ResultModelRequest)
    ui_request = Signal(UIRequest)
    file_request = Signal(FileRequest)
    plot_request = Signal(PlotRequest)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._selection = ResultSelection(entries=[])

    def get_selection(self) -> ResultSelection:
        return self._selection

    def _add_entries(self) -> None:
        selection = self.get_selection()
        entries = selection.entries
        single = "" if len(entries) == 1 else SINGLE_ONLY

        add_action(
            self,
            "Plot",
            self._plot,
            hint="Plots the deviations (several deviation results are plotted together) or the alignment "
            "parameters with their uncertainties.",
            shortcuts=("V",),
            unavailable="" if self._plot_type is not None else "Select deviations, or a single alignment.",
        )
        if isinstance(alignment := entries[0], AlignmentEntry) and all(
            isinstance(entry, AlignmentEntry) for entry in entries
        ):
            add_action(
                self,
                "Choose Parameters to Apply…",
                lambda: self.result_model_request.emit(
                    ResultModelRequest(type=ResultModelRequestType.EDIT_ALIGNMENT, alignment=alignment)
                ),
                hint="Selects which of the estimated parameters are used when this alignment is applied, e.g. "
                "only the rotation.",
                unavailable=single,
            )
        self.addSeparator()
        add_action(
            self,
            "Rename",
            lambda: self.result_model_request.emit(
                ResultModelRequest(type=ResultModelRequestType.RENAME, selection=self.get_selection())
            ),
            shortcuts=("F2", "U"),
            unavailable=single,
        )
        add_action(
            self,
            "Duplicate",
            lambda: self.result_model_request.emit(
                ResultModelRequest(type=ResultModelRequestType.COPY, selection=self.get_selection())
            ),
            hint="Adds a copy of the selected results to the table.",
            shortcuts=("Ctrl+D", "Ctrl+C"),
        )
        add_action(
            self,
            "Export…",
            self._ui(UIRequestType.EXPORT_RES),
            hint="Saves the result as a .result file that can be loaded again.",
            shortcuts=("E",),
            unavailable=single,
        )
        if isinstance(first := entries[0], AbsoluteDeviationEntry) and all(
            isinstance(entry, AbsoluteDeviationEntry) for entry in entries
        ):
            kml_unavailable = single
            if not kml_unavailable and first.deviations.trajectory.positions.local_transformer is None:
                kml_unavailable = "The coordinate system of the evaluated trajectory is unknown (EPSG 0)."
            add_action(
                self,
                "Export to Google Earth (KML)…",
                self._ui(UIRequestType.EXPORT_KML),
                hint="Saves the evaluated trajectory for Google Earth, colored by a deviation of your choice "
                "(e.g. position, along-track or yaw), with playback in Google Earth's time slider.",
                unavailable=kml_unavailable,
            )
            add_action(
                self,
                "3D Playback in the Browser…",
                self._ui(UIRequestType.CESIUM_PLAYBACK),
                hint="Opens a 3D globe in the web browser (CesiumJS) that plays the evaluated trajectory, colored by a deviation of your choice, back with a timeline. Also saves it as a web page.",
                unavailable=kml_unavailable,
            )
        self.addSeparator()
        add_action(
            self,
            "Remove",
            lambda: self.result_model_request.emit(
                ResultModelRequest(type=ResultModelRequestType.REMOVE, selection=self.get_selection())
            ),
            shortcuts=("Del",),
        )

    @property
    def _plot_type(self) -> PlotRequestType | None:
        entries = self._selection.entries
        if entries and all(isinstance(entry, (AbsoluteDeviationEntry, RelativeDeviationEntry)) for entry in entries):
            return PlotRequestType.DEVIATIONS
        if len(entries) == 1 and isinstance(entries[0], AlignmentEntry):
            return PlotRequestType.ALIGNMENT
        return None

    def _plot(self) -> None:
        if (plot_type := self._plot_type) is not None:
            self.plot_request.emit(PlotRequest(type=plot_type, result_selection=self.get_selection()))

    def _ui(self, request_type: UIRequestType) -> Callable[[], None]:
        return lambda: self.ui_request.emit(UIRequest(type=request_type, result_selection=self.get_selection()))
