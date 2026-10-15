"""Trajectopy projects: all trajectories, results and plot settings of a GUI session in one file.

A project file (``.tpy``) is a zip archive::

    project.json                         manifest: format version, order and state of the entries
    plot_settings.json
    trajectories/0000.traj               trajectory (the same file that *Export…* writes)
    trajectories/0000.json               its processing settings
    trajectories/0000_local_transformer.txt  (if it has a local coordinate system)
    results/0000.result

The trajectory and result files can be extracted and used on their own.
"""

import json
import logging
import os
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from trajectopy.__version__ import __version__ as VERSION
from trajectopy.core.settings import PlotSettings
from trajectopy.gui.models.entries import ResultEntry, TrajectoryEntry

logger = logging.getLogger(__name__)

PROJECT_SUFFIX = ".tpy"
PROJECT_FILE_FILTER = f"Trajectopy Project (*{PROJECT_SUFFIX})"

FORMAT_NAME = "trajectopy-project"
FORMAT_VERSION = 1
MANIFEST_FILENAME = "project.json"
PLOT_SETTINGS_FILENAME = "plot_settings.json"

# order files of session folders (written by Trajectopy before projects were introduced)
TRAJECTORY_ORDER_FILENAME = "trajectory_order.txt"
RESULT_ORDER_FILENAME = "result_order.txt"

EntryT = TypeVar("EntryT", TrajectoryEntry, ResultEntry)


class ProjectError(Exception):
    """The file is not a (readable) Trajectopy project."""


@dataclass
class Project:
    """Contents of a project, in the order of the tables."""

    trajectories: list[TrajectoryEntry] = field(default_factory=list)
    results: list[ResultEntry] = field(default_factory=list)
    plot_settings: PlotSettings = field(default_factory=PlotSettings)


def save_project(project: Project, path: str) -> None:
    """Writes ``project`` to ``path``. An existing file is replaced only once the new one is complete."""
    target = Path(path)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_contents(project, root)

        file_descriptor, partial = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
        os.close(file_descriptor)
        try:
            with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(root.rglob("*")):
                    if file.is_file():
                        archive.write(file, file.relative_to(root).as_posix())
            os.replace(partial, target)
        except BaseException:
            Path(partial).unlink(missing_ok=True)
            raise
    logger.info("Saved project %s", target)


def _write_contents(project: Project, root: Path) -> None:
    # files are numbered instead of named after the entries: names may contain any character and ids may repeat
    (root / "trajectories").mkdir()
    (root / "results").mkdir()
    trajectories, results = [], []
    for index, entry in enumerate(project.trajectories):
        file = root / "trajectories" / f"{index:04d}.traj"
        entry.to_file(str(file))
        entry.settings.to_file(str(file.with_suffix(".json")))
        if entry.trajectory.positions.local_transformer is not None:
            entry.trajectory.positions.export_local_transformer(
                str(file.with_name(f"{file.stem}_local_transformer.txt"))
            )
        trajectories.append(
            {
                "id": entry.entry_id,
                "file": file.relative_to(root).as_posix(),
                "reference": entry.set_as_reference,
                "group": entry.group_id,
                "source": entry.full_filename,
            }
        )

    for index, result in enumerate(project.results):
        file = root / "results" / f"{index:04d}.result"
        result.to_file(str(file))
        results.append({"id": result.entry_id, "file": file.relative_to(root).as_posix()})

    project.plot_settings.to_file(str(root / PLOT_SETTINGS_FILENAME))
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "trajectopy_version": VERSION,
        "trajectories": trajectories,
        "results": results,
    }
    (root / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_project(path: str) -> Project:
    """Reads a project file.

    Raises:
        ProjectError: If the file is not a Trajectopy project or was written by a newer version.
    """
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, IsADirectoryError) as error:
        raise ProjectError(f"{path} is not a Trajectopy project.") from error

    with archive, tempfile.TemporaryDirectory() as directory:
        manifest = _read_manifest(archive, path)
        archive.extractall(directory)
        root = Path(directory)

        trajectories = []
        for item in manifest.get("trajectories", []):
            entry = TrajectoryEntry.from_trajectory_file(str(_member(root, item["file"])))
            entry.set_id(item["id"])
            entry.set_as_reference = bool(item.get("reference", False))
            entry.group_id = item.get("group") or entry.group_id
            entry.full_filename = item.get("source", "")
            trajectories.append(entry)

        results = []
        for item in manifest.get("results", []):
            result = ResultEntry.from_file(str(_member(root, item["file"])))
            result.set_id(item["id"])
            results.append(result)

        project = Project(trajectories, results, _read_plot_settings(root / PLOT_SETTINGS_FILENAME))
    logger.info("Opened project %s", path)
    return project


def _read_manifest(archive: zipfile.ZipFile, path: str) -> dict:
    try:
        manifest = json.loads(archive.read(MANIFEST_FILENAME))
    except (KeyError, ValueError) as error:
        raise ProjectError(f"{path} is not a Trajectopy project.") from error

    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT_NAME:
        raise ProjectError(f"{path} is not a Trajectopy project.")
    if manifest.get("version", 0) > FORMAT_VERSION:
        raise ProjectError(
            f"{path} was saved by a newer version of Trajectopy ({manifest.get('trajectopy_version', 'unknown')}). "
            "Update Trajectopy to open it."
        )
    return manifest


def _member(root: Path, name: str) -> Path:
    """Path of an extracted file named in the manifest; never outside of the archive."""
    file = (root / name).resolve()
    if not file.is_relative_to(root.resolve()) or not file.is_file():
        raise ProjectError(f"The project is damaged: {name} is missing.")
    return file


def _read_plot_settings(path: Path) -> PlotSettings:
    return PlotSettings.from_file(str(path)) if path.is_file() else PlotSettings()


def load_session_folder(directory: str) -> Project:
    """Reads a session folder, the format in which Trajectopy saved sessions before projects were introduced.

    Raises:
        ProjectError: If the folder contains neither trajectories nor results.
    """
    root = Path(directory)
    trajectories = [
        _with_session_id(TrajectoryEntry.from_trajectory_file(str(file)), file) for file in sorted(root.glob("*.traj"))
    ]
    results = [_with_session_id(ResultEntry.from_file(str(file)), file) for file in sorted(root.glob("*.result"))]
    if not trajectories and not results:
        raise ProjectError(f"{directory} is not a session folder: it contains no .traj or .result files.")

    return Project(
        trajectories=_ordered(trajectories, root / TRAJECTORY_ORDER_FILENAME),
        results=_ordered(results, root / RESULT_ORDER_FILENAME),
        plot_settings=_read_plot_settings(root / PLOT_SETTINGS_FILENAME),
    )


def _with_session_id(entry: EntryT, file: Path) -> EntryT:
    """Session folders name the files ``<name>_<id>``; trajectory files lack the id in their header, and the name
    may contain underscores, too."""
    _, _, suffix = file.stem.rpartition("_")
    try:
        entry.set_id(str(uuid.UUID(suffix)))
    except ValueError:
        pass  # not written by Trajectopy
    return entry


def _ordered(entries: list[EntryT], order_file: Path) -> list[EntryT]:
    """Sorts the entries by the ids in ``order_file``; entries that it does not list stay at the end."""
    if not order_file.is_file():
        return entries
    order = {entry_id: index for index, entry_id in enumerate(order_file.read_text(encoding="utf-8").split())}
    return sorted(entries, key=lambda entry: order.get(entry.entry_id, len(order)))
