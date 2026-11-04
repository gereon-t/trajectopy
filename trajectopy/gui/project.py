"""Trajectopy projects: all trajectories, results and plot settings of a GUI session in one file.

A project file (``.tpy``) is an uncompressed NumPy archive (see :mod:`trajectopy.gui.project_format`): a JSON
manifest with the metadata of all entries and their numerical data as binary arrays, which makes saving and opening
fast. Projects of the earlier text based format (a zip archive of trajectory and result files) can still be opened,
and session folders of even older versions can be imported.
"""

import json
import logging
import os
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from trajectopy.__version__ import __version__ as VERSION
from trajectopy.core.settings import PlotSettings
from trajectopy.gui import project_format
from trajectopy.gui.models.entries import ResultEntry, TrajectoryEntry

logger = logging.getLogger(__name__)

PROJECT_SUFFIX = ".tpy"
PROJECT_FILE_FILTER = f"Trajectopy Project (*{PROJECT_SUFFIX})"

FORMAT_NAME = "trajectopy-project"
FORMAT_VERSION = 2
MANIFEST_KEY = "manifest"  # array of the JSON manifest in a binary project (version 2)
MANIFEST_FILENAME = "project.json"  # manifest of a text based project (version 1)
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
    encoder = project_format.Encoder()
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "trajectopy_version": VERSION,
        "trajectories": [project_format.encode_trajectory_entry(entry, encoder) for entry in project.trajectories],
        "results": [project_format.encode_result_entry(result, encoder) for result in project.results],
        "plot_settings": project_format.encode_plot_settings(project.plot_settings),
    }

    file_descriptor, partial = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            # uncompressed: the arrays are written as they are, which is much faster than compressing them
            contents: dict[str, Any] = {MANIFEST_KEY: np.array(json.dumps(manifest)), **encoder.arrays}
            np.savez(file, allow_pickle=False, **contents)
        os.replace(partial, target)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise
    logger.info("Saved project %s", target)


def load_project(path: str) -> Project:
    """Reads a project file (of this or of the earlier, text based format).

    Raises:
        ProjectError: If the file is not a Trajectopy project or was written by a newer version.
    """
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, IsADirectoryError, FileNotFoundError) as error:
        raise ProjectError(f"{path} is not a Trajectopy project.") from error

    with archive:
        names = set(archive.namelist())
        if MANIFEST_FILENAME in names:
            project = _load_text_project(archive, path)
        elif f"{MANIFEST_KEY}.npy" in names:
            project = None
        else:
            raise ProjectError(f"{path} is not a Trajectopy project.")
    if project is None:
        project = _load_binary_project(path)
    logger.info("Opened project %s", path)
    return project


def _load_binary_project(path: str) -> Project:
    # without pickle: a project file can never execute code
    with np.load(path, allow_pickle=False) as arrays:
        try:
            manifest = json.loads(str(arrays[MANIFEST_KEY]))
        except ValueError as error:
            raise ProjectError(f"{path} is not a Trajectopy project.") from error
        _check_manifest(manifest, path)
        decoder = project_format.Decoder(arrays)
        try:
            return Project(
                trajectories=[
                    project_format.decode_trajectory_entry(item, decoder) for item in manifest["trajectories"]
                ],
                results=[project_format.decode_result_entry(item, decoder) for item in manifest["results"]],
                plot_settings=project_format.decode_plot_settings(manifest["plot_settings"]),
            )
        except (KeyError, TypeError) as error:
            raise ProjectError(f"The project {path} is damaged ({error}).") from error


def _load_text_project(archive: zipfile.ZipFile, path: str) -> Project:
    """Reads a project of format version 1 (text files in a zip archive, Trajectopy 5.1)."""
    with tempfile.TemporaryDirectory() as directory:
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

        return Project(trajectories, results, _read_plot_settings(root / PLOT_SETTINGS_FILENAME))


def _read_manifest(archive: zipfile.ZipFile, path: str) -> dict:
    try:
        manifest = json.loads(archive.read(MANIFEST_FILENAME))
    except (KeyError, ValueError) as error:
        raise ProjectError(f"{path} is not a Trajectopy project.") from error
    _check_manifest(manifest, path)
    return manifest


def _check_manifest(manifest: object, path: str) -> None:
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT_NAME:
        raise ProjectError(f"{path} is not a Trajectopy project.")
    if manifest.get("version", 0) > FORMAT_VERSION:
        raise ProjectError(
            f"{path} was saved by a newer version of Trajectopy ({manifest.get('trajectopy_version', 'unknown')}). "
            "Update Trajectopy to open it."
        )


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
