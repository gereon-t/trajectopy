"""Project files (.tpy) keep everything that the GUI shows, and session folders of earlier versions can be read."""

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from trajectopy.core.settings import PlotSettings
from trajectopy.gui.models.entries import (
    AbsoluteDeviationEntry,
    AlignmentEntry,
    RelativeDeviationEntry,
    TrajectoryEntry,
)
from trajectopy.gui.project import (
    MANIFEST_FILENAME,
    Project,
    ProjectError,
    load_project,
    load_session_folder,
    save_project,
)

TRAJECTORY_FILES = ["./test/data/generated_trajectory.traj", "./test/data/noisy_trajectory.traj"]


def _project() -> Project:
    reference, test = (TrajectoryEntry.from_trajectory_file(file) for file in TRAJECTORY_FILES)
    reference.set_as_reference = True
    test.name = "run_1 / with:odd*characters"  # not usable as a file name
    test.settings.alignment.preprocessing.min_speed = 1.5
    duplicate = TrajectoryEntry.from_trajectory_file(TRAJECTORY_FILES[0])  # same id as the reference
    duplicate.group_id = reference.group_id

    plot_settings = PlotSettings()
    plot_settings.ate_unit_is_mm = not plot_settings.ate_unit_is_mm
    return Project(
        trajectories=[test, reference, duplicate],
        # the test data was written by an earlier version (#type AbsoluteDeviationEntry, ...), so the type is given
        results=[
            AlignmentEntry.from_file("./test/data/align_entry.result"),
            AbsoluteDeviationEntry.from_file("./test/data/abs_dev.result"),
            RelativeDeviationEntry.from_file("./test/data/rel_dev_meter.result"),
        ],
        plot_settings=plot_settings,
    )


class TestProject(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = str(self.directory / "flight.tpy")

    def test_roundtrip_keeps_entries_order_and_state(self):
        project = _project()
        save_project(project, self.path)
        loaded = load_project(self.path)

        self.assertEqual(len(loaded.trajectories), len(project.trajectories))
        for original, restored in zip(project.trajectories, loaded.trajectories):
            self.assertEqual(restored.entry_id, original.entry_id)
            self.assertEqual(restored.name, original.name)
            self.assertEqual(restored.set_as_reference, original.set_as_reference)
            self.assertEqual(restored.group_id, original.group_id)
            self.assertEqual(restored.full_filename, original.full_filename)
            self.assertEqual(restored.settings.to_dict(), original.settings.to_dict())
            self.assertEqual(len(restored), len(original))

        self.assertEqual(
            [(result.entry_id, result.type, result.name) for result in loaded.results],
            [(result.entry_id, result.type, result.name) for result in project.results],
        )
        self.assertEqual(loaded.plot_settings, project.plot_settings)

    def test_empty_project(self):
        save_project(Project(), self.path)
        loaded = load_project(self.path)
        self.assertEqual((loaded.trajectories, loaded.results), ([], []))
        self.assertEqual(loaded.plot_settings, PlotSettings())

    def test_failed_save_keeps_the_existing_file(self):
        save_project(_project(), self.path)
        before = Path(self.path).read_bytes()

        broken = Project(results=[AlignmentEntry()])  # an alignment without parameters cannot be written
        broken.results[0].alignment_result.position_parameters = None
        with self.assertRaises(ValueError):
            save_project(broken, self.path)

        self.assertEqual(Path(self.path).read_bytes(), before)
        self.assertEqual(os.listdir(self.directory), ["flight.tpy"])  # no partial file left behind

    def test_other_files_are_rejected(self):
        text_file = self.directory / "notes.tpy"
        text_file.write_text("not a project", encoding="utf-8")
        foreign_zip = self.directory / "foreign.tpy"
        with zipfile.ZipFile(foreign_zip, "w") as archive:
            archive.writestr("readme.txt", "hello")

        for path in (text_file, foreign_zip):
            with self.subTest(path=path.name), self.assertRaisesRegex(ProjectError, "not a Trajectopy project"):
                load_project(str(path))

    def test_projects_of_newer_versions_are_rejected(self):
        save_project(Project(), self.path)
        newer = self.directory / "newer.tpy"
        with zipfile.ZipFile(self.path) as source, zipfile.ZipFile(newer, "w") as target:
            for item in source.infolist():
                data = source.read(item)
                if item.filename == MANIFEST_FILENAME:
                    manifest = json.loads(data)
                    manifest.update(version=manifest["version"] + 1, trajectopy_version="99.0.0")
                    data = json.dumps(manifest).encode()
                target.writestr(item, data)

        with self.assertRaisesRegex(ProjectError, r"newer version of Trajectopy \(99.0.0\)"):
            load_project(str(newer))

    def test_session_folder_of_earlier_versions(self):
        # the layout that "Session -> Save" wrote: <name>_<id>.traj/.json/.result and the order of the tables
        project = _project()
        trajectories, results = project.trajectories[:2], project.results
        trajectories[0].name = "run_1"  # an underscore in the name, too
        for entry in trajectories:
            entry.to_file(str(self.directory / f"{entry.name}_{entry.entry_id}.traj"))
            entry.settings.to_file(str(self.directory / f"{entry.name}_{entry.entry_id}.json"))
        for result in results:
            result.to_file(str(self.directory / f"{result.name}_{result.entry_id}.result"))
        (self.directory / "trajectory_order.txt").write_text(
            "\n".join(entry.entry_id for entry in trajectories), encoding="utf-8"
        )
        (self.directory / "result_order.txt").write_text(
            "\n".join(result.entry_id for result in results), encoding="utf-8"
        )
        project.plot_settings.to_file(str(self.directory / "plot_settings.json"))

        loaded = load_session_folder(str(self.directory))

        self.assertEqual([entry.entry_id for entry in loaded.trajectories], [e.entry_id for e in trajectories])
        self.assertEqual(
            loaded.trajectories[0].settings.to_dict(),
            trajectories[0].settings.to_dict(),
        )
        self.assertEqual([result.entry_id for result in loaded.results], [r.entry_id for r in results])
        self.assertEqual(loaded.plot_settings, project.plot_settings)

    def test_folder_without_session_is_rejected(self):
        with self.assertRaisesRegex(ProjectError, "not a session folder"):
            load_session_folder(str(self.directory))


if __name__ == "__main__":
    unittest.main()
