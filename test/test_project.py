"""Project files (.tpy) keep everything that the GUI shows, and session folders of earlier versions can be read."""

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from trajectopy.core.settings import PlotSettings
from trajectopy.gui.models.entries import (
    AbsoluteDeviationEntry,
    AlignmentEntry,
    RelativeDeviationEntry,
    TrajectoryEntry,
)
from trajectopy.gui.project import (
    MANIFEST_FILENAME,
    MANIFEST_KEY,
    Project,
    ProjectError,
    load_project,
    load_session_folder,
    save_project,
)
from trajectopy.utils.definitions import Sorting

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

    def test_damaged_projects_are_reported_as_damaged(self):
        save_project(_project(), self.path)
        data = bytearray(Path(self.path).read_bytes())
        start = data.index(b"manifest.npy") + 100  # inside the stored manifest
        data[start : start + 20] = b"\x00" * 20
        Path(self.path).write_bytes(bytes(data))

        with self.assertRaisesRegex(ProjectError, "damaged"):
            load_project(self.path)

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
        with np.load(self.path, allow_pickle=False) as arrays:
            manifest = json.loads(str(arrays[MANIFEST_KEY]))
            contents = {key: arrays[key] for key in arrays.files if key != MANIFEST_KEY}
        manifest.update(version=manifest["version"] + 1, trajectopy_version="99.0.0")
        with open(newer, "wb") as file:
            np.savez(file, **{MANIFEST_KEY: np.array(json.dumps(manifest))}, **contents)

        with self.assertRaisesRegex(ProjectError, r"newer version of Trajectopy \(99.0.0\)"):
            load_project(str(newer))

    def test_roundtrip_restores_the_data_exactly(self):
        project = _project()
        # spatially sorted with equal path lengths, which the trajectory would otherwise order by time
        trajectory = project.trajectories[0].trajectory
        trajectory.path_lengths = np.repeat(np.arange(len(trajectory) // 2 + 1), 2)[: len(trajectory)].astype(float)
        trajectory._sorting = Sorting.PATH_LENGTH
        alignment = project.results[0].alignment_result.position_parameters
        alignment.sim_trans_x.value = 1.25
        alignment.sim_trans_x.disable()  # keeps its value for when it is enabled again
        save_project(project, self.path)
        loaded = load_project(self.path)

        for original, restored in zip(project.trajectories, loaded.trajectories):
            for attribute in ("timestamps", "path_lengths", "xyz", "velocity_xyz"):
                np.testing.assert_array_equal(
                    getattr(restored.trajectory, attribute), getattr(original.trajectory, attribute)
                )
            # scipy normalizes quaternions when rotations are created, which may change the last digit
            np.testing.assert_allclose(restored.trajectory.quat, original.trajectory.quat, rtol=0, atol=1e-15)
            self.assertEqual(restored.trajectory.sorting, original.trajectory.sorting)
            self.assertEqual(restored.trajectory.positions.epsg, original.trajectory.positions.epsg)
            self.assertEqual(restored.trajectory.positions.pipeline_str, original.trajectory.positions.pipeline_str)
            self.assertEqual(str(restored.state), str(original.state))

        restored_alignment = loaded.results[0].alignment_result
        self.assertEqual(restored_alignment, project.results[0].alignment_result)
        self.assertFalse(restored_alignment.position_parameters.sim_trans_x.enabled)
        restored_alignment.position_parameters.sim_trans_x.enable()
        self.assertEqual(restored_alignment.position_parameters.sim_trans_x.value, 1.25)

        ate_original, ate_restored = project.results[1].deviations, loaded.results[1].deviations
        np.testing.assert_array_equal(ate_restored.abs_dev.pos_dev, ate_original.abs_dev.pos_dev)
        np.testing.assert_array_equal(ate_restored.trajectory.timestamps, ate_original.trajectory.timestamps)
        self.assertEqual(ate_restored.property_dict, ate_original.property_dict)
        self.assertEqual(loaded.results[2].deviations.property_dict, project.results[2].deviations.property_dict)

    def test_projects_of_the_text_format_can_still_be_opened(self):
        # the format of Trajectopy 5.1: a zip archive with a manifest and the trajectory and result files
        project = _project()
        with zipfile.ZipFile(self.path, "w") as archive:
            trajectories = []
            for index, entry in enumerate(project.trajectories):
                file = self.directory / f"{index:04d}.traj"
                entry.to_file(str(file))
                archive.write(file, f"trajectories/{index:04d}.traj")
                trajectories.append(
                    {
                        "id": entry.entry_id,
                        "file": f"trajectories/{index:04d}.traj",
                        "reference": entry.set_as_reference,
                        "group": entry.group_id,
                        "source": "",
                    }
                )
            file = self.directory / "0000.result"
            project.results[1].to_file(str(file))
            archive.write(file, "results/0000.result")
            manifest = {
                "format": "trajectopy-project",
                "version": 1,
                "trajectories": trajectories,
                "results": [{"id": project.results[1].entry_id, "file": "results/0000.result"}],
            }
            archive.writestr(MANIFEST_FILENAME, json.dumps(manifest))

        loaded = load_project(self.path)

        self.assertEqual(
            [entry.entry_id for entry in loaded.trajectories], [entry.entry_id for entry in project.trajectories]
        )
        self.assertEqual(loaded.results[0].entry_id, project.results[1].entry_id)
        np.testing.assert_allclose(loaded.trajectories[0].trajectory.xyz, project.trajectories[0].trajectory.xyz)

    def test_projects_are_read_without_unpickling(self):
        save_project(Project(), self.path)
        with np.load(self.path, allow_pickle=False) as arrays:
            manifest = json.loads(str(arrays[MANIFEST_KEY]))
        with open(self.path, "wb") as file:
            np.savez(file, **{MANIFEST_KEY: np.array(json.dumps(manifest))}, a0=np.array([object()], dtype=object))

        loaded = load_project(self.path)  # the object array is not referenced, so it is never read
        self.assertEqual(loaded.trajectories, [])

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
