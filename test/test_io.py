import unittest
from pathlib import Path
from test.testdata import generated_trajectory

from trajectopy.core.trajectory import Trajectory
from trajectopy.gui.models.entries import (
    AbsoluteDeviationEntry,
    AlignmentEntry,
    RelativeDeviationEntry,
)


class TestIO(unittest.TestCase):
    _file = 0

    def setUp(self) -> None:
        super().setUp()
        Path("./test/tmp").mkdir(parents=True, exist_ok=True)

    def test_trajectory(self):
        input_trajectory = generated_trajectory.copy()
        generated_trajectory.to_file(f"./test/tmp/trajectory_{self._file}.csv")
        output_trajectory = Trajectory.from_file(f"./test/tmp/trajectory_{self._file}.csv")
        self.assertEqual(input_trajectory, output_trajectory)
        TestIO._file += 1

    def test_trajectory_write_read_with_rotations(self):
        """Test writing and reading trajectory with rotation data."""
        import numpy as np

        from trajectopy.core.positions import Positions
        from trajectopy.core.rotations import Rotations

        # Create trajectory with rotations
        xyz = np.random.rand(50, 3) * 100
        quat = np.tile([0, 0, 0, 1], (50, 1))
        timestamps = np.arange(50, dtype=float)

        traj = Trajectory(
            name="TestTraj",
            positions=Positions(xyz=xyz, epsg=4326),
            rotations=Rotations.from_quat(quat),
            timestamps=timestamps,
        )

        filepath = f"./test/tmp/trajectory_{self._file}.csv"
        traj.to_file(filepath)

        # Read back
        traj_read = Trajectory.from_file(filepath)

        # Verify data is preserved
        self.assertEqual(traj.name, traj_read.name)
        self.assertEqual(len(traj), len(traj_read))
        self.assertTrue(traj_read.has_orientation)
        np.testing.assert_array_almost_equal(traj.positions.xyz, traj_read.positions.xyz)
        np.testing.assert_array_almost_equal(traj.rotations.as_quat(), traj_read.rotations.as_quat())
        np.testing.assert_array_almost_equal(traj.timestamps, traj_read.timestamps)

        TestIO._file += 1

    def test_trajectory_write_read_without_rotations(self):
        """Test writing and reading trajectory without rotations."""
        import numpy as np

        from trajectopy.core.positions import Positions

        # Create trajectory without rotations
        xyz = np.random.rand(30, 3) * 50
        timestamps = np.arange(30, dtype=float)

        traj = Trajectory(name="NoRotTraj", positions=Positions(xyz=xyz, epsg=0), timestamps=timestamps)

        filepath = f"./test/tmp/trajectory_{self._file}.csv"
        traj.to_file(filepath)

        # Read back
        traj_read = Trajectory.from_file(filepath)

        # Verify data
        self.assertEqual(len(traj), len(traj_read))
        np.testing.assert_array_almost_equal(traj.positions.xyz, traj_read.positions.xyz)
        np.testing.assert_array_almost_equal(traj.timestamps, traj_read.timestamps)

        TestIO._file += 1

    def test_trajectory_write_read_different_epsg(self):
        """Test writing and reading trajectory with different EPSG codes."""
        import numpy as np

        from trajectopy.core.positions import Positions

        for epsg in [4326, 32632, 0]:
            xyz = np.random.rand(20, 3) * 100
            timestamps = np.arange(20, dtype=float)

            traj = Trajectory(positions=Positions(xyz=xyz, epsg=epsg), timestamps=timestamps)

            filepath = f"./test/tmp/trajectory_{self._file}.csv"
            traj.to_file(filepath)

            traj_read = Trajectory.from_file(filepath)

            # EPSG should be preserved
            self.assertEqual(traj.positions.epsg, traj_read.positions.epsg)

            TestIO._file += 1

    def test_absolute_deviations(self):
        self.deviation_io_test(AbsoluteDeviationEntry, "./test/data/abs_dev.result", "./test/tmp/abs_dev_")

    def test_relative_deviations_metric(self):
        self.deviation_io_test(RelativeDeviationEntry, "./test/data/rel_dev_meter.result", "./test/tmp/rel_dev_")

    def test_relative_deviations_second(self):
        self.deviation_io_test(RelativeDeviationEntry, "./test/data/rel_dev_second.result", "./test/tmp/rel_dev_")

    def test_alignment(self):
        input_align = AlignmentEntry.from_file("./test/data/align_entry.result")
        input_align.to_file(f"./test/tmp/align_{self._file}.result")
        reimport = AlignmentEntry.from_file(f"./test/tmp/align_{self._file}.result")
        TestIO._file += 1
        self.assertEqual(input_align.alignment_result, reimport.alignment_result)

    def deviation_io_test(self, cls, input_file: str, output_file: str):
        input_deviations = cls.from_file(input_file)
        input_deviations.to_file(f"{output_file}{self._file}.result")
        reimport_deviations = cls.from_file(f"{output_file}{self._file}.result")
        TestIO._file += 1
        self.assertEqual(input_deviations.deviations, reimport_deviations.deviations)
