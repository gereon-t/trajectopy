import unittest
from pathlib import Path
from test.testdata import generated_trajectory

from trajectopy_core.trajectory import Trajectory

from trajectopy.models.entries import AbsoluteDeviationEntry, AlignmentEntry, RelativeDeviationEntry


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
