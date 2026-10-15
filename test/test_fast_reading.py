"""The fast paths for reading large trajectory files give the same results as the general ones."""

import itertools
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from trajectopy.core.rotations import Rotations, _euler_to_quat
from trajectopy.io import ascii, header
from trajectopy.io.trajectory_io import _trajectory_from_data, read_trajectory


def _old_scan(filename: str) -> list[str]:
    with open(filename, encoding="utf-8") as file:
        return [line for line in file if line.startswith("#")]


class TestHeaderScan(unittest.TestCase):
    # name: (file content, whether "#" lines follow the first data line)
    CASES = {
        "plain": ("#a 1\n#b 2\n1,2\n3,4\n", False),
        "after_data": ("#a 1\n1,2\n#b 2\n3,4\n#c 3\n", True),
        "after_data_no_final_newline": ("#a 1\n1,2\n#b 2", True),
        "crlf": ("#a 1\r\n1,2\r\n#b 2\r\n3,4\r\n", True),
        "blank_lines": ("\n#a 1\n\n1,2\n\n#b\n", True),
        "only_header": ("#a 1\n#b 2\n", False),
        "empty": ("", False),
        "no_header": ("1,2\n3,4\n", False),
        "consecutive_after_data": ("1,2\n#x\n#y\n#z\n4,5\n", True),
        "hash_inside_line": ("1,2#no\n3,4\n#yes\n", True),
    }

    def test_finds_the_same_lines_as_reading_line_by_line(self):
        with tempfile.TemporaryDirectory() as directory:
            for chunk_size in (1, 2, 3, 7, 64, 1 << 24):  # every chunk boundary position
                for name, (text, lines_after_data) in self.CASES.items():
                    with self.subTest(case=name, chunk_size=chunk_size):
                        filename = os.path.join(directory, f"{name}.txt")
                        with open(filename, "w", newline="", encoding="utf-8") as file:
                            file.write(text)
                        with mock.patch.object(header, "_SCAN_CHUNK_SIZE", chunk_size):
                            scan = header.scan_header(filename)
                        self.assertEqual(scan.lines, _old_scan(filename))
                        self.assertEqual(scan.lines_after_data, lines_after_data)

    def test_leading_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "file.txt")
            with open(filename, "w", encoding="utf-8") as file:
                file.write("\n#a 1\n\n#b 2\n1,2\n#c\n")
            self.assertEqual(header.scan_header(filename).leading_lines, 4)


class TestEulerAngles(unittest.TestCase):
    def test_same_as_scipy_for_all_sequences(self):
        rng = np.random.default_rng(0)
        for length in (1, 2, 3):
            for axes in itertools.product("xyz", repeat=length):
                if any(a == b for a, b in zip(axes, axes[1:])):
                    continue
                for seq, degrees in itertools.product(("".join(axes), "".join(axes).upper()), (False, True)):
                    with self.subTest(seq=seq, degrees=degrees):
                        angles = (
                            rng.uniform(-400, 400, (500, length)) if degrees else rng.uniform(-7, 7, (500, length))
                        )
                        np.testing.assert_allclose(
                            Rotations.from_euler(seq, angles, degrees=degrees).as_quat(),
                            Rotation.from_euler(seq, angles, degrees=degrees).as_quat(),
                            atol=1e-14,
                        )

    def test_unusual_input_is_left_to_scipy(self):
        self.assertIsNone(_euler_to_quat("xyz", np.zeros(3), False))  # single rotation
        self.assertIsNone(_euler_to_quat("xYz", np.zeros((2, 3)), False))  # mixed intrinsic/extrinsic
        self.assertIsNone(_euler_to_quat("xxy", np.zeros((2, 3)), False))  # invalid for scipy
        self.assertEqual(len(Rotations.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()), 4)


class TestReadTrajectory(unittest.TestCase):
    def _assert_same_as_general_reader(self, text: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "trajectory.traj")
            with open(filename, "w", newline="", encoding="utf-8") as file:
                file.write(text)
            fast = read_trajectory(filename)
            general = _trajectory_from_data(*ascii.read_data(filename, dtype=object))
        # pyarrow parses exactly; pandas' default parser is off by one unit in the last place for some values
        np.testing.assert_allclose(fast.timestamps, general.timestamps, rtol=1e-15, atol=0)
        np.testing.assert_allclose(fast.positions.xyz, general.positions.xyz, rtol=1e-15, atol=1e-15)
        if general.has_orientation:
            np.testing.assert_allclose(fast.rotations.as_matrix(), general.rotations.as_matrix(), atol=1e-15)
        self.assertEqual((fast.name, fast.positions.epsg), (general.name, general.positions.epsg))

    def test_test_data(self):
        for filename in ("generated_trajectory", "noisy_trajectory", "open_loop_trajectory"):
            with self.subTest(filename=filename), open(f"test/data/{filename}.traj", encoding="utf-8") as file:
                self._assert_same_as_general_reader(file.read())

    def test_skipped_columns_and_euler_angles(self):
        header_lines = "#name skipped\n#fields t,-,ex,ey,ez,-,px,py,pz\n#nframe ned\n"
        rows = "".join(f"{i},text,0.1,0.2,{0.01 * i},x,{i},{2 * i},3\n" for i in range(20))
        self._assert_same_as_general_reader(header_lines + rows)

    def test_files_the_fast_path_cannot_read(self):
        cases = {
            "whitespace delimited": "#fields t,px,py,pz\n1 2 3 4\n5 6 7 8\n",
            "comment after data": "#fields t,px,py,pz\n1,2,3,4\n#note\n5,6,7,8\n",
            "datetime": (
                "#fields t,t,px,py,pz\n#time_format datetime\n"
                "2024-01-01,10:00:00.0,1,2,3\n2024-01-01,10:00:01.0,4,5,6\n"
            ),
        }
        for name, text in cases.items():
            with self.subTest(case=name):
                self._assert_same_as_general_reader(text)


class TestDatetimeTimeSystems(unittest.TestCase):
    TEMPLATE = (
        "#fields t,px,py,pz\n#time_format datetime\n{header}"
        "2024-01-01 12:00:00.000,0,0,0\n2024-01-01 12:00:00.100,1,0,0\n"
    )
    UTC_2024 = 1704110400.0  # 2024-01-01 12:00:00 UTC

    def _timestamps(self, header: str) -> np.ndarray:
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "trajectory.traj")
            with open(filename, "w", encoding="utf-8") as file:
                file.write(self.TEMPLATE.format(header=header))
            return read_trajectory(filename).timestamps

    def test_utc(self):
        for header_lines in ("", "#datetime_timezone UTC\n"):
            np.testing.assert_allclose(self._timestamps(header_lines), [self.UTC_2024, self.UTC_2024 + 0.1])

    def test_gps_time_subtracts_the_leap_seconds(self):
        np.testing.assert_allclose(
            self._timestamps("#datetime_timezone GPS\n"), [self.UTC_2024 - 18, self.UTC_2024 - 18 + 0.1]
        )

    def test_other_time_zones_are_read_as_utc_with_a_warning(self):
        with self.assertLogs(ascii.logger, level="WARNING") as logs:
            timestamps = self._timestamps("#datetime_timezone Europe/Berlin\n")
        np.testing.assert_allclose(timestamps, [self.UTC_2024, self.UTC_2024 + 0.1])
        self.assertTrue(any("Europe/Berlin" in message for message in logs.output))


class TestFastExtraHint(unittest.TestCase):
    FILE = "test/data/generated_trajectory.traj"

    def _hint_logged(self, *, pyarrow_installed: bool, frozen: bool) -> bool:
        patches = [
            mock.patch.object(ascii, "LARGE_FILE_BYTES", 0),
            mock.patch.object(ascii, "pacsv", object() if pyarrow_installed else None),
            mock.patch.object(ascii.sys, "frozen", frozen, create=True),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        with self.assertLogs(ascii.logger, level="INFO") as logs:
            ascii.logger.info("start")
            ascii._suggest_fast_extra(self.FILE)
        return any("trajectopy[fast]" in message for message in logs.output)

    def test_large_files_suggest_the_fast_extra(self):
        self.assertTrue(self._hint_logged(pyarrow_installed=False, frozen=False))

    def test_no_hint_if_installed_or_in_the_executable(self):
        self.assertFalse(self._hint_logged(pyarrow_installed=True, frozen=False))
        self.assertFalse(self._hint_logged(pyarrow_installed=False, frozen=True))


if __name__ == "__main__":
    unittest.main()
