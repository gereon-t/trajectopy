import unittest

import numpy as np

from trajectopy.core.trajectory import Trajectory
from trajectopy.readers.ascii import (
    HeaderData,
    extract_trajectory_rotations,
    read_string,
)


class TestReadersEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(23)

    def test_header_data_defaults(self):
        hd = HeaderData(data={})
        self.assertEqual(hd.name, "Trajectory")
        self.assertEqual(hd.epsg, 0)
        self.assertEqual(hd.rot_unit, "rad")
        self.assertEqual(hd.delimiter, ",")
        self.assertEqual(hd.nframe, "enu")

    def test_header_data_custom_values(self):
        hd = HeaderData(
            data={
                "name": "Test Trajectory",
                "epsg": 4326,
                "rot_unit": "deg",
                "delimiter": ";",
                "nframe": "ned",
            }
        )
        self.assertEqual(hd.name, "Test Trajectory")
        self.assertEqual(hd.epsg, 4326)
        self.assertEqual(hd.rot_unit, "deg")
        self.assertEqual(hd.delimiter, ";")
        self.assertEqual(hd.nframe, "ned")

    def test_read_string_with_header(self):
        content = "#name TestTraj\n#fields t,px,py,pz\n0,1,2,3\n1,4,5,6\n"
        header, data = read_string(content)
        self.assertEqual(header.name, "TestTraj")
        self.assertEqual(data.shape, (2, 4))

    def test_read_string_with_quaternions(self):
        content = "#fields t,px,py,pz,qx,qy,qz,qw\n0,1,2,3,0,0,0,1\n1,4,5,6,0,0,0,1\n"
        header, data = read_string(content)
        rotations = extract_trajectory_rotations(header, data)
        self.assertIsNotNone(rotations)
        self.assertEqual(len(rotations), 2)

    def test_read_string_with_euler_angles(self):
        content = "#fields t,px,py,pz,ex,ey,ez\n0,1,2,3,0,0,0\n1,4,5,6,0,0,0\n"
        header, data = read_string(content)
        rotations = extract_trajectory_rotations(header, data)
        self.assertIsNotNone(rotations)
        self.assertEqual(len(rotations), 2)

    def test_read_string_without_rotations(self):
        content = "#fields t,px,py,pz\n0,1,2,3\n1,4,5,6\n"
        header, data = read_string(content)
        rotations = extract_trajectory_rotations(header, data)
        self.assertIsNone(rotations)

    def test_trajectory_from_string_io_stream(self):
        content = "#name StreamTest\n#fields t,px,py,pz,qx,qy,qz,qw\n0,1,2,3,0,0,0,1\n1,4,5,6,0,0,0,1\n"
        traj = Trajectory.from_file(content, io_stream=True)
        self.assertEqual(traj.name, "StreamTest")
        self.assertEqual(len(traj), 2)
        self.assertIsNotNone(traj.rotations)

    def test_header_delimiter_detection_single_quote(self):
        content = "#delimiter ','\n#fields t,px,py,pz\n"
        header = HeaderData.from_string(content)
        self.assertEqual(header.delimiter, ",")

    def test_header_delimiter_detection_double_quote(self):
        content = '#delimiter ";"\n#fields t,px,py,pz\n'
        header = HeaderData.from_string(content)
        self.assertEqual(header.delimiter, ";")

    def test_header_time_format_defaults(self):
        hd = HeaderData(data={})
        from trajectopy.utils.definitions import TimeFormat

        self.assertEqual(hd.time_format, TimeFormat.UNIX)

    def test_header_fields_parsing(self):
        content = "#fields t,px,py,pz,qx,qy,qz,qw\n"
        header = HeaderData.from_string(content)
        expected = ["t", "px", "py", "pz", "qx", "qy", "qz", "qw"]
        self.assertEqual(header.fields, expected)


if __name__ == "__main__":
    unittest.main()
