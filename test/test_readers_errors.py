import unittest

import numpy as np

from trajectopy.readers import ascii as ascii_reader
from trajectopy.readers import header as header_reader
from trajectopy.readers.ascii import extract_trajectory_rotations


class TestReadersErrors(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def test_header_from_string_defaults(self):
        content = "#name Test\n#fields t,px,py,pz\n"
        hd = header_reader.HeaderData.from_string(content)
        self.assertEqual(hd.name, "Test")
        self.assertEqual(hd.fields, ["t", "px", "py", "pz"])

    def test_ascii_read_string_empty_data(self):
        # Header present but no data lines -> expect numpy/pandas to fail and return empty array
        content = "#fields t,px,py,pz\n"
        header, data = ascii_reader.read_string(content)
        self.assertIsInstance(header, ascii_reader.HeaderData)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.size, 0)

    def test_extract_rotations_missing_fields_returns_none(self):
        # No rotation fields -> rotations should be None
        content = "#fields t,px,py,pz\n1,0,0,0\n"
        header, data = ascii_reader.read_string(content)
        rot = extract_trajectory_rotations(header, data)
        self.assertIsNone(rot)


if __name__ == "__main__":
    unittest.main()
