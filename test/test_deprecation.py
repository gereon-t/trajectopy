import unittest
import warnings

from trajectopy.utils.deprecation import TrajectopyDeprecationWarning, renamed_parameters


@renamed_parameters(old="new")
def _function(value, *, new=None):
    return value, new


class TestRenamedParameters(unittest.TestCase):
    def test_new_name_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(_function(1, new=2), (1, 2))

    def test_old_name_warns_and_forwards(self):
        with self.assertWarns(TrajectopyDeprecationWarning):
            self.assertEqual(_function(1, old=2), (1, 2))

    def test_both_names_raise(self):
        with self.assertRaises(TypeError):
            _function(1, old=2, new=3)

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(ValueError):
            renamed_parameters(old="missing")(lambda: None)


if __name__ == "__main__":
    unittest.main()
