"""Unittests"""

import unittest

import numpy

from av.helper import HelperTypeHinting


class TestHelperTypeHinting(unittest.TestCase):
    """Test class for class HelperTypeHinting"""

    def test_check_list_of_ndarrays_positive(self):
        """Positive test case for check_list_of_ndarrays function"""
        variable = [numpy.array([1, 2, 3]), numpy.array([4, 5, 6])]
        self.assertTrue(HelperTypeHinting.check_list_of_ndarrays(variable))

    def test_check_list_of_ndarrays_negative(self):
        """Negative test case for check_list_of_ndarrays function"""
        variable = [numpy.array([1, 2, 3]), [4, 5, 6]]  # One element is not numpy.ndarray
        self.assertFalse(HelperTypeHinting.check_list_of_ndarrays(variable))

    def test_ensure_list_of_ndarrays_positive(self):
        """Positive test case for ensure_list_of_ndarrays function"""
        variable = [numpy.array([1, 2, 3]), numpy.array([4, 5, 6])]
        self.assertEqual(HelperTypeHinting.ensure_list_of_ndarrays(variable), variable)

    def test_ensure_list_of_ndarrays_negative(self):
        """Negative test case for ensure_list_of_ndarrays function"""
        variable = [numpy.array([1, 2, 3]), [4, 5, 6]]  # One element is not numpy.ndarray
        self.assertEqual(HelperTypeHinting.ensure_list_of_ndarrays(variable), variable)


if __name__ == "__main__":
    unittest.main()
