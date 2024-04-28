"""Unittests"""

import unittest

import numpy

from av.helper import HelperTypeHinting


class TestHelperTypeHinting(unittest.TestCase):
    """Test class"""

    def test_check_list_of_ndarrays_valid(self):
        """A test function"""
        # Valid case: list of numpy arrays
        variable = [numpy.array([1, 2, 3]), numpy.array([4, 5, 6])]
        self.assertTrue(HelperTypeHinting.check_list_of_ndarrays(variable))

    def test_check_list_of_ndarrays_invalid(self):
        """A test function"""
        # Invalid case: not a list
        variable = numpy.array([1, 2, 3])
        self.assertFalse(HelperTypeHinting.check_list_of_ndarrays(variable))

        # Invalid case: list containing non-numpy array
        variable = [numpy.array([1, 2, 3]), "not an array"]
        self.assertFalse(HelperTypeHinting.check_list_of_ndarrays(variable))

    def test_ensure_list_of_ndarrays_valid(self):
        """A test function"""
        # Valid case: list of numpy arrays
        variable = [numpy.array([1, 2, 3]), numpy.array([4, 5, 6])]
        ensured_variable = HelperTypeHinting.ensure_list_of_ndarrays(variable)
        for x, y in zip(variable, ensured_variable):
            self.assertTrue(numpy.array_equal(x, y))

    # def test_ensure_list_of_ndarrays_invalid(self):
    #     # Invalid case: not a list
    #     variable = numpy.array([1, 2, 3])
    #     ensured_variable = HelperTypeHinting.ensure_list_of_ndarrays(variable)
    #     self.assertEqual(variable, ensured_variable)  # Should return the original variable

    #     # Invalid case: list containing non-numpy array
    #     variable = [numpy.array([1, 2, 3]), "not an array"]
    #     ensured_variable = HelperTypeHinting.ensure_list_of_ndarrays(variable)
    #     for x, y in zip(variable, ensured_variable):
    #         if isinstance(x, numpy.ndarray):
    #             self.assertTrue(numpy.array_equal(x, y))
    #         else:
    #             self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
