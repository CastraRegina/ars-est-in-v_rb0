"""Pytest tests"""

import numpy as np
import pytest

from av.helper import HelperTypeHinting


def test_check_list_of_ndarrays_positive():
    """Positive test case for check_list_of_ndarrays function"""
    variable = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    assert HelperTypeHinting.check_list_of_ndarrays(variable)


def test_check_list_of_ndarrays_negative():
    """Negative test case for check_list_of_ndarrays function"""
    variable = [np.array([1, 2, 3]), [4, 5, 6]]  # One element is not numpy.ndarray
    assert not HelperTypeHinting.check_list_of_ndarrays(variable)


def test_ensure_list_of_ndarrays_positive():
    """Positive test case for ensure_list_of_ndarrays function"""
    variable = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    assert HelperTypeHinting.ensure_list_of_ndarrays(variable) == variable


def test_ensure_list_of_ndarrays_negative():
    """Negative test case for ensure_list_of_ndarrays function"""
    variable = [np.array([1, 2, 3]), [4, 5, 6]]  # One element is not numpy.ndarray
    assert HelperTypeHinting.ensure_list_of_ndarrays(variable) == variable
