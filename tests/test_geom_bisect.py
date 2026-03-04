"""Tests for GeomMath.bisect method."""

from __future__ import annotations

import math

import pytest

from ave.geom import GeomMath


class TestGeomMathBisect:
    """Test suite for GeomMath.bisect static method."""

    def test_basic_transition_ascending(self) -> None:
        """Test basic case with ascending limits and simple transition."""

        def test_fn(x: float) -> bool:
            return x < 5.0

        result = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.01)

        assert abs(result - 5.0) <= 0.01 * 10.0 / 2.0

    def test_basic_transition_descending(self) -> None:
        """Test basic case with descending limits (limit1 > limit2)."""

        def test_fn(x: float) -> bool:
            return x < 5.0

        result = GeomMath.bisect(10.0, 0.0, test_fn, tolerance=0.01)

        assert abs(result - 5.0) <= 0.01 * 10.0 / 2.0

    def test_transition_at_zero(self) -> None:
        """Test transition point at zero."""

        def test_fn(x: float) -> bool:
            return x < 0.0

        result = GeomMath.bisect(-10.0, 10.0, test_fn, tolerance=0.001)

        assert abs(result - 0.0) <= 0.001 * 20.0 / 2.0

    def test_negative_interval(self) -> None:
        """Test with entirely negative interval."""

        def test_fn(x: float) -> bool:
            return x < -5.0

        result = GeomMath.bisect(-10.0, 0.0, test_fn, tolerance=0.01)

        assert abs(result - (-5.0)) <= 0.01 * 10.0 / 2.0

    def test_positive_interval(self) -> None:
        """Test with entirely positive interval."""

        def test_fn(x: float) -> bool:
            return x < 50.0

        result = GeomMath.bisect(0.0, 100.0, test_fn, tolerance=0.005)

        assert abs(result - 50.0) <= 0.005 * 100.0 / 2.0

    def test_very_small_tolerance(self) -> None:
        """Test with very small tolerance for high precision."""

        def test_fn(x: float) -> bool:
            return x < math.pi

        result = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.0001)

        assert abs(result - math.pi) <= 0.0001 * 10.0 / 2.0

    def test_large_tolerance(self) -> None:
        """Test with large tolerance for quick convergence."""

        def test_fn(x: float) -> bool:
            return x < 5.0

        result = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.5)

        assert abs(result - 5.0) <= 0.5 * 10.0 / 2.0

    def test_no_transition_all_true(self) -> None:
        """Test when test_fn is always True (no transition)."""

        def test_fn(_x: float) -> bool:
            return True

        result = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.01)

        assert result == 5.0

    def test_no_transition_all_false(self) -> None:
        """Test when test_fn is always False (no transition)."""

        def test_fn(_x: float) -> bool:
            return False

        result = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.01)

        assert result == 5.0

    def test_transition_false_to_true(self) -> None:
        """Test transition from False to True."""

        def test_fn(x: float) -> bool:
            return x > 7.5

        result = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.01)

        assert abs(result - 7.5) <= 0.01 * 10.0 / 2.0

    def test_transition_true_to_false(self) -> None:
        """Test transition from True to False."""

        def test_fn(x: float) -> bool:
            return x < 3.3

        result = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.01)

        assert abs(result - 3.3) <= 0.01 * 10.0 / 2.0

    def test_very_large_interval(self) -> None:
        """Test with very large interval."""

        def test_fn(x: float) -> bool:
            return x < 500.0

        result = GeomMath.bisect(-1000.0, 1000.0, test_fn, tolerance=0.001)

        assert abs(result - 500.0) <= 0.001 * 2000.0 / 2.0

    def test_very_small_interval(self) -> None:
        """Test with very small interval."""

        def test_fn(x: float) -> bool:
            return x < 0.5

        result = GeomMath.bisect(0.0, 1.0, test_fn, tolerance=0.1)

        assert abs(result - 0.5) <= 0.1 * 1.0 / 2.0

    def test_descending_limits_negative(self) -> None:
        """Test descending limits with negative values."""

        def test_fn(x: float) -> bool:
            return x < -3.0

        result = GeomMath.bisect(0.0, -10.0, test_fn, tolerance=0.01)

        assert abs(result - (-3.0)) <= 0.01 * 10.0 / 2.0

    def test_mathematical_function_sqrt(self) -> None:
        """Test with mathematical function (sqrt(2))."""

        def test_fn(x: float) -> bool:
            return x * x < 2.0

        result = GeomMath.bisect(0.0, 2.0, test_fn, tolerance=0.0001)

        sqrt_2 = math.sqrt(2.0)
        assert abs(result - sqrt_2) <= 0.0001 * 2.0 / 2.0

    def test_mathematical_function_exp(self) -> None:
        """Test with exponential function."""

        def test_fn(x: float) -> bool:
            return math.exp(x) < 10.0

        result = GeomMath.bisect(0.0, 5.0, test_fn, tolerance=0.001)

        ln_10 = math.log(10.0)
        assert abs(result - ln_10) <= 0.001 * 5.0 / 2.0

    def test_equal_limits_raises_error(self) -> None:
        """Test that equal limits raise ValueError."""

        def test_fn(x: float) -> bool:
            return x < 5.0

        with pytest.raises(ValueError, match="limit1 and limit2 must be different"):
            GeomMath.bisect(5.0, 5.0, test_fn, tolerance=0.01)

    def test_tolerance_zero_raises_error(self) -> None:
        """Test that tolerance of 0 raises ValueError."""

        def test_fn(x: float) -> bool:
            return x < 5.0

        with pytest.raises(ValueError, match="tolerance must be in range"):
            GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.0)

    def test_tolerance_one_raises_error(self) -> None:
        """Test that tolerance of 1.0 raises ValueError."""

        def test_fn(x: float) -> bool:
            return x < 5.0

        with pytest.raises(ValueError, match="tolerance must be in range"):
            GeomMath.bisect(0.0, 10.0, test_fn, tolerance=1.0)

    def test_tolerance_negative_raises_error(self) -> None:
        """Test that negative tolerance raises ValueError."""

        def test_fn(x: float) -> bool:
            return x < 5.0

        with pytest.raises(ValueError, match="tolerance must be in range"):
            GeomMath.bisect(0.0, 10.0, test_fn, tolerance=-0.01)

    def test_tolerance_greater_than_one_raises_error(self) -> None:
        """Test that tolerance > 1.0 raises ValueError."""

        def test_fn(x: float) -> bool:
            return x < 5.0

        with pytest.raises(ValueError, match="tolerance must be in range"):
            GeomMath.bisect(0.0, 10.0, test_fn, tolerance=1.5)

    def test_multiple_calls_same_function(self) -> None:
        """Test multiple calls with same function produce consistent results."""

        def test_fn(x: float) -> bool:
            return x < 5.0

        result1 = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.01)
        result2 = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.01)

        assert result1 == result2

    def test_stateful_function_call_count(self) -> None:
        """Test that function is called efficiently (not too many times)."""
        call_count = [0]

        def test_fn(x: float) -> bool:
            call_count[0] += 1
            return x < 5.0

        GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.01)

        # With 64 max iterations, should not exceed this significantly
        assert call_count[0] <= 70

    def test_transition_near_lower_bound(self) -> None:
        """Test transition very close to lower bound."""

        def test_fn(x: float) -> bool:
            return x < 0.1

        result = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.001)

        assert abs(result - 0.1) <= 0.001 * 10.0 / 2.0

    def test_transition_near_upper_bound(self) -> None:
        """Test transition very close to upper bound."""

        def test_fn(x: float) -> bool:
            return x < 9.9

        result = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.001)

        assert abs(result - 9.9) <= 0.001 * 10.0 / 2.0

    def test_floating_point_precision(self) -> None:
        """Test with values that challenge floating point precision."""

        def test_fn(x: float) -> bool:
            return x < 1e-10

        result = GeomMath.bisect(-1e-9, 1e-9, test_fn, tolerance=0.01)

        assert abs(result - 1e-10) <= 0.01 * 2e-9 / 2.0

    def test_large_numbers(self) -> None:
        """Test with very large numbers."""

        def test_fn(x: float) -> bool:
            return x < 1e10

        result = GeomMath.bisect(0.0, 1e11, test_fn, tolerance=0.001)

        assert abs(result - 1e10) <= 0.001 * 1e11 / 2.0

    def test_return_type_is_float(self) -> None:
        """Test that return value is a float."""

        def test_fn(x: float) -> bool:
            return x < 5.0

        result = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.01)

        assert isinstance(result, float)

    def test_max_iterations_parameter(self) -> None:
        """Test that max_iterations parameter limits iterations."""
        call_count = [0]

        def test_fn(x: float) -> bool:
            call_count[0] += 1
            return x < 5.0

        # Use very small tolerance but limit iterations
        result = GeomMath.bisect(0.0, 10.0, test_fn, tolerance=0.00001, max_iterations=10)

        # Should stop at max_iterations
        assert call_count[0] <= 12  # 2 initial + 10 iterations
        assert isinstance(result, float)

    def test_default_tolerance(self) -> None:
        """Test that default tolerance of 0.001 works."""

        def test_fn(x: float) -> bool:
            return x < 5.0

        # Call without specifying tolerance
        result = GeomMath.bisect(0.0, 10.0, test_fn)

        # Should converge to within default tolerance (0.001 * 10.0 = 0.01)
        assert abs(result - 5.0) <= 0.001 * 10.0 / 2.0
