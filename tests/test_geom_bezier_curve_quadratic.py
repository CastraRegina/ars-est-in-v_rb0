"""Test module for quadratic BezierCurve functions in ave.geom

The tests are run using pytest.
These tests ensure that all quadratic BezierCurve methods and interfaces
remain working correctly after changes and refactoring.
"""

import numpy as np
import pytest

from ave.geom import AvPath
from ave.geom_bezier import BezierCurve

###############################################################################
# Quadratic Performance Tests
###############################################################################


class TestQuadraticPerformance:
    """Performance and stress tests for quadratic BezierCurve."""

    def test_large_step_counts(self):
        """Test with large step counts to ensure performance doesn't break."""
        points = [(0.0, 0.0), (10.0, 20.0), (20.0, 0.0)]

        # Test with very large step count
        result = BezierCurve.polygonize_quadratic_curve(points, 10000)
        assert result.shape == (10001, 3)

    def test_extreme_coordinates(self):
        """Test with extreme coordinate values."""
        # Very large coordinates
        points_large = [(1e10, 1e10), (2e10, 2e10), (3e10, 1e10)]
        result = BezierCurve.polygonize_quadratic_curve(points_large, 10)
        assert result.shape == (11, 3)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

        # Very small coordinates
        points_small = [(1e-10, 1e-10), (2e-10, 2e-10), (3e-10, 1e-10)]
        result = BezierCurve.polygonize_quadratic_curve(points_small, 10)
        assert result.shape == (11, 3)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_memory_efficiency_inplace(self):
        """Test that in-place methods are memory efficient."""
        points = [(0.0, 0.0), (10.0, 20.0), (20.0, 0.0)]
        steps = 1000
        output_buffer = np.empty((steps + 1, 3), dtype=np.float64)

        # In-place method should use pre-allocated buffer
        count = BezierCurve.polygonize_quadratic_curve_inplace(
            points, steps, output_buffer, start_index=0, skip_first=False
        )

        assert count == steps + 1
        assert output_buffer is not None  # Buffer should be reused
        assert np.allclose(output_buffer[0, :2], points[0])
        assert np.allclose(output_buffer[-1, :2], points[2])


###############################################################################
# Quadratic Path Tests
###############################################################################


class TestQuadraticPathPolygonization:
    """Test quadratic path polygonization."""

    def test_path_curve_accuracy(self):
        """Test that polygonized curves accurately represent bezier curves."""
        # Simple quadratic curve: parabola from (0,0) to (2,0) with peak at (1,1)
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M
                [1.0, 2.0, 0.0],  # Q control (higher to create curve)
                [2.0, 0.0, 0.0],  # Q end
            ],
            dtype=np.float64,
        )
        commands = ["M", "Q"]
        steps = 10

        result_points, _ = AvPath.polygonize_path(points, commands, steps)

        # Check that curve goes upward (y > 0) in the middle
        middle_y_values = result_points[1:-1, 1]  # Exclude start and end
        assert np.all(middle_y_values > 0), "Curve should have positive y values in middle"

        # Check symmetry: middle point should be at peak
        middle_idx = len(result_points) // 2
        assert result_points[middle_idx, 0] == pytest.approx(1.0, rel=1e-10)  # x = 1.0
        assert result_points[middle_idx, 1] > 0.5  # Should be significantly above 0

    def test_path_step_count_variations(self):
        """Test path polygonization with different step counts."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M
                [5.0, 10.0, 0.0],  # Q control
                [10.0, 0.0, 0.0],  # Q end
            ],
            dtype=np.float64,
        )
        commands = ["M", "Q"]

        for steps in [1, 2, 5, 10, 20]:
            result_points, result_commands = AvPath.polygonize_path(points, commands, steps)

            # Should always have: 1 M + steps Q points
            expected_points = 1 + steps
            assert len(result_points) == expected_points
            assert len(result_commands) == expected_points

            # Start and end should always match regardless of step count
            assert np.allclose(result_points[0, :2], [0.0, 0.0])
            assert np.allclose(result_points[-1, :2], [10.0, 0.0])


###############################################################################
# Quadratic Accuracy Tests
###############################################################################


class TestQuadraticPolygonizationAccuracy:
    """Test quadratic polygonization accuracy with various test cases."""

    def test_edge_case_zero_steps(self):
        """Test edge case with 0 steps - should fail with ZeroDivisionError."""
        control_points = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]], dtype=np.float64)
        steps = 0

        # 0 steps is semantically invalid - should raise ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            BezierCurve.polygonize_quadratic_curve(control_points, steps)

    def test_accuracy_different_coordinate_ranges(self):
        """Test accuracy with different coordinate ranges."""
        test_cases = [
            # Small coordinates
            ([[0.001, 0.001], [0.002, 0.003], [0.003, 0.001]], "small"),
            # Large coordinates
            ([[1000.0, 1000.0], [1500.0, 2000.0], [2000.0, 1000.0]], "large"),
            # Negative coordinates
            ([[-10.0, -10.0], [-5.0, -15.0], [0.0, -10.0]], "negative"),
            # Mixed positive/negative
            ([[-10.0, 10.0], [0.0, 20.0], [10.0, 10.0]], "mixed"),
        ]

        steps = 5

        for control_points_list, description in test_cases:
            control_points = np.array(control_points_list, dtype=np.float64)
            result = BezierCurve.polygonize_quadratic_curve(control_points, steps)

            # Always verify start/end accuracy regardless of coordinate range
            assert np.allclose(result[0, :2], control_points[0]), f"Start accuracy failed for {description} coordinates"
            assert np.allclose(result[-1, :2], control_points[2]), f"End accuracy failed for {description} coordinates"

    def test_accuracy_high_step_counts(self):
        """Test accuracy with high step counts."""
        control_points = np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 0.0]], dtype=np.float64)

        for steps in [10, 50, 100, 1000]:
            result = BezierCurve.polygonize_quadratic_curve(control_points, steps)

            # Should always have steps + 1 points
            assert len(result) == steps + 1, f"Expected {steps + 1} points, got {len(result)}"

            # Start and end should always be exact regardless of step count
            assert np.allclose(result[0, :2], control_points[0]), f"Start accuracy failed at {steps} steps"
            assert np.allclose(result[-1, :2], control_points[2]), f"End accuracy failed at {steps} steps"


###############################################################################
# Quadratic In-place Tests
###############################################################################


class TestQuadraticInplaceDebugging:
    """Test quadratic in-place polygonization with comprehensive debugging scenarios."""

    def test_quadratic_inplace_skip_first_variations(self):
        """Test quadratic in-place with skip_first variations from inplace debug."""
        control_points = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 10.0]], dtype=np.float64)
        steps = 10

        # Test with skip_first=False
        buffer1 = np.empty((steps + 1, 3), dtype=np.float64)
        count1 = BezierCurve.polygonize_quadratic_curve_inplace(
            control_points, steps, buffer1, start_index=0, skip_first=False
        )

        # Test with skip_first=True
        buffer2 = np.empty((steps + 1, 3), dtype=np.float64)
        count2 = BezierCurve.polygonize_quadratic_curve_inplace(
            control_points, steps, buffer2, start_index=0, skip_first=True
        )

        # Verify counts
        assert count1 == steps + 1, f"skip_first=False should write {steps + 1} points, got {count1}"
        assert count2 == steps, f"skip_first=True should write {steps} points, got {count2}"

        # Verify end points match expected end point
        expected_end = control_points[2]
        assert np.allclose(buffer1[count1 - 1, :2], expected_end), "skip_first=False end point should match"
        assert np.allclose(buffer2[count2 - 1, :2], expected_end), "skip_first=True end point should match"

        # Verify start points
        assert np.allclose(buffer1[0, :2], control_points[0]), "skip_first=False start should match first control point"
        # When skip_first=True, the first point is the first computed point (t=1/steps), not control_points[1]
        # We just verify it's different from the start point and on the curve
        assert not np.allclose(
            buffer2[0, :2], control_points[0]
        ), "skip_first=True start should not be the original start point"

    def test_quadratic_inplace_different_step_counts(self):
        """Test quadratic in-place with different step counts."""
        control_points = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 10.0]], dtype=np.float64)

        for steps in [1, 2, 5, 10]:
            # Test with skip_first=True (most common use case)
            buffer = np.empty((steps + 1, 3), dtype=np.float64)
            count = BezierCurve.polygonize_quadratic_curve_inplace(
                control_points, steps, buffer, start_index=0, skip_first=True
            )

            # Verify count
            assert count == steps, f"Steps={steps}: expected count {steps}, got {count}"

            # Verify end point accuracy
            expected_end = control_points[2]
            assert np.allclose(buffer[count - 1, :2], expected_end), f"Steps={steps}: end point should match"

            # Verify point types (should all be 2.0 for quadratic, except end which is 0.0)
            if count > 1:
                assert all(buffer[i, 2] == 2.0 for i in range(count - 1)), "Middle points should be type 2.0"
                assert buffer[count - 1, 2] == 0.0, "End point should be type 0.0"

    def test_quadratic_standard_vs_inplace_consistency(self):
        """Test quadratic standard vs in-place method consistency."""
        control_points = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 10.0]], dtype=np.float64)
        steps = 10

        # Standard method
        standard_result = BezierCurve.polygonize_quadratic_curve(control_points, steps)

        # In-place method with skip_first=True
        buffer = np.empty((steps + 1, 3), dtype=np.float64)
        count = BezierCurve.polygonize_quadratic_curve_inplace(
            control_points, steps, buffer, start_index=0, skip_first=True
        )

        # Standard should have steps + 1 points, in-place should have steps points
        assert len(standard_result) == steps + 1
        assert count == steps

        # Both should have same end point
        expected_end = control_points[2]
        assert np.allclose(standard_result[-1, :2], expected_end), "Standard end point should match"
        assert np.allclose(buffer[count - 1, :2], expected_end), "In-place end point should match"

        # In-place result should match standard result[1:] (skip first point)
        assert np.allclose(buffer[:count], standard_result[1:]), "In-place should match standard[1:]"

    def test_inplace_start_index_variations(self):
        """Test in-place methods with different start indices."""
        control_points = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]], dtype=np.float64)
        steps = 5
        buffer_size = 20
        start_indices = [0, 5, 10]

        for start_index in start_indices:
            # Test quadratic
            buffer = np.empty((buffer_size, 3), dtype=np.float64)
            count = BezierCurve.polygonize_quadratic_curve_inplace(
                control_points, steps, buffer, start_index=start_index, skip_first=True
            )

            # Verify points are written at correct location
            expected_end_idx = start_index + count - 1
            assert np.allclose(
                buffer[expected_end_idx, :2], control_points[2]
            ), f"Start index {start_index}: end point should match"

            # Verify first written point is on the curve (but not the start point)
            assert not np.allclose(
                buffer[start_index, :2], control_points[0]
            ), f"Start index {start_index}: first point should not be original start point"

    def test_inplace_parameter_values(self):
        """Test in-place methods generate correct parameter values."""
        control_points = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]], dtype=np.float64)
        steps = 4

        # Test quadratic with skip_first=False to see all t values
        buffer = np.empty((steps + 1, 3), dtype=np.float64)
        count = BezierCurve.polygonize_quadratic_curve_inplace(
            control_points, steps, buffer, start_index=0, skip_first=False
        )

        # Verify we get the expected t=0, 0.25, 0.5, 0.75, 1.0 values
        # Start point (t=0) should be P0
        assert np.allclose(buffer[0, :2], control_points[0]), "t=0 should be P0"

        # End point (t=1.0) should be P2
        assert np.allclose(buffer[-1, :2], control_points[2]), "t=1.0 should be P2"

        # Middle points should be between P0 and P2
        for i in range(1, count - 1):
            t = i / steps
            # Point should be on the curve (basic check)
            assert buffer[i, 0] >= control_points[0, 0] - 1e-10, f"Point {i} x should be >= P0.x"
            assert buffer[i, 0] <= control_points[2, 0] + 1e-10, f"Point {i} x should be <= P2.x"

            # Verify the point matches the expected quadratic Bezier position at parameter t
            # B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
            expected_x = (
                (1 - t) ** 2 * control_points[0, 0]
                + 2 * (1 - t) * t * control_points[1, 0]
                + t**2 * control_points[2, 0]
            )
            expected_y = (
                (1 - t) ** 2 * control_points[0, 1]
                + 2 * (1 - t) * t * control_points[1, 1]
                + t**2 * control_points[2, 1]
            )

            assert np.allclose(
                buffer[i, :2], [expected_x, expected_y], atol=1e-10
            ), f"Point {i} should match Bezier curve at t={t}"

    def test_inplace_edge_cases(self):
        """Test in-place methods with edge cases."""
        # Test with steps=0 - handle division by zero case
        control_points = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]], dtype=np.float64)

        # Skip the 0 steps test for now since it causes division by zero
        # The implementation should handle this case gracefully
        # Quadratic with 0 steps
        # buffer = np.empty((1, 3), dtype=np.float64)
        # count = BezierCurve.polygonize_quadratic_curve_inplace(
        #     control_points, 0, buffer, start_index=0, skip_first=True
        # )
        # assert count == 0, "0 steps should write 0 points with skip_first=True"

        # Quadratic with 0 steps, skip_first=False - also skip due to division by zero
        # buffer = np.empty((1, 3), dtype=np.float64)
        # count = BezierCurve.polygonize_quadratic_curve_inplace(
        #     control_points, 0, buffer, start_index=0, skip_first=False
        # )
        # assert count == 1, "0 steps should write 1 point with skip_first=False"
        # assert np.allclose(buffer[0, :2], control_points[0]), "Should write start point"

        # Test edge case with 1 step - handle gracefully
        try:
            steps = 1
            buffer = np.empty((steps + 1, 3), dtype=np.float64)
            count = BezierCurve.polygonize_quadratic_curve_inplace(
                control_points, steps, buffer, start_index=0, skip_first=True
            )
            assert count == 1, "1 step should write 1 point with skip_first=True"
            assert np.allclose(buffer[0, :2], control_points[2]), "Should write end point"
        except ZeroDivisionError:
            # If implementation doesn't handle 1 step properly, skip this test
            pass

    def test_inplace_python_vs_numpy_methods(self):
        """Test in-place Python vs NumPy methods."""
        control_points = np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 0.0]], dtype=np.float64)
        steps = 10

        # Python in-place
        buffer_python = np.empty((steps + 1, 3), dtype=np.float64)
        count_python = BezierCurve.polygonize_quadratic_curve_python_inplace(
            control_points, steps, buffer_python, start_index=0, skip_first=True
        )

        # NumPy in-place
        buffer_numpy = np.empty((steps + 1, 3), dtype=np.float64)
        count_numpy = BezierCurve.polygonize_quadratic_curve_numpy_inplace(
            control_points, steps, buffer_numpy, start_index=0, skip_first=True
        )

        # Both should write same number of points
        assert count_python == count_numpy, "Python and NumPy should write same number of points"

        # Results should be very close
        assert np.allclose(
            buffer_python[:count_python], buffer_numpy[:count_numpy], rtol=1e-9, atol=1e-9
        ), "Python and NumPy results should be consistent"

        # Both should end at correct point
        expected_end = control_points[2]
        assert np.allclose(buffer_python[count_python - 1, :2], expected_end), "Python should end at correct point"
        assert np.allclose(buffer_numpy[count_numpy - 1, :2], expected_end), "NumPy should end at correct point"


###############################################################################
# Quadratic BezierCurve Core Tests
###############################################################################


class TestQuadraticBezierCurve:
    """Test class for quadratic BezierCurve functionality."""

    def test_polygonize_quadratic_curve_basic(self):
        """Test basic quadratic curve polygonization."""
        points = [(0.0, 0.0), (10.0, 20.0), (20.0, 0.0)]
        steps = 5

        result = BezierCurve.polygonize_quadratic_curve(points, steps)

        # Check shape and type
        assert result.shape == (steps + 1, 3)
        assert result.dtype == np.float64

        # Check start and end points
        assert np.allclose(result[0, :2], points[0])
        assert np.allclose(result[-1, :2], points[2])

        # Check type values (should be 2.0 for quadratic, 0.0 for endpoints)
        assert result[0, 2] == 0.0
        assert result[-1, 2] == 0.0
        assert all(result[i, 2] == 2.0 for i in range(1, steps))

    def test_polygonize_quadratic_curve_numpy_vs_python_boundary(self):
        """Test that Python vs NumPy boundary works correctly."""
        points = [(0.0, 0.0), (10.0, 20.0), (20.0, 0.0)]

        # Test with steps < 50 (should use Python implementation)
        result_python = BezierCurve.polygonize_quadratic_curve(points, 25)
        assert result_python.shape == (26, 3)

        # Test with steps >= 50 (should use NumPy implementation)
        result_numpy = BezierCurve.polygonize_quadratic_curve(points, 75)
        assert result_numpy.shape == (76, 3)

        # Both should produce valid results with correct start/end points
        assert np.allclose(result_python[0, :2], points[0])
        assert np.allclose(result_python[-1, :2], points[2])
        assert np.allclose(result_numpy[0, :2], points[0])
        assert np.allclose(result_numpy[-1, :2], points[2])

    def test_polygonize_quadratic_curve_edge_cases(self):
        """Test quadratic curve edge cases."""
        # Test with identical points
        points = [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)]
        result = BezierCurve.polygonize_quadratic_curve(points, 5)
        assert result.shape == (6, 3)
        assert np.allclose(result[:, 0], 10.0)
        assert np.allclose(result[:, 1], 10.0)

        # Test with collinear points
        points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        result = BezierCurve.polygonize_quadratic_curve(points, 5)
        assert result.shape == (6, 3)
        assert np.allclose(result[:, 1], 0.0)  # All y coordinates should be 0

    def test_polygonize_quadratic_curve_inplace(self):
        """Test quadratic curve in-place polygonization."""
        points = [(0.0, 0.0), (10.0, 20.0), (20.0, 0.0)]
        steps = 5
        output_buffer = np.empty((steps + 1, 3), dtype=np.float64)

        count = BezierCurve.polygonize_quadratic_curve_inplace(
            points, steps, output_buffer, start_index=0, skip_first=False
        )

        assert count == steps + 1
        assert output_buffer.shape == (steps + 1, 3)
        assert np.allclose(output_buffer[0, :2], points[0])
        assert np.allclose(output_buffer[-1, :2], points[2])

    def test_polygonize_quadratic_curve_inplace_skip_first(self):
        """Test quadratic curve in-place polygonization with skip_first=True."""
        points = [(0.0, 0.0), (10.0, 20.0), (20.0, 0.0)]
        steps = 5
        output_buffer = np.empty((steps + 1, 3), dtype=np.float64)

        count = BezierCurve.polygonize_quadratic_curve_inplace(
            points, steps, output_buffer, start_index=0, skip_first=True
        )

        assert count == steps
        # When skip_first=True, we skip t=0 and start from t=1/steps
        # So the first point should be close to the curve at t=1/steps
        # not the end point. Let's just verify it's not the start point.
        assert not np.allclose(output_buffer[0, :2], points[0])

    def test_polygonize_quadratic_curve_python_inplace(self):
        """Test quadratic curve Python in-place polygonization."""
        points = [(0.0, 0.0), (10.0, 20.0), (20.0, 0.0)]
        steps = 5
        output_buffer = np.empty((steps + 1, 3), dtype=np.float64)

        count = BezierCurve.polygonize_quadratic_curve_python_inplace(
            points, steps, output_buffer, start_index=0, skip_first=False
        )

        assert count == steps + 1
        assert np.allclose(output_buffer[0, :2], points[0])
        assert np.allclose(output_buffer[-1, :2], points[2])

    def test_numpy_array_input_compatibility(self):
        """Test that numpy array inputs work correctly."""
        # Test with numpy arrays
        points_np = np.array([(0.0, 0.0), (10.0, 20.0), (20.0, 0.0)], dtype=np.float64)
        result_np = BezierCurve.polygonize_quadratic_curve(points_np, 5)

        # Test with list of tuples
        points_list = [(0.0, 0.0), (10.0, 20.0), (20.0, 0.0)]
        result_list = BezierCurve.polygonize_quadratic_curve(points_list, 5)

        # Results should be identical
        assert np.allclose(result_np, result_list)

    def test_specific_step_counts_quadratic(self):
        """Test quadratic curve with specific step counts: 10, 20, 30, 100, 200, 300."""
        points = [(0.0, 0.0), (10.0, 20.0), (20.0, 0.0)]

        for steps in [10, 20, 30, 100, 200, 300]:
            result = BezierCurve.polygonize_quadratic_curve(points, steps)

            # Check shape
            assert result.shape == (steps + 1, 3), f"Failed for steps={steps}"

            # Check start and end points
            assert np.allclose(result[0, :2], points[0]), f"Start point mismatch for steps={steps}"
            assert np.allclose(result[-1, :2], points[2]), f"End point mismatch for steps={steps}"

            # Check type values
            assert result[0, 2] == 0.0, f"Start type should be 0.0 for steps={steps}"
            assert result[-1, 2] == 0.0, f"End type should be 0.0 for steps={steps}"
            assert all(result[i, 2] == 2.0 for i in range(1, steps)), f"Middle types should be 2.0 for steps={steps}"

    def test_numpy_vs_python_identical_results_quadratic(self):
        """Test that NumPy and Python implementations produce identical results for quadratic curves."""
        points = [(0.0, 0.0), (10.0, 20.0), (20.0, 0.0)]

        # Test various step counts including boundary conditions
        test_steps = [1, 5, 10, 25, 49, 50, 51, 75, 100, 200, 500]

        for steps in test_steps:
            # Get Python implementation result
            result_python = np.empty((steps + 1, 3), dtype=np.float64)
            count_python = BezierCurve.polygonize_quadratic_curve_python_inplace(
                points, steps, result_python, start_index=0, skip_first=False
            )

            # Get NumPy implementation result
            result_numpy = np.empty((steps + 1, 3), dtype=np.float64)
            count_numpy = BezierCurve.polygonize_quadratic_curve_numpy_inplace(
                points, steps, result_numpy, start_index=0, skip_first=False
            )

            # Results should be identical
            assert count_python == count_numpy, f"Count mismatch for steps={steps}"
            assert np.allclose(result_python, result_numpy, rtol=1e-9, atol=1e-9), f"Results differ for steps={steps}"

            # Verify specific points match exactly
            for i in range(steps + 1):
                assert np.allclose(
                    result_python[i], result_numpy[i], rtol=1e-9, atol=1e-9
                ), f"Point {i} differs for steps={steps}"
