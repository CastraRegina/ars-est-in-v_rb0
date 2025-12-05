"""Test module for cubic BezierCurve functions in ave.geom

The tests are run using pytest.
These tests ensure that all cubic BezierCurve methods and interfaces
remain working correctly after changes and refactoring.
"""

import numpy as np
import pytest

from ave.geom import BezierCurve

###############################################################################
# Cubic In-place Tests
###############################################################################


class TestCubicInplaceDebugging:
    """Test cubic in-place polygonization with comprehensive debugging scenarios."""

    def test_cubic_inplace_skip_first_variations(self):
        """Test cubic in-place with skip_first variations from cubic debug."""
        control_points = np.array([[30.0, 10.0], [35.0, 15.0], [40.0, 15.0], [45.0, 10.0]], dtype=np.float64)
        steps = 10

        # Test with skip_first=False
        buffer1 = np.empty((steps + 1, 3), dtype=np.float64)
        count1 = BezierCurve.polygonize_cubic_curve_inplace(
            control_points, steps, buffer1, start_index=0, skip_first=False
        )

        # Test with skip_first=True
        buffer2 = np.empty((steps + 1, 3), dtype=np.float64)
        count2 = BezierCurve.polygonize_cubic_curve_inplace(
            control_points, steps, buffer2, start_index=0, skip_first=True
        )

        # Verify counts
        assert count1 == steps + 1, f"skip_first=False should write {steps + 1} points, got {count1}"
        assert count2 == steps, f"skip_first=True should write {steps} points, got {count2}"

        # Verify end points match expected end point
        expected_end = control_points[3]
        assert np.allclose(buffer1[count1 - 1, :2], expected_end), "skip_first=False end point should match"
        assert np.allclose(buffer2[count2 - 1, :2], expected_end), "skip_first=True end point should match"

    def test_cubic_inplace_different_step_counts(self):
        """Test cubic in-place with different step counts."""
        control_points = np.array([[30.0, 10.0], [35.0, 15.0], [40.0, 15.0], [45.0, 10.0]], dtype=np.float64)

        for steps in [1, 2, 5, 10]:
            # Test with skip_first=True (most common use case)
            buffer = np.empty((steps + 1, 3), dtype=np.float64)
            count = BezierCurve.polygonize_cubic_curve_inplace(
                control_points, steps, buffer, start_index=0, skip_first=True
            )

            # Verify count
            assert count == steps, f"Steps={steps}: expected count {steps}, got {count}"

            # Verify end point accuracy
            expected_end = control_points[3]
            assert np.allclose(buffer[count - 1, :2], expected_end), f"Steps={steps}: end point should match"

            # Verify point types (should all be 3.0 for cubic, except end which is 0.0)
            if count > 1:
                assert all(buffer[i, 2] == 3.0 for i in range(count - 1)), "Middle points should be type 3.0"
                assert buffer[count - 1, 2] == 0.0, "End point should be type 0.0"

    def test_cubic_standard_vs_inplace_consistency(self):
        """Test cubic standard vs in-place method consistency."""
        control_points = np.array([[30.0, 10.0], [35.0, 15.0], [40.0, 15.0], [45.0, 10.0]], dtype=np.float64)
        steps = 10

        # Standard method
        standard_result = BezierCurve.polygonize_cubic_curve(control_points, steps)

        # In-place method with skip_first=True
        buffer = np.empty((steps + 1, 3), dtype=np.float64)
        count = BezierCurve.polygonize_cubic_curve_inplace(
            control_points, steps, buffer, start_index=0, skip_first=True
        )

        # Standard should have steps + 1 points, in-place should have steps points
        assert len(standard_result) == steps + 1
        assert count == steps

        # Both should have same end point
        expected_end = control_points[3]
        assert np.allclose(standard_result[-1, :2], expected_end), "Standard end point should match"
        assert np.allclose(buffer[count - 1, :2], expected_end), "In-place end point should match"

        # In-place result should match standard result[1:] (skip first point)
        assert np.allclose(buffer[:count], standard_result[1:]), "In-place should match standard[1:]"

    def test_inplace_cubic_python_vs_numpy_methods(self):
        """Test in-place Python vs NumPy methods for cubic curves."""
        control_points = np.array([[0.0, 0.0], [5.0, 20.0], [15.0, 20.0], [20.0, 0.0]], dtype=np.float64)
        steps = 10

        # Python in-place
        buffer_python = np.empty((steps + 1, 3), dtype=np.float64)
        count_python = BezierCurve.polygonize_cubic_curve_python_inplace(
            control_points, steps, buffer_python, start_index=0, skip_first=True
        )

        # NumPy in-place
        buffer_numpy = np.empty((steps + 1, 3), dtype=np.float64)
        count_numpy = BezierCurve.polygonize_cubic_curve_numpy_inplace(
            control_points, steps, buffer_numpy, start_index=0, skip_first=True
        )

        # Both should write same number of points
        assert count_python == count_numpy, "Python and NumPy should write same number of points"

        # Results should be very close
        assert np.allclose(
            buffer_python[:count_python], buffer_numpy[:count_numpy], rtol=1e-9, atol=1e-9
        ), "Python and NumPy results should be consistent"

        # Both should end at correct point
        expected_end = control_points[3]
        assert np.allclose(buffer_python[count_python - 1, :2], expected_end), "Python should end at correct point"
        assert np.allclose(buffer_numpy[count_numpy - 1, :2], expected_end), "NumPy should end at correct point"


###############################################################################
# Cubic BezierCurve Core Tests
###############################################################################


class TestCubicBezierCurve:
    """Test class for cubic BezierCurve functionality."""

    def test_polygonize_cubic_curve_basic(self):
        """Test basic cubic curve polygonization."""
        points = [(0.0, 0.0), (5.0, 20.0), (15.0, 20.0), (20.0, 0.0)]
        steps = 5

        result = BezierCurve.polygonize_cubic_curve(points, steps)

        # Check shape and type
        assert result.shape == (steps + 1, 3)
        assert result.dtype == np.float64

        # Check start and end points
        assert np.allclose(result[0, :2], points[0])
        assert np.allclose(result[-1, :2], points[3])

        # Check type values (should be 3.0 for cubic, 0.0 for endpoints)
        assert result[0, 2] == 0.0
        assert result[-1, 2] == 0.0
        assert all(result[i, 2] == 3.0 for i in range(1, steps))

    def test_polygonize_cubic_curve_numpy_vs_python_boundary(self):
        """Test that Python vs NumPy boundary works correctly for cubic curves."""
        points = [(0.0, 0.0), (5.0, 20.0), (15.0, 20.0), (20.0, 0.0)]

        # Test with steps < 50 (should use Python implementation)
        result_python = BezierCurve.polygonize_cubic_curve(points, 25)
        assert result_python.shape == (26, 3)

        # Test with steps >= 50 (should use NumPy implementation)
        result_numpy = BezierCurve.polygonize_cubic_curve(points, 75)
        assert result_numpy.shape == (76, 3)

        # Both should produce valid results with correct start/end points
        assert np.allclose(result_python[0, :2], points[0])
        assert np.allclose(result_python[-1, :2], points[3])
        assert np.allclose(result_numpy[0, :2], points[0])
        assert np.allclose(result_numpy[-1, :2], points[3])

    def test_polygonize_cubic_curve_edge_cases(self):
        """Test cubic curve edge cases."""
        # Test with identical points
        points = [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0), (10.0, 10.0)]
        result = BezierCurve.polygonize_cubic_curve(points, 5)
        assert result.shape == (6, 3)
        assert np.allclose(result[:, 0], 10.0)
        assert np.allclose(result[:, 1], 10.0)

        # Test with collinear points
        points = [(0.0, 0.0), (5.0, 0.0), (15.0, 0.0), (20.0, 0.0)]
        result = BezierCurve.polygonize_cubic_curve(points, 5)
        assert result.shape == (6, 3)
        assert np.allclose(result[:, 1], 0.0)  # All y coordinates should be 0

    def test_polygonize_cubic_curve_inplace(self):
        """Test cubic curve in-place polygonization."""
        points = [(0.0, 0.0), (5.0, 20.0), (15.0, 20.0), (20.0, 0.0)]
        steps = 5
        output_buffer = np.empty((steps + 1, 3), dtype=np.float64)

        count = BezierCurve.polygonize_cubic_curve_inplace(
            points, steps, output_buffer, start_index=0, skip_first=False
        )

        assert count == steps + 1
        assert output_buffer.shape == (steps + 1, 3)
        assert np.allclose(output_buffer[0, :2], points[0])
        assert np.allclose(output_buffer[-1, :2], points[3])

    def test_polygonize_cubic_curve_inplace_skip_first(self):
        """Test cubic curve in-place polygonization with skip_first=True."""
        points = [(0.0, 0.0), (5.0, 20.0), (15.0, 20.0), (20.0, 0.0)]
        steps = 5
        output_buffer = np.empty((steps + 1, 3), dtype=np.float64)

        count = BezierCurve.polygonize_cubic_curve_inplace(points, steps, output_buffer, start_index=0, skip_first=True)

        assert count == steps
        # When skip_first=True, we skip t=0 and start from t=1/steps
        # So the first point should be close to the curve at t=1/steps
        # not the end point. Let's just verify it's not the start point.
        assert not np.allclose(output_buffer[0, :2], points[0])

    def test_polygonize_cubic_curve_python(self):
        """Test cubic curve Python convenience wrapper."""
        points = [(0.0, 0.0), (5.0, 20.0), (15.0, 20.0), (20.0, 0.0)]
        steps = 5

        result = BezierCurve.polygonize_cubic_curve_python(points, steps)

        assert result.shape == (steps + 1, 3)
        assert np.allclose(result[0, :2], points[0])
        assert np.allclose(result[-1, :2], points[3])

    def test_polygonize_cubic_curve_numpy(self):
        """Test cubic curve NumPy convenience wrapper."""
        points = [(0.0, 0.0), (5.0, 20.0), (15.0, 20.0), (20.0, 0.0)]
        steps = 5

        result = BezierCurve.polygonize_cubic_curve_numpy(points, steps)

        assert result.shape == (steps + 1, 3)
        assert np.allclose(result[0, :2], points[0])
        assert np.allclose(result[-1, :2], points[3])

    def test_polygonize_path_cubic_curve(self):
        """Test path with cubic curve command."""
        points = np.array(
            [[10.0, 10.0, 0.0], [10.0, 30.0, 0.0], [30.0, 30.0, 0.0], [30.0, 10.0, 0.0]],
            dtype=np.float64,
        )
        commands = ["M", "C"]
        steps = 5

        new_points, new_commands = BezierCurve.polygonize_path(points, commands, steps)

        assert len(new_points) == steps + 1
        assert len(new_commands) == steps + 1
        assert new_commands[0] == "M"
        assert all(cmd == "L" for cmd in new_commands[1:])

    def test_specific_step_counts_cubic(self):
        """Test cubic curve with specific step counts: 10, 20, 30, 100, 200, 300."""
        points = [(0.0, 0.0), (5.0, 20.0), (15.0, 20.0), (20.0, 0.0)]

        for steps in [10, 20, 30, 100, 200, 300]:
            result = BezierCurve.polygonize_cubic_curve(points, steps)

            # Check shape
            assert result.shape == (steps + 1, 3), f"Failed for steps={steps}"

            # Check start and end points
            assert np.allclose(result[0, :2], points[0]), f"Start point mismatch for steps={steps}"
            assert np.allclose(result[-1, :2], points[3]), f"End point mismatch for steps={steps}"

            # Check type values
            assert result[0, 2] == 0.0, f"Start type should be 0.0 for steps={steps}"
            assert result[-1, 2] == 0.0, f"End type should be 0.0 for steps={steps}"
            assert all(result[i, 2] == 3.0 for i in range(1, steps)), f"Middle types should be 3.0 for steps={steps}"

    def test_numpy_vs_python_identical_results_cubic(self):
        """Test that NumPy and Python implementations produce identical results for cubic curves."""
        points = [(0.0, 0.0), (5.0, 20.0), (15.0, 20.0), (20.0, 0.0)]

        # Test various step counts including boundary conditions
        test_steps = [1, 5, 10, 25, 49, 50, 51, 75, 100, 200, 500]

        for steps in test_steps:
            # Get Python implementation result
            result_python = np.empty((steps + 1, 3), dtype=np.float64)
            count_python = BezierCurve.polygonize_cubic_curve_python_inplace(
                points, steps, result_python, start_index=0, skip_first=False
            )

            # Get NumPy implementation result
            result_numpy = np.empty((steps + 1, 3), dtype=np.float64)
            count_numpy = BezierCurve.polygonize_cubic_curve_numpy_inplace(
                points, steps, result_numpy, start_index=0, skip_first=False
            )

            # Results should be mathematically equivalent within realistic tolerance
            # Python uses forward differencing, NumPy uses direct evaluation
            # Use step-dependent tolerance: 1e-9 for low steps, 1e-8 for high steps
            tolerance = 1e-9 if steps < 500 else 1e-8
            assert count_python == count_numpy, f"Count mismatch for steps={steps}"
            assert np.allclose(
                result_python[:count_python], result_numpy[:count_numpy], rtol=tolerance, atol=tolerance
            ), f"Results differ for steps={steps}"

            # Verify specific points match exactly
            for i in range(count_python):
                assert np.allclose(
                    result_python[i], result_numpy[i], rtol=tolerance, atol=tolerance
                ), f"Point {i} differs for steps={steps}"

    def test_numpy_vs_python_different_curve_shapes(self):
        """Test NumPy vs Python implementations with various cubic curve shapes."""
        test_cases = [
            # Cubic curves
            ([(0.0, 0.0), (5.0, 20.0), (15.0, 20.0), (20.0, 0.0)], "cubic"),
            ([(0.0, 0.0), (0.0, 0.0), (20.0, 0.0), (20.0, 0.0)], "cubic"),  # Control points at start/end
            ([(0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (20.0, 0.0)], "cubic"),  # Middle control points same
            ([(0.0, 0.0), (5.0, 0.0), (15.0, 0.0), (20.0, 0.0)], "cubic"),  # Collinear
            ([(0.0, 0.0), (5.0, 30.0), (15.0, -10.0), (20.0, 0.0)], "cubic"),  # S-curve
        ]

        steps = 50  # Use step count that triggers NumPy implementation

        for points, curve_type in test_cases:
            # Python implementation
            result_python = np.empty((steps + 1, 3), dtype=np.float64)
            count_python = BezierCurve.polygonize_cubic_curve_python_inplace(
                points, steps, result_python, start_index=0, skip_first=False
            )

            # NumPy implementation
            result_numpy = np.empty((steps + 1, 3), dtype=np.float64)
            count_numpy = BezierCurve.polygonize_cubic_curve_numpy_inplace(
                points, steps, result_numpy, start_index=0, skip_first=False
            )

            # Results should be identical
            assert count_python == count_numpy, f"Count mismatch for {curve_type} curve"
            assert np.allclose(
                result_python, result_numpy, rtol=1e-9, atol=1e-9
            ), f"Results differ for {curve_type} curve"

    def test_high_precision_consistency(self):
        """Test that results remain consistent across different precision requirements."""
        points_cubic = [(0.0, 0.0), (5.0, 20.0), (15.0, 20.0), (20.0, 0.0)]

        # Test with high step counts where precision matters
        high_steps = [1000, 2000, 5000]

        for steps in high_steps:
            # Cubic curve - compare Python vs NumPy
            py_buffer = np.empty((steps + 1, 3), dtype=np.float64)
            np_buffer = np.empty((steps + 1, 3), dtype=np.float64)

            BezierCurve.polygonize_cubic_curve_python_inplace(
                points_cubic, steps, py_buffer, start_index=0, skip_first=False
            )
            BezierCurve.polygonize_cubic_curve_numpy_inplace(
                points_cubic, steps, np_buffer, start_index=0, skip_first=False
            )

            # Use appropriate tolerance for high step counts where numerical accumulation differs
            # Python forward differencing vs NumPy direct evaluation have different error characteristics
            # Allow higher tolerance for very high step counts (5000+) where differences accumulate
            assert np.allclose(
                py_buffer, np_buffer, rtol=2e-7, atol=2e-7
            ), f"High precision cubic mismatch at steps={steps}"

    def test_inplace_vs_original_methods_identical(self):
        """Test that in-place methods produce identical results to original methods."""
        # Test cubic curve
        test_cubic_points = np.array([[0.0, 0.0], [5.0, 15.0], [15.0, 15.0], [20.0, 0.0]], dtype=np.float64)
        cubic_steps = 50

        cubic_original = BezierCurve.polygonize_cubic_curve(test_cubic_points, cubic_steps)
        cubic_inplace_buffer = np.empty((cubic_steps + 1, 3), dtype=np.float64)
        cubic_inplace_count = BezierCurve.polygonize_cubic_curve_inplace(
            test_cubic_points, cubic_steps, cubic_inplace_buffer
        )
        cubic_inplace = cubic_inplace_buffer[:cubic_inplace_count]

        assert np.allclose(cubic_original, cubic_inplace), "Cubic in-place should match original"

    def test_skip_first_vs_original_slice(self):
        """Test that skip_first=True produces identical results to original[1:]."""
        # Test same for cubic curves
        test_cubic_points = np.array([[0.0, 0.0], [5.0, 15.0], [15.0, 15.0], [20.0, 0.0]], dtype=np.float64)
        cubic_steps = 50

        cubic_original = BezierCurve.polygonize_cubic_curve(test_cubic_points, cubic_steps)

        cubic_inplace_skip_buffer = np.empty((cubic_steps + 1, 3), dtype=np.float64)
        cubic_inplace_skip_count = BezierCurve.polygonize_cubic_curve_inplace(
            test_cubic_points, cubic_steps, cubic_inplace_skip_buffer, start_index=0, skip_first=True
        )
        cubic_inplace_skip = cubic_inplace_skip_buffer[:cubic_inplace_skip_count]

        cubic_original_skip = cubic_original[1:]

        assert np.allclose(cubic_original_skip, cubic_inplace_skip), "Cubic skip_first=True should match original[1:]"
