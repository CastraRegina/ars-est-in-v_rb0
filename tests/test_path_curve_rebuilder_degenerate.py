"""Tests for AvPathCurveRebuilder degenerate curve detection and fixing."""

from __future__ import annotations

import numpy as np
import pytest

from ave.path import AvPath
from ave.path_processing import AvPathCurveRebuilder


class TestFixDegenerateCurves:
    """Test suite for _fix_degenerate_curves method."""

    def test_valid_quadratic_curve_unchanged(self):
        """Test that valid quadratic curves are not modified."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [5.0, 10.0, 2.0],  # Q - control
                [10.0, 0.0, 0.0],  # Q - end
            ]
        )
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        assert len(result.points) == 3
        assert result.commands == ["M", "Q"]
        np.testing.assert_array_almost_equal(result.points, points)

    def test_quadratic_control_coincides_with_start(self):
        """Test Q curve where control point coincides with start point."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [0.0, 1e-15, 2.0],  # Q - control (coincides with start)
                [10.0, 0.0, 0.0],  # Q - end
            ]
        )
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # Should be replaced with L
        assert len(result.points) == 2
        assert result.commands == ["M", "L"]
        np.testing.assert_array_almost_equal(result.points[0], points[0])
        np.testing.assert_array_almost_equal(result.points[1], points[2])

    def test_quadratic_control_coincides_with_end(self):
        """Test Q curve where control point coincides with end point."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [10.0, 1e-15, 2.0],  # Q - control (coincides with end)
                [10.0, 0.0, 0.0],  # Q - end
            ]
        )
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # Should be replaced with L
        assert len(result.points) == 2
        assert result.commands == ["M", "L"]

    def test_quadratic_start_coincides_with_end(self):
        """Test Q curve where start and end points coincide."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [5.0, 10.0, 2.0],  # Q - control
                [0.0, 1e-15, 0.0],  # Q - end (coincides with start)
            ]
        )
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # Q should be removed entirely
        assert len(result.points) == 1
        assert result.commands == ["M"]

    def test_quadratic_collinear_points(self):
        """Test Q curve where all three points are collinear."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [5.0, 0.0, 2.0],  # Q - control (on line)
                [10.0, 0.0, 0.0],  # Q - end
            ]
        )
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # Should be replaced with L
        assert len(result.points) == 2
        assert result.commands == ["M", "L"]

    def test_valid_cubic_curve_unchanged(self):
        """Test that valid cubic curves are not modified."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [3.0, 10.0, 3.0],  # C - ctrl1
                [7.0, 10.0, 3.0],  # C - ctrl2
                [10.0, 0.0, 0.0],  # C - end
            ]
        )
        commands = ["M", "C"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        assert len(result.points) == 4
        assert result.commands == ["M", "C"]
        np.testing.assert_array_almost_equal(result.points, points)

    def test_cubic_ctrl1_coincides_with_start(self):
        """Test C curve where ctrl1 coincides with start."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [0.0, 1e-15, 3.0],  # C - ctrl1 (coincides with start)
                [7.0, 10.0, 3.0],  # C - ctrl2
                [10.0, 0.0, 0.0],  # C - end
            ]
        )
        commands = ["M", "C"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # Should be simplified to Q with ctrl2 as control point
        assert len(result.points) == 3
        assert result.commands == ["M", "Q"]
        np.testing.assert_array_almost_equal(result.points[0], points[0])
        np.testing.assert_array_almost_equal(result.points[1], points[2])  # ctrl2
        np.testing.assert_array_almost_equal(result.points[2], points[3])  # end

    def test_cubic_ctrl2_coincides_with_end(self):
        """Test C curve where ctrl2 coincides with end."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [3.0, 10.0, 3.0],  # C - ctrl1
                [10.0, 1e-15, 3.0],  # C - ctrl2 (coincides with end)
                [10.0, 0.0, 0.0],  # C - end
            ]
        )
        commands = ["M", "C"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # Should be simplified to Q with ctrl1 as control point
        assert len(result.points) == 3
        assert result.commands == ["M", "Q"]
        np.testing.assert_array_almost_equal(result.points[0], points[0])
        np.testing.assert_array_almost_equal(result.points[1], points[1])  # ctrl1
        np.testing.assert_array_almost_equal(result.points[2], points[3])  # end

    def test_cubic_both_controls_coincide_with_endpoints(self):
        """Test C curve where ctrl1=start and ctrl2=end."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [0.0, 1e-15, 3.0],  # C - ctrl1 (coincides with start)
                [10.0, 1e-15, 3.0],  # C - ctrl2 (coincides with end)
                [10.0, 0.0, 0.0],  # C - end
            ]
        )
        commands = ["M", "C"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # Should be replaced with L
        assert len(result.points) == 2
        assert result.commands == ["M", "L"]

    def test_cubic_ctrl1_and_ctrl2_coincide(self):
        """Test C curve where ctrl1 and ctrl2 coincide."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [5.0, 10.0, 3.0],  # C - ctrl1
                [5.0, 10.0 + 1e-15, 3.0],  # C - ctrl2 (coincides with ctrl1)
                [10.0, 0.0, 0.0],  # C - end
            ]
        )
        commands = ["M", "C"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # Should be simplified to Q
        assert len(result.points) == 3
        assert result.commands == ["M", "Q"]

    def test_cubic_start_coincides_with_end(self):
        """Test C curve where start and end coincide."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [3.0, 10.0, 3.0],  # C - ctrl1
                [7.0, 10.0, 3.0],  # C - ctrl2
                [0.0, 1e-15, 0.0],  # C - end (coincides with start)
            ]
        )
        commands = ["M", "C"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # C should be removed entirely
        assert len(result.points) == 1
        assert result.commands == ["M"]

    def test_cubic_collinear_points(self):
        """Test C curve where all four points are collinear."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [3.0, 0.0, 3.0],  # C - ctrl1 (on line)
                [7.0, 0.0, 3.0],  # C - ctrl2 (on line)
                [10.0, 0.0, 0.0],  # C - end
            ]
        )
        commands = ["M", "C"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # Should be replaced with L
        assert len(result.points) == 2
        assert result.commands == ["M", "L"]

    def test_multiple_curves_mixed(self):
        """Test path with multiple curves, some degenerate."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [5.0, 10.0, 2.0],  # Q - control (valid)
                [10.0, 0.0, 0.0],  # Q - end
                [10.0, 1e-15, 2.0],  # Q - control (degenerate, coincides with prev end)
                [20.0, 0.0, 0.0],  # Q - end
                [23.0, 10.0, 3.0],  # C - ctrl1 (valid)
                [27.0, 10.0, 3.0],  # C - ctrl2
                [30.0, 0.0, 0.0],  # C - end
            ]
        )
        commands = ["M", "Q", "Q", "C"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # First Q valid, second Q replaced with L, C valid
        assert result.commands == ["M", "Q", "L", "C"]
        assert len(result.points) == 7  # M(1) + Q(2) + L(1) + C(3)

    def test_closed_path_with_z(self):
        """Test that Z commands are preserved."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [5.0, 10.0, 2.0],  # Q - control
                [10.0, 0.0, 0.0],  # Q - end
            ]
        )
        commands = ["M", "Q", "Z"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        assert result.commands == ["M", "Q", "Z"]
        assert len(result.points) == 3

    def test_empty_path(self):
        """Test empty path handling."""
        path = AvPath()
        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        assert len(result.points) == 0
        assert len(result.commands) == 0

    def test_path_with_only_m_command(self):
        """Test path with only M command."""
        points = np.array([[0.0, 0.0, 0.0]])
        commands = ["M"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        assert len(result.points) == 1
        assert result.commands == ["M"]

    def test_path_with_l_commands_only(self):
        """Test path with only L commands (no curves)."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
            ]
        )
        commands = ["M", "L", "L"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        assert len(result.points) == 3
        assert result.commands == ["M", "L", "L"]
        np.testing.assert_array_almost_equal(result.points, points)

    def test_real_world_case_from_glyph_b(self):
        """Test the actual degenerate case found in glyph 'b'."""
        # This is the degenerate Q curve that was causing the duplicate point error
        points = np.array(
            [
                [145.0, 0.0, 0.0],  # Start (previous end point)
                [145.0, 8.200438e-16, 2.0],  # Control (coincides with start)
                [147.0, 8.0, 0.0],  # End
            ]
        )
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)

        # Should be replaced with L
        assert len(result.points) == 2
        assert result.commands == ["M", "L"]
        np.testing.assert_array_almost_equal(result.points[0], points[0])
        np.testing.assert_array_almost_equal(result.points[1], points[2])
