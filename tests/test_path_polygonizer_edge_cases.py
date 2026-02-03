"""Comprehensive edge case tests for PathPolygonizer.

This test file covers edge cases and error conditions for the PathPolygonizer.polygonize_path
function that were not covered in existing tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from ave.common import AvGlyphCmds
from ave.path_support import PathPolygonizer


class TestPathPolygonizerEdgeCases:
    """Test edge cases for PathPolygonizer.polygonize_path."""

    def test_empty_path(self):
        """Test handling of completely empty path."""
        points = np.empty((0, 3), dtype=np.float64)
        commands: list[AvGlyphCmds] = []
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        assert len(result_points) == 0
        assert len(result_commands) == 0

    def test_zero_steps_no_polygonization(self):
        """Test that steps=0 returns original path unchanged."""
        points = np.array([[10.0, 10.0, 0.0], [20.0, 20.0, 0.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M", "L"]
        steps = 0

        def process_2d_to_3d(pts, cmds):
            return pts

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        np.testing.assert_array_equal(result_points, points)
        assert result_commands == commands

    def test_single_move_command(self):
        """Test path with only a MoveTo command."""
        points = np.array([[10.0, 10.0, 0.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        np.testing.assert_array_equal(result_points, points)
        assert result_commands == commands

    def test_only_close_commands(self):
        """Test path with only ClosePath commands (invalid but should handle gracefully)."""
        points = np.array([[10.0, 10.0, 0.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M", "Z", "Z"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        np.testing.assert_array_equal(result_points, points)
        assert result_commands == commands

    def test_quadratic_curve_without_start_point(self):
        """Test quadratic curve command without preceding MoveTo."""
        points = np.array([[20.0, 20.0, 2.0], [30.0, 10.0, 0.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["Q"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        with pytest.raises(ValueError, match="Quadratic Bezier command has no starting point"):
            PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

    def test_cubic_curve_without_start_point(self):
        """Test cubic curve command without preceding MoveTo."""
        points = np.array([[20.0, 20.0, 3.0], [25.0, 25.0, 3.0], [30.0, 10.0, 0.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["C"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        with pytest.raises(ValueError, match="Cubic Bezier command has no starting point"):
            PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

    def test_close_command_without_start_point(self):
        """Test ClosePath command without preceding MoveTo."""
        points = np.empty((0, 3), dtype=np.float64)
        commands: list[AvGlyphCmds] = ["Z"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        with pytest.raises(ValueError, match="ClosePath command has no starting point"):
            PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

    def test_insufficient_points_for_move_command(self):
        """Test MoveTo command with insufficient points."""
        points = np.empty((0, 3), dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        with pytest.raises(ValueError, match="MoveTo command needs 1 point, got 0"):
            PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

    def test_insufficient_points_for_line_command(self):
        """Test LineTo command with insufficient points."""
        points = np.array([[10.0, 10.0, 0.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M", "L"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        with pytest.raises(ValueError, match="LineTo command needs 1 point, got 0"):
            PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

    def test_insufficient_points_for_quadratic_command(self):
        """Test quadratic curve with insufficient points."""
        points = np.array([[10.0, 10.0, 0.0], [20.0, 20.0, 2.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M", "Q"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        with pytest.raises(ValueError, match="Quadratic Bezier command needs 2 points, got 1"):
            PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

    def test_insufficient_points_for_cubic_command(self):
        """Test cubic curve with insufficient points."""
        points = np.array([[10.0, 10.0, 0.0], [20.0, 20.0, 3.0], [25.0, 25.0, 3.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M", "C"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        with pytest.raises(ValueError, match="Cubic Bezier command needs 3 points, got 2"):
            PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

    def test_unknown_command(self):
        """Test handling of unknown command."""
        points = np.array([[10.0, 10.0, 0.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["X"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        # Unknown commands raise KeyError from COMMAND_INFO lookup
        with pytest.raises(KeyError, match="'X'"):
            PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

    def test_2d_points_normalization(self):
        """Test that 2D points are properly normalized to 3D."""
        points = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M", "L"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            # Add z-coordinate of 0.0 and type column
            result = np.column_stack([pts, np.zeros(len(pts))])
            return result

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        assert result_points.shape == (2, 3)
        assert result_points[0, 2] == 0.0  # Z-coordinate should be 0
        assert result_commands == commands

    def test_very_large_number_of_steps(self):
        """Test polygonization with very large number of steps."""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 2.0], [20.0, 0.0, 0.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M", "Q"]
        steps = 1000  # Very large number

        def process_2d_to_3d(pts, cmds):
            return pts

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        # Should have original M point + steps L points
        assert len(result_points) == 1 + steps
        assert result_commands[0] == "M"
        assert all(cmd == "L" for cmd in result_commands[1:])

    def test_negative_steps(self):
        """Test handling of negative steps (should work like positive)."""
        points = np.array([[10.0, 10.0, 0.0], [20.0, 20.0, 0.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M", "L"]
        steps = -5

        def process_2d_to_3d(pts, cmds):
            return pts

        # Negative steps should be treated as positive
        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        np.testing.assert_array_equal(result_points, points)
        assert result_commands == commands

    def test_multiple_curves_in_sequence(self):
        """Test multiple curves without intervening line commands."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M
                [5.0, 10.0, 2.0],  # Q control
                [10.0, 0.0, 0.0],  # Q end
                [12.0, 5.0, 3.0],  # C control1
                [15.0, 5.0, 3.0],  # C control2
                [20.0, 0.0, 0.0],  # C end
            ],
            dtype=np.float64,
        )
        commands: list[AvGlyphCmds] = ["M", "Q", "C"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        # Should have M + 5 L's for Q + 5 L's for C
        assert len(result_points) == 1 + steps + steps
        assert result_commands[0] == "M"
        assert all(cmd == "L" for cmd in result_commands[1:])

    def test_path_with_only_curves(self):
        """Test path consisting only of curve commands (no lines)."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M
                [5.0, 10.0, 2.0],  # Q control
                [10.0, 0.0, 0.0],  # Q end
            ],
            dtype=np.float64,
        )
        commands: list[AvGlyphCmds] = ["M", "Q"]
        steps = 3

        def process_2d_to_3d(pts, cmds):
            return pts

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        assert len(result_points) == 1 + steps
        assert result_commands[0] == "M"
        assert all(cmd == "L" for cmd in result_commands[1:])

    def test_path_with_very_small_coordinates(self):
        """Test polygonization with very small coordinate values."""
        points = np.array(
            [
                [1e-10, 1e-10, 0.0],
                [1e-9, 1e-9, 2.0],
                [1e-10, 1e-9, 0.0],
            ],
            dtype=np.float64,
        )
        commands: list[AvGlyphCmds] = ["M", "Q"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        assert len(result_points) == 1 + steps
        assert not np.any(np.isnan(result_points))
        assert not np.any(np.isinf(result_points))

    def test_path_with_very_large_coordinates(self):
        """Test polygonization with very large coordinate values."""
        points = np.array(
            [
                [1e6, 1e6, 0.0],
                [1e7, 1e7, 2.0],
                [1e6, 1e7, 0.0],
            ],
            dtype=np.float64,
        )
        commands: list[AvGlyphCmds] = ["M", "Q"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        assert len(result_points) == 1 + steps
        assert not np.any(np.isnan(result_points))
        assert not np.any(np.isinf(result_points))

    def test_path_with_nan_coordinates(self):
        """Test handling of NaN coordinates."""
        points = np.array(
            [
                [10.0, 10.0, 0.0],
                [np.nan, 20.0, 2.0],
                [30.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands: list[AvGlyphCmds] = ["M", "Q"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        # Should propagate NaN values
        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        assert len(result_points) == 1 + steps
        assert np.any(np.isnan(result_points))

    def test_path_with_inf_coordinates(self):
        """Test handling of infinite coordinates."""
        points = np.array(
            [
                [10.0, 10.0, 0.0],
                [np.inf, 20.0, 2.0],
                [30.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands: list[AvGlyphCmds] = ["M", "Q"]
        steps = 5

        def process_2d_to_3d(pts, cmds):
            return pts

        # inf values become NaN after bezier calculations
        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        assert len(result_points) == 1 + steps
        # inf values propagate through bezier math and become NaN
        assert np.any(np.isnan(result_points))

    def test_precise_point_count_estimation(self):
        """Test that point count estimation is accurate for buffer sizing."""
        # Create a path with known curve counts
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M
                [10.0, 0.0, 0.0],  # L
                [15.0, 5.0, 2.0],  # Q control
                [20.0, 0.0, 0.0],  # Q end
                [25.0, 5.0, 3.0],  # C control1
                [30.0, 5.0, 3.0],  # C control2
                [35.0, 0.0, 0.0],  # C end
                [40.0, 0.0, 0.0],  # L
            ],
            dtype=np.float64,
        )
        commands: list[AvGlyphCmds] = ["M", "L", "Q", "C", "L"]
        steps = 10

        def process_2d_to_3d(pts, cmds):
            return pts

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        # Expected: M + L + (10 L for Q) + (10 L for C) + L = 14 points
        expected_points = 1 + 1 + steps + steps + 1
        assert len(result_points) == expected_points
        assert len(result_commands) == expected_points

    def test_mixed_2d_and_3d_points(self):
        """Test handling when process_2d_to_3d is needed for some points."""
        # Start with 2D points
        points = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M", "L"]
        steps = 5

        conversion_called = False

        def process_2d_to_3d(pts, cmds):
            nonlocal conversion_called
            conversion_called = True
            # Convert to 3D with type column
            result = np.column_stack([pts, np.zeros(len(pts))])
            return result

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        assert conversion_called
        assert result_points.shape == (2, 3)
        assert result_commands == commands

    def test_steps_parameter_preserved_for_lines(self):
        """Test that steps parameter doesn't affect line commands."""
        points = np.array([[10.0, 10.0, 0.0], [20.0, 20.0, 0.0]], dtype=np.float64)
        commands: list[AvGlyphCmds] = ["M", "L"]
        steps = 100  # Large number

        def process_2d_to_3d(pts, cmds):
            return pts

        result_points, result_commands = PathPolygonizer.polygonize_path(points, commands, steps, process_2d_to_3d)

        # Lines should not be affected by steps
        np.testing.assert_array_equal(result_points, points)
        assert result_commands == commands
