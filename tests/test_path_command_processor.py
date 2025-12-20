"""Tests for PathCommandProcessor integration in path.py methods."""

import numpy as np
import pytest

from ave.path import (
    COMMAND_INFO,
    MULTI_POLYLINE_CONSTRAINTS,
    SINGLE_PATH_CONSTRAINTS,
    AvPath,
    PathCommandInfo,
    PathCommandProcessor,
)


class TestPathCommandInfo:
    """Tests for PathCommandInfo dataclass."""

    def test_command_info_immutable(self):
        """PathCommandInfo should be frozen (immutable)."""
        info = PathCommandInfo(1, False, True)
        with pytest.raises(Exception):  # FrozenInstanceError
            info.consumes_points = 2

    def test_command_info_default_is_drawing(self):
        """PathCommandInfo should default is_drawing to True."""
        info = PathCommandInfo(1, False)
        assert info.is_drawing is True


class TestPathCommandInfoRegistry:
    """Tests for COMMAND_INFO registry."""

    def test_all_commands_registered(self):
        """All SVG path commands should be in the registry."""
        expected_commands = {"M", "L", "Q", "C", "Z"}
        assert set(COMMAND_INFO.keys()) == expected_commands

    def test_point_consumption_values(self):
        """Verify correct point consumption for each command."""
        assert COMMAND_INFO["M"].consumes_points == 1
        assert COMMAND_INFO["L"].consumes_points == 1
        assert COMMAND_INFO["Q"].consumes_points == 2
        assert COMMAND_INFO["C"].consumes_points == 3
        assert COMMAND_INFO["Z"].consumes_points == 0

    def test_curve_flags(self):
        """Verify correct curve flags for each command."""
        assert COMMAND_INFO["M"].is_curve is False
        assert COMMAND_INFO["L"].is_curve is False
        assert COMMAND_INFO["Q"].is_curve is True
        assert COMMAND_INFO["C"].is_curve is True
        assert COMMAND_INFO["Z"].is_curve is False


class TestPathCommandProcessor:
    """Tests for PathCommandProcessor static methods."""

    def test_get_point_consumption(self):
        """get_point_consumption should return correct values."""
        assert PathCommandProcessor.get_point_consumption("M") == 1
        assert PathCommandProcessor.get_point_consumption("L") == 1
        assert PathCommandProcessor.get_point_consumption("Q") == 2
        assert PathCommandProcessor.get_point_consumption("C") == 3
        assert PathCommandProcessor.get_point_consumption("Z") == 0

    def test_is_curve_command(self):
        """is_curve_command should correctly identify curves."""
        assert PathCommandProcessor.is_curve_command("M") is False
        assert PathCommandProcessor.is_curve_command("L") is False
        assert PathCommandProcessor.is_curve_command("Q") is True
        assert PathCommandProcessor.is_curve_command("C") is True
        assert PathCommandProcessor.is_curve_command("Z") is False

    def test_is_drawing_command(self):
        """is_drawing_command should correctly identify drawing commands."""
        assert PathCommandProcessor.is_drawing_command("M") is False  # MoveTo is not drawing
        assert PathCommandProcessor.is_drawing_command("L") is True
        assert PathCommandProcessor.is_drawing_command("Q") is True
        assert PathCommandProcessor.is_drawing_command("C") is True
        assert PathCommandProcessor.is_drawing_command("Z") is True

    def test_validate_command_sequence_valid(self):
        """validate_command_sequence should pass for valid sequences."""
        commands = ["M", "L", "L", "Z"]
        points = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)
        # Should not raise
        PathCommandProcessor.validate_command_sequence(commands, points)

    def test_validate_command_sequence_with_curves(self):
        """validate_command_sequence should handle curves correctly."""
        commands = ["M", "Q", "C", "Z"]
        # M(1) + Q(2) + C(3) = 6 points
        points = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 2], [5, 0]], dtype=np.float64)
        # Should not raise
        PathCommandProcessor.validate_command_sequence(commands, points)

    def test_validate_command_sequence_insufficient_points(self):
        """validate_command_sequence should fail for insufficient points."""
        commands = ["M", "L", "L", "Z"]
        points = np.array([[0, 0], [1, 0]], dtype=np.float64)  # Need 3 points
        with pytest.raises(ValueError):
            PathCommandProcessor.validate_command_sequence(commands, points)


class TestPolygonize:
    """Tests for AvPath.polygonize() method."""

    def test_polygonize_no_curves(self):
        """Polygonize should return self when no curves present."""
        points = [(0, 0), (10, 0), (10, 10)]
        commands = ["M", "L", "L", "Z"]
        path = AvPath(points, commands)
        result = path.polygonize(10)
        assert result is path  # Should return same object

    def test_polygonize_steps_zero(self):
        """Polygonize with steps=0 should return self."""
        points = [(0, 0), (5, 5), (10, 0)]
        commands = ["M", "Q", "Z"]
        path = AvPath(points, commands)
        result = path.polygonize(0)
        assert result is path

    def test_polygonize_quadratic_bezier(self):
        """Polygonize should convert quadratic bezier to lines."""
        # M(1) + Q(2) = 3 points
        points = [(0, 0), (5, 10), (10, 0)]
        commands = ["M", "Q", "Z"]
        path = AvPath(points, commands)
        result = path.polygonize(5)

        # Should have more points than original
        assert len(result.points) >= len(path.points)
        # Should have no curve commands
        assert "Q" not in result.commands
        assert "C" not in result.commands
        # Should start with M and have L commands
        assert result.commands[0] == "M"
        assert "L" in result.commands

    def test_polygonize_cubic_bezier(self):
        """Polygonize should convert cubic bezier to lines."""
        # M(1) + C(3) = 4 points
        points = [(0, 0), (3, 10), (7, 10), (10, 0)]
        commands = ["M", "C", "Z"]
        path = AvPath(points, commands)
        result = path.polygonize(5)

        # Should have no curve commands
        assert "Q" not in result.commands
        assert "C" not in result.commands
        # Should have constraints for polyline
        assert result.constraints == MULTI_POLYLINE_CONSTRAINTS

    def test_polygonize_mixed_commands(self):
        """Polygonize should handle mixed L, Q, C commands."""
        # M(1) + L(1) + Q(2) + C(3) = 7 points
        points = [(0, 0), (5, 0), (7, 5), (10, 0), (12, 5), (15, 5), (17, 0)]
        commands = ["M", "L", "Q", "C", "Z"]
        path = AvPath(points, commands)
        result = path.polygonize(3)

        # Should have no curve commands after polygonization
        assert not any(PathCommandProcessor.is_curve_command(cmd) for cmd in result.commands)

    def test_polygonize_preserves_structure(self):
        """Polygonize should preserve path structure (M at start, Z at end if closed)."""
        points = [(0, 0), (5, 10), (10, 0)]
        commands = ["M", "Q", "Z"]
        path = AvPath(points, commands)
        result = path.polygonize(5)

        # Structure should be preserved
        assert result.commands[0] == "M"
        assert result.commands[-1] == "Z"

    def test_polygonize_multiple_segments(self):
        """Polygonize should handle multiple segments."""
        # Segment 1: M(1) + Q(2) + Z(0) = 3 points
        # Segment 2: M(1) + L(1) + Z(0) = 2 points
        points = [(0, 0), (5, 10), (10, 0), (20, 0), (20, 10)]
        commands = ["M", "Q", "Z", "M", "L", "Z"]
        path = AvPath(points, commands)
        result = path.polygonize(3)

        # Should have two M commands (two segments)
        assert result.commands.count("M") == 2


class TestSplitIntoSinglePaths:
    """Tests for AvPath.split_into_single_paths() method."""

    def test_split_empty_path(self):
        """Splitting empty path should return empty list."""
        path = AvPath()
        result = path.split_into_single_paths()
        assert result == []

    def test_split_single_segment(self):
        """Splitting single segment path should return one path."""
        points = [(0, 0), (10, 0), (10, 10)]
        commands = ["M", "L", "L", "Z"]
        path = AvPath(points, commands)
        result = path.split_into_single_paths()

        assert len(result) == 1
        assert result[0].commands == ["M", "L", "L", "Z"]
        assert len(result[0].points) == 3

    def test_split_two_segments(self):
        """Splitting two segment path should return two paths."""
        # Segment 1: M(1) + L(1) + L(1) + Z(0) = 3 points
        # Segment 2: M(1) + L(1) + L(1) + Z(0) = 3 points
        points = [(0, 0), (1, 0), (1, 1), (10, 10), (11, 10), (11, 11)]
        commands = ["M", "L", "L", "Z", "M", "L", "L", "Z"]
        path = AvPath(points, commands)
        result = path.split_into_single_paths()

        assert len(result) == 2
        assert result[0].commands == ["M", "L", "L", "Z"]
        assert result[1].commands == ["M", "L", "L", "Z"]

    def test_split_with_quadratic_bezier(self):
        """Splitting should preserve quadratic bezier commands."""
        # M(1) + Q(2) + Z(0) = 3 points
        points = [(0, 0), (5, 10), (10, 0)]
        commands = ["M", "Q", "Z"]
        path = AvPath(points, commands)
        result = path.split_into_single_paths()

        assert len(result) == 1
        assert result[0].commands == ["M", "Q", "Z"]
        assert len(result[0].points) == 3

    def test_split_with_cubic_bezier(self):
        """Splitting should preserve cubic bezier commands."""
        # M(1) + C(3) + Z(0) = 4 points
        points = [(0, 0), (3, 10), (7, 10), (10, 0)]
        commands = ["M", "C", "Z"]
        path = AvPath(points, commands)
        result = path.split_into_single_paths()

        assert len(result) == 1
        assert result[0].commands == ["M", "C", "Z"]
        assert len(result[0].points) == 4

    def test_split_mixed_segments(self):
        """Splitting should handle segments with different command types."""
        # Segment 1: M(1) + Q(2) + Z(0) = 3 points
        # Segment 2: M(1) + C(3) + Z(0) = 4 points
        points = [(0, 0), (5, 10), (10, 0), (20, 0), (23, 10), (27, 10), (30, 0)]
        commands = ["M", "Q", "Z", "M", "C", "Z"]
        path = AvPath(points, commands)
        result = path.split_into_single_paths()

        assert len(result) == 2
        assert result[0].commands == ["M", "Q", "Z"]
        assert len(result[0].points) == 3
        assert result[1].commands == ["M", "C", "Z"]
        assert len(result[1].points) == 4

    def test_split_constraints(self):
        """Split paths should have SINGLE_PATH_CONSTRAINTS."""
        points = [(0, 0), (1, 0), (1, 1)]
        commands = ["M", "L", "L", "Z"]
        path = AvPath(points, commands)
        result = path.split_into_single_paths()

        assert len(result) == 1
        assert result[0].constraints == SINGLE_PATH_CONSTRAINTS


class TestPathCommandProcessorIntegration:
    """Integration tests verifying PathCommandProcessor is properly used."""

    def test_consumed_variable_used_in_polygonize(self):
        """Verify consumed variable is actually used in polygonize."""
        # This test verifies the fix for the unused 'consumed' variable issue
        # M(1) + Q(2) + L(1) + C(3) = 7 points
        points = [(0, 0), (2, 5), (4, 0), (6, 0), (8, 5), (10, 5), (12, 0)]
        commands = ["M", "Q", "L", "C", "Z"]
        path = AvPath(points, commands)

        # Should not raise - verifies point counting works correctly
        result = path.polygonize(3)

        # Verify structure is correct
        assert result.commands[0] == "M"
        assert result.commands[-1] == "Z"
        assert not any(PathCommandProcessor.is_curve_command(cmd) for cmd in result.commands)

    def test_consumed_variable_used_in_split(self):
        """Verify consumed variable is actually used in split_into_single_paths."""
        # Multi-segment with various command types
        # Segment 1: M(1) + Q(2) = 3 points
        # Segment 2: M(1) + C(3) = 4 points
        points = [(0, 0), (5, 10), (10, 0), (20, 0), (23, 10), (27, 10), (30, 0)]
        commands = ["M", "Q", "Z", "M", "C", "Z"]
        path = AvPath(points, commands)

        result = path.split_into_single_paths()

        # Verify correct splitting
        assert len(result) == 2
        # First segment should have 3 points (M + Q)
        assert len(result[0].points) == 3
        # Second segment should have 4 points (M + C)
        assert len(result[1].points) == 4
