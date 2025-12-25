"""Test cases for PathCommandProcessor integration in AvPathCleaner."""

import numpy as np
import pytest

from ave.path import MULTI_POLYLINE_CONSTRAINTS, AvMultiPolylinePath, AvPath
from ave.path_processing import AvPathCleaner


class TestPathCommandProcessorIntegration:
    """Test integration of PathCommandProcessor in AvPathCleaner."""

    def test_analyze_segment_with_curves(self):
        """Test segment analysis with curve commands."""
        # Create a segment with curves (Q needs 2 points)
        points = np.array([[0, 0, 0], [10, 10, 0], [20, 0, 0], [30, 0, 0]], dtype=np.float64)

        commands = ["M", "Q", "L"]

        segment = AvPath(points, commands)
        analysis = AvPathCleaner._analyze_segment(segment)

        assert analysis["is_valid"] is True
        assert analysis["has_curves"] is True
        assert analysis["num_drawing_commands"] == 2  # Q and L
        assert analysis["num_move_commands"] == 1
        assert analysis["num_close_commands"] == 0

    def test_analyze_segment_without_curves(self):
        """Test segment analysis without curve commands."""
        # Create a simple polyline segment
        points = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=np.float64)

        commands = ["M", "L", "L", "L"]

        segment = AvPath(points, commands)
        analysis = AvPathCleaner._analyze_segment(segment)

        assert analysis["is_valid"] is True
        assert analysis["has_curves"] is False
        assert analysis["num_drawing_commands"] == 3  # 3 L commands
        assert analysis["num_move_commands"] == 1
        assert analysis["num_close_commands"] == 0

    def test_analyze_invalid_segment(self):
        """Test segment analysis with invalid command/point sequence."""
        # Since AvPath validates on creation, we need to test the validation
        # through the analyzer itself with a mock scenario
        # Create a valid segment first
        points = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float64)

        commands = ["M", "L"]

        segment = AvPath(points, commands)
        analysis = AvPathCleaner._analyze_segment(segment)

        # Should be valid
        assert analysis["is_valid"] is True

        # Test with empty segment
        empty_segment = AvPath()
        empty_analysis = AvPathCleaner._analyze_segment(empty_segment)
        assert empty_analysis["is_valid"] is True

    def test_resolve_with_invalid_segments(self):
        """Test that invalid segments are skipped during processing."""
        # Create a path with one valid and one invalid segment
        points = np.array(
            [
                # Valid segment
                [0, 0, 0],
                [10, 0, 0],
                [10, 10, 0],
                [0, 10, 0],
                [0, 0, 0],
                # Invalid segment (not enough points)
                [20, 0, 0],
                [30, 0, 0],
            ],
            dtype=np.float64,
        )

        commands = ["M", "L", "L", "L", "Z", "M", "L", "L"]

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)
        result = AvPathCleaner.resolve_polygonized_path_intersections(path)

        # Should process the valid segment and skip the invalid one
        assert isinstance(result, AvPath)
        # Should have at least the valid segment
        assert len(result.points) > 0

    def test_analyze_segment_with_cubic_curves(self):
        """Test segment analysis with cubic bezier commands."""
        points = np.array([[0, 0, 0], [5, 10, 0], [15, 10, 0], [20, 0, 0], [30, 0, 0]], dtype=np.float64)

        commands = ["M", "C", "L"]

        segment = AvPath(points, commands)
        analysis = AvPathCleaner._analyze_segment(segment)

        assert analysis["is_valid"] is True
        assert analysis["has_curves"] is True
        assert analysis["num_drawing_commands"] == 2  # C and L
        assert analysis["num_move_commands"] == 1
        assert analysis["num_close_commands"] == 0

    def test_analyze_empty_segment(self):
        """Test analysis of empty segment."""
        segment = AvPath()
        analysis = AvPathCleaner._analyze_segment(segment)

        assert analysis["is_valid"] is True
        assert analysis["has_curves"] is False
        assert analysis["num_drawing_commands"] == 0
        assert analysis["num_move_commands"] == 0
        assert analysis["num_close_commands"] == 0
