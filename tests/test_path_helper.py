"""Tests for path helper functions"""

import numpy as np
import pytest

from ave.path import AvPath
from ave.path_helper import AvPathCleaner


class TestAvPathCleaner:
    """Test cases for AvPathCleaner class"""

    def test_resolve_path_intersections_simple_square(self):
        """Test resolving intersections with a simple square"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_path_intersections(path)

        # Should return a valid path (no intersections to resolve)
        assert isinstance(result, AvPath)
        assert len(result.points) >= 4  # At least the original points
        assert result.commands[-1] == "Z"  # Should be closed

    def test_resolve_path_intersections_self_intersecting_figure_eight(self):
        """Test resolving intersections with a self-intersecting figure-eight"""
        # Create a figure-eight path that self-intersects
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Start
                [5.0, 5.0, 0.0],  # Center intersection point
                [10.0, 0.0, 0.0],  # Right
                [5.0, -5.0, 0.0],  # Bottom intersection
                [0.0, 0.0, 0.0],  # Back to start
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "L"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_path_intersections(path)

        # Should return a valid path after resolving intersections
        assert isinstance(result, AvPath)
        # The result should be a proper path (may be split into multiple contours)
        assert len(result.points) > 0

    def test_resolve_path_intersections_open_path(self):
        """Test resolving intersections with an open path (no closing)"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L"]  # No closing 'Z'
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_path_intersections(path)

        # Open paths should be handled gracefully
        assert isinstance(result, AvPath)

    def test_resolve_path_intersections_empty_path(self):
        """Test resolving intersections with an empty path"""
        points = np.array([], dtype=np.float64).reshape(0, 3)
        commands = []
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_path_intersections(path)

        # Empty path should return empty path
        assert isinstance(result, AvPath)
        assert len(result.points) == 0
        assert len(result.commands) == 0

    def test_resolve_path_intersections_degenerate_polygon(self):
        """Test resolving intersections with a degenerate polygon (collinear points)"""
        # Create a degenerate "polygon" with collinear points
        points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_path_intersections(path)

        # Should handle degenerate cases gracefully
        assert isinstance(result, AvPath)

    def test_resolve_path_intersections_multiple_contours(self):
        """Test resolving intersections with multiple separate contours"""
        # Create two separate squares
        points = np.array(
            [
                # First square
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                # Move to second square
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [30.0, 10.0, 0.0],
                [20.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_path_intersections(path)

        # Should handle multiple contours
        assert isinstance(result, AvPath)
        assert len(result.points) > 0
        # Should still have closed contours
        assert "Z" in result.commands
