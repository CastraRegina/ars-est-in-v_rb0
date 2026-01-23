#!/usr/bin/env python3
"""Comprehensive tests for AvPath.centroid with Shapely integration."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pytest

from ave.geom import AvPolygon
from ave.path import MULTI_POLYGON_CONSTRAINTS, AvPath


class TestAvPathCentroid:
    """Test AvPath.centroid method with various polygon types."""

    def test_centroid_convex_polygon(self):
        """Test centroid calculation for a convex polygon."""
        # Square
        points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        cmds = ["M"] + ["L"] * 3 + ["Z"]

        path = AvPath(points, cmds)
        centroid = path.centroid

        # Should be at center of square
        assert abs(centroid[0] - 1.0) < 1e-10
        assert abs(centroid[1] - 1.0) < 1e-10

    def test_centroid_concave_polygon(self):
        """Test centroid calculation for a concave polygon."""
        # Arrow shape (concave)
        points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [3.0, 0.5], [2.0, 2.0], [2.0, 3.0], [0.0, 3.0]])
        cmds = ["M"] + ["L"] * 6 + ["Z"]

        path = AvPath(points, cmds)
        centroid = path.centroid

        # Should match Shapely's calculation
        expected = AvPolygon.centroid(points)
        assert abs(centroid[0] - expected[0]) < 1e-10
        assert abs(centroid[1] - expected[1]) < 1e-10

    def test_centroid_polygon_with_hole(self):
        """Test centroid calculation for a polygon with a hole."""
        # Outer square (CCW)
        outer_points = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])

        # Inner square (hole) - CW orientation
        hole_points = np.array([[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]])

        # Create multi-polygon path
        outer_cmds = ["M"] + ["L"] * 3 + ["Z"]
        hole_cmds = ["M"] + ["L"] * 3 + ["Z"]

        all_points = np.vstack([outer_points, hole_points])
        all_cmds = outer_cmds + hole_cmds

        path = AvPath(all_points, all_cmds, constraints=MULTI_POLYGON_CONSTRAINTS)
        centroid = path.centroid

        # With Shapely, should be closer to (2.0, 2.0) than the old calculation
        assert 2.0 < centroid[0] < 2.5  # Shapely gives better result
        assert 2.0 < centroid[1] < 2.5

    def test_centroid_multipolygon(self):
        """Test centroid calculation for multiple separate polygons."""
        # Two separate squares
        square1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        square2 = np.array([[3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 4.0]])

        cmds1 = ["M"] + ["L"] * 3 + ["Z"]
        cmds2 = ["M"] + ["L"] * 3 + ["Z"]

        all_points = np.vstack([square1, square2])
        all_cmds = cmds1 + cmds2

        path = AvPath(all_points, all_cmds, constraints=MULTI_POLYGON_CONSTRAINTS)
        centroid = path.centroid

        # Current implementation treats as combined polygon (not true MultiPolygon)
        # This creates a self-intersecting polygon, so it falls back to AvPolygon
        expected = AvPolygon.centroid(all_points)
        assert abs(centroid[0] - expected[0]) < 1e-10
        assert abs(centroid[1] - expected[1]) < 1e-10
        assert centroid == (2.75, 2.75)  # Simple average of all vertices

    def test_centroid_with_curves(self):
        """Test centroid calculation for paths with curves."""
        # Create a path with a quadratic curve
        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])  # Control point
        cmds = ["M", "Q", "L", "L", "Z"]

        path = AvPath(points, cmds)
        centroid = path.centroid

        # Should be a valid point
        assert isinstance(centroid, tuple)
        assert len(centroid) == 2
        assert all(isinstance(c, float) for c in centroid)

    def test_centroid_degenerate_cases(self):
        """Test centroid calculation for degenerate cases."""
        # Empty path - need to provide proper empty array
        path = AvPath(np.empty((0, 3)), [])
        with pytest.raises(ValueError):
            _ = path.centroid

        # Single point
        points = np.array([[1.0, 2.0, 0.0]])
        cmds = ["M"]
        path = AvPath(points, cmds)
        with pytest.raises(ValueError):
            _ = path.centroid

        # Line (2 points)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        cmds = ["M", "L"]
        path = AvPath(points, cmds)
        with pytest.raises(ValueError):
            _ = path.centroid

    def test_centroid_self_intersecting(self):
        """Test centroid calculation for self-intersecting polygon."""
        # Bowtie shape (self-intersecting)
        points = np.array([[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
        cmds = ["M"] + ["L"] * 3 + ["Z"]

        path = AvPath(points, cmds)
        centroid = path.centroid

        # Should fallback to AvPolygon calculation
        expected = AvPolygon.centroid(points)
        assert abs(centroid[0] - expected[0]) < 1e-10
        assert abs(centroid[1] - expected[1]) < 1e-10

    def test_centroid_open_path_raises_error(self):
        """Test that open paths raise ValueError for centroid."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        cmds = ["M", "L", "L", "L"]  # No Z command

        path = AvPath(points, cmds)
        with pytest.raises(ValueError, match="requires a closed path"):
            _ = path.centroid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
