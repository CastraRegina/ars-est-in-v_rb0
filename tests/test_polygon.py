"""Tests for AvPolygon class methods."""

import numpy as np
import pytest

from ave.geom import AvPolygon


class TestAvPolygon:
    """Test class for AvPolygon static methods."""

    def test_area_empty_polygon(self):
        """Test area calculation for empty polygon."""
        points = np.array([], dtype=np.float64).reshape(0, 3)
        assert AvPolygon.area(points) == 0.0

    def test_area_single_point(self):
        """Test area calculation for single point."""
        points = np.array([[1.0, 2.0, 0.0]], dtype=np.float64)
        assert AvPolygon.area(points) == 0.0

    def test_area_two_points(self):
        """Test area calculation for two points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        assert AvPolygon.area(points) == 0.0

    def test_area_triangle(self):
        """Test area calculation for a simple triangle."""
        # Right triangle with vertices (0,0), (2,0), (0,2)
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
        # Area should be 0.5 * base * height = 0.5 * 2 * 2 = 2
        assert AvPolygon.area(points) == 2.0

    def test_area_square(self):
        """Test area calculation for a square."""
        # Unit square
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        assert AvPolygon.area(points) == 1.0

    def test_area_rectangle(self):
        """Test area calculation for a rectangle."""
        # Rectangle 3x2
        points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
        assert AvPolygon.area(points) == 6.0

    def test_area_pentagon(self):
        """Test area calculation for a regular pentagon."""
        # Regular pentagon with circumradius 1
        n_sides = 5
        points = []
        for i in range(n_sides):
            angle = 2 * np.pi * i / n_sides
            points.append([np.cos(angle), np.sin(angle), 0.0])
        points = np.array(points, dtype=np.float64)

        # Area of regular pentagon with circumradius 1
        expected_area = 5 * 0.5 * np.sin(2 * np.pi / 5)
        assert np.isclose(AvPolygon.area(points), expected_area, rtol=1e-10)

    def test_area_clockwise_vs_counterclockwise(self):
        """Test that area is positive regardless of vertex order."""
        # Square vertices
        square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        # Clockwise order (should give same area)
        square_cw = square[::-1]

        assert AvPolygon.area(square) == AvPolygon.area(square_cw) == 1.0

    def test_area_degenerate_polygon(self):
        """Test area calculation for degenerate polygon (collinear points)."""
        # Collinear points on a line
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]], dtype=np.float64)
        assert AvPolygon.area(points) == 0.0

    def test_centroid_empty_polygon(self):
        """Test centroid calculation for empty polygon."""
        points = np.array([], dtype=np.float64).reshape(0, 3)
        assert AvPolygon.centroid(points) == (0.0, 0.0)

    def test_centroid_single_point(self):
        """Test centroid calculation for single point."""
        points = np.array([[3.0, 4.0, 0.0]], dtype=np.float64)
        assert AvPolygon.centroid(points) == (3.0, 4.0)

    def test_centroid_two_points(self):
        """Test centroid calculation for two points."""
        points = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]], dtype=np.float64)
        assert AvPolygon.centroid(points) == (1.0, 2.0)

    def test_centroid_triangle(self):
        """Test centroid calculation for a triangle."""
        # Triangle vertices (0,0), (2,0), (0,2)
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
        # Centroid of triangle is average of vertices
        expected = ((0.0 + 2.0 + 0.0) / 3, (0.0 + 0.0 + 2.0) / 3)
        assert AvPolygon.centroid(points) == expected

    def test_centroid_square(self):
        """Test centroid calculation for a square."""
        # Unit square
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        assert AvPolygon.centroid(points) == (0.5, 0.5)

    def test_centroid_rectangle(self):
        """Test centroid calculation for a rectangle."""
        # Rectangle 3x2
        points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
        assert AvPolygon.centroid(points) == (1.5, 1.0)

    def test_centroid_irregular_polygon(self):
        """Test centroid calculation for an irregular polygon."""
        # L-shaped polygon
        points = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float64,
        )
        centroid = AvPolygon.centroid(points)
        # Should be somewhere in the interior of the L-shape
        assert 0.0 < centroid[0] < 2.0
        assert 0.0 < centroid[1] < 2.0

    def test_centroid_degenerate_polygon(self):
        """Test centroid calculation for degenerate polygon (collinear points)."""
        # Collinear points on a line
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]], dtype=np.float64)
        # Should return mean of points for degenerate case
        expected = (1.5, 1.5)
        assert AvPolygon.centroid(points) == expected

    def test_is_ccw_empty_polygon(self):
        """Test CCW detection for empty polygon."""
        points = np.array([], dtype=np.float64).reshape(0, 3)
        assert not AvPolygon.is_ccw(points)

    def test_is_ccw_single_point(self):
        """Test CCW detection for single point."""
        points = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
        assert not AvPolygon.is_ccw(points)

    def test_is_ccw_two_points(self):
        """Test CCW detection for two points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        assert not AvPolygon.is_ccw(points)

    def test_is_ccw_triangle_counterclockwise(self):
        """Test CCW detection for counter-clockwise triangle."""
        # Triangle in CCW order: (0,0), (1,0), (0,1)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        assert AvPolygon.is_ccw(points)

    def test_is_ccw_triangle_clockwise(self):
        """Test CCW detection for clockwise triangle."""
        # Triangle in CW order: (0,0), (0,1), (1,0)
        points = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        assert not AvPolygon.is_ccw(points)

    def test_is_ccw_square_counterclockwise(self):
        """Test CCW detection for counter-clockwise square."""
        # Square in CCW order
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        assert AvPolygon.is_ccw(points)

    def test_is_ccw_square_clockwise(self):
        """Test CCW detection for clockwise square."""
        # Square in CW order
        points = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        assert not AvPolygon.is_ccw(points)

    def test_is_ccw_regular_polygon_counterclockwise(self):
        """Test CCW detection for regular polygon in CCW order."""
        # Regular pentagon in CCW order
        n_sides = 5
        points = []
        for i in range(n_sides):
            angle = 2 * np.pi * i / n_sides
            points.append([np.cos(angle), np.sin(angle), 0.0])
        points = np.array(points, dtype=np.float64)
        assert AvPolygon.is_ccw(points)

    def test_is_ccw_regular_polygon_clockwise(self):
        """Test CCW detection for regular polygon in CW order."""
        # Regular pentagon in CW order
        n_sides = 5
        points = []
        for i in range(n_sides):
            angle = -2 * np.pi * i / n_sides  # Negative for CW
            points.append([np.cos(angle), np.sin(angle), 0.0])
        points = np.array(points, dtype=np.float64)
        assert not AvPolygon.is_ccw(points)

    def test_is_ccw_degenerate_polygon(self):
        """Test CCW detection for degenerate polygon (collinear points)."""
        # Collinear points
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]], dtype=np.float64)
        assert not AvPolygon.is_ccw(points)

    def test_is_ccw_self_intersecting_polygon(self):
        """Test CCW detection for self-intersecting polygon (bowtie)."""
        # Bowtie shape: self-intersecting
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        # This should return some result (implementation dependent)
        result = AvPolygon.is_ccw(points)
        assert isinstance(result, bool)

    def test_combined_methods_triangle(self):
        """Test all methods together on a triangle."""
        # Right triangle with vertices (0,0), (3,0), (0,4)
        points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float64)

        # Area should be 0.5 * base * height = 0.5 * 3 * 4 = 6
        assert AvPolygon.area(points) == 6.0

        # Centroid should be average of vertices
        expected_centroid = ((0.0 + 3.0 + 0.0) / 3, (0.0 + 0.0 + 4.0) / 3)
        assert AvPolygon.centroid(points) == expected_centroid

        # Should be counter-clockwise
        assert AvPolygon.is_ccw(points)

    def test_combined_methods_clockwise_triangle(self):
        """Test all methods together on a clockwise triangle."""
        # Same triangle but in clockwise order
        points = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)

        # Area should still be positive 6
        assert AvPolygon.area(points) == 6.0

        # Centroid should be the same
        expected_centroid = ((0.0 + 0.0 + 3.0) / 3, (0.0 + 4.0 + 0.0) / 3)
        assert AvPolygon.centroid(points) == expected_centroid

        # Should be clockwise (not CCW)
        assert not AvPolygon.is_ccw(points)

    def test_large_coordinates(self):
        """Test methods with large coordinate values."""
        # Large triangle
        points = np.array([[1000.0, 2000.0, 0.0], [3000.0, 2000.0, 0.0], [1000.0, 4000.0, 0.0]], dtype=np.float64)

        # Area should be 0.5 * base * height = 0.5 * 2000 * 2000 = 2000000
        assert AvPolygon.area(points) == 2000000.0

        # Centroid
        expected_centroid = ((1000.0 + 3000.0 + 1000.0) / 3, (2000.0 + 2000.0 + 4000.0) / 3)
        centroid = AvPolygon.centroid(points)
        assert np.isclose(centroid[0], expected_centroid[0], rtol=1e-12)
        assert np.isclose(centroid[1], expected_centroid[1], rtol=1e-12)

        # Should be counter-clockwise
        assert AvPolygon.is_ccw(points)

    def test_small_coordinates(self):
        """Test methods with very small coordinate values."""
        # Small triangle
        points = np.array([[0.001, 0.002, 0.0], [0.003, 0.002, 0.0], [0.001, 0.004, 0.0]], dtype=np.float64)

        # Area should be 0.5 * base * height = 0.5 * 0.002 * 0.002 = 0.000002
        expected_area = 0.5 * 0.002 * 0.002
        assert np.isclose(AvPolygon.area(points), expected_area, rtol=1e-10)

        # Centroid
        expected_centroid = ((0.001 + 0.003 + 0.001) / 3, (0.002 + 0.002 + 0.004) / 3)
        assert np.allclose(AvPolygon.centroid(points), expected_centroid, rtol=1e-10)

        # Should be counter-clockwise
        assert AvPolygon.is_ccw(points)

    def test_negative_coordinates(self):
        """Test methods with negative coordinate values."""
        # Triangle with negative coordinates
        points = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=np.float64)

        # Area should be 0.5 * base * height = 0.5 * 2 * 2 = 2
        assert AvPolygon.area(points) == 2.0

        # Centroid should be at (-1/3, -1/3)
        expected_centroid = ((-1.0 + 1.0 + -1.0) / 3, (-1.0 + -1.0 + 1.0) / 3)
        assert AvPolygon.centroid(points) == expected_centroid

        # Should be counter-clockwise
        assert AvPolygon.is_ccw(points)
