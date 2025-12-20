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

    # Tests for ray_casting_single method
    def test_ray_casting_single_input_validation(self):
        """Test input validation for ray casting."""
        # Empty polygon
        points = np.array([], dtype=np.float64).reshape(0, 3)
        assert not AvPolygon.ray_casting_single(points, (0.5, 0.5))

        # Invalid point inputs
        square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="Point must be a tuple or list of 2 numeric values"):
            AvPolygon.ray_casting_single(square, (0.5,))  # Too few coordinates

        with pytest.raises(ValueError, match="Point must be a tuple or list of 2 numeric values"):
            AvPolygon.ray_casting_single(square, (0.5, 0.5, 0.5))  # Too many coordinates

        with pytest.raises(ValueError, match="Point must be a tuple or list of 2 numeric values"):
            AvPolygon.ray_casting_single(square, "invalid")  # Wrong type

    def test_ray_casting_single_unit_square(self):
        """Test ray casting on unit square."""
        # Unit square vertices
        square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        # Points inside square
        assert AvPolygon.ray_casting_single(square, (0.5, 0.5))
        assert AvPolygon.ray_casting_single(square, (0.1, 0.1))
        assert AvPolygon.ray_casting_single(square, (0.9, 0.9))

        # Points outside square
        assert not AvPolygon.ray_casting_single(square, (-0.1, 0.5))
        assert not AvPolygon.ray_casting_single(square, (1.1, 0.5))
        assert not AvPolygon.ray_casting_single(square, (0.5, -0.1))
        assert not AvPolygon.ray_casting_single(square, (0.5, 1.1))

    def test_ray_casting_single_triangle(self):
        """Test ray casting on triangle."""
        # Right triangle with vertices (0,0), (2,0), (0,2)
        triangle = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)

        # Points inside triangle
        assert AvPolygon.ray_casting_single(triangle, (0.5, 0.5))
        assert AvPolygon.ray_casting_single(triangle, (0.1, 0.1))
        assert AvPolygon.ray_casting_single(triangle, (1.0, 0.5))

        # Points outside triangle
        assert not AvPolygon.ray_casting_single(triangle, (1.5, 1.5))
        assert not AvPolygon.ray_casting_single(triangle, (-0.5, 0.5))
        assert not AvPolygon.ray_casting_single(triangle, (0.5, -0.5))

    def test_ray_casting_single_edge_cases(self):
        """Test ray casting on edge cases."""
        # Unit square
        square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        # Points on edges (implementation dependent, but should be consistent)
        edge_points = [(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)]
        for point in edge_points:
            result = AvPolygon.ray_casting_single(square, point)
            assert isinstance(result, bool)  # Should return a boolean

    def test_ray_casting_single_vertices(self):
        """Test ray casting on polygon vertices."""
        # Unit square
        square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        # Test each vertex
        vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        for vertex in vertices:
            result = AvPolygon.ray_casting_single(square, vertex)
            assert isinstance(result, bool)  # Should return a boolean

    def test_ray_casting_single_complex_polygon(self):
        """Test ray casting on complex polygon."""
        # Octagon
        n_sides = 8
        points = []
        for i in range(n_sides):
            angle = 2 * np.pi * i / n_sides
            points.append([np.cos(angle), np.sin(angle), 0.0])
        octagon = np.array(points, dtype=np.float64)

        # Center should be inside
        assert AvPolygon.ray_casting_single(octagon, (0.0, 0.0))

        # Point far outside should be outside
        assert not AvPolygon.ray_casting_single(octagon, (2.0, 2.0))

    def test_ray_casting_single_concave_polygon(self):
        """Test ray casting on concave polygon."""
        # Concave L-shaped polygon
        concave = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float64,
        )

        # Points in the solid parts of the L-shape
        assert AvPolygon.ray_casting_single(concave, (0.5, 0.5))  # Lower-left area
        assert AvPolygon.ray_casting_single(concave, (1.5, 0.5))  # Lower-right area
        assert AvPolygon.ray_casting_single(concave, (0.5, 1.5))  # Upper-left area

        # Point in the "hole" (concave part - upper-right missing area)
        assert not AvPolygon.ray_casting_single(concave, (1.5, 1.5))

        # Points outside the entire shape
        assert not AvPolygon.ray_casting_single(concave, (2.5, 1.0))
        assert not AvPolygon.ray_casting_single(concave, (0.5, 2.5))

    def test_ray_casting_single_clockwise_vs_counterclockwise(self):
        """Test that ray casting works regardless of vertex order."""
        # Square vertices
        square_ccw = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        square_cw = square_ccw[::-1]  # Reverse order

        test_point = (0.5, 0.5)

        # Should give same result regardless of orientation
        result_ccw = AvPolygon.ray_casting_single(square_ccw, test_point)
        result_cw = AvPolygon.ray_casting_single(square_cw, test_point)
        assert result_ccw == result_cw == True

    def test_ray_casting_single_degenerate_cases(self):
        """Test ray casting on degenerate polygons."""
        # Single point
        single_point = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
        assert not AvPolygon.ray_casting_single(single_point, (1.0, 1.0))
        assert not AvPolygon.ray_casting_single(single_point, (2.0, 2.0))

        # Line segment (2 points)
        line = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        assert not AvPolygon.ray_casting_single(line, (0.5, 0.5))
        assert not AvPolygon.ray_casting_single(line, (2.0, 2.0))

        # Collinear points
        collinear = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float64)
        assert not AvPolygon.ray_casting_single(collinear, (1.0, 1.0))

    def test_ray_casting_single_large_coordinates(self):
        """Test ray casting with large coordinate values."""
        # Large square
        large_square = np.array(
            [[1000.0, 1000.0, 0.0], [2000.0, 1000.0, 0.0], [2000.0, 2000.0, 0.0], [1000.0, 2000.0, 0.0]],
            dtype=np.float64,
        )

        # Point inside large square
        assert AvPolygon.ray_casting_single(large_square, (1500.0, 1500.0))

        # Point outside large square
        assert not AvPolygon.ray_casting_single(large_square, (500.0, 1500.0))

    def test_ray_casting_single_small_coordinates(self):
        """Test ray casting with very small coordinate values."""
        # Small square
        small_square = np.array(
            [[0.001, 0.001, 0.0], [0.002, 0.001, 0.0], [0.002, 0.002, 0.0], [0.001, 0.002, 0.0]], dtype=np.float64
        )

        # Point inside small square
        assert AvPolygon.ray_casting_single(small_square, (0.0015, 0.0015))

        # Point outside small square
        assert not AvPolygon.ray_casting_single(small_square, (0.003, 0.0015))

    def test_ray_casting_single_negative_coordinates(self):
        """Test ray casting with negative coordinate values."""
        # Square in negative quadrant
        neg_square = np.array(
            [[-2.0, -2.0, 0.0], [-1.0, -2.0, 0.0], [-1.0, -1.0, 0.0], [-2.0, -1.0, 0.0]], dtype=np.float64
        )

        # Point inside negative square
        assert AvPolygon.ray_casting_single(neg_square, (-1.5, -1.5))

        # Point outside negative square
        assert not AvPolygon.ray_casting_single(neg_square, (0.0, -1.5))

    def test_ray_casting_single_2d_points(self):
        """Test ray casting with 2D points (without type column)."""
        # 2D square
        square_2d = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)

        # Should work the same as 3D points
        assert AvPolygon.ray_casting_single(square_2d, (0.5, 0.5))
        assert not AvPolygon.ray_casting_single(square_2d, (1.5, 0.5))

    def test_ray_casting_single_performance_consistency(self):
        """Test that ray casting gives consistent results across multiple calls."""
        square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        test_point = (0.5, 0.5)

        # Multiple calls should give same result
        results = [AvPolygon.ray_casting_single(square, test_point) for _ in range(10)]
        assert all(results)  # All should be True

    def test_ray_casting_single_numeric_types(self):
        """Test ray casting with different numeric types for points."""
        square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        # Test with different numeric types
        assert AvPolygon.ray_casting_single(square, (0.5, 0.5))  # float
        assert AvPolygon.ray_casting_single(square, (0, 0))  # int
        assert AvPolygon.ray_casting_single(square, (0.5, 0))  # mixed
        assert AvPolygon.ray_casting_single(square, [0.5, 0.5])  # list instead of tuple

    # Tests for interior_point_scanlines method
    def test_interior_point_scanlines_degenerate_cases(self):
        """Test interior point finding on degenerate polygons."""
        # Empty polygon
        points = np.array([], dtype=np.float64).reshape(0, 3)
        assert AvPolygon.interior_point_scanlines(points) is None

        # Single point
        points = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
        assert AvPolygon.interior_point_scanlines(points) is None

        # Two points
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        assert AvPolygon.interior_point_scanlines(points) is None

        # Collinear points (horizontal line)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        assert AvPolygon.interior_point_scanlines(points) is None

        # Collinear points (vertical line)
        points = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
        assert AvPolygon.interior_point_scanlines(points) is None

    def test_interior_point_scanlines_triangle(self):
        """Test interior point finding on a triangle."""
        # Right triangle with vertices (0,0), (2,0), (0,2)
        triangle = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)

        point = AvPolygon.interior_point_scanlines(triangle)
        assert point is not None

        # Point should be inside triangle
        assert AvPolygon.ray_casting_single(triangle, point)

        # Point should be within triangle bounds
        assert 0.0 <= point[0] <= 2.0
        assert 0.0 <= point[1] <= 2.0
        assert point[0] + point[1] <= 2.0  # Below hypotenuse

    def test_interior_point_scanlines_square(self):
        """Test interior point finding on a square."""
        # Unit square
        square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        point = AvPolygon.interior_point_scanlines(square)
        assert point is not None

        # Point should be inside square
        assert AvPolygon.ray_casting_single(square, point)

        # Point should be within square bounds
        assert 0.0 < point[0] < 1.0
        assert 0.0 < point[1] < 1.0

    def test_interior_point_scanlines_rectangle(self):
        """Test interior point finding on a rectangle."""
        # Rectangle 3x2
        rectangle = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)

        point = AvPolygon.interior_point_scanlines(rectangle)
        assert point is not None

        # Point should be inside rectangle
        assert AvPolygon.ray_casting_single(rectangle, point)

        # Point should be within rectangle bounds
        assert 0.0 < point[0] < 3.0
        assert 0.0 < point[1] < 2.0

    def test_interior_point_scanlines_concave_polygon(self):
        """Test interior point finding on a concave polygon."""
        # Concave L-shaped polygon
        concave = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float64,
        )

        point = AvPolygon.interior_point_scanlines(concave)
        assert point is not None

        # Point should be inside concave polygon
        assert AvPolygon.ray_casting_single(concave, point)

        # Point should be within overall bounds
        assert 0.0 < point[0] < 2.0
        assert 0.0 < point[1] < 2.0

    def test_interior_point_scanlines_regular_polygon(self):
        """Test interior point finding on regular polygons."""
        # Regular pentagon
        n_sides = 5
        points = []
        for i in range(n_sides):
            angle = 2 * np.pi * i / n_sides
            points.append([np.cos(angle), np.sin(angle), 0.0])
        pentagon = np.array(points, dtype=np.float64)

        point = AvPolygon.interior_point_scanlines(pentagon)
        assert point is not None

        # Point should be inside pentagon
        assert AvPolygon.ray_casting_single(pentagon, point)

        # Point should be near center for regular polygon
        assert abs(point[0]) < 1.0  # Should be within polygon bounds
        assert abs(point[1]) < 1.0

    def test_interior_point_scanlines_parameters(self):
        """Test interior point finding with different parameters."""
        square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        # Test with different sample counts
        for samples in [1, 3, 5, 10, 20]:
            point = AvPolygon.interior_point_scanlines(square, samples=samples)
            assert point is not None
            assert AvPolygon.ray_casting_single(square, point)

        # Test with different epsilon values
        for epsilon in [1e-6, 1e-9, 1e-12]:
            point = AvPolygon.interior_point_scanlines(square, epsilon=epsilon)
            assert point is not None
            assert AvPolygon.ray_casting_single(square, point)

    def test_interior_point_scanlines_clockwise_vs_counterclockwise(self):
        """Test that interior point finding works regardless of vertex order."""
        # Square vertices
        square_ccw = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        square_cw = square_ccw[::-1]  # Reverse order

        point_ccw = AvPolygon.interior_point_scanlines(square_ccw)
        point_cw = AvPolygon.interior_point_scanlines(square_cw)

        # Both should find valid interior points
        assert point_ccw is not None
        assert point_cw is not None
        assert AvPolygon.ray_casting_single(square_ccw, point_ccw)
        assert AvPolygon.ray_casting_single(square_cw, point_cw)

    def test_interior_point_scanlines_large_coordinates(self):
        """Test interior point finding with large coordinate values."""
        # Large square
        large_square = np.array(
            [[1000.0, 1000.0, 0.0], [2000.0, 1000.0, 0.0], [2000.0, 2000.0, 0.0], [1000.0, 2000.0, 0.0]],
            dtype=np.float64,
        )

        point = AvPolygon.interior_point_scanlines(large_square)
        assert point is not None

        # Point should be inside large square
        assert AvPolygon.ray_casting_single(large_square, point)

        # Point should be within large square bounds
        assert 1000.0 < point[0] < 2000.0
        assert 1000.0 < point[1] < 2000.0

    def test_interior_point_scanlines_small_coordinates(self):
        """Test interior point finding with very small coordinate values."""
        # Small square
        small_square = np.array(
            [[0.001, 0.001, 0.0], [0.002, 0.001, 0.0], [0.002, 0.002, 0.0], [0.001, 0.002, 0.0]], dtype=np.float64
        )

        point = AvPolygon.interior_point_scanlines(small_square)
        assert point is not None

        # Point should be inside small square
        assert AvPolygon.ray_casting_single(small_square, point)

        # Point should be within small square bounds
        assert 0.001 < point[0] < 0.002
        assert 0.001 < point[1] < 0.002

    def test_interior_point_scanlines_negative_coordinates(self):
        """Test interior point finding with negative coordinate values."""
        # Square in negative quadrant
        neg_square = np.array(
            [[-2.0, -2.0, 0.0], [-1.0, -2.0, 0.0], [-1.0, -1.0, 0.0], [-2.0, -1.0, 0.0]], dtype=np.float64
        )

        point = AvPolygon.interior_point_scanlines(neg_square)
        assert point is not None

        # Point should be inside negative square
        assert AvPolygon.ray_casting_single(neg_square, point)

        # Point should be within negative square bounds
        assert -2.0 < point[0] < -1.0
        assert -2.0 < point[1] < -1.0

    def test_interior_point_scanlines_2d_points(self):
        """Test interior point finding with 2D points (without type column)."""
        # 2D square
        square_2d = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)

        point = AvPolygon.interior_point_scanlines(square_2d)
        assert point is not None

        # Should work the same as 3D points
        assert AvPolygon.ray_casting_single(square_2d, point)
        assert 0.0 < point[0] < 1.0
        assert 0.0 < point[1] < 1.0

    def test_interior_point_scanlines_skinny_polygon(self):
        """Test interior point finding on skinny polygons."""
        # Very thin rectangle
        skinny = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.01, 0.0], [0.0, 0.01, 0.0]], dtype=np.float64)

        point = AvPolygon.interior_point_scanlines(skinny)
        assert point is not None

        # Point should be inside skinny rectangle
        assert AvPolygon.ray_casting_single(skinny, point)

        # Point should be within skinny rectangle bounds
        assert 0.0 < point[0] < 10.0
        assert 0.0 < point[1] < 0.01

    def test_interior_point_scanlines_complex_concave(self):
        """Test interior point finding on complex concave polygon."""
        # Star-shaped concave polygon
        star_points = []
        n_points = 10
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            if i % 2 == 0:
                # Outer points
                radius = 2.0
            else:
                # Inner points (creating concavity)
                radius = 1.0
            star_points.append([radius * np.cos(angle), radius * np.sin(angle), 0.0])
        star = np.array(star_points, dtype=np.float64)

        point = AvPolygon.interior_point_scanlines(star)
        assert point is not None

        # Point should be inside star
        assert AvPolygon.ray_casting_single(star, point)

        # Point should be within overall bounds
        assert -2.0 < point[0] < 2.0
        assert -2.0 < point[1] < 2.0

    def test_interior_point_scanlines_performance_consistency(self):
        """Test that interior point finding gives consistent results."""
        square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        # Multiple calls should give valid interior points
        points = [AvPolygon.interior_point_scanlines(square) for _ in range(10)]
        for point in points:
            assert point is not None
            assert AvPolygon.ray_casting_single(square, point)

    def test_interior_point_scanlines_edge_cases(self):
        """Test interior point finding on edge case polygons."""
        # Triangle with very small area
        tiny_triangle = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]], dtype=np.float64)

        point = AvPolygon.interior_point_scanlines(tiny_triangle)
        assert point is not None
        assert AvPolygon.ray_casting_single(tiny_triangle, point)

        # Very tall skinny triangle
        tall_triangle = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 100.0, 0.0]], dtype=np.float64)

        point = AvPolygon.interior_point_scanlines(tall_triangle)
        assert point is not None
        assert AvPolygon.ray_casting_single(tall_triangle, point)
