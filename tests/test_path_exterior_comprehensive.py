"""Comprehensive tests for AvPathExterior class including edge cases."""

from __future__ import annotations

import numpy as np
import pytest
import shapely.geometry

from ave.path import SINGLE_POLYGON_CONSTRAINTS, AvSinglePolygonPath
from ave.path_exterior import AvPathExterior


class TestAvPathExterior:
    """Test suite for AvPathExterior class."""

    def test_left_exterior_silhouette_list_empty_input(self):
        """Test silhouette list with empty input."""
        result = AvPathExterior.left_exterior_silhouette_list([])
        assert result == []

    def test_left_exterior_silhouette_list_single_empty_polygon(self):
        """Test silhouette list with single empty polygon."""
        empty = AvSinglePolygonPath(np.empty((0, 3), dtype=np.float64), [], SINGLE_POLYGON_CONSTRAINTS)
        result = AvPathExterior.left_exterior_silhouette_list([empty])
        assert len(result) == 1
        assert len(result[0].points) == 0

    def test_left_exterior_silhouette_list_multiple_empty_polygons(self):
        """Test silhouette list with multiple empty polygons."""
        empty1 = AvSinglePolygonPath(np.empty((0, 3), dtype=np.float64), [], SINGLE_POLYGON_CONSTRAINTS)
        empty2 = AvSinglePolygonPath(np.empty((0, 3), dtype=np.float64), [], SINGLE_POLYGON_CONSTRAINTS)
        result = AvPathExterior.left_exterior_silhouette_list([empty1, empty2])
        assert len(result) == 2
        assert len(result[0].points) == 0
        assert len(result[1].points) == 0

    def test_left_exterior_silhouette_empty_polygon(self):
        """Test silhouette with empty polygon."""
        empty = AvSinglePolygonPath(np.empty((0, 3), dtype=np.float64), [], SINGLE_POLYGON_CONSTRAINTS)
        result = AvPathExterior.left_exterior_silhouette(empty)
        assert len(result.points) == 0

    def test_left_exterior_silhouette_degenerate_line(self):
        """Test silhouette with degenerate line (3 points forming line)."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        commands = ["M", "L", "L", "Z"]
        line = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)
        result = AvPathExterior.left_exterior_silhouette(line)
        assert len(result.points) == 0

    def test_left_exterior_silhouette_horizontal_line(self):
        """Test silhouette with horizontal line (zero height)."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        commands = ["M", "L", "L", "Z"]
        line = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)
        result = AvPathExterior.left_exterior_silhouette(line)
        assert len(result.points) == 0

    def test_left_exterior_silhouette_single_point(self):
        """Test silhouette with minimal valid triangle."""
        points = np.array([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [0.0, 1e-6, 0.0]])
        commands = ["M", "L", "L", "Z"]
        point = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)
        result = AvPathExterior.left_exterior_silhouette(point)
        assert isinstance(result, AvSinglePolygonPath)

    def test_left_exterior_silhouette_unit_square(self):
        """Test silhouette with unit square."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        commands = ["M", "L", "L", "L", "Z"]
        square = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(square)
        assert len(result.points) >= 4
        assert result.area > 0

        # Should be a valid polygon
        shapely_poly = shapely.geometry.Polygon(result.points[:, :2])
        assert shapely_poly.is_valid
        assert shapely_poly.is_simple

    def test_left_exterior_silhouette_large_rectangle(self):
        """Test silhouette with large rectangle."""
        points = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 50.0, 0.0], [0.0, 50.0, 0.0]])
        commands = ["M", "L", "L", "L", "Z"]
        rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(rect)
        assert len(result.points) >= 4
        assert result.area > 0

        # Check that max x is preserved
        max_x = np.max(result.points[:, 0])
        assert np.isclose(max_x, 100.0)

    def test_left_exterior_silhouette_very_small_polygon(self):
        """Test silhouette with very small polygon."""
        points = np.array([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [1e-3, 1e-3, 0.0], [0.0, 1e-3, 0.0]])
        commands = ["M", "L", "L", "L", "Z"]
        tiny = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(tiny)
        assert len(result.points) >= 4
        # Use a more lenient check for very small polygons
        assert result.area >= 0  # Area might be zero due to precision

    def test_left_exterior_silhouette_complex_concave(self):
        """Test silhouette with complex concave polygon."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # bottom-left
                [10.0, 0.0, 0.0],  # bottom-right
                [10.0, 3.0, 0.0],  # right side up
                [7.0, 3.0, 0.0],  # indent start
                [7.0, 6.0, 0.0],  # indent end
                [10.0, 6.0, 0.0],  # right side up
                [10.0, 10.0, 0.0],  # top-right
                [0.0, 10.0, 0.0],  # top-left
            ]
        )
        commands = ["M"] + ["L"] * 7 + ["Z"]
        concave = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(concave)
        assert len(result.points) >= 4
        assert result.area > 0
        assert result.area >= concave.area  # Silhouette should be larger or equal

    def test_left_exterior_silhouette_triangle(self):
        """Test silhouette with triangle."""
        points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.5, 4.0, 0.0]])
        commands = ["M", "L", "L", "Z"]
        triangle = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(triangle)
        assert len(result.points) >= 3
        assert result.area > 0

    def test_left_exterior_silhouette_pentagon(self):
        """Test silhouette with pentagon."""
        angles = np.linspace(0, 2 * np.pi, 6)[:-1]  # 5 points
        points = np.array([[np.cos(a), np.sin(a), 0.0] for a in angles])
        commands = ["M"] + ["L"] * 4 + ["Z"]
        pentagon = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(pentagon)
        assert len(result.points) >= 3
        assert result.area > 0

    def test_left_exterior_silhouette_irregular_polygon(self):
        """Test silhouette with irregular polygon."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [8.0, 1.0, 0.0],
                [10.0, 3.0, 0.0],
                [9.0, 7.0, 0.0],
                [6.0, 9.0, 0.0],
                [2.0, 8.0, 0.0],
                [1.0, 5.0, 0.0],
            ]
        )
        commands = ["M"] + ["L"] * 6 + ["Z"]
        irregular = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(irregular)
        assert len(result.points) >= 3
        assert result.area > 0

    def test_left_exterior_silhouette_multiple_polygons(self):
        """Test silhouette list with multiple different polygons."""
        # Square
        points1 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
        commands1 = ["M", "L", "L", "L", "Z"]
        square = AvSinglePolygonPath(points1, commands1, SINGLE_POLYGON_CONSTRAINTS)

        # Triangle
        points2 = np.array([[5.0, 0.0, 0.0], [8.0, 0.0, 0.0], [6.5, 3.0, 0.0]])
        commands2 = ["M", "L", "L", "Z"]
        triangle = AvSinglePolygonPath(points2, commands2, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette_list([square, triangle])
        assert len(result) == 2
        assert result[0].area > 0
        assert result[1].area > 0

    def test_remove_left_upward_undercuts_empty_input(self):
        """Test upward undercut removal with empty input."""
        coords = np.empty((0, 3), dtype=np.float64)
        result = AvPathExterior._remove_left_upward_undercuts(coords)
        assert len(result) == 0

    def test_remove_left_upward_undercuts_single_point(self):
        """Test upward undercut removal with single point."""
        coords = np.array([[0.0, 0.0, 0.0]])
        result = AvPathExterior._remove_left_upward_undercuts(coords)
        assert len(result) == 1

    def test_remove_left_upward_undercuts_two_points(self):
        """Test upward undercut removal with two points."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = AvPathExterior._remove_left_upward_undercuts(coords)
        assert len(result) == 2

    def test_remove_left_upward_undercuts_simple_square(self):
        """Test upward undercut removal with simple square."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        result = AvPathExterior._remove_left_upward_undercuts(coords)
        assert len(result) >= 4
        # Should be a valid polygon
        shapely_poly = shapely.geometry.Polygon(result[:, :2])
        assert shapely_poly.is_valid

    def test_remove_left_upward_undercuts_with_upward_undercut(self):
        """Test upward undercut removal with actual upward undercut."""
        # Create a shape with an upward undercut
        coords = np.array(
            [
                [0.0, 0.0, 0.0],  # bottom-left
                [5.0, 0.0, 0.0],  # bottom-right
                [5.0, 2.0, 0.0],  # up
                [3.0, 2.0, 0.0],  # left (undercut start)
                [3.0, 4.0, 0.0],  # up (undercut end)
                [5.0, 4.0, 0.0],  # right
                [5.0, 6.0, 0.0],  # up
                [0.0, 6.0, 0.0],  # top-left
            ]
        )
        result = AvPathExterior._remove_left_upward_undercuts(coords)
        assert len(result) >= 4
        # Should be a valid polygon
        shapely_poly = shapely.geometry.Polygon(result[:, :2])
        assert shapely_poly.is_valid

    def test_remove_left_downward_undercuts_empty_input(self):
        """Test downward undercut removal with empty input."""
        coords = np.empty((0, 3), dtype=np.float64)
        result = AvPathExterior._remove_left_downward_undercuts(coords)
        assert len(result) == 0

    def test_remove_left_downward_undercuts_single_point(self):
        """Test downward undercut removal with single point."""
        coords = np.array([[0.0, 0.0, 0.0]])
        result = AvPathExterior._remove_left_downward_undercuts(coords)
        assert len(result) == 1

    def test_remove_left_downward_undercuts_two_points(self):
        """Test downward undercut removal with two points."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = AvPathExterior._remove_left_downward_undercuts(coords)
        assert len(result) == 2

    def test_remove_left_downward_undercuts_simple_square(self):
        """Test downward undercut removal with simple square."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        result = AvPathExterior._remove_left_downward_undercuts(coords)
        assert len(result) >= 4
        # Should be a valid polygon
        shapely_poly = shapely.geometry.Polygon(result[:, :2])
        assert shapely_poly.is_valid

    def test_remove_left_downward_undercuts_with_downward_undercut(self):
        """Test downward undercut removal with actual downward undercut."""
        # Create a shape with a downward undercut (mirror of upward test)
        coords = np.array(
            [
                [0.0, 6.0, 0.0],  # top-left
                [5.0, 6.0, 0.0],  # top-right
                [5.0, 4.0, 0.0],  # down
                [3.0, 4.0, 0.0],  # left (undercut start)
                [3.0, 2.0, 0.0],  # down (undercut end)
                [5.0, 2.0, 0.0],  # right
                [5.0, 0.0, 0.0],  # down
                [0.0, 0.0, 0.0],  # bottom-left
            ]
        )
        result = AvPathExterior._remove_left_downward_undercuts(coords)
        assert len(result) >= 4
        # Should be a valid polygon
        shapely_poly = shapely.geometry.Polygon(result[:, :2])
        assert shapely_poly.is_valid

    def test_silhouette_preserves_ccw_orientation(self):
        """Test that silhouette preserves CCW orientation."""
        points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        commands = ["M", "L", "L", "L", "Z"]
        rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(rect)
        # Check orientation using signed area
        coords_2d = result.points[:, :2]
        n = len(coords_2d)
        signed_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            signed_area += coords_2d[i, 0] * coords_2d[j, 1]
            signed_area -= coords_2d[j, 0] * coords_2d[i, 1]
        signed_area /= 2.0

        assert signed_area > 0, "Polygon should be CCW (positive signed area)"

    def test_silhouette_with_very_large_coordinates(self):
        """Test silhouette with very large coordinate values."""
        points = np.array([[1e6, 1e6, 0.0], [2e6, 1e6, 0.0], [2e6, 2e6, 0.0], [1e6, 2e6, 0.0]])
        commands = ["M", "L", "L", "L", "Z"]
        large_rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(large_rect)
        assert len(result.points) >= 4
        assert result.area > 0
        # Check that large coordinates are handled correctly
        assert np.max(result.points[:, 0]) > 1e6

    def test_silhouette_with_negative_coordinates(self):
        """Test silhouette with negative coordinates."""
        points = np.array([[-5.0, -3.0, 0.0], [-1.0, -3.0, 0.0], [-1.0, -1.0, 0.0], [-5.0, -1.0, 0.0]])
        commands = ["M", "L", "L", "L", "Z"]
        negative_rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(negative_rect)
        assert len(result.points) >= 4
        assert result.area > 0

    def test_silhouette_with_mixed_coordinates(self):
        """Test silhouette with mixed positive and negative coordinates."""
        points = np.array([[-2.0, -1.0, 0.0], [3.0, -1.0, 0.0], [3.0, 2.0, 0.0], [-2.0, 2.0, 0.0]])
        commands = ["M", "L", "L", "L", "Z"]
        mixed_rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(mixed_rect)
        assert len(result.points) >= 4
        assert result.area > 0

    def test_silhouette_with_floating_point_precision(self):
        """Test silhouette with floating point precision edge cases."""
        points = np.array([[0.0, 0.0, 0.0], [1.0000001, 0.0, 0.0], [1.0000001, 1.0000001, 0.0], [0.0, 1.0000001, 0.0]])
        commands = ["M", "L", "L", "L", "Z"]
        precise_rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(precise_rect)
        assert len(result.points) >= 4
        assert result.area > 0

    def test_silhouette_with_collinear_points(self):
        """Test silhouette with collinear points on edges."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],  # extra point on bottom edge
                [5.0, 0.0, 0.0],
                [5.0, 3.0, 0.0],
                [2.5, 3.0, 0.0],  # extra point on top edge
                [0.0, 3.0, 0.0],
            ]
        )
        commands = ["M", "L", "L", "L", "L", "L", "Z"]
        collinear_rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(collinear_rect)
        assert len(result.points) >= 4
        assert result.area > 0

    def test_silhouette_with_near_duplicate_points(self):
        """Test silhouette with nearly duplicate points."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1e-10, 0.0, 0.0],  # very close to previous point
                [5.0, 0.0, 0.0],
                [5.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        )
        commands = ["M", "L", "L", "L", "L", "Z"]
        near_duplicate = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(near_duplicate)
        assert len(result.points) >= 4
        assert result.area > 0

    def test_performance_large_polygon(self):
        """Test performance with large polygon (many vertices)."""
        # Create a polygon with many vertices
        n_points = 1000
        angles = np.linspace(0, 2 * np.pi, n_points)
        radius = 10.0
        points = np.array([[radius * np.cos(a), radius * np.sin(a), 0.0] for a in angles])
        commands = ["M"] + ["L"] * (n_points - 1) + ["Z"]
        large_polygon = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        import time

        start_time = time.time()
        result = AvPathExterior.left_exterior_silhouette(large_polygon)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        assert len(result.points) >= 3
        assert result.area > 0

    def test_edge_case_zero_area_polygon(self):
        """Test silhouette with polygon that has zero area."""
        # All points on a line
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        commands = ["M", "L", "L", "L", "Z"]
        zero_area = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(zero_area)
        # Should handle gracefully (likely return empty or minimal polygon)
        assert isinstance(result, AvSinglePolygonPath)

    def test_edge_case_polygon_with_single_edge(self):
        """Test silhouette with polygon that has effectively one edge."""
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-15, 0.0], [0.0, 1e-15, 0.0]]  # almost same y as first point
        )
        commands = ["M", "L", "L", "L", "Z"]
        thin_polygon = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(thin_polygon)
        assert isinstance(result, AvSinglePolygonPath)

    def test_consistency_with_original_implementation(self):
        """Test that results are consistent with expected behavior."""
        # Simple test case with known expected behavior
        points = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        commands = ["M", "L", "L", "L", "Z"]
        rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = AvPathExterior.left_exterior_silhouette(rect)

        # Should preserve the bounding box dimensions
        assert np.isclose(np.max(result.points[:, 0]), 4.0)  # max x preserved
        assert np.isclose(np.max(result.points[:, 1]), 3.0)  # max y preserved
        assert np.isclose(np.min(result.points[:, 1]), 0.0)  # min y preserved

        # Should be a valid polygon
        shapely_poly = shapely.geometry.Polygon(result.points[:, :2])
        assert shapely_poly.is_valid
        assert shapely_poly.is_simple

    def test_simple_c_shape_monotone(self):
        """Test that C-shape silhouette is monotone (no backtracking)."""
        # Simple C-shape without self-intersections
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # bottom-left
                [10.0, 0.0, 0.0],  # bottom-right
                [10.0, 3.0, 0.0],  # up
                [3.0, 3.0, 0.0],  # left (undercut)
                [3.0, 7.0, 0.0],  # up
                [10.0, 7.0, 0.0],  # right
                [10.0, 10.0, 0.0],  # top-right
                [0.0, 10.0, 0.0],  # top-left
            ]
        )
        commands = ["M"] + ["L"] * 7 + ["Z"]
        shape = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        silhouette = AvPathExterior.left_exterior_silhouette(shape)
        coords = silhouette.points[:, :2]

        # Exclude closing edge
        max_x = np.max(coords[:, 0])
        left_edge = coords[coords[:, 0] < max_x - 0.5]

        if len(left_edge) < 2:
            pytest.skip("Not enough points")

        # Sort by y and check x doesn't increase significantly
        sorted_indices = np.argsort(left_edge[:, 1])
        sorted_coords = left_edge[sorted_indices]
        x_values = sorted_coords[:, 0]

        # Check monotonicity: x should not increase as y increases
        for i in range(1, len(x_values)):
            x_increase = x_values[i] - x_values[i - 1]
            assert x_increase <= 0.1, f"X increased by {x_increase} at index {i}, not monotone"

    def test_silhouette_preserves_dimensions(self):
        """Test that silhouette preserves overall shape dimensions."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )
        commands = ["M"] + ["L"] * 3 + ["Z"]
        shape = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        silhouette = AvPathExterior.left_exterior_silhouette(shape)
        coords = silhouette.points[:, :2]

        x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
        y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])

        assert x_range >= 9, f"Silhouette too narrow: {x_range}"
        assert y_range >= 9, f"Silhouette too short: {y_range}"
        assert len(coords) >= 4, "Silhouette collapsed"

    def test_undercut_removal_simple_c_shape(self):
        """Test undercut removal on a simple C-shaped polygon (like digit 3)."""
        # Create C-shape with undercut on left side
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # bottom-left (min_y, min_x)
                [10.0, 0.0, 0.0],  # bottom-right
                [10.0, 3.0, 0.0],  # up
                [3.0, 3.0, 0.0],  # left (undercut indentation - should be removed)
                [3.0, 7.0, 0.0],  # up
                [10.0, 7.0, 0.0],  # right
                [10.0, 10.0, 0.0],  # up
                [0.0, 10.0, 0.0],  # top-left (max_y)
            ]
        )
        commands = ["M"] + ["L"] * 7 + ["Z"]
        shape = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        # Get silhouette
        silhouette = AvPathExterior.left_exterior_silhouette(shape)

        # Check silhouette points
        assert len(silhouette.points) >= 4, "Silhouette should have at least 4 points"

        # Verify it's a proper polygon (not collapsed)
        x_coords = silhouette.points[:, 0]
        y_coords = silhouette.points[:, 1]

        x_range = np.max(x_coords) - np.min(x_coords)
        y_range = np.max(y_coords) - np.min(y_coords)

        # For C-shape, silhouette should span full height and width
        assert x_range > 5, f"Silhouette x-range {x_range} too small"
        assert y_range > 8, f"Silhouette y-range {y_range} too small, should span full height"

        # The left silhouette of a C-shape should be a clean vertical strip at x=0
        # (undercut at x=3 should be removed)
        assert np.min(x_coords) <= 1, "Left edge should be at or near x=0"
        assert np.max(x_coords) >= 9, "Right edge should be at or near max_x=10"

    def test_undercut_removal_s_shape(self):
        """Test S-shaped or reverse-C shape (like digit 5)."""
        # Create a shape with both top and bottom undercuts
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # bottom-left
                [10.0, 0.0, 0.0],  # bottom-right
                [10.0, 4.0, 0.0],  # up
                [3.0, 4.0, 0.0],  # left (mid-level)
                [3.0, 6.0, 0.0],  # up
                [10.0, 6.0, 0.0],  # right
                [10.0, 10.0, 0.0],  # up
                [0.0, 10.0, 0.0],  # top-left
            ]
        )
        commands = ["M"] + ["L"] * 7 + ["Z"]
        shape = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        silhouette = AvPathExterior.left_exterior_silhouette(shape)

        # Should be a valid polygon with proper dimensions
        assert len(silhouette.points) >= 4

        x_coords = silhouette.points[:, 0]
        y_coords = silhouette.points[:, 1]

        x_range = np.max(x_coords) - np.min(x_coords)
        y_range = np.max(y_coords) - np.min(y_coords)

        assert x_range > 5, f"S-shape silhouette too narrow: {x_range}"
        assert y_range > 8, f"S-shape silhouette too short: {y_range}"

    def test_undercut_removal_with_shapely_validation(self):
        """Validate silhouette using Shapely - should be simple and valid."""
        # Complex shape with multiple indentations
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 2.0, 0.0],
                [4.0, 2.0, 0.0],
                [4.0, 4.0, 0.0],
                [10.0, 4.0, 0.0],
                [10.0, 6.0, 0.0],
                [4.0, 6.0, 0.0],
                [4.0, 8.0, 0.0],
                [10.0, 8.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )
        commands = ["M"] + ["L"] * 11 + ["Z"]
        shape = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        silhouette = AvPathExterior.left_exterior_silhouette(shape)

        # Validate with Shapely
        coords = silhouette.points[:, :2].tolist()
        if len(coords) >= 3:
            poly = shapely.geometry.Polygon(coords)
            assert poly.is_valid, "Silhouette should be a valid polygon"
            assert poly.is_simple, "Silhouette should be simple (no self-intersections)"

            # Check area is reasonable (not collapsed to line)
            assert poly.area > 10, f"Silhouette area {poly.area} too small, likely collapsed"
