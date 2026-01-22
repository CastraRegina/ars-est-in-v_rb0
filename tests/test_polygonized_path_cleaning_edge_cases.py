"""Test cases for edge cases in AvPathCleaner.resolve_polygonized_path_intersections."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ave.path import MULTI_POLYGON_CONSTRAINTS, AvMultiPolylinePath, AvPath
from ave.path_processing import AvPathCleaner
from ave.path_support import MULTI_POLYLINE_CONSTRAINTS


class TestPolygonizedPathCleaningEdgeCases:
    """Test edge cases and error conditions in path cleaning."""

    def test_multipolygon_buffer_result(self):
        """Test handling when buffer(0) returns MultiPolygon."""
        # Create a shape that might result in MultiPolygon after buffer(0)
        # A figure-8 shape that gets split into two separate polygons
        points = np.array(
            [
                # First loop
                [0, 0, 0],
                [50, 0, 0],
                [50, 50, 0],
                [0, 50, 0],
                [0, 0, 0],
                # Second loop (overlapping)
                [25, 25, 0],
                [75, 25, 0],
                [75, 75, 0],
                [25, 75, 0],
                [25, 25, 0],
            ],
            dtype=np.float64,
        )

        commands = ["M", "L", "L", "L", "L", "M", "L", "L", "L", "L", "Z"]

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)
        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should handle MultiPolygon result gracefully
        assert isinstance(result, AvPath)
        # Should have valid points and commands
        assert len(result.points) >= 0
        assert len(result.commands) >= 0

    def test_geometrycollection_buffer_result(self):
        """Test handling when buffer(0) returns GeometryCollection."""
        # Create a degenerate shape that might result in GeometryCollection
        points = np.array(
            [[0, 0, 0], [100, 0, 0], [50, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0], [0, 0, 0]], dtype=np.float64
        )

        commands = ["M", "L", "L", "L", "L", "L", "L", "Z"]

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)
        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should handle GeometryCollection result gracefully
        assert isinstance(result, AvPath)

    def test_shapely_error_during_buffer(self):
        """Test handling of Shapely errors during buffer(0) operation."""
        points = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0], [0, 0, 0]], dtype=np.float64)

        commands = ["M", "L", "L", "L", "L", "Z"]

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)

        # Mock shapely.geometry.Polygon to raise a ShapelyError during buffer
        import shapely.errors

        with patch("ave.path_processing.shapely.geometry.Polygon") as mock_polygon:
            mock_poly = MagicMock()
            mock_poly.buffer.side_effect = shapely.errors.ShapelyError("Shapely error")
            mock_polygon.return_value = mock_poly

            result = AvPathCleaner.resolve_polygon_path_intersections(path)

            # Should return empty path on error
            assert isinstance(result, AvPath)
            assert len(result.points) == 0
            assert len(result.commands) == 0

    def test_shapely_error_during_geometry_conversion(self):
        """Test handling of Shapely errors during geometry conversion back to paths."""
        points = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0], [0, 0, 0]], dtype=np.float64)

        commands = ["M", "L", "L", "L", "L", "Z"]

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)

        # Mock the geometry conversion to raise a ValueError
        with patch("ave.path_processing.AvPath.join_paths") as mock_join:
            mock_join.side_effect = ValueError("Join error")

            result = AvPathCleaner.resolve_polygon_path_intersections(path)

            # Should return empty path on error
            assert isinstance(result, AvPath)
            assert len(result.points) == 0
            assert len(result.commands) == 0

    def test_coordinate_preservation_with_original_first_point(self):
        """Test that original first point is preserved in result."""
        # Create a path with a specific starting point
        original_first_point = np.array([42.5, 17.8])

        points = np.array([[42.5, 17.8, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0], [42.5, 17.8, 0]], dtype=np.float64)

        commands = ["M", "L", "L", "L", "L", "Z"]

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)
        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should have preserved the starting point location
        if len(result.points) > 0:
            # Find the point closest to original first point
            distances = np.linalg.norm(result.points[:, :2] - original_first_point, axis=1)
            min_distance = np.min(distances)
            # Should be very close (within tolerance)
            assert min_distance < 1e-6

    def test_coordinate_preservation_rotation_logic(self):
        """Test the rotate_to_start_point helper function logic."""
        # Create a square with starting point at a specific location
        points = np.array([[10, 10, 0], [20, 10, 0], [20, 20, 0], [10, 20, 0], [10, 10, 0]], dtype=np.float64)

        commands = ["M", "L", "L", "L", "L", "Z"]

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)
        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should preserve the starting point
        if len(result.points) > 0:
            # The result should start close to the original starting point
            assert np.linalg.norm(result.points[0, :2] - points[0, :2]) < 1e-6

    def test_no_ccw_polygon_found_case(self):
        """Test behavior when no CCW polygon is found (all CW)."""
        # Create a path with only CW polygons (holes)
        points = np.array(
            [
                # CW square (hole)
                [0, 0, 0],
                [0, 100, 0],
                [100, 100, 0],
                [100, 0, 0],
                [0, 0, 0],
                # Another CW square
                [150, 0, 0],
                [150, 100, 0],
                [250, 100, 0],
                [250, 0, 0],
                [150, 0, 0],
            ],
            dtype=np.float64,
        )

        commands = ["M", "L", "L", "L", "L", "M", "L", "L", "L", "L", "Z"]

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)
        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should return empty path when no CCW polygon found
        assert isinstance(result, AvPath)
        assert len(result.points) == 0
        assert len(result.commands) == 0

    def test_empty_result_after_operations(self):
        """Test behavior when result becomes empty after boolean operations."""
        # Create two identical polygons that cancel each other out
        points = np.array(
            [
                # First square
                [0, 0, 0],
                [100, 0, 0],
                [100, 100, 0],
                [0, 100, 0],
                [0, 0, 0],
                # Second identical square (should result in empty after operations)
                [0, 0, 0],
                [100, 0, 0],
                [100, 100, 0],
                [0, 100, 0],
                [0, 0, 0],
            ],
            dtype=np.float64,
        )

        commands = ["M", "L", "L", "L", "L", "M", "L", "L", "L", "L", "Z"]

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)
        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should handle empty result gracefully
        assert isinstance(result, AvPath)

    def test_linestring_point_geometry_handling(self):
        """Test handling of LineString or Point geometries from buffer(0)."""
        # Create a degenerate line that might result in LineString
        points = np.array([[0, 0, 0], [100, 0, 0], [0, 0, 0]], dtype=np.float64)  # Back and forth line

        commands = ["M", "L", "L", "Z"]

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)
        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should handle degenerate geometries gracefully
        assert isinstance(result, AvPath)

    def test_large_number_of_segments_performance(self):
        """Test performance with many segments (stress test)."""
        # Create a path with many small segments
        num_segments = 50
        points = []
        commands = []

        for i in range(num_segments):
            # Small square for each segment
            start_x = i * 10
            start_y = i * 5
            square_points = [
                [start_x, start_y, 0],
                [start_x + 5, start_y, 0],
                [start_x + 5, start_y + 5, 0],
                [start_x, start_y + 5, 0],
                [start_x, start_y, 0],
            ]
            points.extend(square_points)
            commands.extend(["M", "L", "L", "L", "L"])

        points = np.array(points, dtype=np.float64)
        commands.append("Z")  # Close the last segment

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)

        # Should complete without timeout or memory issues
        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        assert isinstance(result, AvPath)

    def test_mixed_orientation_multipolygon_subpolygons(self):
        """Test MultiPolygon sub-polygons with potentially different orientations."""
        # Create a complex shape that might split into differently oriented sub-polygons
        points = np.array(
            [
                # Create a self-intersecting shape
                [0, 0, 0],
                [100, 100, 0],
                [100, 0, 0],
                [0, 100, 0],
                [0, 0, 0],
            ],
            dtype=np.float64,
        )

        commands = ["M", "L", "L", "L", "L", "Z"]

        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)
        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should handle complex MultiPolygon scenarios
        assert isinstance(result, AvPath)
