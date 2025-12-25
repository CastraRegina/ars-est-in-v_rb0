#!/usr/bin/env python3
"""
Test that rotate_to_start_point() is applied independently to each polygon
regardless of orientation (CCW/CW).
"""

import numpy as np
import shapely.geometry

from ave.path import AvPath, AvPolygon
from ave.path_processing import AvPathCleaner


def test_rotation_applied_to_all_polygons():
    """Test that rotation is applied to both exterior and interior rings independently."""

    # Create a test case with a polygon that has holes
    # Exterior: square starting at (10,10)
    exterior_coords = [(10, 10), (20, 10), (20, 20), (15, 20), (10, 20)]

    # Interior (hole): smaller square starting at (12,12)
    interior_coords = [(12, 12), (18, 12), (18, 18), (12, 18)]

    # Create Shapely polygon
    exterior = shapely.geometry.Polygon(exterior_coords, [interior_coords])

    # Original first point (should be preserved after rotation)
    original_first_point = np.array([10.0, 10.0])

    # Convert to paths using the function
    paths = AvPathCleaner._convert_shapely_to_paths(exterior, original_first_point)

    # Should have 2 paths: exterior and interior
    assert len(paths) == 2, f"Expected 2 paths, got {len(paths)}"

    # Check exterior path
    exterior_path = paths[0]
    assert exterior_path.commands[0] == "M", "Exterior should start with M"
    assert exterior_path.commands[-1] == "Z", "Exterior should end with Z"

    # The exterior should start at or very close to the original first point
    exterior_start = exterior_path.points[0, :2]
    distance_to_original = np.linalg.norm(exterior_start - original_first_point)
    assert distance_to_original < 1e-6, f"Exterior should start near original point, distance: {distance_to_original}"

    # Check interior path
    interior_path = paths[1]
    assert interior_path.commands[0] == "M", "Interior should start with M"
    assert interior_path.commands[-1] == "Z", "Interior should end with Z"

    # The interior should also be rotated to start closest to original_first_point
    interior_start = interior_path.points[0, :2]

    # Find the point in interior that's closest to original_first_point
    interior_points = np.array(interior_coords)
    distances = [np.linalg.norm(point - original_first_point) for point in interior_points]
    expected_interior_start = interior_points[np.argmin(distances)]

    distance_to_expected = np.linalg.norm(interior_start - expected_interior_start)
    assert (
        distance_to_expected < 1e-6
    ), f"Interior should start at point closest to original, distance: {distance_to_expected}"


def test_rotation_with_multipolygon():
    """Test that rotation is applied to each polygon in a MultiPolygon independently."""

    # Create two separate polygons
    poly1_coords = [(5, 5), (10, 5), (10, 10), (5, 10)]  # Square at (5,5)
    poly2_coords = [(20, 20), (25, 20), (25, 25), (20, 25)]  # Square at (20,20)

    poly1 = shapely.geometry.Polygon(poly1_coords)
    poly2 = shapely.geometry.Polygon(poly2_coords)

    # Create MultiPolygon
    multi = shapely.geometry.MultiPolygon([poly1, poly2])

    # Original first point (closer to second polygon)
    original_first_point = np.array([22.0, 22.0])

    # Convert to paths
    paths = AvPathCleaner._convert_shapely_to_paths(multi, original_first_point)

    # Should have 2 paths
    assert len(paths) == 2, f"Expected 2 paths, got {len(paths)}"

    # Each path should start at the point closest to original_first_point
    for i, path in enumerate(paths):
        start_point = path.points[0, :2]
        distance_to_original = np.linalg.norm(start_point - original_first_point)

        # Find expected start point for this polygon
        if i == 0:
            expected_points = np.array(poly1_coords)
        else:
            expected_points = np.array(poly2_coords)

        distances = [np.linalg.norm(point - original_first_point) for point in expected_points]
        expected_start = expected_points[np.argmin(distances)]

        distance_to_expected = np.linalg.norm(start_point - expected_start)
        assert distance_to_expected < 1e-6, f"Polygon {i} should start at closest point to original"


def test_rotation_with_complex_holes():
    """Test rotation with multiple holes of different orientations."""

    # Main exterior
    exterior = [(0, 0), (30, 0), (30, 30), (0, 30)]

    # Multiple holes with different starting points
    hole1 = [(5, 5), (10, 5), (10, 10), (5, 10)]  # Starts at (5,5)
    hole2 = [(15, 15), (20, 15), (20, 20), (15, 20)]  # Starts at (15,15)
    hole3 = [(25, 5), (28, 5), (28, 8), (25, 8)]  # Starts at (25,5)

    polygon = shapely.geometry.Polygon(exterior, [hole1, hole2, hole3])

    # Original first point
    original_first_point = np.array([17.0, 17.0])

    # Convert to paths
    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_first_point)

    # Should have 4 paths (1 exterior + 3 holes)
    assert len(paths) == 4, f"Expected 4 paths, got {len(paths)}"

    # Check that all paths start at points closest to original_first_point
    for i, path in enumerate(paths):
        start_point = path.points[0, :2]

        # Determine which coordinates to check against
        if i == 0:
            coords = exterior
        elif i == 1:
            coords = hole1
        elif i == 2:
            coords = hole2
        else:
            coords = hole3

        # Find expected start point
        points_array = np.array(coords)
        distances = [np.linalg.norm(point - original_first_point) for point in points_array]
        expected_start = points_array[np.argmin(distances)]

        distance_to_expected = np.linalg.norm(start_point - expected_start)
        assert distance_to_expected < 1e-6, f"Path {i} should start at closest point to original"


def test_rotation_preserves_orientation():
    """Test that rotation doesn't change the orientation of polygons."""

    # Create a CW exterior (should be reversed to CCW)
    cw_exterior = [(0, 0), (0, 10), (10, 10), (10, 0)]  # Clockwise

    # Create a CCW interior (should be reversed to CW)
    ccw_interior = [(2, 2), (8, 2), (8, 8), (2, 8)]  # Counter-clockwise

    polygon = shapely.geometry.Polygon(cw_exterior, [ccw_interior])
    original_first_point = np.array([0.0, 0.0])

    # Convert to paths
    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_first_point)

    # Check exterior is CCW
    exterior_path = paths[0]
    exterior_coords = exterior_path.points[:, :2]
    assert AvPolygon.is_ccw(exterior_coords), "Exterior should be CCW after processing"

    # Check interior is CW
    interior_path = paths[1]
    interior_coords = interior_path.points[:, :2]
    assert not AvPolygon.is_ccw(interior_coords), "Interior should be CW after processing"

    # Both should still be rotated to start near original point
    exterior_start = exterior_path.points[0, :2]
    interior_start = interior_path.points[0, :2]

    # Find expected starts
    exterior_points = np.array(cw_exterior)
    exterior_distances = [np.linalg.norm(point - original_first_point) for point in exterior_points]
    expected_exterior_start = exterior_points[np.argmin(exterior_distances)]

    interior_points = np.array(ccw_interior)
    interior_distances = [np.linalg.norm(point - original_first_point) for point in interior_points]
    expected_interior_start = interior_points[np.argmin(interior_distances)]

    assert np.linalg.norm(exterior_start - expected_exterior_start) < 1e-6
    assert np.linalg.norm(interior_start - expected_interior_start) < 1e-6


if __name__ == "__main__":
    test_rotation_applied_to_all_polygons()
    test_rotation_with_multipolygon()
    test_rotation_with_complex_holes()
    test_rotation_preserves_orientation()
    print("All tests passed!")
