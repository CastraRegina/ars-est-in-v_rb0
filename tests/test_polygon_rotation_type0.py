#!/usr/bin/env python3
"""
Test that rotation is applied to first matching type=0 point from original path.
The rotation algorithm prioritizes the first point of the input path,
but the target point MUST be of type=0.
"""

import numpy as np
import shapely.geometry

from ave.path import AvPolygon
from ave.path_processing import AvPathCleaner


def test_rotation_to_first_type0_point():
    """Test that rotation targets the first type=0 point from original."""

    # Create a test case with a simple square
    exterior_coords = [(10, 10), (20, 10), (20, 20), (10, 20)]

    # Create Shapely polygon
    polygon = shapely.geometry.Polygon(exterior_coords)

    # Original points with types - first point is type=0 at (10, 10)
    original_points = np.array(
        [
            [10.0, 10.0, 0.0],  # type=0, should be rotation target
            [20.0, 10.0, 0.0],
            [20.0, 20.0, 0.0],
            [10.0, 20.0, 0.0],
        ]
    )

    # Convert to paths using the function
    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    # Should have 1 path
    assert len(paths) == 1, f"Expected 1 path, got {len(paths)}"

    # The path should start at (10, 10) - the first type=0 point
    path = paths[0]
    start_point = path.points[0, :2]
    expected_start = np.array([10.0, 10.0])

    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"Path should start at first type=0 point, got {start_point}"


def test_rotation_skips_non_type0_first_point():
    """Test that rotation skips non-type=0 points and finds next type=0."""

    # Create a simple square
    exterior_coords = [(10, 10), (20, 10), (20, 20), (10, 20)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # Original points - first point is type=2 (not type=0), second is type=0
    original_points = np.array(
        [
            [10.0, 10.0, 2.0],  # type=2, should be skipped
            [20.0, 10.0, 0.0],  # type=0, should be rotation target
            [20.0, 20.0, 2.0],
            [10.0, 20.0, 0.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 1
    path = paths[0]
    start_point = path.points[0, :2]
    expected_start = np.array([20.0, 10.0])  # Second point (first type=0)

    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"Path should start at first type=0 point (20,10), got {start_point}"


def test_rotation_with_holes():
    """Test that rotation is applied independently to exterior and holes."""

    # Main exterior
    exterior = [(0, 0), (30, 0), (30, 30), (0, 30)]

    # Hole
    hole = [(5, 5), (10, 5), (10, 10), (5, 10)]

    polygon = shapely.geometry.Polygon(exterior, [hole])

    # Original points with types for both exterior and hole
    original_points = np.array(
        [
            [0.0, 0.0, 0.0],  # Exterior point, type=0
            [30.0, 0.0, 0.0],
            [30.0, 30.0, 0.0],
            [0.0, 30.0, 0.0],
            [5.0, 5.0, 0.0],  # Hole point, type=0
            [10.0, 5.0, 0.0],
            [10.0, 10.0, 0.0],
            [5.0, 10.0, 0.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    # Should have 2 paths (exterior + hole)
    assert len(paths) == 2, f"Expected 2 paths, got {len(paths)}"

    # Exterior should start at (0, 0) - first type=0 point
    exterior_path = paths[0]
    exterior_start = exterior_path.points[0, :2]
    expected_exterior_start = np.array([0.0, 0.0])
    assert np.linalg.norm(exterior_start - expected_exterior_start) < 1e-9

    # Hole should start at (5, 5) - first matching type=0 point for hole
    hole_path = paths[1]
    hole_start = hole_path.points[0, :2]
    expected_hole_start = np.array([5.0, 5.0])
    assert np.linalg.norm(hole_start - expected_hole_start) < 1e-9


def test_rotation_with_no_matching_point():
    """Test behavior when no type=0 point matches in coords."""

    # Square at (100, 100) area
    exterior_coords = [(100, 100), (110, 100), (110, 110), (100, 110)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # Original points are at completely different location (no match)
    original_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    # Should still return 1 path, just no rotation (starts at index 0)
    assert len(paths) == 1
    # No assertion on start point - just verify it doesn't crash


def test_rotation_preserves_orientation():
    """Test that rotation doesn't change the orientation of polygons."""

    # Create a CW exterior (should be reversed to CCW)
    cw_exterior = [(0, 0), (0, 10), (10, 10), (10, 0)]  # Clockwise

    # Create a CCW interior (should be reversed to CW)
    ccw_interior = [(2, 2), (8, 2), (8, 8), (2, 8)]  # Counter-clockwise

    polygon = shapely.geometry.Polygon(cw_exterior, [ccw_interior])

    # Original points with types
    original_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
            [10.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [8.0, 2.0, 0.0],
            [8.0, 8.0, 0.0],
            [2.0, 8.0, 0.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    # Check exterior is CCW
    exterior_path = paths[0]
    exterior_coords = exterior_path.points[:, :2]
    assert AvPolygon.is_ccw(exterior_coords), "Exterior should be CCW after processing"

    # Check interior is CW
    interior_path = paths[1]
    interior_coords = interior_path.points[:, :2]
    assert not AvPolygon.is_ccw(interior_coords), "Interior should be CW after processing"


def test_rotation_with_tolerance():
    """Test that TOLERANCE is applied correctly for coordinate matching."""

    # Square with exact coordinates
    exterior_coords = [(10, 10), (20, 10), (20, 20), (10, 20)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # Original points with very slight offset (within tolerance 1e-10)
    original_points = np.array(
        [
            [10.0 + 1e-11, 10.0 + 1e-11, 0.0],  # Within tolerance of (10, 10)
            [20.0, 10.0, 0.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 1
    path = paths[0]
    start_point = path.points[0, :2]

    # Should start at (10, 10) since the original point is within tolerance
    expected_start = np.array([10.0, 10.0])
    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"Should match within tolerance, got {start_point}"


def test_all_points_non_type0():
    """Test that no rotation happens when all original points are non-type=0."""

    exterior_coords = [(10, 10), (20, 10), (20, 20), (10, 20)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # All original points are type=2 (no type=0)
    original_points = np.array(
        [
            [10.0, 10.0, 2.0],
            [20.0, 10.0, 2.0],
            [20.0, 20.0, 2.0],
            [10.0, 20.0, 2.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    # Should return 1 path without rotation (default index 0)
    assert len(paths) == 1


if __name__ == "__main__":
    test_rotation_to_first_type0_point()
    test_rotation_skips_non_type0_first_point()
    test_rotation_with_holes()
    test_rotation_with_no_matching_point()
    test_rotation_preserves_orientation()
    test_rotation_with_tolerance()
    test_all_points_non_type0()
    print("All tests passed!")
