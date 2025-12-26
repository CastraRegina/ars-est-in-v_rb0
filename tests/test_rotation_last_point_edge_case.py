#!/usr/bin/env python3
"""
Test rotation when the target point is the last point (before Z command).
This is an edge case where the rotation needs to handle the last point correctly.
"""

import numpy as np
import shapely.geometry

from ave.path import AvPolygon
from ave.path_processing import AvPathCleaner


def test_rotation_to_last_point_simple():
    """Test rotation when the matching point is the last vertex."""

    # Create a triangle
    exterior_coords = [(10, 10), (20, 10), (15, 20)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # Original points where the last point is type=0 and should be rotation target
    original_points = np.array(
        [
            [10.0, 10.0, 2.0],  # type=2, skip
            [20.0, 10.0, 2.0],  # type=2, skip
            [15.0, 20.0, 0.0],  # type=0, last point, should be rotation target
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 1
    path = paths[0]
    start_point = path.points[0, :2]
    expected_start = np.array([15.0, 20.0])  # Should start at the last point

    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"Path should start at last point (15,20), got {start_point}"

    # Check that the path is still valid (CCW after rotation)
    assert path.commands[0] == "M"
    assert path.commands[-1] == "Z"
    assert AvPolygon.is_ccw(path.points[:, :2])


def test_rotation_to_last_point_with_holes():
    """Test rotation when both exterior and hole targets are last points."""

    # Exterior square
    exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]

    # Hole square
    hole = [(2, 2), (8, 2), (8, 8), (2, 8)]

    polygon = shapely.geometry.Polygon(exterior, [hole])

    # Original points where last points of both exterior and hole are type=0
    original_points = np.array(
        [
            # Exterior points
            [0.0, 0.0, 2.0],  # type=2, skip
            [10.0, 0.0, 2.0],  # type=2, skip
            [10.0, 10.0, 2.0],  # type=2, skip
            [0.0, 10.0, 0.0],  # type=0, last exterior point, rotation target
            # Hole points
            [2.0, 2.0, 2.0],  # type=2, skip
            [8.0, 2.0, 2.0],  # type=2, skip
            [8.0, 8.0, 2.0],  # type=2, skip
            [2.0, 8.0, 0.0],  # type=0, last hole point, rotation target
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 2

    # Exterior should start at (0, 10) - last exterior point
    exterior_path = paths[0]
    exterior_start = exterior_path.points[0, :2]
    expected_exterior_start = np.array([0.0, 10.0])
    assert np.linalg.norm(exterior_start - expected_exterior_start) < 1e-9
    assert AvPolygon.is_ccw(exterior_path.points[:, :2])  # Should still be CCW

    # Hole should start at (2, 8) - last hole point
    hole_path = paths[1]
    hole_start = hole_path.points[0, :2]
    expected_hole_start = np.array([2.0, 8.0])
    assert np.linalg.norm(hole_start - expected_hole_start) < 1e-9
    assert not AvPolygon.is_ccw(hole_path.points[:, :2])  # Should still be CW


def test_rotation_to_last_point_complex():
    """Test with a more complex polygon where rotation target is last point."""

    # Pentagon
    exterior_coords = [(10, 0), (20, 5), (18, 15), (12, 18), (5, 10)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # Original points where only the last point is type=0
    original_points = np.array(
        [
            [10.0, 0.0, 2.0],  # type=2, skip
            [20.0, 5.0, 2.0],  # type=2, skip
            [18.0, 15.0, 2.0],  # type=2, skip
            [12.0, 18.0, 2.0],  # type=2, skip
            [5.0, 10.0, 0.0],  # type=0, last point, rotation target
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 1
    path = paths[0]
    start_point = path.points[0, :2]
    expected_start = np.array([5.0, 10.0])

    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"Path should start at last point (5,10), got {start_point}"

    # Verify the path order is correct after rotation
    expected_order = [
        [5.0, 10.0],  # Start at last point
        [10.0, 0.0],  # Then first point
        [20.0, 5.0],  # Then second point
        [18.0, 15.0],  # Then third point
        [12.0, 18.0],  # Then fourth point
    ]

    for i, expected in enumerate(expected_order):
        actual = path.points[i, :2]
        assert np.linalg.norm(actual - expected) < 1e-9, f"Point {i} should be {expected}, got {actual}"


def test_rotation_last_point_with_tolerance():
    """Test rotation to last point with coordinate tolerance."""

    exterior_coords = [(10, 10), (20, 10), (20, 20), (10, 20)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # Original points where last point has slight offset but within tolerance
    original_points = np.array(
        [
            [10.0, 10.0, 2.0],
            [20.0, 10.0, 2.0],
            [20.0, 20.0, 2.0],
            [10.0 + 1e-11, 20.0 + 1e-11, 0.0],  # Within tolerance of (10, 20)
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 1
    path = paths[0]
    start_point = path.points[0, :2]
    expected_start = np.array([10.0, 20.0])

    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"Should match last point within tolerance, got {start_point}"


def test_no_rotation_when_last_point_not_type0():
    """Test that no rotation happens when last point is not type=0."""

    exterior_coords = [(10, 10), (20, 10), (20, 20), (10, 20)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # Original points where no point is type=0
    original_points = np.array(
        [
            [10.0, 10.0, 2.0],
            [20.0, 10.0, 2.0],
            [20.0, 20.0, 2.0],
            [10.0, 20.0, 2.0],  # Last point is also type=2
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 1
    path = paths[0]
    start_point = path.points[0, :2]

    # Should not rotate (start at index 0)
    expected_start = np.array([10.0, 10.0])
    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"No rotation should occur, got {start_point}"


def test_rotation_multiple_last_points():
    """Test when multiple polygons have last points as targets."""

    # Create two separate polygons
    poly1_coords = [(0, 0), (5, 0), (5, 5), (0, 5)]
    poly2_coords = [(10, 10), (15, 10), (15, 15), (10, 15)]

    poly1 = shapely.geometry.Polygon(poly1_coords)
    poly2 = shapely.geometry.Polygon(poly2_coords)
    multi = shapely.geometry.MultiPolygon([poly1, poly2])

    # Original points where last points of both polygons are type=0
    original_points = np.array(
        [
            # First polygon
            [0.0, 0.0, 2.0],
            [5.0, 0.0, 2.0],
            [5.0, 5.0, 2.0],
            [0.0, 5.0, 0.0],  # First polygon last point
            # Second polygon
            [10.0, 10.0, 2.0],
            [15.0, 10.0, 2.0],
            [15.0, 15.0, 2.0],
            [10.0, 15.0, 0.0],  # Second polygon last point
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(multi, original_points)

    assert len(paths) == 2

    # Both should start at their respective last points
    assert np.linalg.norm(paths[0].points[0, :2] - np.array([0.0, 5.0])) < 1e-9
    assert np.linalg.norm(paths[1].points[0, :2] - np.array([10.0, 15.0])) < 1e-9


if __name__ == "__main__":
    test_rotation_to_last_point_simple()
    test_rotation_to_last_point_with_holes()
    test_rotation_to_last_point_complex()
    test_rotation_last_point_with_tolerance()
    test_no_rotation_when_last_point_not_type0()
    test_rotation_multiple_last_points()
    print("All edge case tests passed!")
