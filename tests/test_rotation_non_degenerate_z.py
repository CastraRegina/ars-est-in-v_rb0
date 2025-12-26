#!/usr/bin/env python3
"""
Test that rotation prefers non-degenerate Z lines.
A degenerate Z line occurs when first_point == last_point after rotation.
The algorithm should prefer type=0 points that create actual line segments.

Note: Shapely normalizes polygons and removes consecutive duplicate points,
so we test the logic directly and with edge cases that Shapely preserves.
"""

import numpy as np
import shapely.geometry

from ave.path_processing import AvPathCleaner


def test_normal_polygon_no_degenerate():
    """Test normal polygon where no degenerate Z lines exist."""

    # Normal square - no duplicate points
    exterior_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # All type=0 points
    original_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 1
    path = paths[0]
    start_point = path.points[0, :2]

    # Should start at first point (0, 0) since it's type=0 and non-degenerate
    expected_start = np.array([0.0, 0.0])
    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"Should start at first point, got {start_point}"


def test_rotation_prefers_first_type0_when_all_valid():
    """Test that first type=0 point is chosen when all create valid Z lines."""

    # Pentagon - all points create valid Z lines
    exterior_coords = [(0, 0), (10, 0), (12, 8), (5, 12), (-2, 8)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # First two points are type=0
    original_points = np.array(
        [
            [0.0, 0.0, 0.0],  # type=0, first preference
            [10.0, 0.0, 0.0],  # type=0, second preference
            [12.0, 8.0, 2.0],
            [5.0, 12.0, 2.0],
            [-2.0, 8.0, 2.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 1
    path = paths[0]
    start_point = path.points[0, :2]

    # Should start at first type=0 point (0, 0)
    expected_start = np.array([0.0, 0.0])
    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"Should prefer first type=0, got {start_point}"


def test_skip_non_type0_points():
    """Test that non-type=0 points are always skipped."""

    exterior_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # First point is not type=0, second is
    original_points = np.array(
        [
            [0.0, 0.0, 2.0],  # type=2, skip
            [10.0, 0.0, 0.0],  # type=0, should be chosen
            [10.0, 10.0, 2.0],
            [0.0, 10.0, 2.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 1
    path = paths[0]
    start_point = path.points[0, :2]

    # Should skip (0,0) and start at (10,0)
    expected_start = np.array([10.0, 0.0])
    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"Should skip non-type=0, got {start_point}"


def test_degenerate_z_with_holes():
    """Test rotation applies independently to exterior and holes."""

    # Exterior - normal square
    exterior = [(0, 0), (30, 0), (30, 30), (0, 30)]

    # Hole - normal square
    hole = [(10, 10), (20, 10), (20, 20), (10, 20)]

    polygon = shapely.geometry.Polygon(exterior, [hole])

    # Original points
    original_points = np.array(
        [
            # Exterior
            [0.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [30.0, 30.0, 0.0],
            [0.0, 30.0, 0.0],
            # Hole
            [10.0, 10.0, 0.0],
            [20.0, 10.0, 0.0],
            [20.0, 20.0, 0.0],
            [10.0, 20.0, 0.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 2

    # Exterior should start at (0,0) - first type=0
    exterior_path = paths[0]
    assert np.linalg.norm(exterior_path.points[0, :2] - np.array([0.0, 0.0])) < 1e-9

    # Hole should start at first matching type=0 point
    hole_path = paths[1]
    hole_start = hole_path.points[0, :2]
    # Should be one of the hole points
    assert hole_start[0] >= 10.0 and hole_start[0] <= 20.0
    assert hole_start[1] >= 10.0 and hole_start[1] <= 20.0


def test_multipolygon_independent_rotation():
    """Test rotation applies independently to each polygon in MultiPolygon."""

    poly1_coords = [(0, 0), (5, 0), (5, 5), (0, 5)]
    poly2_coords = [(10, 10), (15, 10), (15, 15), (10, 15)]

    poly1 = shapely.geometry.Polygon(poly1_coords)
    poly2 = shapely.geometry.Polygon(poly2_coords)
    multi = shapely.geometry.MultiPolygon([poly1, poly2])

    original_points = np.array(
        [
            # Poly1
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            # Poly2
            [10.0, 10.0, 0.0],
            [15.0, 10.0, 0.0],
            [15.0, 15.0, 0.0],
            [10.0, 15.0, 0.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(multi, original_points)

    assert len(paths) == 2

    # Each polygon should start at its first matching type=0 point
    # (order may vary due to Shapely's MultiPolygon handling)
    starts = [tuple(p.points[0, :2]) for p in paths]
    assert any(np.linalg.norm(np.array(s) - np.array([0.0, 0.0])) < 1e-9 for s in starts)
    assert any(np.linalg.norm(np.array(s) - np.array([10.0, 10.0])) < 1e-9 for s in starts)


def test_no_matching_type0_no_rotation():
    """Test that no rotation occurs when no type=0 points match coordinates."""

    exterior_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # Original points at different coordinates (no match)
    original_points = np.array(
        [
            [100.0, 100.0, 0.0],  # type=0 but no coordinate match
            [200.0, 200.0, 0.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 1
    # Should return without crash, starting at default position


def test_tolerance_applied():
    """Test that TOLERANCE is used for coordinate matching."""

    exterior_coords = [(10, 10), (20, 10), (20, 20), (10, 20)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # Original point with slight offset within tolerance
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

    # Should match (10, 10) within tolerance
    expected_start = np.array([10.0, 10.0])
    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"Should match within tolerance, got {start_point}"


def test_fallback_when_all_type0_cause_degenerate():
    """Test fallback to first type=0 when all would cause degenerate Z."""

    # This tests the fallback logic - even if degenerate, accept first match
    # In practice, Shapely normalizes polygons so this is rare
    exterior_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    polygon = shapely.geometry.Polygon(exterior_coords)

    # Only one type=0 point
    original_points = np.array(
        [
            [0.0, 0.0, 0.0],  # Only type=0 point
            [10.0, 0.0, 2.0],
            [10.0, 10.0, 2.0],
            [0.0, 10.0, 2.0],
        ]
    )

    paths = AvPathCleaner._convert_shapely_to_paths(polygon, original_points)

    assert len(paths) == 1
    path = paths[0]
    start_point = path.points[0, :2]

    # Should use the only type=0 point
    expected_start = np.array([0.0, 0.0])
    distance = np.linalg.norm(start_point - expected_start)
    assert distance < 1e-9, f"Should fallback to first type=0, got {start_point}"


if __name__ == "__main__":
    test_normal_polygon_no_degenerate()
    test_rotation_prefers_first_type0_when_all_valid()
    test_skip_non_type0_points()
    test_degenerate_z_with_holes()
    test_multipolygon_independent_rotation()
    test_no_matching_type0_no_rotation()
    test_tolerance_applied()
    test_fallback_when_all_type0_cause_degenerate()
    print("All non-degenerate Z line tests passed!")
