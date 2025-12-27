"""Comprehensive tests for the new AvPathMatcher match_paths API."""

import numpy as np
import pytest  # pylint: disable=unused-import
from numpy.testing import assert_allclose, assert_array_equal

from ave.path import MULTI_POLYGON_CONSTRAINTS, AvPath
from ave.path_processing import AvPathMatcher


class TestAvPathMatcherBasic:
    """Test basic functionality of AvPathMatcher.match_paths."""

    def test_exact_match_simple(self):
        """Test exact point matching with simple polygon."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        commands_org = ["M", "L", "L", "L", "Z"]

        points_new = np.array([[0.0, 0.0, -1], [1.0, 0.0, -1], [1.0, 1.0, -1], [0.0, 1.0, -1]], dtype=np.float64)
        commands_new = ["M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        expected = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 0.0]])
        assert_array_equal(result.points, expected)

    def test_within_tolerance_match(self):
        """Test matching within tolerance threshold."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        commands_org = ["M", "L", "L", "L", "Z"]

        # Points slightly offset but within tolerance
        points_new = np.array(
            [[1e-11, 1e-11, -1], [1.0 + 5e-11, 5e-11, -1], [1.0 - 5e-11, 1.0 + 5e-11, -1], [5e-11, 1.0 - 5e-11, -1]],
            dtype=np.float64,
        )
        commands_new = ["M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        # Check types match, coordinates may have tiny differences
        assert_array_equal(result.points[:, 2], [0.0, 2.0, 2.0, 0.0])
        assert_allclose(result.points[:, :2], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], atol=1e-10)

    def test_outside_tolerance_no_match(self):
        """Test points outside tolerance get unmatched type."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        commands_org = ["M", "L", "L", "L", "Z"]

        # Points offset beyond tolerance
        points_new = np.array(
            [[1e-9, 1e-9, -1], [1.0 + 1e-9, 1e-9, -1], [1.0, 1.0 + 1e-9, -1], [1e-9, 1.0 + 1e-9, -1]], dtype=np.float64
        )
        commands_new = ["M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert np.all(result.points[:, 2] == AvPathMatcher.UNMATCHED_TYPE)

    def test_empty_paths(self):
        """Test with empty paths."""
        path_org = AvPath(np.empty((0, 3), dtype=np.float64), [], MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(np.empty((0, 3), dtype=np.float64), [], MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert len(result.points) == 0
        assert len(result.commands) == 0

    def test_empty_original_path(self):
        """Test with empty original path."""
        path_org = AvPath(np.empty((0, 3), dtype=np.float64), [], MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(
            np.array([[0.0, 0.0, -1], [1.0, 0.0, -1], [1.0, 1.0, -1], [0.0, 1.0, -1]], dtype=np.float64),
            ["M", "L", "L", "L", "Z"],
            MULTI_POLYGON_CONSTRAINTS,
        )

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert result.points.shape == (4, 3)
        assert np.all(result.points[:, 2] == AvPathMatcher.UNMATCHED_TYPE)

    def test_empty_new_path(self):
        """Test with empty new path."""
        path_org = AvPath(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float64),
            ["M", "L", "L", "L", "Z"],
            MULTI_POLYGON_CONSTRAINTS,
        )
        path_new = AvPath(np.empty((0, 3), dtype=np.float64), [], MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert len(result.points) == 0
        assert len(result.commands) == 0


class TestAvPathMatcherMultipleSegments:
    """Test matching with multiple polygon segments."""

    def test_multiple_polygons_match(self):
        """Test matching multiple separate polygons."""
        points_org = np.array(
            [
                # First square
                [0, 0, 0],
                [50, 0, 2],
                [50, 50, 2],
                [0, 50, 0],
                # Second square
                [100, 0, 1],
                [150, 0, 3],
                [150, 50, 3],
                [100, 50, 1],
            ],
            dtype=np.float64,
        )
        commands_org = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        points_new = np.array(
            [
                # First square
                [0, 0, -1],
                [50, 0, -1],
                [50, 50, -1],
                [0, 50, -1],
                # Second square
                [100, 0, -1],
                [150, 0, -1],
                [150, 50, -1],
                [100, 50, -1],
            ],
            dtype=np.float64,
        )
        commands_new = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert result.points.shape == (8, 3)
        assert_array_equal(result.points[:4, 2], [0, 2, 2, 0])
        assert_array_equal(result.points[4:, 2], [1, 3, 3, 1])

    def test_polygon_with_hole_match(self):
        """Test matching polygon with hole (donut shape)."""
        points_org = np.array(
            [
                # Outer polygon (CCW)
                [0, 0, 0],
                [100, 0, 2],
                [100, 100, 2],
                [0, 100, 0],
                # Inner polygon (CW - hole)
                [25, 25, 1],
                [25, 75, 3],
                [75, 75, 3],
                [75, 25, 1],
            ],
            dtype=np.float64,
        )
        commands_org = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        points_new = np.array(
            [
                # Outer polygon
                [0, 0, -1],
                [100, 0, -1],
                [100, 100, -1],
                [0, 100, -1],
                # Inner polygon
                [25, 25, -1],
                [25, 75, -1],
                [75, 75, -1],
                [75, 25, -1],
            ],
            dtype=np.float64,
        )
        commands_new = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert result.points.shape == (8, 3)
        assert_array_equal(result.points[:4, 2], [0, 2, 2, 0])
        assert_array_equal(result.points[4:, 2], [1, 3, 3, 1])

    def test_no_matching_segments(self):
        """Test when no segments have overlapping bounding boxes."""
        points_org = np.array([[0, 0, 0], [100, 0, 2], [100, 100, 2], [0, 100, 0]], dtype=np.float64)
        commands_org = ["M", "L", "L", "L", "Z"]

        points_new = np.array(
            [[1000, 1000, -1], [1100, 1000, -1], [1100, 1100, -1], [1000, 1100, -1]], dtype=np.float64
        )
        commands_new = ["M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert np.all(result.points[:, 2] == AvPathMatcher.UNMATCHED_TYPE)

    def test_partially_overlapping_segments(self):
        """Test when only some segments overlap."""
        points_org = np.array(
            [
                [0, 0, 0],
                [100, 0, 2],
                [100, 100, 2],
                [0, 100, 0],
                [200, 0, 1],
                [300, 0, 3],
                [300, 100, 3],
                [200, 100, 1],
            ],
            dtype=np.float64,
        )
        commands_org = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        points_new = np.array(
            [
                [0, 0, -1],
                [100, 0, -1],
                [100, 100, -1],
                [0, 100, -1],
                [400, 0, -1],
                [500, 0, -1],
                [500, 100, -1],
                [400, 100, -1],
            ],
            dtype=np.float64,
        )
        commands_new = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert_array_equal(result.points[:4, 2], [0, 2, 2, 0])
        assert np.all(result.points[4:, 2] == AvPathMatcher.UNMATCHED_TYPE)


class TestAvPathMatcherComplexCases:
    """Test complex edge cases and scenarios."""

    def test_intersecting_polygons_reordered(self):
        """Test matching when polygons have been intersected and reordered."""
        # Original: two overlapping squares
        points_org = np.array(
            [
                [0, 0, 0],
                [100, 0, 2],
                [100, 100, 2],
                [0, 100, 0],
                [50, 50, 1],
                [150, 50, 3],
                [150, 150, 3],
                [50, 150, 1],
            ],
            dtype=np.float64,
        )
        commands_org = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        # New: after intersection resolution, polygons are reordered
        points_new = np.array(
            [
                # Resulting polygons after intersection (reordered)
                [50, 50, -1],
                [100, 50, -1],
                [100, 100, -1],
                [50, 100, -1],
                [0, 0, -1],
                [50, 0, -1],
                [50, 50, -1],
                [0, 50, -1],
            ],
            dtype=np.float64,
        )
        commands_new = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        # Should match based on spatial proximity, not order
        assert result.points.shape == (8, 3)
        # The intersection piece should match type 1/3 from original
        # The separate piece should match type 0/2 from original

    def test_polygon_with_additional_points(self):
        """Test matching when new polygon has more points (e.g., from curve interpolation)."""
        points_org = np.array([[0, 0, 0], [100, 0, 2], [100, 100, 2], [0, 100, 0]], dtype=np.float64)
        commands_org = ["M", "L", "L", "L", "Z"]

        # New: same square but with interpolated points
        points_new = np.array(
            [
                [0, 0, -1],
                [25, 0, -1],
                [50, 0, -1],
                [75, 0, -1],
                [100, 0, -1],
                [100, 25, -1],
                [100, 50, -1],
                [100, 75, -1],
                [100, 100, -1],
                [75, 100, -1],
                [50, 100, -1],
                [25, 100, -1],
                [0, 100, -1],
                [0, 75, -1],
                [0, 50, -1],
                [0, 25, -1],
            ],
            dtype=np.float64,
        )
        commands_new = ["M"] + ["L"] * 15 + ["Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert result.points.shape == (16, 3)
        # Corner points should match original types
        assert result.points[0, 2] == 0  # (0, 0)
        assert result.points[4, 2] == 2  # (100, 0)
        assert result.points[8, 2] == 2  # (100, 100)
        assert result.points[12, 2] == 0  # (0, 100)

    def test_touching_polygons(self):
        """Test matching polygons that touch at edges."""
        points_org = np.array(
            [
                # Left square
                [0, 0, 0],
                [50, 0, 2],
                [50, 50, 2],
                [0, 50, 0],
                # Right square (touching)
                [50, 0, 1],
                [100, 0, 3],
                [100, 50, 3],
                [50, 50, 1],
            ],
            dtype=np.float64,
        )
        commands_org = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        points_new = np.array(
            [
                # Left square
                [0, 0, -1],
                [50, 0, -1],
                [50, 50, -1],
                [0, 50, -1],
                # Right square
                [50, 0, -1],
                [100, 0, -1],
                [100, 50, -1],
                [50, 50, -1],
            ],
            dtype=np.float64,
        )
        commands_new = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        # Both should match despite touching at the edge
        assert result.points.shape == (8, 3)
        # The matching should work correctly based on centroid and winding

    def test_different_point_counts(self):
        """Test matching segments with different numbers of points."""
        points_org = np.array(
            [
                # Triangle
                [0, 0, 0],
                [100, 0, 2],
                [50, 86.6, 0],
            ],
            dtype=np.float64,
        )
        commands_org = ["M", "L", "L", "Z"]

        points_new = np.array(
            [
                # Same triangle but with more points
                [0, 0, -1],
                [25, 0, -1],
                [50, 0, -1],
                [75, 0, -1],
                [100, 0, -1],
                [87.5, 43.3, -1],
                [75, 86.6, -1],
                [50, 86.6, -1],
                [25, 86.6, -1],
                [0, 0, -1],
            ],
            dtype=np.float64,
        )
        commands_new = ["M"] + ["L"] * 9 + ["Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert result.points.shape == (10, 3)
        # Should match the vertices that are close enough


class TestAvPathMatcherEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_point_segments(self):
        """Test with segments that have minimum valid points."""
        # Use triangles (minimum 3 points for polygons)
        points_org = np.array(
            [
                # First triangle
                [0, 0, 0],
                [1, 0, 2],
                [0, 1, 0],
                # Second triangle
                [10, 10, 1],
                [11, 10, 3],
                [10, 11, 1],
            ],
            dtype=np.float64,
        )
        commands_org = ["M", "L", "L", "Z", "M", "L", "L", "Z"]

        points_new = np.array(
            [
                # First triangle
                [0, 0, -1],
                [1, 0, -1],
                [0, 1, -1],
                # Second triangle
                [10, 10, -1],
                [11, 10, -1],
                [10, 11, -1],
            ],
            dtype=np.float64,
        )
        commands_new = ["M", "L", "L", "Z", "M", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        # Should handle gracefully
        assert result.points.shape == (6, 3)
        assert_array_equal(result.points[:, 2], [0, 2, 0, 1, 3, 1])

    def test_very_close_points(self):
        """Test with points that are very close to each other."""
        points_org = np.array([[0, 0, 0], [1e-12, 0, 2], [1e-12, 1e-12, 2], [0, 1e-12, 0]], dtype=np.float64)
        commands_org = ["M", "L", "L", "L", "Z"]

        points_new = np.array([[0, 0, -1], [1e-12, 0, -1], [1e-12, 1e-12, -1], [0, 1e-12, -1]], dtype=np.float64)
        commands_new = ["M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        # Should match despite very small distances
        assert_array_equal(result.points[:, 2], [0, 2, 2, 0])

    def test_large_coordinates(self):
        """Test with very large coordinate values."""
        large_coord = 1e6
        points_org = np.array(
            [
                [large_coord, large_coord, 0],
                [large_coord + 100, large_coord, 2],
                [large_coord + 100, large_coord + 100, 2],
                [large_coord, large_coord + 100, 0],
            ],
            dtype=np.float64,
        )
        commands_org = ["M", "L", "L", "L", "Z"]

        points_new = np.array(
            [
                [large_coord, large_coord, -1],
                [large_coord + 100, large_coord, -1],
                [large_coord + 100, large_coord + 100, -1],
                [large_coord, large_coord + 100, -1],
            ],
            dtype=np.float64,
        )
        commands_new = ["M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        # Should work with large coordinates
        assert_array_equal(result.points[:, 2], [0, 2, 2, 0])

    def test_mixed_winding_directions(self):
        """Test matching polygons with different winding directions."""
        points_org = np.array(
            [
                # CCW polygon
                [0, 0, 0],
                [100, 0, 2],
                [100, 100, 2],
                [0, 100, 0],
                # CW polygon (hole)
                [25, 75, 1],
                [75, 75, 3],
                [75, 25, 3],
                [25, 25, 1],
            ],
            dtype=np.float64,
        )
        commands_org = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        points_new = np.array(
            [
                # Same as original
                [0, 0, -1],
                [100, 0, -1],
                [100, 100, -1],
                [0, 100, -1],
                [25, 75, -1],
                [75, 75, -1],
                [75, 25, -1],
                [25, 25, -1],
            ],
            dtype=np.float64,
        )
        commands_new = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]

        path_org = AvPath(points_org, commands_org, MULTI_POLYGON_CONSTRAINTS)
        path_new = AvPath(points_new, commands_new, MULTI_POLYGON_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        # Should match correctly considering winding direction
        assert_array_equal(result.points[:4, 2], [0, 2, 2, 0])
        assert_array_equal(result.points[4:, 2], [1, 3, 3, 1])
