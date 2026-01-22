"""Tests for AvPathCleaner.remove_duplicate_consecutive_points method for AvMultiPolylinePath."""

from __future__ import annotations

import numpy as np
import pytest

from ave.path import AvMultiPolylinePath
from ave.path_processing import AvPathCleaner
from ave.path_support import MULTI_POLYLINE_CONSTRAINTS


class TestRemoveDuplicateConsecutivePointsPolyline:
    """Test suite for remove_duplicate_consecutive_points method for polygonized paths."""

    def test_empty_path(self):
        """Test with empty path."""
        path = AvMultiPolylinePath()
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 0
        assert len(result.commands) == 0

    def test_single_point(self):
        """Test with single point path."""
        points = [(10.0, 20.0, 0.0)]
        commands = ["M"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 1
        assert len(result.commands) == 1
        np.testing.assert_array_almost_equal(result.points[0], [10.0, 20.0, 0.0])

    def test_no_duplicates(self):
        """Test path with no duplicates."""
        points = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)]
        commands = ["M", "L", "L", "L", "Z"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 4
        assert len(result.commands) == 5
        assert result.commands == ["M", "L", "L", "L", "Z"]

    def test_duplicate_type0_points(self):
        """Test duplicate type=0 points - keep first."""
        points = [(0.0, 0.0, 0.0), (10.0, 10.0, 0.0), (10.0, 10.0, 0.0), (20.0, 20.0, 0.0)]  # Duplicate type=0
        commands = ["M", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 3
        assert len(result.commands) == 3
        assert result.commands == ["M", "L", "L"]
        np.testing.assert_array_almost_equal(result.points[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result.points[1], [10.0, 10.0, 0.0])
        np.testing.assert_array_almost_equal(result.points[2], [20.0, 20.0, 0.0])

    def test_duplicate_type2_points(self):
        """Test duplicate type=2 points - keep first."""
        points = [(0.0, 0.0, 0.0), (5.0, 5.0, 2.0), (5.0, 5.0, 2.0), (10.0, 10.0, 0.0)]  # Duplicate type=2
        commands = ["M", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 3
        assert len(result.commands) == 3
        assert result.commands == ["M", "L", "L"]

    def test_type0_replaces_type2(self):
        """Test type=0 point replaces duplicate type=2 point."""
        points = [
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 2.0),  # type=2 curve sample
            (10.0, 10.0, 0.0),  # type=0 vertex at same location - should replace
            (20.0, 20.0, 0.0),
        ]
        commands = ["M", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 3
        assert len(result.commands) == 3
        # Should keep type=0 point, not type=2
        assert result.points[1][2] == 0.0
        np.testing.assert_array_almost_equal(result.points[1], [10.0, 10.0, 0.0])

    def test_type0_replaces_type3(self):
        """Test type=0 point replaces duplicate type=3 point."""
        points = [
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 3.0),  # type=3 curve sample
            (10.0, 10.0, 0.0),  # type=0 vertex at same location - should replace
            (20.0, 20.0, 0.0),
        ]
        commands = ["M", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 3
        assert len(result.commands) == 3
        # Should keep type=0 point, not type=3
        assert result.points[1][2] == 0.0

    def test_type2_kept_when_followed_by_type3(self):
        """Test type=2 point kept when followed by duplicate type=3 point."""
        points = [
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 2.0),  # type=2 curve sample
            (10.0, 10.0, 3.0),  # type=3 at same location - type=2 has priority
            (20.0, 20.0, 0.0),
        ]
        commands = ["M", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 3
        assert len(result.commands) == 3
        # Should keep type=2 point (lower type value has priority among non-zero types)
        assert result.points[1][2] == 2.0

    def test_multiple_consecutive_duplicates_mixed_types(self):
        """Test multiple consecutive duplicates with mixed types."""
        points = [
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 2.0),  # type=2
            (10.0, 10.0, 2.0),  # type=2 duplicate
            (10.0, 10.0, 3.0),  # type=3 duplicate
            (10.0, 10.0, 0.0),  # type=0 - should win
            (20.0, 20.0, 0.0),
        ]
        commands = ["M", "L", "L", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 3
        # Should keep type=0 point
        assert result.points[1][2] == 0.0
        np.testing.assert_array_almost_equal(result.points[1], [10.0, 10.0, 0.0])

    def test_polygonized_quadratic_curve_pattern(self):
        """Test realistic pattern from polygonized quadratic curve."""
        # Simulates: vertex (0) -> curve samples (2,2,2,...) -> vertex (0)
        points = [
            (0.0, 0.0, 0.0),  # Start vertex
            (2.0, 4.0, 2.0),  # Curve sample
            (4.0, 6.0, 2.0),  # Curve sample
            (6.0, 6.0, 2.0),  # Curve sample
            (8.0, 4.0, 2.0),  # Curve sample
            (10.0, 0.0, 0.0),  # End vertex
        ]
        commands = ["M", "L", "L", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        # No duplicates, all should be kept
        assert len(result.points) == 6
        assert len(result.commands) == 6

    def test_polygonized_cubic_curve_pattern(self):
        """Test realistic pattern from polygonized cubic curve."""
        # Simulates: vertex (0) -> curve samples (3,3,3,...) -> vertex (0)
        points = [
            (0.0, 0.0, 0.0),  # Start vertex
            (2.0, 5.0, 3.0),  # Curve sample
            (5.0, 8.0, 3.0),  # Curve sample
            (8.0, 5.0, 3.0),  # Curve sample
            (10.0, 0.0, 0.0),  # End vertex
        ]
        commands = ["M", "L", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        # No duplicates, all should be kept
        assert len(result.points) == 5
        assert len(result.commands) == 5

    def test_closed_path_with_z(self):
        """Test closed path with Z command."""
        points = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)]
        commands = ["M", "L", "L", "L", "Z"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 4
        assert len(result.commands) == 5
        assert result.commands == ["M", "L", "L", "L", "Z"]

    def test_multiple_segments(self):
        """Test path with multiple segments."""
        points = [(0.0, 0.0, 0.0), (10.0, 10.0, 0.0), (20.0, 0.0, 0.0), (30.0, 30.0, 0.0)]
        commands = ["M", "L", "M", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 4
        assert len(result.commands) == 4
        assert result.commands == ["M", "L", "M", "L"]

    def test_tolerance_parameter(self):
        """Test custom tolerance parameter."""
        points = [
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (10.0 + 1e-10, 10.0 + 1e-10, 2.0),  # Very close to previous
            (20.0, 20.0, 0.0),
        ]
        commands = ["M", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)

        # With default tolerance (1e-9), should remove duplicate
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)
        assert len(result.points) == 3

        # With very small tolerance, should keep all points
        result = AvPathCleaner.remove_duplicate_consecutive_points(path, tolerance=1e-12)
        assert len(result.points) == 4

    def test_real_world_case_vertex_and_curve_sample_coincide(self):
        """Test real-world case where vertex and curve sample have same coordinates."""
        # This simulates the case from the original bug report
        points = [
            (160.0, 82.0, 0.0),  # Vertex point
            (160.0, 82.0, 2.0),  # Quadratic sample at same location
            (170.0, 90.0, 0.0),  # Next vertex
        ]
        commands = ["M", "L", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        # Should keep type=0 vertex, remove type=2 duplicate
        assert len(result.points) == 2
        assert len(result.commands) == 2
        assert result.points[0][2] == 0.0
        assert result.points[1][2] == 0.0
        np.testing.assert_array_almost_equal(result.points[0], [160.0, 82.0, 0.0])
        np.testing.assert_array_almost_equal(result.points[1], [170.0, 90.0, 0.0])

    def test_preserves_point_types(self):
        """Test that point types are preserved correctly."""
        points = [
            (0.0, 0.0, 0.0),
            (5.0, 5.0, 2.0),
            (10.0, 10.0, 0.0),
            (15.0, 15.0, 3.0),
            (20.0, 20.0, 0.0),
        ]
        commands = ["M", "L", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert len(result.points) == 5
        # Check that types are preserved
        assert result.points[0][2] == 0.0
        assert result.points[1][2] == 2.0
        assert result.points[2][2] == 0.0
        assert result.points[3][2] == 3.0
        assert result.points[4][2] == 0.0

    def test_all_points_duplicate_different_types(self):
        """Test edge case where all points after M are duplicates with different types."""
        points = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 2.0),
            (0.0, 0.0, 3.0),
            (0.0, 0.0, 0.0),  # type=0 should win
        ]
        commands = ["M", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        # Should only keep one point with type=0
        assert len(result.points) == 1
        assert len(result.commands) == 1
        assert result.commands == ["M"]
        assert result.points[0][2] == 0.0

    def test_constraints_preserved(self):
        """Test that path constraints are preserved."""
        points = [(0.0, 0.0, 0.0), (10.0, 10.0, 0.0), (10.0, 10.0, 2.0), (20.0, 20.0, 0.0)]
        commands = ["M", "L", "L", "L", "Z"]
        path = AvMultiPolylinePath(points, commands, MULTI_POLYLINE_CONSTRAINTS)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        assert result.constraints == MULTI_POLYLINE_CONSTRAINTS

    def test_complex_mixed_type_sequence(self):
        """Test complex sequence with various type transitions."""
        points = [
            (0.0, 0.0, 0.0),  # M - vertex
            (5.0, 5.0, 2.0),  # L - quadratic sample
            (10.0, 10.0, 2.0),  # L - quadratic sample
            (15.0, 15.0, 0.0),  # L - vertex
            (20.0, 20.0, 3.0),  # L - cubic sample
            (25.0, 25.0, 3.0),  # L - cubic sample
            (30.0, 30.0, 0.0),  # L - vertex
        ]
        commands = ["M", "L", "L", "L", "L", "L", "L", "Z"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        # No duplicates, all should be kept
        assert len(result.points) == 7
        assert len(result.commands) == 8
        assert result.commands[-1] == "Z"

    def test_duplicate_at_segment_boundary(self):
        """Test duplicate at the boundary between two segments."""
        points = [
            (0.0, 0.0, 0.0),  # M - first segment
            (10.0, 10.0, 0.0),  # L
            (10.0, 10.0, 0.0),  # M - second segment (duplicate of previous)
            (20.0, 20.0, 0.0),  # L
        ]
        commands = ["M", "L", "M", "L"]
        path = AvMultiPolylinePath(points, commands)
        result = AvPathCleaner.remove_duplicate_consecutive_points(path)

        # M commands are always kept even if duplicate (they mark segment boundaries)
        assert len(result.points) == 4
        assert len(result.commands) == 4
        assert result.commands == ["M", "L", "M", "L"]
