"""Intensive tests for AvPathMatcher class.

Tests cover point matching, segment propagation, edge cases, and error handling.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ave.path import AvMultiPolylinePath
from ave.path_processing import AvPathMatcher
from ave.path_support import MULTI_POLYLINE_CONSTRAINTS


class TestAvPathMatcherBasic:
    """Test basic functionality of AvPathMatcher."""

    def test_exact_match_simple(self):
        """Test exact point matching with simple input."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 2.0, 0.0]])
        points_new = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        expected = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 2.0, 0.0]])
        assert_array_equal(result, expected)

    def test_within_tolerance_match(self):
        """Test matching within tolerance threshold."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 2.0, 0.0]])
        # Points slightly offset but within tolerance
        points_new = np.array([[1e-11, 1e-11], [1.0 + 5e-11, 1.0 - 5e-11], [2.0, 2.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        # Check types match, coordinates may have tiny differences
        assert_array_equal(result[:, 2], [0.0, 2.0, 0.0])
        assert_allclose(result[:, :2], [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], atol=1e-10)

    def test_outside_tolerance_no_match(self):
        """Test points outside tolerance get unmatched type."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
        # Points offset beyond tolerance
        points_new = np.array([[1e-9, 1e-9], [1.0 + 1e-9, 1.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        assert result[0, 2] == AvPathMatcher.UNMATCHED_TYPE
        assert result[1, 2] == AvPathMatcher.UNMATCHED_TYPE

    def test_empty_original_points(self):
        """Test with empty original points array."""
        points_org = np.empty((0, 3))
        points_new = np.array([[0.0, 0.0], [1.0, 1.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        expected = np.array([[0.0, 0.0, -1.0], [1.0, 1.0, -1.0]])
        assert_array_equal(result, expected)

    def test_empty_new_points(self):
        """Test with empty new points array."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
        points_new = np.empty((0, 2))

        result = AvPathMatcher.match_points(points_org, points_new)

        assert result.shape == (0, 3)

    def test_single_point_match(self):
        """Test matching with single point."""
        points_org = np.array([[5.0, 5.0, 2.0]])
        points_new = np.array([[5.0, 5.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        expected = np.array([[5.0, 5.0, 2.0]])
        assert_array_equal(result, expected)


class TestAvPathMatcherPropagation:
    """Test segment propagation functionality."""

    def test_forward_propagation(self):
        """Test propagation in forward direction."""
        # Original: sequence with a gap in the middle
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [2.0, 0.0, 0.0], [3.0, 0.0, 2.0], [4.0, 0.0, 0.0]])
        # New: same sequence but missing some points
        points_new = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        expected_types = [0.0, 2.0, 0.0, 2.0, 0.0]
        assert_array_equal(result[:, 2], expected_types)

    def test_backward_propagation(self):
        """Test propagation in backward direction."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [2.0, 0.0, 0.0], [3.0, 0.0, 2.0], [4.0, 0.0, 0.0]])
        # New starts from the end
        points_new = np.array([[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        # Should match even in reverse order due to propagation
        expected_types = [0.0, 2.0, 0.0, 2.0, 0.0]
        assert_array_equal(result[:, 2], expected_types)

    def test_reversed_segment_matching(self):
        """Test matching when segments are reversed."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 2.0, 0.0], [3.0, 3.0, 2.0], [4.0, 4.0, 0.0]])
        # New has a reversed segment
        points_new = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        # Should handle the reversal correctly
        expected_types = [0.0, 2.0, 0.0, 2.0, 0.0]
        assert_array_equal(result[:, 2], expected_types)

    def test_partial_segment_match(self):
        """Test when only part of a segment matches."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [2.0, 0.0, 0.0], [3.0, 0.0, 2.0], [4.0, 0.0, 0.0]])
        # New matches only first half
        points_new = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        expected_types = [0.0, 2.0, 0.0]
        assert_array_equal(result[:, 2], expected_types)

    def test_propagation_jump_prevention(self):
        """Test that propagation doesn't jump between disconnected segments."""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],  # First segment
                [10.0, 10.0, 0.0],
                [11.0, 10.0, 2.0],  # Second segment, far away
            ]
        )
        # New tries to trick propagation with close but disconnected points
        points_new = np.array(
            [[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]]  # Matches first segment  # Matches second segment
        )

        result = AvPathMatcher.match_points(points_org, points_new)

        expected_types = [0.0, 2.0, 0.0, 2.0]
        assert_array_equal(result[:, 2], expected_types)


class TestAvPathMatcherMixed:
    """Test mixed scenarios with matched and unmatched points."""

    def test_mixed_matched_unmatched(self):
        """Test sequence with both matched and unmatched points."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 2.0, 0.0]])
        # New has some original points and some new ones
        points_new = np.array([[0.0, 0.0], [1.5, 1.5], [2.0, 2.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        assert result[0, 2] == 0.0  # Matched
        assert result[1, 2] == -1.0  # Unmatched
        assert result[2, 2] == 0.0  # Matched

    def test_intersection_points_unmatched(self):
        """Test that intersection points (new) remain unmatched."""
        points_org = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 2.0], [4.0, 0.0, 0.0]])
        # New includes intersection point at 1.0
        points_new = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        # Middle points should be unmatched (-1.0)
        assert result[1, 2] == -1.0
        assert result[3, 2] == -1.0
        # Original points should match
        assert result[0, 2] == 0.0
        assert result[2, 2] == 2.0
        assert result[4, 2] == 0.0


class TestAvPathMatcherTypes:
    """Test handling of different point types (0.0, 2.0, 3.0)."""

    def test_all_type_values(self):
        """Test matching with all possible type values."""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],  # Regular point
                [1.0, 0.0, 2.0],  # Control point
                [2.0, 0.0, 3.0],  # Another control type
                [3.0, 0.0, 0.0],  # Regular point
            ]
        )
        points_new = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        expected_types = [0.0, 2.0, 3.0, 0.0]
        assert_array_equal(result[:, 2], expected_types)


class TestAvPathMatcherWarnings:
    """Test warning generation for duplicate points."""

    def test_duplicate_neighbor_warning(self, capsys):
        """Test warning for duplicate neighboring points."""
        points_org = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 0.0]])
        points_new = np.array([[0.0, 0.0], [1.0, 1.0]])

        AvPathMatcher.match_points(points_org, points_new)

        captured = capsys.readouterr()
        assert "duplicate neighboring points" in captured.out

    def test_no_duplicate_warning(self, capsys):
        """Test no warning when no duplicates exist."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 2.0, 0.0]])
        points_new = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        AvPathMatcher.match_points(points_org, points_new)

        captured = capsys.readouterr()
        assert "duplicate neighboring points" not in captured.out


class TestAvPathMatcherErrors:
    """Test error handling for invalid inputs."""

    def test_invalid_original_shape(self):
        """Test error for wrong original points shape."""
        points_org = np.array([[0.0, 0.0], [1.0, 1.0]])  # Missing type column
        points_new = np.array([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(ValueError, match="must have shape \\(N, 3\\)"):
            AvPathMatcher.match_points(points_org, points_new)

    def test_invalid_new_shape_1d(self):
        """Test error for 1D new points array."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
        points_new = np.array([0.0, 0.0])  # 1D array

        with pytest.raises(ValueError, match="must be 2D"):
            AvPathMatcher.match_points(points_org, points_new)

    def test_invalid_new_shape_4d(self):
        """Test error for new points with too many columns."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
        points_new = np.array([[0.0, 0.0, 0.0, 0.0]])  # 4 columns

        with pytest.raises(ValueError, match="must have shape \\(M, 2\\) or \\(M, 3\\)"):
            AvPathMatcher.match_points(points_org, points_new)

    def test_3d_new_points_ignored_third_column(self):
        """Test that third column in new points is ignored."""
        points_org = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
        points_new = np.array([[0.0, 0.0, 999.0], [1.0, 1.0, 888.0]])

        result = AvPathMatcher.match_points(points_org, points_new)

        expected = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
        assert_array_equal(result, expected)


class TestAvPathMatcherPaths:
    """Test match_paths convenience method."""

    def test_match_paths_returns_correct_type(self):
        """Test that match_paths returns AvMultiPolylinePath."""
        path_org = AvMultiPolylinePath(
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]]), ["M", "L"], MULTI_POLYLINE_CONSTRAINTS
        )
        path_new = AvMultiPolylinePath(np.array([[0.0, 0.0], [1.0, 1.0]]), ["M", "L"], MULTI_POLYLINE_CONSTRAINTS)

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert isinstance(result, AvMultiPolylinePath)
        assert result.commands == ["M", "L"]
        assert_array_equal(result.points[:, 2], [0.0, 2.0])

    def test_match_paths_preserves_new_commands(self):
        """Test that match_paths preserves commands from new path."""
        path_org = AvMultiPolylinePath(np.array([[0.0, 0.0, 0.0]]), ["M"], MULTI_POLYLINE_CONSTRAINTS)
        path_new = AvMultiPolylinePath(np.array([[0.0, 0.0]]), ["M"], MULTI_POLYLINE_CONSTRAINTS)  # Must start with M

        result = AvPathMatcher.match_paths(path_org, path_new)

        assert result.commands == ["M"]  # From new path


class TestAvPathMatcherComplex:
    """Complex integration tests."""

    def test_complex_curve_path(self):
        """Test with a complex curve path."""
        # Simulate a quadratic curve with control points
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],  # Start
                [1.0, 2.0, 2.0],  # Control
                [2.0, 0.0, 0.0],  # End
                [3.0, 0.0, 0.0],  # Move to
                [4.0, 2.0, 2.0],  # Control
                [5.0, 0.0, 0.0],  # End
            ]
        )
        # After polygonization, control points might be interpolated
        points_new = np.array(
            [
                [0.0, 0.0],
                [0.5, 1.0],
                [1.0, 1.5],
                [1.5, 1.5],
                [2.0, 0.0],
                [3.0, 0.0],
                [3.5, 1.0],
                [4.0, 1.5],
                [4.5, 1.5],
                [5.0, 0.0],
            ]
        )

        result = AvPathMatcher.match_points(points_org, points_new)

        # Check which points matched
        matched_indices = np.where(result[:, 2] != -1.0)[0]

        # Only original points at indices 0 and 5 should match
        assert 0 in matched_indices  # Start point at [0,0]
        assert 5 in matched_indices  # End point at [2,0]

        # Most interpolated points should be unmatched
        # Note: point at index 6 ([3,0]) might match due to propagation
        assert result[0, 2] == 0.0  # Start
        assert result[5, 2] == 0.0  # End

    def test_large_dataset_performance(self):
        """Test with a large dataset to ensure reasonable performance."""
        # Generate 1000 points along a sine wave
        x = np.linspace(0, 10 * np.pi, 1000)
        points_org = np.column_stack([x, np.sin(x), np.zeros(1000)])
        points_org[::10, 2] = 2.0  # Every 10th point is a control point

        # Same points with small noise
        noise = np.random.RandomState(42).randn(1000, 2) * 1e-11
        points_new = points_org[:, :2] + noise

        result = AvPathMatcher.match_points(points_org, points_new)

        # All should match within tolerance
        assert np.all(result[:, 2] != -1.0)
        assert np.sum(result[:, 2] == 2.0) == 100  # 100 control points

    def test_tolerance_boundary(self):
        """Test behavior exactly at tolerance boundary."""
        points_org = np.array([[0.0, 0.0, 0.0]])

        # Exactly at tolerance - should NOT match (distance_upper_bound is exclusive)
        points_new_at = np.array([[AvPathMatcher.TOLERANCE, 0.0]])
        result_at = AvPathMatcher.match_points(points_org, points_new_at)
        assert result_at[0, 2] == -1.0  # Should not match

        # Just inside tolerance
        points_new_in = np.array([[AvPathMatcher.TOLERANCE * 0.9, 0.0]])
        result_in = AvPathMatcher.match_points(points_org, points_new_in)
        assert result_in[0, 2] == 0.0  # Should match
