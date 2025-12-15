"""Tests for path helper functions"""

import numpy as np
import pytest

from ave.path_helper import AvPointMatcher


class TestAvPointMatcher:
    """Test cases for AvPointMatcher class"""

    def test_transfer_point_types_basic(self):
        """Test basic point type transfer with non-exact matches"""
        # Original points with types: on-curve, quadratic control, cubic control
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],  # on-curve
                [1.0, 1.0, 2.0],  # quadratic control
                [2.0, 2.0, 3.0],  # cubic control
            ],
            dtype=np.float64,
        )

        # New points close to originals but not exact (> epsilon)
        points_new = np.array(
            [
                [0.1, 0.1],  # should match first (type 0.0 -> -1.0)
                [1.9, 1.9],  # should match third (type 3.0 -> -3.0)
            ],
            dtype=np.float64,
        )

        result = AvPointMatcher.transfer_point_types(points_org, points_new)

        assert result.shape == (2, 3)
        assert result[0, 2] == -1.0  # non-exact match to on-curve
        assert result[1, 2] == -3.0  # non-exact match to cubic control

    def test_transfer_point_types_exact_match(self):
        """Test point type transfer with exact coordinate matches"""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 2.0],
                [10.0, 10.0, 3.0],
            ],
            dtype=np.float64,
        )

        points_new = np.array(
            [
                [5.0, 5.0],  # exact match
                [10.0, 10.0],  # exact match
            ],
            dtype=np.float64,
        )

        result = AvPointMatcher.transfer_point_types(points_org, points_new)

        assert result[0, 2] == 2.0
        assert result[1, 2] == 3.0

    def test_transfer_point_types_empty_org(self):
        """Test with empty original points"""
        points_org = np.empty((0, 3), dtype=np.float64)
        points_new = np.array([[1.0, 1.0]], dtype=np.float64)

        result = AvPointMatcher.transfer_point_types(points_org, points_new)

        assert result.shape == (1, 3)
        assert result[0, 2] == -1.0  # default to non-exact on-curve

    def test_transfer_point_types_empty_new(self):
        """Test with empty new points"""
        points_org = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
        points_new = np.empty((0, 2), dtype=np.float64)

        result = AvPointMatcher.transfer_point_types(points_org, points_new)

        assert result.shape == (0, 3)

    def test_transfer_point_types_ordered_shifted_start_closed(self):
        """Test ordered transfer with shifted start on closed sequence (cyclic matching)"""
        points_org_xy = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=np.float64,
        )
        points_org = np.column_stack(
            [
                points_org_xy,
                np.array([0.0, 2.0, 3.0, 0.0], dtype=np.float64),
            ]
        )

        points_new = points_org_xy[[2, 3, 0, 1]]
        result = AvPointMatcher.transfer_point_types_ordered(points_org, points_new, search_window=1, is_closed=True)

        assert result.shape == (4, 3)
        np.testing.assert_array_equal(result[:, 2], [3.0, 0.0, 0.0, 2.0])

    def test_transfer_point_types_ordered_realignment_mid_sequence(self):
        """Test ordered transfer with realignment in the middle of a closed sequence"""
        n = 12
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        org_xy = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float64)

        org_types = np.array([0.0, 2.0, 3.0] * 4, dtype=np.float64)
        points_org = np.column_stack([org_xy, org_types])

        idx = np.array([3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6], dtype=int)
        points_new = org_xy[idx]

        result = AvPointMatcher.transfer_point_types_ordered(points_org, points_new, search_window=2, is_closed=True)
        assert result.shape == (n, 3)
        np.testing.assert_array_equal(result[:, 2], org_types[idx])

    def test_transfer_point_types_ordered_explicit_open(self):
        """Test ordered transfer with explicit open sequence (should anchor at start)"""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 3.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Shifted sequence but still open
        points_new = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float64,
        )

        result = AvPointMatcher.transfer_point_types_ordered(points_org, points_new, search_window=1, is_closed=False)

        assert result.shape == (3, 3)
        # With explicit open, should anchor at start (index 0) and not use cyclic matching
        np.testing.assert_array_equal(result[:, 2], [2.0, 3.0, 0.0])

    def test_transfer_point_types_both_empty(self):
        """Test with both arrays empty"""
        points_org = np.empty((0, 3), dtype=np.float64)
        points_new = np.empty((0, 2), dtype=np.float64)

        result = AvPointMatcher.transfer_point_types(points_org, points_new)

        assert result.shape == (0, 3)

    def test_transfer_point_types_single_point(self):
        """Test with single points (non-exact match)"""
        points_org = np.array([[5.0, 5.0, 2.0]], dtype=np.float64)
        points_new = np.array([[0.0, 0.0]], dtype=np.float64)

        result = AvPointMatcher.transfer_point_types(points_org, points_new)

        assert result.shape == (1, 3)
        assert result[0, 2] == -2.0  # non-exact match to quadratic

    def test_transfer_point_types_many_to_one(self):
        """Test multiple new points matching same original point (non-exact)"""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 100.0, 3.0],
            ],
            dtype=np.float64,
        )

        points_new = np.array(
            [
                [0.1, 0.0],
                [0.0, 0.1],
                [-0.1, 0.0],
            ],
            dtype=np.float64,
        )

        result = AvPointMatcher.transfer_point_types(points_org, points_new)

        # All should match the first point (non-exact)
        assert all(result[:, 2] == -1.0)

    def test_transfer_point_types_ordered_forward(self):
        """Test ordered transfer with forward-aligned sequences (non-exact)"""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 3.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        points_new = np.array(
            [
                [0.1, 0.0],
                [1.1, 0.0],
                [2.1, 0.0],
                [3.1, 0.0],
            ],
            dtype=np.float64,
        )

        result = AvPointMatcher.transfer_point_types_ordered(points_org, points_new)

        assert result.shape == (4, 3)
        np.testing.assert_array_equal(result[:, 2], [-1.0, -2.0, -3.0, -1.0])

    def test_transfer_point_types_ordered_reversed(self):
        """Test ordered transfer with reversed sequences (non-exact)"""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 3.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Reversed order
        points_new = np.array(
            [
                [3.1, 0.0],
                [2.1, 0.0],
                [1.1, 0.0],
                [0.1, 0.0],
            ],
            dtype=np.float64,
        )

        result = AvPointMatcher.transfer_point_types_ordered(points_org, points_new)

        assert result.shape == (4, 3)
        np.testing.assert_array_equal(result[:, 2], [-1.0, -3.0, -2.0, -1.0])

    def test_transfer_point_types_ordered_empty(self):
        """Test ordered transfer with empty arrays"""
        points_org = np.empty((0, 3), dtype=np.float64)
        points_new = np.empty((0, 2), dtype=np.float64)

        result = AvPointMatcher.transfer_point_types_ordered(points_org, points_new)

        assert result.shape == (0, 3)

    def test_find_nearest_indices_basic(self):
        """Test finding nearest indices"""
        points_org = np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
            ],
            dtype=np.float64,
        )

        points_new = np.array(
            [
                [1.0, 0.0],  # nearest to index 0
                [9.0, 0.0],  # nearest to index 1
                [10.0, 9.0],  # nearest to index 2
            ],
            dtype=np.float64,
        )

        indices, distances = AvPointMatcher.find_nearest_indices(points_org, points_new)

        np.testing.assert_array_equal(indices, [0, 1, 2])
        assert distances[0] == pytest.approx(1.0)
        assert distances[1] == pytest.approx(1.0)
        assert distances[2] == pytest.approx(1.0)

    def test_find_nearest_indices_empty(self):
        """Test finding nearest indices with empty arrays"""
        points_org = np.empty((0, 2), dtype=np.float64)
        points_new = np.array([[1.0, 1.0]], dtype=np.float64)

        indices, distances = AvPointMatcher.find_nearest_indices(points_org, points_new)

        assert len(indices) == 1
        assert len(distances) == 1

    def test_find_exact_matches_basic(self):
        """Test finding exact matches"""
        points_org = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=np.float64,
        )

        points_new = np.array(
            [
                [0.0, 0.0],  # exact match with index 0
                [1.5, 1.5],  # no exact match
                [2.0, 2.0],  # exact match with index 2
            ],
            dtype=np.float64,
        )

        matches = AvPointMatcher.find_exact_matches(points_org, points_new)

        assert (0, 0) in matches
        assert (2, 2) in matches
        assert len(matches) == 2

    def test_find_exact_matches_with_tolerance(self):
        """Test finding exact matches with custom tolerance"""
        points_org = np.array([[0.0, 0.0]], dtype=np.float64)
        points_new = np.array([[0.001, 0.0]], dtype=np.float64)

        # Default tolerance should not match
        matches_tight = AvPointMatcher.find_exact_matches(points_org, points_new)
        assert len(matches_tight) == 0

        # Larger tolerance should match
        matches_loose = AvPointMatcher.find_exact_matches(points_org, points_new, tolerance=0.01)
        assert len(matches_loose) == 1

    def test_find_exact_matches_empty(self):
        """Test finding exact matches with empty arrays"""
        points_org = np.empty((0, 2), dtype=np.float64)
        points_new = np.array([[1.0, 1.0]], dtype=np.float64)

        matches = AvPointMatcher.find_exact_matches(points_org, points_new)

        assert not matches

    def test_transfer_types_two_stage_basic(self):
        """Test two-stage transfer with mostly ordered points (non-exact)"""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 3.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )

        # Mostly ordered, but one point is displaced
        points_new = np.array(
            [
                [0.1, 0.0],  # close to index 0
                [1.1, 0.0],  # close to index 1
                [1.8, 0.0],  # close to index 2
                [4.5, 0.0],  # displaced - should trigger KD-tree fallback
                [3.1, 0.0],  # close to index 3
            ],
            dtype=np.float64,
        )

        result = AvPointMatcher.transfer_types_two_stage(points_org, points_new)

        assert result.shape == (5, 3)
        # All are non-exact matches (distance > epsilon)
        assert result[0, 2] == -1.0
        assert result[1, 2] == -2.0
        assert result[2, 2] == -3.0
        assert result[4, 2] == -1.0
        # Displaced point should use KD-tree and match closest (index 4)
        assert result[3, 2] == -2.0

    def test_transfer_types_two_stage_no_fallback(self):
        """Test two-stage transfer when all points are well-ordered (non-exact)"""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        )

        points_new = np.array(
            [
                [0.1, 0.0],
                [1.1, 0.0],
                [2.1, 0.0],
            ],
            dtype=np.float64,
        )

        result = AvPointMatcher.transfer_types_two_stage(points_org, points_new)

        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result[:, 2], [-1.0, -2.0, -3.0])

    def test_transfer_types_two_stage_all_fallback(self):
        """Test two-stage transfer when all points exceed threshold (non-exact)"""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        )

        # Points are far from expected positions
        points_new = np.array(
            [
                [10.0, 10.0],
                [11.0, 10.0],
                [12.0, 10.0],
            ],
            dtype=np.float64,
        )

        result = AvPointMatcher.transfer_types_two_stage(points_org, points_new, max_residual=0.1)

        assert result.shape == (3, 3)
        # All should use KD-tree fallback (non-exact matches)
        # Each new point should match nearest original point
        # Since all new points are at y=10.0, distance is purely based on x coordinate
        # (10.0, 10.0) is closest to (2.0, 0.0) with distance sqrt(8^2 + 10^2)
        # (11.0, 10.0) is closest to (2.0, 0.0) with distance sqrt(9^2 + 10^2)
        # (12.0, 10.0) is closest to (2.0, 0.0) with distance sqrt(10^2 + 10^2)
        assert result[0, 2] == -3.0  # closest to (2,0) which has type 3.0 -> -3.0
        assert result[1, 2] == -3.0  # closest to (2,0) which has type 3.0 -> -3.0
        assert result[2, 2] == -3.0  # closest to (2,0) which has type 3.0 -> -3.0

    def test_transfer_types_two_stage_reversed(self):
        """Test two-stage transfer with reversed sequences (non-exact)"""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 3.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Reversed order
        points_new = np.array(
            [
                [3.1, 0.0],
                [2.1, 0.0],
                [1.1, 0.0],
                [0.1, 0.0],
            ],
            dtype=np.float64,
        )

        result = AvPointMatcher.transfer_types_two_stage(points_org, points_new)

        assert result.shape == (4, 3)
        np.testing.assert_array_equal(result[:, 2], [-1.0, -3.0, -2.0, -1.0])

    def test_transfer_types_two_stage_empty(self):
        """Test two-stage transfer with empty arrays"""
        points_org = np.empty((0, 3), dtype=np.float64)
        points_new = np.empty((0, 2), dtype=np.float64)

        result = AvPointMatcher.transfer_types_two_stage(points_org, points_new)

        assert result.shape == (0, 3)

    def test_transfer_types_two_stage_single_points(self):
        """Test two-stage transfer with single points (non-exact)"""
        points_org = np.array([[5.0, 5.0, 2.0]], dtype=np.float64)
        points_new = np.array([[5.1, 5.1]], dtype=np.float64)

        result = AvPointMatcher.transfer_types_two_stage(points_org, points_new)

        assert result.shape == (1, 3)
        assert result[0, 2] == -2.0

    def test_transfer_types_two_stage_custom_threshold(self):
        """Test two-stage transfer with custom residual threshold (non-exact)"""
        points_org = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        )

        # One point slightly displaced
        points_new = np.array(
            [
                [0.1, 0.0],
                [1.5, 0.0],  # 0.5 units from expected
                [2.1, 0.0],
            ],
            dtype=np.float64,
        )

        # With low threshold, middle point should trigger KD-tree
        result_low = AvPointMatcher.transfer_types_two_stage(points_org, points_new, max_residual=0.1)
        # With high threshold, all should use ordered matching
        result_high = AvPointMatcher.transfer_types_two_stage(points_org, points_new, max_residual=1.0)

        # Both should produce correct types (all non-exact)
        np.testing.assert_array_equal(result_low[:, 2], [-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(result_high[:, 2], [-1.0, -2.0, -3.0])
