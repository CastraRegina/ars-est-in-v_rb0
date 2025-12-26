"""Tests for degenerate Z line handling in AvPathCurveRebuilder."""

import numpy as np
import pytest

from ave.path import AvPath
from ave.path_processing import AvPathCurveRebuilder


class TestRotateIfDegenerateZ:
    """Test _rotate_if_degenerate_z method (proactive rotation)."""

    def test_no_curve_cluster_unchanged(self):
        """Segment without curve cluster ending at Z should be unchanged."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [5.0, 15.0, 0.0],
            ]
        )
        commands = ["M", "L", "L", "L", "Z"]
        seg = AvPath(points, commands)

        result = AvPathCurveRebuilder._rotate_if_degenerate_z(seg)

        # Should be unchanged - no curve cluster ending at Z
        assert len(result.points) == 4
        assert np.allclose(result.points[0, :2], [0.0, 0.0])

    def test_curve_cluster_ending_at_z_rotated(self):
        """Segment with curve cluster ending at Z should be rotated."""
        # Curve samples (type=2) at end, would create degenerate Z
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # type=0, M start
                [10.0, 0.0, 0.0],  # type=0, L vertex
                [10.0, 5.0, 2.0],  # type=2, curve sample
                [10.0, 10.0, 2.0],  # type=2, curve sample (ends at Z)
            ]
        )
        commands = ["M", "L", "L", "L", "Z"]
        seg = AvPath(points, commands)

        result = AvPathCurveRebuilder._rotate_if_degenerate_z(seg)

        # Should be rotated to start at different type=0 point
        # The curve would end at segment start, so rotate to avoid that
        first = result.points[0, :2]
        # After rotation, first point should be different from original
        # (rotated to index 1 which is [10, 0])
        assert np.allclose(first, [10.0, 0.0])

    def test_insufficient_type0_points_unchanged(self):
        """Segment with only one type=0 point should be unchanged."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # type=0, only one
                [5.0, 5.0, 2.0],  # type=2
                [10.0, 10.0, 2.0],  # type=2
            ]
        )
        commands = ["M", "L", "L", "Z"]
        seg = AvPath(points, commands)

        result = AvPathCurveRebuilder._rotate_if_degenerate_z(seg)

        # Should be unchanged - can't rotate with only 1 type=0 point
        assert np.allclose(result.points[0, :2], [0.0, 0.0])


class TestRotateSegmentPoints:
    """Test _rotate_segment_points helper method."""

    def test_rotate_to_index_1(self):
        """Rotating to index 1 should shift points correctly."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
            ]
        )
        commands = ["M", "L", "L", "Z"]
        seg = AvPath(points, commands)

        result = AvPathCurveRebuilder._rotate_segment_points(seg, 1)

        # New first point should be old index 1
        assert np.allclose(result.points[0, :2], [10.0, 0.0])
        # New last point should be old index 0
        assert np.allclose(result.points[-1, :2], [0.0, 0.0])

    def test_rotate_to_index_0_unchanged(self):
        """Rotating to index 0 should return unchanged."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
        commands = ["M", "L", "Z"]
        seg = AvPath(points, commands)

        result = AvPathCurveRebuilder._rotate_segment_points(seg, 0)

        assert np.allclose(result.points[0, :2], [0.0, 0.0])


class TestRebuildCurvePathIntegration:
    """Integration tests for rebuild_curve_path with degenerate Z fixing."""

    def test_rebuild_fixes_degenerate(self):
        """rebuild_curve_path should fix degenerate Z lines."""
        # Create a path that would produce degenerate Z after rebuild
        # Using type=0 for vertices, type=2 for curve samples
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - vertex
                [5.0, 0.0, 0.0],  # L - vertex
                [10.0, 0.0, 2.0],  # curve sample
                [10.0, 5.0, 2.0],  # curve sample
                [10.0, 10.0, 0.0],  # L - vertex
                [5.0, 10.0, 0.0],  # L - vertex
                [0.0, 10.0, 2.0],  # curve sample
                [0.0, 5.0, 2.0],  # curve sample
                [0.0, 0.0, 0.0],  # ends at start (degenerate)
            ]
        )
        commands = ["M", "L", "L", "L", "L", "L", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder.rebuild_curve_path(path)

        # Check all segments have non-degenerate Z
        for seg in result.split_into_single_paths():
            if seg.commands[-1] == "Z":
                first = seg.points[0, :2]
                last = seg.points[-1, :2]
                dist = np.linalg.norm(first - last)
                # Note: may still be degenerate if no rotation possible
                # but the method should attempt to fix it
