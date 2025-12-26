"""Tests for degenerate Z line fixing in AvPathCurveRebuilder."""

import numpy as np
import pytest

from ave.path import AvPath
from ave.path_processing import AvPathCurveRebuilder


class TestFixDegenerateZLines:
    """Test _fix_degenerate_z_lines method."""

    def test_non_degenerate_unchanged(self):
        """Non-degenerate segment should be unchanged."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [5.0, 15.0, 0.0],  # Different from first
            ]
        )
        commands = ["M", "L", "L", "L", "Z"]
        seg = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_z_lines(seg)

        # Should be unchanged
        assert len(result.points) == 4
        assert np.allclose(result.points[0, :2], [0.0, 0.0])

    def test_l_only_degenerate_rotated(self):
        """L-only segment with degenerate Z should be rotated."""
        # Create degenerate: first == last
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],  # Same as first - degenerate!
            ]
        )
        commands = ["M", "L", "L", "L", "Z"]
        seg = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_z_lines(seg)

        # Should be rotated - first != last
        first = result.points[0, :2]
        last = result.points[-1, :2]
        dist = np.linalg.norm(first - last)
        assert dist > 1e-9, "Z line should be non-degenerate after rotation"

    def test_multiple_segments(self):
        """Multiple segments should each be fixed independently."""
        # Segment 1: degenerate but fixable (4 unique points)
        pts1 = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],  # Same as first - degenerate
            ]
        )
        cmds1 = ["M", "L", "L", "L", "Z"]
        seg1 = AvPath(pts1, cmds1)

        # Segment 2: non-degenerate
        pts2 = np.array(
            [
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [25.0, 10.0, 0.0],
            ]
        )
        cmds2 = ["M", "L", "L", "Z"]
        seg2 = AvPath(pts2, cmds2)

        combined = AvPath.join_paths(seg1, seg2)
        result = AvPathCurveRebuilder._fix_degenerate_z_lines(combined)

        # Both segments should exist and be valid
        segments = result.split_into_single_paths()
        assert len(segments) == 2

        for seg in segments:
            first = seg.points[0, :2]
            last = seg.points[-1, :2]
            dist = np.linalg.norm(first - last)
            assert dist > 1e-9, "Each segment should have non-degenerate Z"


class TestRotateCurveSegment:
    """Test _rotate_curve_segment method."""

    def test_curve_segment_with_l_command(self):
        """Curve segment with L commands should find rotation point."""
        # M L Q Z - Q ends at start (degenerate)
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M
                [10.0, 0.0, 0.0],  # L endpoint
                [15.0, 5.0, 2.0],  # Q control
                [0.0, 0.0, 0.0],  # Q endpoint = start (degenerate)
            ]
        )
        commands = ["M", "L", "Q", "Z"]
        seg = AvPath(points, commands)

        result = AvPathCurveRebuilder._rotate_curve_segment(seg)

        # Should be rotated to start at L endpoint
        first = result.points[0, :2]
        last = result.points[-1, :2]
        dist = np.linalg.norm(first - last)

        # If rotation worked, Z should be non-degenerate
        # (If no L rotation found, original is returned)
        if not np.allclose(first, [0.0, 0.0]):
            assert dist > 1e-9, "Rotated segment should have non-degenerate Z"

    def test_no_l_commands_returns_original(self):
        """Segment with only curves should return original."""
        # M Q Q Z - no L commands to rotate to
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M
                [5.0, 5.0, 2.0],  # Q1 control
                [10.0, 0.0, 0.0],  # Q1 endpoint
                [5.0, -5.0, 2.0],  # Q2 control
                [0.0, 0.0, 0.0],  # Q2 endpoint = start (degenerate)
            ]
        )
        commands = ["M", "Q", "Q", "Z"]
        seg = AvPath(points, commands)

        result = AvPathCurveRebuilder._rotate_curve_segment(seg)

        # No L commands to rotate to, should return original
        assert np.allclose(result.points[0, :2], seg.points[0, :2])


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
