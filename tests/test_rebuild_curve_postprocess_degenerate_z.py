"""Tests for post-processing degenerate Z handling in AvPathCurveRebuilder."""

import numpy as np
import pytest

from ave.path import AvPath
from ave.path_processing import AvPathCurveRebuilder


class TestPostProcessDegenerateZ:
    """Test _fix_degenerate_z_lines method (post-processing fallback)."""

    def test_degenerate_with_l_commands_rotates(self):
        """Degenerate segment with L commands should be rotated."""
        # Create a reconstructed path with degenerate Z and L commands
        points = np.array(
            [
                [0.0, 0.0],  # First point (M)
                [5.0, 0.0],  # L endpoint
                [10.0, 0.0],  # Q control
                [10.0, 5.0],  # Q endpoint
                [10.0, 10.0],  # L endpoint
                [5.0, 10.0],  # L endpoint
                [0.0, 10.0],  # Q control
                [0.0, 5.0],  # Q endpoint (same as first - degenerate)
            ]
        )
        commands = ["M", "L", "Q", "L", "L", "Q", "Z"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_z_lines(path)
        segments = result.split_into_single_paths()

        # Should have rotated to avoid degenerate Z
        assert len(segments) == 1
        seg = segments[0]
        first = seg.points[0, :2]
        last = seg.points[-1, :2]
        dist = np.linalg.norm(first - last)
        assert dist > 1e-9, f"Should be non-degenerate, got distance {dist}"

    def test_degenerate_without_l_commands_unchanged(self):
        """Degenerate segment without L commands should be unchanged."""
        # Create a reconstructed path with only Q commands (like 'j' dot)
        points = np.array(
            [
                [164.0, 1291.0],  # First point (M)
                [170.0, 1291.0],  # Q control
                [170.0, 1295.0],  # Q endpoint
                [170.0, 1299.0],  # Q control
                [164.0, 1299.0],  # Q endpoint
                [158.0, 1299.0],  # Q control
                [158.0, 1295.0],  # Q endpoint
                [158.0, 1291.0],  # Q control
                [164.0, 1291.0],  # Q endpoint (same as first - degenerate)
            ]
        )
        commands = ["M", "Q", "Q", "Q", "Q", "Z"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_z_lines(path)
        segments = result.split_into_single_paths()

        # Should remain unchanged (no L commands to rotate to)
        assert len(segments) == 1
        seg = segments[0]
        first = seg.points[0, :2]
        last = seg.points[-1, :2]
        dist = np.linalg.norm(first - last)
        assert dist < 1e-9, f"Should remain degenerate, got distance {dist}"
        # Verify commands are unchanged
        assert seg.commands == commands

    def test_non_degenerate_unchanged(self):
        """Non-degenerate segment should be unchanged."""
        points1 = np.array(
            [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [10.0, 5.0], [10.0, 10.0], [5.0, 10.0], [0.0, 10.0], [0.0, 5.0]]
        )
        commands1 = ["M", "L", "Q", "L", "L", "Q", "Z"]
        path = AvPath(points1, commands1)

        result = AvPathCurveRebuilder._fix_degenerate_z_lines(path)

        # Should be unchanged
        assert np.allclose(result.points, path.points)
        assert result.commands == path.commands

    def test_multiple_segments_mixed(self):
        """Test multiple segments with mixed degenerate cases."""
        # Create two segments: one with L commands, one without
        points1 = np.array(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [10.0, 0.0],
                [10.0, 5.0],
                [10.0, 10.0],
                [5.0, 10.0],
                [0.0, 10.0],
                [0.0, 5.0],
            ]
        )
        commands1 = ["M", "L", "Q", "L", "L", "Q", "Z"]

        points2 = np.array([[100.0, 100.0], [105.0, 100.0], [105.0, 105.0], [100.0, 105.0], [100.0, 100.0]])
        commands2 = ["M", "Q", "Q", "Z"]

        # Join segments
        joined = AvPath.join_paths(AvPath(points1, commands1), AvPath(points2, commands2))

        result = AvPathCurveRebuilder._fix_degenerate_z_lines(joined)
        segments = result.split_into_single_paths()

        assert len(segments) == 2

        # First segment should be non-degenerate (has L commands)
        first = segments[0].points[0, :2]
        last = segments[0].points[-1, :2]
        dist1 = np.linalg.norm(first - last)
        assert dist1 > 1e-9, f"First segment should be non-degenerate"

        # Second segment should remain degenerate (no L commands)
        first = segments[1].points[0, :2]
        last = segments[1].points[-1, :2]
        dist2 = np.linalg.norm(first - last)
        assert dist2 < 1e-9, f"Second segment should remain degenerate"


class TestRotateReconstructedSegment:
    """Test _rotate_reconstructed_segment method."""

    def test_rotates_to_l_command(self):
        """Should rotate to L command endpoint to fix degenerate Z."""
        points = np.array(
            [
                [0.0, 0.0],  # M
                [10.0, 0.0],  # L (rotation target)
                [10.0, 5.0],  # Q control
                [10.0, 10.0],  # Q endpoint
                [5.0, 10.0],  # L
                [0.0, 10.0],  # Q control
                [0.0, 5.0],  # Q endpoint (degenerate with first)
            ]
        )
        commands = ["M", "L", "Q", "L", "Q", "Z"]
        seg = AvPath(points, commands)

        result = AvPathCurveRebuilder._rotate_reconstructed_segment(seg)

        # Should rotate to start at L command endpoint
        first = result.points[0, :2]
        # After rotation, should start at [10, 0] (first L endpoint)
        assert np.allclose(first, [10.0, 0.0])

    def test_no_l_commands_returns_original(self):
        """Should return original when no L commands present."""
        points = np.array(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [10.0, 5.0],
                [5.0, 10.0],
                [0.0, 5.0],
            ]
        )
        commands = ["M", "Q", "Q", "Z"]
        seg = AvPath(points, commands)

        result = AvPathCurveRebuilder._rotate_reconstructed_segment(seg)

        # Should return unchanged
        assert np.allclose(result.points, seg.points)
        assert result.commands == seg.commands


if __name__ == "__main__":
    # Run tests
    test_degenerate_with_l_commands_rotates()
    test_degenerate_without_l_commands_unchanged()
    test_non_degenerate_unchanged()
    test_multiple_segments_mixed()
    test_rotates_to_l_command()
    test_no_l_commands_returns_original()
    print("All post-processing degenerate Z tests passed!")
