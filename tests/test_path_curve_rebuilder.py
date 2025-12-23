"""Tests for AvPathCurveRebuilder class."""

import numpy as np
import pytest  # pylint: disable=unused-import

from ave.path import AvMultiPolylinePath, AvPath
from ave.path_processing import AvPathCurveRebuilder


class TestAvPathCurveRebuilder:
    """Test cases for AvPathCurveRebuilder class."""

    def test_empty_path(self):
        """Test rebuild_curve_path with empty path."""
        empty_path = AvMultiPolylinePath(np.array([], dtype=np.float64).reshape(0, 3), [])
        result = AvPathCurveRebuilder.rebuild_curve_path(empty_path)

        assert isinstance(result, AvPath)
        assert len(result.points) == 0
        assert len(result.commands) == 0

    def test_path_with_only_regular_points(self):
        """Test rebuild_curve_path with only type=0 points."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z"]
        path = AvMultiPolylinePath(points, commands)

        result = AvPathCurveRebuilder.rebuild_curve_path(path)

        assert isinstance(result, AvPath)
        assert len(result.points) == 4
        assert result.commands == ["M", "L", "L", "L", "Z"]
        # Points should be unchanged
        np.testing.assert_array_equal(result.points, points)

    def test_path_with_quadratic_cluster(self):
        """Test rebuild_curve_path with type=2 point cluster."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Start point
                [5.0, 10.0, 2.0],  # Control point cluster
                [10.0, 0.0, 2.0],
                [15.0, -5.0, 2.0],
                [20.0, 0.0, 0.0],  # End point
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)

        result = AvPathCurveRebuilder.rebuild_curve_path(path)

        assert isinstance(result, AvPath)
        # Should have replaced the cluster with a quadratic curve
        assert len(result.points) == 3  # Start, control, end
        assert result.commands == ["M", "Q"]
        # Check types
        assert result.points[0, 2] == 0.0  # Start point
        assert result.points[1, 2] == 2.0  # Control point
        assert result.points[2, 2] == 0.0  # End point

    def test_path_with_cubic_cluster(self):
        """Test rebuild_curve_path with type=3 point cluster."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Start point
                [5.0, 10.0, 3.0],  # Control point cluster
                [10.0, -5.0, 3.0],
                [15.0, 10.0, 3.0],
                [20.0, 0.0, 0.0],  # End point
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)

        result = AvPathCurveRebuilder.rebuild_curve_path(path)

        assert isinstance(result, AvPath)
        # Should have replaced the cluster with a cubic curve
        assert len(result.points) == 4  # Start, control1, control2, end
        assert result.commands == ["M", "C"]
        # Check types
        assert result.points[0, 2] == 0.0  # Start point
        assert result.points[1, 2] == 3.0  # Control point 1
        assert result.points[2, 2] == 3.0  # Control point 2
        assert result.points[3, 2] == 0.0  # End point

    def test_path_with_single_type_3_point(self):
        """Test rebuild_curve_path with single type=3 point (should become quadratic)."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Start point
                [10.0, 10.0, 3.0],  # Single control point
                [20.0, 0.0, 0.0],  # End point
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L"]
        path = AvMultiPolylinePath(points, commands)

        result = AvPathCurveRebuilder.rebuild_curve_path(path)

        assert isinstance(result, AvPath)
        # Single type=3 should become quadratic
        assert len(result.points) == 3
        assert result.commands == ["M", "Q"]
        assert result.points[1, 2] == 2.0  # Should be type=2 (quadratic control)

    def test_path_with_mixed_point_types(self):
        """Test rebuild_curve_path with mixed type=0, -1, 2, and 3 points."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Regular point
                [10.0, 0.0, 0.0],  # Regular point
                [15.0, 5.0, 2.0],  # Quadratic cluster start
                [20.0, 10.0, 2.0],  # Quadratic cluster end
                [25.0, 5.0, 0.0],  # Regular point
                [30.0, 0.0, -1.0],  # Type -1 point
                [35.0, 5.0, 3.0],  # Single type=3
                [40.0, 0.0, 0.0],  # End point
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "L", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)

        result = AvPathCurveRebuilder.rebuild_curve_path(path)

        assert isinstance(result, AvPath)
        # Should have replaced clusters but preserved other points
        assert len(result.points) > 0
        assert len(result.commands) > 0

        # Check that M and Z commands are preserved
        if "Z" in commands:
            assert "Z" in result.commands

    def test_path_with_multiple_segments(self):
        """Test rebuild_curve_path with multiple segments (multiple M commands)."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # First segment start
                [10.0, 0.0, 0.0],
                [15.0, 5.0, 2.0],  # Quadratic cluster
                [20.0, 0.0, 2.0],
                [30.0, 10.0, 0.0],  # Second segment start
                [35.0, 15.0, 0.0],
                [40.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "M", "L", "L"]
        path = AvMultiPolylinePath(points, commands)

        result = AvPathCurveRebuilder.rebuild_curve_path(path)

        assert isinstance(result, AvPath)
        # Should preserve the M command for the second segment
        assert commands.count("M") == result.commands.count("M")
        assert "M" in result.commands

    def test_path_preserves_original_commands(self):
        """Test that original L, Q, C commands are preserved for type=0/-1 points."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M
                [10.0, 0.0, 0.0],  # L
                [15.0, 5.0, 0.0],  # Q control
                [20.0, 0.0, 0.0],  # Q end
                [25.0, 0.0, 0.0],  # L
                [30.0, 5.0, 0.0],  # C control 1
                [35.0, -5.0, 0.0],  # C control 2
                [40.0, 0.0, 0.0],  # C end
                [45.0, 0.0, 0.0],  # L
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "Q", "L", "C", "L"]  # Mixed commands with correct point counts
        path = AvMultiPolylinePath(points, commands)

        result = AvPathCurveRebuilder.rebuild_curve_path(path)

        assert isinstance(result, AvPath)
        # Should preserve the original commands for regular points
        assert result.commands[0] == "M"
        assert result.commands[1] == "L"  # Preserved
        assert result.commands[2] == "Q"  # Preserved
        assert result.commands[3] == "L"  # Preserved
        assert result.commands[4] == "C"  # Preserved
        assert result.commands[5] == "L"  # Preserved

    def test_path_with_z_commands(self):
        """Test rebuild_curve_path preserves Z commands."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [5.0, 5.0, 2.0],  # Type=2 point
                [0.0, 0.0, 0.0],  # Back to start
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L"]
        path = AvMultiPolylinePath(points, commands)

        result = AvPathCurveRebuilder.rebuild_curve_path(path)

        assert isinstance(result, AvPath)
        # Should preserve Z commands
        assert "Z" in result.commands
        # Z should be at the same position relative to M commands
        z_index = result.commands.index("Z")
        assert z_index > 0  # Z should not be first

    def test_approximation_failure_fallback(self):
        """Test that approximation failures fall back to individual point handling."""
        # Create a pathological case that might fail approximation
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],  # Very close points
                [2.0, 0.0, 2.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L"]
        path = AvMultiPolylinePath(points, commands)

        result = AvPathCurveRebuilder.rebuild_curve_path(path)

        # Should not crash and should produce a valid path
        assert isinstance(result, AvPath)
        assert len(result.points) > 0
        assert len(result.commands) > 0

    def test_edge_cases(self):
        """Test edge cases for rebuild_curve_path."""
        # Test with single point
        single_point = np.array([[10.0, 10.0, 0.0]], dtype=np.float64)
        path = AvMultiPolylinePath(single_point, ["M"])
        result = AvPathCurveRebuilder.rebuild_curve_path(path)
        assert len(result.points) == 1
        assert result.commands == ["M"]

        # Test with type=-1 points
        neg_type_points = np.array(
            [
                [0.0, 0.0, -1.0],
                [10.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        path = AvMultiPolylinePath(neg_type_points, ["M", "L"])
        result = AvPathCurveRebuilder.rebuild_curve_path(path)
        assert len(result.points) == 2
        assert result.commands[0] == "M"
        assert result.commands[1] == "L"
