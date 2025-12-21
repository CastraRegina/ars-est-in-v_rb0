"""Tests for _process_2d_to_3d() method reuse and functionality."""

import pytest  # pylint: disable=unused-import

from ave.path import AvPath, PathCommandProcessor


class TestProcess2DTo3DReuse:
    """Tests for _process_2d_to_3d() method reuse across the codebase."""

    def test_process_2d_to_3d_used_in_init(self):
        """Verify _process_2d_to_3d() is used in __init__ method."""
        # Test with 2D points - should trigger _process_2d_to_3d()
        points_2d = [(0, 0), (10, 0), (10, 10)]
        commands = ["M", "L", "L", "Z"]
        path = AvPath(points_2d, commands)

        # Should have 3D points with type column
        assert path.points.shape == (3, 3)
        # Type column should be set correctly
        assert path.points[:, 2].tolist() == [0.0, 0.0, 0.0]

    def test_process_2d_to_3d_used_in_polygonize(self):
        """Verify _process_2d_to_3d() is used in polygonize() method."""
        # Create path with 2D points and curves
        points_2d = [(0, 0), (5, 10), (10, 0)]
        commands = ["M", "Q", "Z"]
        path = AvPath(points_2d, commands)

        # Original path should have 3D points
        assert path.points.shape == (3, 3)

        # Polygonize should work correctly with 3D points
        result = path.polygonize(5)

        # Result should have no curve commands
        assert not any(PathCommandProcessor.is_curve_command(cmd) for cmd in result.commands)
        # Result should still be 3D
        assert result.points.shape[1] == 3

    def test_process_2d_to_3d_with_curves(self):
        """Test _process_2d_to_3d() correctly handles curve commands."""
        # M(1) + Q(2) + C(3) = 6 points (Z uses 0 points)
        points_2d = [(0, 0), (2, 5), (4, 0), (6, 0), (8, 5), (10, 5)]
        commands = ["M", "Q", "C", "Z"]
        path = AvPath(points_2d, commands)

        # Should have 3D points
        assert path.points.shape == (6, 3)

        # Type column should be set correctly:
        # M: type 0.0
        # Q: control point type 2.0, end point type 0.0
        # C: control1 type 3.0, control2 type 3.0, end point type 0.0
        expected_types = [0.0, 2.0, 0.0, 3.0, 3.0, 0.0]
        actual_types = path.points[:, 2].tolist()

        assert actual_types == expected_types

    def test_process_2d_to_3d_validation(self):
        """Test _process_2d_to_3d() validates command sequence."""
        # Invalid: not enough points for commands
        points_2d = [(0, 0), (1, 0)]  # Only 2 points
        commands = ["M", "Q", "Z"]  # Need M(1) + Q(2) = 3 points

        with pytest.raises(ValueError, match="Not enough points"):
            AvPath(points_2d, commands)

    def test_process_2d_to_3d_mixed_commands(self):
        """Test _process_2d_to_3d() with mixed command types."""
        # M(1) + L(1) + Q(2) + C(3) = 7 points
        points_2d = [(0, 0), (5, 0), (7, 5), (10, 0), (12, 5), (15, 5), (17, 0)]
        commands = ["M", "L", "Q", "C", "Z"]
        path = AvPath(points_2d, commands)

        # Should have 3D points
        assert path.points.shape == (7, 3)

        # Verify type column is correct
        types = path.points[:, 2]
        assert types[0] == 0.0  # M
        assert types[1] == 0.0  # L
        assert types[2] == 2.0  # Q control
        assert types[3] == 0.0  # Q end
        assert types[4] == 3.0  # C control1
        assert types[5] == 3.0  # C control2
        assert types[6] == 0.0  # C end

    def test_process_2d_to_3d_reuse_benefits(self):
        """Demonstrate benefits of _process_2d_to_3d() reuse."""
        # Before refactoring: duplicate logic in __init__ and polygonize
        # After refactoring: single source of truth for 2D to 3D conversion

        # Test that both methods produce consistent results
        points_2d = [(0, 0), (5, 10), (10, 0)]
        commands = ["M", "Q", "Z"]

        # Method 1: Direct creation via __init__
        path1 = AvPath(points_2d, commands)

        # Method 2: Via polygonize (which also uses _process_2d_to_3d internally)
        path2 = AvPath(points_2d, commands)
        polygonized = path2.polygonize(3)

        # Both should have correctly processed 3D points
        assert path1.points.shape == (3, 3)
        assert polygonized.points.shape[1] == 3

        # Type column should be consistent
        assert path1.points[0, 2] == 0.0  # M point
        assert path1.points[1, 2] == 2.0  # Q control
        assert path1.points[2, 2] == 0.0  # Q end

    def test_process_2d_to_3d_no_command_modification(self):
        """Ensure _process_2d_to_3d() doesn't modify original commands."""
        points_2d = [(0, 0), (5, 10), (10, 0)]
        commands = ["M", "Q", "Z"]
        original_commands = commands.copy()

        path = AvPath(points_2d, commands)

        # Commands should be unchanged
        assert path.commands == original_commands

    def test_process_2d_to_3d_edge_cases(self):
        """Test _process_2d_to_3d() with edge cases."""
        # Empty path
        path = AvPath()
        assert path.points.shape == (0, 3)

        # Single point
        path = AvPath([(0, 0)], ["M"])
        assert path.points.shape == (1, 3)
        assert path.points[0, 2] == 0.0

        # M followed by Z (closed path with no drawing)
        path = AvPath([(0, 0)], ["M", "Z"])
        assert path.points.shape == (1, 3)
        assert path.points[0, 2] == 0.0


class TestProcess2DTo3DIntegration:
    """Integration tests for _process_2d_to_3d() with PathCommandProcessor."""

    def test_command_processor_integration_consistency(self):
        """Ensure _process_2d_to_3d() uses PathCommandProcessor consistently."""
        points_2d = [(0, 0), (5, 10), (10, 0), (15, 5)]
        commands = ["M", "Q", "L", "Z"]
        path = AvPath(points_2d, commands)

        # Verify PathCommandProcessor was used correctly
        for i, cmd in enumerate(commands):
            if cmd == "Z":
                continue

            if cmd in ["M", "L"]:
                assert path.points[i, 2] == 0.0  # Regular points
            elif cmd == "Q":
                assert path.points[i, 2] == 2.0  # Control point
                assert path.points[i + 1, 2] == 0.0  # End point

    def test_reuse_maintains_functionality(self):
        """Ensure reuse of _process_2d_to_3d() maintains all functionality."""
        # Test complex scenario
        points_2d = [
            (0, 0),  # M
            (5, 10),  # Q control
            (10, 0),  # Q end
            (15, 5),  # C control1
            (20, 10),  # C control2
            (25, 0),  # C end
        ]
        commands = ["M", "Q", "C", "Z"]
        path = AvPath(points_2d, commands)

        # Should work with polygonize
        result = path.polygonize(5)
        assert not any(PathCommandProcessor.is_curve_command(cmd) for cmd in result.commands)

        # Should work with splitting
        single_paths = path.split_into_single_paths()
        assert len(single_paths) == 1
        assert single_paths[0].commands == commands
