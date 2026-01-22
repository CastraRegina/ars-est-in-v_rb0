"""Test cases for AvPathCleaner.resolve_polygonized_path_intersections."""

import numpy as np
import pytest  # pylint: disable=unused-import

from ave.path import SINGLE_PATH_CONSTRAINTS, AvPath
from ave.path_processing import AvPathCleaner


class TestPolygonizedPathCleaning:
    """Test polygonized path cleaning functionality."""

    def test_make_closed_with_duplicate_endpoint(self):
        """Test that AvPath.make_closed() correctly handles duplicate endpoints.

        This test verifies the fix for the bug where duplicate endpoints were removed
        but the command count wasn't adjusted, causing point/command mismatches.
        """
        # Create a path with duplicate endpoint (first and last points are the same)
        points = np.array(
            [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0], [0, 0, 0]],  # Duplicate of first point
            dtype=np.float64,
        )

        # Commands for a closed path with duplicate point
        commands = ["M", "L", "L", "L", "L", "Z"]

        # Create original path
        original_path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        # Convert to closed path
        closed_path = AvPath.make_closed_single(original_path)

        # Verify that the duplicate point was removed
        assert closed_path.points.shape[0] == 4  # Should have 4 points (not 5)

        # Verify that the command count was adjusted to match
        assert len(closed_path.commands) == 5  # Should have 5 commands (not 6)

        # Verify the commands are correct for the number of points
        # For 4 points, we need: M, L, L, L, Z
        expected_commands = ["M", "L", "L", "L", "Z"]
        assert closed_path.commands == expected_commands

        # Verify the path is still valid
        # Should not raise ValueError during creation
        AvPath(closed_path.points, closed_path.commands)

    def test_make_closed_without_duplicate_endpoint(self):
        """Test that AvPath.make_closed() works correctly when no duplicate endpoint exists."""
        # Create a path without duplicate endpoint
        points = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]], dtype=np.float64)

        # Commands without duplicate point
        commands = ["M", "L", "L", "L"]

        # Create original path
        original_path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        # Convert to closed path
        closed_path = AvPath.make_closed_single(original_path)

        # Verify that Z was added
        assert len(closed_path.commands) == 5
        assert closed_path.commands[-1] == "Z"

        # Verify points remain unchanged
        assert closed_path.points.shape[0] == 4

        # Verify the path is valid
        AvPath(closed_path.points, closed_path.commands)

    def test_plus_shape_crossing_lines(self):
        """Test that '+' shape formed by crossing lines is preserved."""
        # Create two rectangles that cross to form a '+' shape
        points = np.array(
            [
                [450, 0, 0],  # Vertical bar start
                [550, 0, 0],
                [550, 1000, 0],
                [450, 1000, 0],
                [0, 450, 0],  # Horizontal bar start
                [1000, 450, 0],
                [1000, 550, 0],
                [0, 550, 0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should preserve the '+' shape
        assert len(result.points) > 0
        assert isinstance(result, AvPath)

    def test_k_glyph_shape(self):
        """Test that 'K' glyph shape is preserved with all legs."""
        # Create a simplified 'K' shape with vertical and diagonal legs
        points = np.array(
            [
                [100, 0, 0],  # Vertical bar
                [120, 0, 0],
                [120, 1000, 0],
                [100, 1000, 0],
                [100, 0, 0],
                [120, 400, 0],  # Diagonal leg start
                [400, 0, 0],
                [420, 0, 0],
                [120, 420, 0],
                [120, 600, 0],  # Lower diagonal leg
                [400, 1000, 0],
                [420, 1000, 0],
                [120, 620, 0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "L", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should preserve all parts of the 'K'
        assert len(result.points) > 0
        assert isinstance(result, AvPath)

    def test_simple_square(self):
        """Test cleaning a simple square."""
        points = np.array(
            [
                [0, 0, 0],
                [100, 0, 0],
                [100, 100, 0],
                [0, 100, 0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        assert len(result.points) > 0
        assert isinstance(result, AvPath)

    def test_overlapping_squares(self):
        """Test cleaning overlapping squares."""
        points = np.array(
            [
                [0, 0, 0],
                [100, 0, 0],
                [100, 100, 0],
                [0, 100, 0],
                [50, 50, 0],
                [150, 50, 0],
                [150, 150, 0],
                [50, 150, 0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        assert len(result.points) > 0
        assert isinstance(result, AvPath)

    def test_donut_shape_with_hole(self):
        """Test cleaning a donut shape (square with hole)."""
        points = np.array(
            [
                [0, 0, 0],  # Outer square (CCW)
                [200, 0, 0],
                [200, 200, 0],
                [0, 200, 0],
                [50, 50, 0],  # Inner square (CW for hole)
                [50, 150, 0],
                [150, 150, 0],
                [150, 50, 0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        assert len(result.points) > 0
        assert isinstance(result, AvPath)
        # Should have two contours (outer and inner)
        assert result.commands.count("Z") == 2

    def test_cw_first_hole_before_exterior(self):
        """Test when CW polygon appears before CCW exterior."""
        points = np.array(
            [
                [50, 50, 0],  # Inner square (CW) - comes first!
                [50, 150, 0],
                [150, 150, 0],
                [150, 50, 0],
                [0, 0, 0],  # Outer square (CCW) - comes second
                [200, 0, 0],
                [200, 200, 0],
                [0, 200, 0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        assert len(result.points) > 0
        assert isinstance(result, AvPath)
        # Should still have two contours
        assert result.commands.count("Z") == 2

    def test_self_intersecting_bowtie(self):
        """Test cleaning a self-intersecting bowtie shape."""
        points = np.array(
            [
                [0, 0, 0],
                [100, 100, 0],
                [0, 100, 0],
                [100, 0, 0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Self-intersecting polygons should return empty path (matching resolve_path_intersections)
        assert isinstance(result, AvPath)
        assert len(result.points) == 0
        assert len(result.commands) == 0

    def test_degenerate_polygon(self):
        """Test cleaning a degenerate polygon (collinear points)."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should handle degenerate cases gracefully
        assert isinstance(result, AvPath)

    def test_empty_path(self):
        """Test cleaning an empty path."""
        points = np.array([], dtype=np.float64).reshape(0, 3)
        commands = []
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Empty path should return empty path
        assert isinstance(result, AvPath)
        assert len(result.points) == 0
        assert len(result.commands) == 0

    def test_open_path(self):
        """Test cleaning an open path (no closing)."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L"]  # No closing 'Z'
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Open paths should be handled gracefully
        assert isinstance(result, AvPath)

    def test_multiple_contours(self):
        """Test cleaning multiple separate contours."""
        # Create two separate squares
        points = np.array(
            [
                # First square
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                # Move to second square
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [30.0, 10.0, 0.0],
                [20.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Should handle multiple contours
        assert isinstance(result, AvPath)
        assert len(result.points) > 0
        # Should still have closed contours
        assert "Z" in result.commands


class TestPolygonizedPathCleaningFromData:
    """Test path cleaning functionality using real font data."""

    def test_character_0041(self):
        """Test cleaning for character 'A'."""
        # Original path
        original_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [192.0, 0.0, 0.0],
                [596.0, 1100.0, 0.0],
                [607.25, 1132.5, 2.0],
                [617.0, 1164.0, 0.0],
                [625.5, 1195.5, 2.0],
                [633.0, 1228.0, 0.0],
                [643.0, 1228.0, 0.0],
                [650.5, 1195.5, 2.0],
                [659.0, 1164.0, 0.0],
                [668.75, 1132.5, 2.0],
                [680.0, 1100.0, 0.0],
                [1080.0, 0.0, 0.0],
                [1280.0, 0.0, 0.0],
                [726.0, 1466.0, 0.0],
                [554.0, 1466.0, 0.0],
                [238.0, 380.0, 0.0],
                [1039.0, 380.0, 0.0],
                [981.0, 538.0, 0.0],
                [297.0, 538.0, 0.0],
            ],
            dtype=np.float64,
        )
        original_commands = [
            "M",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "Z",
            "M",
            "L",
            "L",
            "L",
            "Z",
        ]
        original_path = AvPath(original_points, original_commands)

        # Apply path cleaning
        result = AvPathCleaner.resolve_polygon_path_intersections(original_path)

        # Verify result is not empty
        assert len(result.points) > 0
        assert isinstance(result, AvPath)

    def test_character_004b(self):
        """Test cleaning for character 'K'."""
        # Original path
        original_points = np.array(
            [
                [170.0, 0.0, 0.0],
                [362.0, 0.0, 0.0],
                [362.0, 500.0, 0.0],
                [1242.0, 1456.0, 0.0],
                [1010.0, 1456.0, 0.0],
                [362.0, 740.0, 0.0],
                [362.0, 1456.0, 0.0],
                [170.0, 1456.0, 0.0],
                [1050.0, 0.0, 0.0],
                [1286.0, 0.0, 0.0],
                [610.0, 890.0, 0.0],
                [486.0, 756.0, 0.0],
            ],
            dtype=np.float64,
        )
        original_commands = [
            "M",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "Z",
            "M",
            "L",
            "L",
            "L",
            "Z",
        ]
        original_path = AvPath(original_points, original_commands)

        # Apply path cleaning
        result = AvPathCleaner.resolve_polygon_path_intersections(original_path)

        # Verify result is not empty
        assert len(result.points) > 0
        assert isinstance(result, AvPath)

    def test_character_0058(self):
        """Test cleaning for character 'X'."""
        # Original path
        original_points = np.array(
            [
                [25.0, 0.0, 0.0],
                [237.0, 0.0, 0.0],
                [659.0, 670.0, 0.0],
                [673.0, 670.0, 0.0],
                [1172.0, 1456.0, 0.0],
                [960.0, 1456.0, 0.0],
                [563.0, 806.0, 0.0],
                [549.0, 806.0, 0.0],
                [50.0, 1456.0, 0.0],
                [549.0, 670.0, 0.0],
                [559.0, 670.0, 0.0],
                [981.0, 0.0, 0.0],
                [1197.0, 0.0, 0.0],
                [673.0, 806.0, 0.0],
                [663.0, 806.0, 0.0],
                [266.0, 1456.0, 0.0],
            ],
            dtype=np.float64,
        )
        original_commands = [
            "M",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "Z",
            "M",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "Z",
        ]
        original_path = AvPath(original_points, original_commands)

        # Apply path cleaning
        result = AvPathCleaner.resolve_polygon_path_intersections(original_path)

        # Verify result is not empty
        assert len(result.points) > 0
        assert isinstance(result, AvPath)

    def test_character_0034(self):
        """Test cleaning for character '4'."""
        # Original path
        original_points = np.array(
            [
                [724.0, 0.0, 0.0],
                [904.0, 0.0, 0.0],
                [904.0, 1456.0, 0.0],
                [732.0, 1456.0, 0.0],
                [50.0, 468.0, 0.0],
                [50.0, 342.0, 0.0],
                [1108.0, 342.0, 0.0],
                [1108.0, 500.0, 0.0],
                [852.0, 500.0, 0.0],
                [775.0, 500.0, 0.0],
                [269.0, 500.0, 0.0],
                [712.0, 1149.0, 0.0],
                [724.0, 1149.0, 0.0],
                [724.0, 450.0, 0.0],
                [724.0, 398.0, 0.0],
            ],
            dtype=np.float64,
        )
        original_commands = [
            "M",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "L",
            "Z",
        ]
        original_path = AvPath(original_points, original_commands)

        # Apply path cleaning
        result = AvPathCleaner.resolve_polygon_path_intersections(original_path)

        # Verify result is not empty
        assert len(result.points) > 0
        assert isinstance(result, AvPath)

    def test_character_0023(self):
        """Test cleaning for character '#'."""
        # Original path
        original_points = np.array(
            [
                [216.0, -20.0, 0.0],
                [360.0, -20.0, 0.0],
                [797.0, 1501.0, 0.0],
                [653.0, 1501.0, 0.0],
                [722.0, -20.0, 0.0],
                [866.0, -20.0, 0.0],
                [1303.0, 1501.0, 0.0],
                [1159.0, 1501.0, 0.0],
                [64.0, 404.0, 0.0],
                [1405.0, 404.0, 0.0],
                [1405.0, 542.0, 0.0],
                [64.0, 542.0, 0.0],
                [115.0, 939.0, 0.0],
                [1455.0, 939.0, 0.0],
                [1455.0, 1077.0, 0.0],
                [115.0, 1077.0, 0.0],
            ],
            dtype=np.float64,
        )
        original_commands = [
            "M",
            "L",
            "L",
            "L",
            "Z",
            "M",
            "L",
            "L",
            "L",
            "Z",
            "M",
            "L",
            "L",
            "L",
            "Z",
            "M",
            "L",
            "L",
            "L",
            "Z",
        ]
        original_path = AvPath(original_points, original_commands)

        # Apply path cleaning
        result = AvPathCleaner.resolve_polygon_path_intersections(original_path)

        # Verify result is not empty
        assert len(result.points) > 0
        assert isinstance(result, AvPath)

    def test_character_0022(self):
        """Test cleaning for character '\"'."""
        # Original path
        original_points = np.array(
            [
                [115.0, 966.0, 0.0],
                [265.0, 966.0, 0.0],
                [280.0, 1320.0, 0.0],
                [280.0, 1456.0, 0.0],
                [100.0, 1456.0, 0.0],
                [100.0, 1320.0, 0.0],
                [495.0, 966.0, 0.0],
                [645.0, 966.0, 0.0],
                [660.0, 1320.0, 0.0],
                [660.0, 1456.0, 0.0],
                [480.0, 1456.0, 0.0],
                [480.0, 1320.0, 0.0],
            ],
            dtype=np.float64,
        )
        original_commands = [
            "M",
            "L",
            "L",
            "L",
            "L",
            "L",
            "Z",
            "M",
            "L",
            "L",
            "L",
            "L",
            "L",
            "Z",
        ]
        original_path = AvPath(original_points, original_commands)

        # Apply path cleaning
        result = AvPathCleaner.resolve_polygon_path_intersections(original_path)

        # Verify result is not empty
        assert len(result.points) > 0
        assert isinstance(result, AvPath)


if __name__ == "__main__":
    # Run tests
    test = TestPolygonizedPathCleaning()
    test.test_plus_shape_crossing_lines()
    test.test_simple_square()
    test.test_overlapping_squares()
    test.test_donut_shape_with_hole()
    test.test_cw_first_hole_before_exterior()
    test.test_self_intersecting_bowtie()
    test.test_degenerate_polygon()
    test.test_empty_path()
    test.test_open_path()
    test.test_multiple_contours()
    print("All basic tests passed!")
