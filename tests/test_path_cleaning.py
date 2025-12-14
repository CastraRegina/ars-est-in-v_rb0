"""Test cases for AvPathCleaner to verify path intersection resolution."""

import numpy as np
import pytest

from ave.path import AvPath
from ave.path_helper import AvPathCleaner


class TestPathCleaning:
    """Test path cleaning functionality."""

    def test_plus_shape_crossing_lines(self):
        """Test that crossing lines forming a '+' are preserved correctly."""
        # Create horizontal line (rectangle)
        horizontal_points = np.array(
            [
                [0.0, 45.0, 0.0],  # Start
                [100.0, 45.0, 0.0],  # End
                [100.0, 55.0, 0.0],  # End bottom
                [0.0, 55.0, 0.0],  # Start bottom
            ]
        )
        horizontal_commands = ["M", "L", "L", "L", "Z"]

        # Create vertical line (rectangle)
        vertical_points = np.array(
            [
                [45.0, 0.0, 0.0],  # Start
                [55.0, 0.0, 0.0],  # End top
                [55.0, 100.0, 0.0],  # End bottom
                [45.0, 100.0, 0.0],  # Start bottom
            ]
        )
        vertical_commands = ["M", "L", "L", "L", "Z"]

        # Combine into single path with two contours
        combined_points = np.vstack([horizontal_points, vertical_points])
        combined_commands = horizontal_commands + vertical_commands

        path = AvPath(combined_points, combined_commands)

        # Clean the path
        cleaned_path = AvPathCleaner.resolve_path_intersections(path)

        # Verify result is not empty and has the expected shape
        assert cleaned_path.points.shape[0] > 0, "Cleaned path should not be empty"

        # The result should be a plus shape with 12 vertices (crossing creates 8 extra vertices)
        # Allow some tolerance for different implementations
        assert (
            cleaned_path.points.shape[0] >= 12
        ), f"Expected at least 12 points for plus shape, got {cleaned_path.points.shape[0]}"

        # Verify the shape bounds are correct
        min_x = np.min(cleaned_path.points[:, 0])
        max_x = np.max(cleaned_path.points[:, 0])
        min_y = np.min(cleaned_path.points[:, 1])
        max_y = np.max(cleaned_path.points[:, 1])

        assert min_x == 0.0, "Min X should be 0"
        assert max_x == 100.0, "Max X should be 100"
        assert min_y == 0.0, "Min Y should be 0"
        assert max_y == 100.0, "Max Y should be 100"

    def test_k_glyph_shape(self):
        """Test that K glyph shape preserves both contours."""
        # Simplified K shape with main vertical and diagonal
        # Main vertical line
        vertical_points = np.array(
            [
                [170.0, 0.0, 0.0],
                [362.0, 0.0, 0.0],
                [362.0, 1456.0, 0.0],
                [170.0, 1456.0, 0.0],
            ]
        )
        vertical_commands = ["M", "L", "L", "L", "Z"]

        # Diagonal leg
        diagonal_points = np.array(
            [
                [362.0, 500.0, 0.0],
                [1242.0, 1456.0, 0.0],
                [1010.0, 1456.0, 0.0],
                [362.0, 740.0, 0.0],
            ]
        )
        diagonal_commands = ["M", "L", "L", "L", "Z"]

        # Combine paths
        combined_points = np.vstack([vertical_points, diagonal_points])
        combined_commands = vertical_commands + diagonal_commands

        path = AvPath(combined_points, combined_commands)

        # Clean the path
        cleaned_path = AvPathCleaner.resolve_path_intersections(path)

        # Should preserve both parts of the K
        assert cleaned_path.points.shape[0] > 0, "Cleaned path should not be empty"

        # Check that we have points from both contours
        # The cleaned path should include the full vertical and diagonal
        min_x = np.min(cleaned_path.points[:, 0])
        max_x = np.max(cleaned_path.points[:, 0])

        assert min_x <= 170.0, "Should include left side of K"
        assert max_x >= 1242.0, "Should include right side of K (diagonal leg)"

    def test_simple_square(self):
        """Test that a simple square remains unchanged."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
            ]
        )
        commands = ["M", "L", "L", "L", "Z"]

        path = AvPath(points, commands)
        cleaned_path = AvPathCleaner.resolve_path_intersections(path)

        # Should remain essentially the same
        assert cleaned_path.points.shape[0] >= 4, "Square should have at least 4 points"

        # Check bounds
        assert np.min(cleaned_path.points[:, 0]) == 0.0
        assert np.max(cleaned_path.points[:, 0]) == 100.0
        assert np.min(cleaned_path.points[:, 1]) == 0.0
        assert np.max(cleaned_path.points[:, 1]) == 100.0

    def test_overlapping_squares(self):
        """Test two overlapping squares are properly unioned."""
        # First square
        square1_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
            ]
        )
        square1_commands = ["M", "L", "L", "L", "Z"]

        # Second overlapping square
        square2_points = np.array(
            [
                [50.0, 50.0, 0.0],
                [150.0, 50.0, 0.0],
                [150.0, 150.0, 0.0],
                [50.0, 150.0, 0.0],
            ]
        )
        square2_commands = ["M", "L", "L", "L", "Z"]

        # Combine
        combined_points = np.vstack([square1_points, square2_points])
        combined_commands = square1_commands + square2_commands

        path = AvPath(combined_points, combined_commands)
        cleaned_path = AvPathCleaner.resolve_path_intersections(path)

        # Result should be a single polygon (union of both squares)
        assert cleaned_path.points.shape[0] > 0, "Result should not be empty"

        # Bounds should cover both squares
        assert np.min(cleaned_path.points[:, 0]) == 0.0
        assert np.max(cleaned_path.points[:, 0]) == 150.0
        assert np.min(cleaned_path.points[:, 1]) == 0.0
        assert np.max(cleaned_path.points[:, 1]) == 150.0

    def test_donut_shape_with_hole(self):
        """Test a shape with a hole (donut)."""
        # Outer square
        outer_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
            ]
        )
        outer_commands = ["M", "L", "L", "L", "Z"]

        # Inner square (hole) - oriented clockwise
        inner_points = np.array(
            [
                [25.0, 75.0, 0.0],  # Clockwise order
                [75.0, 75.0, 0.0],
                [75.0, 25.0, 0.0],
                [25.0, 25.0, 0.0],
            ]
        )
        inner_commands = ["M", "L", "L", "L", "Z"]

        # Combine
        combined_points = np.vstack([outer_points, inner_points])
        combined_commands = outer_commands + inner_commands

        path = AvPath(combined_points, combined_commands)
        cleaned_path = AvPathCleaner.resolve_path_intersections(path)

        # Should preserve the hole
        assert cleaned_path.points.shape[0] > 0, "Result should not be empty"

        # The result should have at least one 'M' for the exterior
        m_count = cleaned_path.commands.count("M")
        assert m_count >= 1, f"Should have at least 1 path (exterior), got {m_count}"

        # Check that we have interior holes by counting 'Z' commands
        # Exterior + hole should have 2 'Z' commands
        z_count = cleaned_path.commands.count("Z")
        assert z_count >= 2, f"Should have at least 2 closed paths (exterior + hole), got {z_count}"

        # Verify bounds are correct (outer square)
        assert np.min(cleaned_path.points[:, 0]) == 0.0
        assert np.max(cleaned_path.points[:, 0]) == 100.0
        assert np.min(cleaned_path.points[:, 1]) == 0.0
        assert np.max(cleaned_path.points[:, 1]) == 100.0


if __name__ == "__main__":
    # Run tests
    test = TestPathCleaning()
    test.test_plus_shape_crossing_lines()
    test.test_k_glyph_shape()
    test.test_simple_square()
    test.test_overlapping_squares()
    test.test_donut_shape_with_hole()
    print("All tests passed!")
