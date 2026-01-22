"""Test orientation enforcement in AvPathCleaner.resolve_polygonized_path_intersections."""

import numpy as np
import pytest

from ave.geom import AvPolygon
from ave.path import SINGLE_PATH_CONSTRAINTS, AvPath
from ave.path_processing import AvPathCleaner


class TestPolygonizedPathOrientation:
    """Test that polygon orientations are correctly enforced after cleaning."""

    def test_ccw_exterior_enforcement(self):
        """Test that exterior rings are forced to CCW orientation."""
        # Create a CCW square (should remain CCW)
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

        # Split into segments to check orientation
        segments = result.split_into_single_paths()
        assert len(segments) == 1

        # Check that the resulting exterior is CCW
        exterior = segments[0]
        assert exterior.is_ccw, "Exterior ring should be CCW after orientation enforcement"

    def test_cw_hole_enforcement(self):
        """Test that interior rings (holes) are forced to CW orientation."""
        # Create a donut shape with CW hole (should remain CW)
        points = np.array(
            [
                # Outer square (CCW)
                [0, 0, 0],
                [100, 0, 0],
                [100, 100, 0],
                [0, 100, 0],
                # Inner square (CW - proper hole orientation)
                [25, 75, 0],
                [75, 75, 0],
                [75, 25, 0],
                [25, 25, 0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Split into segments
        segments = result.split_into_single_paths()
        assert len(segments) == 2

        # First segment should be exterior (CCW)
        assert segments[0].is_ccw, "Exterior ring should be CCW"

        # Second segment should be hole (CW)
        assert not segments[1].is_ccw, "Interior hole should be CW"

    def test_multipolygon_orientation(self):
        """Test orientation enforcement for MultiPolygon results."""
        # Create two separate squares, both CCW (valid exteriors)
        points = np.array(
            [
                # First square (CCW)
                [0, 0, 0],
                [50, 0, 0],
                [50, 50, 0],
                [0, 50, 0],
                # Second square (CCW)
                [100, 0, 0],
                [150, 0, 0],
                [150, 50, 0],
                [100, 50, 0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Split into segments
        segments = result.split_into_single_paths()
        assert len(segments) == 2

        # Both should be CCW (both are exteriors)
        assert segments[0].is_ccw, "First exterior should be CCW"
        assert segments[1].is_ccw, "Second exterior should be CCW"

    def test_complex_shape_with_holes(self):
        """Test orientation with complex shape containing multiple holes."""
        # Create a rectangle with two holes (both CW for proper subtraction)
        points = np.array(
            [
                # Outer rectangle (CCW)
                [0, 0, 0],
                [200, 0, 0],
                [200, 100, 0],
                [0, 100, 0],
                # First hole (CW - proper hole orientation)
                [20, 80, 0],
                [80, 80, 0],
                [80, 20, 0],
                [20, 20, 0],
                # Second hole (CW - proper hole orientation)
                [120, 80, 0],
                [180, 80, 0],
                [180, 20, 0],
                [120, 20, 0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)

        result = AvPathCleaner.resolve_polygon_path_intersections(path)

        # Split into segments
        segments = result.split_into_single_paths()
        assert len(segments) == 3

        # Check orientations
        assert segments[0].is_ccw, "Exterior should be CCW"
        assert not segments[1].is_ccw, "First hole should be CW"
        assert not segments[2].is_ccw, "Second hole should be CW"

    def test_orientation_after_boolean_operations(self):
        """Test that orientation is preserved after complex boolean operations."""
        # Create overlapping shapes that will trigger union operations
        points = np.array(
            [
                # First square (CCW)
                [0, 0, 0],
                [100, 0, 0],
                [100, 100, 0],
                [0, 100, 0],
                # Second overlapping square (CCW)
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

        # Result should be at least one CCW polygon (union of two squares)
        segments = result.split_into_single_paths()
        assert len(segments) >= 1

        # All exterior segments should be CCW
        for segment in segments:
            assert segment.is_ccw, f"All exterior segments should be CCW, got: {segment.is_ccw}"


if __name__ == "__main__":
    # Run tests
    test = TestPolygonizedPathOrientation()
    test.test_ccw_exterior_enforcement()
    test.test_cw_hole_enforcement()
    test.test_multipolygon_orientation()
    test.test_complex_shape_with_holes()
    test.test_orientation_after_boolean_operations()
    print("All orientation tests passed!")
