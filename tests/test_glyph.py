"""Tests for glyph functions"""

import numpy as np
import pytest

from ave.geom import AvBox
from ave.glyph import AvGlyph


class TestAvGlyph:
    """Test cases for AvGlyph class"""

    def test_glyph_init(self):
        """Test basic glyph initialization"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "L"]

        glyph = AvGlyph(character="A", width=10, points=points, commands=commands)

        assert glyph.character == "A"
        assert glyph.width() == 10
        np.testing.assert_array_equal(glyph.points, points)
        assert glyph.commands == commands
        # Test that bounding box calculation works (public interface)
        bbox = glyph.bounding_box()
        assert bbox is not None

    def test_bounding_box_empty_points(self):
        """Test bounding box with empty points"""
        points = np.array([], dtype=np.float64).reshape(0, 3)
        commands = []

        glyph = AvGlyph(character=" ", width=0, points=points, commands=commands)
        bbox = glyph.bounding_box()

        expected = AvBox(0.0, 0.0, 0.0, 0.0)
        assert bbox == expected

    def test_bounding_box_linear_path(self):
        """Test bounding box with linear path (no curves)"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 5.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L", "L"]

        glyph = AvGlyph(character="L", width=10, points=points, commands=commands)
        bbox = glyph.bounding_box()

        expected = AvBox(0.0, 0.0, 10.0, 5.0)
        assert bbox == expected

    def test_bounding_box_quadratic_curve(self):
        """Test bounding box with quadratic curve"""
        # Simple quadratic curve that goes up and back down
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # start
                [10.0, 20.0, 0.0],  # control point (higher than actual curve)
                [20.0, 0.0, 0.0],  # end
            ],
            dtype=np.float64,
        )
        commands = ["M", "Q"]

        glyph = AvGlyph(character="Q", width=20, points=points, commands=commands)
        bbox = glyph.bounding_box()

        # The actual quadratic curve should not reach y=20 (control point)
        # With 100 steps, the max y should be around 10.0 (midpoint of curve)
        assert bbox.xmin == 0.0
        assert bbox.xmax == 20.0
        assert bbox.ymin == 0.0
        assert bbox.ymax < 20.0  # Should not include control point
        assert bbox.ymax > 5.0  # Should be higher than endpoints

    def test_bounding_box_cubic_curve(self):
        """Test bounding box with cubic curve"""
        # Cubic curve with control points that extend beyond actual curve
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # start
                [5.0, 25.0, 0.0],  # control point 1
                [15.0, 25.0, 0.0],  # control point 2
                [20.0, 0.0, 0.0],  # end
            ],
            dtype=np.float64,
        )
        commands = ["M", "C"]

        glyph = AvGlyph(character="C", width=20, points=points, commands=commands)
        bbox = glyph.bounding_box()

        # The actual cubic curve should not reach y=25 (control points)
        assert bbox.xmin == 0.0
        assert bbox.xmax == 20.0
        assert bbox.ymin == 0.0
        assert bbox.ymax < 25.0  # Should not include control points
        assert bbox.ymax > 10.0  # Should be higher than endpoints

    def test_bounding_box_mixed_path(self):
        """Test bounding box with mixed linear and curve commands"""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [10.0, 0.0, 0.0],  # L - line
                [15.0, 20.0, 0.0],  # Q - control point
                [20.0, 0.0, 0.0],  # Q - end
                [25.0, 0.0, 0.0],  # L - line
                [30.0, 15.0, 0.0],  # C - control point 1
                [35.0, 15.0, 0.0],  # C - control point 2
                [40.0, 0.0, 0.0],  # C - end
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "Q", "L", "C"]

        glyph = AvGlyph(character="M", width=40, points=points, commands=commands)
        bbox = glyph.bounding_box()

        assert bbox.xmin == 0.0
        assert bbox.xmax == 40.0
        assert bbox.ymin == 0.0
        assert bbox.ymax < 20.0  # Should not include highest control point

    def test_bounding_box_caching(self):
        """Test that bounding box is cached after first calculation"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q"]

        glyph = AvGlyph(character="Q", width=20, points=points, commands=commands)

        # First call should calculate and cache
        bbox1 = glyph.bounding_box()
        # Check that bounding box is cached (internal testing)
        # pylint: disable=protected-access
        assert glyph._bounding_box is not None

        # Second call should return cached result
        bbox2 = glyph.bounding_box()
        assert bbox1 == bbox2
        # Verify the cached value is the same (internal testing)
        # pylint: disable=protected-access
        assert glyph._bounding_box == bbox1

    def test_right_side_bearing(self):
        """Test right side bearing calculation"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L", "L"]

        glyph = AvGlyph(character="R", width=15, points=points, commands=commands)
        rsb = glyph.right_side_bearing

        # Width=15, bbox xmax=10, so RSB should be 5
        assert rsb == 5.0

    def test_right_side_bearing_with_curves(self):
        """Test right side bearing with curves"""
        points = np.array([[0.0, 0.0, 0.0], [5.0, 15.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q"]

        glyph = AvGlyph(character="S", width=12, points=points, commands=commands)
        rsb = glyph.right_side_bearing

        # Width=12, bbox xmax should be 10.0, so RSB should be 2.0
        assert rsb == 2.0

    def test_bounding_box_accuracy_comparison(self):
        """Test that polygonized bounding box is more accurate than control point box"""
        # Create a curve where control points are far from actual curve
        points = np.array(
            [[0.0, 0.0, 0.0], [10.0, 50.0, 0.0], [20.0, 0.0, 0.0]],  # start  # control point (very high)  # end
            dtype=np.float64,
        )
        commands = ["M", "Q"]

        glyph = AvGlyph(character="T", width=20, points=points, commands=commands)

        # Get accurate bounding box using polygonization
        accurate_bbox = glyph.bounding_box()

        # Calculate naive bounding box using control points
        naive_ymax = points[:, 1].max()

        # The accurate bounding box should be smaller than naive
        # For a quadratic curve, the max y is typically around 1/4 to 1/2 of the control point height
        assert accurate_bbox.ymax < naive_ymax * 0.8  # Should be significantly smaller
        assert accurate_bbox.ymax > 0.0  # But still positive

    def test_complex_glyph_path(self):
        """Test bounding box with a complex glyph-like path"""
        # Simulate a simple 'A' glyph with curves
        points = np.array(
            [
                # Left leg
                [10.0, 0.0, 0.0],  # M - start bottom left
                [15.0, 30.0, 0.0],  # L - middle left
                # Cross bar with quadratic curve
                [20.0, 40.0, 0.0],  # Q - control point
                [25.0, 30.0, 0.0],  # Q - end middle right
                # Right leg
                [30.0, 0.0, 0.0],  # L - bottom right
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "Q", "L"]

        glyph = AvGlyph(character="A", width=40, points=points, commands=commands)
        bbox = glyph.bounding_box()

        assert bbox.xmin == 10.0
        assert bbox.xmax == 30.0
        assert bbox.ymin == 0.0
        assert bbox.ymax < 40.0  # Should not include control point
        assert bbox.ymax > 30.0  # Should include the curve peak

    def test_negative_coordinates(self):
        """Test bounding box with negative coordinates"""
        points = np.array(
            [[-10.0, -5.0, 0.0], [0.0, 10.0, 0.0], [10.0, -5.0, 0.0]], dtype=np.float64  # start  # control point  # end
        )
        commands = ["M", "Q"]

        glyph = AvGlyph(character="N", width=20, points=points, commands=commands)
        bbox = glyph.bounding_box()

        assert bbox.xmin == -10.0
        assert bbox.xmax == 10.0
        assert bbox.ymin == -5.0
        assert bbox.ymax < 10.0  # Should not include control point
        assert bbox.ymax > -5.0  # Should be higher than endpoints


if __name__ == "__main__":
    pytest.main([__file__])
