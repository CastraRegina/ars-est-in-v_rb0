"""Tests for glyph functions"""

import numpy as np
import pytest

from ave.geom import AvBox, AvPath
from ave.glyph import AvFont, AvFontProperties, AvGlyph, AvGlyphCachedFactory


class TestAvGlyph:
    """Test cases for AvGlyph class"""

    def test_glyph_init(self):
        """Test basic glyph initialization"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "L"]
        path = AvPath(points, commands)

        glyph = AvGlyph(character="A", width=10, path=path)

        assert glyph.character == "A"
        assert glyph.width() == 10
        np.testing.assert_array_equal(glyph.path.points, points)
        assert glyph.path.commands == commands
        # Test that bounding box calculation works (public interface)
        bbox = glyph.bounding_box()
        assert bbox is not None

    def test_bounding_box_empty_points(self):
        """Test bounding box with empty points"""
        points = np.array([], dtype=np.float64).reshape(0, 3)
        commands = []
        path = AvPath(points, commands)

        glyph = AvGlyph(character=" ", width=0, path=path)
        bbox = glyph.bounding_box()

        expected = AvBox(0.0, 0.0, 0.0, 0.0)
        assert bbox == expected

    def test_bounding_box_linear_path(self):
        """Test bounding box with linear path (no curves)"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 5.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L", "L"]
        path = AvPath(points, commands)

        glyph = AvGlyph(character="L", width=10, path=path)
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
        path = AvPath(points, commands)

        glyph = AvGlyph(character="Q", width=20, path=path)
        bbox = glyph.bounding_box()

        # The actual quadratic curve should not reach y=20 (control point)
        # With 100 steps, the max y should be around 10.0 (midpoint of curve)
        assert np.isclose(bbox.xmin, 0.0)
        assert np.isclose(bbox.xmax, 20.0)
        assert np.isclose(bbox.ymin, 0.0, atol=1e-12)  # Allow tiny negative due to floating point
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
        path = AvPath(points, commands)

        glyph = AvGlyph(character="C", width=20, path=path)
        bbox = glyph.bounding_box()

        # The actual cubic curve should not reach y=25 (control points)
        assert np.isclose(bbox.xmin, 0.0)
        assert np.isclose(bbox.xmax, 20.0)
        assert np.isclose(bbox.ymin, 0.0)
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
        path = AvPath(points, commands)

        glyph = AvGlyph(character="M", width=40, path=path)
        bbox = glyph.bounding_box()

        assert np.isclose(bbox.xmin, 0.0)
        assert np.isclose(bbox.xmax, 40.0)
        assert np.isclose(bbox.ymin, 0.0)
        assert bbox.ymax < 20.0  # Should not include highest control point

    def test_bounding_box_caching(self):
        """Test that bounding box is cached after first calculation"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        glyph = AvGlyph(character="Q", width=20, path=path)

        # First call should calculate and cache
        bbox1 = glyph.bounding_box()
        # Check that bounding box is cached in AvPath (internal testing)
        # pylint: disable=protected-access
        assert glyph._path._bounding_box is not None

        # Second call should return cached result
        bbox2 = glyph.bounding_box()
        assert bbox1 == bbox2
        # Verify the cached value is the same (internal testing)
        # pylint: disable=protected-access
        assert glyph._path._bounding_box == bbox1

    def test_right_side_bearing(self):
        """Test right side bearing calculation"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L", "L"]
        path = AvPath(points, commands)

        glyph = AvGlyph(character="R", width=15, path=path)
        rsb = glyph.right_side_bearing

        # Width=15, bbox xmax=10, so RSB should be 5
        assert rsb == 5.0

    def test_right_side_bearing_with_curves(self):
        """Test right side bearing with curves"""
        points = np.array([[0.0, 0.0, 0.0], [5.0, 15.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        glyph = AvGlyph(character="S", width=12, path=path)
        rsb = glyph.right_side_bearing

        # Width=12, bbox xmax should be 10.0, so RSB should be 2.0
        assert np.isclose(rsb, 2.0)

    def test_bounding_box_accuracy_comparison(self):
        """Test that polygonized bounding box is more accurate than control point box"""
        # Create a curve where control points are far from actual curve
        points = np.array(
            [[0.0, 0.0, 0.0], [10.0, 50.0, 0.0], [20.0, 0.0, 0.0]],  # start  # control point (very high)  # end
            dtype=np.float64,
        )
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        glyph = AvGlyph(character="T", width=20, path=path)

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
        path = AvPath(points, commands)

        glyph = AvGlyph(character="A", width=40, path=path)
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
        path = AvPath(points, commands)

        glyph = AvGlyph(character="N", width=20, path=path)
        bbox = glyph.bounding_box()

        assert np.isclose(bbox.xmin, -10.0)
        assert np.isclose(bbox.xmax, 10.0)
        assert np.isclose(bbox.ymin, -5.0)
        assert bbox.ymax < 10.0  # Should not include control point
        assert bbox.ymax > -5.0  # Should be higher than endpoints


class TestAvFontCache:
    """Tests for AvFont cache serialization helpers."""

    def test_font_cache_roundtrip_in_memory(self):
        """Test AvFont.to_cache_dict and AvFont.from_cache_dict round-trip.

        This test uses a minimal in-memory font setup without touching the
        examples package or external font files.
        """

        # Create a simple glyph
        points = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]],
            dtype=np.float64,
        )
        commands = ["M", "L", "L"]
        path = AvPath(points, commands)
        glyph = AvGlyph(character="A", width=10.0, path=path)

        # Create a cached glyph factory with one glyph
        glyphs = {"A": glyph}
        glyph_factory = AvGlyphCachedFactory(glyphs=glyphs, source_factory=None)

        # Minimal font properties
        font_props = AvFontProperties(
            ascender=800.0,
            descender=-200.0,
            line_gap=0.0,
            x_height=400.0,
            cap_height=700.0,
            units_per_em=1000.0,
            family_name="TestFamily",
            subfamily_name="Regular",
            full_name="TestFamily Regular",
            license_description="Test license",
        )

        avfont = AvFont(glyph_factory=glyph_factory, font_properties=font_props)

        # Serialize to cache dict and reconstruct
        cache_dict = avfont.to_cache_dict()
        avfont_restored = AvFont.from_cache_dict(cache_dict)

        # Basic properties
        assert avfont_restored.props.family_name == avfont.props.family_name
        assert avfont_restored.props.units_per_em == avfont.props.units_per_em
        assert avfont_restored.props.ascender == avfont.props.ascender
        assert avfont_restored.props.descender == avfont.props.descender

        # Glyphs in cache
        glyph_restored = avfont_restored.get_glyph("A")

        assert glyph_restored.character == glyph.character
        assert glyph_restored.width() == glyph.width()

        np.testing.assert_allclose(
            glyph_restored.path.points,
            glyph.path.points,
            rtol=1e-12,
            atol=1e-12,
        )
        assert glyph_restored.path.commands == glyph.path.commands


if __name__ == "__main__":
    pytest.main([__file__])
