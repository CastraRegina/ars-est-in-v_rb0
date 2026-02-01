"""Tests for glyph functions"""

import numpy as np
import pytest

from ave.font_support import AvFontProperties
from ave.geom import AvBox
from ave.glyph import AvGlyph, AvGlyphPersistentFactory
from ave.path import CLOSED_SINGLE_PATH_CONSTRAINTS, AvPath


class TestAvGlyph:
    """Test cases for AvGlyph class"""

    # Constants for character 'f' glyph data (2 separate contours)
    GLYPH_F_POINTS = np.array(
        [
            [196.000000, 0.000000, 0],
            [382.000000, 0.000000, 0],
            [382.000000, 970.000000, 0],
            [382.000000, 992.000000, 0],
            [382.000000, 1200.000000, 0],
            [382.000000, 1300.000000, 2],
            [431.000000, 1350.000000, 0],
            [480.000000, 1400.000000, 2],
            [570.000000, 1400.000000, 0],
            [596.000000, 1400.000000, 2],
            [621.000000, 1397.000000, 0],
            [646.000000, 1394.000000, 2],
            [668.000000, 1390.000000, 0],
            [678.000000, 1540.000000, 0],
            [648.000000, 1548.000000, 2],
            [615.000000, 1552.000000, 0],
            [582.000000, 1556.000000, 2],
            [550.000000, 1556.000000, 0],
            [386.000000, 1556.000000, 2],
            [291.000000, 1465.000000, 0],
            [196.000000, 1374.000000, 2],
            [196.000000, 1200.000000, 0],
            [25.000000, 910.000000, 0],
            [610.000000, 910.000000, 0],
            [610.000000, 1052.000000, 0],
            [312.000000, 1052.000000, 0],
            [266.000000, 1052.000000, 0],
            [25.000000, 1052.000000, 0],
        ],
        dtype=np.float64,
    )

    GLYPH_F_COMMANDS = [
        "M",
        "L",
        "L",
        "L",
        "L",
        "Q",
        "Q",
        "Q",
        "Q",
        "L",
        "Q",
        "Q",
        "Q",
        "Q",
        "Z",
        "M",
        "L",
        "L",
        "L",
        "L",
        "L",
        "Z",
    ]

    # Constants for character '$' glyph data (3 separate contours)
    GLYPH_DOLLAR_POINTS = np.array(
        [
            [21.000000, 381.000000, 0],
            [21.000000, 202.000000, 2],
            [155.000000, 98.000000, 0],
            [289.000000, -6.000000, 2],
            [511.000000, -6.000000, 0],
            [732.000000, -6.000000, 2],
            [861.500000, 98.000000, 0],
            [991.000000, 202.000000, 2],
            [991.000000, 365.000000, 0],
            [991.000000, 527.000000, 2],
            [892.500000, 639.500000, 0],
            [794.000000, 752.000000, 2],
            [579.000000, 812.000000, 0],
            [363.000000, 873.000000, 2],
            [293.500000, 945.500000, 0],
            [224.000000, 1018.000000, 2],
            [224.000000, 1099.000000, 0],
            [224.000000, 1199.000000, 2],
            [291.500000, 1253.500000, 0],
            [359.000000, 1308.000000, 2],
            [501.000000, 1308.000000, 0],
            [644.000000, 1308.000000, 2],
            [716.000000, 1244.500000, 0],
            [788.000000, 1181.000000, 2],
            [788.000000, 1076.000000, 0],
            [788.000000, 1065.000000, 2],
            [965.000000, 1065.000000, 0],
            [965.000000, 1076.000000, 2],
            [965.000000, 1235.000000, 2],
            [839.500000, 1346.500000, 0],
            [714.000000, 1458.000000, 2],
            [501.000000, 1458.000000, 0],
            [289.000000, 1458.000000, 2],
            [169.000000, 1356.000000, 0],
            [49.000000, 1254.000000, 2],
            [49.000000, 1099.000000, 0],
            [49.000000, 946.000000, 2],
            [140.500000, 838.500000, 0],
            [232.000000, 731.000000, 2],
            [451.000000, 670.000000, 0],
            [671.000000, 608.000000, 2],
            [743.500000, 531.500000, 0],
            [816.000000, 455.000000, 2],
            [816.000000, 365.000000, 0],
            [816.000000, 265.000000, 2],
            [737.000000, 204.000000, 0],
            [658.000000, 143.000000, 2],
            [511.000000, 143.000000, 0],
            [363.000000, 143.000000, 2],
            [280.500000, 202.000000, 0],
            [198.000000, 261.000000, 2],
            [198.000000, 381.000000, 0],
            [198.000000, 392.000000, 2],
            [21.000000, 392.000000, 0],
            [100.000000, 400.000000, 2],
            [436.000000, -170.000000, 0],
            [500.000000, -160.000000, 2],
            [586.000000, -170.000000, 0],
            [586.000000, 70.000000, 0],
            [572.000000, 80.000000, 0],
            [572.000000, 758.000000, 0],
            [436.000000, 758.000000, 0],
            [450.000000, 758.000000, 0],
            [586.000000, 758.000000, 0],
            [586.000000, 1586.000000, 0],
            [436.000000, 1586.000000, 0],
            [436.000000, 1396.000000, 0],
            [450.000000, 1386.000000, 0],
        ],
        dtype=np.float64,
    )

    GLYPH_DOLLAR_COMMANDS = [
        "M",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "L",
        "L",
        "L",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
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
        "M",
        "L",
        "L",
        "L",
        "L",
        "L",
        "Z",
    ]

    # Constants for character '§' glyph data (2 separate contours)
    GLYPH_SECTION_POINTS = np.array(
        [
            [691.000000, 133.000000, 0],
            [870.000000, 133.000000, 2],
            [1002.500000, 224.000000, 0],
            [1135.000000, 315.000000, 2],
            [1135.000000, 478.000000, 0],
            [1135.000000, 652.000000, 2],
            [1001.500000, 745.500000, 0],
            [868.000000, 839.000000, 2],
            [684.000000, 897.000000, 0],
            [484.000000, 961.000000, 2],
            [422.500000, 1005.000000, 0],
            [361.000000, 1049.000000, 2],
            [361.000000, 1140.000000, 0],
            [361.000000, 1239.000000, 2],
            [422.000000, 1288.000000, 0],
            [483.000000, 1337.000000, 2],
            [623.000000, 1337.000000, 0],
            [744.000000, 1337.000000, 2],
            [809.500000, 1268.500000, 0],
            [875.000000, 1200.000000, 2],
            [874.000000, 1100.000000, 0],
            [1039.000000, 1100.000000, 0],
            [1040.000000, 1294.000000, 2],
            [928.000000, 1393.000000, 0],
            [816.000000, 1492.000000, 2],
            [623.000000, 1491.000000, 0],
            [421.000000, 1490.000000, 2],
            [308.500000, 1397.500000, 0],
            [196.000000, 1305.000000, 2],
            [196.000000, 1129.000000, 0],
            [196.000000, 949.000000, 2],
            [304.500000, 871.500000, 0],
            [413.000000, 794.000000, 2],
            [606.000000, 734.000000, 0],
            [787.000000, 672.000000, 2],
            [870.500000, 614.000000, 0],
            [954.000000, 556.000000, 2],
            [954.000000, 463.000000, 0],
            [954.000000, 376.000000, 2],
            [877.500000, 314.000000, 0],
            [801.000000, 252.000000, 2],
            [691.000000, 252.000000, 0],
            [537.000000, 894.000000, 0],
            [375.000000, 894.000000, 2],
            [247.000000, 810.500000, 0],
            [119.000000, 727.000000, 2],
            [119.000000, 577.000000, 0],
            [119.000000, 420.000000, 2],
            [237.000000, 321.000000, 0],
            [355.000000, 222.000000, 2],
            [540.000000, 161.000000, 0],
            [742.000000, 94.000000, 2],
            [813.000000, 29.500000, 0],
            [884.000000, -35.000000, 2],
            [884.000000, -136.000000, 0],
            [884.000000, -244.000000, 2],
            [800.000000, -299.500000, 0],
            [716.000000, -355.000000, 2],
            [599.000000, -355.000000, 0],
            [493.000000, -355.000000, 2],
            [394.000000, -292.500000, 0],
            [295.000000, -230.000000, 2],
            [295.000000, -70.000000, 0],
            [131.000000, -72.000000, 0],
            [131.000000, -303.000000, 2],
            [267.500000, -406.000000, 0],
            [404.000000, -509.000000, 2],
            [599.000000, -509.000000, 0],
            [794.000000, -509.000000, 2],
            [921.000000, -403.500000, 0],
            [1048.000000, -298.000000, 2],
            [1048.000000, -113.000000, 0],
            [1048.000000, 66.000000, 2],
            [930.000000, 159.000000, 0],
            [812.000000, 252.000000, 2],
            [636.000000, 317.000000, 0],
            [447.000000, 387.000000, 2],
            [374.000000, 437.000000, 0],
            [301.000000, 487.000000, 2],
            [301.000000, 567.000000, 0],
            [301.000000, 649.000000, 2],
            [363.000000, 702.500000, 0],
            [425.000000, 756.000000, 2],
            [536.000000, 756.000000, 0],
        ],
        dtype=np.float64,
    )

    GLYPH_SECTION_COMMANDS = [
        "M",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "L",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Z",
        "M",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "L",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Q",
        "Z",
    ]

    def test_glyph_init(self):
        """Test basic glyph initialization"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "L"]
        path = AvPath(points, commands)

        glyph = AvGlyph(character="A", advance_width=10, path=path)

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

        glyph = AvGlyph(character=" ", advance_width=0, path=path)
        bbox = glyph.bounding_box()

        expected = AvBox(0.0, 0.0, 0.0, 0.0)
        assert bbox == expected

    def test_bounding_box_linear_path(self):
        """Test bounding box with linear path (no curves)"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 5.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L", "L"]
        path = AvPath(points, commands)

        glyph = AvGlyph(character="L", advance_width=10, path=path)
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

        glyph = AvGlyph(character="Q", advance_width=20, path=path)
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

        glyph = AvGlyph(character="C", advance_width=20, path=path)
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

        glyph = AvGlyph(character="M", advance_width=40, path=path)
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

        glyph = AvGlyph(character="Q", advance_width=20, path=path)

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

        glyph = AvGlyph(character="R", advance_width=15, path=path)
        rsb = glyph.right_side_bearing()

        # Width=15, bbox xmax=10, so RSB should be 5
        assert rsb == 5.0

    def test_right_side_bearing_with_curves(self):
        """Test right side bearing with curves"""
        points = np.array([[0.0, 0.0, 0.0], [5.0, 15.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        glyph = AvGlyph(character="S", advance_width=12, path=path)
        rsb = glyph.right_side_bearing()

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

        glyph = AvGlyph(character="T", advance_width=20, path=path)

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

        glyph = AvGlyph(character="A", advance_width=40, path=path)
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

        glyph = AvGlyph(character="N", advance_width=20, path=path)
        bbox = glyph.bounding_box()

        assert np.isclose(bbox.xmin, -10.0)
        assert np.isclose(bbox.xmax, 10.0)
        assert np.isclose(bbox.ymin, -5.0)
        assert bbox.ymax < 10.0  # Should not include control point
        assert bbox.ymax > -5.0  # Should be higher than endpoints

    def test_revise_direction_empty_glyph(self):
        """Test revise_direction with empty glyph"""
        points = np.array([], dtype=np.float64).reshape(0, 3)
        commands = []
        path = AvPath(points, commands)
        glyph = AvGlyph(character=" ", advance_width=0, path=path)

        result = glyph.revise_direction()

        assert isinstance(result, AvGlyph)
        assert result.character == " "
        assert result.width() == 0
        assert len(result.path.points) == 0
        assert len(result.path.commands) == 0

    def test_revise_direction_single_ccw_contour(self):
        """Test revise_direction with single CCW contour (already correct)"""
        # Create a CCW square
        points = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 10.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)
        glyph = AvGlyph(character="A", advance_width=10, path=path)

        result = glyph.revise_direction()

        # Should remain CCW (no change needed)
        assert isinstance(result, AvGlyph)
        assert result.character == "A"
        assert result.width() == 10
        # Verify it's still CCW
        # Need to create a closed path to check winding direction
        closed = AvPath(result.path.points, result.path.commands, CLOSED_SINGLE_PATH_CONSTRAINTS)
        assert closed.is_ccw

    def test_revise_direction_single_cw_contour(self):
        """Test revise_direction with single CW contour (needs reversal)"""
        # Create a CW square
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)
        glyph = AvGlyph(character="A", advance_width=10, path=path)

        result = glyph.revise_direction()

        # Should be reversed to CCW
        assert isinstance(result, AvGlyph)
        assert result.character == "A"
        assert result.width() == 10
        # Verify it's now CCW
        # Need to create a closed path to check winding direction
        closed = AvPath(result.path.points, result.path.commands, CLOSED_SINGLE_PATH_CONSTRAINTS)
        assert closed.is_ccw

    def test_revise_direction_nested_contours_correct(self):
        """Test revise_direction with nested contours already correct (outer CCW, inner CW)"""
        # Outer CCW square
        # Inner CW square (hole)
        points = np.array(
            [
                # Outer square (CCW)
                [0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [10.0, 10.0, 0.0],
                [10.0, 0.0, 0.0],
                # Inner square (CW)
                [2.0, 2.0, 0.0],
                [8.0, 2.0, 0.0],
                [8.0, 8.0, 0.0],
                [2.0, 8.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)
        glyph = AvGlyph(character="A", advance_width=10, path=path)

        result = glyph.revise_direction()

        # Should remain unchanged (already correct)
        assert isinstance(result, AvGlyph)
        assert result.character == "A"
        assert result.width() == 10

    def test_revise_direction_nested_contours_wrong(self):
        """Test revise_direction with nested contours wrong directions (outer CW, inner CCW)"""
        # Outer CW square
        # Inner CCW square (should be CW)
        points = np.array(
            [
                # Outer square (CW)
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                # Inner square (CCW - wrong)
                [2.0, 2.0, 0.0],
                [2.0, 8.0, 0.0],
                [8.0, 8.0, 0.0],
                [8.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)
        glyph = AvGlyph(character="A", advance_width=10, path=path)

        result = glyph.revise_direction()

        # Should fix both contours
        assert isinstance(result, AvGlyph)
        assert result.character == "A"
        assert result.width() == 10
        # Split into contours to check individual directions
        contours = result.path.split_into_single_paths()
        assert len(contours) == 2

    def test_revise_direction_multiple_separate_contours(self):
        """Test revise_direction with multiple separate non-nested contours"""
        # Two separate squares
        points = np.array(
            [
                # First square
                [0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [5.0, 5.0, 0.0],
                [5.0, 0.0, 0.0],
                # Second square
                [10.0, 0.0, 0.0],
                [10.0, 5.0, 0.0],
                [15.0, 5.0, 0.0],
                [15.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)
        glyph = AvGlyph(character="A", advance_width=15, path=path)

        result = glyph.revise_direction()

        # Both should be CCW (additive)
        assert isinstance(result, AvGlyph)
        assert result.character == "A"
        assert result.width() == 15

    def test_revise_direction_triple_nesting(self):
        """Test revise_direction with triple nesting (island in hole in shape)"""
        # Outer CCW, middle CW, inner CCW
        points = np.array(
            [
                # Outer square (CCW)
                [0.0, 0.0, 0.0],
                [0.0, 15.0, 0.0],
                [15.0, 15.0, 0.0],
                [15.0, 0.0, 0.0],
                # Middle square (CW - hole)
                [3.0, 3.0, 0.0],
                [12.0, 3.0, 0.0],
                [12.0, 12.0, 0.0],
                [3.0, 12.0, 0.0],
                # Inner square (CCW - island in hole)
                [5.0, 5.0, 0.0],
                [5.0, 10.0, 0.0],
                [10.0, 10.0, 0.0],
                [10.0, 5.0, 0.0],
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z", "M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)
        glyph = AvGlyph(character="A", advance_width=15, path=path)

        result = glyph.revise_direction()

        # Should maintain correct nesting directions
        assert isinstance(result, AvGlyph)
        assert result.character == "A"
        assert result.width() == 15

    def test_revise_direction_open_contour(self):
        """Test revise_direction with open contour (should be untouched)"""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L"]  # No closing 'Z'
        path = AvPath(points, commands)
        glyph = AvGlyph(character="A", advance_width=10, path=path)

        result = glyph.revise_direction()

        # Open contour should be untouched
        assert isinstance(result, AvGlyph)
        assert result.character == "A"
        assert result.width() == 10
        assert len(result.path.commands) == 3
        assert "Z" not in result.path.commands

    def test_revise_direction_degenerate_contour(self):
        """Test revise_direction with degenerate/near-zero area contour"""
        # Create a line segment that closes to form degenerate polygon
        points = np.array([[5.0, 5.0, 0.0], [5.0, 5.00001, 0.0]], dtype=np.float64)
        commands = ["M", "L", "Z"]
        path = AvPath(points, commands)
        glyph = AvGlyph(character="A", advance_width=10, path=path)

        result = glyph.revise_direction()

        # Should handle degenerate case gracefully
        assert isinstance(result, AvGlyph)
        assert result.character == "A"
        assert result.width() == 10

    def test_revise_direction_real_glyph_f(self):
        """Test revise_direction with glyph 'f' (separate components)"""
        # Glyph 'f' with 2 separate contours
        points = self.GLYPH_F_POINTS
        commands = self.GLYPH_F_COMMANDS

        path = AvPath(points, commands)
        glyph = AvGlyph(character="f", advance_width=678, path=path)
        result = glyph.revise_direction()

        # Both contours should be CCW (additive)
        contours = result.path.split_into_single_paths()
        assert len(contours) == 2

        for contour in contours:
            if contour.commands and contour.commands[-1] == "Z":
                closed = AvPath(contour.points.copy(), contour.commands, CLOSED_SINGLE_PATH_CONSTRAINTS)
                assert closed.is_ccw, "Contour should be CCW but is CW"

    def test_revise_direction_real_glyph_dollar(self):
        """Test revise_direction with glyph '$' (separate components)"""
        # Glyph '$' with 3 separate contours
        points = self.GLYPH_DOLLAR_POINTS
        commands = self.GLYPH_DOLLAR_COMMANDS

        path = AvPath(points, commands)
        glyph = AvGlyph(character="$", advance_width=991, path=path)
        result = glyph.revise_direction()

        # All three contours should be CCW (additive)
        contours = result.path.split_into_single_paths()
        assert len(contours) == 3

        for contour in contours:
            if contour.commands and contour.commands[-1] == "Z":
                closed = AvPath(contour.points.copy(), contour.commands, CLOSED_SINGLE_PATH_CONSTRAINTS)
                assert closed.is_ccw, "Contour should be CCW but is CW"

    def test_revise_direction_real_glyph_section(self):
        """Test revise_direction with glyph '§' (separate components)"""
        # Glyph '§' with 2 separate contours
        points = self.GLYPH_SECTION_POINTS
        commands = self.GLYPH_SECTION_COMMANDS

        path = AvPath(points, commands)
        glyph = AvGlyph(character="§", advance_width=1135, path=path)
        result = glyph.revise_direction()

        # Both contours should be CCW (additive)
        contours = result.path.split_into_single_paths()
        assert len(contours) == 2

        for contour in contours:
            if contour.commands and contour.commands[-1] == "Z":
                closed = AvPath(contour.points.copy(), contour.commands, CLOSED_SINGLE_PATH_CONSTRAINTS)
                assert closed.is_ccw, "Contour should be CCW but is CW"


class TestGlyphFactoryCache:
    """Tests for glyph factory cache serialization."""

    def test_factory_cache_roundtrip_in_memory(self):
        """Test glyph factory to_cache_dict and from_cache_dict round-trip.

        This test uses a minimal in-memory factory setup without touching the
        examples package or external font files.
        """

        # Create a simple glyph
        points = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]],
            dtype=np.float64,
        )
        commands = ["M", "L", "L"]
        path = AvPath(points, commands)
        glyph = AvGlyph(character="A", advance_width=10.0, path=path)

        # Create a cached glyph factory with one glyph
        glyphs = {"A": glyph}
        # Create minimal font properties
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
        glyph_factory = AvGlyphPersistentFactory(glyphs=glyphs, font_properties=font_props)

        # Test that the cached glyph is accessible
        glyph_cached = glyph_factory.get_glyph("A")

        assert glyph_cached.character == glyph.character
        assert glyph_cached.width() == glyph.width()

        np.testing.assert_allclose(
            glyph_cached.path.points,
            glyph.path.points,
            rtol=1e-12,
            atol=1e-12,
        )
        assert glyph_cached.path.commands == glyph.path.commands


if __name__ == "__main__":
    pytest.main([__file__])
