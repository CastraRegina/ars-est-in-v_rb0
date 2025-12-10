"""Test module for ave.geom

The tests are run using pytest.
These tests ensure that all functions and interfaces in src/ave/geom.py
remain working correctly after changes and refactoring.
"""

import numpy as np
import pytest

from ave.geom import AvBox, AvPath, AvSinglePath, GeomMath
from ave.geom_bezier import BezierCurve

###############################################################################
# GeomMath Tests
###############################################################################


class TestGeomMath:
    """Test class for GeomMath functionality."""

    def test_transform_point_identity(self):
        """Test point transformation with identity matrix."""
        affine_trafo = [1, 0, 0, 1, 0, 0]
        point = (10.0, 20.0)

        result = GeomMath.transform_point(affine_trafo, point)

        assert result == (10.0, 20.0)

    def test_transform_point_translation(self):
        """Test point transformation with translation."""
        affine_trafo = [1, 0, 0, 1, 5.0, 10.0]
        point = (10.0, 20.0)

        result = GeomMath.transform_point(affine_trafo, point)

        assert result == (15.0, 30.0)

    def test_transform_point_scale(self):
        """Test point transformation with scaling."""
        affine_trafo = [2, 0, 0, 3, 0, 0]
        point = (10.0, 20.0)

        result = GeomMath.transform_point(affine_trafo, point)

        assert result == (20.0, 60.0)

    def test_transform_point_complex(self):
        """Test point transformation with scaling, rotation, and translation."""
        # Scale by 2, rotate 90 degrees, translate by (5, 10)
        affine_trafo = [0, -2, 2, 0, 5.0, 10.0]
        point = (10.0, 20.0)

        result = GeomMath.transform_point(affine_trafo, point)

        # x' = 0*10 + (-2)*20 + 5 = -40 + 5 = -35
        # y' = 2*10 + 0*20 + 10 = 20 + 10 = 30
        assert result == (-35.0, 30.0)

    def test_transform_point_input_types(self):
        """Test transform_point with different input types."""
        affine_trafo = [1, 0, 0, 1, 5.0, 10.0]

        # Test with tuple
        result_tuple = GeomMath.transform_point(affine_trafo, (10.0, 20.0))
        assert result_tuple == (15.0, 30.0)

        # Test with list
        result_list = GeomMath.transform_point(affine_trafo, [10.0, 20.0])
        assert result_list == (15.0, 30.0)

        # Test with integer inputs
        result_int = GeomMath.transform_point([1, 0, 0, 1, 5, 10], (10, 20))
        assert result_int == (15.0, 30.0)

    def test_transform_point_return_type(self):
        """Test that transform_point always returns float tuple."""
        result = GeomMath.transform_point([1, 0, 0, 1, 0, 0], (10, 20))

        assert isinstance(result, tuple)
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


###############################################################################
# AvBox Tests
###############################################################################


class TestAvBox:
    """Test class for AvBox functionality."""

    def test_avbox_initialization_normal(self):
        """Test AvBox initialization with normal coordinates."""
        box = AvBox(10.0, 20.0, 30.0, 40.0)

        assert box.xmin == 10.0
        assert box.ymin == 20.0
        assert box.xmax == 30.0
        assert box.ymax == 40.0

    def test_avbox_initialization_reversed(self):
        """Test AvBox initialization with reversed coordinates."""
        box = AvBox(30.0, 40.0, 10.0, 20.0)

        # Should automatically reorder
        assert box.xmin == 10.0
        assert box.ymin == 20.0
        assert box.xmax == 30.0
        assert box.ymax == 40.0

    def test_avbox_properties(self):
        """Test AvBox property calculations."""
        box = AvBox(10.0, 20.0, 30.0, 40.0)

        assert box.width == 20.0
        assert box.height == 20.0
        assert box.area == 400.0
        assert box.centroid == (20.0, 30.0)
        assert box.extent == (10.0, 20.0, 30.0, 40.0)

    def test_avbox_zero_area(self):
        """Test AvBox with zero area."""
        box = AvBox(10.0, 20.0, 10.0, 20.0)

        assert box.width == 0.0
        assert box.height == 0.0
        assert box.area == 0.0
        assert box.centroid == (10.0, 20.0)

    def test_avbox_negative_coordinates(self):
        """Test AvBox with negative coordinates."""
        box = AvBox(-30.0, -40.0, -10.0, -20.0)

        assert box.xmin == -30.0
        assert box.ymin == -40.0
        assert box.xmax == -10.0
        assert box.ymax == -20.0
        assert box.width == 20.0
        assert box.height == 20.0

    def test_avbox_transform_affine_identity(self):
        """Test AvBox affine transformation with identity matrix."""
        box = AvBox(10.0, 20.0, 30.0, 40.0)
        affine_trafo = [1, 0, 0, 1, 0, 0]

        transformed = box.transform_affine(affine_trafo)

        assert transformed.xmin == 10.0
        assert transformed.ymin == 20.0
        assert transformed.xmax == 30.0
        assert transformed.ymax == 40.0

    def test_avbox_transform_affine_translation(self):
        """Test AvBox affine transformation with translation."""
        box = AvBox(10.0, 20.0, 30.0, 40.0)
        affine_trafo = [1, 0, 0, 1, 5.0, 10.0]

        transformed = box.transform_affine(affine_trafo)

        assert transformed.xmin == 15.0
        assert transformed.ymin == 30.0
        assert transformed.xmax == 35.0
        assert transformed.ymax == 50.0

    def test_avbox_transform_affine_scale(self):
        """Test AvBox affine transformation with scaling."""
        box = AvBox(10.0, 20.0, 30.0, 40.0)
        affine_trafo = [2, 0, 0, 3, 0, 0]

        transformed = box.transform_affine(affine_trafo)

        assert transformed.xmin == 20.0
        assert transformed.ymin == 60.0
        assert transformed.xmax == 60.0
        assert transformed.ymax == 120.0

    def test_avbox_transform_scale_translate(self):
        """Test AvBox scale and translate transformation."""
        box = AvBox(10.0, 20.0, 30.0, 40.0)

        transformed = box.transform_scale_translate(2.0, 5.0, 10.0)

        assert transformed.xmin == 25.0  # 10*2 + 5
        assert transformed.ymin == 50.0  # 20*2 + 10
        assert transformed.xmax == 65.0  # 30*2 + 5
        assert transformed.ymax == 90.0  # 40*2 + 10

    def test_avbox_str_representation(self):
        """Test AvBox string representation."""
        box = AvBox(10.0, 20.0, 30.0, 40.0)

        str_repr = str(box)

        assert "AvBox" in str_repr
        assert "xmin=10.0" in str_repr
        assert "ymin=20.0" in str_repr
        assert "xmax=30.0" in str_repr
        assert "ymax=40.0" in str_repr
        assert "width=20.0" in str_repr
        assert "height=20.0" in str_repr

    def test_avbox_immutability(self):
        """Test that AvBox properties are read-only."""
        box = AvBox(10.0, 20.0, 30.0, 40.0)

        # Properties should be read-only
        with pytest.raises(AttributeError):
            box.xmin = 15.0

        with pytest.raises(AttributeError):
            box.ymin = 25.0

        with pytest.raises(AttributeError):
            box.xmax = 35.0

        with pytest.raises(AttributeError):
            box.ymax = 45.0


class TestAvBoxSerialization:
    """Tests for AvBox.to_dict and AvBox.from_dict."""

    def test_avbox_to_from_dict_roundtrip(self):
        """Round-trip AvBox through to_dict and from_dict."""
        box = AvBox(10.0, 20.0, 30.0, 40.0)

        data = box.to_dict()
        restored = AvBox.from_dict(data)

        assert isinstance(data, dict)
        assert data == {"xmin": 10.0, "ymin": 20.0, "xmax": 30.0, "ymax": 40.0}
        assert restored.xmin == box.xmin
        assert restored.ymin == box.ymin
        assert restored.xmax == box.xmax
        assert restored.ymax == box.ymax

    def test_avbox_from_dict_missing_keys_defaults_to_zero(self):
        """from_dict should default missing values to 0.0."""
        data = {"xmin": 1.0}

        box = AvBox.from_dict(data)

        # Note: AvBox constructor reorders coordinates, so xmin=1.0, xmax=0.0 becomes xmin=0.0, xmax=1.0
        assert box.xmin == 0.0  # min(1.0, 0.0) = 0.0
        assert box.ymin == 0.0  # min(0.0, 0.0) = 0.0
        assert box.xmax == 1.0  # max(1.0, 0.0) = 1.0
        assert box.ymax == 0.0  # max(0.0, 0.0) = 0.0


###############################################################################
# Integration Tests
###############################################################################


class TestIntegration:
    """Integration tests to ensure overall functionality."""

    def test_complete_workflow_path_and_box(self):
        """Test complete workflow with path and box transformation."""

        # Create test path
        points = np.array([[10.0, 10.0, 0.0], [20.0, 20.0, 0.0], [30.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q"]

        # Polygonize path
        new_points, _ = AvPath.polygonize_path(points, commands, 5)

        # Create bounding box from polygonized points
        x_coords = new_points[:, 0]
        y_coords = new_points[:, 1]
        box = AvBox(np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords))

        # Transform box
        transformed_box = box.transform_scale_translate(2.0, 10.0, 20.0)

        # Verify transformation
        assert transformed_box.width == box.width * 2.0
        assert transformed_box.height == box.height * 2.0


###############################################################################
# AvPath Tests
###############################################################################


class TestAvPath:
    """Test class for AvPath functionality."""

    def test_avpath_init_2d_points_linear(self):
        """Test AvPath initialization with 2D points and linear commands."""
        points_2d = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]], dtype=np.float64)
        commands = ["M", "L", "L"]

        path = AvPath(points_2d, commands)

        # Points should be converted to 3D with type column
        assert path.points.shape == (3, 3)
        np.testing.assert_allclose(path.points[:, :2], points_2d)  # x,y coordinates should match

        # Type column should be all 0.0 for linear commands
        expected_types = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(path.points[:, 2], expected_types)

        assert path.commands == commands

    def test_avpath_init_2d_points_with_curves(self):
        """Test AvPath initialization with 2D points and curve commands."""
        points_2d = np.array(
            [[0.0, 0.0], [5.0, 10.0], [10.0, 0.0], [15.0, 5.0], [20.0, 5.0], [25.0, 0.0]], dtype=np.float64
        )
        commands = ["M", "Q", "C"]

        path = AvPath(points_2d, commands)

        # Points should be converted to 3D with type column
        assert path.points.shape == (6, 3)
        np.testing.assert_allclose(path.points[:, :2], points_2d)  # x,y coordinates should match

        # Type column should reflect command types:
        # M: 1 point -> 0.0
        # Q: 2 points -> 2.0 (control), 0.0 (end)
        # C: 3 points -> 3.0, 3.0, 0.0
        expected_types = np.array([0.0, 2.0, 0.0, 3.0, 3.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(path.points[:, 2], expected_types)

        assert path.commands == commands

    def test_avpath_init_3d_points_passthrough(self):
        """Test AvPath initialization with 3D points (should pass through unchanged)."""
        points_3d = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 2.0], [10.0, 5.0, 3.0]], dtype=np.float64)
        commands = ["M", "L", "L"]

        path = AvPath(points_3d, commands)

        # Points should remain exactly as provided
        assert path.points.shape == (3, 3)
        np.testing.assert_allclose(path.points, points_3d)  # Should be identical

        assert path.commands == commands

    def test_avpath_init_empty_points(self):
        """Test AvPath initialization with empty points."""
        points_empty = np.array([], dtype=np.float64).reshape(0, 2)
        commands = []

        path = AvPath(points_empty, commands)

        assert path.points.shape == (0, 3)
        assert path.commands == []

    def test_avpath_init_3d_points_empty(self):
        """Test AvPath initialization with empty 3D points."""
        points_empty = np.array([], dtype=np.float64).reshape(0, 3)
        commands = []

        path = AvPath(points_empty, commands)

        assert path.points.shape == (0, 3)
        assert path.commands == []

    def test_avpath_init_default_empty(self):
        """Test AvPath default initialization with no arguments."""

        path = AvPath()

        assert path.points.shape == (0, 3)
        assert path.commands == []

    def test_avpath_bounding_box_empty(self):
        """Bounding box for an empty path should be zero-sized at origin."""
        path = AvPath()

        bbox = path.bounding_box()

        assert bbox.xmin == 0.0
        assert bbox.ymin == 0.0
        assert bbox.xmax == 0.0
        assert bbox.ymax == 0.0

    def test_avpath_polygonize_empty(self):
        """Polygonizing an empty path should return an empty path."""
        path = AvPath()

        poly = path.polygonize(steps=5)

        assert poly.points.shape == (0, 3)
        assert poly.commands == []

    def test_avpath_validate_requires_segment_start_with_move(self):
        """Segments must start with an 'M' command."""
        points_2d = np.array([[0.0, 0.0]], dtype=np.float64)
        commands = ["L"]

        with pytest.raises(ValueError):
            AvPath(points_2d, commands)

    def test_avpath_validate_z_must_terminate_segment(self):
        """'Z' must terminate a segment and be followed only by 'M' or end."""
        points_2d = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        commands = ["M", "Z", "L"]

        with pytest.raises(ValueError):
            AvPath(points_2d, commands)

    def test_avpath_validate_points_commands_mismatch(self):
        """Number of points must match what commands require."""
        points_2d = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float64)
        commands = ["M", "L"]

        with pytest.raises(ValueError):
            AvPath(points_2d, commands)


###############################################################################
# AvPath Serialization Tests
###############################################################################


class TestAvPathSerialization:
    """Tests for AvPath.to_dict and AvPath.from_dict."""

    def test_avpath_serialization_without_cached_bounding_box(self):
        """AvPath serialization when bounding_box has not been computed yet."""
        points_2d = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]], dtype=np.float64)
        commands = ["M", "L", "L"]

        path = AvPath(points_2d, commands)

        # Ensure bounding box has not been computed/cached yet
        # pylint: disable=protected-access
        assert path._bounding_box is None

        data = path.to_dict()

        # Points must be 3D and serialized as list
        assert isinstance(data["points"], list)
        arr = np.asarray(data["points"], dtype=np.float64)
        assert arr.shape == (3, 3)

        # Commands round-trip as strings
        assert data["commands"] == commands

        # No bounding box cached yet -> None
        assert data["bounding_box"] is None

        restored = AvPath.from_dict(data)

        np.testing.assert_allclose(restored.points, path.points)
        assert restored.commands == path.commands
        # pylint: disable=protected-access
        assert restored._bounding_box is None

    def test_avpath_serialization_empty_path(self):
        path = AvPath()

        # pylint: disable=protected-access
        assert path._bounding_box is None

        data = path.to_dict()

        assert data["points"] == []
        assert data["commands"] == []
        assert data["bounding_box"] is None

        restored = AvPath.from_dict(data)

        assert restored.points.shape == (0, 3)
        assert restored.commands == []
        # pylint: disable=protected-access
        assert restored._bounding_box is None

    def test_avpath_serialization_with_cached_bounding_box(self):
        """AvPath serialization when bounding_box has been computed and cached."""
        points_2d = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]], dtype=np.float64)
        commands = ["M", "L", "L"]

        path = AvPath(points_2d, commands)

        # Trigger bounding box computation to populate cache
        bbox = path.bounding_box()

        data = path.to_dict()

        assert isinstance(data["bounding_box"], dict)
        assert data["bounding_box"] == bbox.to_dict()

        restored = AvPath.from_dict(data)

        np.testing.assert_allclose(restored.points, path.points)
        assert restored.commands == path.commands

        # Restored path should have an equivalent bounding box
        restored_bbox = restored.bounding_box()
        assert restored_bbox.xmin == bbox.xmin
        assert restored_bbox.ymin == bbox.ymin
        assert restored_bbox.xmax == bbox.xmax
        assert restored_bbox.ymax == bbox.ymax


###############################################################################
# AvSinglePath Tests
###############################################################################


class TestAvSinglePath:
    """Tests for AvSinglePath behavior."""

    def test_avsinglepath_empty(self):
        """Empty AvSinglePath should behave like an empty AvPath."""
        path = AvSinglePath()

        assert path.points.shape == (0, 3)
        assert path.commands == []

    def test_avsinglepath_single_segment_ok(self):
        """Single-segment AvSinglePath with M and L commands is valid."""
        points_2d = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]], dtype=np.float64)
        commands = ["M", "L", "L"]

        path = AvSinglePath(points_2d, commands)

        assert path.points.shape == (3, 3)
        assert path.commands == commands

    def test_avsinglepath_multiple_segments_raises(self):
        """AvSinglePath must not contain more than one segment."""
        points_2d = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "M", "L"]

        with pytest.raises(ValueError):
            AvSinglePath(points_2d, commands)

    def test_avsinglepath_reversed_path_empty(self):
        """Test reversed_path on empty path returns equivalent empty path."""
        path = AvSinglePath()
        reversed_path = path.reversed()

        assert len(reversed_path.points) == 0
        assert reversed_path.commands == []

    def test_avsinglepath_reversed_path_move_only(self):
        """Test reversed_path on path with only M command."""
        points = np.array([[5.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M"]

        path = AvSinglePath(points, commands)
        reversed_path = path.reversed()

        np.testing.assert_array_equal(reversed_path.points, points)
        assert reversed_path.commands == commands

    def test_avsinglepath_reversed_path_line_segment(self):
        """Test reversed_path on simple line segment."""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L"]

        path = AvSinglePath(points, commands)
        reversed_path = path.reversed()

        # Expected: start from end point, draw to start point
        expected_points = np.array([[10.0, 10.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "L"]

        np.testing.assert_array_equal(reversed_path.points, expected_points)
        assert reversed_path.commands == expected_commands

    def test_avsinglepath_reversed_path_multiple_lines(self):
        """Test reversed_path on multiple line segments."""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L"]

        path = AvSinglePath(points, commands)
        reversed_path = path.reversed()

        # Expected: start from last point, draw backwards
        expected_points = np.array([[10.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "L", "L"]

        np.testing.assert_array_equal(reversed_path.points, expected_points)
        assert reversed_path.commands == expected_commands

    def test_avsinglepath_reversed_path_quadratic_bezier(self):
        """Test reversed_path on quadratic bezier curve."""
        points = np.array([[0.0, 0.0, 0.0], [5.0, 10.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q"]

        path = AvSinglePath(points, commands)
        reversed_path = path.reversed()

        # Expected: start from end point, use same control point, end at start
        expected_points = np.array([[10.0, 0.0, 0.0], [5.0, 10.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "Q"]

        np.testing.assert_array_equal(reversed_path.points, expected_points)
        assert reversed_path.commands == expected_commands

    def test_avsinglepath_reversed_path_cubic_bezier(self):
        """Test reversed_path on cubic bezier curve."""
        points = np.array([[0.0, 0.0, 0.0], [3.0, 10.0, 0.0], [7.0, 10.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "C"]

        path = AvSinglePath(points, commands)
        reversed_path = path.reversed()

        # Expected: start from end point, control points swapped, end at start
        expected_points = np.array(
            [[10.0, 0.0, 0.0], [7.0, 10.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
        )
        expected_commands = ["M", "C"]

        np.testing.assert_array_equal(reversed_path.points, expected_points)
        assert reversed_path.commands == expected_commands

    def test_avsinglepath_reversed_path_closed_triangle(self):
        """Test reversed_path on closed triangle."""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L", "Z"]

        path = AvSinglePath(points, commands)
        reversed_path = path.reversed()

        # Expected: start from last point, draw backwards, close
        expected_points = np.array([[10.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "L", "L", "Z"]

        np.testing.assert_array_equal(reversed_path.points, expected_points)
        assert reversed_path.commands == expected_commands

    def test_avsinglepath_reversed_path_mixed_commands(self):
        """Test reversed_path on mixed commands (lines and curves)."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M
                [5.0, 0.0, 0.0],  # L
                [7.5, 5.0, 0.0],  # Q control
                [10.0, 0.0, 0.0],  # Q end
                [12.0, 2.0, 0.0],  # C control1
                [13.0, 8.0, 0.0],  # C control2
                [15.0, 0.0, 0.0],  # C end
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "Q", "C"]

        path = AvSinglePath(points, commands)
        reversed_path = path.reversed()

        # Expected: start from last point, process commands in reverse
        expected_points = np.array(
            [
                [15.0, 0.0, 0.0],  # M (original C end)
                [13.0, 8.0, 0.0],  # C control2 (swapped)
                [12.0, 2.0, 0.0],  # C control1 (swapped)
                [10.0, 0.0, 0.0],  # C end (original Q end)
                [7.5, 5.0, 0.0],  # Q control
                [5.0, 0.0, 0.0],  # Q end (original L end)
                [0.0, 0.0, 0.0],  # L end (original M point)
            ],
            dtype=np.float64,
        )
        expected_commands = ["M", "C", "Q", "L"]

        np.testing.assert_array_equal(reversed_path.points, expected_points)
        assert reversed_path.commands == expected_commands

    def test_avsinglepath_reversed_path_twice_returns_original(self):
        """Test that reversing twice returns to the original path."""
        points = np.array([[0.0, 0.0, 0.0], [5.0, 10.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q"]

        original_path = AvSinglePath(points, commands)
        reversed_once = original_path.reversed()
        reversed_twice = reversed_once.reversed()

        np.testing.assert_array_equal(reversed_twice.points, original_path.points)
        assert reversed_twice.commands == original_path.commands
