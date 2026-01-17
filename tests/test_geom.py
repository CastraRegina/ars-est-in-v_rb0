"""Test module for ave.geom

The tests are run using pytest.
These tests ensure that all functions and interfaces in src/ave/geom.py
remain working correctly after changes and refactoring.
"""

import numpy as np
import pytest

from ave.common import sgn_sci
from ave.geom import AvBox, GeomMath
from ave.path import SINGLE_PATH_CONSTRAINTS, AvPath

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
        box = AvBox(xmin=10.0, ymin=20.0, xmax=30.0, ymax=40.0)

        assert box.xmin == 10.0
        assert box.ymin == 20.0
        assert box.xmax == 30.0
        assert box.ymax == 40.0

    def test_avbox_initialization_reversed(self):
        """Test AvBox initialization with reversed coordinates."""
        box = AvBox(xmin=30.0, ymin=40.0, xmax=10.0, ymax=20.0)

        # Should automatically reorder
        assert box.xmin == 10.0
        assert box.ymin == 20.0
        assert box.xmax == 30.0
        assert box.ymax == 40.0

    def test_avbox_properties(self):
        """Test AvBox property calculations."""
        box = AvBox(xmin=10.0, ymin=20.0, xmax=30.0, ymax=40.0)

        assert box.width == 20.0
        assert box.height == 20.0
        assert box.area == 400.0
        assert box.centroid == (20.0, 30.0)
        assert box.extent == (10.0, 20.0, 30.0, 40.0)

    def test_avbox_zero_area(self):
        """Test AvBox with zero area."""
        box = AvBox(xmin=10.0, ymin=20.0, xmax=10.0, ymax=20.0)

        assert box.width == 0.0
        assert box.height == 0.0
        assert box.area == 0.0
        assert box.centroid == (10.0, 20.0)

    def test_avbox_negative_coordinates(self):
        """Test AvBox with negative coordinates."""
        box = AvBox(xmin=-30.0, ymin=-40.0, xmax=-10.0, ymax=-20.0)

        assert box.xmin == -30.0
        assert box.ymin == -40.0
        assert box.xmax == -10.0
        assert box.ymax == -20.0
        assert box.width == 20.0
        assert box.height == 20.0

    def test_avbox_transform_affine_identity(self):
        """Test AvBox affine transformation with identity matrix."""
        box = AvBox(xmin=10.0, ymin=20.0, xmax=30.0, ymax=40.0)
        affine_trafo = [1, 0, 0, 1, 0, 0]

        transformed = box.transform_affine(affine_trafo)

        assert transformed.xmin == 10.0
        assert transformed.ymin == 20.0
        assert transformed.xmax == 30.0
        assert transformed.ymax == 40.0

    def test_avbox_transform_affine_translation(self):
        """Test AvBox affine transformation with translation."""
        box = AvBox(xmin=10.0, ymin=20.0, xmax=30.0, ymax=40.0)
        affine_trafo = [1, 0, 0, 1, 5.0, 10.0]

        transformed = box.transform_affine(affine_trafo)

        assert transformed.xmin == 15.0
        assert transformed.ymin == 30.0
        assert transformed.xmax == 35.0
        assert transformed.ymax == 50.0

    def test_avbox_transform_affine_scale(self):
        """Test AvBox affine transformation with scaling."""
        box = AvBox(xmin=10.0, ymin=20.0, xmax=30.0, ymax=40.0)
        affine_trafo = [2, 0, 0, 3, 0, 0]

        transformed = box.transform_affine(affine_trafo)

        assert transformed.xmin == 20.0
        assert transformed.ymin == 60.0
        assert transformed.xmax == 60.0
        assert transformed.ymax == 120.0

    def test_avbox_transform_scale_translate(self):
        """Test AvBox scale and translate transformation."""
        box = AvBox(xmin=10.0, ymin=20.0, xmax=30.0, ymax=40.0)

        transformed = box.transform_scale_translate(2.0, 5.0, 10.0)

        assert transformed.xmin == 25.0  # 10*2 + 5
        assert transformed.ymin == 50.0  # 20*2 + 10
        assert transformed.xmax == 65.0  # 30*2 + 5
        assert transformed.ymax == 90.0  # 40*2 + 10

    def test_avbox_str_representation(self):
        """Test AvBox string representation."""
        box = AvBox(xmin=10.0, ymin=20.0, xmax=30.0, ymax=40.0)

        str_repr = str(box)

        assert "AvBox" in str_repr
        assert f"xmin={sgn_sci(10.0)}" in str_repr
        assert f"ymin={sgn_sci(20.0)}" in str_repr
        assert f"xmax={sgn_sci(30.0)}" in str_repr
        assert f"ymax={sgn_sci(40.0)}" in str_repr
        assert f"width={sgn_sci(20.0, always_positive=True)}" in str_repr
        assert f"height={sgn_sci(20.0, always_positive=True)}" in str_repr

    def test_avbox_immutability(self):
        """Test that AvBox properties are read-only."""
        box = AvBox(xmin=10.0, ymin=20.0, xmax=30.0, ymax=40.0)

        # Properties should be read-only
        with pytest.raises(AttributeError):
            box.xmin = 15.0

        with pytest.raises(AttributeError):
            box.ymin = 25.0

        with pytest.raises(AttributeError):
            box.xmax = 35.0

        with pytest.raises(AttributeError):
            box.ymax = 45.0

    def test_avbox_overlaps(self):
        """Test AvBox overlaps method."""
        # Test overlapping boxes
        box1 = AvBox(xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0)
        box2 = AvBox(xmin=5.0, ymin=5.0, xmax=15.0, ymax=15.0)
        assert box1.overlaps(box2) is True
        assert box2.overlaps(box1) is True

        # Test non-overlapping boxes (separated horizontally)
        box3 = AvBox(xmin=20.0, ymin=0.0, xmax=30.0, ymax=10.0)
        assert box1.overlaps(box3) is False
        assert box3.overlaps(box1) is False

        # Test non-overlapping boxes (separated vertically)
        box4 = AvBox(xmin=0.0, ymin=20.0, xmax=10.0, ymax=30.0)
        assert box1.overlaps(box4) is False
        assert box4.overlaps(box1) is False

        # Test touching boxes (edges touch but don't overlap)
        box5 = AvBox(xmin=10.0, ymin=0.0, xmax=20.0, ymax=10.0)
        assert box1.overlaps(box5) is True  # Touching edges are considered overlapping
        assert box5.overlaps(box1) is True

        # Test touching boxes (corners touch)
        box6 = AvBox(xmin=10.0, ymin=10.0, xmax=20.0, ymax=20.0)
        assert box1.overlaps(box6) is True  # Touching corners are considered overlapping
        assert box6.overlaps(box1) is True

        # Test one box completely inside another
        box7 = AvBox(xmin=2.0, ymin=2.0, xmax=8.0, ymax=8.0)
        assert box1.overlaps(box7) is True
        assert box7.overlaps(box1) is True

        # Test identical boxes
        assert box1.overlaps(box1) is True

        # Test with negative coordinates
        box8 = AvBox(xmin=-10.0, ymin=-10.0, xmax=0.0, ymax=0.0)
        box9 = AvBox(xmin=-5.0, ymin=-5.0, xmax=5.0, ymax=5.0)
        assert box8.overlaps(box9) is True
        assert box9.overlaps(box8) is True


class TestAvBoxSerialization:
    """Tests for AvBox.to_dict and AvBox.from_dict."""

    def test_avbox_to_from_dict_roundtrip(self):
        """Round-trip AvBox through to_dict and from_dict."""
        box = AvBox(xmin=10.0, ymin=20.0, xmax=30.0, ymax=40.0)

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
        path = AvPath(points, commands)
        new_points = path.polygonize(5).points

        # Create bounding box from polygonized points
        x_coords = new_points[:, 0]
        y_coords = new_points[:, 1]
        box = AvBox(xmin=np.min(x_coords), ymin=np.min(y_coords), xmax=np.max(x_coords), ymax=np.max(y_coords))

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

    def test_avpath_split_and_join_roundtrip(self):
        """Splitting into single paths and joining should reproduce the original path."""

        # Create a path with two segments
        points_2d = np.array(
            [
                [0.0, 0.0],  # First segment M
                [10.0, 0.0],  # L
                [10.0, 10.0],  # L
                [20.0, 20.0],  # Second segment M
                [30.0, 20.0],  # L
                [30.0, 30.0],  # L
            ],
            dtype=np.float64,
        )
        commands = ["M", "L", "L", "Z", "M", "L", "L"]

        original_path = AvPath(points_2d, commands)

        # Split into single-segment paths and join again
        single_paths = original_path.split_into_single_paths()

        # Join back using the first segment as the base path
        if len(single_paths) == 1:
            joined_path = single_paths[0]
        else:
            joined_path = single_paths[0].append(single_paths[1:])

        np.testing.assert_allclose(joined_path.points, original_path.points)
        assert joined_path.commands == original_path.commands

    def test_avpath_append_varargs_and_sequence(self):
        """append should handle varargs and sequence inputs equivalently."""

        # Three simple one-segment paths
        p1 = AvPath(np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64), ["M", "L"])
        p2 = AvPath(np.array([[20.0, 0.0], [30.0, 0.0]], dtype=np.float64), ["M", "L"])
        p3 = AvPath(np.array([[40.0, 0.0], [50.0, 0.0]], dtype=np.float64), ["M", "L"])

        joined_varargs = p1.append(p2, p3)
        joined_seq = p1.append([p2, p3])

        expected_points = np.concatenate([p1.points, p2.points, p3.points], axis=0)
        expected_commands = p1.commands + p2.commands + p3.commands

        np.testing.assert_allclose(joined_varargs.points, expected_points)
        assert joined_varargs.commands == expected_commands

        np.testing.assert_allclose(joined_seq.points, expected_points)
        assert joined_seq.commands == expected_commands

    def test_avpath_append_empty_base(self):
        """append on an empty base path should return the other path unchanged."""

        empty = AvPath()
        other_points = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        other_commands = ["M", "L"]
        other = AvPath(other_points, other_commands)

        joined = empty.append(other)

        np.testing.assert_allclose(joined.points, other.points)
        assert joined.commands == other.commands

    def test_avpath_join_paths_various_inputs(self):
        """join_paths should support varargs and sequence inputs."""

        p1 = AvPath(np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64), ["M", "L"])
        p2 = AvPath(np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float64), ["M", "L"])

        joined_varargs = AvPath.join_paths(p1, p2)
        joined_seq = AvPath.join_paths([p1, p2])

        expected_points = np.concatenate([p1.points, p2.points], axis=0)
        expected_commands = p1.commands + p2.commands

        np.testing.assert_allclose(joined_varargs.points, expected_points)
        assert joined_varargs.commands == expected_commands

        np.testing.assert_allclose(joined_seq.points, expected_points)
        assert joined_seq.commands == expected_commands

    def test_avpath_join_paths_empty_input(self):
        """join_paths with no arguments should return an empty path."""

        joined = AvPath.join_paths()

        assert joined.points.shape == (0, 3)
        assert joined.commands == []

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

    def test_avpath_reversed_test_cases(self):
        """AvPath.reversed should match expected commands and 2D coords for multiple scenarios."""

        test_cases = [
            # SIMPLE 1 - Open polyline (L only)
            dict(
                name="simple_line",
                commands=["M", "L", "L"],
                coords=[(0, 0), (10, 0), (10, 10)],
                exp_commands=["M", "L", "L"],
                exp_coords=[(10, 10), (10, 0), (0, 0)],
            ),
            # SIMPLE 2 - Closed polygon (with Z)
            dict(
                name="simple_closed_square",
                commands=["M", "L", "L", "L", "Z"],
                coords=[(0, 0), (10, 0), (10, 10), (0, 10)],
                exp_commands=["M", "L", "L", "L", "Z"],
                exp_coords=[(0, 10), (10, 10), (10, 0), (0, 0)],
            ),
            # SIMPLE 3 - Quadratic curve
            dict(
                name="simple_quad",
                commands=["M", "Q"],
                coords=[(0, 0), (5, 10), (10, 0)],
                exp_commands=["M", "Q"],
                exp_coords=[(10, 0), (5, 10), (0, 0)],
            ),
            # SIMPLE 4 - Cubic curve
            dict(
                name="simple_cubic",
                commands=["M", "C"],
                coords=[(0, 0), (5, 10), (10, 10), (15, 0)],
                exp_commands=["M", "C"],
                exp_coords=[(15, 0), (10, 10), (5, 10), (0, 0)],
            ),
            # SIMPLE 5 - Mixed Q + L + C
            dict(
                name="mixed_simple",
                commands=["M", "Q", "L", "C"],
                coords=[
                    (0, 0),  # M
                    (5, 10),
                    (10, 0),  # Q control + end
                    (20, 0),  # L
                    (25, 10),
                    (30, 10),
                    (35, 0),  # C c1, c2, end
                ],
                exp_commands=["M", "C", "L", "Q"],
                exp_coords=[
                    (35, 0),
                    (30, 10),
                    (25, 10),
                    (20, 0),
                    (10, 0),
                    (5, 10),
                    (0, 0),
                ],
            ),
            # MULTI 1 - Two contours, quadratic only
            dict(
                name="multi_quad",
                commands=["M", "Q", "Q", "Z", "M", "Q", "L"],
                coords=[
                    (0, 0),
                    (5, 10),
                    (10, 0),
                    (15, 10),
                    (20, 0),  # first contour
                    (30, 0),
                    (35, 10),
                    (40, 0),
                    (50, 0),  # second contour
                ],
                exp_commands=["M", "Q", "Q", "Z", "M", "L", "Q"],
                exp_coords=[
                    (20, 0),
                    (15, 10),
                    (10, 0),
                    (5, 10),
                    (0, 0),
                    (50, 0),
                    (40, 0),
                    (35, 10),
                    (30, 0),
                ],
            ),
            # MULTI 2 - Two contours mixing C and Q
            dict(
                name="multi_cq",
                commands=["M", "C", "Q", "Z", "M", "C"],
                coords=[
                    (0, 0),
                    (5, 10),
                    (10, 10),
                    (15, 0),
                    (20, 10),
                    (25, 0),  # first contour
                    (40, 0),
                    (45, 10),
                    (50, 10),
                    (55, 0),  # second contour
                ],
                exp_commands=["M", "Q", "C", "Z", "M", "C"],
                exp_coords=[
                    (25, 0),
                    (20, 10),
                    (15, 0),
                    (10, 10),
                    (5, 10),
                    (0, 0),
                    (55, 0),
                    (50, 10),
                    (45, 10),
                    (40, 0),
                ],
            ),
            # MULTI 3 - Open + closed, mixed commands
            dict(
                name="multi_mixed_three",
                commands=["M", "L", "Q", "M", "C", "L", "Z", "M", "Q", "C"],
                coords=[
                    (0, 0),
                    (10, 0),
                    (15, 10),
                    (20, 0),  # contour 1 open
                    (30, 0),
                    (35, 10),
                    (40, 10),
                    (45, 0),
                    (50, 0),  # contour 2 closed
                    (100, 100),
                    (110, 120),
                    (120, 100),
                    (130, 110),
                    (140, 110),
                    (150, 100),  # contour 3
                ],
                # Corrected expected commands to match AvPath.reversed semantics
                exp_commands=["M", "Q", "L", "M", "L", "C", "Z", "M", "C", "Q"],
                exp_coords=[
                    (20, 0),
                    (15, 10),
                    (10, 0),
                    (0, 0),
                    (50, 0),
                    (45, 0),
                    (40, 10),
                    (35, 10),
                    (30, 0),
                    (150, 100),
                    (140, 110),
                    (130, 110),
                    (120, 100),
                    (110, 120),
                    (100, 100),
                ],
            ),
            # MULTI 4 - Z then open contour with Q + C
            dict(
                name="multi_z_plus_open",
                commands=["M", "L", "C", "Z", "M", "Q", "C", "L"],
                coords=[
                    (0, 0),
                    (20, 0),
                    (25, 10),
                    (30, 10),
                    (35, 0),  # first closed
                    (50, 0),
                    (55, 10),
                    (60, 0),
                    (65, 10),
                    (70, 10),
                    (75, 0),
                    (80, 0),  # second open
                ],
                exp_commands=["M", "C", "L", "Z", "M", "L", "C", "Q"],
                exp_coords=[
                    (35, 0),
                    (30, 10),
                    (25, 10),
                    (20, 0),
                    (0, 0),
                    (80, 0),
                    (75, 0),
                    (70, 10),
                    (65, 10),
                    (60, 0),
                    (55, 10),
                    (50, 0),
                ],
            ),
        ]

        for case in test_cases:
            commands = case["commands"]
            coords_2d = np.array(case["coords"], dtype=np.float64)

            path = AvPath(coords_2d, commands)
            reversed_path = path.reverse()

            exp_coords_2d = np.array(case["exp_coords"], dtype=np.float64)

            # Check commands and 2D coordinates; the 3rd column encodes internal types.
            assert reversed_path.commands == case["exp_commands"], case["name"]
            np.testing.assert_allclose(
                reversed_path.points[:, :2],
                exp_coords_2d,
                err_msg=case["name"],
            )


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
# AvPath Single Path Tests (with SINGLE_PATH_CONSTRAINTS)
###############################################################################


class TestAvPathSinglePath:
    """Tests for AvPath with SINGLE_PATH_CONSTRAINTS behavior."""

    def test_avpath_single_empty(self):
        """Empty single-segment AvPath should behave like an empty AvPath."""
        path = AvPath(constraints=SINGLE_PATH_CONSTRAINTS)

        assert path.points.shape == (0, 3)
        assert path.commands == []

    def test_avpath_single_segment_ok(self):
        """Single-segment AvPath with M and L commands is valid."""
        points_2d = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]], dtype=np.float64)
        commands = ["M", "L", "L"]

        path = AvPath(points_2d, commands, SINGLE_PATH_CONSTRAINTS)

        assert path.points.shape == (3, 3)
        assert path.commands == commands

    def test_avpath_single_multiple_segments_raises(self):
        """Single-segment AvPath must not contain more than one segment."""
        points_2d = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "M", "L"]

        with pytest.raises(ValueError):
            AvPath(points_2d, commands, SINGLE_PATH_CONSTRAINTS)

    def test_avpath_reversed_path_empty(self):
        """Test reversed_path on empty path returns equivalent empty path."""
        path = AvPath(constraints=SINGLE_PATH_CONSTRAINTS)
        reversed_path = path.reverse()

        assert len(reversed_path.points) == 0
        assert reversed_path.commands == []

    def test_avpath_reversed_path_move_only(self):
        """Test reversed_path on path with only M command."""
        points = np.array([[5.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M"]

        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)
        reversed_path = path.reverse()

        np.testing.assert_array_equal(reversed_path.points, points)
        assert reversed_path.commands == commands

    def test_avpath_reversed_path_line_segment(self):
        """Test reversed_path on simple line segment."""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L"]

        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)
        reversed_path = path.reverse()

        # Expected: start from end point, draw to start point
        expected_points = np.array([[10.0, 10.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "L"]

        np.testing.assert_array_equal(reversed_path.points, expected_points)
        assert reversed_path.commands == expected_commands

    def test_avpath_reversed_path_multiple_lines(self):
        """Test reversed_path on multiple line segments."""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L"]

        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)
        reversed_path = path.reverse()

        # Expected: start from last point, draw backwards
        expected_points = np.array([[10.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "L", "L"]

        np.testing.assert_array_equal(reversed_path.points, expected_points)
        assert reversed_path.commands == expected_commands

    def test_avpath_reversed_path_quadratic_bezier(self):
        """Test reversed_path on quadratic bezier curve."""
        points = np.array([[0.0, 0.0, 0.0], [5.0, 10.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q"]

        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)
        reversed_path = path.reverse()

        # Expected: start from end point, use same control point, end at start
        expected_points = np.array([[10.0, 0.0, 0.0], [5.0, 10.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "Q"]

        np.testing.assert_array_equal(reversed_path.points, expected_points)
        assert reversed_path.commands == expected_commands

    def test_avpath_reversed_path_cubic_bezier(self):
        """Test reversed_path on cubic bezier curve."""
        points = np.array([[0.0, 0.0, 0.0], [3.0, 10.0, 0.0], [7.0, 10.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "C"]

        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)
        reversed_path = path.reverse()

        # Expected: start from end point, control points swapped, end at start
        expected_points = np.array(
            [[10.0, 0.0, 0.0], [7.0, 10.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
        )
        expected_commands = ["M", "C"]

        np.testing.assert_array_equal(reversed_path.points, expected_points)
        assert reversed_path.commands == expected_commands

    def test_avpath_reversed_path_closed_triangle(self):
        """Test reversed_path on closed triangle."""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L", "Z"]

        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)
        reversed_path = path.reverse()

        # Expected: start from last point, draw backwards, close
        expected_points = np.array([[10.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "L", "L", "Z"]

        np.testing.assert_array_equal(reversed_path.points, expected_points)
        assert reversed_path.commands == expected_commands

    def test_avpath_reversed_path_mixed_commands(self):
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

        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)
        reversed_path = path.reverse()

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

    def test_avpath_reversed_path_twice_returns_original(self):
        """Test that reversing twice returns to the original path."""
        points = np.array([[0.0, 0.0, 0.0], [5.0, 10.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q"]

        original_path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)
        reversed_once = original_path.reverse()
        reversed_twice = reversed_once.reverse()

        np.testing.assert_array_equal(reversed_twice.points, original_path.points)
        assert reversed_twice.commands == original_path.commands


###############################################################################
# make_closed_single() Tests
###############################################################################


class TestMakeClosedSingle:
    """Test class for AvPath.make_closed_single() functionality."""

    def test_empty_path(self):
        """Test make_closed_single with empty path."""
        points = np.array([], dtype=np.float64).reshape(0, 3)
        commands = []
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        assert closed.points.shape == (0, 3)
        assert closed.commands == []

    def test_single_point_path(self):
        """Test make_closed_single with single point."""
        points = np.array([[1.0, 2.0, 0.0]], dtype=np.float64)
        commands = ["M"]
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        # For single point, adding Z creates a closed path
        np.testing.assert_array_equal(closed.points, points)
        assert closed.commands == ["M", "Z"]

    def test_line_command_with_duplicate_endpoint(self):
        """Test line command where last point duplicates first point."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L"]
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        # Should remove duplicate point for line command
        expected_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "L", "Z"]

        np.testing.assert_array_equal(closed.points, expected_points)
        assert closed.commands == expected_commands

    def test_line_command_without_duplicate_endpoint(self):
        """Test line command where last point does not duplicate first point."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L"]
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        # Should keep all points and add Z
        expected_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "L", "L", "Z"]

        np.testing.assert_array_equal(closed.points, expected_points)
        assert closed.commands == expected_commands

    def test_quadratic_command_with_duplicate_endpoint(self):
        """Test quadratic curve command where last point duplicates first point."""
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q"]  # Q is the last drawing command
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        # Should keep duplicate point for curve command
        expected_points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "Q", "Z"]

        np.testing.assert_array_equal(closed.points, expected_points)
        assert closed.commands == expected_commands

    def test_cubic_command_with_duplicate_endpoint(self):
        """Test cubic curve command where last point duplicates first point."""
        points = np.array([[0.0, 0.0, 0.0], [0.3, 0.5, 0.0], [0.7, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "C"]  # C is the last drawing command
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        # Should keep duplicate point for curve command
        expected_points = np.array(
            [[0.0, 0.0, 0.0], [0.3, 0.5, 0.0], [0.7, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
        )
        expected_commands = ["M", "C", "Z"]

        np.testing.assert_array_equal(closed.points, expected_points)
        assert closed.commands == expected_commands

    def test_mixed_commands_with_duplicate_endpoint(self):
        """Test mixed commands where last command is line with duplicate endpoint."""
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "Q", "L"]  # L is the last drawing command
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        # Last command is L, should remove duplicate L command and its point
        expected_points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "Q", "Z"]  # L removed, Z added

        np.testing.assert_array_equal(closed.points, expected_points)
        assert closed.commands == expected_commands

    def test_path_already_closed_with_z(self):
        """Test path that already ends with Z command."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        commands = ["M", "L", "Z"]
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        # Should handle existing Z correctly
        expected_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "L", "Z"]

        np.testing.assert_array_equal(closed.points, expected_points)
        assert closed.commands == expected_commands

    def test_near_duplicate_points_within_tolerance(self):
        """Test points that are very close but not exactly equal."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1e-11, 1e-11, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L"]
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        # Should remove near-duplicate point for line command
        expected_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "L", "Z"]

        np.testing.assert_array_equal(closed.points, expected_points)
        assert closed.commands == expected_commands

    def test_points_outside_tolerance(self):
        """Test points that are close but outside tolerance."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1e-9, 1e-9, 0.0]], dtype=np.float64)
        commands = ["M", "L", "L"]
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        # Should keep all points as they're outside tolerance
        expected_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1e-9, 1e-9, 0.0]], dtype=np.float64)
        expected_commands = ["M", "L", "L", "Z"]

        np.testing.assert_array_equal(closed.points, expected_points)
        assert closed.commands == expected_commands

    def test_move_command_with_duplicate_endpoint(self):
        """Test line command where last point duplicates first point."""
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        commands = ["M", "L"]  # M moves to first point, L draws to second (duplicate) point
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        # Should remove duplicate point and L command since L endpoint duplicates M startpoint
        expected_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        expected_commands = ["M", "Z"]  # L removed, Z added

        np.testing.assert_array_equal(closed.points, expected_points)
        assert closed.commands == expected_commands

    def test_complex_curve_path(self):
        """Test complex path with multiple curve commands."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M
                [0.5, 1.0, 0.0],  # Q control
                [1.0, 0.0, 0.0],  # Q end
                [1.5, -0.5, 0.0],  # C control1
                [2.0, 0.5, 0.0],  # C control2
                [2.5, 0.0, 0.0],  # C end
                [0.0, 0.0, 0.0],  # L duplicate
            ],
            dtype=np.float64,
        )
        commands = ["M", "Q", "C", "L"]
        path = AvPath(points, commands, SINGLE_PATH_CONSTRAINTS)

        closed = AvPath.make_closed_single(path)

        # Last command is L, should remove duplicate L command and its point
        expected_points = np.array(
            [
                [0.0, 0.0, 0.0],  # M
                [0.5, 1.0, 0.0],  # Q control
                [1.0, 0.0, 0.0],  # Q end
                [1.5, -0.5, 0.0],  # C control1
                [2.0, 0.5, 0.0],  # C control2
                [2.5, 0.0, 0.0],  # C end
            ],
            dtype=np.float64,
        )
        expected_commands = ["M", "Q", "C", "Z"]  # L removed, Z added

        np.testing.assert_array_equal(closed.points, expected_points)
        assert closed.commands == expected_commands


###############################################################################
# AvBox Calculation Methods Tests
###############################################################################


class TestAvBoxCalculations:
    """Test class for AvBox calculation methods."""

    def test_width_calculation(self):
        """Test width calculation for various box configurations."""
        # Normal box
        box = AvBox(0, 0, 10, 20)
        assert box.width == 10.0

        # Negative coordinates
        box = AvBox(-10, -5, -2, -1)
        assert box.width == 8.0

        # Reversed coordinates (should auto-correct)
        box = AvBox(10, 0, 0, 20)
        assert box.width == 10.0

        # Zero width
        box = AvBox(5, 0, 5, 10)
        assert box.width == 0.0

    def test_height_calculation(self):
        """Test height calculation for various box configurations."""
        # Normal box
        box = AvBox(0, 0, 10, 20)
        assert box.height == 20.0

        # Negative coordinates
        box = AvBox(-10, -15, -2, -5)
        assert box.height == 10.0

        # Reversed coordinates (should auto-correct)
        box = AvBox(0, 20, 10, 0)
        assert box.height == 20.0

        # Zero height
        box = AvBox(0, 5, 10, 5)
        assert box.height == 0.0

    def test_area_calculation(self):
        """Test area calculation for various box configurations."""
        # Normal rectangle
        box = AvBox(0, 0, 10, 20)
        assert box.area == 200.0

        # Square
        box = AvBox(-5, -5, 5, 5)
        assert box.area == 100.0

        # Zero area (zero width)
        box = AvBox(5, 0, 5, 10)
        assert box.area == 0.0

        # Zero area (zero height)
        box = AvBox(0, 5, 10, 5)
        assert box.area == 0.0

        # Zero area (point)
        box = AvBox(5, 5, 5, 5)
        assert box.area == 0.0

    def test_centroid_calculation(self):
        """Test centroid calculation for various box configurations."""
        # Normal box
        box = AvBox(0, 0, 10, 20)
        assert box.centroid == (5.0, 10.0)

        # Negative coordinates
        box = AvBox(-10, -20, 0, 0)
        assert box.centroid == (-5.0, -10.0)

        # Asymmetric box
        box = AvBox(2, 3, 8, 15)
        assert box.centroid == (5.0, 9.0)

        # Zero area box
        box = AvBox(5, 5, 5, 5)
        assert box.centroid == (5.0, 5.0)

    def test_extent_property(self):
        """Test extent property returns correct tuple."""
        box = AvBox(10, 20, 30, 40)
        assert box.extent == (10.0, 20.0, 30.0, 40.0)

        # Reversed coordinates
        box = AvBox(30, 40, 10, 20)
        assert box.extent == (10.0, 20.0, 30.0, 40.0)


class TestAvBoxCombine:
    """Test class for AvBox.combine static method."""

    def test_combine_two_boxes(self):
        """Test combining two valid boxes."""
        box1 = AvBox(0, 0, 10, 10)
        box2 = AvBox(5, 5, 15, 15)

        result = AvBox.combine(box1, box2)

        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 15.0
        assert result.ymax == 15.0
        assert result.width == 15.0
        assert result.height == 15.0

    def test_combine_disjoint_boxes(self):
        """Test combining non-overlapping boxes."""
        box1 = AvBox(0, 0, 5, 5)
        box2 = AvBox(10, 10, 15, 15)
        box3 = AvBox(-10, -10, -5, -5)

        result = AvBox.combine(box1, box2, box3)

        assert result.xmin == -10.0
        assert result.ymin == -10.0
        assert result.xmax == 15.0
        assert result.ymax == 15.0

    def test_combine_static_method_with_list_input(self):
        """Test combining using a list of boxes."""
        boxes = [AvBox(0, 0, 10, 10), AvBox(5, 5, 15, 15), AvBox(-5, -5, 5, 5)]

        result = AvBox.combine(boxes)

        assert result.xmin == -5.0
        assert result.ymin == -5.0
        assert result.xmax == 15.0
        assert result.ymax == 15.0

    def test_combine_ignores_zero_width_boxes(self):
        """Test that boxes with zero width are ignored."""
        box1 = AvBox(0, 0, 10, 10)
        zero_width = AvBox(20, 20, 20, 30)  # width = 0
        box2 = AvBox(5, 5, 15, 15)

        result = AvBox.combine(box1, zero_width, box2)

        # Should ignore zero_width box
        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 15.0
        assert result.ymax == 15.0

    def test_combine_ignores_zero_height_boxes(self):
        """Test that boxes with zero height are ignored."""
        box1 = AvBox(0, 0, 10, 10)
        zero_height = AvBox(20, 20, 30, 20)  # height = 0
        box2 = AvBox(5, 5, 15, 15)

        result = AvBox.combine(box1, zero_height, box2)

        # Should ignore zero_height box
        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 15.0
        assert result.ymax == 15.0

    def test_combine_ignores_zero_area_boxes(self):
        """Test that boxes with zero area (point) are ignored."""
        box1 = AvBox(0, 0, 10, 10)
        point = AvBox(20, 20, 20, 20)  # width = 0, height = 0
        box2 = AvBox(5, 5, 15, 15)

        result = AvBox.combine(box1, point, box2)

        # Should ignore point box
        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 15.0
        assert result.ymax == 15.0

    def test_combine_zero_width_box_extending_bounds_is_ignored(self):
        """Test that zero-width box extending beyond valid bounds is ignored."""
        box1 = AvBox(0, 0, 10, 10)
        # Zero-width box that extends beyond valid box bounds
        zero_width_outside = AvBox(-10, 5, -10, 15)  # Would extend xmin if considered
        zero_width_outside2 = AvBox(20, 5, 20, 15)  # Would extend xmax if considered
        box2 = AvBox(5, 5, 15, 15)

        result = AvBox.combine(box1, zero_width_outside, zero_width_outside2, box2)

        # Should ignore zero-width boxes even though they extend bounds
        assert result.xmin == 0.0  # Not -10
        assert result.ymin == 0.0
        assert result.xmax == 15.0  # Not 20
        assert result.ymax == 15.0

    def test_combine_zero_height_box_extending_bounds_is_ignored(self):
        """Test that zero-height box extending beyond valid bounds is ignored."""
        box1 = AvBox(0, 0, 10, 10)
        # Zero-height box that extends beyond valid box bounds
        zero_height_outside = AvBox(5, -10, 15, -10)  # Would extend ymin if considered
        zero_height_outside2 = AvBox(5, 20, 15, 20)  # Would extend ymax if considered
        box2 = AvBox(5, 5, 15, 15)

        result = AvBox.combine(box1, zero_height_outside, zero_height_outside2, box2)

        # Should ignore zero-height boxes even though they extend bounds
        assert result.xmin == 0.0
        assert result.ymin == 0.0  # Not -10
        assert result.xmax == 15.0
        assert result.ymax == 15.0  # Not 20

    def test_combine_zero_area_point_extending_bounds_is_ignored(self):
        """Test that zero-area point extending beyond valid bounds is ignored."""
        box1 = AvBox(0, 0, 10, 10)
        # Points that extend beyond valid box bounds
        point_outside1 = AvBox(-10, -10, -10, -10)  # Would extend both xmin and ymin
        point_outside2 = AvBox(20, 20, 20, 20)  # Would extend both xmax and ymax
        box2 = AvBox(5, 5, 15, 15)

        result = AvBox.combine(box1, point_outside1, point_outside2, box2)

        # Should ignore point boxes even though they extend bounds
        assert result.xmin == 0.0  # Not -10
        assert result.ymin == 0.0  # Not -10
        assert result.xmax == 15.0  # Not 20
        assert result.ymax == 15.0  # Not 20

    def test_combine_mixed_zero_area_boxes_all_ignored(self):
        """Test that various zero-area boxes are all ignored."""
        valid_box = AvBox(10, 10, 20, 20)

        # Various types of zero-area boxes
        zero_width1 = AvBox(5, 15, 5, 25)  # Zero width
        zero_width2 = AvBox(25, 15, 25, 25)  # Zero width
        zero_height1 = AvBox(15, 5, 25, 5)  # Zero height
        zero_height2 = AvBox(15, 25, 25, 25)  # Zero height
        point1 = AvBox(5, 5, 5, 5)  # Point
        point2 = AvBox(30, 30, 30, 30)  # Point

        result = AvBox.combine(zero_width1, valid_box, zero_width2, zero_height1, point1, zero_height2, point2)

        # Should only consider the valid box
        assert result.xmin == 10.0
        assert result.ymin == 10.0
        assert result.xmax == 20.0
        assert result.ymax == 20.0

    def test_combine_zero_area_boxes_with_negative_coordinates(self):
        """Test zero-area boxes with negative coordinates are ignored."""
        valid_box = AvBox(-5, -5, 5, 5)

        # Zero-area boxes with negative coordinates
        zero_width_neg = AvBox(-10, -5, -10, 5)  # Zero width, negative x
        zero_height_neg = AvBox(-5, -10, 5, -10)  # Zero height, negative y
        point_neg = AvBox(-10, -10, -10, -10)  # Point, negative coordinates

        result = AvBox.combine(valid_box, zero_width_neg, zero_height_neg, point_neg)

        # Should ignore zero-area boxes
        assert result.xmin == -5.0  # Not -10
        assert result.ymin == -5.0  # Not -10
        assert result.xmax == 5.0
        assert result.ymax == 5.0

    def test_combine_zero_width_box_inside_valid_bounds(self):
        """Test zero-width box inside valid bounds is ignored."""
        valid_box = AvBox(0, 0, 20, 20)
        zero_width_inside = AvBox(10, 5, 10, 15)  # Zero width, inside valid box

        result = AvBox.combine(valid_box, zero_width_inside)

        # Should ignore zero-width box
        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 20.0
        assert result.ymax == 20.0

    def test_combine_zero_height_box_inside_valid_bounds(self):
        """Test zero-height box inside valid bounds is ignored."""
        valid_box = AvBox(0, 0, 20, 20)
        zero_height_inside = AvBox(5, 10, 15, 10)  # Zero height, inside valid box

        result = AvBox.combine(valid_box, zero_height_inside)

        # Should ignore zero-height box
        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 20.0
        assert result.ymax == 20.0

    def test_combine_line_boxes_ignored(self):
        """Test that horizontal and vertical line boxes are ignored."""
        valid_box1 = AvBox(0, 0, 10, 10)
        valid_box2 = AvBox(20, 20, 30, 30)

        # Horizontal line (zero height)
        h_line = AvBox(5, 15, 25, 15)
        # Vertical line (zero width)
        v_line = AvBox(15, 5, 15, 25)

        result = AvBox.combine(valid_box1, h_line, v_line, valid_box2)

        # Should ignore line boxes
        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 30.0
        assert result.ymax == 30.0

    def test_combine_single_box(self):
        """Test combining a single box returns equivalent box."""
        box = AvBox(5, 10, 15, 20)

        result = AvBox.combine(box)

        assert result.xmin == box.xmin
        assert result.ymin == box.ymin
        assert result.xmax == box.xmax
        assert result.ymax == box.ymax

    def test_combine_nested_boxes(self):
        """Test combining boxes where one is inside another."""
        outer = AvBox(0, 0, 20, 20)
        inner = AvBox(5, 5, 15, 15)

        result = AvBox.combine(outer, inner)

        # Should return the outer box
        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 20.0
        assert result.ymax == 20.0

    def test_combine_edge_touching_boxes(self):
        """Test combining boxes that touch at edges."""
        box1 = AvBox(0, 0, 10, 10)
        box2 = AvBox(10, 0, 20, 10)  # Touches at right edge
        box3 = AvBox(0, 10, 10, 20)  # Touches at top edge

        result = AvBox.combine(box1, box2, box3)

        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 20.0
        assert result.ymax == 20.0

    def test_combine_negative_coordinates(self):
        """Test combining boxes with negative coordinates."""
        box1 = AvBox(-20, -30, -10, -20)
        box2 = AvBox(-15, -25, -5, -15)
        box3 = AvBox(0, 0, 10, 10)

        result = AvBox.combine(box1, box2, box3)

        assert result.xmin == -20.0
        assert result.ymin == -30.0
        assert result.xmax == 10.0
        assert result.ymax == 10.0

    def test_combine_empty_list_raises_error(self):
        """Test that combining empty list raises ValueError."""
        with pytest.raises(ValueError, match="At least one AvBox must be provided"):
            AvBox.combine([])

    def test_combine_only_zero_area_boxes_returns_first(self):
        """Test that combining only zero-area boxes returns a copy of the first box."""
        zero_width = AvBox(5, 5, 5, 10)
        zero_height = AvBox(5, 5, 10, 5)
        point = AvBox(5, 5, 5, 5)

        result = AvBox.combine(zero_width, zero_height, point)

        # Should return a copy of the first box
        assert result.xmin == zero_width.xmin
        assert result.ymin == zero_width.ymin
        assert result.xmax == zero_width.xmax
        assert result.ymax == zero_width.ymax
        # Ensure it's a copy, not the same object
        assert result is not zero_width

    def test_combine_with_mixed_valid_and_invalid_boxes(self):
        """Test combining mix of valid and invalid boxes."""
        valid1 = AvBox(0, 0, 10, 10)
        invalid1 = AvBox(5, 5, 5, 5)  # Point
        valid2 = AvBox(20, 20, 30, 30)
        invalid2 = AvBox(25, 25, 25, 35)  # Zero width

        result = AvBox.combine(valid1, invalid1, valid2, invalid2)

        # Should only consider valid boxes
        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 30.0
        assert result.ymax == 30.0

    def test_combine_preserves_input_boxes(self):
        """Test that combine doesn't modify input boxes."""
        box1 = AvBox(0, 0, 10, 10)
        box2 = AvBox(5, 5, 15, 15)

        original_box1 = (box1.xmin, box1.ymin, box1.xmax, box1.ymax)
        original_box2 = (box2.xmin, box2.ymin, box2.xmax, box2.ymax)

        result = AvBox.combine(box1, box2)

        # Input boxes should be unchanged
        assert (box1.xmin, box1.ymin, box1.xmax, box1.ymax) == original_box1
        assert (box2.xmin, box2.ymin, box2.xmax, box2.ymax) == original_box2
        # Result should be new box
        assert result is not box1
        assert result is not box2

    def test_combine_with_generator(self):
        """Test combining using a generator expression."""
        boxes = (AvBox(i, i, i + 10, i + 10) for i in range(0, 30, 10))

        result = AvBox.combine(boxes)

        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 30.0
        assert result.ymax == 30.0

    def test_combine_with_tuple(self):
        """Test combining using a tuple of boxes."""
        boxes = (AvBox(0, 0, 10, 10), AvBox(5, 5, 15, 15), AvBox(20, 20, 30, 30))

        result = AvBox.combine(boxes)

        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 30.0
        assert result.ymax == 30.0

    def test_combine_with_single_box(self):
        """Test combine_with instance method with a single other box."""
        box1 = AvBox(0, 0, 10, 10)
        box2 = AvBox(5, 5, 15, 15)

        result = box1.combine_with(box2)

        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 15.0
        assert result.ymax == 15.0

    def test_combine_with_multiple_boxes(self):
        """Test combine_with instance method with multiple other boxes."""
        box1 = AvBox(0, 0, 10, 10)
        box2 = AvBox(5, 5, 15, 15)
        box3 = AvBox(-5, -5, 5, 5)

        result = box1.combine_with(box2, box3)

        assert result.xmin == -5.0
        assert result.ymin == -5.0
        assert result.xmax == 15.0
        assert result.ymax == 15.0

    def test_combine_with_instance_method_list_input(self):
        """Test combine_with instance method with a list of boxes."""
        box1 = AvBox(0, 0, 10, 10)
        boxes = [AvBox(5, 5, 15, 15), AvBox(-5, -5, 5, 5)]

        result = box1.combine_with(boxes)

        assert result.xmin == -5.0
        assert result.ymin == -5.0
        assert result.xmax == 15.0
        assert result.ymax == 15.0

    def test_combine_with_ignores_zero_area_boxes(self):
        """Test combine_with ignores zero-area boxes including itself if zero area."""
        valid_box = AvBox(0, 0, 10, 10)
        zero_width = AvBox(5, 5, 5, 15)
        zero_height = AvBox(5, 5, 15, 5)
        point = AvBox(20, 20, 20, 20)

        result = valid_box.combine_with(zero_width, zero_height, point)

        # Should only consider the valid box (self)
        assert result.xmin == 0.0
        assert result.ymin == 0.0
        assert result.xmax == 10.0
        assert result.ymax == 10.0

    def test_combine_with_self_zero_area_returns_self(self):
        """Test combine_with returns copy of self when self has zero area and no valid boxes provided."""
        zero_area_self = AvBox(5, 5, 5, 5)  # Point
        zero_width = AvBox(10, 10, 10, 20)
        zero_height = AvBox(10, 10, 20, 10)

        result = zero_area_self.combine_with(zero_width, zero_height)

        # Should return a copy of self
        assert result.xmin == zero_area_self.xmin
        assert result.ymin == zero_area_self.ymin
        assert result.xmax == zero_area_self.xmax
        assert result.ymax == zero_area_self.ymax
        # Ensure it's a copy, not the same object
        assert result is not zero_area_self

    def test_combine_with_preserves_self(self):
        """Test combine_with doesn't modify the original box."""
        box1 = AvBox(0, 0, 10, 10)
        box2 = AvBox(5, 5, 15, 15)

        original_box1 = (box1.xmin, box1.ymin, box1.xmax, box1.ymax)

        result = box1.combine_with(box2)

        # Original box should be unchanged
        assert (box1.xmin, box1.ymin, box1.xmax, box1.ymax) == original_box1
        # Result should be new box
        assert result is not box1

    def test_combine_with_no_arguments(self):
        """Test combine_with no arguments returns a copy of self."""
        box = AvBox(5, 10, 15, 20)

        result = box.combine_with()

        assert result.xmin == box.xmin
        assert result.ymin == box.ymin
        assert result.xmax == box.xmax
        assert result.ymax == box.ymax
        assert result is not box  # Should be a new instance

    def test_combine_with_empty_list(self):
        """Test combine_with empty list returns a copy of self."""
        box = AvBox(5, 10, 15, 20)

        result = box.combine_with([])

        assert result.xmin == box.xmin
        assert result.ymin == box.ymin
        assert result.xmax == box.xmax
        assert result.ymax == box.ymax
        assert result is not box  # Should be a new instance
