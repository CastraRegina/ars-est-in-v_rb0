"""Test module for ave.geom

The tests are run using pytest.
These tests ensure that all functions and interfaces in src/ave/geom.py
remain working correctly after changes and refactoring.
"""

import numpy as np
import pytest

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
        assert "xmin=10.0" in str_repr
        assert "ymin=20.0" in str_repr
        assert "xmax=30.0" in str_repr
        assert "ymax=40.0" in str_repr
        assert "width=20.0" in str_repr
        assert "height=20.0" in str_repr

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
