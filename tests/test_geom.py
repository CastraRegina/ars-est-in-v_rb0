"""Test module for ave.geom

The tests are run using pytest.
These tests ensure that all functions and interfaces in src/ave/geom.py
remain working correctly after changes and refactoring.
"""

import numpy as np
import pytest

from ave.geom import AvBox, BezierCurve, GeomMath

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
        new_points, _ = BezierCurve.polygonize_path(points, commands, 5)

        # Create bounding box from polygonized points
        x_coords = new_points[:, 0]
        y_coords = new_points[:, 1]
        box = AvBox(np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords))

        # Transform box
        transformed_box = box.transform_scale_translate(2.0, 10.0, 20.0)

        # Verify transformation
        assert transformed_box.width == box.width * 2.0
        assert transformed_box.height == box.height * 2.0
