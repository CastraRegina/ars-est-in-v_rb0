"""Test module for AvImage class

This module tests all functionality of the AvImage class including:
- Basic properties and methods
- Region extraction
- Region statistics (mean, weighted mean, histogram)
- Edge cases and error handling
"""

import numpy as np
import pytest

from ave.image import AvImage


class TestAvImage:
    """Test class for AvImage functionality"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample 10x10 test image with known values"""
        # Create a gradient image from 0 to 255
        data = np.arange(100, dtype=np.uint8).reshape(10, 10)
        return AvImage(data)

    @pytest.fixture
    def uniform_image(self):
        """Create a uniform image with all pixels having value 128"""
        data = np.full((20, 30), 128, dtype=np.uint8)
        return AvImage(data)

    @pytest.fixture
    def binary_image(self):
        """Create a binary image with only 0 and 255 values"""
        data = np.zeros((10, 10), dtype=np.uint8)
        data[:5, :] = 255  # Top half white, bottom half black
        return AvImage(data)

    def test_init_with_valid_array(self):
        """Test initialization with valid numpy array"""
        data = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        img = AvImage(data)
        assert img.image is data

    def test_init_with_invalid_type(self):
        """Test initialization with non-numpy array"""
        with pytest.raises(TypeError, match="Image must be a NumPy array"):
            AvImage([[0, 255], [128, 64]])

    def test_init_with_invalid_dtype(self):
        """Test initialization with wrong dtype"""
        data = np.array([[0, 255], [128, 64]], dtype=np.int32)
        with pytest.raises(ValueError, match="Image array must be of type uint8"):
            AvImage(data)

    def test_init_with_invalid_dimensions(self):
        """Test initialization with wrong number of dimensions"""
        data = np.array([0, 255, 128, 64], dtype=np.uint8)
        with pytest.raises(ValueError, match="Image must be 2-dimensional"):
            AvImage(data)

    def test_init_with_zero_size(self):
        """Test initialization with zero-sized image"""
        data = np.array([], dtype=np.uint8).reshape(0, 10)
        with pytest.raises(ValueError, match="Image cannot have zero width or height"):
            AvImage(data)

    def test_properties(self, sample_image):
        """Test image properties"""
        assert sample_image.width_px == 10
        assert sample_image.height_px == 10
        assert sample_image.width_rel == 1.0
        assert sample_image.height_rel == 1.0
        assert sample_image.scale_rel == 0.1

    def test_get_region_px_basic(self, sample_image):
        """Test basic region extraction with pixel coordinates"""
        region = sample_image.get_region_px(2, 3, 7, 8)
        expected = sample_image.image[3:8, 2:7]
        np.testing.assert_array_equal(region, expected)

    def test_get_region_px_out_of_bounds(self, sample_image):
        """Test region extraction with out-of-bounds coordinates"""
        # Should clip to image bounds
        region = sample_image.get_region_px(-5, -5, 15, 15)
        assert region.shape == (10, 10)
        np.testing.assert_array_equal(region, sample_image.image)

    def test_get_region_px_swapped_coordinates(self, sample_image):
        """Test region extraction with swapped coordinates"""
        # Should automatically swap if x2 < x1 or y2 < y1
        region1 = sample_image.get_region_px(2, 3, 7, 8)
        region2 = sample_image.get_region_px(7, 8, 2, 3)
        np.testing.assert_array_equal(region1, region2)

    def test_get_region_px_single_pixel(self, sample_image):
        """Test extraction of single pixel region"""
        region = sample_image.get_region_px(5, 5, 6, 6)
        assert region.shape == (1, 1)
        assert region[0, 0] == sample_image.image[5, 5]

    def test_get_region_rel_basic(self, sample_image):
        """Test basic region extraction with relative coordinates"""
        region = sample_image.get_region_rel(0.2, 0.3, 0.7, 0.8)
        # Convert to pixel coordinates and compare
        px1, py1 = int(round(0.2 / 0.1)), int(round((1.0 - 0.8) / 0.1))
        px2, py2 = int(round(0.7 / 0.1)), int(round((1.0 - 0.3) / 0.1))
        expected = sample_image.get_region_px(px1, py1, px2, py2)
        np.testing.assert_array_equal(region, expected)

    def test_get_region_mean_px(self, sample_image):
        """Test mean calculation with pixel coordinates"""
        mean = sample_image.get_region_mean_px(0, 0, 10, 10)
        expected = float(sample_image.image.mean())
        assert mean == expected

    def test_get_region_mean_px_uniform(self, uniform_image):
        """Test mean calculation on uniform image"""
        mean = uniform_image.get_region_mean_px(5, 5, 15, 15)
        assert mean == 128.0

    def test_get_region_mean_rel(self, sample_image):
        """Test mean calculation with relative coordinates"""
        mean = sample_image.get_region_mean_rel(0.0, 0.0, 1.0, 1.0)
        expected = float(sample_image.image.mean())
        assert mean == expected

    def test_get_region_weighted_mean_px(self):
        """Test weighted mean calculation with pixel coordinates"""
        # Test on uniform image where weighted mean should equal regular mean
        uniform_data = np.full((10, 10), 128, dtype=np.uint8)
        uniform_img = AvImage(uniform_data)
        wmean = uniform_img.get_region_weighted_mean_px(0, 0, 10, 10)
        mean = uniform_img.get_region_mean_px(0, 0, 10, 10)
        assert abs(wmean - mean) < 0.01

    def test_get_region_weighted_mean_rel(self, sample_image):
        """Test weighted mean calculation with relative coordinates"""
        wmean = sample_image.get_region_weighted_mean_rel(0.0, 0.0, 1.0, 1.0)
        mean = sample_image.get_region_mean_rel(0.0, 0.0, 1.0, 1.0)
        assert abs(wmean - mean) < 0.01

    def test_get_region_histogram_px_basic(self, sample_image):
        """Test histogram calculation with pixel coordinates"""
        hist = sample_image.get_region_histogram_px(0, 0, 10, 10, bins=10)
        assert len(hist) == 10
        assert abs(sum(hist) - 1.0) < 1e-10  # Should sum to 1.0
        assert all(0 <= h <= 1 for h in hist)  # All values should be between 0 and 1

    def test_get_region_histogram_px_uniform(self, uniform_image):
        """Test histogram on uniform image"""
        hist = uniform_image.get_region_histogram_px(0, 0, 20, 30, bins=5)
        assert len(hist) == 5
        assert abs(sum(hist) - 1.0) < 1e-10
        # All pixels have value 128, which should be in bin 2 (for 5 bins)
        assert hist[2] == 1.0
        assert all(h == 0 for i, h in enumerate(hist) if i != 2)

    def test_get_region_histogram_px_binary(self, binary_image):
        """Test histogram on binary image"""
        hist = binary_image.get_region_histogram_px(0, 0, 10, 10, bins=2)
        assert len(hist) == 2
        assert abs(sum(hist) - 1.0) < 1e-10
        # Should have 50% in bin 0 (values 0-127) and 50% in bin 1 (values 128-255)
        assert hist[0] == 0.5
        assert hist[1] == 0.5

    def test_get_region_histogram_px_single_bin(self, sample_image):
        """Test histogram with single bin"""
        hist = sample_image.get_region_histogram_px(0, 0, 10, 10, bins=1)
        assert len(hist) == 1
        assert hist[0] == 1.0  # All pixels in single bin

    def test_get_region_histogram_px_many_bins(self, sample_image):
        """Test histogram with many bins"""
        hist = sample_image.get_region_histogram_px(0, 0, 10, 10, bins=256)
        assert len(hist) == 256
        assert abs(sum(hist) - 1.0) < 1e-10

    def test_get_region_histogram_rel(self, sample_image):
        """Test histogram calculation with relative coordinates"""
        hist = sample_image.get_region_histogram_rel(0.0, 0.0, 1.0, 1.0, bins=10)
        assert len(hist) == 10
        assert abs(sum(hist) - 1.0) < 1e-10

    def test_get_region_histogram_small_region(self, sample_image):
        """Test histogram on small region"""
        hist = sample_image.get_region_histogram_px(5, 5, 6, 6, bins=5)
        assert len(hist) == 5
        assert abs(sum(hist) - 1.0) < 1e-10
        # Only one pixel, so it should be entirely in one bin
        assert sum(h == 1.0 for h in hist) == 1
        assert sum(h == 0.0 for h in hist) == 4

    def test_get_region_histogram_edge_values(self):
        """Test histogram with edge values (0 and 255)"""
        data = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        img = AvImage(data)
        hist = img.get_region_histogram_px(0, 0, 2, 2, bins=2)
        assert hist[0] == 0.5  # Values 0-127
        assert hist[1] == 0.5  # Values 128-255

    def test_from_file(self, tmp_path):
        """Test loading image from file"""
        # Create a test image
        data = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        img = AvImage(data)

        # Save to file
        filename = tmp_path / "test.png"
        img.to_file(filename)

        # Load from file
        loaded_img = AvImage.from_file(filename)
        np.testing.assert_array_equal(loaded_img.image, data)

    def test_to_file(self, tmp_path):
        """Test saving image to file"""
        data = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        img = AvImage(data)

        filename = tmp_path / "test_output.png"
        img.to_file(filename)

        assert filename.exists()
        assert filename.stat().st_size > 0

    def test_region_view_shares_memory(self, sample_image):
        """Test that region views share memory with original image"""
        region = sample_image.get_region_px(2, 3, 7, 8)
        # Check that they share memory base
        assert np.shares_memory(region, sample_image.image)
        # The view should be read-only as documented

    def test_empty_region(self, sample_image):
        """Test behavior with empty region"""
        # Region with zero area should still work due to coordinate adjustment
        region = sample_image.get_region_px(5, 5, 5, 5)
        assert region.size > 0  # Should be adjusted to at least 1x1

    def test_large_number_of_bins(self, sample_image):
        """Test histogram with more bins than unique values"""
        hist = sample_image.get_region_histogram_px(0, 0, 10, 10, bins=1000)
        assert len(hist) == 1000
        assert abs(sum(hist) - 1.0) < 1e-10
        # Most bins should be empty
        assert sum(h > 0 for h in hist) <= 100  # Only 100 unique values in test image

    def test_histogram_precision(self, uniform_image):
        """Test histogram precision with uniform values"""
        hist = uniform_image.get_region_histogram_px(0, 0, 20, 30, bins=4)
        # Value 128 should be exactly in the middle
        assert hist[1] == 0.0  # 0-63
        assert hist[2] == 1.0  # 64-127
        assert hist[3] == 0.0  # 128-191
        assert hist[4] == 0.0 if len(hist) > 4 else True  # 192-255 (if 5 bins)
