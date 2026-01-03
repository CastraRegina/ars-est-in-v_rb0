"""Image representation module for handling grayscale images using NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import PIL.Image


@dataclass
class AvImage:
    """An image representation storing grayscale data as NumPy array.

    The image data is stored as Pillow Type L (8-bit pixels, grayscale),
    where black=0 and white=255. The internal representation uses NumPy
    arrays for efficient processing.

    Two coordinate systems are supported:

    1. Pixel Coordinate System (_px suffix):
        - Origin (0, 0) is at the top-left corner
        - X increases from left to right
        - Y increases from top to bottom
        - Valid pixel coordinates: X in [0, width-1], Y in [0, height-1]
        - Example: For a 100x200 image, pixels range from (0,0) to (99,199)

    2. Relative Coordinate System (_rel suffix):
        - Origin (0, 0) is at the bottom-left corner
        - X increases from left to right
        - Y increases from bottom to top
        - X normalized to [0, 1] where 1 = width_px
        - Y normalized to [0, height_px*scale] where scale = 1/width_px
        - Uses float parameters for precise positioning

    Region extraction uses zero-copy NumPy views for maximum performance.
    Coordinates are automatically clipped to bounds and swapped if needed.
    """

    _image: np.ndarray[np.uint8, :]

    def __init__(self, image: np.ndarray[np.uint8, :]):
        """Initialize with a grayscale image as NumPy array.

        Args:
            image: NumPy array of type uint8 representing grayscale image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array")

        if image.dtype != np.uint8:
            raise ValueError("Image array must be of type uint8")

        if len(image.shape) != 2:
            raise ValueError("Image must be 2-dimensional (grayscale)")

        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Image cannot have zero width or height")

        self._image = image

    @property
    def image(self) -> np.ndarray:
        """Get the image as NumPy array.

        Returns:
            NumPy array of shape (height, width) with uint8 values
        """
        return self._image

    @property
    def width_px(self) -> int:
        """Get the image width in pixels.

        Returns:
            Width of the image in pixels
        """
        return self._image.shape[1]

    @property
    def height_px(self) -> int:
        """Get the image height in pixels.

        Returns:
            Height of the image in pixels
        """
        return self._image.shape[0]

    @property
    def width_rel(self) -> float:
        """Get the image width in relative coordinates.

        Returns:
            Width as 1.0 (normalized)
        """
        return 1.0

    @property
    def height_rel(self) -> float:
        """Get the image height in relative coordinates.

        Returns:
            Height normalized to width (height_px / width_px)
        """
        return self.height_px / self.width_px

    @property
    def scale_rel(self) -> float:
        """Get the scale factor for relative coordinates.

        Returns:
            Scale factor (1.0 / width_px)
        """
        return 1.0 / self.width_px

    def get_region_px(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray[np.uint8, :]:
        """Get a read-only view of a rectangular region of the image.

        This method returns a NumPy array view, which is zero-copy and
        extremely efficient. The view shares memory with the original
        image, so modifications to the view will affect the original.

        Coordinates are automatically clipped to image bounds and swapped
        if provided in wrong order for convenience.

        Coordinate System:
            - Origin (0, 0) is at the top-left corner
            - X increases from left to right
            - Y increases from top to bottom
            - Valid pixel coordinates: X in [0, width-1], Y in [0, height-1]
            - Example: For a 100x200 image, pixels range from (0,0) to (99,199)

        Examples:
            # Get the entire image:
            region = img.get_region_px(0, 0, img.width_px, img.height_px)

            # Get a single pixel at (10, 20):
            pixel = img.get_region_px(10, 20, 11, 21)

            # Get a 50x50 region starting at top-left:
            region = img.get_region_px(0, 0, 50, 50)

        Args:
            x1: Left coordinate (inclusive, 0-based)
            y1: Top coordinate (inclusive, 0-based)
            x2: Right coordinate (exclusive, 0-based)
            y2: Bottom coordinate (exclusive, 0-based)

        Returns:
            Read-only view of the specified region as NumPy array

        Note:
            This function never raises errors for invalid coordinates. Coordinates are
            automatically clipped to image bounds, swapped if needed, and adjusted to
            ensure at least one pixel is returned.
        """
        # Ensure x1 <= x2 and y1 <= y2 by swapping if needed
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Clip coordinates to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.width_px, x2)
        y2 = min(self.height_px, y2)

        # Ensure at least one pixel is returned (simplified logic)
        if x1 >= x2:
            x1 = max(0, x1 - 1)
            x2 = min(x1 + 1, self.width_px)

        if y1 >= y2:
            y1 = max(0, y1 - 1)
            y2 = min(y1 + 1, self.height_px)

        # Return a view of the region (zero-copy operation)
        region = self._image[y1:y2, x1:x2]

        # Make the array read-only to prevent accidental modification
        region.flags.writeable = False

        return region

    def get_region_rel(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray[np.uint8, :]:
        """Get a read-only view of a rectangular region using relative coordinates.

        This method works with the relative coordinate system where:
        - Origin (0, 0) is at the bottom-left corner
        - X increases from left to right, normalized to [0, 1]
        - Y increases from bottom to top, normalized to [0, height_px*scale]

        The method converts relative coordinates to pixel coordinates
        and delegates to get_region_px() for actual region extraction.

        Args:
            x1: Left coordinate in relative units (float, 0 to 1)
            y1: Bottom coordinate in relative units (float, 0 to height_px*scale)
            x2: Right coordinate in relative units (float, 0 to 1)
            y2: Top coordinate in relative units (float, 0 to height_px*scale)

        Returns:
            Read-only view of the specified region as NumPy array

        Note:
            This function never raises errors for invalid coordinates. Coordinates are
            converted to pixel space and then processed by get_region_px(), which
            automatically clips to bounds and ensures at least one pixel is returned.
        """
        # Convert relative X coordinates to pixel coordinates
        px1 = int(round(x1 / self.scale_rel))
        px2 = int(round(x2 / self.scale_rel))

        # Convert relative Y coordinates to pixel coordinates
        # Y is inverted because relative system starts from bottom
        py1 = int(round((self.height_rel - y1) / self.scale_rel))
        py2 = int(round((self.height_rel - y2) / self.scale_rel))

        # Delegate to pixel-based method
        return self.get_region_px(px1, py1, px2, py2)

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> AvImage:
        """Load an image from file and convert to grayscale.

        Args:
            filename: Path to the image file

        Returns:
            AvImage instance with grayscale data
        """
        # Load image using Pillow
        with PIL.Image.open(filename) as img:
            # Convert to grayscale (Type L) if not already
            if img.mode != "L":
                img = img.convert("L")

            # Convert to NumPy array
            image_array = np.array(img, dtype=np.uint8)

        return cls(image_array)

    def to_file(self, filename: Union[str, Path]) -> None:
        """Save the image to file.

        Args:
            filename: Path where to save the image file
        """
        # Convert NumPy array back to Pillow Image
        img = PIL.Image.fromarray(self._image, mode="L")

        # Save to file
        img.save(filename)
