"""Handling geometries for SVG processing and 2D transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ave.common import sgn_sci

###############################################################################
# GeomMath
###############################################################################


class GeomMath:
    """Class to provide various static methods related to geometry handling."""

    @staticmethod
    def transform_point(
        affine_trafo: Sequence[Union[int, float]], point: Sequence[Union[int, float]]
    ) -> Tuple[float, float]:
        """Perform an affine transformation on the given 2D point.

        The given affine_trafo is a list of 6 floats, performing an affine transformation.
        The transformation is defined as:
            | x' | = | a00 a01 b0 |   | x |
            | y' | = | a10 a11 b1 | * | y |
            | 1  | = |  0   0  1  |   | 1 |
        with
            affine_trafo = [a00, a01, a10, a11, b0, b1]
        See also shapely - Affine Transformations

        Args:
            affine_trafo: List of 6 floats defining the affine transformation
            point: 2D point as sequence of (x, y) coordinates

        Returns:
            Transformed point as (x, y) tuple

        Raises:
            ValueError: If affine_trafo does not have exactly 6 elements
        """
        x_new = float(affine_trafo[0] * point[0] + affine_trafo[1] * point[1] + affine_trafo[4])
        y_new = float(affine_trafo[2] * point[0] + affine_trafo[3] * point[1] + affine_trafo[5])
        return (x_new, y_new)


###############################################################################
# AvPolygon
###############################################################################


class AvPolygon:
    """Class to provide various static methods related to simple single polygons."""

    @staticmethod
    def area(points: NDArray[np.float64]) -> float:
        """Calculate area using shoelace formula for a single polygon.

        Note: This method always returns 0 or a positive value, regardless of
        the polygon's winding direction (clockwise or counter-clockwise).

        Args:
            points: Array of 2D points defining the polygon vertices

        Returns:
            float: The absolute area of the polygon (always >= 0.0)
        """
        if points.shape[0] < 3:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        cross_sum = cross.sum()
        if np.isclose(cross_sum, 0.0):
            return 0.0
        return float(0.5 * abs(cross_sum))

    @staticmethod
    def centroid(points: NDArray[np.float64]) -> Tuple[float, float]:
        """Calculate centroid for a single polygon."""
        if points.shape[0] == 0:
            return (0.0, 0.0)
        if points.shape[0] < 3:
            x_mean = float(points[:, 0].mean())
            y_mean = float(points[:, 1].mean())
            return (x_mean, y_mean)
        x = points[:, 0]
        y = points[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        cross_sum = cross.sum()
        if np.isclose(cross_sum, 0.0):
            x_mean = float(x.mean())
            y_mean = float(y.mean())
            return (x_mean, y_mean)
        factor = 1.0 / (3.0 * cross_sum)
        cx = float(((x + x_next) * cross).sum() * factor)
        cy = float(((y + y_next) * cross).sum() * factor)
        return (cx, cy)

    @staticmethod
    def is_ccw(points: NDArray[np.float64]) -> bool:
        """Calculate if single polygon vertices are ordered counter-clockwise."""
        if points.shape[0] < 3:
            return False
        x = points[:, 0]
        y = points[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        return bool(cross.sum() > 0.0)

    @staticmethod
    def ray_casting_single(points: NDArray[np.float64], point: Tuple[float, float]) -> bool:
        """Check if a point lies inside a single polygon using ray casting algorithm.

        Args:
            points: Array of polygon vertices (shape: n_points, 3 or n_points, 2)
            point: Point to test as (x, y) tuple

        Returns:
            bool: True if point is inside the polygon, False otherwise

        Raises:
            ValueError: If point is not a tuple/list of 2 numeric values
        """
        # Input validation
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError("Point must be a tuple or list of 2 numeric values")

        n = points.shape[0]
        if n == 0:
            return False

        x, y = float(point[0]), float(point[1])
        inside = False
        j = n - 1

        # Optimized ray casting with local variables for speed
        pts_x = points[:, 0]
        pts_y = points[:, 1]

        for i in range(n):
            xi, yi = float(pts_x[i]), float(pts_y[i])
            xj, yj = float(pts_x[j]), float(pts_y[j])

            # Check if ray intersects edge
            if (yi > y) != (yj > y):
                # Calculate intersection point
                dy = yj - yi
                if dy != 0:  # Avoid division by zero
                    x_intersect = xi + (y - yi) * (xj - xi) / dy
                    if x < x_intersect:
                        inside = not inside
            j = i
        return inside

    @staticmethod
    def interior_point_scanlines(
        points: NDArray[np.float64], samples: int = 9, epsilon: float = 1e-9
    ) -> Optional[Tuple[float, float]]:
        """Find an interior point in a single polygon using horizontal scanlines.

        This method samples horizontal lines across the polygon's y-range,
        finds intersections with polygon edges, and returns the midpoint of
        the widest interior interval. This is useful for finding a representative
        point inside potentially concave polygons where the centroid might lie
        outside the filled region.

        Args:
            points: Array of polygon vertices (shape: n_points, 3 or n_points, 2)
            samples: Number of horizontal scanlines to try between ymin and ymax
            epsilon: Small relative offset applied to scanline y values to avoid
                pathological cases where the scanline hits vertices exactly

        Returns:
            Optional[Tuple[float, float]]: An interior point if one can be found,
                None if the polygon is degenerate or no interior point is found

        Note:
            This method works on single polygons only. For multi-polygon or
            polygon-with-holes scenarios, use higher-level path methods.
        """
        if points.shape[0] < 3:
            return None

        y_min = float(points[:, 1].min())
        y_max = float(points[:, 1].max())
        height = y_max - y_min
        if np.isclose(height, 0.0):
            return None

        n = int(points.shape[0])
        n_samples = max(int(samples), 1)
        y_tol = abs(epsilon) * height

        for k in range(n_samples):
            y = y_min + (k + 0.5) / n_samples * height + epsilon * height

            xs: List[float] = []
            j = n - 1
            for i in range(n):
                xi, yi = float(points[i, 0]), float(points[i, 1])
                xj, yj = float(points[j, 0]), float(points[j, 1])

                if (yi > y) != (yj > y):
                    dy = yj - yi
                    if abs(dy) >= y_tol:
                        x_int = xi + (y - yi) * (xj - xi) / dy
                        xs.append(float(x_int))

                j = i

            xs.sort()

            if len(xs) % 2 != 0:
                continue

            best: Optional[Tuple[float, float]] = None
            best_w = -1.0
            for i in range(0, len(xs) - 1, 2):
                w = xs[i + 1] - xs[i]
                if w > best_w:
                    best_w = w
                    best = ((xs[i] + xs[i + 1]) * 0.5, y)

            if best is not None:
                # Verify the point is actually inside using ray casting
                if AvPolygon.ray_casting_single(points, best):
                    return (float(best[0]), float(best[1]))

        return None


###############################################################################
# AvBox
###############################################################################


@dataclass(frozen=True)
class AvBox:
    """
    Represents a rectangular box with coordinates and dimensions.

    Attributes:
        xmin (float): The minimum x-coordinate.
        ymin (float): The minimum y-coordinate.
        xmax (float): The maximum x-coordinate.
        ymax (float): The maximum y-coordinate.
    """

    _xmin: float
    _ymin: float
    _xmax: float
    _ymax: float

    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        """Initialize AvBox with coordinates.

        Args:
            xmin: The minimum x-coordinate
            ymin: The minimum y-coordinate
            xmax: The maximum x-coordinate
            ymax: The maximum y-coordinate
        """
        x0, x1 = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        y0, y1 = (ymin, ymax) if ymin <= ymax else (ymax, ymin)

        object.__setattr__(self, "_xmin", x0)
        object.__setattr__(self, "_ymin", y0)
        object.__setattr__(self, "_xmax", x1)
        object.__setattr__(self, "_ymax", y1)

    @property
    def xmin(self) -> float:
        """float: The minimum x-coordinate."""
        return self._xmin

    @property
    def ymin(self) -> float:
        """float: The minimum y-coordinate."""
        return self._ymin

    @property
    def xmax(self) -> float:
        """float: The maximum x-coordinate."""
        return self._xmax

    @property
    def ymax(self) -> float:
        """float: The maximum y-coordinate."""
        return self._ymax

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """The extent of the box as Tuple (xmin, ymin, xmax, ymax)."""
        return self._xmin, self._ymin, self._xmax, self._ymax

    @property
    def width(self) -> float:
        """float: The width of the box (difference between xmax and xmin)."""

        return self._xmax - self._xmin

    @property
    def height(self) -> float:
        """float: The height of the box (difference between ymax and ymin)."""

        return self._ymax - self._ymin

    @property
    def area(self) -> float:
        """float: The area of the box (always >= 0.0)."""

        return self.width * self.height

    @property
    def centroid(self) -> Tuple[float, float]:
        """
        The centroid of the box.

        Returns:
            Tuple[float, float]: The coordinates of the centroid as (x, y)
        """
        return (self._xmin + self._xmax) / 2, (self._ymin + self._ymax) / 2

    def transform_affine(self, affine_trafo: Sequence[Union[int, float]]) -> AvBox:
        """
        Transform the AvBox using the given affine transformation [a00, a01, a10, a11, b0, b1].

        Args:
            affine_trafo (List[float]): Affine transformation [a00, a01, a10, a11, b0, b1]

        Returns:
            AvBox: The transformed box
        """
        (xmin, ymin, xmax, ymax) = self.extent
        (x0, y0) = GeomMath.transform_point(affine_trafo, (xmin, ymin))
        (x1, y1) = GeomMath.transform_point(affine_trafo, (xmax, ymax))
        return AvBox(xmin=x0, ymin=y0, xmax=x1, ymax=y1)

    def transform_scale_translate(self, scale_factor: float, translate_x: float, translate_y: float) -> AvBox:
        """
        Transform the AvBox using the given scale and translation.

        Args:
            scale_factor (float): The scale factor.
            translate_x (float): The translation in x-direction.
            translate_y (float): The translation in y-direction.

        Returns:
            AvBox: The transformed box
        """
        return self.transform_affine((scale_factor, 0, 0, scale_factor, translate_x, translate_y))

    def overlaps(self, other: AvBox) -> bool:
        """Check if this box overlaps with another box.

        Args:
            other: Another AvBox to check overlap with

        Returns:
            True if the boxes overlap, False otherwise
        """
        # Check if bounding boxes intersect
        return not (
            self.xmax < other.xmin or other.xmax < self.xmin or self.ymax < other.ymin or other.ymax < self.ymin
        )

    @classmethod
    def from_dict(cls, data: dict) -> AvBox:
        """Create an AvBox instance from a dictionary."""
        return cls(
            xmin=data.get("xmin", 0.0),
            ymin=data.get("ymin", 0.0),
            xmax=data.get("xmax", 0.0),
            ymax=data.get("ymax", 0.0),
        )

    def to_dict(self) -> dict:
        """Convert the AvBox instance to a dictionary."""
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
        }

    def __str__(self):
        """Returns a string representation of the AvBox instance."""
        return (
            f"AvBox(xmin={sgn_sci(self.xmin)}, "
            f"ymin={sgn_sci(self.ymin)}, "
            f"xmax={sgn_sci(self.xmax)}, "
            f"ymax={sgn_sci(self.ymax)}, "
            f"width={sgn_sci(self.width, always_positive=True)}, "
            f"height={sgn_sci(self.height, always_positive=True)}, "
            f"area={sgn_sci(self.area, always_positive=True)}, "
            f"centroid=({sgn_sci(self.centroid[0])}, {sgn_sci(self.centroid[1])}))"
        )

    @staticmethod
    def combine(*boxes: Union[AvBox, Sequence[AvBox]]) -> AvBox:
        """Create a new AvBox that is the overall sum of the given AvBoxes.

        The resulting box has xmin as the minimum of all xmin values and xmax as
        the maximum of all xmax values (only from boxes with non-zero width and height).
        The same applies to the y-direction.

        If all boxes have zero width and/or height, returns a copy of the first AvBox.

        Args:
            *boxes: One or more AvBoxes, or a single iterable/list of AvBoxes

        Returns:
            AvBox: A new box representing the overall bounds

        Raises:
            ValueError: If no AvBoxes are provided
        """
        # Handle the case where a single iterable is passed
        if len(boxes) == 1 and not isinstance(boxes[0], AvBox):
            # Assume it's an iterable of AvBoxes
            boxes_iterable = boxes[0]
        else:
            boxes_iterable = boxes

        # Filter out boxes with zero width or height
        valid_boxes = [box for box in boxes_iterable if isinstance(box, AvBox) and box.width != 0 and box.height != 0]

        # If no boxes with non-zero area, return a copy of the first AvBox
        if not valid_boxes:
            # Find the first AvBox in the original iterable
            first_box = None
            for box in boxes_iterable:
                if isinstance(box, AvBox):
                    first_box = box
                    break

            if first_box is None:
                raise ValueError("At least one AvBox must be provided")

            # Return a copy of the first box (which has zero area)
            return AvBox(xmin=first_box.xmin, ymin=first_box.ymin, xmax=first_box.xmax, ymax=first_box.ymax)

        # Find the overall bounds
        xmin = min(box.xmin for box in valid_boxes)
        xmax = max(box.xmax for box in valid_boxes)
        ymin = min(box.ymin for box in valid_boxes)
        ymax = max(box.ymax for box in valid_boxes)

        return AvBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    def combine_with(self, *boxes: Union[AvBox, Sequence[AvBox]]) -> AvBox:
        """Combine this AvBox with other AvBoxes to create a new overall bounding box.

        This is an instance method that calls the static combine() method,
        including this box along with the provided boxes.

        If all boxes (including this one) have zero width and/or height,
        returns a copy of self.

        Args:
            *boxes: One or more AvBoxes, or a single iterable/list of AvBoxes to combine with this box

        Returns:
            AvBox: A new box representing the overall bounds of this box and all provided boxes

        Raises:
            ValueError: If no AvBoxes are provided
        """
        # Prepend this box to the list of boxes to combine
        if len(boxes) == 1 and not isinstance(boxes[0], AvBox):
            # If boxes is a single iterable, create a new list with this box prepended
            all_boxes = [self] + list(boxes[0])
        else:
            # If boxes are varargs, create a tuple with this box prepended
            all_boxes = (self,) + boxes

        return AvBox.combine(*all_boxes)


###############################################################################
# main
###############################################################################


def main():
    """Main"""
    example_box = AvBox(xmin=10.0, ymin=20.0, xmax=40.0, ymax=60.0)
    print(example_box)


if __name__ == "__main__":
    main()
