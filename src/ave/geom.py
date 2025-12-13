"""Handling geometries"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ave.common import AvGlyphCmds
from ave.geom_bezier import BezierCurve


###############################################################################
# GeomMath
###############################################################################
class GeomMath:
    """Class to provide various static methods related to geometry handling."""

    @staticmethod
    def transform_point(
        affine_trafo: Sequence[Union[int, float]], point: Sequence[Union[int, float]]
    ) -> Tuple[float, float]:
        """
        Perform an affine transformation on the given 2D point.

        The given _affine_trafo_ is a list of 6 floats, performing an affine transformation.
        The transformation is defined as:
            | x' | = | a00 a01 b0 |   | x |
            | y' | = | a10 a11 b1 | * | y |
            | 1  | = |  0   0  1  |   | 1 |
        with
            affine_trafo = [a00, a01, a10, a11, b0, b1]
        See also shapely - Affine Transformations

        Args:
            affine_trafo (Tuple/List[float]): Affine transformation - [a00, a01, a10, a11, b0, b1]
            point (Tuple/List[float]): 2D point - (x, y)

        Returns:
            Tuple[float, float]: the transformed point
        """
        x_new = float(affine_trafo[0] * point[0] + affine_trafo[1] * point[1] + affine_trafo[4])
        y_new = float(affine_trafo[2] * point[0] + affine_trafo[3] * point[1] + affine_trafo[5])
        return (x_new, y_new)


###############################################################################
# AvBox
###############################################################################
@dataclass
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
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

        # Normalize coordinates to ensure xmin ≤ xmax and ymin ≤ ymax
        if self._xmin > self._xmax:
            self._xmin, self._xmax = self._xmax, self._xmin
        if self._ymin > self._ymax:
            self._ymin, self._ymax = self._ymax, self._ymin

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
        """float: The area of the box."""

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

    @classmethod
    def from_dict(cls, data: dict) -> AvBox:
        """Create an AvBox instance from a dictionary."""
        return cls(
            xmin=data.get("xmin", 0.0),
            ymin=data.get("ymin", 0.0),
            xmax=data.get("xmax", 0.0),
            ymax=data.get("ymax", 0.0),
        )

    def __str__(self):
        """Returns a string representation of the AvBox instance."""
        return (
            f"AvBox(xmin={self.xmin}, ymin={self.ymin}, "
            f"xmax={self.xmax}, ymax={self.ymax}, "
            f"width={self.width}, height={self.height})"
        )

    def to_dict(self) -> dict:
        """Convert the AvBox instance to a dictionary."""
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
        }


def main():
    """Main"""


if __name__ == "__main__":
    main()
