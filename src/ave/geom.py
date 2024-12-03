"""Handling geometries"""

from __future__ import annotations


class AvBox:
    """
    Represents a rectangular box with coordinates and dimensions.

    Attributes:
        xmin (float): The minimum x-coordinate.
        ymin (float): The minimum y-coordinate.
        xmax (float): The maximum x-coordinate.
        ymax (float): The maximum y-coordinate.
    """

    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        """
        Initializes a new Box instance.

        Args:
            xmin (float): The minimum x-coordinate.
            ymin (float): The minimum y-coordinate.
            xmax (float): The maximum x-coordinate.
            ymax (float): The maximum y-coordinate.
        """

        self._xmin = min(xmin, xmax)
        self._xmax = max(xmin, xmax)
        self._ymin = min(ymin, ymax)
        self._ymax = max(ymin, ymax)

    @property
    def xmin(self) -> float:
        """float: The minimum x-coordinate."""

        return self._xmin

    @property
    def xmax(self) -> float:
        """float: The maximum x-coordinate."""

        return self._xmax

    @property
    def ymin(self) -> float:
        """float: The minimum y-coordinate."""

        return self._ymin

    @property
    def ymax(self) -> float:
        """float: The maximum y-coordinate."""

        return self._ymax

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


def main():
    """Main"""

    my_box = AvBox(xmin=10, ymin=40, xmax=30, ymax=70)

    print(f"Width : {my_box.width}")
    print(f"Height: {my_box.height}")
    print(f"Area  : {my_box.area}")
    print(f"xmin  : {my_box.xmin}, ymin: {my_box.ymin}, xmax: {my_box.xmax}, ymax: {my_box.ymax}")


if __name__ == "__main__":
    main()
