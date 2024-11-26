"""Provide various helper functionalities."""

from __future__ import annotations

import math
from typing import Any, List, Tuple, Union, cast

import numpy
import svgwrite
import svgwrite.elementfactory

import av.consts


class HelperTypeHinting:
    """Helper-class to check for certain types and do type casting"""

    @staticmethod
    def check_list_of_ndarrays(variable: Union[List[numpy.ndarray], Any]) -> bool:
        """
        Check if a given variable is of type list containing elements of type numpy.ndarray.

        Parameters:
            variable (Union[List[numpy.ndarray], Any]): The variable to be checked.

        Returns:
            bool: True if the variable is of type list[numpy.ndarray], False otherwise.
        """
        if isinstance(variable, list):
            if all(isinstance(item, numpy.ndarray) for item in variable):
                return True
        return False

    @staticmethod
    def ensure_list_of_ndarrays(variable: Union[List[numpy.ndarray], Any]) -> List[numpy.ndarray]:
        """
        Ensure that a given variable is of type list containing elements of type numpy.ndarray.
        If the variable is not of this type, return the variable without the type and print an error message.

        Parameters:
            variable (Union[List[numpy.ndarray], Any]): The variable to be checked.

        Returns:
            Union[List[numpy.ndarray], Any]: The variable with type List[numpy.ndarray] if it passes the check,
                                            otherwise return the variable without the type.
        """
        if HelperTypeHinting.check_list_of_ndarrays(variable):
            return cast(list[numpy.ndarray], variable)
        else:
            print("Error: The variable is not of type List[numpy.ndarray]. Nevertheless going on...")
            return cast(list[numpy.ndarray], variable)


class HelperSvg:
    """Helper-class to provide various functionalities to handle SVG topics."""

    @staticmethod
    def rect_to_path(rect: Tuple[float, float, float, float]) -> str:
        """Provide a rectangle as SVG path (polygon)

        Args:
            rect (Tuple[float, float, float, float]): (x_pos, y_pos, width, height)

        Returns:
            str: SVG string representing the rectangle as path
        """
        (x_pos, y_pos, width, height) = rect
        (x00, y00) = (x_pos, y_pos)
        (x01, y01) = (x_pos + width, y_pos)
        (x11, y11) = (x_pos + width, y_pos + height)
        (x10, y10) = (x_pos, y_pos + height)
        ret_path = f"M{x00:g} {y00:g} " + f"L{x01:g} {y01:g} " + f"L{x11:g} {y11:g} " + f"L{x10:g} {y10:g} Z"
        return ret_path

    @staticmethod
    def circle_to_path(
        x_pos: float,
        y_pos: float,
        radius: float,
        angle_degree: float = av.consts.POLYGONIZE_ANGLE_MAX_DEG,
    ) -> str:
        """Provide a circle as SVG path (polygon)

        Args:
            x_pos (float): _description_
            y_pos (float): _description_
            radius (float): _description_
            angle_degree (float, optional): _description_.
                Defaults to av.consts.POLYGONIZE_ANGLE_MAX_DEG.

        Returns:
            str: SVG string representing the circle as polygon-path
        """
        ret_path = ""
        num_points = math.ceil(360 / angle_degree)
        for i in range(num_points):
            angle_rad = 2 * math.pi * i / num_points
            x_circ = x_pos + radius * math.sin(angle_rad)
            y_circ = y_pos + radius * math.cos(angle_rad)
            if i <= 0:
                ret_path += f"M{x_circ:g} {y_circ:g} "
            else:
                ret_path += f"L{x_circ:g} {y_circ:g} "
        ret_path += "Z"
        return ret_path

    @staticmethod
    def svg_rect(
        dwg: svgwrite.Drawing,
        rect: Tuple[float, float, float, float],
        stroke: str,
        stroke_width: float,
        **svg_properties,
    ) -> svgwrite.elementfactory.ElementBuilder:
        """Create a SVG-element which can be added to a SVG-layer

        Args:
            dwg (svgwrite.Drawing): dwg.rect()-function is called to create the rect
            rect (Tuple[float, float, float, float]): (x_pos, y_pos, width, height)
            stroke (str): color of the stroke
            stroke_width (float): width of the stroke
            svg_properties: further SVG properties

        Returns:
            svgwrite.elementfactory.ElementBuilder: the rect which can be added to a SVG-layer
        """
        (x_pos, y_pos, width, height) = rect
        rect_properties = {
            "insert": (x_pos, y_pos),
            "size": (width, height),
            "stroke": stroke,  # color
            "stroke_width": stroke_width,
            "fill": "none",
        }
        rect_properties.update(svg_properties)
        return dwg.rect(**rect_properties)


class AvBox:
    """
    Represents a rectangular box with coordinates and dimensions.

    Attributes:
        xmin (float): The minimum x-coordinate.
        xmax (float): The maximum x-coordinate.
        ymin (float): The minimum y-coordinate.
        ymax (float): The maximum y-coordinate.
    """

    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float):
        """
        Initializes a new Box instance.

        Args:
            xmin (float): The minimum x-coordinate.
            xmax (float): The maximum x-coordinate.
            ymin (float): The minimum y-coordinate.
            ymax (float): The maximum y-coordinate.
        """

        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

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

    def svg_rect(
        self,
        dwg: svgwrite.Drawing,
        stroke: str,
        stroke_width: float,
        **svg_properties,
    ) -> svgwrite.elementfactory.ElementBuilder:
        """Create a SVG-element which can be added to a SVG-layer

        Args:y
            dwg (svgwrite.Drawing): dwg.rect()-function is called to create the rect
            stroke (str): color of the stroke
            stroke_width (float): width of the stroke
            svg_properties: further SVG properties

        Returns:
            svgwrite.elementfactory.ElementBuilder: the rect which can be added to a SVG-layer
        """
        x_pos = self.xmin
        y_pos = self.ymin  # TODO: check the direction top/bottom as SVG-coordinate-system is from top to bottom
        width = self.width
        height = self.height
        rect = (x_pos, y_pos, width, height)

        svg_rect = HelperSvg.svg_rect(dwg, rect, stroke, stroke_width, **svg_properties)
        return svg_rect


def main():
    """Main"""

    my_box = AvBox(xmin=10, xmax=30, ymin=40, ymax=70)

    print(f"Width : {my_box.width}")
    print(f"Height: {my_box.height}")
    print(f"Area  : {my_box.area}")
    print(f"xmin  : {my_box.xmin}, xmax: {my_box.xmax}, ymin: {my_box.ymin}, ymax: {my_box.ymax}")


if __name__ == "__main__":
    main()
