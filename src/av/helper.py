"""Provide various SVG helper functionalities."""

from __future__ import annotations

import math
from typing import Any, List, Tuple, Union, cast

import numpy

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
