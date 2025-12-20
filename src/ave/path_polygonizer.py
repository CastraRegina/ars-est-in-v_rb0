"""Path polygonization utilities for converting curves to line segments."""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

from ave.bezier import BezierCurve
from ave.common import AvGlyphCmds
from ave.path_support import PathCommandProcessor


class PathPolygonizer:
    """Utility class for polygonizing paths with curves into line segments."""

    @staticmethod
    def polygonize_path(
        points: NDArray[np.float64], commands: List[AvGlyphCmds], steps: int, process_2d_to_3d_func
    ) -> tuple[NDArray[np.float64], List[AvGlyphCmds]]:
        """Convert a path with curves to a polygonized path with line segments.

        Args:
            points: Array of path points (shape: n_points, 2 or 3)
            commands: List of SVG commands
            steps: Number of segments to use for curve approximation
            process_2d_to_3d_func: Function to convert 2D points to 3D with type column

        Returns:
            Tuple of (polygonized_points, polygonized_commands)

        Raises:
            ValueError: If steps is 0 or path has invalid structure
        """
        if steps == 0:
            return points, commands

        # Input normalization: ensure all points are 3D
        if points.shape[1] == 2:
            points = process_2d_to_3d_func(points, commands)

        # Pre-allocation: estimate final size more accurately
        # Count actual curves and estimate points needed
        num_quadratic = commands.count("Q")
        num_cubic = commands.count("C")
        # Each Q becomes (steps) L's, each C becomes (steps) L's
        estimated_points = (
            len(points)  # Original points
            - num_quadratic * 2  # Remove original curve points
            - num_cubic * 3  # Remove original curve points
            + (num_quadratic + num_cubic) * steps  # Add polygonized points
        )
        new_points_array = np.empty((estimated_points, 3), dtype=np.float64)
        new_commands_list = []

        point_index = 0
        array_index = 0
        last_point = None

        for cmd in commands:
            consumed = PathCommandProcessor.get_point_consumption(cmd)

            if cmd == "M":  # MoveTo - uses consumed points
                if point_index + consumed > len(points):
                    raise ValueError(f"MoveTo command needs {consumed} point, got {len(points) - point_index}")

                pt = points[point_index]
                new_points_array[array_index] = pt
                new_commands_list.append(cmd)
                last_point = pt[:2]  # Store 2D for curve calculations
                array_index += 1
                point_index += consumed

            elif cmd == "L":  # LineTo - uses consumed points
                if point_index + consumed > len(points):
                    raise ValueError(f"LineTo command needs {consumed} point, got {len(points) - point_index}")

                pt = points[point_index]
                new_points_array[array_index] = pt
                new_commands_list.append(cmd)
                last_point = pt[:2]  # Store 2D for curve calculations
                array_index += 1
                point_index += consumed

            elif cmd == "Q":  # Quadratic Bezier To - uses consumed points
                if point_index + consumed > len(points):
                    raise ValueError(
                        f"Quadratic Bezier command needs {consumed} points, got {len(points) - point_index}"
                    )

                if array_index == 0:
                    raise ValueError("Quadratic Bezier command has no starting point")

                # Get start point (last point) + control and end points
                control_point = points[point_index][:2]
                end_point = points[point_index + 1][:2]

                control_points = np.array([last_point, control_point, end_point], dtype=np.float64)

                # Polygonize the quadratic bezier directly into output buffer
                num_curve_points = BezierCurve.polygonize_quadratic_curve_inplace(
                    control_points, steps, new_points_array, array_index, skip_first=True
                )
                new_commands_list.extend(["L"] * num_curve_points)
                array_index += num_curve_points
                last_point = end_point  # Update last point
                point_index += consumed

            elif cmd == "C":  # Cubic Bezier To - uses consumed points
                if point_index + consumed > len(points):
                    raise ValueError(f"Cubic Bezier command needs {consumed} points, got {len(points) - point_index}")

                if array_index == 0:
                    raise ValueError("Cubic Bezier command has no starting point")

                # Get start point (last point) + control1, control2, and end points
                control1_point = points[point_index][:2]
                control2_point = points[point_index + 1][:2]
                end_point = points[point_index + 2][:2]

                control_points = np.array([last_point, control1_point, control2_point, end_point], dtype=np.float64)

                # Polygonize the cubic bezier directly into output buffer
                num_curve_points = BezierCurve.polygonize_cubic_curve_inplace(
                    control_points, steps, new_points_array, array_index, skip_first=True
                )
                new_commands_list.extend(["L"] * num_curve_points)
                array_index += num_curve_points
                last_point = end_point  # Update last point
                point_index += consumed

            elif cmd == "Z":  # ClosePath - uses 0 points, no point data in SVG
                if array_index == 0:
                    raise ValueError("ClosePath command has no starting point")

                # Z command doesn't add a new point, it just closes the path
                # The closing line is implicit from current point to first MoveTo point
                new_commands_list.append(cmd)

            else:
                raise ValueError(f"Unknown command '{cmd}'")

        # Trim the pre-allocated array to actual size
        new_points = new_points_array[:array_index]
        return new_points, new_commands_list
