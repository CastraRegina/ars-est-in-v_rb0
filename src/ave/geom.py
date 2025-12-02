"""Handling geometries"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ave.common import AvGlyphCmds


###############################################################################
# BezierCurve
###############################################################################
class BezierCurve:
    """Class to represent a Bezier curve."""

    @classmethod
    def polygonize_quadratic_curve_python(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]], steps: int
    ) -> NDArray[np.float64]:
        """
        Polygonize a quadratic Bézier curve into line segments using pure Python.

        Args:
            points: Control points as Sequence[Tuple[float, float]] or NDArray[np.float64]
                    Must contain exactly 3 points: start, control, end
            steps: Number of segments to divide the curve into

        Returns:
            NDArray[np.float64] of shape (steps+1, 3) containing the polygonized points (x, y, type=2.0)
        """
        # Extract control points - works for both numpy arrays and sequences
        pt0, pt1, pt2 = points

        # Direct coordinate extraction for minimal overhead
        p0x, p0y = pt0[0], pt0[1]
        p1x, p1y = pt1[0], pt1[1]
        p2x, p2y = pt2[0], pt2[1]

        # Create NumPy array with correct size (steps+1 to include starting point)
        result = np.empty((steps + 1, 3), dtype=np.float64)

        # Add starting point
        result[0, 0] = p0x
        result[0, 1] = p0y
        result[0, 2] = 0.0

        inv_steps = 1.0 / steps

        # Fill NumPy array directly during iteration
        for i in range(steps):
            t = (i + 1) * inv_steps
            omt = 1.0 - t
            omt2 = omt * omt
            t2 = t * t

            x = omt2 * p0x + 2.0 * omt * t * p1x + t2 * p2x
            y = omt2 * p0y + 2.0 * omt * t * p1y + t2 * p2y

            # Direct assignment to NumPy array (starting from index 1)
            result[i + 1, 0] = x
            result[i + 1, 1] = y
            result[i + 1, 2] = 2.0

        # Set last point type to 0.0
        result[steps, 2] = 0.0

        return result

    @classmethod
    def polygonize_cubic_curve_python(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]], steps: int
    ) -> NDArray[np.float64]:
        """
        Polygonize a cubic Bézier curve into line segments using pure Python.

        Args:
            points: Control points as Sequence[Tuple[float, float]] or NDArray[np.float64]
                    Must contain exactly 4 points: start, control1, control2, end
            steps: Number of segments to divide the curve into

        Returns:
            NDArray[np.float64] of shape (steps+1, 3) containing the polygonized points (x, y, type=3.0)
        """
        # Extract control points - works for both numpy arrays and sequences
        pt0, pt1, pt2, pt3 = points

        # Direct coordinate extraction for minimal overhead
        p0x, p0y = pt0[0], pt0[1]
        p1x, p1y = pt1[0], pt1[1]
        p2x, p2y = pt2[0], pt2[1]
        p3x, p3y = pt3[0], pt3[1]

        # Create NumPy array with correct size (steps+1 to include starting point)
        result = np.empty((steps + 1, 3), dtype=np.float64)

        # Add starting point
        result[0, 0] = p0x
        result[0, 1] = p0y
        result[0, 2] = 0.0

        inv_steps = 1.0 / steps

        # Fill NumPy array directly during iteration
        for i in range(steps):
            t = (i + 1) * inv_steps
            omt = 1.0 - t
            omt2 = omt * omt
            omt3 = omt2 * omt
            t2 = t * t
            t3 = t2 * t

            x = omt3 * p0x + 3.0 * omt2 * t * p1x + 3.0 * omt * t2 * p2x + t3 * p3x
            y = omt3 * p0y + 3.0 * omt2 * t * p1y + 3.0 * omt * t2 * p2y + t3 * p3y

            # Direct assignment to NumPy array (starting from index 1)
            result[i + 1, 0] = x
            result[i + 1, 1] = y
            result[i + 1, 2] = 3.0

        # Set last point type to 0.0
        result[steps, 2] = 0.0

        return result

    @classmethod
    def polygonize_quadratic_curve_numpy(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]], steps: int
    ) -> NDArray[np.float64]:
        """
        Polygonize a quadratic Bézier curve into line segments using NumPy.

        Args:
            points: Control points as Sequence[Tuple[float, float]] or NDArray[np.float64]
                    Must contain exactly 3 points: start, control, end
            steps: Number of segments to divide the curve into

        Returns:
            NDArray[np.float64] of shape (steps+1, 3) containing the polygonized points (x, y, type=2.0)
        """
        # Extract control points - works for both numpy arrays and sequences
        pt0, pt1, pt2 = points

        # Include starting point (t=0) and steps points
        t = np.arange(0, steps + 1) / steps
        omt = 1.0 - t
        x = omt**2 * pt0[0] + 2.0 * omt * t * pt1[0] + t**2 * pt2[0]
        y = omt**2 * pt0[1] + 2.0 * omt * t * pt1[1] + t**2 * pt2[1]

        # Create types array with first and last points as 0.0, others as 2.0
        types = np.full(steps + 1, 2.0, dtype=np.float64)
        types[0] = 0.0
        types[-1] = 0.0

        return np.column_stack([x, y, types])

    @classmethod
    def polygonize_cubic_curve_numpy(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]], steps: int
    ) -> NDArray[np.float64]:
        """
        Polygonize a cubic Bézier curve into line segments using NumPy.

        Args:
            points: Control points as Sequence[Tuple[float, float]] or NDArray[np.float64]
                    Must contain exactly 4 points: start, control1, control2, end
            steps: Number of segments to divide the curve into

        Returns:
            NDArray[np.float64] of shape (steps+1, 3) containing the polygonized points (x, y, type=3.0)
        """
        # Extract control points - works for both numpy arrays and sequences
        pt0, pt1, pt2, pt3 = points

        # Include starting point (t=0) and steps points
        t = np.arange(0, steps + 1) / steps
        omt = 1.0 - t
        x = omt**3 * pt0[0] + 3.0 * omt**2 * t * pt1[0] + 3.0 * omt * t**2 * pt2[0] + t**3 * pt3[0]
        y = omt**3 * pt0[1] + 3.0 * omt**2 * t * pt1[1] + 3.0 * omt * t**2 * pt2[1] + t**3 * pt3[1]

        # Create types array with first and last points as 0.0, others as 3.0
        types = np.full(steps + 1, 3.0, dtype=np.float64)
        types[0] = 0.0
        types[-1] = 0.0

        return np.column_stack([x, y, types])

    @classmethod
    def polygonize_quadratic_curve(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]], steps: int
    ) -> NDArray[np.float64]:
        """
        Polygonize a quadratic Bézier curve into line segments.
        Uses pure Python for small step counts, NumPy for larger ones.

        Args:
            points: Control points as Sequence[Tuple[float, float]] or NDArray[np.float64]
            steps: Number of segments to divide the curve into

        Returns:
            NDArray[np.float64] of shape (steps+1, 3) containing the polygonized points (x, y, type=2.0)
        """
        if steps < 50:
            return cls.polygonize_quadratic_curve_python(points, steps)
        else:
            return cls.polygonize_quadratic_curve_numpy(points, steps)

    @classmethod
    def polygonize_cubic_curve(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]], steps: int
    ) -> NDArray[np.float64]:
        """
        Polygonize a cubic Bézier curve into line segments.
        Uses pure Python for small step counts, NumPy for larger ones.

        Args:
            points: Control points as Sequence[Tuple[float, float]] or NDArray[np.float64]
                    Must contain exactly 4 points: start, control1, control2, end
            steps: Number of segments to divide the curve into

        Returns:
            NDArray[np.float64] of shape (steps+1, 3) containing the polygonized points (x, y, type=3.0)
        """
        if steps < 50:
            return cls.polygonize_cubic_curve_python(points, steps)
        else:
            return cls.polygonize_cubic_curve_numpy(points, steps)

    # TODO: check and correct the function
    @staticmethod
    def polygonize_path(
        points: NDArray[np.float64], commands: List[AvGlyphCmds], steps: int
    ) -> Tuple[NDArray[np.float64], List[AvGlyphCmds]]:
        """
        Polygonize a path by converting curve commands (C, Q) to line segments.

        Args:
            points: Array of points with shape (n, 3) containing (x, y, type)
            commands: List of path commands (M, L, C, Q, Z)
            polygonize_steps: Number of segments to use for curve polygonization

        Returns:
            Tuple of (new_points, new_commands) where curves are replaced by line segments
        """
        # if len(points) != len(commands):
        #     raise ValueError(f"Points ({len(points)}) and commands ({len(commands)}) must have same length")

        # TODO: modify so that it fits to new polygonize-methods which are returning the first control points.
        # Therefore the first point has to be removed before concatenate to the whole path

        new_points_list = []
        new_commands_list = []

        i = 0
        while i < len(commands):
            cmd = commands[i]
            pt = points[i]

            if cmd == "M":  # MoveTo
                new_points_list.append(pt)
                new_commands_list.append(cmd)
                i += 1

            elif cmd == "L":  # LineTo
                new_points_list.append(pt)
                new_commands_list.append(cmd)
                i += 1

            elif cmd == "Q":  # Quadratic Bezier To
                if i + 2 >= len(points):
                    raise ValueError(f"Quadratic Bezier command at index {i} needs 3 points, got {len(points) - i}")

                # Get control points: current point, control point, end point
                if len(new_points_list) == 0:
                    raise ValueError(f"Quadratic Bezier at index {i} has no starting point")

                # Extract all control points using NumPy slicing for better performance
                # Get start point (previous point) + control, end points
                control_points = points[i - 1 : i + 2, :2]  # Get start, control, end points (x, y only)

                # Polygonize the quadratic bezier
                curve_points = BezierCurve.polygonize_quadratic_curve(control_points, steps)

                # Add all polygonized points as line segments
                new_points_list.extend(curve_points)
                new_commands_list.extend(["L"] * len(curve_points))

                i += 2  # Skip control and end points (they're now polygonized)

            elif cmd == "C":  # Cubic Bezier To
                if i + 3 >= len(points):
                    raise ValueError(f"Cubic Bezier command at index {i} needs 4 points, got {len(points) - i}")

                # Get control points: current point, control1, control2, end point
                if len(new_points_list) == 0:
                    raise ValueError(f"Cubic Bezier at index {i} has no starting point")

                # Extract all control points using NumPy slicing for better performance
                # Get start point (previous point) + control1, control2, end points
                control_points = points[i - 1 : i + 3, :2]  # Get start, control1, control2, end points (x, y only)

                # Polygonize the cubic bezier
                curve_points = BezierCurve.polygonize_cubic_curve(control_points, steps)

                # Add all polygonized points as line segments
                new_points_list.extend(curve_points)
                new_commands_list.extend(["L"] * len(curve_points))

                i += 3  # Skip control1, control2, and end points (they're now polygonized)

            elif cmd == "Z":  # ClosePath
                new_points_list.append(pt)
                new_commands_list.append(cmd)
                i += 1

            else:
                raise ValueError(f"Unknown command '{cmd}' at index {i}")

        new_points = np.array(new_points_list, dtype=np.float64)
        return new_points, new_commands_list


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

    def __str__(self):
        """Returns a string representation of the AvBox instance."""

        return (
            f"AvBox(xmin={self._xmin}, ymin={self._ymin}, "
            f"      xmax={self._xmax}, ymax={self._ymax}, "
            f"      width={self.width}, height={self.height})"
        )

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


def main():
    """Main"""

    my_box = AvBox(xmin=10, ymin=40, xmax=30, ymax=70)

    print(f"Width : {my_box.width}")
    print(f"Height: {my_box.height}")
    print(f"Area  : {my_box.area}")
    print(f"xmin  : {my_box.xmin}, ymin: {my_box.ymin}, xmax: {my_box.xmax}, ymax: {my_box.ymax}")

    my_box = my_box.transform_scale_translate(1, 10, 20)
    print(f"Width : {my_box.width}")
    print(f"Height: {my_box.height}")
    print(f"Area  : {my_box.area}")
    print(f"xmin  : {my_box.xmin}, ymin: {my_box.ymin}, xmax: {my_box.xmax}, ymax: {my_box.ymax}")

    my_box = my_box.transform_affine((2, 0, 0, 2, 0, 0))
    print(f"Width : {my_box.width}")
    print(f"Height: {my_box.height}")
    print(f"Area  : {my_box.area}")
    print(f"xmin  : {my_box.xmin}, ymin: {my_box.ymin}, xmax: {my_box.xmax}, ymax: {my_box.ymax}")

    # Create a Bezier curve and polygonize it
    control_points = [(0, 0), (10, 10), (20, 0)]
    polygon = BezierCurve.polygonize_quadratic_curve(control_points, 2)

    # Print the result
    print("Polygonized quadratic Bezier curve:")
    print("  Number of points:", len(polygon))
    print("  Points:")
    for point in polygon:
        print("    ", point)

    # # Example for polygonize_path
    # print("\n" + "=" * 50)
    # print("Example for polygonize_path method:")
    # print("=" * 50)

    # # Create a simple path with MoveTo, LineTo, and Quadratic Bezier
    # # Points format: (x, y, type) - type is not used in polygonize_path but required for format
    # path_points = np.array(
    #     [
    #         [10.0, 10.0, 0.0],  # M - MoveTo starting point
    #         [20.0, 20.0, 2.0],  # Q - Quadratic Bezier control point
    #         [30.0, 10.0, 0.0],  # Q - Quadratic Bezier end point
    #     ],
    #     dtype=np.float64,
    # )

    # path_commands = ["M", "Q"]

    # print("Original path:")
    # print("  Number of points:", len(path_points))
    # print("  Commands:", " -> ".join(path_commands))
    # print("  Points:")
    # for i, (point, cmd) in enumerate(zip(path_points, path_commands)):
    #     print(f"    {i:2d}: {cmd} ({point[0]:6.1f}, {point[1]:6.1f})")

    # # Polygonize the path with 2 steps per curve
    # polygonize_steps = 2
    # new_points, new_commands = BezierCurve.polygonize_path(path_points, path_commands, polygonize_steps)

    # print(f"\nPolygonized path (steps={polygonize_steps}):")
    # print("  Number of points:", len(new_points))
    # print("  Commands:", " -> ".join(new_commands))
    # print("  Points:")
    # for i, (point, cmd) in enumerate(zip(new_points, new_commands)):
    #     print(f"    {i:2d}: {cmd} ({point[0]:6.1f}, {point[1]:6.1f})")

    # # Show statistics
    # original_curves = sum(1 for cmd in path_commands if cmd in ["Q", "C"])
    # total_segments = sum(1 for cmd in new_commands if cmd == "L")

    # print(f"\nPolygonization statistics:")
    # print(f"  Original curves: {original_curves}")
    # print(f"  Total line segments after polygonization: {total_segments}")
    # print(
    #     f"  Average segments per curve: {total_segments / original_curves:.1f}"
    #     if original_curves > 0
    #     else "  No curves to polygonize"
    # )


if __name__ == "__main__":
    main()
