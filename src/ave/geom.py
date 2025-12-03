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

    @staticmethod
    def polygonize_path(
        points: NDArray[np.float64], commands: List[AvGlyphCmds], steps: int
    ) -> Tuple[NDArray[np.float64], List[AvGlyphCmds]]:
        """
        Polygonize a path by converting curve commands (C, Q) to line segments.

        Args:
            points: Array of points with shape (n, 3) containing (x, y, type)
            commands: List of path commands (M, L, C, Q, Z)
            steps: Number of segments to use for curve polygonization

        Returns:
            Tuple of (new_points, new_commands) where curves are replaced by line segments
        """
        new_points_list = []
        new_commands_list = []

        point_index = 0

        for cmd in commands:
            if cmd == "M":  # MoveTo - uses 1 point
                if point_index >= len(points):
                    raise ValueError(f"MoveTo command needs 1 point, got {len(points) - point_index}")

                pt = points[point_index]
                new_points_list.append(pt)
                new_commands_list.append(cmd)
                point_index += 1

            elif cmd == "L":  # LineTo - uses 1 point
                if point_index >= len(points):
                    raise ValueError(f"LineTo command needs 1 point, got {len(points) - point_index}")

                pt = points[point_index]
                new_points_list.append(pt)
                new_commands_list.append(cmd)
                point_index += 1

            elif cmd == "Q":  # Quadratic Bezier To - uses 2 points (control, end)
                if point_index + 1 >= len(points):
                    raise ValueError(f"Quadratic Bezier command needs 2 points, got {len(points) - point_index}")

                if len(new_points_list) == 0:
                    raise ValueError("Quadratic Bezier command has no starting point")

                # Get start point (last point in new_points_list) + control and end points
                start_point = new_points_list[-1][:2]  # Get x,y from last point
                control_point = points[point_index][:2]
                end_point = points[point_index + 1][:2]

                control_points = np.array([start_point, control_point, end_point], dtype=np.float64)

                # Polygonize the quadratic bezier
                curve_points = BezierCurve.polygonize_quadratic_curve(control_points, steps)

                # Remove first point since it's the same as start_point to avoid duplication
                curve_points = curve_points[1:]

                # Add polygonized points as line segments
                new_points_list.extend(curve_points)
                new_commands_list.extend(["L"] * len(curve_points))

                point_index += 2  # Skip control and end points

            elif cmd == "C":  # Cubic Bezier To - uses 3 points (control1, control2, end)
                if point_index + 2 >= len(points):
                    raise ValueError(f"Cubic Bezier command needs 3 points, got {len(points) - point_index}")

                if len(new_points_list) == 0:
                    raise ValueError("Cubic Bezier command has no starting point")

                # Get start point (last point in new_points_list) + control1, control2, and end points
                start_point = new_points_list[-1][:2]  # Get x,y from last point
                control1_point = points[point_index][:2]
                control2_point = points[point_index + 1][:2]
                end_point = points[point_index + 2][:2]

                control_points = np.array([start_point, control1_point, control2_point, end_point], dtype=np.float64)

                # Polygonize the cubic bezier
                curve_points = BezierCurve.polygonize_cubic_curve(control_points, steps)

                # Remove first point since it's the same as start_point to avoid duplication
                curve_points = curve_points[1:]

                # Add polygonized points as line segments
                new_points_list.extend(curve_points)
                new_commands_list.extend(["L"] * len(curve_points))

                point_index += 3  # Skip control1, control2, and end points

            elif cmd == "Z":  # ClosePath - uses 0 points, no point data in SVG
                if len(new_points_list) == 0:
                    raise ValueError("ClosePath command has no starting point")

                # Z command doesn't add a new point, it just closes the path
                # The closing line is implicit from current point to first MoveTo point
                new_commands_list.append(cmd)

            else:
                raise ValueError(f"Unknown command '{cmd}'")

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

    ###########################################################################
    # Example path_points and path_commands containing only a Move followed by one Quadratic Bezier curve
    path_points = np.array(
        [
            [10.0, 10.0, 0.0],  # M - MoveTo destination point
            [20.0, 20.0, 2.0],  # Q - Quadratic Bezier control point
            [30.0, 10.0, 0.0],  # Q - Quadratic Bezier end point
        ],
        dtype=np.float64,
    )

    path_commands = ["M", "Q"]  # M uses 1 point, Q uses 2 points
    print("\n" + "=" * 50)
    print("Example path_points and path_commands:")
    print("=" * 50)
    print("  Number of points:", len(path_points))
    print("  Number of commands:", len(path_commands))
    print("  Points:")
    for point in path_points:
        print("    ", (point[0], point[1], point[2]))
    print("  Commands:")
    for cmd in path_commands:
        print("    ", cmd)

    ###########################################################################
    # Example for MoveTo followed by Cubic Bezier
    path_points_mc = np.array(
        [
            [10.0, 10.0, 0.0],  # M - MoveTo destination point
            [10.0, 30.0, 3.0],  # C - Cubic Bezier control point 1
            [30.0, 30.0, 3.0],  # C - Cubic Bezier control point 2
            [30.0, 10.0, 0.0],  # C - Cubic Bezier end point
        ],
        dtype=np.float64,
    )

    path_commands_mc = ["M", "C"]  # M uses 1 point, C uses 3 points
    print("\n" + "=" * 50)
    print("Example M,C path_points and path_commands:")
    print("=" * 50)
    print("  Number of points:", len(path_points_mc))
    print("  Number of commands:", len(path_commands_mc))
    print("  Points:")
    for point in path_points_mc:
        print("    ", (point[0], point[1], point[2]))
    print("  Commands:")
    for cmd in path_commands_mc:
        print("    ", cmd)

    ###########################################################################
    ###########################################################################
    # Test the corrected polygonize_path method with both examples

    polygonize_steps = 5
    print("\n" + "=" * 50)
    print("Testing polygonize_path method:")
    print("=" * 50)

    ###########################################################################
    # Test M,Q example
    print("\nTesting M,Q example:")
    new_points_q, new_commands_q = BezierCurve.polygonize_path(path_points, path_commands, polygonize_steps)

    print(f"Original: {len(path_points)} points, {len(path_commands)} commands")
    print(f"Polygonized: {len(new_points_q)} points, {len(new_commands_q)} commands")
    print("Polygonized points:")
    for i, (point, cmd) in enumerate(zip(new_points_q, new_commands_q)):
        print(f"  {i:2d}: {cmd} ({point[0]:6.1f}, {point[1]:6.1f})")

    ###########################################################################
    # Test M,C example
    print("\nTesting M,C example:")
    new_points_c, new_commands_c = BezierCurve.polygonize_path(path_points_mc, path_commands_mc, polygonize_steps)

    print(f"Original: {len(path_points_mc)} points, {len(path_commands_mc)} commands")
    print(f"Polygonized: {len(new_points_c)} points, {len(new_commands_c)} commands")
    print("Polygonized points:")
    for i, (point, cmd) in enumerate(zip(new_points_c, new_commands_c)):
        if cmd == "Z":
            print(f"  {i:2d}: {cmd} (close path - no point data)")
        else:
            print(f"  {i:2d}: {cmd} ({point[0]:6.1f}, {point[1]:6.1f})")

    ###########################################################################
    # Complex path example with multiple command types
    print("\n" + "=" * 50)
    print("Complex path example with multiple command types:")
    print("=" * 50)

    # Create a complex path: M -> L -> Q -> L -> C -> L -> Z
    complex_path_points = np.array(
        [
            # M - MoveTo starting point
            [10.0, 10.0, 0.0],
            # L - LineTo
            [30.0, 10.0, 0.0],
            # Q - Quadratic Bezier (control point, end point)
            [40.0, 20.0, 2.0],  # Control point
            [50.0, 10.0, 0.0],  # End point
            # L - LineTo
            [70.0, 10.0, 0.0],
            # C - Cubic Bezier (control1, control2, end point)
            [80.0, 20.0, 3.0],  # Control point 1
            [90.0, 20.0, 3.0],  # Control point 2
            [100.0, 10.0, 0.0],  # End point
            # L - LineTo
            [100.0, 30.0, 0.0],
        ],
        dtype=np.float64,
    )

    complex_path_commands = ["M", "L", "Q", "L", "C", "L", "Z"]  # Z uses 0 points

    print("Complex path:")
    print("  Number of points:", len(complex_path_points))
    print("  Number of commands:", len(complex_path_commands))
    print("  Points:")
    for point in complex_path_points:
        print("    ", (point[0], point[1], point[2]))
    print("  Commands:")
    for cmd in complex_path_commands:
        print("    ", cmd)

    # Test the complex path
    print("\nTesting complex path:")
    new_points_complex, new_commands_complex = BezierCurve.polygonize_path(
        complex_path_points, complex_path_commands, polygonize_steps
    )

    print(f"Original: {len(complex_path_points)} points, {len(complex_path_commands)} commands")
    print(f"Polygonized: {len(new_points_complex)} points, {len(new_commands_complex)} commands")
    print("Polygonized points:")
    for i, cmd in enumerate(new_commands_complex):
        if cmd == "Z":
            print(f"  {i:2d}: {cmd} (close path - no point data)")
        else:
            point = new_points_complex[i]
            print(f"  {i:2d}: {cmd} ({point[0]:6.1f}, {point[1]:6.1f})")

    # Show statistics
    original_curves = sum(1 for cmd in complex_path_commands if cmd in ["Q", "C"])
    original_lines = sum(1 for cmd in complex_path_commands if cmd in ["L"])
    total_segments = sum(1 for cmd in new_commands_complex if cmd == "L")

    print("\nPolygonization statistics:")
    print(f"  Original curves: {original_curves}")
    print(f"  Original lines: {original_lines}")
    print(f"  Total line segments after polygonization: {total_segments}")
    print(
        f"  Average segments per curve: {total_segments / original_curves:.1f}"
        if original_curves > 0
        else "  No curves to polygonize"
    )


if __name__ == "__main__":
    main()
