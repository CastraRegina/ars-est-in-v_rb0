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
    def polygonize_quadratic_curve_python_inplace(
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        cls,
        points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]],
        steps: int,
        output_buffer: NDArray[np.float64],
        start_index: int = 0,
        skip_first: bool = False,
    ) -> int:
        """
        Polygonize a quadratic Bezier curve directly into pre-allocated buffer using pure Python.
        Optimized using forward differencing for O(1) per point computation.
        """
        # Extract control points
        pt0, pt1, pt2 = points
        p0x, p0y = pt0[0], pt0[1]
        p1x, p1y = pt1[0], pt1[1]
        p2x, p2y = pt2[0], pt2[1]

        # Precompute Bezier coefficients using forward differencing
        # B(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
        # Expanding: B(t) = P0 + 2*t*(P1-P0) + t^2*(P0-2*P1+P2)
        inv_steps = 1.0 / steps
        inv_steps_sq = inv_steps * inv_steps

        # First point (t=0)
        x = p0x
        y = p0y

        # Second differences (d^2*B/dt^2 * inv_steps^2) - constant for quadratic
        # Second derivative of B(t) is 2*(P0-2*P1+P2), scaled by inv_steps^2
        dx_second = 2.0 * inv_steps_sq * (p0x - 2.0 * p1x + p2x)
        dy_second = 2.0 * inv_steps_sq * (p0y - 2.0 * p1y + p2y)

        # Initialize first differences with midpoint correction
        # First derivative at t=0: B'(0) = 2*(P1-P0), scaled by inv_steps
        # Plus half second difference for midpoint correction (since we update position before derivatives)
        dx_first = 2.0 * inv_steps * (p1x - p0x) + 0.5 * dx_second
        dy_first = 2.0 * inv_steps * (p1y - p0y) + 0.5 * dy_second

        output_idx = start_index

        # Handle skip_first by adjusting loop bounds
        start_i = 1  # Always start at 1 since first point is handled separately
        end_i = steps + 1

        # Write first point if not skipped
        if not skip_first:
            output_buffer[output_idx, 0] = x
            output_buffer[output_idx, 1] = y
            output_buffer[output_idx, 2] = 0.0
            output_idx += 1

        # Use forward differencing: B(t+dt) = B(t) + dB(t) + d^2*B
        # where dB(t+dt) = dB(t) + d^2*B (constant second difference)
        for i in range(start_i, end_i):
            # Update position first using current derivatives
            x += dx_first
            y += dy_first

            # Then update derivatives for next iteration
            dx_first += dx_second
            dy_first += dy_second

            # Write to buffer
            output_buffer[output_idx, 0] = x
            output_buffer[output_idx, 1] = y
            output_buffer[output_idx, 2] = 2.0 if i < steps else 0.0
            output_idx += 1

        return steps + (1 if not skip_first else 0)

    @classmethod
    def polygonize_cubic_curve_python_inplace(
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        cls,
        points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]],
        steps: int,
        output_buffer: NDArray[np.float64],
        start_index: int = 0,
        skip_first: bool = False,
    ) -> int:
        """
        Polygonize a cubic Bezier curve directly into pre-allocated buffer using pure Python.
        Optimized using forward differencing for O(1) per point computation.
        """
        # Extract control points
        pt0, pt1, pt2, pt3 = points
        p0x, p0y = pt0[0], pt0[1]
        p1x, p1y = pt1[0], pt1[1]
        p2x, p2y = pt2[0], pt2[1]
        p3x, p3y = pt3[0], pt3[1]

        # Precompute Bezier coefficients using forward differencing
        # B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
        inv_steps = 1.0 / steps

        # First point (t=0)
        x = p0x
        y = p0y

        # Compute discrete differences directly from curve points for exact forward differencing
        # Calculate first 4 points using Bezier formula to derive exact differences
        h = inv_steps  # step size

        # B(0) = P0
        b0_x, b0_y = p0x, p0y

        # B(h) using Bezier formula
        t = h
        omt = 1.0 - t
        omt2 = omt * omt
        omt3 = omt2 * omt
        t2 = t * t
        t3 = t2 * t
        b1_x = omt3 * p0x + 3.0 * omt2 * t * p1x + 3.0 * omt * t2 * p2x + t3 * p3x
        b1_y = omt3 * p0y + 3.0 * omt2 * t * p1y + 3.0 * omt * t2 * p2y + t3 * p3y

        # B(2h) using Bezier formula
        t = 2.0 * h
        omt = 1.0 - t
        omt2 = omt * omt
        omt3 = omt2 * omt
        t2 = t * t
        t3 = t2 * t
        b2_x = omt3 * p0x + 3.0 * omt2 * t * p1x + 3.0 * omt * t2 * p2x + t3 * p3x
        b2_y = omt3 * p0y + 3.0 * omt2 * t * p1y + 3.0 * omt * t2 * p2y + t3 * p3y

        # B(3h) using Bezier formula
        t = 3.0 * h
        omt = 1.0 - t
        omt2 = omt * omt
        omt3 = omt2 * omt
        t2 = t * t
        t3 = t2 * t
        b3_x = omt3 * p0x + 3.0 * omt2 * t * p1x + 3.0 * omt * t2 * p2x + t3 * p3x
        b3_y = omt3 * p0y + 3.0 * omt2 * t * p1y + 3.0 * omt * t2 * p2y + t3 * p3y

        # Derive discrete differences from actual curve points
        # First differences: ΔB = B(h) - B(0)
        dx_first = b1_x - b0_x
        dy_first = b1_y - b0_y

        # Second differences: Δ²B = B(2h) - 2*B(h) + B(0)
        dx_second = b2_x - 2.0 * b1_x + b0_x
        dy_second = b2_y - 2.0 * b1_y + b0_y

        # Third differences: Δ³B = B(3h) - 3*B(2h) + 3*B(h) - B(0) (constant for cubic)
        dx_third = b3_x - 3.0 * b2_x + 3.0 * b1_x - b0_x
        dy_third = b3_y - 3.0 * b2_y + 3.0 * b1_y - b0_y

        output_idx = start_index

        # Handle skip_first by adjusting loop bounds
        start_i = 1  # Always start at 1 since first point is handled separately
        end_i = steps + 1

        # Write first point if not skipped
        if not skip_first:
            output_buffer[output_idx, 0] = x
            output_buffer[output_idx, 1] = y
            output_buffer[output_idx, 2] = 0.0
            output_idx += 1

        # Use forward differencing: B(t+dt) = B(t) + dB(t) + d^2*B(t) + d^3*B
        # where d^3*B is constant for cubic curves
        for i in range(start_i, end_i):
            # Update position first using current derivatives
            x += dx_first
            y += dy_first

            # Then update derivatives for next iteration
            dx_first += dx_second
            dy_first += dy_second
            dx_second += dx_third
            dy_second += dy_third

            # Write to buffer
            output_buffer[output_idx, 0] = x
            output_buffer[output_idx, 1] = y
            output_buffer[output_idx, 2] = 3.0 if i < steps else 0.0
            output_idx += 1

        return steps + (1 if not skip_first else 0)

    @classmethod
    def polygonize_cubic_curve_python(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]], steps: int
    ) -> NDArray[np.float64]:
        """Polygonize a cubic Bezier curve using the pure Python implementation.

        This is a convenience wrapper around polygonize_cubic_curve_python_inplace
        that allocates the output buffer and returns it.
        """
        result = np.empty((steps + 1, 3), dtype=np.float64)
        cls.polygonize_cubic_curve_python_inplace(points, steps, result, start_index=0, skip_first=False)
        return result

    @classmethod
    def polygonize_cubic_curve_numpy(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]], steps: int
    ) -> NDArray[np.float64]:
        """Polygonize a cubic Bezier curve using the NumPy implementation.

        This is a convenience wrapper around polygonize_cubic_curve_numpy_inplace
        that allocates the output buffer and returns it.
        """
        result = np.empty((steps + 1, 3), dtype=np.float64)
        cls.polygonize_cubic_curve_numpy_inplace(points, steps, result, start_index=0, skip_first=False)
        return result

    @classmethod
    def polygonize_quadratic_curve(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]], steps: int
    ) -> NDArray[np.float64]:
        """
        Polygonize a quadratic Bezier curve into line segments.
        Uses pure Python for small step counts, NumPy for larger ones.

        Args:
            points: Control points as Sequence[Tuple[float, float]] or NDArray[np.float64]
            steps: Number of segments to divide the curve into

        Returns:
            NDArray[np.float64] of shape (steps+1, 3) containing the polygonized points (x, y, type=2.0)
        """
        result = np.empty((steps + 1, 3), dtype=np.float64)
        cls.polygonize_quadratic_curve_inplace(points, steps, result, start_index=0, skip_first=False)
        return result

    @classmethod
    def polygonize_cubic_curve(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]], steps: int
    ) -> NDArray[np.float64]:
        """
        Polygonize a cubic Bezier curve into line segments.
        Uses pure Python for small step counts, NumPy for larger ones.

        Args:
            points: Control points as Sequence[Tuple[float, float]] or NDArray[np.float64]
                    Must contain exactly 4 points: start, control1, control2, end
            steps: Number of segments to divide the curve into

        Returns:
            NDArray[np.float64] of shape (steps+1, 3) containing the polygonized points (x, y, type=3.0)
        """
        # Create buffer and call in-place implementation
        result = np.empty((steps + 1, 3), dtype=np.float64)
        cls.polygonize_cubic_curve_inplace(points, steps, result, start_index=0, skip_first=False)
        return result

    @classmethod
    def polygonize_quadratic_curve_inplace(
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        cls,
        points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]],
        steps: int,
        output_buffer: NDArray[np.float64],
        start_index: int = 0,
        skip_first: bool = False,
    ) -> int:
        """
        Polygonize a quadratic Bezier curve directly into pre-allocated buffer.
        Uses pure Python for small step counts, NumPy for larger ones.

        Args:
            points: Control points as Sequence[Tuple[float, float]] or NDArray[np.float64]
            steps: Number of segments to divide the curve into
            output_buffer: Pre-allocated buffer to write points into
            start_index: Starting index in output_buffer
            skip_first: If True, skip writing the first point (to avoid duplication)

        Returns:
            Number of points written to buffer
        """
        if steps < 70:
            return cls.polygonize_quadratic_curve_python_inplace(points, steps, output_buffer, start_index, skip_first)
        else:
            return cls.polygonize_quadratic_curve_numpy_inplace(points, steps, output_buffer, start_index, skip_first)

    @classmethod
    def polygonize_cubic_curve_inplace(
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        cls,
        points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]],
        steps: int,
        output_buffer: NDArray[np.float64],
        start_index: int = 0,
        skip_first: bool = False,
    ) -> int:
        """
        Polygonize a cubic Bezier curve directly into pre-allocated buffer.
        Uses pure Python for small step counts, NumPy for larger ones.

        Args:
            points: Control points as Sequence[Tuple[float, float]] or NDArray[np.float64]
                    Must contain exactly 4 points: start, control1, control2, end
            steps: Number of segments to divide the curve into
            output_buffer: Pre-allocated buffer to write points into
            start_index: Starting index in output_buffer
            skip_first: If True, skip writing the first point (to avoid duplication)

        Returns:
            Number of points written to buffer
        """
        if steps < 70:
            return cls.polygonize_cubic_curve_python_inplace(points, steps, output_buffer, start_index, skip_first)
        else:
            return cls.polygonize_cubic_curve_numpy_inplace(points, steps, output_buffer, start_index, skip_first)

    @classmethod
    def polygonize_quadratic_curve_numpy_inplace(
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        cls,
        points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]],
        steps: int,
        output_buffer: NDArray[np.float64],
        start_index: int = 0,
        skip_first: bool = False,
    ) -> int:
        """
        Polygonize a quadratic Bezier curve directly into pre-allocated buffer using NumPy.
        Uses direct evaluation with vectorized operations for optimal NumPy performance.
        """
        # Convert to numpy array if needed
        points_array = np.array(points, dtype=np.float64)

        # Create parameter array
        t = np.linspace(0, 1, steps + 1, dtype=np.float64)
        if skip_first:
            t = t[1:]  # Skip t=0, but keep t=1.0

        # Quadratic Bezier basis functions
        omt = 1 - t
        omt2 = omt**2
        t2 = t**2

        # Calculate curve points
        x = omt2 * points_array[0, 0] + 2 * omt * t * points_array[1, 0] + t2 * points_array[2, 0]
        y = omt2 * points_array[0, 1] + 2 * omt * t * points_array[1, 1] + t2 * points_array[2, 1]

        # Set types
        types = np.full(len(t), 2.0, dtype=np.float64)
        if len(types) > 0:
            if not skip_first:
                types[0] = 0.0  # First point is start point when not skipping
            types[-1] = 0.0  # End point is always 0.0

        # Write directly to output buffer
        end_idx = start_index + len(t)
        output_buffer[start_index:end_idx, 0] = x
        output_buffer[start_index:end_idx, 1] = y
        output_buffer[start_index:end_idx, 2] = types

        return len(t)

    @classmethod
    def polygonize_cubic_curve_numpy_inplace(
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        cls,
        points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]],
        steps: int,
        output_buffer: NDArray[np.float64],
        start_index: int = 0,
        skip_first: bool = False,
    ) -> int:
        """
        Polygonize a cubic Bezier curve directly into pre-allocated buffer using NumPy.
        Uses direct evaluation with vectorized operations for optimal NumPy performance.
        """
        # Convert to numpy array if needed
        points_array = np.array(points, dtype=np.float64)

        # Create parameter array
        t = np.linspace(0, 1, steps + 1, dtype=np.float64)
        if skip_first:
            t = t[1:]  # Skip t=0, but keep t=1.0

        # Cubic Bezier basis functions
        omt = 1 - t
        omt2 = omt**2
        omt3 = omt2 * omt
        t2 = t**2
        t3 = t2 * t

        # Calculate curve points
        x = (
            omt3 * points_array[0, 0]
            + 3 * omt2 * t * points_array[1, 0]
            + 3 * omt * t2 * points_array[2, 0]
            + t3 * points_array[3, 0]
        )
        y = (
            omt3 * points_array[0, 1]
            + 3 * omt2 * t * points_array[1, 1]
            + 3 * omt * t2 * points_array[2, 1]
            + t3 * points_array[3, 1]
        )

        # Set types
        types = np.full(len(t), 3.0, dtype=np.float64)
        if len(types) > 0:
            if not skip_first:
                types[0] = 0.0  # First point is start point when not skipping
            types[-1] = 0.0  # End point is always 0.0

        # Write directly to output buffer
        end_idx = start_index + len(t)
        output_buffer[start_index:end_idx, 0] = x
        output_buffer[start_index:end_idx, 1] = y
        output_buffer[start_index:end_idx, 2] = types

        return len(t)

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
        # Input normalization: ensure all points are 3D
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(len(points), dtype=np.float64)])

        # Pre-allocation: estimate final size
        num_curves = sum(1 for cmd in commands if cmd in "QC")
        estimated_points = len(points) + num_curves * steps
        new_points_array = np.empty((estimated_points, 3), dtype=np.float64)
        new_commands_list = []

        point_index = 0
        array_index = 0

        for cmd in commands:
            if cmd == "M":  # MoveTo - uses 1 point
                if point_index >= len(points):
                    raise ValueError(f"MoveTo command needs 1 point, got {len(points) - point_index}")

                pt = points[point_index]
                new_points_array[array_index] = pt
                new_commands_list.append(cmd)
                array_index += 1
                point_index += 1

            elif cmd == "L":  # LineTo - uses 1 point
                if point_index >= len(points):
                    raise ValueError(f"LineTo command needs 1 point, got {len(points) - point_index}")

                pt = points[point_index]
                new_points_array[array_index] = pt
                new_commands_list.append(cmd)
                array_index += 1
                point_index += 1

            elif cmd == "Q":  # Quadratic Bezier To - uses 2 points (control, end)
                if point_index + 1 >= len(points):
                    raise ValueError(f"Quadratic Bezier command needs 2 points, got {len(points) - point_index}")

                if array_index == 0:
                    raise ValueError("Quadratic Bezier command has no starting point")

                # Get start point (last point in new_points_array) + control and end points
                start_point = new_points_array[array_index - 1][:2]  # Get x,y from last point
                control_point = points[point_index][:2]
                end_point = points[point_index + 1][:2]

                control_points = np.array([start_point, control_point, end_point], dtype=np.float64)

                # Polygonize the quadratic bezier directly into output buffer
                num_curve_points = BezierCurve.polygonize_quadratic_curve_inplace(
                    control_points, steps, new_points_array, array_index, skip_first=True
                )
                new_commands_list.extend(["L"] * num_curve_points)
                array_index += num_curve_points
                point_index += 2  # Skip control and end points

            elif cmd == "C":  # Cubic Bezier To - uses 3 points (control1, control2, end)
                if point_index + 2 >= len(points):
                    raise ValueError(f"Cubic Bezier command needs 3 points, got {len(points) - point_index}")

                if array_index == 0:
                    raise ValueError("Cubic Bezier command has no starting point")

                # Get start point (last point in new_points_array) + control1, control2, and end points
                start_point = new_points_array[array_index - 1][:2]  # Get x,y from last point
                control1_point = points[point_index][:2]
                control2_point = points[point_index + 1][:2]
                end_point = points[point_index + 2][:2]

                control_points = np.array([start_point, control1_point, control2_point, end_point], dtype=np.float64)

                # Polygonize the cubic bezier directly into output buffer
                num_curve_points = BezierCurve.polygonize_cubic_curve_inplace(
                    control_points, steps, new_points_array, array_index, skip_first=True
                )
                new_commands_list.extend(["L"] * num_curve_points)
                array_index += num_curve_points
                point_index += 3  # Skip control1, control2, and end points

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

    def __eq__(self, other):
        """Check if two AvBox instances are equal."""
        if not isinstance(other, AvBox):
            return False
        return (
            self._xmin == other._xmin
            and self._ymin == other._ymin
            and self._xmax == other._xmax
            and self._ymax == other._ymax
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
            [10.0, 10.0, 0.0],  # M - MoveTo destination point (type 0.0)
            [20.0, 20.0, 0.0],  # Q - Quadratic Bezier control point (type 0.0)
            [30.0, 10.0, 0.0],  # Q - Quadratic Bezier end point (type 0.0)
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
            [10.0, 10.0, 0.0],  # M - MoveTo destination point (type 0.0)
            [10.0, 30.0, 0.0],  # C - Cubic Bezier control point 1 (type 0.0)
            [30.0, 30.0, 0.0],  # C - Cubic Bezier control point 2 (type 0.0)
            [30.0, 10.0, 0.0],  # C - Cubic Bezier end point (type 0.0)
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

    #
    #
    #
    #
    #
    #
    #
    ###########################################################################
    ###########################################################################
    # Test the corrected polygonize_path method with both examples

    polygonize_steps = 3

    print("\n" + "=" * 50)
    print("=" * 50)
    print("=" * 50)
    print("=" * 50)
    print("Testing polygonize_path method:")
    print("=" * 50)

    ###########################################################################
    # Test M,Q example
    print("\nTesting M,Q example:")
    new_points_q, new_commands_q = BezierCurve.polygonize_path(path_points, path_commands, polygonize_steps)

    print(f"Original points: {len(path_points)} points, {len(path_commands)} commands")
    for i, point in enumerate(path_points):
        if len(point) >= 3:
            print(f"  {i:2d}: M({point[0]:6.1f}, {point[1]:6.1f}, type={point[2]:1.0f})")
        else:
            print(f"  {i:2d}: M({point[0]:6.1f}, {point[1]:6.1f})")
    print(f"Original commands: {path_commands}")

    print(f"Polygonized: {len(new_points_q)} points, {len(new_commands_q)} commands")
    print("Polygonized points:")
    for i, (point, cmd) in enumerate(zip(new_points_q, new_commands_q)):
        if len(point) >= 3:
            print(f"  {i:2d}: {cmd} ({point[0]:6.1f}, {point[1]:6.1f}, type={point[2]:1.0f})")
        else:
            print(f"  {i:2d}: {cmd} ({point[0]:6.1f}, {point[1]:6.1f})")

    ###########################################################################
    # Test M,C example
    print("\nTesting M,C example:")
    new_points_c, new_commands_c = BezierCurve.polygonize_path(path_points_mc, path_commands_mc, polygonize_steps)

    print(f"Original points: {len(path_points_mc)} points, {len(path_commands_mc)} commands")
    for i, point in enumerate(path_points_mc):
        if len(point) >= 3:
            print(f"  {i:2d}: M({point[0]:6.1f}, {point[1]:6.1f}, type={point[2]:1.0f})")
        else:
            print(f"  {i:2d}: M({point[0]:6.1f}, {point[1]:6.1f})")
    print(f"Original commands: {path_commands_mc}")

    print(f"Polygonized: {len(new_points_c)} points, {len(new_commands_c)} commands")
    print("Polygonized points:")
    for i, (point, cmd) in enumerate(zip(new_points_c, new_commands_c)):
        if cmd == "Z":
            print(f"  {i:2d}: {cmd} (close path - no point data)")
        else:
            if len(point) >= 3:
                print(f"  {i:2d}: {cmd} ({point[0]:6.1f}, {point[1]:6.1f}, type={point[2]:1.0f})")
            else:
                print(f"  {i:2d}: {cmd} ({point[0]:6.1f}, {point[1]:6.1f})")

    #
    #
    #
    #
    #
    #
    #
    ###########################################################################
    # Complex path example with multiple command types
    print("\n" + "=" * 50)
    print("=" * 50)
    print("=" * 50)
    print("=" * 50)
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
    for i, point in enumerate(complex_path_points):
        if len(point) >= 3:
            print(f"    {i:2d}: {point[0]:6.1f}, {point[1]:6.1f}, type={point[2]:1.0f}")
        else:
            print(f"    {i:2d}: {point[0]:6.1f}, {point[1]:6.1f}")
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
            if len(point) >= 3:
                print(f"  {i:2d}: {cmd} ({point[0]:6.1f}, {point[1]:6.1f}, type={point[2]:1.0f})")
            else:
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
