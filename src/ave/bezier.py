"""Bezier curve handling utilities for SVG path processing and geometry operations."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class BezierCurve:
    """Class to handle quadratic and cubic Bezier curve operations.

    Provides methods for polygonizing Bezier curves into point sequences,
    supporting both pure Python and NumPy-optimized implementations.
    """

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
        # First differences: delta_B = B(h) - B(0)
        dx_first = b1_x - b0_x
        dy_first = b1_y - b0_y

        # Second differences: delta2_B = B(2h) - 2*B(h) + B(0)
        dx_second = b2_x - 2.0 * b1_x + b0_x
        dy_second = b2_y - 2.0 * b1_y + b0_y

        # Third differences: delta3_B = B(3h) - 3*B(2h) + 3*B(h) - B(0) (constant for cubic)
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
    def polygonize_quadratic_curve_python_inplace(
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        cls,
        points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]],
        steps: int,
        output_buffer: NDArray[np.float64],
        start_index: int = 0,
        skip_first: bool = False,
    ) -> int:
        """Polygonize a quadratic Bezier curve directly into pre-allocated buffer using pure Python.

        Optimized using forward differencing for O(1) per point computation.

        Args:
            points: Control points as sequence of (x, y) tuples or array
            steps: Number of interpolation steps
            output_buffer: Pre-allocated buffer for output points (shape: N, 3)
            start_index: Starting index in output buffer
            skip_first: Whether to skip the first point

        Returns:
            Number of points written to buffer

        Raises:
            ValueError: If points length is not 3 or steps is invalid
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
    def approximate_quadratic_control_points(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """Approximate quadratic Bezier control points from sampled curve points.

        Uses a least-squares fit assuming the first and last points are the start and
        end of the desired quadratic Bezier curve. The remaining points are used to
        compute the single quadratic control point that best fits the samples.

        Args:
            points: Sampled points along the curve as (x, y) or (x, y, type) tuples.

        Returns:
            Array of shape (3, 3) containing the start point, fitted control point,
            and end point, each encoded as (x, y, type).

        Raises:
            ValueError: If fewer than three points are provided or if the input does
                not contain at least two coordinate columns.
        """
        points_array = np.asarray(points, dtype=np.float64)
        if points_array.ndim != 2 or points_array.shape[1] < 2:
            raise ValueError("Quadratic control approximation requires (x, y) formatted points.")

        num_points = len(points_array)
        if num_points < 3:
            raise ValueError("At least three points are required to approximate a quadratic curve.")

        xy_points = points_array[:, :2]
        start_point = xy_points[0]
        end_point = xy_points[-1]

        best_control = None
        best_error = np.inf

        # Uniform parameterization (most common case)
        if num_points == 1:
            inv_span = 0.0
        else:
            inv_span = 1.0 / float(num_points - 1)
        params_uniform = np.arange(num_points, dtype=np.float64) * inv_span
        control_uniform = cls._solve_control_optimized(params_uniform, xy_points, start_point, end_point)
        if control_uniform is not None:
            best_control = control_uniform

            # For small inputs, the extra error computation is overhead and chord-length
            # parameterization is not attempted anyway.
            if num_points <= 10:
                result = np.empty((3, 3), dtype=np.float64)
                result[0, :2] = start_point
                result[1, :2] = best_control
                result[2, :2] = end_point
                result[:, 2] = np.array([0.0, 2.0, 0.0], dtype=np.float64)
                return result

            error_uniform = cls._evaluate_error_optimized(
                params_uniform, xy_points, start_point, end_point, control_uniform
            )
            best_error = error_uniform

            # Exact fits cannot be improved by re-parameterization.
            if best_error == 0.0:
                result = np.empty((3, 3), dtype=np.float64)
                result[0, :2] = start_point
                result[1, :2] = best_control
                result[2, :2] = end_point
                result[:, 2] = np.array([0.0, 2.0, 0.0], dtype=np.float64)
                return result

        # Chord-length parameterization only if uniform sampling is not sufficient
        # or if points appear irregularly spaced
        if num_points > 10:
            deltas = np.diff(xy_points, axis=0)
            diffs = np.hypot(deltas[:, 0], deltas[:, 1])
            total_length = np.sum(diffs)
            if total_length > 0.0 and np.isfinite(total_length):
                cumulative = np.empty(num_points, dtype=np.float64)
                cumulative[0] = 0.0
                cumulative[1:] = np.cumsum(diffs) / total_length

                control_chord = cls._solve_control_optimized(cumulative, xy_points, start_point, end_point)
                if control_chord is not None:
                    error_chord = cls._evaluate_error_optimized(
                        cumulative, xy_points, start_point, end_point, control_chord
                    )
                    if error_chord < best_error:
                        best_control = control_chord
                        best_error = error_chord

        if best_control is None:
            raise ValueError("Unable to approximate quadratic control point from provided samples.")

        result = np.empty((3, 3), dtype=np.float64)
        result[0, :2] = start_point
        result[1, :2] = best_control
        result[2, :2] = end_point
        result[:, 2] = np.array([0.0, 2.0, 0.0], dtype=np.float64)
        return result

    @staticmethod
    def _solve_control_optimized(
        params: NDArray[np.float64],
        xy_points: NDArray[np.float64],
        start_point: NDArray[np.float64],
        end_point: NDArray[np.float64],
    ) -> Union[NDArray[np.float64], None]:
        """Optimized control point solver for given parameterization."""
        if len(params) < 3:
            return None

        # Params are expected to cover [0, 1] with endpoints included.
        t_values = params[1:-1]
        if len(t_values) == 0:
            return None

        omt = 1.0 - t_values
        weights = 2.0 * omt * t_values

        denominator = float(np.dot(weights, weights))
        if denominator <= 0.0 or not np.isfinite(denominator):
            return None

        interior_xy = xy_points[1:-1]
        omt2 = omt * omt
        t2 = t_values * t_values

        base_x = omt2 * float(start_point[0]) + t2 * float(end_point[0])
        base_y = omt2 * float(start_point[1]) + t2 * float(end_point[1])
        residual_x = interior_xy[:, 0] - base_x
        residual_y = interior_xy[:, 1] - base_y

        numerator_x = float(np.dot(weights, residual_x))
        numerator_y = float(np.dot(weights, residual_y))
        control_candidate = np.array([numerator_x / denominator, numerator_y / denominator], dtype=np.float64)

        if not np.all(np.isfinite(control_candidate)):
            return None
        return control_candidate

    @staticmethod
    def _evaluate_error_optimized(
        params: NDArray[np.float64],
        xy_points: NDArray[np.float64],
        start_point: NDArray[np.float64],
        end_point: NDArray[np.float64],
        control_point: NDArray[np.float64],
    ) -> float:
        """Optimized error evaluation for given parameterization and control point."""
        omt = 1.0 - params
        omt2 = omt * omt
        t2 = params * params

        bx = omt2 * float(start_point[0])
        bx += 2.0 * omt * params * float(control_point[0])
        bx += t2 * float(end_point[0])

        by = omt2 * float(start_point[1])
        by += 2.0 * omt * params * float(control_point[1])
        by += t2 * float(end_point[1])

        rx = xy_points[:, 0] - bx
        ry = xy_points[:, 1] - by
        return float(np.dot(rx, rx) + np.dot(ry, ry))
