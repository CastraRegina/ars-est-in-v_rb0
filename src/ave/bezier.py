"""Bezier curve handling utilities for SVG path processing and geometry operations."""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Pre-allocated constant for quadratic control point type values
_APPROX_QUADRATIC_TYPES: NDArray[np.float64] = np.array([0.0, 2.0, 0.0], dtype=np.float64)
_APPROX_QUADRATIC_ERROR_EPS: float = 1.0e-14

# Pre-allocated constant for cubic control point type values
_APPROX_CUBIC_TYPES: NDArray[np.float64] = np.array([0.0, 3.0, 3.0, 0.0], dtype=np.float64)
_APPROX_CUBIC_ERROR_EPS: float = 1.0e-14


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
        if isinstance(points, np.ndarray) and points.dtype == np.float64:
            points_array = points
        else:
            points_array = np.asarray(points, dtype=np.float64)
        if points_array.ndim != 2 or points_array.shape[1] < 2:
            raise ValueError("Quadratic control approximation requires (x, y) formatted points.")

        num_points = points_array.shape[0]
        if num_points < 3:
            raise ValueError("At least three points are required to approximate a quadratic curve.")

        xy_points = points_array[:, :2]
        start_x, start_y = xy_points[0, 0], xy_points[0, 1]
        end_x, end_y = xy_points[-1, 0], xy_points[-1, 1]

        best_ctrl_x: float = 0.0
        best_ctrl_y: float = 0.0
        best_error = float("inf")
        have_solution = False

        # Uniform parameterization (most common case)
        inv_span = 1.0 / float(num_points - 1)
        params_uniform = np.arange(num_points, dtype=np.float64) * inv_span
        ctrl_result = cls._approximate_quadratic_solve_control(
            params_uniform, xy_points, start_x, start_y, end_x, end_y
        )
        if ctrl_result is not None:
            best_ctrl_x, best_ctrl_y = ctrl_result
            have_solution = True

            # For small inputs, skip error computation and chord-length parameterization
            if num_points <= 10:
                result = np.empty((3, 3), dtype=np.float64)
                result[0, 0] = start_x
                result[0, 1] = start_y
                result[1, 0] = best_ctrl_x
                result[1, 1] = best_ctrl_y
                result[2, 0] = end_x
                result[2, 1] = end_y
                result[:, 2] = _APPROX_QUADRATIC_TYPES
                return result

            best_error = cls._approximate_quadratic_evaluate_error(
                params_uniform, xy_points, start_x, start_y, end_x, end_y, best_ctrl_x, best_ctrl_y
            )

            # Near-exact fits cannot be improved by re-parameterization
            if best_error <= _APPROX_QUADRATIC_ERROR_EPS:
                result = np.empty((3, 3), dtype=np.float64)
                result[0, 0] = start_x
                result[0, 1] = start_y
                result[1, 0] = best_ctrl_x
                result[1, 1] = best_ctrl_y
                result[2, 0] = end_x
                result[2, 1] = end_y
                result[:, 2] = _APPROX_QUADRATIC_TYPES
                return result

        # Chord-length parameterization for larger inputs
        if num_points > 10:
            deltas = np.diff(xy_points, axis=0)
            diffs = np.hypot(deltas[:, 0], deltas[:, 1])
            total_length = float(np.sum(diffs))
            if total_length > 0.0 and math.isfinite(total_length):
                cumulative = np.empty(num_points, dtype=np.float64)
                cumulative[0] = 0.0
                cumulative[1:] = np.cumsum(diffs) / total_length

                ctrl_chord = cls._approximate_quadratic_solve_control(
                    cumulative, xy_points, start_x, start_y, end_x, end_y
                )
                if ctrl_chord is not None:
                    chord_ctrl_x, chord_ctrl_y = ctrl_chord
                    error_chord = cls._approximate_quadratic_evaluate_error(
                        cumulative, xy_points, start_x, start_y, end_x, end_y, chord_ctrl_x, chord_ctrl_y
                    )
                    if error_chord < best_error:
                        best_ctrl_x, best_ctrl_y = chord_ctrl_x, chord_ctrl_y
                        best_error = error_chord
                        have_solution = True

        if not have_solution:
            raise ValueError("Unable to approximate quadratic control point from provided samples.")

        result = np.empty((3, 3), dtype=np.float64)
        result[0, 0] = start_x
        result[0, 1] = start_y
        result[1, 0] = best_ctrl_x
        result[1, 1] = best_ctrl_y
        result[2, 0] = end_x
        result[2, 1] = end_y
        result[:, 2] = _APPROX_QUADRATIC_TYPES
        return result

    @staticmethod
    def _approximate_quadratic_solve_control(
        params: NDArray[np.float64],
        xy_points: NDArray[np.float64],
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> Union[Tuple[float, float], None]:
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
        if denominator <= 0.0 or not math.isfinite(denominator):
            return None

        interior_xy = xy_points[1:-1]
        omt2 = omt * omt
        t2 = t_values * t_values

        base_x = omt2 * start_x + t2 * end_x
        base_y = omt2 * start_y + t2 * end_y
        residual_x = interior_xy[:, 0] - base_x
        residual_y = interior_xy[:, 1] - base_y

        ctrl_x = float(np.dot(weights, residual_x)) / denominator
        ctrl_y = float(np.dot(weights, residual_y)) / denominator

        if not (math.isfinite(ctrl_x) and math.isfinite(ctrl_y)):
            return None
        return (ctrl_x, ctrl_y)

    @staticmethod
    def _approximate_quadratic_evaluate_error(
        params: NDArray[np.float64],
        xy_points: NDArray[np.float64],
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        ctrl_x: float,
        ctrl_y: float,
    ) -> float:
        """Optimized error evaluation for given parameterization and control point."""
        omt = 1.0 - params
        omt2 = omt * omt
        t2 = params * params
        two_omt_t = 2.0 * omt * params

        bx = omt2 * start_x + two_omt_t * ctrl_x + t2 * end_x
        by = omt2 * start_y + two_omt_t * ctrl_y + t2 * end_y

        rx = xy_points[:, 0] - bx
        ry = xy_points[:, 1] - by
        return float(np.dot(rx, rx) + np.dot(ry, ry))

    @classmethod
    def approximate_cubic_control_points(
        cls, points: Union[Sequence[Tuple[float, float]], NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """Approximate cubic Bezier control points from sampled curve points.

        Uses a least-squares fit assuming the first and last points are the start and
        end of the desired cubic Bezier curve. The remaining points are used to
        compute the two cubic control points that best fit the samples.

        Args:
            points: Sampled points along the curve as (x, y) or (x, y, type) tuples.

        Returns:
            Array of shape (4, 3) containing the start point, fitted control points,
            and end point, each encoded as (x, y, type).

        Raises:
            ValueError: If fewer than four points are provided or if the input does
                not contain at least two coordinate columns.
        """
        if isinstance(points, np.ndarray) and points.dtype == np.float64:
            points_array = points
        else:
            points_array = np.asarray(points, dtype=np.float64)
        if points_array.ndim != 2 or points_array.shape[1] < 2:
            raise ValueError("Cubic control approximation requires (x, y) formatted points.")

        num_points = points_array.shape[0]
        if num_points < 4:
            raise ValueError("At least four points are required to approximate a cubic curve.")

        xy_points = points_array[:, :2]
        start_x, start_y = xy_points[0, 0], xy_points[0, 1]
        end_x, end_y = xy_points[-1, 0], xy_points[-1, 1]

        best_ctrl1_x: float = 0.0
        best_ctrl1_y: float = 0.0
        best_ctrl2_x: float = 0.0
        best_ctrl2_y: float = 0.0
        best_error = float("inf")
        have_solution = False

        inv_span = 1.0 / float(num_points - 1)
        params_uniform = np.arange(num_points, dtype=np.float64) * inv_span
        ctrl_result = cls._approximate_cubic_solve_controls(params_uniform, xy_points, start_x, start_y, end_x, end_y)
        if ctrl_result is not None:
            best_ctrl1_x, best_ctrl1_y, best_ctrl2_x, best_ctrl2_y = ctrl_result
            have_solution = True

            if num_points <= 10:
                result = np.empty((4, 3), dtype=np.float64)
                result[0, 0] = start_x
                result[0, 1] = start_y
                result[1, 0] = best_ctrl1_x
                result[1, 1] = best_ctrl1_y
                result[2, 0] = best_ctrl2_x
                result[2, 1] = best_ctrl2_y
                result[3, 0] = end_x
                result[3, 1] = end_y
                result[:, 2] = _APPROX_CUBIC_TYPES
                return result

            best_error = cls._approximate_cubic_evaluate_error(
                params_uniform,
                xy_points,
                start_x,
                start_y,
                end_x,
                end_y,
                best_ctrl1_x,
                best_ctrl1_y,
                best_ctrl2_x,
                best_ctrl2_y,
            )

            if best_error <= _APPROX_CUBIC_ERROR_EPS:
                result = np.empty((4, 3), dtype=np.float64)
                result[0, 0] = start_x
                result[0, 1] = start_y
                result[1, 0] = best_ctrl1_x
                result[1, 1] = best_ctrl1_y
                result[2, 0] = best_ctrl2_x
                result[2, 1] = best_ctrl2_y
                result[3, 0] = end_x
                result[3, 1] = end_y
                result[:, 2] = _APPROX_CUBIC_TYPES
                return result

        if num_points > 10:
            deltas = np.diff(xy_points, axis=0)
            diffs = np.hypot(deltas[:, 0], deltas[:, 1])
            total_length = float(np.sum(diffs))
            if total_length > 0.0 and math.isfinite(total_length):
                cumulative = np.empty(num_points, dtype=np.float64)
                cumulative[0] = 0.0
                cumulative[1:] = np.cumsum(diffs) / total_length

                ctrl_chord = cls._approximate_cubic_solve_controls(
                    cumulative, xy_points, start_x, start_y, end_x, end_y
                )
                if ctrl_chord is not None:
                    chord_ctrl1_x, chord_ctrl1_y, chord_ctrl2_x, chord_ctrl2_y = ctrl_chord
                    error_chord = cls._approximate_cubic_evaluate_error(
                        cumulative,
                        xy_points,
                        start_x,
                        start_y,
                        end_x,
                        end_y,
                        chord_ctrl1_x,
                        chord_ctrl1_y,
                        chord_ctrl2_x,
                        chord_ctrl2_y,
                    )
                    if error_chord < best_error:
                        best_ctrl1_x, best_ctrl1_y = chord_ctrl1_x, chord_ctrl1_y
                        best_ctrl2_x, best_ctrl2_y = chord_ctrl2_x, chord_ctrl2_y
                        best_error = error_chord
                        have_solution = True

        if not have_solution:
            raise ValueError("Unable to approximate cubic control points from provided samples.")

        result = np.empty((4, 3), dtype=np.float64)
        result[0, 0] = start_x
        result[0, 1] = start_y
        result[1, 0] = best_ctrl1_x
        result[1, 1] = best_ctrl1_y
        result[2, 0] = best_ctrl2_x
        result[2, 1] = best_ctrl2_y
        result[3, 0] = end_x
        result[3, 1] = end_y
        result[:, 2] = _APPROX_CUBIC_TYPES
        return result

    @staticmethod
    def _approximate_cubic_solve_controls(
        params: NDArray[np.float64],
        xy_points: NDArray[np.float64],
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> Union[Tuple[float, float, float, float], None]:
        """Optimized cubic control point solver for given parameterization."""
        if len(params) < 4:
            return None

        t_values = params[1:-1]
        if len(t_values) < 2:
            return None

        omt = 1.0 - t_values
        omt2 = omt * omt
        t2 = t_values * t_values

        w1 = 3.0 * omt2 * t_values
        w2 = 3.0 * omt * t2

        s11 = float(np.dot(w1, w1))
        s12 = float(np.dot(w1, w2))
        s22 = float(np.dot(w2, w2))

        det = s11 * s22 - s12 * s12
        if det <= 0.0 or not math.isfinite(det):
            return None

        interior_xy = xy_points[1:-1]
        omt3 = omt2 * omt
        t3 = t2 * t_values
        base_x = omt3 * start_x + t3 * end_x
        base_y = omt3 * start_y + t3 * end_y

        residual_x = interior_xy[:, 0] - base_x
        residual_y = interior_xy[:, 1] - base_y

        r1x = float(np.dot(w1, residual_x))
        r2x = float(np.dot(w2, residual_x))
        r1y = float(np.dot(w1, residual_y))
        r2y = float(np.dot(w2, residual_y))

        inv_det = 1.0 / det
        ctrl1_x = (r1x * s22 - r2x * s12) * inv_det
        ctrl2_x = (r2x * s11 - r1x * s12) * inv_det
        ctrl1_y = (r1y * s22 - r2y * s12) * inv_det
        ctrl2_y = (r2y * s11 - r1y * s12) * inv_det

        if not (
            math.isfinite(ctrl1_x) and math.isfinite(ctrl1_y) and math.isfinite(ctrl2_x) and math.isfinite(ctrl2_y)
        ):
            return None

        return (ctrl1_x, ctrl1_y, ctrl2_x, ctrl2_y)

    @staticmethod
    def _approximate_cubic_evaluate_error(
        params: NDArray[np.float64],
        xy_points: NDArray[np.float64],
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        ctrl1_x: float,
        ctrl1_y: float,
        ctrl2_x: float,
        ctrl2_y: float,
    ) -> float:
        """Optimized cubic error evaluation for given parameterization and control points."""
        omt = 1.0 - params
        t2 = params * params
        omt2 = omt * omt

        w0 = omt2 * omt
        w1 = 3.0 * omt2 * params
        w2 = 3.0 * omt * t2
        w3 = t2 * params

        bx = w0 * start_x + w1 * ctrl1_x + w2 * ctrl2_x + w3 * end_x
        by = w0 * start_y + w1 * ctrl1_y + w2 * ctrl2_y + w3 * end_y

        rx = xy_points[:, 0] - bx
        ry = xy_points[:, 1] - by
        return float(np.dot(rx, rx) + np.dot(ry, ry))
