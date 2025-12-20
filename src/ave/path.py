"""SVG path handling and geometric operations for vector graphics processing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ave.bezier import BezierCurve
from ave.common import AvGlyphCmds
from ave.geom import AvBox

###############################################################################
# PathConstraints
###############################################################################


@dataclass(frozen=True)
class PathConstraints:
    """Constraints defining valid path structures.

    Attributes:
        allows_curves: If False, only M, L, Z commands are allowed (no Q, C).
        max_segments: Maximum number of segments (M commands). None means unlimited.
        must_close: If True, each segment must end with Z command.
        min_points_per_segment: Minimum points required per segment. None means no minimum.
    """

    allows_curves: bool = True
    max_segments: Optional[int] = None
    must_close: bool = False
    min_points_per_segment: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert constraints to a dictionary for serialization."""
        return {
            "allows_curves": self.allows_curves,
            "max_segments": self.max_segments,
            "must_close": self.must_close,
            "min_points_per_segment": self.min_points_per_segment,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PathConstraints":
        """Create PathConstraints from a dictionary."""
        return cls(
            allows_curves=data.get("allows_curves", True),
            max_segments=data.get("max_segments"),
            must_close=data.get("must_close", False),
            min_points_per_segment=data.get("min_points_per_segment"),
        )


# Constraint constants for different path types
GENERAL_CONSTRAINTS = PathConstraints()

MULTI_POLYLINE_CONSTRAINTS = PathConstraints(
    allows_curves=False,
    max_segments=None,
    must_close=False,
    min_points_per_segment=None,
)

SINGLE_PATH_CONSTRAINTS = PathConstraints(
    allows_curves=True,
    max_segments=1,
    must_close=False,
    min_points_per_segment=None,
)

CLOSED_SINGLE_PATH_CONSTRAINTS = PathConstraints(
    allows_curves=True,
    max_segments=1,
    must_close=True,
    min_points_per_segment=None,
)

SINGLE_POLYGON_CONSTRAINTS = PathConstraints(
    allows_curves=False,
    max_segments=1,
    must_close=True,
    min_points_per_segment=3,
)

MULTI_POLYGON_CONSTRAINTS = PathConstraints(
    allows_curves=False,
    max_segments=None,
    must_close=True,
    min_points_per_segment=3,
)


###############################################################################
# PathValidator
###############################################################################


class PathValidator:
    """Validates path structure against constraints."""

    @staticmethod
    def validate(
        commands: List[AvGlyphCmds],
        points: NDArray[np.float64],
        constraints: PathConstraints,
    ) -> None:
        """Validate path against constraints.

        Args:
            commands: List of path commands.
            points: Array of path points.
            constraints: Constraints to validate against.

        Raises:
            ValueError: If path violates any constraint.
        """
        if not commands:
            if points.shape[0] != 0:
                raise ValueError("Empty command list must have zero points")
            return

        # Validate command types
        if not constraints.allows_curves:
            for i, cmd in enumerate(commands):
                if cmd not in ("M", "L", "Z"):
                    raise ValueError(f"Path cannot contain curve commands (found '{cmd}' at position {i})")

        # Count segments and validate structure
        segments = AvPathUtils.split_into_segments(commands)

        # Validate segment count
        if constraints.max_segments is not None:
            if len(segments) > constraints.max_segments:
                raise ValueError(f"Path has {len(segments)} segments but max is {constraints.max_segments}")

        # Validate each segment
        for seg_idx, (seg_cmds, seg_point_count) in enumerate(segments):
            # Check closure
            if constraints.must_close:
                if not seg_cmds or seg_cmds[-1] != "Z":
                    raise ValueError(f"Segment {seg_idx} must end with 'Z' command")

            # Check minimum points
            if constraints.min_points_per_segment is not None:
                if seg_point_count < constraints.min_points_per_segment:
                    raise ValueError(
                        f"Segment {seg_idx} has {seg_point_count} points "
                        f"but minimum is {constraints.min_points_per_segment}"
                    )


###############################################################################
# AvPathUtils
###############################################################################


class AvPathUtils:
    """Collection of static utility functions for path operations."""

    @staticmethod
    def split_into_segments(commands: List[AvGlyphCmds]) -> List[Tuple[List[AvGlyphCmds], int]]:
        """Split commands into segments, returning list of (commands, point_count) tuples."""
        if not commands:
            return []

        segments: List[Tuple[List[AvGlyphCmds], int]] = []
        current_cmds: List[AvGlyphCmds] = []
        current_point_count = 0

        for cmd in commands:
            if cmd == "M":
                if current_cmds:
                    segments.append((current_cmds, current_point_count))
                current_cmds = ["M"]
                current_point_count = 1
            elif cmd == "L":
                current_cmds.append("L")
                current_point_count += 1
            elif cmd == "Q":
                current_cmds.append("Q")
                current_point_count += 2
            elif cmd == "C":
                current_cmds.append("C")
                current_point_count += 3
            elif cmd == "Z":
                current_cmds.append("Z")

        if current_cmds:
            segments.append((current_cmds, current_point_count))

        return segments


###############################################################################
# AvPath
###############################################################################


@dataclass
class AvPath:
    """SVG path represented by points and corresponding commands.

    A path contains 0..n segments; each segment starts with M, is followed by
    an arbitrary mix of L/Q/C, and may optionally end with Z.
    A path may also be empty (no points and no commands), representing 0 segments.

    Attributes:
        _points: Array of 3D points (shape: n_points, 3)
        _commands: List of SVG commands for each point
        _bounding_box: Cached bounding box for performance
        _polygonized_path: Cached polygonized path for performance
    """

    _points: NDArray[np.float64]  # shape (n_points, 3)
    _commands: List[AvGlyphCmds]  # shape (n_commands, 1)
    _constraints: PathConstraints = GENERAL_CONSTRAINTS  # path constraints
    _bounding_box: Optional[AvBox] = None  # caching variable
    _polygonized_path: Optional[AvMultiPolylinePath] = None  # caching variable

    # Number of steps to use when polygonizing curves for internal approximations
    POLYGONIZE_STEPS_INTERNAL: int = 50  # pylint: disable=invalid-name

    def __init__(
        self,
        points: Optional[
            Union[
                Sequence[Tuple[float, float]],
                Sequence[Tuple[float, float, float]],
                NDArray[np.float64],
            ]
        ] = None,
        commands: Optional[List[AvGlyphCmds]] = None,
        constraints: Optional[PathConstraints] = None,
    ):
        """
        Initialize an AvPath from 2D points.

        Args:
            points: a sequence of (x, y) or (x, y, type).
            commands: List of drawing commands corresponding to the points.
        """
        if points is None:
            arr = np.empty((0, 3), dtype=np.float64)
        elif isinstance(points, np.ndarray):
            arr = points.astype(np.float64, copy=False)
        else:
            arr = np.asarray(points, dtype=np.float64)

        if arr.ndim != 2:
            raise ValueError(f"points must have 2 dimensions, got {arr.ndim}")

        commands_list = [] if commands is None else list(commands)

        if arr.shape[1] == 2:
            # Generate type column based on commands
            type_column = np.zeros(arr.shape[0], dtype=np.float64)
            point_idx = 0

            for cmd in commands_list:
                if cmd == "M":  # MoveTo - 1 point
                    if point_idx >= arr.shape[0]:
                        raise ValueError(f"Not enough points for MoveTo command at index {point_idx}")
                    type_column[point_idx] = 0.0
                    point_idx += 1
                elif cmd == "L":  # LineTo - 1 point
                    if point_idx >= arr.shape[0]:
                        raise ValueError(f"Not enough points for LineTo command at index {point_idx}")
                    type_column[point_idx] = 0.0
                    point_idx += 1
                elif cmd == "Q":  # Quadratic Bezier - 2 points (control + end)
                    if point_idx + 1 >= arr.shape[0]:
                        raise ValueError(f"Not enough points for Quadratic Bezier command at index {point_idx}")
                    type_column[point_idx] = 2.0  # control point
                    type_column[point_idx + 1] = 0.0  # end point
                    point_idx += 2
                elif cmd == "C":  # Cubic Bezier - 3 points (control1 + control2 + end)
                    if point_idx + 2 >= arr.shape[0]:
                        raise ValueError(f"Not enough points for Cubic Bezier command at index {point_idx}")
                    type_column[point_idx] = 3.0  # control point 1
                    type_column[point_idx + 1] = 3.0  # control point 2
                    type_column[point_idx + 2] = 0.0  # end point
                    point_idx += 3
                elif cmd == "Z":  # ClosePath - no points
                    pass  # No points consumed

            self._points = np.column_stack([arr, type_column])
        elif arr.shape[1] == 3:
            self._points = arr
        else:
            raise ValueError(f"points must have shape (n, 2) or (n, 3), got {arr.shape}")

        self._commands = commands_list
        self._constraints = constraints if constraints is not None else GENERAL_CONSTRAINTS
        self._bounding_box = None
        self._polygonized_path = None
        self._validate()

    def _validate(self) -> None:
        """Validate path structure using PathValidator.

        This method validates basic path structure and point/command consistency,
        then delegates constraint-specific validation to PathValidator.
        """
        cmds = self._commands
        points = self._points

        # Constraint-based validation (handles empty path check, curve allowance, segment limits)
        PathValidator.validate(cmds, points, self._constraints)

        # Nothing more to check for empty paths
        if not cmds:
            return

        # Use PathValidator's segment parsing to validate structure and point count
        segments = AvPathUtils.split_into_segments(cmds)

        # Validate that segments start with M and Z properly terminates segments
        idx = 0
        for seg_idx, (seg_cmds, _) in enumerate(segments):
            if not seg_cmds or seg_cmds[0] != "M":
                raise ValueError(
                    f"Each segment must start with 'M' command (segment {seg_idx} starts with '{seg_cmds[0] if seg_cmds else 'None'}')"
                )

            # Check Z termination within segment (except for last segment which may not end with Z)
            for cmd_idx, cmd in enumerate(seg_cmds):
                if cmd == "Z" and cmd_idx < len(seg_cmds) - 1:
                    raise ValueError(
                        f"'Z' must terminate a segment "
                        f"(found 'Z' at position {idx + cmd_idx} followed by '{seg_cmds[cmd_idx + 1]}')"
                    )

            idx += len(seg_cmds)

        # Validate total point count matches segments
        total_expected = sum(point_count for _, point_count in segments)
        if points.shape[0] != total_expected:
            raise ValueError(
                f"Number of points ({points.shape[0]}) does not match commands (requires {total_expected} points)"
            )

    @property
    def points(self) -> NDArray[np.float64]:
        """
        The points of this path as a numpy array of shape (n_points, 3).
        """
        return self._points

    @property
    def commands(self) -> List[AvGlyphCmds]:
        """
        The commands of this path as a list of SVG path commands.
        """
        return self._commands

    @property
    def constraints(self) -> PathConstraints:
        """
        The constraints defining valid path structures.
        """
        return self._constraints

    @property
    def is_polygon_like(self) -> bool:
        """
        Return True if this path has polygon-like constraints.

        Polygon-like paths support geometric operations like area, centroid, is_ccw.
        This includes single_polygon and multi_polygon constraint types.
        """
        return self._constraints.must_close and not self._constraints.allows_curves

    @property
    def is_closed(self) -> bool:
        """
        Return True if this path has closure constraints (must_close=True).
        """
        return self._constraints.must_close

    @property
    def is_single_segment(self) -> bool:
        """
        Return True if this path is constrained to a single segment.
        """
        return self._constraints.max_segments == 1

    @property
    def is_single_path(self) -> bool:
        """Return True if this path has SINGLE_PATH_CONSTRAINTS."""
        return self._constraints == SINGLE_PATH_CONSTRAINTS

    @property
    def is_closed_path(self) -> bool:
        """Return True if this path has CLOSED_SINGLE_PATH_CONSTRAINTS."""
        return self._constraints == CLOSED_SINGLE_PATH_CONSTRAINTS

    @property
    def is_polylines_path(self) -> bool:
        """Return True if this path has MULTI_POLYLINE_CONSTRAINTS."""
        return self._constraints == MULTI_POLYLINE_CONSTRAINTS

    @property
    def is_polygon_path(self) -> bool:
        """Return True if this path has SINGLE_POLYGON_CONSTRAINTS."""
        return self._constraints == SINGLE_POLYGON_CONSTRAINTS

    @property
    def is_multi_polygon_path(self) -> bool:
        """Return True if this path has MULTI_POLYGON_CONSTRAINTS."""
        return self._constraints == MULTI_POLYGON_CONSTRAINTS

    @cached_property
    def area(self) -> float:
        """
        Return the area of this path if it's polygon-like.

        For paths with curves, the path is first polygonized.
        For multi-segment paths, returns the sum of all segment areas.

        Returns:
            float: The area of the path. Returns 0.0 if path has fewer than 3 points.

        Raises:
            ValueError: If the path is not closed (must_close constraint required).
        """
        if not self._constraints.must_close:
            raise ValueError("Area calculation requires a closed path (must_close=True)")

        # For polygon-like paths, use direct calculation
        if self.is_polygon_like:
            return self._calculate_polygon_area()

        # For closed paths with curves, polygonize first
        polygonized = self.polygonized_path()
        return polygonized._calculate_polygon_area()

    def _calculate_polygon_area(self) -> float:
        """Calculate area using shoelace formula for a single polygon."""
        pts = self._points
        if pts.shape[0] < 3:
            return 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        cross_sum = cross.sum()
        if np.isclose(cross_sum, 0.0):
            return 0.0
        return float(0.5 * abs(cross_sum))

    @cached_property
    def centroid(self) -> Tuple[float, float]:
        """
        Return the centroid of this path if it's polygon-like.

        For paths with curves, the path is first polygonized.

        Returns:
            Tuple[float, float]: The x and y coordinates of the centroid.

        Raises:
            ValueError: If the path is not closed (must_close constraint required).
        """
        if not self._constraints.must_close:
            raise ValueError("Centroid calculation requires a closed path (must_close=True)")

        # For polygon-like paths, use direct calculation
        if self.is_polygon_like:
            return self._calculate_polygon_centroid()

        # For closed paths with curves, polygonize first
        polygonized = self.polygonized_path()
        return polygonized._calculate_polygon_centroid()

    def _calculate_polygon_centroid(self) -> Tuple[float, float]:
        """Calculate centroid for a polygon."""
        pts = self._points
        if pts.shape[0] == 0:
            return (0.0, 0.0)
        if pts.shape[0] < 3:
            x_mean = float(pts[:, 0].mean())
            y_mean = float(pts[:, 1].mean())
            return (x_mean, y_mean)
        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        cross_sum = cross.sum()
        if np.isclose(cross_sum, 0.0):
            x_mean = float(x.mean())
            y_mean = float(y.mean())
            return (x_mean, y_mean)
        factor = 1.0 / (3.0 * cross_sum)
        cx = float(((x + x_next) * cross).sum() * factor)
        cy = float(((y + y_next) * cross).sum() * factor)
        return (cx, cy)

    @cached_property
    def is_ccw(self) -> bool:
        """
        Return True if this path runs counter-clockwise.

        For paths with curves, the path is first polygonized.

        Returns:
            bool: True if counter-clockwise, False otherwise.

        Raises:
            ValueError: If the path is not closed (must_close constraint required).
        """
        if not self._constraints.must_close:
            raise ValueError("CCW check requires a closed path (must_close=True)")

        # For polygon-like paths, use direct calculation
        if self.is_polygon_like:
            return self._calculate_is_ccw()

        # For closed paths with curves, polygonize first
        polygonized = self.polygonized_path()
        return polygonized._calculate_is_ccw()

    def _calculate_is_ccw(self) -> bool:
        """Calculate if polygon vertices are ordered counter-clockwise."""
        pts = self._points
        if pts.shape[0] < 3:
            return False
        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        return bool(cross.sum() > 0.0)

    def reverse(self) -> AvPath:
        """Return a new AvPath with reversed drawing direction.

        The same point coordinates are used, but their order (and the
        corresponding commands) is reversed, so traversal runs from the
        original last point back to the original first point.

        For curve commands (Q, C), control points are reordered to preserve
        the exact curve geometry when traversed in reverse.

        For multi-segment paths, each segment is reversed individually.
        """
        # Early return for empty paths
        if not self._commands or self._points.size == 0:
            return AvPath(self._points.copy(), list(self._commands), self._constraints)

        # Check if this is a multi-segment path
        m_count = self._commands.count("M")
        if m_count > 1:
            # Multi-segment path: split, reverse each, join
            single_paths = self.split_into_single_paths()
            reversed_segments = [self._reverse_single_segment(seg) for seg in single_paths]
            return AvPath.join_paths(reversed_segments)

        # Single-segment path: reverse directly
        return self._reverse_single_segment(self)

    def _reverse_single_segment(self, path: AvSinglePath) -> AvSinglePath:
        """Reverse a single-segment path."""
        if not path.commands or path._points.size == 0:
            return AvSinglePath(path._points.copy(), list(path.commands), path._constraints)

        # Check if path is closed
        is_closed = path.commands[-1] == "Z"

        # Build segments by iterating forward once
        segments = []
        last_point = path._points[0].copy()  # Start with M point
        point_idx = 1  # Skip M's point

        for cmd in path.commands[1:]:
            if cmd == "Z":
                break

            start_point = last_point

            if cmd == "L":
                end_point = path._points[point_idx].copy()
                segments.append(("L", [], start_point, end_point))
                last_point = end_point
                point_idx += 1

            elif cmd == "Q":
                control = path._points[point_idx].copy()
                end_point = path._points[point_idx + 1].copy()
                segments.append(("Q", [control], start_point, end_point))
                last_point = end_point
                point_idx += 2

            elif cmd == "C":
                control1 = path._points[point_idx].copy()
                control2 = path._points[point_idx + 1].copy()
                end_point = path._points[point_idx + 2].copy()
                segments.append(("C", [control1, control2], start_point, end_point))
                last_point = end_point
                point_idx += 3

        # Build reversed path
        new_commands = ["M"]
        new_points = []

        if segments:
            # Start from last segment's end point
            new_points.append(segments[-1][3])  # Last end point

            # Process segments in reverse
            for cmd_type, controls, start_point, end_point in reversed(segments):
                if cmd_type == "L":
                    new_commands.append("L")
                    new_points.append(start_point)  # Original start becomes new end

                elif cmd_type == "Q":
                    new_commands.append("Q")
                    new_points.append(controls[0])  # Control point
                    new_points.append(start_point)  # Original start becomes new end

                elif cmd_type == "C":
                    new_commands.append("C")
                    new_points.append(controls[1])  # Swapped control points
                    new_points.append(controls[0])
                    new_points.append(start_point)  # Original start becomes new end
        else:
            # Only M command
            new_points.append(path._points[0].copy())

        # Add Z if original was closed
        if is_closed:
            new_commands.append("Z")

        # Convert to numpy array
        points_array = np.array(new_points, dtype=np.float64) if new_points else np.empty((0, 3), dtype=np.float64)

        return AvSinglePath(points_array, new_commands, path.constraints)

    @classmethod
    def make_closed_single(cls, path: "AvSinglePath") -> "AvClosedSinglePath":
        """Create a closed AvPath from an existing AvSinglePath, ensuring it's properly closed.

        Args:
            path: An AvSinglePath instance to convert to a closed path

        Returns:
            AvClosedSinglePath: A new closed path with proper Z command and no duplicate endpoints
        """
        # Handle empty path
        if path.points.size == 0 or not path.commands:
            return AvClosedSinglePath(constraints=CLOSED_SINGLE_PATH_CONSTRAINTS)

        # Copy points and commands to avoid modifying original
        points = path.points.copy()
        commands = list(path.commands)

        # Ensure path ends with Z command
        if commands[-1] != "Z":
            commands.append("Z")

        # Check if first and last points are duplicates (or very close)
        if len(points) > 0:
            first_point = points[0]
            last_point = points[-1]

            # Calculate distance between first and last points
            distance = np.linalg.norm(first_point[:2] - last_point[:2])

            # If points are very close (within a small tolerance), remove the duplicate
            tolerance = 1e-10
            if distance < tolerance:
                points = points[:-1]  # Remove last point
                # Also remove one command to maintain point/command ratio
                # Remove the command before Z (which should be the last command)
                if len(commands) > 1 and commands[-1] == "Z":
                    commands = commands[:-2] + ["Z"]
                else:
                    commands = commands[:-1]

        return AvClosedSinglePath(points, commands, CLOSED_SINGLE_PATH_CONSTRAINTS)

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Return True if the point lies inside this path (ray casting).

        This method is most meaningful for closed polygon-like paths.
        For paths with curves, the path is first polygonized.
        """
        # For paths with curves, polygonize first
        if any(cmd in ["Q", "C"] for cmd in self._commands):
            return self.polygonized_path().contains_point(point)

        pts = self._points
        n = pts.shape[0]
        if n == 0:
            return False
        x, y = point
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i][:2]
            xj, yj = pts[j][:2]
            intersects = (yi > y) != (yj > y)
            if intersects:
                slope = (xj - xi) / (yj - yi)
                x_intersect = slope * (y - yi) + xi
                if x < x_intersect:
                    inside = not inside
            j = i
        return inside

    def representative_point(self, samples: int = 9, epsilon: float = 1e-9) -> Tuple[float, float]:
        """Return a point intended to lie inside the path.

        The centroid of a concave polygon can lie outside the filled region.
        This method finds interior points by intersecting several horizontal
        scanlines with the polygon edges and returning the midpoint of an
        interior interval.

        For paths with curves, the path is first polygonized.

        Args:
            samples: Number of scanlines to try between ymin and ymax.
            epsilon: Small relative offset applied to the scanline y value to
                avoid pathological cases where the scanline hits vertices
                exactly.

        Returns:
            Tuple[float, float]: A point inside the path when the contour is
                a simple (non self-intersecting) ring. For degenerate cases a
                best-effort fallback is returned.
        """
        # For paths with curves, polygonize first
        if any(cmd in ["Q", "C"] for cmd in self._commands):
            return self.polygonized_path().representative_point(samples, epsilon)

        pts = self._points
        if pts.shape[0] == 0:
            return (0.0, 0.0)

        if pts.shape[0] < 3:
            return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

        y_min = float(pts[:, 1].min())
        y_max = float(pts[:, 1].max())
        height = y_max - y_min
        if np.isclose(height, 0.0):
            return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

        n = int(pts.shape[0])
        n_samples = max(int(samples), 1)

        y_tol = abs(epsilon) * height

        for k in range(n_samples):
            y = y_min + (k + 0.5) / n_samples * height + epsilon * height

            xs: List[float] = []
            j = n - 1
            for i in range(n):
                xi, yi = float(pts[i, 0]), float(pts[i, 1])
                xj, yj = float(pts[j, 0]), float(pts[j, 1])

                if (yi > y) != (yj > y):
                    dy = yj - yi
                    if abs(dy) >= y_tol:
                        x_int = xi + (y - yi) * (xj - xi) / dy
                        xs.append(float(x_int))

                j = i

            xs.sort()

            if len(xs) % 2 != 0:
                continue

            best: Optional[Tuple[float, float]] = None
            best_w = -1.0
            for i in range(0, len(xs) - 1, 2):
                w = xs[i + 1] - xs[i]
                if w > best_w:
                    best_w = w
                    best = ((xs[i] + xs[i + 1]) * 0.5, y)

            if best is not None and self.contains_point(best):
                return (float(best[0]), float(best[1]))

        # Fallback to centroid if available
        if self._constraints.must_close:
            candidate = self.centroid
            if self.contains_point(candidate):
                return candidate

        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

    def bounding_box(self) -> AvBox:
        """
        Returns bounding box (tightest box around Path)
        Coordinates are relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top
        Uses dimensions in unitsPerEm.
        """

        if self._bounding_box is not None:
            return self._bounding_box

        # No points, so bounding box is set to size 0.
        if not self._points.size:
            self._bounding_box = AvBox(0.0, 0.0, 0.0, 0.0)
            return self._bounding_box

        # Check if path contains curves that need polygonization for accurate bounding box
        has_curves = any(cmd in ["Q", "C"] for cmd in self._commands)

        if not has_curves:
            # No curves, use simple min/max calculation on existing points
            points_x = self._points[:, 0]
            points_y = self._points[:, 1]
            x_min, x_max, y_min, y_max = points_x.min(), points_x.max(), points_y.min(), points_y.max()
        else:
            # Has curves, polygonize temporarily to get accurate bounding box
            polygonized_path = self.polygonized_path()
            polygonized_points = polygonized_path.points

            if polygonized_points.size == 0:
                self._bounding_box = AvBox(0.0, 0.0, 0.0, 0.0)
                return self._bounding_box

            # Calculate bounding box from polygonized points
            points_x = polygonized_points[:, 0]
            points_y = polygonized_points[:, 1]
            x_min, x_max, y_min, y_max = points_x.min(), points_x.max(), points_y.min(), points_y.max()

        self._bounding_box = AvBox(x_min, y_min, x_max, y_max)
        return self._bounding_box

    @classmethod
    def from_dict(cls, data: dict) -> AvPath:
        """Create an AvPath instance from a dictionary."""
        raw_points = data.get("points", None)
        if "points" in data:
            if raw_points:
                points = np.array(raw_points, dtype=np.float64)
            else:
                points = None
        else:
            points = None

        # Convert commands back from strings to enums
        commands = [cmd for cmd in data.get("commands", [])]

        # Convert bounding box back if present
        bounding_box = None
        if data.get("bounding_box") is not None:
            bounding_box = AvBox.from_dict(data["bounding_box"])

        # Convert constraints back if present
        constraints = None
        if data.get("constraints") is not None:
            constraints = PathConstraints.from_dict(data["constraints"])

        # Create instance with 3D points (already has type column)
        path = cls(points, commands, constraints)
        path._bounding_box = bounding_box
        return path

    def to_dict(self) -> dict:
        """Convert the AvPath instance to a dictionary."""
        # Convert numpy array to list for JSON serialization
        points_list = self._points.tolist() if self._points.size > 0 else []

        # Commands are already strings
        commands_list = list(self._commands)

        # Convert bounding box to dict if present
        bbox_dict = None
        if self._bounding_box is not None:
            bbox_dict = self._bounding_box.to_dict()

        return {
            "points": points_list,
            "commands": commands_list,
            "bounding_box": bbox_dict,
            "constraints": self._constraints.to_dict(),
        }

    def polygonized_path(self) -> AvMultiPolylinePath:
        """Return the polygonized path."""
        if self._polygonized_path is None:
            self._polygonized_path = self.polygonize(self.POLYGONIZE_STEPS_INTERNAL)
        return self._polygonized_path

    def polygonize(self, steps: int) -> AvMultiPolylinePath:
        """Return a polygonized copy of this path.

        Args:
            steps: Number of segments used to approximate curve segments.
                If 0, the original path is returned unchanged.

        Returns:
            AvPath: New path with curves replaced by line segments.
                If this instance already has no curves, it is returned unchanged.
        """
        if steps == 0:
            return self

        # Check if path already has no curves
        if not any(cmd in ["Q", "C"] for cmd in self._commands):
            return self

        points = self._points
        commands = self._commands

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
        return AvMultiPolylinePath(new_points, new_commands_list, MULTI_POLYLINE_CONSTRAINTS)

    def split_into_single_paths(self) -> List[AvSinglePath]:
        """Split an AvPath into single-segment AvSinglePath instances at each 'M' command."""

        # Empty path: nothing to split
        if not self.commands:
            return []

        pts = self.points
        cmds = self.commands

        single_paths: List[AvSinglePath] = []

        point_idx = 0
        cmd_idx = 0

        while cmd_idx < len(cmds):
            cmd = cmds[cmd_idx]

            if cmd != "M":
                # AvPath._validate should guarantee segments start with 'M'
                raise ValueError(f"Expected 'M' at command index {cmd_idx}, got '{cmd}'")

            # Start a new segment
            seg_cmds: List[str] = []
            seg_points: List[np.ndarray] = []

            # Handle MoveTo (always uses one point)
            if point_idx >= len(pts):
                raise ValueError("MoveTo command has no corresponding point")

            seg_cmds.append("M")
            seg_points.append(pts[point_idx].copy())
            point_idx += 1
            cmd_idx += 1

            # Consume commands until next 'M' or end
            while cmd_idx < len(cmds) and cmds[cmd_idx] != "M":
                cmd = cmds[cmd_idx]

                if cmd == "L":
                    if point_idx >= len(pts):
                        raise ValueError("LineTo command has no corresponding point")
                    seg_cmds.append("L")
                    seg_points.append(pts[point_idx].copy())
                    point_idx += 1
                    cmd_idx += 1

                elif cmd == "Q":
                    if point_idx + 1 >= len(pts):
                        raise ValueError("Quadratic Bezier command needs 2 points")
                    seg_cmds.append("Q")
                    seg_points.append(pts[point_idx].copy())  # control
                    seg_points.append(pts[point_idx + 1].copy())  # end
                    point_idx += 2
                    cmd_idx += 1

                elif cmd == "C":
                    if point_idx + 2 >= len(pts):
                        raise ValueError("Cubic Bezier command needs 3 points")
                    seg_cmds.append("C")
                    seg_points.append(pts[point_idx].copy())  # control1
                    seg_points.append(pts[point_idx + 1].copy())  # control2
                    seg_points.append(pts[point_idx + 2].copy())  # end
                    point_idx += 3
                    cmd_idx += 1

                elif cmd == "Z":
                    seg_cmds.append("Z")
                    cmd_idx += 1

                else:
                    raise ValueError(f"Unknown command '{cmd}'")

            # Create AvSinglePath for this segment with single-path constraints
            seg_points_array = (
                np.array(seg_points, dtype=np.float64) if seg_points else np.empty((0, 3), dtype=np.float64)
            )

            single_paths.append(AvSinglePath(seg_points_array, seg_cmds, SINGLE_PATH_CONSTRAINTS))

        return single_paths

    def append(self, *paths: Union[AvPath, Sequence[AvPath]]) -> AvPath:
        """Return a new AvPath consisting of this path followed by other paths.

        The given paths are concatenated by keeping each segment's initial 'M'
        command and appending all points and commands in order. The original
        paths are not modified.
        """

        # Start with this path and flatten additional arguments: accept single
        # AvPath, multiple AvPaths, or sequences (tuple, list, etc.) of AvPath
        # instances.
        flat_paths: List[AvPath] = [self]

        for arg in paths:
            if isinstance(arg, AvPath):
                flat_paths.append(arg)
            elif isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
                for item in arg:
                    if not isinstance(item, AvPath):
                        raise TypeError("append expects only AvPath instances")
                    flat_paths.append(item)
            else:
                raise TypeError("append expects AvPath instances or sequences of AvPath instances")

        # Collect points and commands without modifying originals
        points_arrays: List[NDArray[np.float64]] = []
        commands_lists: List[List[AvGlyphCmds]] = []

        for path in flat_paths:
            if path.points.size > 0:
                points_arrays.append(path.points)
            if path.commands:
                commands_lists.append(path.commands)

        if not points_arrays:
            # All paths were empty (including self)
            return AvPath()

        if len(points_arrays) == 1:
            new_points = points_arrays[0].copy()
        else:
            new_points = np.concatenate(points_arrays, axis=0)

        new_commands: List[AvGlyphCmds] = []
        for cmds in commands_lists:
            new_commands.extend(cmds)

        return AvPath(new_points, new_commands)

    @classmethod
    def join_paths(cls, *paths: Union[AvPath, Sequence[AvPath]]) -> AvPath:
        """Join one or more paths into a single AvPath using append()."""

        # Flatten arguments: accept single AvPath, multiple AvPaths, or
        # sequences (tuple, list, etc.) of AvPath instances.
        flat_paths: List[AvPath] = []

        for arg in paths:
            if isinstance(arg, AvPath):
                flat_paths.append(arg)
            elif isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
                for item in arg:
                    if not isinstance(item, AvPath):
                        raise TypeError("join_paths expects only AvPath instances")
                    flat_paths.append(item)
            else:
                raise TypeError("join_paths expects AvPath instances or sequences of AvPath instances")

        # No paths -> return empty path
        if not flat_paths:
            return cls()

        # Use the first path as base and append the rest
        base = flat_paths[0]
        if len(flat_paths) == 1:
            return base

        return base.append(flat_paths[1:])


###############################################################################
# Type Aliases for Path Types
###############################################################################
# These type aliases provide clear type hints for paths with specific constraints.
# They are all AvPath at runtime but communicate intent in type annotations.
# Use runtime properties (is_single_path, is_polylines_path, etc.) to verify.
# Using string literals to avoid circular import issues.

AvSinglePath = AvPath
"""Type alias for AvPath with SINGLE_PATH_CONSTRAINTS (single segment, may have curves)."""

AvClosedSinglePath = AvPath
"""Type alias for AvPath with CLOSED_SINGLE_PATH_CONSTRAINTS (single closed segment, may have curves)."""

AvMultiPolylinePath = AvPath
"""Type alias for AvPath with MULTI_POLYLINE_CONSTRAINTS (multiple segments, no curves)."""

AvSinglePolygonPath = AvPath
"""Type alias for AvPath with SINGLE_POLYGON_CONSTRAINTS (single closed polygon, no curves)."""

AvMultiPolygonPath = AvPath
"""Type alias for AvPath with MULTI_POLYGON_CONSTRAINTS (multiple closed polygons, no curves)."""


###############################################################################
# Main
###############################################################################


def main():
    """Main"""


if __name__ == "__main__":
    main()
