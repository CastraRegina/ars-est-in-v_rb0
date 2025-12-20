"""SVG path handling and geometric operations for vector graphics processing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ave.bezier import BezierCurve
from ave.common import AvGlyphCmds
from ave.geom import AvBox, AvPolygon
from ave.path_support import (  # pylint: disable=unused-import
    CLOSED_SINGLE_PATH_CONSTRAINTS,
    COMMAND_INFO,
    GENERAL_CONSTRAINTS,
    MULTI_POLYGON_CONSTRAINTS,
    MULTI_POLYLINE_CONSTRAINTS,
    SINGLE_PATH_CONSTRAINTS,
    SINGLE_POLYGON_CONSTRAINTS,
    AvPathUtils,
    PathCommandInfo,
    PathCommandProcessor,
    PathConstraints,
    PathValidator,
)

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
            # Generate type column based on commands using PathCommandProcessor
            self._points = self._process_2d_to_3d(arr, commands_list)
        elif arr.shape[1] == 3:
            self._points = arr
        else:
            raise ValueError(f"points must have shape (n, 2) or (n, 3), got {arr.shape}")

        self._commands = commands_list
        self._constraints = constraints if constraints is not None else GENERAL_CONSTRAINTS
        self._bounding_box = None
        self._polygonized_path = None
        self._validate()

    def _process_2d_to_3d(self, points: NDArray[np.float64], commands: List[AvGlyphCmds]) -> NDArray[np.float64]:
        """Convert 2D points to 3D with type column based on commands using PathCommandProcessor."""
        # Validate command sequence first
        PathCommandProcessor.validate_command_sequence(commands, points)

        type_column = np.zeros(points.shape[0], dtype=np.float64)
        point_idx = 0

        for cmd in commands:
            if cmd == "Z":
                continue

            consumed = PathCommandProcessor.get_point_consumption(cmd)

            if cmd in ["M", "L"]:
                # Regular points get type 0.0 (no change needed)
                point_idx += consumed
            elif cmd == "Q":
                # Quadratic: control point (2.0), end point (0.0)
                type_column[point_idx] = 2.0  # Control point
                type_column[point_idx + 1] = 0.0  # End point
                point_idx += consumed
            elif cmd == "C":
                # Cubic: control1 (3.0), control2 (3.0), end point (0.0)
                type_column[point_idx] = 3.0  # Control point 1
                type_column[point_idx + 1] = 3.0  # Control point 2
                type_column[point_idx + 2] = 0.0  # End point
                point_idx += consumed

        return np.column_stack([points, type_column])

    def _validate(self) -> None:
        """Validate path structure using PathValidator.

        This method validates basic path structure and point/command consistency,
        then delegates constraint-specific validation to PathValidator.
        """
        cmds = self.commands
        points = self.points

        # Constraint-based validation (handles empty path check, curve allowance, segment limits)
        PathValidator.validate(cmds, points, self.constraints)

        # Nothing more to check for empty paths
        if not cmds:
            return

        # Use PathValidator's segment parsing to validate structure and point count
        segments = AvPathUtils.split_commands_into_segments(cmds)

        # Validate that segments start with M and Z properly terminates segments
        idx = 0
        for seg_idx, (seg_cmds, _) in enumerate(segments):
            if not seg_cmds or seg_cmds[0] != "M":
                raise ValueError(
                    f"Each segment must start with 'M' command (segment {seg_idx} "
                    f"starts with '{seg_cmds[0] if seg_cmds else 'None'}')"
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
        return self.constraints.must_close and not self.constraints.allows_curves

    @property
    def is_closed(self) -> bool:
        """
        Return True if this path has closure constraints (must_close=True).
        """
        return self.constraints.must_close

    @property
    def is_single_segment(self) -> bool:
        """
        Return True if this path is constrained to a single segment.
        """
        return self.constraints.max_segments == 1

    @property
    def is_single_path(self) -> bool:
        """Return True if this path has SINGLE_PATH_CONSTRAINTS."""
        return self.constraints == SINGLE_PATH_CONSTRAINTS

    @property
    def is_closed_path(self) -> bool:
        """Return True if this path has CLOSED_SINGLE_PATH_CONSTRAINTS."""
        return self.constraints == CLOSED_SINGLE_PATH_CONSTRAINTS

    @property
    def is_polylines_path(self) -> bool:
        """Return True if this path has MULTI_POLYLINE_CONSTRAINTS."""
        return self.constraints == MULTI_POLYLINE_CONSTRAINTS

    @property
    def is_polygon_path(self) -> bool:
        """Return True if this path has SINGLE_POLYGON_CONSTRAINTS."""
        return self.constraints == SINGLE_POLYGON_CONSTRAINTS

    @property
    def is_multi_polygon_path(self) -> bool:
        """Return True if this path has MULTI_POLYGON_CONSTRAINTS."""
        return self.constraints == MULTI_POLYGON_CONSTRAINTS

    @cached_property
    def has_curves(self) -> bool:
        """Return True if this path contains curve commands (Q, C)."""
        return any(PathCommandProcessor.is_curve_command(cmd) for cmd in self.commands)

    @cached_property
    def area(self) -> float:
        """
        Return the area of this path if it's polygon-like.

        For paths with curves, the path is first polygonized.
        For multi-segment paths, returns the sum of all segment areas.

        Returns:
            float: The area of the path. Returns 0.0 if path has fewer than 3 points.

        Raises:
            ValueError: If the path is not closed (must_close constraint required or Z command present).
        """
        # Check if path is closed - either by constraint (which guarantees Z) or by Z command
        if not self.constraints.must_close and not (self.commands and self.commands[-1] == "Z"):
            raise ValueError("Area calculation requires a closed path (must_close=True or Z command)")

        # For polygon-like paths, use direct calculation
        if self.is_polygon_like:
            return AvPolygon.area(self.points)

        # For closed paths with curves, polygonize first
        polygonized = self.polygonized_path()
        return AvPolygon.area(polygonized.points)

    @cached_property
    def centroid(self) -> Tuple[float, float]:
        """
        Return the centroid of this path if it's polygon-like.

        For paths with curves, the path is first polygonized.

        Returns:
            Tuple[float, float]: The x and y coordinates of the centroid.

        Raises:
            ValueError: If the path is not closed (must_close constraint required or Z command present).
        """
        # Check if path is closed - either by constraint (which guarantees Z) or by Z command
        if not self.constraints.must_close and not (self.commands and self.commands[-1] == "Z"):
            raise ValueError("Centroid calculation requires a closed path (must_close=True or Z command)")

        # For polygon-like paths, use direct calculation
        if self.is_polygon_like:
            return AvPolygon.centroid(self.points)

        # For closed paths with curves, polygonize first
        polygonized = self.polygonized_path()
        return AvPolygon.centroid(polygonized.points)

    @cached_property
    def is_ccw(self) -> bool:
        """
        Return True if this path runs counter-clockwise.

        For paths with curves, the path is first polygonized.

        Returns:
            bool: True if counter-clockwise, False otherwise.

        Raises:
            ValueError: If the path is not closed (must_close constraint required or Z command present).
        """
        # Check if path is closed - either by constraint (which guarantees Z) or by Z command
        if not self.constraints.must_close and not (self.commands and self.commands[-1] == "Z"):
            raise ValueError("CCW check requires a closed path (must_close=True or Z command)")

        # For polygon-like paths, use direct calculation
        if self.is_polygon_like:
            return AvPolygon.is_ccw(self.points)

        # For closed paths with curves, polygonize first
        polygonized = self.polygonized_path()
        return AvPolygon.is_ccw(polygonized.points)

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
        if not self.commands or self.points.size == 0:
            return AvPath(self.points.copy(), list(self.commands), self.constraints)

        # Check if this is a multi-segment path
        m_count = self.commands.count("M")
        if m_count > 1:
            # Multi-segment path: split, reverse each, join
            single_paths = self.split_into_single_paths()
            reversed_segments = [self._reverse_single_segment(seg) for seg in single_paths]
            return AvPath.join_paths(reversed_segments)

        # Single-segment path: reverse directly
        return self._reverse_single_segment(self)

    def _reverse_single_segment(self, path: AvSinglePath) -> AvSinglePath:
        """Reverse a single-segment path."""
        if not path.commands or path.points.size == 0:
            return AvSinglePath(path.points.copy(), list(path.commands), path.constraints)

        # Check if path is closed
        is_closed = path.commands[-1] == "Z"

        # Build segments by iterating forward once
        segments = []
        last_point = path.points[0].copy()  # Start with M point
        point_idx = 1  # Skip M's point

        for cmd in path.commands[1:]:
            if cmd == "Z":
                break

            start_point = last_point

            if cmd == "L":
                end_point = path.points[point_idx].copy()
                segments.append(("L", [], start_point, end_point))
                last_point = end_point
                point_idx += 1

            elif cmd == "Q":
                control = path.points[point_idx].copy()
                end_point = path.points[point_idx + 1].copy()
                segments.append(("Q", [control], start_point, end_point))
                last_point = end_point
                point_idx += 2

            elif cmd == "C":
                control1 = path.points[point_idx].copy()
                control2 = path.points[point_idx + 1].copy()
                end_point = path.points[point_idx + 2].copy()
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
            new_points.append(path.points[0].copy())

        # Add Z if original was closed
        if is_closed:
            new_commands.append("Z")

        # Convert to numpy array
        points_array = np.array(new_points, dtype=np.float64) if new_points else np.empty((0, 3), dtype=np.float64)

        return AvSinglePath(points_array, new_commands, path.constraints)

    @classmethod
    def make_closed_single(cls, path: AvSinglePath) -> AvClosedSinglePath:
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

            # If points are very close (within a small tolerance), check command type
            tolerance = 1e-10
            if distance < tolerance:
                # Find the actual drawing command (not Z)
                if len(commands) > 1:
                    last_draw_cmd_idx = -2 if commands[-1] == "Z" else -1
                    last_draw_cmd = commands[last_draw_cmd_idx]

                    # Only remove duplicate for line commands (L, M)
                    # Preserve curve endpoints (Q, C) to maintain geometric integrity
                    if last_draw_cmd in ["L", "M"]:
                        # Special case: don't remove M if it would leave only Z command
                        if last_draw_cmd == "M" and len(commands) == 2 and commands[0] == "M" and commands[1] == "Z":
                            # Keep the M command for single point paths
                            pass
                        else:
                            points = points[:-1]  # Remove last point
                            # Also remove one command to maintain point/command ratio
                            # Remove the command before Z (which should be the last command)
                            if len(commands) > 1 and commands[-1] == "Z":
                                commands = commands[:-2] + ["Z"]
                            else:
                                commands = commands[:-1]
                    # For curve commands (Q, C), keep the duplicate point

        return AvClosedSinglePath(points, commands, CLOSED_SINGLE_PATH_CONSTRAINTS)

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Return True if the point lies inside this path (ray casting).

        This method is most meaningful for closed polygon-like paths.
        For paths with curves, the path is first polygonized.
        For multi-segment paths, handles polygons with holes using winding rule.
        """
        # Input validation
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError("Point must be a tuple or list of 2 numeric values")

        # For paths with curves, polygonize first using cached curve detection
        if self.has_curves:
            return self.polygonized_path().contains_point(point)

        # Handle empty path
        if self.points.shape[0] == 0:
            return False

        # For single-segment paths, use direct ray casting
        if self.commands.count("M") <= 1:
            return AvPolygon.ray_casting_single(self.points, point)

        # For multi-segment paths, handle each segment and apply winding rule
        segments = self.split_into_single_paths()
        winding_number = 0

        for segment in segments:
            if segment.points.shape[0] == 0:
                continue

            # Check if point is in this segment
            if AvPolygon.ray_casting_single(segment.points, point):
                # Determine winding direction based on segment's orientation
                if segment.is_ccw:
                    winding_number += 1
                else:
                    winding_number -= 1

        # Point is inside if winding number is non-zero
        return winding_number != 0

    @cached_property
    def _representative_point_cache(self) -> Tuple[float, float]:
        """Cached representative point for performance."""
        return self._compute_representative_point()

    def _compute_representative_point(self, samples: int = 9, epsilon: float = 1e-9) -> Tuple[float, float]:
        """Internal method to compute representative point."""
        # For paths with curves, polygonize first using cached curve detection
        if self.has_curves:
            return self.polygonized_path().representative_point(samples, epsilon)

        pts = self.points
        if pts.shape[0] == 0:
            return (0.0, 0.0)

        if pts.shape[0] < 3:
            return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

        # Use the new polygon scanline method for single polygons
        scanline_point = AvPolygon.interior_point_scanlines(pts, samples, epsilon)
        if scanline_point is not None:
            return scanline_point

        # Fallback to centroid if available
        if self.constraints.must_close:
            candidate = self.centroid
            if self.contains_point(candidate):
                return candidate

        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

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
        # Use cached version for default parameters, compute for custom params
        if samples == 9 and epsilon == 1e-9:
            return self._representative_point_cache
        else:
            return self._compute_representative_point(samples, epsilon)

    def bounding_box(self) -> AvBox:
        """
        Returns bounding box (tightest box around Path)
        Coordinates are relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top
        Uses dimensions in unitsPerEm.
        """

        if self._bounding_box is not None:
            return self._bounding_box

        # No points, so bounding box is set to size 0.
        if not self.points.size:
            self._bounding_box = AvBox(0.0, 0.0, 0.0, 0.0)
            return self._bounding_box

        # Check if path contains curves that need polygonization for accurate bounding box
        has_curves = self.has_curves

        if not has_curves:
            # No curves, use simple min/max calculation on existing points
            points_x = self.points[:, 0]
            points_y = self.points[:, 1]
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
        points_list = self.points.tolist() if self.points.size > 0 else []

        # Commands are already strings
        commands_list = list(self.commands)

        # Convert bounding box to dict if present
        bbox_dict = None
        if self._bounding_box is not None:
            bbox_dict = self.bounding_box().to_dict()

        return {
            "points": points_list,
            "commands": commands_list,
            "bounding_box": bbox_dict,
            "constraints": self.constraints.to_dict(),
        }

    def polygonized_path(self) -> AvMultiPolylinePath:
        """Return the polygonized path with lazy evaluation and caching."""
        if self._polygonized_path is None:
            # Only polygonize if we actually have curves
            if self.has_curves:
                self._polygonized_path = self.polygonize(self.POLYGONIZE_STEPS_INTERNAL)
            else:
                # No curves - return a copy with polyline constraints
                self._polygonized_path = AvMultiPolylinePath(
                    self.points.copy(), list(self.commands), MULTI_POLYLINE_CONSTRAINTS
                )
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

        # Check if path already has no curves using cached curve detection
        if not self.has_curves:
            return self

        points = self.points
        commands = self.commands

        # Input normalization: ensure all points are 3D using PathCommandProcessor
        if points.shape[1] == 2:
            points = self._process_2d_to_3d(points, commands)

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

            # Consume commands until next 'M' or end using PathCommandProcessor
            while cmd_idx < len(cmds) and cmds[cmd_idx] != "M":
                cmd = cmds[cmd_idx]
                consumed = PathCommandProcessor.get_point_consumption(cmd)

                if cmd == "L":
                    if point_idx + consumed > len(pts):
                        raise ValueError(f"LineTo command needs {consumed} point, got {len(pts) - point_idx}")
                    seg_cmds.append("L")
                    seg_points.append(pts[point_idx].copy())
                    point_idx += consumed
                    cmd_idx += 1

                elif cmd == "Q":
                    if point_idx + consumed > len(pts):
                        raise ValueError(
                            f"Quadratic Bezier command needs {consumed} points, got {len(pts) - point_idx}"
                        )
                    seg_cmds.append("Q")
                    seg_points.append(pts[point_idx].copy())  # control
                    seg_points.append(pts[point_idx + 1].copy())  # end
                    point_idx += consumed
                    cmd_idx += 1

                elif cmd == "C":
                    if point_idx + consumed > len(pts):
                        raise ValueError(f"Cubic Bezier command needs {consumed} points, got {len(pts) - point_idx}")
                    seg_cmds.append("C")
                    seg_points.append(pts[point_idx].copy())  # control1
                    seg_points.append(pts[point_idx + 1].copy())  # control2
                    seg_points.append(pts[point_idx + 2].copy())  # end
                    point_idx += consumed
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
