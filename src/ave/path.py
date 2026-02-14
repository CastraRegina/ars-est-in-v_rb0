"""SVG path handling and geometric operations for vector graphics processing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import shapely.geometry
from numpy.typing import NDArray

from ave.common import AffineTransform, AvGlyphCmds, sgn_sci
from ave.geom import AvBox, AvPolygon
from ave.path_support import (  # pylint: disable=unused-import
    CLOSED_SINGLE_PATH_CONSTRAINTS,
    COMMAND_INFO,
    GENERAL_CONSTRAINTS,
    MULTI_POLYGON_CONSTRAINTS,
    MULTI_POLYLINE_CONSTRAINTS,
    SINGLE_PATH_CONSTRAINTS,
    SINGLE_POLYGON_CONSTRAINTS,
    PathCommandInfo,
    PathCommandProcessor,
    PathConstraints,
    PathPolygonizer,
    PathSplitter,
    PathSvgString,
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

    Note: bounding_box and polygonized_path use @cached_property for automatic caching.
    """

    _points: NDArray[np.float64]  # shape (n_points, 3)
    _commands: List[AvGlyphCmds]  # shape (n_commands, 1)
    _constraints: PathConstraints = GENERAL_CONSTRAINTS  # path constraints

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

        # Validate array shape and dimensions with improved empty array handling
        self._validate_points_array(arr)

        commands_list = [] if commands is None else list(commands)

        if arr.shape[1] == 2:
            # Generate type column based on commands using PathCommandProcessor
            self._points = self._process_2d_to_3d(arr, commands_list)
        elif arr.shape[1] == 3:
            self._points = arr

        self._commands = commands_list
        self._constraints = constraints if constraints is not None else GENERAL_CONSTRAINTS
        self._validate()

    def _validate_points_array(self, arr: NDArray[np.float64]) -> None:
        """Validate points array has correct shape and dimensions."""
        if arr.size == 0:
            # Empty array is valid, will be handled by validation
            return
        if arr.ndim != 2:
            raise ValueError(f"AvPath.__init__: points must have 2 dimensions, got {arr.ndim}")
        if arr.shape[1] not in (2, 3):
            raise ValueError(f"AvPath.__init__: points must have shape (n, 2) or (n, 3), got {arr.shape}")

    def _require_closed_path(self, operation: str) -> None:
        """Helper method to check if path is closed for geometric operations."""
        if not self.are_all_segments_closed():
            raise ValueError(f"AvPath.{operation}: requires a closed path (must_close=True or Z command)")

    def _validate_point_input(self, point: Tuple[float, float]) -> None:
        """Validate point input for contains_point and similar methods."""
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError("AvPath.contains_point: Point must be a tuple or list of 2 numeric values")

    def _process_2d_to_3d(self, points: NDArray[np.float64], commands: List[AvGlyphCmds]) -> NDArray[np.float64]:
        """Convert 2D points to 3D with type column based on commands using PathCommandProcessor."""
        # Validate command sequence first
        PathCommandProcessor.validate_command_sequence(commands, points)

        type_column = np.zeros(points.shape[0], dtype=np.float64)
        point_idx = 0

        for cmd in commands:
            consumed = PathCommandProcessor.get_point_consumption(cmd)

            # Check bounds before accessing type_column
            if point_idx + consumed > points.shape[0]:
                raise ValueError(f"Not enough points for {cmd} command at index {point_idx}")

            if cmd == "Z":
                continue
            if cmd in ["M", "L"]:
                # Regular points get type 0.0
                type_column[point_idx] = 0.0
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
            else:
                raise ValueError(f"Unsupported command: {cmd}")

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
        segments = PathSplitter.split_commands_into_segments(cmds)

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

        Returns a read-only view to avoid expensive copy operations.
        Attempting to modify the returned array will raise an error.
        """
        # Return read-only view for performance - avoids expensive copies
        view = self._points.view()
        view.flags.writeable = False
        return view

    @property
    def commands(self) -> List[AvGlyphCmds]:
        """
        The commands of this path as a list of SVG path commands.

        Note: Modifying the returned list is not recommended and may lead to
        inconsistent state. Treat as read-only.
        """
        # Return direct reference for performance - commands are small and rarely modified
        return self._commands

    @property
    def constraints(self) -> PathConstraints:
        """
        The constraints defining valid path structures.
        """
        return self._constraints

    def with_constraints(self, constraints: PathConstraints) -> AvPath:
        """Return a new AvPath with the specified constraints.

        This method creates a new path with the same points and commands
        but with different constraints. The new path will be validated
        against the new constraints.

        Args:
            constraints: The new constraints to apply

        Returns:
            AvPath: A new path instance with the specified constraints

        Raises:
            ValueError: If the path violates the new constraints
        """
        return AvPath(self.points.copy(), list(self.commands), constraints)

    def transformed_copy(self, affine_trafo: AffineTransform) -> AvPath:
        """Return a copy with coordinates transformed by the given affine matrix.

        Applies the affine transformation:
            x' = a00 * x + a01 * y + b0
            y' = a10 * x + a11 * y + b1

        The third column (point-type flags) and commands are preserved unchanged.

        Args:
            affine: 6-element list [a00, a01, a10, a11, b0, b1] defining the transformation.

        Returns:
            New ``AvPath`` with transformed coordinates and the same
            commands and constraints.
        """
        # TODO: check if performance could be improved
        a00, a01, a10, a11, b0, b1 = affine_trafo
        pts = self.points.copy()
        x, y = pts[:, 0], pts[:, 1]
        pts[:, 0] = a00 * x + a01 * y + b0
        pts[:, 1] = a10 * x + a11 * y + b1
        return AvPath(pts, list(self.commands), self.constraints)

    def determine_appropriate_constraints(self) -> PathConstraints:
        """Analyze this path and determine the most appropriate constraints.

        Returns:
            PathConstraints: Constraints that match the path's actual structure
        """
        # Check if path has curves
        allows_curves = self.has_curves

        # Count segments (number of 'M' commands)
        max_segments = 1 if self.num_segments == 1 else None

        # Check if all segments are closed
        must_close = self.are_all_segments_closed()

        # Determine minimum points per segment
        if not self.commands:
            min_points_per_segment = None
        else:
            # Split into segments and find minimum points
            segments = PathSplitter.split_commands_into_segments(self.commands)
            if segments:
                min_points_per_segment = min(point_count for _, point_count in segments)
            else:
                min_points_per_segment = None

        # Create custom constraints based on analysis
        return PathConstraints.from_attributes(
            allows_curves=allows_curves,
            max_segments=max_segments,
            must_close=must_close,
            min_points_per_segment=min_points_per_segment,
        )

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

    @cached_property
    def num_segments(self) -> int:
        """
        Return the number of segments in this path.

        Segments are defined as sequences starting with 'M' commands.
        Returns 0 for empty paths.
        """
        return self.commands.count("M")

    @cached_property
    def is_single_path(self) -> bool:
        """Return True if this path has SINGLE_PATH_CONSTRAINTS."""
        return self.constraints == SINGLE_PATH_CONSTRAINTS

    @cached_property
    def is_closed_single_path(self) -> bool:
        """Return True if this path has CLOSED_SINGLE_PATH_CONSTRAINTS."""
        return self.constraints == CLOSED_SINGLE_PATH_CONSTRAINTS

    @cached_property
    def is_multi_polyline_path(self) -> bool:
        """Return True if this path has MULTI_POLYLINE_CONSTRAINTS."""
        return self.constraints == MULTI_POLYLINE_CONSTRAINTS

    @cached_property
    def is_single_polygon_path(self) -> bool:
        """Return True if this path has SINGLE_POLYGON_CONSTRAINTS."""
        return self.constraints == SINGLE_POLYGON_CONSTRAINTS

    @cached_property
    def is_multi_polygon_path(self) -> bool:
        """Return True if this path has MULTI_POLYGON_CONSTRAINTS."""
        return self.constraints == MULTI_POLYGON_CONSTRAINTS

    @cached_property
    def has_curves(self) -> bool:
        """Return True if this path contains curve commands (Q, C)."""
        return any(PathCommandProcessor.is_curve_command(cmd) for cmd in self.commands)

    def are_all_segments_closed(self) -> bool:
        """Check if path is properly closed by examining segments."""
        # If constraints require closure, assume it's properly closed
        if self.constraints.must_close:
            return True

        # For non-constraint paths, check if all segments end with Z
        if not self.commands:
            return False

        segments = PathSplitter.split_commands_into_segments(self.commands)

        # Check if all segments (except possibly empty ones) end with Z
        for seg_cmds, _ in segments:
            if seg_cmds and seg_cmds[-1] != "Z":
                return False

        return True

    @cached_property
    def area(self) -> float:
        """
        Return the area of this path if it's polygon-like.

        For paths with curves, the path is first polygonized.
        For multi-segment paths, returns the sum of all segment areas.

        Note: This method always returns 0 or a positive value, regardless of
        the path's winding direction (clockwise or counter-clockwise).

        Returns:
            float: The absolute area of the path (always >= 0.0).
                    Returns 0.0 if path has fewer than 3 points.

        Raises:
            ValueError: If the path is not closed (must_close constraint required or Z command present).
        """
        # Check if path is closed using proper segment analysis
        self._require_closed_path("area")

        # For polygon-like paths, use direct calculation
        if self.is_polygon_like:
            return AvPolygon.area(self.points)

        # For closed paths with curves, polygonize first
        polygonized = self.polygonized_path
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
        # Check if path is closed using proper segment analysis
        self._require_closed_path("centroid")

        # Get polygonized path (handles both polygon and curve cases)
        polygonized = self.polygonized_path

        # Convert to Shapely polygon and get centroid
        if len(polygonized.points) >= 3:
            poly = shapely.geometry.Polygon(polygonized.points[:, :2].tolist())
            if poly.is_valid and not poly.is_empty:
                return (poly.centroid.x, poly.centroid.y)

        # Fallback to original method for invalid polygons
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
        # Check if path is closed using proper segment analysis
        self._require_closed_path("is_ccw")

        # For polygon-like paths, use direct calculation
        if self.is_polygon_like:
            return AvPolygon.is_ccw(self.points)

        # For closed paths with curves, polygonize first
        polygonized = self.polygonized_path
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
        if self.num_segments > 1:
            # Multi-segment path: split, reverse each, join
            single_paths = self.split_into_single_paths()
            reversed_segments = [self._reverse_single_segment(seg) for seg in single_paths]
            return AvPath.join_paths(reversed_segments, constraints=self.constraints)

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
        # Input validation using standardized helper
        self._validate_point_input(point)

        # Check if path is closed for consistency with other geometric operations
        self._require_closed_path("contains_point")

        # For paths with curves, polygonize first and use the points directly
        if self.has_curves:
            polygonized = self.polygonized_path
            points = polygonized.points
            commands = polygonized.commands
        else:
            # Handle empty path
            if self.points.shape[0] == 0:
                return False
            points = self.points
            commands = self.commands

        # For single-segment paths, use direct ray casting
        if self.num_segments <= 1:
            return AvPolygon.ray_casting_single(points, point)

        # For multi-segment paths, handle each segment and apply winding rule
        segments = PathSplitter.split_into_single_paths(points, commands)
        winding_number = 0

        for segment in segments:
            if segment.points.shape[0] == 0:
                continue

            # Check if point is in this segment
            if AvPolygon.ray_casting_single(segment.points, point):
                # Determine winding direction based on segment's orientation
                # Only use is_ccw if segment is properly closed, otherwise use polygon calculation
                try:
                    if segment.are_all_segments_closed():
                        if segment.is_ccw:
                            winding_number += 1
                        else:
                            winding_number -= 1
                    else:
                        # For unclosed segments, calculate orientation from points directly
                        if AvPolygon.is_ccw(segment.points):
                            winding_number += 1
                        else:
                            winding_number -= 1
                except (ValueError, IndexError):
                    # If orientation calculation fails, skip this segment
                    continue

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
            return self.polygonized_path.representative_point(samples, epsilon)

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

    @cached_property
    def bounding_box(self) -> AvBox:
        """
        Returns bounding box (tightest box around Path)
        Coordinates are relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top
        Uses dimensions in unitsPerEm.
        """
        # No points, so bounding box is set to size 0.
        if not self.points.size:
            return AvBox(0.0, 0.0, 0.0, 0.0)

        # Check if path contains curves that need polygonization for accurate bounding box
        has_curves = self.has_curves

        if not has_curves:
            # No curves, use simple min/max calculation on existing points
            points_x = self.points[:, 0]
            points_y = self.points[:, 1]
            x_min, x_max, y_min, y_max = points_x.min(), points_x.max(), points_y.min(), points_y.max()
        else:
            # Has curves, polygonize temporarily to get accurate bounding box
            polygonized_path = self.polygonized_path
            polygonized_points = polygonized_path.points

            if polygonized_points.size == 0:
                return AvBox(0.0, 0.0, 0.0, 0.0)

            # Calculate bounding box from polygonized points
            points_x = polygonized_points[:, 0]
            points_y = polygonized_points[:, 1]
            x_min, x_max, y_min, y_max = points_x.min(), points_x.max(), points_y.min(), points_y.max()

        return AvBox(x_min, y_min, x_max, y_max)

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

        # Convert constraints back if present
        constraints = None
        if data.get("constraints") is not None:
            constraints = PathConstraints.from_dict(data["constraints"])

        # Create instance with 3D points (already has type column) - this triggers validation
        path = cls(points, commands, constraints)

        # Convert bounding box back if present and set it in __dict__ for @cached_property
        if data.get("bounding_box") is not None:
            bounding_box = AvBox.from_dict(data["bounding_box"])
            path.__dict__["bounding_box"] = bounding_box  # Set cached value for @cached_property

        return path

    def to_dict(self) -> dict:
        """Convert the AvPath instance to a dictionary."""
        # Convert numpy array to list for JSON serialization
        points_list = self.points.tolist() if self.points.size > 0 else []

        # Commands are already strings
        commands_list = list(self.commands)

        # Convert bounding box to dict if it has been cached
        bbox_dict = None
        if "bounding_box" in self.__dict__:
            bbox_dict = self.bounding_box.to_dict()

        return {
            "points": points_list,
            "commands": commands_list,
            "bounding_box": bbox_dict,
            "constraints": self.constraints.to_dict(),
        }

    def approx_equal(self, other: AvPath, rtol: float = 1e-9, atol: float = 1e-9) -> bool:
        """Check if two paths are approximately equal within numerical tolerances.

        Args:
            other: Another AvPath to compare with
            rtol: Relative tolerance for floating point comparison
            atol: Absolute tolerance for floating point comparison

        Returns:
            True if paths are approximately equal, False otherwise
        """
        if not isinstance(other, AvPath):
            return False

        # Compare points with numpy allclose
        if not np.allclose(self.points, other.points, rtol=rtol, atol=atol):
            return False

        # Compare commands exactly (they're strings/enums)
        if self.commands != other.commands:
            return False

        # Compare constraints (simple values, direct comparison is fine)
        if self.constraints != other.constraints:
            return False

        return True

    @cached_property
    def polygonized_path(self) -> AvMultiPolylinePath:
        """Return the polygonized path with lazy evaluation and caching."""
        # Only polygonize if we actually have curves
        if self.has_curves:
            return self.polygonize(self.POLYGONIZE_STEPS_INTERNAL)
        else:
            # No curves - return a copy with polyline constraints
            return AvMultiPolylinePath(self.points.copy(), list(self.commands), MULTI_POLYLINE_CONSTRAINTS)

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

        new_points, new_commands = PathPolygonizer.polygonize_path(
            self.points, self.commands, steps, self._process_2d_to_3d
        )

        return AvMultiPolylinePath(new_points, new_commands, MULTI_POLYLINE_CONSTRAINTS)

    def split_into_single_paths(self) -> List[AvSinglePath]:
        """Split an AvPath into single-segment AvSinglePath instances at each 'M' command."""
        # Use the extracted PathSplitter
        return PathSplitter.split_into_single_paths(self.points, self.commands)

    def append(self, *paths: Union[AvPath, Sequence[AvPath]], constraints: Optional[PathConstraints] = None) -> AvPath:
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
            return AvPath(constraints=constraints or GENERAL_CONSTRAINTS)

        if len(points_arrays) == 1:
            new_points = points_arrays[0].copy()
        else:
            new_points = np.concatenate(points_arrays, axis=0)

        new_commands: List[AvGlyphCmds] = []
        for cmds in commands_lists:
            new_commands.extend(cmds)

        return AvPath(new_points, new_commands, constraints or GENERAL_CONSTRAINTS)

    @classmethod
    def join_paths(
        cls, *paths: Union[AvPath, Sequence[AvPath]], constraints: Optional[PathConstraints] = None
    ) -> AvPath:
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
            return cls(constraints=constraints or GENERAL_CONSTRAINTS)

        # Use the first path as base and append the rest
        base = flat_paths[0]
        if len(flat_paths) == 1:
            if constraints is None:
                return base
            return cls(base.points.copy(), list(base.commands), constraints)

        return base.append(flat_paths[1:], constraints=constraints or GENERAL_CONSTRAINTS)

    def __str__(self) -> str:
        """Return detailed information about the path including points, commands, and distances as a string."""
        lines = []

        # Header
        formatted_commands: List[str] = []
        for cmd in self.commands:
            if cmd == "M" and formatted_commands:
                formatted_commands.append(" ")
            formatted_commands.append(cmd)
        lines.append(f"Path: {''.join(formatted_commands)}")
        lines.append(f"  #commands: {len(self.commands)}, #points: {len(self.points)}")

        # Area and centroid info if all segments are closed
        if self.are_all_segments_closed():
            area = self.area
            centroid = self.centroid
            rep_point = self.representative_point()
            fmt_area = sgn_sci(area, always_positive=True)
            fmt_centroid = f"({sgn_sci(centroid[0])}, {sgn_sci(centroid[1])})"
            fmt_rep_point = f"({sgn_sci(rep_point[0])}, {sgn_sci(rep_point[1])})"
            lines.append(f"  segments \\     overall_area={fmt_area} centroid={fmt_centroid} repr_pt={fmt_rep_point}")

        # Segment information
        if self.num_segments > 0:
            segments = self.split_into_single_paths()
            for i, segment in enumerate(segments):
                if segment.are_all_segments_closed():
                    is_ccw = segment.is_ccw
                    area = segment.area
                    centroid = segment.centroid
                    rep_point = segment.representative_point()
                    fmt_area = sgn_sci(area, always_positive=True)
                    fmt_centroid = f"({sgn_sci(centroid[0])}, {sgn_sci(centroid[1])})"
                    fmt_rep_point = f"({sgn_sci(rep_point[0])}, {sgn_sci(rep_point[1])})"
                    lines.append(
                        f"    [{i:5d}] is_ccw={str(is_ccw):5s}"
                        f" area={fmt_area}"
                        f" centroid={fmt_centroid}"
                        f" repr_pt={fmt_rep_point}"
                    )
                else:  # not closed
                    lines.append(f"    [{i:5d}] not closed")

        lines.append("  commands and points:")
        # Format all points with proper formatting
        cmd_idx = 0
        points_in_current_cmd = 0

        segment_start_point: Optional[NDArray[np.float64]] = None
        next_cmd_width = 2
        distance_column_width = 10

        for i, point in enumerate(self.points):
            cmd_display = ""
            next_cmd = ""
            # Skip Z commands to find active command
            while cmd_idx < len(self.commands) and self.commands[cmd_idx] == "Z":
                cmd_idx += 1

            if cmd_idx < len(self.commands):
                cmd = self.commands[cmd_idx]
                cmd_display = cmd
                if cmd == "M":
                    segment_start_point = point
                    next_cmd = " "  # Move commands do not close path
                else:
                    # Only show Z if this is the last point of the current command
                    # and the next command is Z
                    consumed = PathCommandProcessor.get_point_consumption(cmd)
                    is_last_point_of_cmd = points_in_current_cmd + 1 >= consumed
                    next_cmd = (
                        " Z"
                        if (
                            is_last_point_of_cmd
                            and cmd_idx + 1 < len(self.commands)
                            and self.commands[cmd_idx + 1] == "Z"
                        )
                        else ""
                    )

                consumed = PathCommandProcessor.get_point_consumption(cmd)
                points_in_current_cmd += 1
                if points_in_current_cmd >= consumed:
                    cmd_idx += 1
                    points_in_current_cmd = 0
            else:
                cmd_display = " "
                next_cmd = ""

            # Format: [index] cmd (x, y, type) Z
            type_val = int(point[2]) if point[2].is_integer() else point[2]

            # Calculate distance to next point if Z is not shown
            distance_text = ""
            if next_cmd == " Z":
                if segment_start_point is not None:
                    dx = segment_start_point[0] - point[0]
                    dy = segment_start_point[1] - point[1]
                    distance = (dx**2 + dy**2) ** 0.5
                    distance_text = sgn_sci(distance, always_positive=True)
            else:
                if i + 1 < len(self.points):
                    next_point = self.points[i + 1]
                    dx = next_point[0] - point[0]
                    dy = next_point[1] - point[1]
                    distance = (dx**2 + dy**2) ** 0.5
                    distance_text = sgn_sci(distance, always_positive=True)

            if distance_text:
                distance_str = f"{distance_text:>{distance_column_width}}"
            else:
                distance_str = " " * distance_column_width

            next_cmd_display = f"{next_cmd or '':>{next_cmd_width}}"

            lines.append(
                f"    [{i:5d}]"
                f" {cmd_display}"  # command
                f" ({sgn_sci(point[0])}, {sgn_sci(point[1])}, {type_val:2d})"  # point
                f"{next_cmd_display}"
                f" {distance_str}"  # next command and distance
            )

        return "\n".join(lines)

    def svg_path_string(
        self,
        scale: float = 1.0,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
    ) -> str:
        """
        Returns the SVG path representation (absolute coordinates) of the path.
        The SVG path is a string that defines the outline using
        SVG path commands. This path can be used to render the path as a
        vector graphic.

        Supported commands:
            M (move-to), L (line-to),
            C (cubic bezier), Q (quadratic bezier),
            Z (close-path).

        Args:
            scale (float): The scale factor to apply to the points before generating the SVG path string.
            translate_x (float): X-coordinate translation before generating the SVG path string.
            translate_y (float): Y-coordinate translation before generating the SVG path string.

        Returns:
            str: The SVG path string (absolute coordinates) representing the path.
                    Returns "M 0 0" if there are no points.
        """
        return PathSvgString.svg_path_string(self.points, self.commands, scale, translate_x, translate_y)

    def svg_path_string_debug_polyline(
        self,
        scale: float = 1.0,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
        stroke_width: float = 1.0,
    ) -> str:
        """
        Returns a debug SVG path representation using only polylines with visual markers.

        This method converts all path commands to straight lines and adds markers to visualize
        the path structure, making it ideal for debugging complex paths and understanding
        control point relationships.

        Command conversion:
            M, L, Z: Preserved as-is (move-to, line-to, close-path)
            Q: Converted to lines connecting all three points (start, control, end)
            C: Converted to lines connecting all four points (start, control1, control2, end)

        Visual markers (size based on stroke_width):
            - Squares: Path points (L commands and curve endpoints)
            - Circles: Control points (intermediate points in Q and C commands)
            - Right triangles: Move-to points (M commands - segment starts)
            - Left triangles: Points before close-path (Z commands - segment ends)

        Args:
            scale: Scale factor for coordinates (default: 1.0)
            translate_x: X-axis translation (default: 0.0)
            translate_y: Y-axis translation (default: 0.0)
            stroke_width: Determines marker sizes (default: 1.0)

        Returns:
            str: Complete SVG path string with polylines and markers.
                    Returns "M 0 0" if path has no points.
        """
        return PathSvgString.svg_path_string_debug_polyline(
            self.points, self.commands, scale, translate_x, translate_y, stroke_width
        )


###############################################################################
# Type Aliases for Path Types
###############################################################################
# These type aliases provide clear type hints for paths with specific constraints.
# They are all AvPath at runtime but communicate intent in type annotations.
# Use runtime properties (is_single_path, is_multi_polyline_path, etc.) to verify.
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
