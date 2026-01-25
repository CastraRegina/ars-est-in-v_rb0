"""Supporting utilities and constraints for AvPath.

This module contains command metadata, validation helpers, and utility
functions that are used by the core path implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ave.bezier import BezierCurve
from ave.common import AvGlyphCmds

if TYPE_CHECKING:
    from ave.path import AvPath  # pylint: disable=unused-import

###############################################################################
# PathCommandInfo
###############################################################################


@dataclass(frozen=True)
class PathCommandInfo:
    """Metadata for SVG path commands.

    Attributes:
        consumes_points: Number of points this command consumes
        is_curve: Whether this command represents a curve
        is_drawing: Whether this command draws (vs. move)
    """

    consumes_points: int
    is_curve: bool
    is_drawing: bool = True


# Command registry with metadata
COMMAND_INFO = {
    "M": PathCommandInfo(1, False, False),  # MoveTo - not drawing
    "L": PathCommandInfo(1, False, True),  # LineTo - drawing
    "Q": PathCommandInfo(2, True, True),  # Quadratic - curve, drawing
    "C": PathCommandInfo(3, True, True),  # Cubic - curve, drawing
    "Z": PathCommandInfo(0, False, True),  # ClosePath - drawing, no points
}


###############################################################################
# PathCommandProcessor
###############################################################################


class PathCommandProcessor:
    """Handles command/point processing operations."""

    @staticmethod
    def get_point_consumption(cmd: str) -> int:
        """Return number of points consumed by command."""
        return COMMAND_INFO[cmd].consumes_points

    @staticmethod
    def is_curve_command(cmd: str) -> bool:
        """Return True if command represents a curve."""
        return COMMAND_INFO[cmd].is_curve

    @staticmethod
    def is_drawing_command(cmd: str) -> bool:
        """Return True if command draws (vs. move)."""
        return COMMAND_INFO[cmd].is_drawing

    @staticmethod
    def validate_command_sequence(commands: List[AvGlyphCmds], points: NDArray) -> None:
        """Validate that commands match available points."""
        point_idx = 0
        for cmd in commands:
            if cmd == "Z":
                continue

            consumed = PathCommandProcessor.get_point_consumption(cmd)
            if point_idx + consumed > len(points):
                raise ValueError(f"Not enough points for {cmd} command at index {point_idx}")
            point_idx += consumed

    @staticmethod
    def process_commands_with_points(
        commands: List[AvGlyphCmds], points: NDArray, processor_func: Callable[[str, Optional[NDArray], int], Any]
    ) -> List[Any]:
        """Generic command processing with point advancement.

        Args:
            commands: List of commands to process
            points: Array of points
            processor_func: Function that takes (cmd, cmd_points, point_idx)

        Returns:
            List of results from processor_func
        """
        results = []
        point_idx = 0

        for cmd in commands:
            if cmd == "Z":
                results.append(processor_func(cmd, None, -1))
                continue

            consumed = PathCommandProcessor.get_point_consumption(cmd)
            cmd_points = points[point_idx : point_idx + consumed]
            results.append(processor_func(cmd, cmd_points, point_idx))
            point_idx += consumed

        return results


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
    def from_dict(cls, data: dict) -> PathConstraints:
        """Create PathConstraints from a dictionary."""
        return cls(
            allows_curves=data.get("allows_curves", True),
            max_segments=data.get("max_segments"),
            must_close=data.get("must_close", False),
            min_points_per_segment=data.get("min_points_per_segment"),
        )

    @classmethod
    def from_attributes(
        cls,
        *,
        allows_curves: bool,
        max_segments: Optional[int],
        must_close: bool,
        min_points_per_segment: Optional[int],
    ) -> PathConstraints:
        """Return the most specific predefined constraints matching the attributes.

        Falls back to a generic PathConstraints instance if no preset matches.
        """
        candidate = cls(
            allows_curves=allows_curves,
            max_segments=max_segments,
            must_close=must_close,
            min_points_per_segment=min_points_per_segment,
        )

        for preset in _PRIORITIZED_CONSTRAINTS:
            if cls._matches_attributes(
                preset,
                allows_curves=allows_curves,
                max_segments=max_segments,
                must_close=must_close,
                min_points_per_segment=min_points_per_segment,
            ):
                return preset

        return candidate

    @classmethod
    def _matches_attributes(
        cls,
        preset: PathConstraints,
        *,
        allows_curves: bool,
        max_segments: Optional[int],
        must_close: bool,
        min_points_per_segment: Optional[int],
    ) -> bool:
        """Return True if the preset satisfies the detected attributes."""
        if preset.allows_curves != allows_curves:
            return False

        if preset.must_close != must_close:
            return False

        if preset.max_segments is not None:
            if max_segments is None or max_segments > preset.max_segments:
                return False
        # If preset.max_segments is None it imposes no constraint.

        if preset.min_points_per_segment is not None:
            if min_points_per_segment is None or min_points_per_segment < preset.min_points_per_segment:
                return False

        return True

    def __str__(self) -> str:
        base_name = "      PathConstraints"
        if self == GENERAL_CONSTRAINTS:
            base_name = "         GENERAL_CSTR"
        elif self == MULTI_POLYLINE_CONSTRAINTS:
            base_name = "  MULTI_POLYLINE_CSTR"
        elif self == SINGLE_PATH_CONSTRAINTS:
            base_name = "     SINGLE_PATH_CSTR"
        elif self == CLOSED_SINGLE_PATH_CONSTRAINTS:
            base_name = "CLSD_SINGLE_PATH_CSTR"
        elif self == SINGLE_POLYGON_CONSTRAINTS:
            base_name = "  SINGLE_POLYGON_CSTR"
        elif self == MULTI_POLYGON_CONSTRAINTS:
            base_name = "   MULTI_POLYGON_CSTR"

        constraint_str = (
            f"curves_ok={str(self.allows_curves):>5}"
            f", max_segs={str(self.max_segments):>5}"
            f", must_close={str(self.must_close):>5}"
            f", min_pts_per_seg={str(self.min_points_per_segment):>5}"
        )
        return f"{base_name}({constraint_str})"


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

_PRIORITIZED_CONSTRAINTS = [
    SINGLE_POLYGON_CONSTRAINTS,
    MULTI_POLYGON_CONSTRAINTS,
    CLOSED_SINGLE_PATH_CONSTRAINTS,
    SINGLE_PATH_CONSTRAINTS,
    MULTI_POLYLINE_CONSTRAINTS,
    GENERAL_CONSTRAINTS,
]


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
        segments = PathSplitter.split_commands_into_segments(commands)

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
# PathSplitter
###############################################################################


class PathSplitter:
    """Utility class for splitting paths into single-segment components."""

    @staticmethod
    def split_commands_into_segments(commands: List[AvGlyphCmds]) -> List[Tuple[List[AvGlyphCmds], int]]:
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

    @staticmethod
    def split_into_single_paths(points: NDArray[np.float64], commands: List[AvGlyphCmds]) -> List["AvPath"]:
        """Split an AvPath into single-segment AvPath instances at each 'M' command.

        Args:
            points: Array of path points
            commands: List of SVG commands

        Returns:
            List of single-segment AvPath instances

        Raises:
            ValueError: If path structure is invalid
        """
        # Import here to avoid circular import
        from ave.path import AvPath  # pylint: disable=import-outside-toplevel

        # Empty path: nothing to split
        if not commands:
            return []

        pts = points
        cmds = commands

        single_paths: List[AvPath] = []
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

            # Create AvPath for this segment with single-path constraints
            seg_points_array = (
                np.array(seg_points, dtype=np.float64) if seg_points else np.empty((0, 3), dtype=np.float64)
            )

            single_paths.append(AvPath(seg_points_array, seg_cmds, SINGLE_PATH_CONSTRAINTS))

        return single_paths


###############################################################################
# PathPolygonizer
###############################################################################


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


###############################################################################
# PathSvgString
###############################################################################


class PathSvgString:
    """Helper class for SVG path string generation utilities."""

    @staticmethod
    def svg_path_string(
        points: NDArray[np.float64],
        commands: List[str],
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
            points (NDArray[np.float64]): The array of points defining the path.
            commands (List[str]): The list of commands defining the path.
            scale (float): The scale factor to apply to the points before generating the SVG path string.
            translate_x (float): X-coordinate translation before generating the SVG path string.
            translate_y (float): Y-coordinate translation before generating the SVG path string.

        Returns:
            str: The SVG path string (absolute coordinates) representing the path.
                    Returns "M 0 0" if there are no points.
        """
        # Apply scale and translation to the points, make points to be 2 dimensions (x, y)
        points_transformed = points[:, :2] * scale + (translate_x, translate_y)

        parts: List[str] = []
        p_idx = 0
        for cmd in commands:
            if cmd in ("M", "L"):
                # Move-to or Line-to: one point (x,y)
                if p_idx >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x, y = points_transformed[p_idx]
                parts.append(f"{cmd} {x:g} {y:g}")
                p_idx += 1
            elif cmd == "Q":
                # Quadratic bezier: control point + end point (2 points total)
                if p_idx + 1 >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x1, y1 = points_transformed[p_idx]
                x2, y2 = points_transformed[p_idx + 1]
                parts.append(f"{cmd} {x1:g} {y1:g} {x2:g} {y2:g}")
                p_idx += 2
            elif cmd == "C":
                # Cubic bezier: control1 + control2 + end point (3 points total)
                if p_idx + 2 >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x1, y1 = points_transformed[p_idx]
                x2, y2 = points_transformed[p_idx + 1]
                x3, y3 = points_transformed[p_idx + 2]
                parts.append(f"{cmd} {x1:g} {y1:g} {x2:g} {y2:g} {x3:g} {y3:g}")
                p_idx += 3
            elif cmd == "Z":
                # Close-path: no coordinates
                parts.append("Z")
            else:
                # Unsupported command (should not occur from AvPointCommandPen)
                raise ValueError(f"Unsupported SVG command: {cmd}")

        # Return the composed absolute-path string or "M 0 0" string if parts is empty
        return " ".join(parts) if parts else "M 0 0"

    @staticmethod
    def svg_path_string_debug_polyline(
        points: NDArray[np.float64],
        commands: List[str],
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
            points: Array of coordinates defining the path shape
            commands: List of SVG commands (M, L, Q, C, Z) that operate on points
            scale: Scale factor for coordinates (default: 1.0)
            translate_x: X-axis translation (default: 0.0)
            translate_y: Y-axis translation (default: 0.0)
            stroke_width: Determines marker sizes (default: 1.0)

        Returns:
            str: Complete SVG path string with polylines and markers.
                    Returns "M 0 0" if path has no points.
        """
        # Apply scale and translation to the points, make points to be 2 dimensions (x, y)
        points_transformed = points[:, :2] * scale + (translate_x, translate_y)

        parts: List[str] = []
        p_idx = 0

        for cmd in commands:
            if cmd == "M":
                # Move-to: one point (x,y), start new segment
                if p_idx >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x, y = points_transformed[p_idx]
                parts.append(f"M {x:g} {y:g}")
                p_idx += 1
            elif cmd == "L":
                # Line-to: one point (x,y)
                if p_idx >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x, y = points_transformed[p_idx]
                parts.append(f"L {x:g} {y:g}")
                p_idx += 1
            elif cmd == "Q":
                # Quadratic bezier: control point + end point -> convert to 2 L commands
                if p_idx + 1 >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x1, y1 = points_transformed[p_idx]  # Control point
                x2, y2 = points_transformed[p_idx + 1]  # End point
                parts.append(f"L {x1:g} {y1:g}")  # Line to control point
                parts.append(f"L {x2:g} {y2:g}")  # Line to end point
                p_idx += 2
            elif cmd == "C":
                # Cubic bezier: control1 + control2 + end point -> convert to 3 L commands
                if p_idx + 2 >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x1, y1 = points_transformed[p_idx]  # Control point 1
                x2, y2 = points_transformed[p_idx + 1]  # Control point 2
                x3, y3 = points_transformed[p_idx + 2]  # End point
                parts.append(f"L {x1:g} {y1:g}")  # Line to control point 1
                parts.append(f"L {x2:g} {y2:g}")  # Line to control point 2
                parts.append(f"L {x3:g} {y3:g}")  # Line to end point
                p_idx += 3
            elif cmd == "Z":
                # Close-path: draw line to start of current segment
                parts.append("Z")
                # No need to track segment_start_point anymore as segment is closed
            else:
                # Unsupported command (should not occur from AvPointCommandPen)
                raise ValueError(f"Unsupported SVG command: {cmd}")

        # Add markers at each point
        square_size = stroke_width * 2
        half_size = square_size / 2
        circle_radius = square_size / 2

        # Track which points are control points and special markers
        p_idx = 0
        is_control_point = [False] * len(points_transformed)
        is_m_point = [False] * len(points_transformed)  # M command points
        is_before_z_point = [False] * len(points_transformed)  # Points before Z commands

        # First pass: identify control points and special markers
        for i, cmd in enumerate(commands):
            if cmd == "M":
                # Move-to: mark this point
                if p_idx < len(is_m_point):
                    is_m_point[p_idx] = True
                p_idx += 1
            elif cmd == "L":
                # Line-to: regular point
                p_idx += 1
            elif cmd == "Q":
                # Quadratic bezier: control point + end point (2 points total)
                # First point is control point, second is end point
                if p_idx < len(is_control_point):
                    is_control_point[p_idx] = True  # Control point
                p_idx += 2
            elif cmd == "C":
                # Cubic bezier: control1 + control2 + end point (3 points total)
                # First two points are control points, third is end point
                if p_idx < len(is_control_point):
                    is_control_point[p_idx] = True  # Control point 1
                if p_idx + 1 < len(is_control_point):
                    is_control_point[p_idx + 1] = True  # Control point 2
                p_idx += 3
            elif cmd == "Z":
                # Close-path: mark the previous point as before-Z
                # Find the last point before this Z command
                if i > 0:
                    # Count points used before this Z to find the last point index
                    temp_p_idx = 0
                    for j in range(i):
                        prev_cmd = commands[j]
                        if prev_cmd in ("M", "L"):
                            temp_p_idx += 1
                        elif prev_cmd == "Q":
                            temp_p_idx += 2
                        elif prev_cmd == "C":
                            temp_p_idx += 3
                        # Z commands don't use points

                    if temp_p_idx > 0:  # There is a point before this Z
                        last_point_idx = temp_p_idx - 1
                        if last_point_idx < len(is_before_z_point):
                            is_before_z_point[last_point_idx] = True
            else:
                # Unsupported command (should not occur from AvPointCommandPen)
                raise ValueError(f"Unsupported SVG command: {cmd}")

        # Add markers for all transformed points
        for i, (x, y) in enumerate(points_transformed):
            # Always add the base marker (square or circle)
            if is_control_point[i]:
                # Control point: circle
                parts.append(f"M {x - circle_radius:g} {y:g}")
                parts.append(f"A {circle_radius:g} {circle_radius:g} 0 1 0 {x + circle_radius:g} {y:g}")
                parts.append(f"A {circle_radius:g} {circle_radius:g} 0 1 0 {x - circle_radius:g} {y:g}")
            else:
                # Regular point: square
                square_x1 = x - half_size
                square_y1 = y - half_size
                square_x2 = x + half_size
                square_y2 = y + half_size

                parts.append(f"M {square_x1:g} {square_y1:g}")
                parts.append(f"L {square_x2:g} {square_y1:g}")
                parts.append(f"L {square_x2:g} {square_y2:g}")
                parts.append(f"L {square_x1:g} {square_y2:g}")
                parts.append("Z")

            # Add additional triangle markers for M points and before-Z points
            if is_m_point[i]:
                # M command point: equilateral triangle with left side vertical (pointing right)
                triangle_size = stroke_width * 2
                height = triangle_size * (3**0.5) / 2  # Height of equilateral triangle

                # Triangle with left side vertical (pointing right)
                # Left side is vertical, so vertices are:
                # Top: (x, y + height/2)
                # Bottom: (x, y - height/2)
                # Right: (x + triangle_size, y)
                parts.append(f"M {x:g} {y + height/2:g}")
                parts.append(f"L {x + triangle_size:g} {y:g}")
                parts.append(f"L {x:g} {y - height/2:g}")
                parts.append("Z")

            if is_before_z_point[i]:
                # Point before Z command: equilateral triangle with right side vertical (pointing left)
                triangle_size = stroke_width * 2
                height = triangle_size * (3**0.5) / 2  # Height of equilateral triangle

                # Triangle with right side vertical (pointing left)
                # Right side is vertical, so vertices are:
                # Top: (x, y + height/2)
                # Bottom: (x, y - height/2)
                # Left: (x - triangle_size, y)
                parts.append(f"M {x:g} {y + height/2:g}")
                parts.append(f"L {x - triangle_size:g} {y:g}")
                parts.append(f"L {x:g} {y - height/2:g}")
                parts.append("Z")

        # Return the composed absolute-path string or "M 0 0" string if parts is empty
        return " ".join(parts) if parts else "M 0 0"
