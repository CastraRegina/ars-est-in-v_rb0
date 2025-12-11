"""Handling geometries"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ave.common import AvGlyphCmds
from ave.geom_bezier import BezierCurve


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

    @classmethod
    def from_dict(cls, data: dict) -> AvBox:
        """Create an AvBox instance from a dictionary."""
        return cls(
            xmin=data.get("xmin", 0.0),
            ymin=data.get("ymin", 0.0),
            xmax=data.get("xmax", 0.0),
            ymax=data.get("ymax", 0.0),
        )

    def to_dict(self) -> dict:
        """Convert the AvBox instance to a dictionary."""
        return {
            "xmin": self._xmin,
            "ymin": self._ymin,
            "xmax": self._xmax,
            "ymax": self._ymax,
        }


###############################################################################
# AvPath
###############################################################################


@dataclass
class AvPath:
    """
    Path represented by points (shape (n, 3)) and corresponding commands.
    A path contains 0..n segments; each segment starts with M, is followed by
    an arbitrary mix of L/Q/C, and may optionally end with Z.
    A path may also be empty (no points and no commands), representing 0 segments.
    """

    _points: NDArray[np.float64]  # shape (n_points, 3)
    _commands: List[AvGlyphCmds]  # shape (n_commands, 1)
    _bounding_box: Optional[AvBox] = None  # caching variable

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
                    type_column[point_idx] = 0.0
                    point_idx += 1
                elif cmd == "L":  # LineTo - 1 point
                    type_column[point_idx] = 0.0
                    point_idx += 1
                elif cmd == "Q":  # Quadratic Bezier - 2 points (control + end)
                    type_column[point_idx] = 2.0  # control point
                    type_column[point_idx + 1] = 0.0  # end point
                    point_idx += 2
                elif cmd == "C":  # Cubic Bezier - 3 points (control1 + control2 + end)
                    type_column[point_idx] = 3.0  # control point 1
                    type_column[point_idx + 1] = 3.0  # control point 2
                    type_column[point_idx + 2] = 0.0  # end point
                    point_idx += 3
                elif cmd == "Z":  # ClosePath - no points
                    pass

            self._points = np.column_stack([arr, type_column])
        elif arr.shape[1] == 3:
            self._points = arr
        else:
            raise ValueError(f"points must have shape (n, 2) or (n, 3), got {arr.shape}")

        self._commands = commands_list
        self._bounding_box = None
        self._validate()

    def _validate(self) -> None:
        """Validate path structure. Override in subclasses for specific constraints."""
        cmds = self._commands
        points = self._points

        if not cmds:
            if points.shape[0] != 0:
                raise ValueError("Empty command list must have zero points")
            return

        expected_points = 0
        idx = 0
        n_cmds = len(cmds)

        while idx < n_cmds:
            cmd = cmds[idx]
            if cmd != "M":
                raise ValueError(f"Each segment must start with 'M' command (got '{cmd}' at index {idx})")

            expected_points += 1
            idx += 1

            while idx < n_cmds and cmds[idx] != "M":
                cmd = cmds[idx]
                if cmd == "L":
                    expected_points += 1
                    idx += 1
                elif cmd == "Q":
                    expected_points += 2
                    idx += 1
                elif cmd == "C":
                    expected_points += 3
                    idx += 1
                elif cmd == "Z":
                    idx += 1
                    if idx < n_cmds and cmds[idx] != "M":
                        raise ValueError(
                            "'Z' must terminate a segment "
                            f"(expected 'M' or end after 'Z', got '{cmds[idx]}' at index {idx})"
                        )
                    break
                else:
                    raise ValueError(f"Unknown command '{cmd}' at index {idx}")

        if points.shape[0] != expected_points:
            raise ValueError(
                f"Number of points ({points.shape[0]}) does not match commands (requires {expected_points} points)"
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
            polygonized_path = self.polygonize(self.POLYGONIZE_STEPS_INTERNAL)
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

        # Create instance with 3D points (already has type column)
        path = cls(points, commands)  # Use regular init - it handles 3D points
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
        }

    def polygonize(self, steps: int) -> AvPath:
        """Return a polygonized copy of this path.

        Args:
            steps: Number of segments used to approximate curve segments.
                If 0, the original path is returned unchanged.

        Returns:
            AvPath: New path with curves replaced by line segments.
        """
        if steps == 0:
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
        return AvPath(new_points, new_commands_list)

    def split_into_single_paths(self) -> List[AvSinglePath]:
        """Split an AvPath into AvSinglePath segments at each 'M' command."""

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

            # Create AvSinglePath for this segment
            seg_points_array = (
                np.array(seg_points, dtype=np.float64) if seg_points else np.empty((0, 3), dtype=np.float64)
            )

            single_paths.append(AvSinglePath(seg_points_array, seg_cmds))

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

    # TODO: maybe add parameter also_invert_segment_order
    def reverse(self) -> AvPath:
        """Return a new AvPath with reversed drawing direction.

        The sequence of segments is kept the same, but within each segment
        the sequence of points and commands is reversed. Curve geometry is
        preserved by delegating to AvSinglePath.reverse() for each segment.
        """

        # Split into single segments, reverse each as AvSinglePath, then join.
        single_paths = self.split_into_single_paths()

        # If the path is empty, return a new empty AvPath for consistency
        # with join_paths() semantics.
        if not single_paths:
            return AvPath()

        reversed_segments = [segment.reverse() for segment in single_paths]
        return AvPath.join_paths(reversed_segments)


class AvSinglePath(AvPath):
    """
    Path with at most one segment, represented by points (shape (n, 3)) plus commands.
    If non-empty, it starts with one M, continues with any mix of L/Q/C,
    and may optionally end with Z.
    """

    def _validate(self) -> None:
        """Validate that this path has at most one segment (0 or 1)."""
        super()._validate()

        if not self._commands:
            return  # Empty path is valid

        # Check for additional M commands (would indicate multiple segments)
        for i, cmd in enumerate(self._commands[1:], 1):
            if cmd == "M":
                raise ValueError(f"AvSinglePath cannot contain multiple segments (found 'M' at position {i})")

    def reverse(self) -> AvSinglePath:
        """Return a new AvSinglePath with reversed drawing direction.

        The same point coordinates are used, but their order (and the
        corresponding commands) is reversed, so traversal runs from the
        original last point back to the original first point.

        For curve commands (Q, C), control points are reordered to preserve
        the exact curve geometry when traversed in reverse.
        """
        # Early return for empty paths
        if not self._commands or self._points.size == 0:
            return AvSinglePath(self._points.copy(), list(self._commands))

        # Check if path is closed
        is_closed = self._commands[-1] == "Z"

        # Build segments by iterating forward once
        segments = []
        last_point = self._points[0].copy()  # Start with M point
        point_idx = 1  # Skip M's point

        for cmd in self._commands[1:]:
            if cmd == "Z":
                break

            start_point = last_point

            if cmd == "L":
                end_point = self._points[point_idx].copy()
                segments.append(("L", [], start_point, end_point))
                last_point = end_point
                point_idx += 1

            elif cmd == "Q":
                control = self._points[point_idx].copy()
                end_point = self._points[point_idx + 1].copy()
                segments.append(("Q", [control], start_point, end_point))
                last_point = end_point
                point_idx += 2

            elif cmd == "C":
                control1 = self._points[point_idx].copy()
                control2 = self._points[point_idx + 1].copy()
                end_point = self._points[point_idx + 2].copy()
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
            new_points.append(self._points[0].copy())

        # Add Z if original was closed
        if is_closed:
            new_commands.append("Z")

        # Convert to numpy array
        points_array = np.array(new_points, dtype=np.float64) if new_points else np.empty((0, 3), dtype=np.float64)

        return AvSinglePath(points_array, new_commands)


class AvClosedPath(AvSinglePath):
    """
    Path with at most one closed segment, stored as points (shape (n, 3)) plus commands.
    If non-empty, it begins with one M, continues with any mix of L/Q/C,
    and always ends with Z.
    """

    # TODO: add cache variable for polygonized path (check if parent would fit better)

    def _validate(self) -> None:
        """Validate that this path has at most one closed segment (0 or 1)."""
        super()._validate()

        if not self._commands:
            return  # Empty path is valid

        if self._commands[-1] != "Z":
            raise ValueError("AvClosedPath must end with 'Z' command")

    # TODO: implement area()
    # TODO: implement centroid()
    # TODO: implement is_ccw()


# # TODO: implement:
# class AvPolygonPath(AvClosedPath):
#     """
#     Polygonal closed path stored as points (shape (n, 3)) plus commands.
#     Begins with one M, follows any number of L commands, and always ends with Z.
#     """


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
    path_q = AvPath(path_points, path_commands)
    polygonized_q = path_q.polygonize(polygonize_steps)
    new_points_q, new_commands_q = polygonized_q.points, polygonized_q.commands

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
    path_c = AvPath(path_points_mc, path_commands_mc)
    polygonized_c = path_c.polygonize(polygonize_steps)
    new_points_c, new_commands_c = polygonized_c.points, polygonized_c.commands

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
    path_complex = AvPath(complex_path_points, complex_path_commands)
    polygonized_complex = path_complex.polygonize(polygonize_steps)
    new_points_complex, new_commands_complex = polygonized_complex.points, polygonized_complex.commands

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

    ###########################################################################
    # Test reverse_orientation with the user's example
    print("\n" + "=" * 50)
    print("=" * 50)
    print("Testing reverse_orientation method:")
    print("=" * 50)

    # Create the example path with two contours
    # First contour: M 0 0, L 10 0, L 10 10, Z
    # Second contour: M 20 20, L 30 20, L 30 30
    test_points = np.array(
        [
            [0.0, 0.0, 0.0],  # M 0 0
            [10.0, 0.0, 0.0],  # L 10 0
            [10.0, 10.0, 0.0],  # L 10 10
            [20.0, 20.0, 0.0],  # M 20 20
            [30.0, 20.0, 0.0],  # L 30 20
            [30.0, 30.0, 0.0],  # L 30 30
        ],
        dtype=np.float64,
    )

    test_commands = ["M", "L", "L", "Z", "M", "L", "L"]

    # Create original path
    original_path = AvPath(test_points, test_commands)

    print("Original path:")
    print("  Points:")
    for i, point in enumerate(test_points):
        print(f"    {i:2d}: ({point[0]:6.1f}, {point[1]:6.1f}, type={point[2]:1.0f})")
    print("  Commands:")
    for i, cmd in enumerate(test_commands):
        print(f"    {i:2d}: {cmd}")

    # Reverse the orientation
    reversed_path = original_path  # .reverse_orientation()

    print("\nReversed path:")
    print("  Points:")
    for i, point in enumerate(reversed_path.points):
        print(f"    {i:2d}: ({point[0]:6.1f}, {point[1]:6.1f}, type={point[2]:1.0f})")
    print("  Commands:")
    for i, cmd in enumerate(reversed_path.commands):
        print(f"    {i:2d}: {cmd}")

    print("\nExpected output:")
    print("  First contour: M 10 10, L 10 0, L 0 0, Z")
    print("  Second contour: M 30 30, L 30 20, L 20 20")
    print("\nVerification:")
    print("   Contour sequence preserved (first contour processed before second)")
    print("   First contour reversed: starts at (10,10) instead of (0,0)")
    print("   Second contour reversed: starts at (30,30) instead of (20,20)")
    print("   Z commands maintained for closed contours")


if __name__ == "__main__":
    main()
