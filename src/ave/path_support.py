"""Supporting utilities and constraints for AvPath.

This module contains command metadata, validation helpers, and utility
functions that are used by the core path implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ave.common import AvGlyphCmds

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
        segments = AvPathUtils.split_commands_into_segments(commands)

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
