"""Support utilities for letter spacing calculations.

This module contains geometric calculation utilities for determining
horizontal space between letters using Shapely operations.
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional, Protocol, Tuple

import numpy as np
import shapely
import shapely.affinity
import shapely.geometry
import shapely.ops
import shapely.prepared
from numpy.typing import NDArray

from ave.geom import AvBox
from ave.path import AvPath

# Tolerance factor for space calculation: multiplied by advance_width.
LEFT_SPACE_TOL_FACTOR: float = 0.001

# Polygonization resolution for left_space() calculations.
# Lower value = faster performance, less accuracy. Standard is 50.
LEFT_SPACE_POLYGONIZE_RESOLUTION: int = 10


class LetterProtocol(Protocol):
    """Protocol for letter objects used in spacing calculations.

    Any object exposing these members can participate in
    :class:`LetterSpacing` calculations without depending on
    concrete letter classes.
    """

    @property
    def advance_width(self) -> float:
        """The advance width of the letter."""

    @property
    def bounding_box(self) -> AvBox:
        """The bounding box around the letter's outline."""

    def geometry_path(self) -> Optional[AvPath]:
        """Return the polygonized outline in world coordinates."""


class _GeometryContext(NamedTuple):
    """Pre-built Shapely geometries used during spacing probes.

    Bundling these values avoids passing them as separate
    arguments through the bisection helpers.

    Attributes:
        left_geom: Shapely geometry of the left letter.
        right_geom: Shapely geometry of the right letter.
        left_bounds: ``left_geom.bounds`` cached as
            ``(minx, miny, maxx, maxy)``.
        right_bounds: ``right_geom.bounds`` cached as
            ``(minx, miny, maxx, maxy)``.
        left_prep: Prepared left geometry for faster intersection
            tests (``shapely.prepared.prep(left_geom)``).
        shift_holder: Single-element list whose ``[0]`` entry is
            mutated before each ``shapely.transform`` call to set
            the current horizontal offset.
        shift_fn: Coordinate-transform callable that reads
            ``shift_holder[0]`` and adds it to the x-column.
            Created once per ``space_between`` invocation.
    """

    left_geom: shapely.geometry.base.BaseGeometry
    right_geom: shapely.geometry.base.BaseGeometry
    left_bounds: Tuple[float, float, float, float]
    right_bounds: Tuple[float, float, float, float]
    left_prep: shapely.prepared.PreparedGeometry
    shift_holder: List[float]
    shift_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]]


class LetterSpacing:
    """Utility class for letter spacing calculations using Shapely geometries."""

    @staticmethod
    def letter_to_shapely_geometry(
        letter: LetterProtocol,
        dx_offset: float = 0.0,
        steps: Optional[int] = None,
    ) -> Optional[shapely.geometry.base.BaseGeometry]:
        """Convert a letter to Shapely geometry in world coordinates.

        The glyph path is polygonized, split into contours, and each closed
        exterior contour is converted to a ``shapely.geometry.Polygon``.
        All polygons are combined via ``shapely.ops.unary_union``.
        Interior rings (holes) are ignored.

        Args:
            letter: The letter to convert.
            dx_offset: Additional horizontal offset in world coordinates
                applied on top of the letter's current position.
            steps: Optional polygonization resolution. If None, uses the
                letter's default polygonized path (standard=50 steps).
                If provided, uses custom resolution for performance/accuracy trade-off.

        Returns:
            A Shapely geometry representing the letter outline, or ``None``
            if the path is empty or has no valid contours.
        """
        world_path = letter.geometry_path()
        if world_path is None:
            return None

        # If custom steps provided, re-polygonize the raw path for performance
        if steps is not None and steps > 0:
            world_path = LetterSpacing._repolygonize_path(world_path, steps)
            if world_path is None:
                return None

        return LetterSpacing._path_to_geometry(world_path, dx_offset)

    @staticmethod
    def _repolygonize_path(
        world_path: AvPath,
        steps: int,
    ) -> Optional[AvPath]:
        """Re-polygonize a world path with custom resolution.

        Extracts the raw (unpolygonized) path from the letter's geometry
        and polygonizes it with the specified steps for performance tuning.

        Args:
            world_path: The world path to re-polygonize.
            steps: Number of segments for curve approximation.

        Returns:
            Re-polygonized path or None if empty.
        """
        # Access the raw underlying path before polygonization
        # The world_path may be already polygonized; we need to get raw curves
        # and re-polygonize with custom steps
        if world_path.points.size == 0:
            return None

        # Re-polygonize with custom steps if the path has curves
        if world_path.has_curves:
            return world_path.polygonize(steps)

        # No curves - return as-is (already a polyline)
        return world_path

    @staticmethod
    def _path_to_geometry(
        world_path: AvPath,
        dx_offset: float = 0.0,
    ) -> Optional[shapely.geometry.base.BaseGeometry]:
        """Convert an ``AvPath`` in world coordinates to Shapely geometry.

        Args:
            world_path: Polygonized path already in world coordinates.
            dx_offset: Extra horizontal shift in world coordinates.

        Returns:
            Combined Shapely geometry or ``None`` if no valid contours.
        """
        pts = world_path.points[:, :2].copy()
        if dx_offset != 0.0:
            pts[:, 0] += dx_offset

        return LetterSpacing._build_union_from_contours(
            world_path.split_into_single_paths(),
            pts,
        )

    @staticmethod
    def _build_union_from_contours(
        contours: List,
        pts: object,
    ) -> Optional[shapely.geometry.base.BaseGeometry]:
        """Build a unary union of exterior-ring polygons from contours.

        Args:
            contours: List of single-path contours from a polygonized path.
            pts: Transformed 2-D point array (world coordinates).

        Returns:
            Combined Shapely geometry or ``None`` if no valid contours.
        """
        shapely_polys: List[shapely.geometry.Polygon] = []
        pt_idx = 0
        for contour in contours:
            n_pts = contour.points.shape[0]
            is_closed = contour.commands and contour.commands[-1] == "Z"
            if is_closed and n_pts >= 3:
                poly = shapely.geometry.Polygon(
                    pts[pt_idx : pt_idx + n_pts].tolist(),
                )
                if poly.is_valid and not poly.is_empty and poly.area > 1e-12:
                    shapely_polys.append(
                        shapely.geometry.Polygon(poly.exterior.coords),
                    )
            pt_idx += n_pts

        if not shapely_polys:
            return None
        if len(shapely_polys) == 1:
            return shapely_polys[0]
        return shapely.ops.unary_union(shapely_polys)

    @staticmethod
    def space_between(
        left: LetterProtocol,
        right: LetterProtocol,
        tolerance: Optional[float] = None,
    ) -> float:
        """Calculate horizontal space between two letters.

        Determines how far the right letter can be moved in the negative
        x-direction before its outline intersects with the left letter,
        or how far it must move in the positive x-direction if the outlines
        already overlap.

        Args:
            left: The left letter.
            right: The right letter.
            tolerance: Convergence tolerance. If None, defaults to
                0.001 * right.advance_width.

        Returns:
            float: Positive when the right letter can shift left by that
                amount before touching.  Negative when it already overlaps
                and must shift right.  Returns 0.0 when either letter has
                an empty path.
        """
        # Build geometries once -- all subsequent probes use
        # shapely.transform instead of rebuilding from scratch.
        # Use lower polygonization resolution (10 vs 50) for performance.
        left_geom = LetterSpacing.letter_to_shapely_geometry(left, steps=LEFT_SPACE_POLYGONIZE_RESOLUTION)
        right_geom = LetterSpacing.letter_to_shapely_geometry(right, steps=LEFT_SPACE_POLYGONIZE_RESOLUTION)

        if left_geom is None or right_geom is None:
            return 0.0

        # Use tolerance based on right letter's advance width
        if tolerance is None:
            tolerance = LEFT_SPACE_TOL_FACTOR * right.advance_width

        # Mutable holder lets the closure read the current shift
        # without creating a new function object per probe.
        shift_holder: List[float] = [0.0]

        def _shift_fn(
            coords: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            shifted = coords.copy()
            shifted[:, 0] += shift_holder[0]
            return shifted

        # Bundle pre-built geometries for reuse during probes
        ctx = _GeometryContext(
            left_geom=left_geom,
            right_geom=right_geom,
            left_bounds=left_geom.bounds,
            right_bounds=right_geom.bounds,
            left_prep=shapely.prepared.prep(left_geom),
            shift_holder=shift_holder,
            shift_fn=_shift_fn,
        )

        # Check current intersection state
        if right_geom.intersects(left_geom):
            # Case B: already overlapping -- find rightward shift to separate
            return LetterSpacing._calculate_separation_distance(
                ctx,
                tolerance,
                right,
                left,
            )

        # Case A: not intersecting -- find max leftward shift before touching
        return LetterSpacing._calculate_approach_distance(
            ctx,
            tolerance,
            right,
            left,
        )

    @staticmethod
    def _calculate_separation_distance(
        ctx: _GeometryContext,
        tolerance: float,
        right: LetterProtocol,
        left: LetterProtocol,
    ) -> float:
        """Find rightward shift to eliminate intersection (Case B).

        Args:
            ctx: Pre-built geometry context.
            tolerance: Convergence threshold.
            right: The right letter (for bounding_box property).
            left: The left letter (for bounding_box property).

        Returns:
            Negative float (rightward shift magnitude).
        """
        left_bb = left.bounding_box
        right_bb = right.bounding_box
        overlap = left_bb.xmax - right_bb.xmin

        # Better initial estimate: use actual overlap plus safety margin
        # This gets us closer to the separation point in fewer iterations
        hi = abs(overlap) * 1.5

        # Verify upper bound is sufficient
        if LetterSpacing._translated_intersects(ctx, hi):
            hi *= 2.0

        lo, hi = LetterSpacing._bisect_shift(
            (0.0, hi),
            tolerance,
            ctx,
            seek_intersecting=True,
        )
        return -(lo + hi) / 2.0

    @staticmethod
    def _calculate_approach_distance(
        ctx: _GeometryContext,
        tolerance: float,
        right: LetterProtocol,
        left: LetterProtocol,
    ) -> float:
        """Find max leftward shift before intersection (Case A).

        Args:
            ctx: Pre-built geometry context.
            tolerance: Convergence threshold.
            right: The right letter (for bounding_box property).
            left: The left letter (for bounding_box/advance_width).

        Returns:
            Positive float (leftward shift amount) or 0.0.
        """
        # Check bounding box gap first (fast)
        left_bb = left.bounding_box
        right_bb = right.bounding_box
        bbox_gap = right_bb.xmin - left_bb.xmax

        # For large bbox gaps, skip expensive Shapely distance and use bbox_gap directly
        # For small gaps or overlaps, use precise Shapely distance
        if bbox_gap > tolerance * 10:
            # Large separation: bbox_gap is good enough as starting point
            hi = bbox_gap
        else:
            # Small gap or overlap: need accurate Shapely distance
            gap = ctx.right_geom.distance(ctx.left_geom)
            if gap < tolerance:
                return 0.0
            # Use bbox_gap if it's larger (better starting point)
            hi = max(gap, bbox_gap) if bbox_gap > 0 else gap

        # Exponential search to find intersection
        # Limit to avoid infinite loop for vertically disjoint letters
        limit = max(1000.0, (left.advance_width + right.advance_width) * 10.0)

        while hi < limit:
            if LetterSpacing._translated_intersects(ctx, -hi):
                break
            hi = max(hi * 2.0, tolerance)

        # If we exceeded limit without intersection, return the large value (essentially infinity)
        if not LetterSpacing._translated_intersects(ctx, -hi):
            return hi

        lo, hi = LetterSpacing._bisect_shift(
            (0.0, hi),
            tolerance,
            ctx,
            seek_intersecting=False,
        )
        return (lo + hi) / 2.0

    @staticmethod
    def _shifted_intersects(
        letter: LetterProtocol,
        shift: float,
        other_geom: shapely.geometry.base.BaseGeometry,
    ) -> bool:
        """Check if letter shifted by *dx* intersects *other*.

        This is the legacy entry point that rebuilds geometry from
        scratch.  Internal callers should prefer
        :meth:`_translated_intersects` which reuses a pre-built
        geometry.

        Args:
            letter: The letter to shift.
            shift: Horizontal offset in world coordinates.
            other_geom: Reference geometry.

        Returns:
            True if geometries intersect after shifting.
        """
        geom = LetterSpacing.letter_to_shapely_geometry(letter, dx_offset=shift)
        return geom is not None and geom.intersects(other_geom)

    @staticmethod
    def _translated_intersects(
        ctx: _GeometryContext,
        shift: float,
    ) -> bool:
        """Check intersection using ``shapely.transform`` on a pre-built geometry.

        Uses a 2-D bounding-box pre-filter to skip the expensive
        Shapely ``intersects`` call when the shifted right bbox
        cannot overlap the left bbox.  The actual coordinate shift
        is performed via ``shapely.transform`` with a mutable
        closure, which is significantly faster than
        ``shapely.affinity.translate``.

        Args:
            ctx: Pre-built geometry context with cached bounds,
                prepared left geometry, and shift closure.
            shift: Horizontal offset applied to *right_geom*.

        Returns:
            True if the shifted right geometry intersects *left_geom*.
        """
        rb = ctx.right_bounds
        lb = ctx.left_bounds
        # Fast reject: bounding boxes don't overlap horizontally
        if rb[0] + shift > lb[2] or rb[2] + shift < lb[0]:
            return False
        # Fast reject: bounding boxes don't overlap vertically
        if rb[1] > lb[3] or rb[3] < lb[1]:
            return False
        ctx.shift_holder[0] = shift
        shifted = shapely.transform(ctx.right_geom, ctx.shift_fn)
        return ctx.left_prep.intersects(shifted)

    @staticmethod
    def _bisect_shift(
        bounds: Tuple[float, float],
        tolerance: float,
        ctx: _GeometryContext,
        seek_intersecting: bool = False,
    ) -> Tuple[float, float]:
        """Run bisection to refine the intersection boundary.

        When *seek_intersecting* is True the shift is applied in the
        positive-x direction (rightward, Case B).  Otherwise the shift
        is negated (leftward, Case A).

        Uses :meth:`_translated_intersects` with a pre-built right
        geometry for each probe instead of rebuilding from scratch.

        Args:
            bounds: (lo, hi) interval to bisect.
            tolerance: Convergence threshold.
            ctx: Pre-built geometry context.
            seek_intersecting: Direction flag (keyword-only).

        Returns:
            Refined (lo, hi) interval.
        """
        lo, hi = bounds
        sign = 1.0 if seek_intersecting else -1.0
        for _ in range(60):
            if hi - lo < tolerance:
                break
            mid = (lo + hi) / 2.0
            hits = LetterSpacing._translated_intersects(ctx, sign * mid)
            if hits == seek_intersecting:
                lo = mid
            else:
                hi = mid
        return lo, hi
