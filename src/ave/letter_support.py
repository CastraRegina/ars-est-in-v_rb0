"""Support utilities for letter spacing calculations.

This module contains geometric calculation utilities for determining
horizontal space between letters using Shapely operations.
"""

from __future__ import annotations

from typing import List, Optional, Protocol

import numpy as np
import shapely
import shapely.geometry
import shapely.ops
import shapely.prepared
from numpy.typing import NDArray

from ave.geom import AvBox
from ave.path import AvSinglePolygonPath


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

    # @property
    # def centroid(self) -> Tuple[float, float]:
    #     """The centroid of the letter in real dimensions."""

    # def polygonize_path(self, steps: int = 50) -> Optional[AvPath]:
    #     """Return the polygonized outline in world coordinates.

    #     Args:
    #         steps: Number of segments for curve approximation (default: 50).
    #     """

    # @property
    # def exterior_path(self) -> List[AvSinglePolygonPath]:
    #     """Convert glyph outline to one or more polygons without holes using fixed internal steps."""

    @property
    def exterior_path_left_silhouette(self) -> List[AvSinglePolygonPath]:
        """Get left orthographic silhouette of the letter's exterior polygons."""

    @property
    def exterior_path_right_silhouette(self) -> List[AvSinglePolygonPath]:
        """Get right orthographic silhouette of the letter's exterior polygons."""

    # def exterior(self, steps: int = 20) -> List[AvSinglePolygonPath]:
    #     """Get the exterior paths of the letter without holes.

    #     Args:
    #         steps: Number of segments to use for curve approximation during polygonization

    #     Returns:
    #         List of AvSinglePolygonPath objects representing only positive polygons
    #         (exterior rings without any holes) in the letter's coordinate system
    #     """


class LetterSpacing:
    """Letter spacing calculation using bisection algorithm.

    This class provides methods to calculate optimal horizontal spacing between
    letters by finding the transition point where letter outlines intersect.

    The spacing value indicates:
    - Positive: Overlap (right letter must move right by this amount to separate)
    - Negative: Gap (right letter can move left by this amount to touch)
    """

    class _SimpleGeometryContext:
        """Simplified geometry context for performance optimization.

        Caches geometries and provides fast intersection testing.
        """

        def __init__(
            self, left_geom: shapely.geometry.base.BaseGeometry, right_geom: shapely.geometry.base.BaseGeometry
        ):
            self.left_geom = left_geom
            self.right_geom = right_geom
            self.left_prep = shapely.prepared.prep(left_geom)
            self.left_bounds = left_geom.bounds
            self.right_bounds = right_geom.bounds
            # Mutable holder for shift value to avoid array creation
            self.shift_holder = [0.0]

        def test_intersection(self, shift: float) -> bool:
            """Fast intersection test using cached data and optimized transformations.

            Args:
                shift: Horizontal shift amount for right geometry.

            Returns:
                True if geometries intersect after shift, False otherwise.
            """
            # Fast bbox pre-filter using cached bounds
            if self.right_bounds[0] + shift > self.left_bounds[2] or self.right_bounds[2] + shift < self.left_bounds[0]:
                return False
            if self.right_bounds[1] > self.left_bounds[3] or self.right_bounds[3] < self.left_bounds[1]:
                return False

            # Use shapely.transform with in-place coordinate modification
            # This avoids creating new geometry objects
            self.shift_holder[0] = shift
            shifted = shapely.transform(self.right_geom, self._shift_fn)

            # Fast intersection using prepared geometry
            return self.left_prep.intersects(shifted)

        def _shift_fn(self, coords: NDArray[np.float64]) -> NDArray[np.float64]:
            """Optimized shift function that modifies coordinates in-place."""
            # Add shift directly to x coordinates without copying
            coords[:, 0] += self.shift_holder[0]
            return coords

    @staticmethod
    def _build_shapely_geometry_from_silhouette(
        letter: LetterProtocol,
        use_left_silhouette: bool = False,
    ) -> Optional[shapely.geometry.base.BaseGeometry]:
        """Build optimized Shapely geometry from letter silhouette paths.

        Uses the appropriate silhouette property for better spacing accuracy.

        Args:
            letter: Letter protocol object with silhouette properties
            use_left_silhouette: True to use left silhouette, False to use right silhouette

        Returns:
            Combined Shapely geometry or None if no valid contours.
        """
        # Choose the appropriate silhouette based on the side
        if use_left_silhouette:
            silhouette_paths = letter.exterior_path_left_silhouette
        else:
            silhouette_paths = letter.exterior_path_right_silhouette

        polygons = []
        for path in silhouette_paths:
            if len(path.points) >= 3:  # Need at least 3 points for a polygon
                polygons.append(shapely.geometry.Polygon(path.points[:, :2]))

        if not polygons:
            return None

        # Use unary_union for optimized combination (faster than MultiPolygon)
        return shapely.ops.unary_union(polygons)

    @staticmethod
    def find_transition_shift(
        left: Optional[LetterProtocol],
        right: Optional[LetterProtocol],
        tolerance: Optional[float] = None,
        max_iterations: int = 64,
    ) -> float:
        """Calculate horizontal shift needed for optimal letter spacing.

        Uses bisection to find the optimal shift by testing when letter outlines
        intersect as the right letter is shifted horizontally.

        Args:
            left: The left letter.
            right: The right letter.
            tolerance: Absolute tolerance for convergence. If None, uses 0.001 *
                right.advance_width.
            max_iterations: Maximum number of iterations (default: 64).

        Returns:
            Horizontal shift amount:
            - Positive value: Move RIGHT to separate letters (overlap case).
            - Negative value: Move LEFT till letters are touching (gap case).
            - Zero: Letters are exactly touching or one/both letters are None.
        """
        if left is None or right is None:
            return 0.0

        # Use same tolerance calculation as original for compatibility
        if tolerance is None:
            tolerance = 0.001 * right.advance_width

        # Build geometries using silhouette properties for better accuracy
        left_geom = LetterSpacing._build_shapely_geometry_from_silhouette(left, use_left_silhouette=False)
        right_geom = LetterSpacing._build_shapely_geometry_from_silhouette(right, use_left_silhouette=True)

        # Check if we have valid geometries
        if left_geom is None or right_geom is None:
            return 0.0

        # Create geometry context for performance optimization
        ctx = LetterSpacing._SimpleGeometryContext(left_geom, right_geom)

        # Check if exteriors intersect to determine movement direction
        if left_geom.intersects(right_geom):
            # Letters overlap -> need to move right letter RIGHT (positive direction)
            # Use a simple, safe bound based on letter sizes
            max_shift = (left.bounding_box.width + right.bounding_box.width) * 2.0

            # Ensure we don't exceed reasonable limits
            limit = max(1000.0, (left.advance_width + right.advance_width) * 10.0)
            max_shift = min(max_shift, limit)
        else:
            # Letters don't overlap -> can move right letter LEFT (negative direction)
            # Use actual distance for better accuracy
            actual_distance = left_geom.distance(right_geom)
            max_shift = -(actual_distance + left.bounding_box.width)

            # For gap case, ensure bounds are sufficient
            limit = max(1000.0, (left.advance_width + right.advance_width) * 10.0)
            if abs(max_shift) > limit:
                max_shift = -limit

        # Find transition point where intersection status changes
        # max_shift is signed: positive for rightward, negative for leftward search
        # Specialized inline bisect for performance
        lo, hi = 0.0, max_shift

        # Cache test values at boundaries
        test_lo = ctx.test_intersection(lo)
        test_hi = ctx.test_intersection(hi)

        # If no transition exists, return midpoint
        if test_lo == test_hi:
            return max_shift / 2.0

        # Inline bisection loop - works with signed bounds directly
        for _ in range(max_iterations):
            if abs(hi - lo) <= tolerance:
                break

            mid = (lo + hi) / 2.0
            test_mid = ctx.test_intersection(mid)

            # Update bounds to maintain: test(lo) != test(hi)
            if test_mid == test_lo:
                lo = mid
            else:
                hi = mid

        # Return midpoint of refined interval
        return (lo + hi) / 2.0
