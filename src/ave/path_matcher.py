"""Path matching module for transferring type information between polygon paths.

This module provides functionality to transfer point type information from an
original path to a new path that may have been modified by intersection resolution.

After resolve_polygonized_path_intersections():
- Segments may be split, merged, or completely reorganized
- Winding directions are enforced (CCW for exteriors, CW for holes)
- New intersection points may be introduced
- Original points should remain at the same coordinates

The matching algorithm uses spatial proximity (KD-tree) to find corresponding
points within a tight tolerance, regardless of segment structure or order.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from ave.path import AvMultiPolygonPath, AvPath
from ave.path_support import MULTI_POLYGON_CONSTRAINTS


class AvPathMatcher:  # pylint: disable=too-few-public-methods
    """Transfer type information between polygon paths using spatial point matching.

    This class matches points between an original path (with type information)
    and a new path (typically output from resolve_polygonized_path_intersections),
    transferring type values from original to new based on coordinate proximity.

    After intersection resolution:
    - Segments may be split, merged, or reorganized
    - Winding directions may change (original could be any, new is enforced)
    - Points that existed in original should be at same coordinates
    - New intersection points will not match and remain UNMATCHED

    The algorithm:
    1. Build KD-tree from ALL original points
    2. For each new point, find nearest original point
    3. If within tolerance, copy type; otherwise mark as UNMATCHED
    """

    TOLERANCE: float = 1e-10
    UNMATCHED_TYPE: float = -1.0

    @classmethod
    def match_paths(
        cls,
        path_org: AvMultiPolygonPath,
        path_new: AvMultiPolygonPath,
    ) -> AvMultiPolygonPath:
        """Transfer type information from original path to new path.

        Matches points by spatial proximity using KD-tree search. Points in
        path_new that are within TOLERANCE of points in path_org receive the
        corresponding type value. Other points are marked as UNMATCHED_TYPE.

        This method makes NO assumptions about:
        - Segment correspondence (segments may be split/merged/reordered)
        - Winding direction (CCW vs CW is irrelevant for matching)
        - Point order within segments (may be reversed or rotated)

        Args:
            path_org: Original path with type information in 3rd column of points.
                Type values are typically {0.0, 2.0, 3.0} for vertex types.
            path_new: New path to receive type information.
                Typically output from resolve_polygonized_path_intersections().

        Returns:
            New AvMultiPolygonPath with matched type information.
            Points that match get their type from path_org.
            Points without a match get UNMATCHED_TYPE (-1.0).
        """
        # Handle empty paths
        if len(path_new.points) == 0:
            return AvPath(
                np.empty((0, 3), dtype=np.float64),
                list(path_new.commands),
                MULTI_POLYGON_CONSTRAINTS,
            )

        # Initialize result with new path coordinates and unmatched types
        result_points = path_new.points.copy()
        result_points[:, 2] = cls.UNMATCHED_TYPE

        # If no original points, all new points are unmatched
        if len(path_org.points) == 0:
            return AvPath(result_points, list(path_new.commands), MULTI_POLYGON_CONSTRAINTS)

        # Extract coordinates and types from original path
        org_xy = path_org.points[:, :2]
        org_types = path_org.points[:, 2]
        new_xy = path_new.points[:, :2]

        # Build KD-tree from all original points for efficient nearest-neighbor search
        tree = KDTree(org_xy)

        # Find nearest original point for each new point
        distances, indices = tree.query(new_xy, k=1)

        # Flatten arrays (query returns 2D even for k=1)
        distances = distances.flatten()
        indices = indices.flatten()

        # Transfer types for points within tolerance
        within_tolerance = distances < cls.TOLERANCE
        result_points[within_tolerance, 2] = org_types[indices[within_tolerance]]

        return AvPath(result_points, list(path_new.commands), MULTI_POLYGON_CONSTRAINTS)
