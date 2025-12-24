"""Path cleaning and manipulation utilities for vector graphics processing."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import shapely.geometry
from numpy.typing import NDArray
from scipy.spatial import KDTree

from ave.bezier import BezierCurve
from ave.geom import AvPolygon
from ave.path import (
    MULTI_POLYGON_CONSTRAINTS,
    AvMultiPolygonPath,
    AvMultiPolylinePath,
    AvPath,
    AvSinglePolygonPath,
    PathConstraints,
)


###############################################################################
# AvPathCleaner
###############################################################################
class AvPathCleaner:
    """Collection of static path-cleaning utilities."""

    @staticmethod
    def resolve_polygonized_path_intersections(path: AvMultiPolylinePath) -> AvMultiPolygonPath:
        """Resolve self-intersections in a polygonized path with winding direction rules.

        The input path consists of 0..n closed segments. Each segment must end with 'Z',
        and if a segment is not explicitly closed, it will be automatically closed by
        appending a 'Z' command. Segments follow the standard winding rule where:
        - Counter-clockwise (CCW) segments represent positive/additive polygons
        - Clockwise (CW) segments represent subtractive polygons (holes)

        Algorithm Strategy:
        The function resolves complex path intersections by processing polygonized segments
        through Shapely geometric operations, carefully handling winding directions and
        deferring CW polygons until the first CCW polygon is found.

        Step-by-step process:
        1. Split input path into individual segments (sub-paths)
        2. Convert each segment to closed path and then to polygonized format
        3. Store CCW orientation from each closed path for later processing
        4. Apply buffer(0) operation to each polygon to remove self-intersections:
            - buffer(0) cleans up topology and resolves intersections
            - Handles Polygon, MultiPolygon, and GeometryCollection results
            - Skips invalid or empty geometries with warnings
        5. Perform sequential boolean operations with special ordering:
            - Wait for first CCW polygon to initialize the result
            - Defer all CW polygons encountered before first CCW
            - Once first CCW is found, process deferred CW polygons as holes
            - Subsequent CCW polygons are unioned (additive)
            - Subsequent CW polygons are differenced (subtractive)
        6. Handle different geometry types from buffer(0):
            - Polygon: processed directly
            - MultiPolygon: each sub-polygon processed with same orientation
            - GeometryCollection: Polygon types extracted and processed
        7. Convert final Shapely geometry back to AvMultiPolygonPath format:
            - Extract exterior rings as closed paths with 'Z' command
            - Extract interior rings (holes) as separate paths
            - Join all paths using AvPath.join_paths
            - Return result with MULTI_POLYGON_CONSTRAINTS

        Key technical details:
        - Uses orientation from closed path's is_ccw() to determine winding
        - Implements deferred processing for CW polygons before first CCW
        - Comprehensive error handling with fallback to original path
        - Removes duplicate closing points when converting coordinates

        The function handles the following cases:
        - Empty input paths: returns empty AvMultiPolygonPath
        - Degenerate polygons (< 3 points): skips with warning
        - Invalid geometries after buffer(0): skips with warning
        - Different geometry types from buffer(0):
            - Polygon: processed directly
            - MultiPolygon: each sub-polygon processed with same orientation
            - GeometryCollection: Polygon types extracted and processed
        - CW polygons before first CCW: deferred until first CCW is found
        - No CCW polygon found: returns empty AvMultiPolygonPath with warning
        - Empty result after operations: returns empty AvMultiPolygonPath with warning
        - Shapely errors during processing: returns empty AvMultiPolygonPath with warning
        - Errors during geometry conversion: returns empty AvMultiPolygonPath with warning
        - Errors during path joining: returns empty AvMultiPolygonPath with warning

        Args:
            path: An AvMultiPolylinePath containing the segments to process

        Returns:
            AvMultiPolygonPath: A new path with resolved intersections and proper winding,
                                or empty AvMultiPolygonPath with warning if errors occur
        """

        # Split path into individual segments
        segments = path.split_into_single_paths()

        # Store the original first point to preserve it after Shapely operations
        original_first_point = None
        if path.points.shape[0] > 0:
            original_first_point = path.points[0, :2].copy()

        # Process each segment to ensure it's closed
        polygons: List[AvSinglePolygonPath] = []
        orientations: List[bool] = []  # Store CCW orientation from closed paths

        for segment in segments:
            # Create closed path, then get polygonized path
            try:
                closed_path = AvPath.make_closed_single(segment)
                polygonized = closed_path.polygonized_path()
                polygons.append(polygonized)
                orientations.append(closed_path.is_ccw)  # Store orientation from closed path
            except (TypeError, ValueError) as e:
                print(f"Error processing segment: {e}. Skipping.")
                continue

        if not polygons:
            return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)

        # Sequentially combine polygons using the first CCW polygon as base
        # Store early CW polygons to defer them until we find the first CCW
        deferred_cw_polygons: List[shapely.geometry.base.BaseGeometry] = []
        result: Optional[shapely.geometry.base.BaseGeometry] = None
        first_ccw_found = False

        try:
            for polygon, is_ccw in zip(polygons, orientations):
                # Use stored orientation from closed path

                # Skip degenerate polygons
                if polygon.points.shape[0] < 3:
                    print("Warning: Contour has fewer than 3 points. Skipping.")
                    continue

                # Convert to Shapely polygon
                shapely_poly = shapely.geometry.Polygon(polygon.points[:, :2].tolist())

                # Clean intersections with buffer(0)
                try:
                    cleaned = shapely_poly.buffer(0)
                    # Skip if buffer(0) results in empty or invalid geometry
                    if cleaned.is_empty or not cleaned.is_valid:
                        print("Warning: Contour became empty or invalid after buffer(0). Skipping.")
                        continue
                except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
                    print(f"Warning: Failed to clean contour with buffer(0): {e}. Skipping.")
                    continue

                # Handle different geometry types from buffer(0)
                if isinstance(cleaned, shapely.geometry.MultiPolygon):
                    # Process each sub-polygon with the same orientation
                    for sub_poly in cleaned.geoms:
                        if not sub_poly.is_empty:
                            if result is None:
                                # Wait for first CCW polygon to initialize result
                                if is_ccw:
                                    result = sub_poly
                                    first_ccw_found = True
                                    # Now process any deferred CW polygons
                                    for cw_poly in deferred_cw_polygons:
                                        result = result.difference(cw_poly)
                                    deferred_cw_polygons.clear()
                                else:
                                    # Defer CW polygon until we find first CCW
                                    deferred_cw_polygons.append(sub_poly)
                            elif first_ccw_found:
                                # We have a base, now process all polygons
                                if is_ccw:
                                    # CCW polygons are additive
                                    result = result.union(sub_poly)
                                else:
                                    # CW polygons are subtractive (holes)
                                    result = result.difference(sub_poly)
                elif isinstance(cleaned, shapely.geometry.Polygon) and not cleaned.is_empty:
                    if result is None:
                        # Wait for first CCW polygon to initialize result
                        if is_ccw:
                            result = cleaned
                            first_ccw_found = True
                            # Now process any deferred CW polygons
                            for cw_poly in deferred_cw_polygons:
                                result = result.difference(cw_poly)
                            deferred_cw_polygons.clear()
                        else:
                            # Defer CW polygon until we find first CCW
                            deferred_cw_polygons.append(cleaned)
                    elif first_ccw_found:
                        # We have a base, now process all polygons
                        if is_ccw:
                            # CCW polygons are additive
                            result = result.union(cleaned)
                        else:
                            # CW polygons are subtractive (holes)
                            result = result.difference(cleaned)
                elif isinstance(cleaned, shapely.geometry.GeometryCollection):
                    # Extract Polygon types from GeometryCollection
                    for geom in cleaned.geoms:
                        if isinstance(geom, shapely.geometry.Polygon) and not geom.is_empty:
                            if result is None:
                                # Wait for first CCW polygon to initialize result
                                if is_ccw:
                                    result = geom
                                    first_ccw_found = True
                                    # Now process any deferred CW polygons
                                    for cw_poly in deferred_cw_polygons:
                                        result = result.difference(cw_poly)
                                    deferred_cw_polygons.clear()
                                else:
                                    # Defer CW polygon until we find first CCW
                                    deferred_cw_polygons.append(geom)
                            elif first_ccw_found:
                                # We have a base, now process all polygons
                                if is_ccw:
                                    # CCW polygons are additive
                                    result = result.union(geom)
                                else:
                                    # CW polygons are subtractive (holes)
                                    result = result.difference(geom)
                # Skip empty geometries

            # If no CCW polygon was found, return empty path
            if result is None or not first_ccw_found:
                print("Warning: No CCW polygon found. Returning empty path.")
                return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)

            if result.is_empty:
                print("Warning: Result is empty after operations. Returning empty path.")
                return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)

        except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
            print(f"Error during polygon processing: {e}. Returning empty path.")
            return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)

        # Convert final Shapely geometry back to AvMultiPolygonPath
        cleaned_paths: List[AvSinglePolygonPath] = []

        # Helper function to rotate coordinates to start from original first point
        def rotate_to_start_point(
            coords: List[Tuple[float, float]], start_point: Optional[np.ndarray]
        ) -> List[Tuple[float, float]]:
            """Rotate coordinate list to start from the point closest to start_point."""
            if start_point is None or not coords:
                return coords

            # Find index of point closest to original start point
            distances = [np.linalg.norm(np.array(coord) - start_point) for coord in coords]
            min_idx = distances.index(min(distances))

            # Rotate list to start from min_idx
            return coords[min_idx:] + coords[:min_idx]

        try:
            if isinstance(result, shapely.geometry.Polygon) and not result.is_empty:
                # Convert exterior
                exterior_coords = list(result.exterior.coords)
                if len(exterior_coords) >= 4:
                    exterior_coords = exterior_coords[:-1]  # Remove closing point
                    if len(exterior_coords) >= 3:  # Need at least 3 points for a polygon
                        # Enforce CCW for exterior rings (positive polygons)
                        was_reversed = False
                        if not AvPolygon.is_ccw(np.asarray(exterior_coords)):
                            exterior_coords = list(reversed(exterior_coords))
                            was_reversed = True
                        # Rotate to start from original first point
                        exterior_coords = rotate_to_start_point(exterior_coords, original_first_point)
                        # If we reversed for CCW, we need to rotate again since reversal changed start
                        if was_reversed:
                            # After reversing, the point closest to original might be at different position
                            exterior_coords = rotate_to_start_point(exterior_coords, original_first_point)
                        exterior_cmds = ["M"] + ["L"] * (len(exterior_coords) - 1) + ["Z"]
                        cleaned_paths.append(AvPath(exterior_coords, exterior_cmds))

                # Convert interiors (holes)
                for interior in result.interiors:
                    interior_coords = list(interior.coords)
                    if len(interior_coords) >= 4:
                        interior_coords = interior_coords[:-1]  # Remove closing point
                        if len(interior_coords) >= 3:  # Need at least 3 points for a polygon
                            # Enforce CW for interior rings (holes)
                            if AvPolygon.is_ccw(np.asarray(interior_coords)):
                                interior_coords = list(reversed(interior_coords))
                            interior_cmds = ["M"] + ["L"] * (len(interior_coords) - 1) + ["Z"]
                            cleaned_paths.append(AvPath(interior_coords, interior_cmds))

            elif isinstance(result, shapely.geometry.MultiPolygon):
                # Handle MultiPolygon result
                for poly in result.geoms:
                    if not poly.is_empty:
                        # Convert exterior
                        exterior_coords = list(poly.exterior.coords)
                        if len(exterior_coords) >= 4:
                            exterior_coords = exterior_coords[:-1]
                            if len(exterior_coords) >= 3:  # Need at least 3 points for a polygon
                                # Enforce CCW for exterior rings (positive polygons)
                                was_reversed = False
                                if not AvPolygon.is_ccw(np.asarray(exterior_coords)):
                                    exterior_coords = list(reversed(exterior_coords))
                                    was_reversed = True
                                # Rotate to start from original first point
                                exterior_coords = rotate_to_start_point(exterior_coords, original_first_point)
                                # If we reversed for CCW, we need to rotate again since reversal changed start
                                if was_reversed:
                                    # After reversing, the point closest to original might be at different position
                                    exterior_coords = rotate_to_start_point(exterior_coords, original_first_point)
                                exterior_cmds = ["M"] + ["L"] * (len(exterior_coords) - 1) + ["Z"]
                                cleaned_paths.append(AvPath(exterior_coords, exterior_cmds))

                        # Convert interiors
                        for interior in poly.interiors:
                            interior_coords = list(interior.coords)
                            if len(interior_coords) >= 4:
                                interior_coords = interior_coords[:-1]
                                if len(interior_coords) >= 3:  # Need at least 3 points for a polygon
                                    # Enforce CW for interior rings (holes)
                                    if AvPolygon.is_ccw(np.asarray(interior_coords)):
                                        interior_coords = list(reversed(interior_coords))
                                    interior_cmds = ["M"] + ["L"] * (len(interior_coords) - 1) + ["Z"]
                                    cleaned_paths.append(AvPath(interior_coords, interior_cmds))
        except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
            print(f"Error during geometry conversion: {e}. Returning empty path.")
            return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)

        # Join all paths and return as AvMultiPolygonPath with MULTI_POLYGON_CONSTRAINTS
        if cleaned_paths:
            try:
                joined = AvPath.join_paths(*cleaned_paths)
                # Return with MULTI_POLYGON_CONSTRAINTS
                return AvMultiPolygonPath(joined.points, joined.commands, MULTI_POLYGON_CONSTRAINTS)
            except (TypeError, ValueError) as e:
                print(f"Error during path joining: {e}. Returning empty path.")
                return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)
        else:
            print("Warning: No valid paths to join. Returning empty path.")
            return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)


###############################################################################
# AvPathMatcher
###############################################################################
class AvPathMatcher:
    """Transfer type information from original path points to new path points.

    This class matches points between an original path (with type information)
    and a new path (without type information), typically used after intersection
    resolution where new points may have been introduced.

    The matching uses segment-wise correspondence with KD-tree spatial search
    to efficiently find matching points within a small tolerance.
    """

    TOLERANCE: float = 1e-10
    UNMATCHED_TYPE: float = -1.0

    @classmethod
    def match_points(cls, points_org: NDArray[np.float64], points_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transfer type information from original points to new points.

        Args:
            points_org: Original points as array (N, 3)
                with columns (x, y, type). Type values are in {0.0, 2.0, 3.0}.
            points_new: New points as array (M, 2) or (M, 3)
                with columns (x, y) or (x, y, ignored).

        Returns:
            NDArray of shape (M, 3) with columns (x, y, type) where type is:
            - The matched type from points_org if found within tolerance
            - -1.0 if no match found (typically segment jump points)

        Raises:
            ValueError: If input arrays have invalid shapes.
        """
        org_arr = cls._validate_points_array(points_org, require_type=True)
        new_arr = cls._validate_points_array(points_new, require_type=False)

        cls._warn_duplicate_neighbors(org_arr, "points_org")
        cls._warn_duplicate_neighbors(new_arr, "points_new")

        return cls._transfer_types(org_arr, new_arr)

    @classmethod
    def match_paths(cls, path_org: AvMultiPolylinePath, path_new: AvMultiPolylinePath) -> AvMultiPolylinePath:
        """Transfer type information between paths, returning a new path.

        This is a convenience method that wraps match_points and returns
        an AvMultiPolylinePath with the matched type information.

        Args:
            path_org: Original path with type information.
            path_new: New path to receive type information.

        Returns:
            New AvMultiPolylinePath with matched type information.
        """
        matched_points = cls.match_points(path_org.points, path_new.points)
        return AvMultiPolylinePath(
            matched_points,
            list(path_new.commands),
            PathConstraints.from_attributes(
                allows_curves=False,
                max_segments=None,
                must_close=False,
                min_points_per_segment=None,
            ),
        )

    @classmethod
    def _validate_points_array(
        cls,
        data: NDArray[np.float64],
        require_type: bool,
    ) -> NDArray[np.float64]:
        """Validate points array shape.

        Args:
            data: Input as numpy array.
            require_type: If True, array must have shape (N, 3).

        Returns:
            NDArray of shape (N, 2) or (N, 3) depending on require_type.

        Raises:
            ValueError: If array shape is invalid.
        """
        arr = np.asarray(data, dtype=np.float64)

        if arr.ndim != 2:
            raise ValueError(f"Points array must be 2D, got {arr.ndim}D")

        if require_type:
            if arr.shape[1] != 3:
                raise ValueError(f"Original points must have shape (N, 3), got {arr.shape}")
        else:
            if arr.shape[1] not in (2, 3):
                raise ValueError(f"New points must have shape (M, 2) or (M, 3), got {arr.shape}")

        return arr

    @classmethod
    def _warn_duplicate_neighbors(
        cls,
        points: NDArray[np.float64],
        name: str,
    ) -> None:
        """Warn if consecutive points have identical coordinates.

        Args:
            points: Points array of shape (N, 2) or (N, 3).
            name: Name of the array for warning messages.
        """
        if points.shape[0] < 2:
            return

        xy = points[:, :2]
        diffs = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        duplicates = np.where(diffs < cls.TOLERANCE)[0]

        for idx in duplicates:
            print(f"Warning: {name} has duplicate neighboring points at " f"indices {idx} and {idx + 1}: {xy[idx]}")

    @classmethod
    def _transfer_types(
        cls,
        org: NDArray[np.float64],
        new: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Transfer type information using KD-tree and segment propagation.

        Args:
            org: Original points array of shape (N, 3) with (x, y, type).
            new: New points array of shape (M, 2) or (M, 3).

        Returns:
            NDArray of shape (M, 3) with (x, y, type).
        """
        n_new = new.shape[0]
        if n_new == 0:
            return np.empty((0, 3), dtype=np.float64)

        n_org = org.shape[0]
        if n_org == 0:
            result = np.empty((n_new, 3), dtype=np.float64)
            result[:, :2] = new[:, :2]
            result[:, 2] = cls.UNMATCHED_TYPE
            return result

        result = np.empty((n_new, 3), dtype=np.float64)
        result[:, :2] = new[:, :2]
        result[:, 2] = cls.UNMATCHED_TYPE

        matched = np.zeros(n_new, dtype=bool)

        org_xy = org[:, :2]
        new_xy = new[:, :2]
        org_types = org[:, 2]

        tree = KDTree(org_xy)

        distances, indices = tree.query(new_xy, distance_upper_bound=cls.TOLERANCE)

        direct_matches = distances < cls.TOLERANCE
        for i in range(n_new):
            if direct_matches[i]:
                result[i, 2] = org_types[indices[i]]
                matched[i] = True

        cls._propagate_matches(result, matched, org_xy, org_types, new_xy, tree)

        return result

    @classmethod
    def _propagate_matches(
        cls,
        result: NDArray[np.float64],
        matched: NDArray[np.bool_],
        org_xy: NDArray[np.float64],
        org_types: NDArray[np.float64],
        new_xy: NDArray[np.float64],
        tree: KDTree,
    ) -> None:
        """Propagate type assignments along segment correspondences.

        When a match is found, attempt to propagate in both directions
        along the segment to find additional matches. This exploits
        the fact that segments often match partially, even if reversed.

        Args:
            result: Result array to update in place.
            matched: Boolean array tracking matched points.
            org_xy: Original XY coordinates.
            org_types: Original type values.
            new_xy: New XY coordinates.
            tree: KD-tree built from org_xy.
        """
        n_new = new_xy.shape[0]
        n_org = org_xy.shape[0]

        match_indices = np.where(matched)[0]

        for new_idx in match_indices:
            dist, org_idx = tree.query(new_xy[new_idx])
            if dist >= cls.TOLERANCE:
                continue

            cls._propagate_direction(
                result, matched, org_xy, org_types, new_xy, new_idx, org_idx, 1, n_new, n_org, tree
            )
            cls._propagate_direction(
                result, matched, org_xy, org_types, new_xy, new_idx, org_idx, -1, n_new, n_org, tree
            )

    @classmethod
    def _propagate_direction(
        cls,
        result: NDArray[np.float64],
        matched: NDArray[np.bool_],
        org_xy: NDArray[np.float64],
        org_types: NDArray[np.float64],
        new_xy: NDArray[np.float64],
        start_new: int,
        start_org: int,
        direction: int,
        n_new: int,
        n_org: int,
        tree: KDTree,
    ) -> None:
        """Propagate matches in a single direction along segments.

        Tries both forward and reversed segment directions in the original
        array to handle partial reversals.

        Args:
            result: Result array to update in place.
            matched: Boolean array tracking matched points.
            org_xy: Original XY coordinates.
            org_types: Original type values.
            new_xy: New XY coordinates.
            start_new: Starting index in new array.
            start_org: Starting index in original array.
            direction: +1 for forward, -1 for backward in new array.
            n_new: Length of new array.
            n_org: Length of original array.
            tree: KD-tree built from org_xy.
        """
        for org_direction in (1, -1):
            new_idx = start_new + direction
            org_idx = start_org + org_direction

            while 0 <= new_idx < n_new and 0 <= org_idx < n_org:
                if matched[new_idx]:
                    break

                dist = np.linalg.norm(new_xy[new_idx] - org_xy[org_idx])
                if dist < cls.TOLERANCE:
                    result[new_idx, 2] = org_types[org_idx]
                    matched[new_idx] = True
                    new_idx += direction
                    org_idx += org_direction
                else:
                    query_dist, query_idx = tree.query(new_xy[new_idx])
                    if query_dist < cls.TOLERANCE:
                        result[new_idx, 2] = org_types[query_idx]
                        matched[new_idx] = True
                        org_idx = query_idx + org_direction
                        new_idx += direction
                    else:
                        break


class AvPathCurveRebuilder:
    """Rebuild polygon paths by replacing point clusters with SVG Bezier curves.

    This class provides functionality to convert polygonized paths back to
    smooth Bezier curves using least-squares fitting. Input paths must be
    AvMultiPolygonPath or AvSinglePolygonPath with properly closed segments.
    """

    @staticmethod
    def rebuild_curve_path(path: AvMultiPolylinePath) -> AvPath:
        """Rebuild a polygon path by replacing point clusters with Bezier curves.

        Iterates through the input path linearly, replacing clusters of sampled
        curve points (type=2 for quadratic, type=3 for cubic) with proper SVG
        Bezier curve commands (Q, C) using least-squares fitting.

        Args:
            path: Input polygon path with point type annotations.
                    Must have segments starting with M and ending with Z.

        Returns:
            New AvPath with Bezier curves replacing point clusters.
            Non-curve vertices are preserved exactly.
        """
        points = path.points
        commands = path.commands

        # Handle empty path
        if points.shape[0] == 0:
            return AvPath()

        # Output buffers
        out_points: List[NDArray[np.float64]] = []
        out_commands: List[str] = []

        # State tracking
        n_points = points.shape[0]
        point_idx = 0
        cmd_idx = 0
        segment_start_point: Optional[NDArray[np.float64]] = None

        while cmd_idx < len(commands):
            cmd = commands[cmd_idx]

            if cmd == "M":
                # Start of new segment
                segment_start_point = points[point_idx].copy()
                out_points.append(points[point_idx].copy())
                out_commands.append("M")
                point_idx += 1
                cmd_idx += 1

            elif cmd == "L":
                pt = points[point_idx]
                pt_type = pt[2]

                if pt_type in (0.0, -1.0):
                    # Regular vertex - emit as L
                    out_points.append(pt.copy())
                    out_commands.append("L")
                    point_idx += 1
                    cmd_idx += 1

                elif pt_type in (2.0, 3.0):
                    # Start of a Bezier cluster
                    cluster_type = pt_type

                    # The predecessor point is the last emitted point
                    if not out_points:
                        raise ValueError("Bezier cluster without predecessor point")
                    predecessor = out_points[-1]

                    # Collect all consecutive points of the same cluster type
                    cluster_points: List[NDArray[np.float64]] = []
                    while point_idx < n_points and cmd_idx < len(commands):
                        if commands[cmd_idx] != "L":
                            break
                        current_pt = points[point_idx]
                        if current_pt[2] != cluster_type:
                            break
                        cluster_points.append(current_pt.copy())
                        point_idx += 1
                        cmd_idx += 1

                    # Determine the successor (endpoint)
                    # Check if next command is Z (closure) or another point
                    if cmd_idx < len(commands) and commands[cmd_idx] == "Z":
                        # Cluster ends at segment closure - use segment start as endpoint
                        successor = segment_start_point
                    elif point_idx < n_points and commands[cmd_idx] == "L":
                        # Next point is the endpoint
                        successor = points[point_idx].copy()
                        # Consume the endpoint
                        point_idx += 1
                        cmd_idx += 1
                    else:
                        raise ValueError("Bezier cluster without successor point")

                    # Build sample list: [predecessor, cluster_points..., successor]
                    sample_list = [predecessor] + cluster_points + [successor]
                    sample_array = np.array(sample_list, dtype=np.float64)

                    # Call appropriate approximation method
                    if cluster_type == 2.0:
                        # Quadratic Bezier
                        approx = BezierCurve.approximate_quadratic_control_points(sample_array)
                        # approx shape: (3, 3) = [start, ctrl, end]
                        # Skip start (already emitted), emit ctrl and end
                        ctrl_pt = approx[1].copy()
                        end_pt = approx[2].copy()
                        out_points.append(ctrl_pt)
                        out_points.append(end_pt)
                        out_commands.append("Q")
                    else:
                        # Cubic Bezier (cluster_type == 3.0)
                        approx = BezierCurve.approximate_cubic_control_points(sample_array)
                        # approx shape: (4, 3) = [start, ctrl1, ctrl2, end]
                        # Skip start (already emitted), emit ctrl1, ctrl2, end
                        ctrl1_pt = approx[1].copy()
                        ctrl2_pt = approx[2].copy()
                        end_pt = approx[3].copy()
                        out_points.append(ctrl1_pt)
                        out_points.append(ctrl2_pt)
                        out_points.append(end_pt)
                        out_commands.append("C")
                else:
                    # Unknown type - treat as regular vertex
                    out_points.append(pt.copy())
                    out_commands.append("L")
                    point_idx += 1
                    cmd_idx += 1

            elif cmd == "Z":
                # Segment closure
                out_commands.append("Z")
                cmd_idx += 1

            else:
                # Skip unknown commands
                cmd_idx += 1

        # Build output path
        if out_points:
            points_array = np.array(out_points, dtype=np.float64)
            return AvPath(points_array, out_commands)
        return AvPath()
