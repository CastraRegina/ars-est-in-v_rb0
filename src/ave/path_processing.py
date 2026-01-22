"""Path cleaning and manipulation utilities for vector graphics processing."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely.geometry
from numpy.typing import NDArray
from scipy.spatial import KDTree

from ave.bezier import BezierCurve
from ave.geom import AvBox, AvPolygon
from ave.path import (
    MULTI_POLYGON_CONSTRAINTS,
    AvMultiPolygonPath,
    AvMultiPolylinePath,
    AvPath,
    AvSinglePolygonPath,
)
from ave.path_support import PathCommandProcessor

###############################################################################
# AvPathCleaner
###############################################################################


class AvPathCleaner:
    """Collection of static path-cleaning utilities."""

    @staticmethod
    def remove_duplicate_consecutive_points(path: AvMultiPolylinePath, tolerance: float = 1e-9) -> AvMultiPolylinePath:
        """Remove duplicate consecutive points from a polygonized path, prioritizing type=0 points.

        This method is specifically designed for polygonized paths (AvMultiPolylinePath) with only M, L, Z commands.
        Paths containing curves (Q, C commands) are not supported and will raise an error.

        Point types in polygonized paths:
        - type=0: Vertex points (endpoints of original curves, regular line points)
        - type=2: Intermediate points from quadratic curve polygonization
        - type=3: Intermediate points from cubic curve polygonization

        When duplicate consecutive points are found, the algorithm:
        1. Keeps type=0 points (vertices) - these are the important structural points
        2. Removes type=2 and type=3 points (curve samples) when they duplicate type=0 points
        3. Among duplicates of the same type, keeps the first occurrence

        Args:
            path: Polygonized path (AvMultiPolylinePath) with potential duplicate consecutive points
            tolerance: Distance threshold for detecting duplicates (default: 1e-9)

        Returns:
            New AvMultiPolylinePath with duplicate consecutive points removed

        Raises:
            ValueError: If path contains curve commands (Q, C) or has invalid structure
        """
        points = path.points
        commands = path.commands

        # Validate that path contains no curves
        for cmd in commands:
            if cmd in ["Q", "C"]:
                raise ValueError(
                    f"remove_duplicate_consecutive_points only supports polygonized paths (M, L, Z commands). "
                    f"Found curve command '{cmd}'. Use this method only on AvMultiPolylinePath."
                )

        if len(points) == 0:
            return AvMultiPolylinePath(points.copy(), list(commands), path.constraints)

        if len(points) == 1:
            return AvMultiPolylinePath(points.copy(), list(commands), path.constraints)

        # Track which points and commands to keep by iterating through commands
        new_points = []
        new_commands = []
        point_idx = 0
        last_kept_point = None

        for cmd in commands:
            if cmd == "Z":
                # Always keep Z commands
                new_commands.append(cmd)
                continue

            if cmd == "M":
                # Always keep M commands and their points
                pt = points[point_idx]
                new_points.append(pt.copy())
                new_commands.append(cmd)
                last_kept_point = pt[:2]
                point_idx += 1
                continue

            if cmd == "L":
                # Check if this point is duplicate of last kept point
                pt = points[point_idx]
                if last_kept_point is not None:
                    distance = np.sqrt((pt[0] - last_kept_point[0]) ** 2 + (pt[1] - last_kept_point[1]) ** 2)
                    if distance < tolerance:
                        # Points are duplicates - check if we should replace based on type
                        if len(new_points) > 0:
                            last_pt = new_points[-1]
                            last_type = last_pt[2]
                            curr_type = pt[2]

                            # Priority: type=0 > type=2 > type=3
                            if curr_type == 0.0 and last_type != 0.0:
                                # Current is vertex, previous is curve sample - replace
                                new_points[-1] = pt.copy()
                                last_kept_point = pt[:2]
                            # else: keep the previous point

                        # Skip this duplicate command
                        point_idx += 1
                        continue

                # Keep this point
                new_points.append(pt.copy())
                new_commands.append(cmd)
                last_kept_point = pt[:2]
                point_idx += 1
                continue

        # Convert new_points list to numpy array
        if new_points:
            new_points_array = np.array(new_points, dtype=np.float64)
        else:
            new_points_array = np.empty((0, 3), dtype=np.float64)

        return AvMultiPolylinePath(new_points_array, new_commands, path.constraints)

    @staticmethod
    def _analyze_segment(segment: AvPath) -> dict:
        """Analyze a path segment using PathCommandProcessor.

        Args:
            segment: The path segment to analyze

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "has_curves": False,
            "num_drawing_commands": 0,
            "num_move_commands": 0,
            "num_close_commands": 0,
            "is_valid": True,
        }

        try:
            # Validate command sequence
            PathCommandProcessor.validate_command_sequence(segment.commands, segment.points)

            # Analyze commands
            for cmd in segment.commands:
                if PathCommandProcessor.is_curve_command(cmd):
                    analysis["has_curves"] = True
                if PathCommandProcessor.is_drawing_command(cmd):
                    analysis["num_drawing_commands"] += 1
                if cmd == "M":
                    analysis["num_move_commands"] += 1
                if cmd == "Z":
                    analysis["num_close_commands"] += 1

        except (TypeError, ValueError) as e:
            analysis["is_valid"] = False
            analysis["error"] = str(e)

        return analysis

    @staticmethod
    def _process_cleaned_geometry(
        cleaned: shapely.geometry.base.BaseGeometry,
        is_ccw: bool,
        result: Optional[shapely.geometry.base.BaseGeometry],
        first_ccw_found: bool,
        deferred_cw_polygons: List[shapely.geometry.base.BaseGeometry],
    ) -> Tuple[Optional[shapely.geometry.base.BaseGeometry], bool]:
        """Process a cleaned geometry (Polygon, MultiPolygon, or GeometryCollection).

        Args:
            cleaned: The cleaned geometry from buffer(0)
            is_ccw: Whether the original polygon was CCW
            result: Current result geometry
            first_ccw_found: Whether first CCW polygon has been found
            deferred_cw_polygons: List of deferred CW polygons

        Returns:
            Tuple of (updated_result, updated_first_ccw_found)
        """
        # Handle different geometry types from buffer(0)
        if isinstance(cleaned, shapely.geometry.MultiPolygon):
            # Process each sub-polygon with the same orientation
            for sub_poly in cleaned.geoms:
                if not sub_poly.is_empty:
                    result, first_ccw_found = AvPathCleaner._process_single_geometry(
                        sub_poly, is_ccw, result, first_ccw_found, deferred_cw_polygons
                    )
        elif isinstance(cleaned, shapely.geometry.Polygon) and not cleaned.is_empty:
            result, first_ccw_found = AvPathCleaner._process_single_geometry(
                cleaned, is_ccw, result, first_ccw_found, deferred_cw_polygons
            )
        elif isinstance(cleaned, shapely.geometry.GeometryCollection):
            # Extract Polygon types from GeometryCollection
            for geom in cleaned.geoms:
                if isinstance(geom, shapely.geometry.Polygon) and not geom.is_empty:
                    result, first_ccw_found = AvPathCleaner._process_single_geometry(
                        geom, is_ccw, result, first_ccw_found, deferred_cw_polygons
                    )
                # Skip other geometry types (LineString, Point) with warning
                elif isinstance(geom, (shapely.geometry.LineString, shapely.geometry.Point)):
                    print(f"Warning: Skipping {geom.geom_type} geometry in GeometryCollection.")

        return result, first_ccw_found

    @staticmethod
    def _process_single_geometry(
        geometry: shapely.geometry.Polygon,
        is_ccw: bool,
        result: Optional[shapely.geometry.base.BaseGeometry],
        first_ccw_found: bool,
        deferred_cw_polygons: List[shapely.geometry.base.BaseGeometry],
    ) -> Tuple[Optional[shapely.geometry.base.BaseGeometry], bool]:
        """Process a single polygon geometry.

        Args:
            geometry: The polygon geometry to process
            is_ccw: Whether the original polygon was CCW
            result: Current result geometry
            first_ccw_found: Whether first CCW polygon has been found
            deferred_cw_polygons: List of deferred CW polygons

        Returns:
            Tuple of (updated_result, updated_first_ccw_found)
        """
        if result is None:
            # Wait for first CCW polygon to initialize result
            if is_ccw:
                result = geometry
                first_ccw_found = True
                # Now process any deferred CW polygons
                for cw_poly in deferred_cw_polygons:
                    result = result.difference(cw_poly)
                deferred_cw_polygons.clear()
            else:
                # Defer CW polygon until we find first CCW
                deferred_cw_polygons.append(geometry)
        elif first_ccw_found:
            # We have a base, now process all polygons
            if is_ccw:
                # CCW polygons are additive
                result = result.union(geometry)
            else:
                # CW polygons are subtractive (holes)
                result = result.difference(geometry)

        return result, first_ccw_found

    @staticmethod
    def _convert_shapely_to_paths(
        result: shapely.geometry.base.BaseGeometry,
        original_points_with_types: Optional[np.ndarray],
    ) -> List[AvPath]:
        """Convert Shapely geometry back to AvPath objects.

        The rotation algorithm prioritizes the first point of the input path,
        but the target point MUST be of type=0. If the first point is not found
        or is not type=0, it searches for the next matching type=0 point from
        the original path. The algorithm correctly handles all edge cases,
        including when the target point is the last point (before Z command).

        Args:
            result: The Shapely geometry to convert
            original_points_with_types: Original points array with types (x, y, type)
                for coordinate preservation and type=0 matching

        Returns:
            List of AvPath objects with proper rotation applied

        Note:
            The rotation works correctly for all positions including:
            - First point (index 0)
            - Middle points (indices 1 to n-2)
            - Last point (index n-1, before Z command)
            - No rotation when no type=0 points match
        """

        TOLERANCE: float = 1e-10  # pylint: disable=C0103
        cleaned_paths: List[AvPath] = []

        def find_rotation_target(
            coords: List[Tuple[float, float]],
            original_pts: Optional[np.ndarray],
        ) -> int:
            """Find the index to rotate to: first matching type=0 point from original.

            Algorithm:
            1. Iterate through original points in order (preferring first point)
            2. For each original point with type=0:
                - Check if it exists in coords within TOLERANCE
                - If found, return that index in coords
            3. If no match found, return 0 (no rotation)

            This algorithm correctly handles all edge cases including:
            - Target point being the last point (index n-1)
            - Multiple matching points (returns first occurrence)
            - Coordinate tolerance matching within TOLERANCE

            Args:
                coords: List of (x, y) coordinates from Shapely result
                original_pts: Original points array with types (x, y, type)

            Returns:
                Index in coords to rotate to (0 if no match found)
            """
            if original_pts is None or len(coords) == 0:
                return 0

            # Iterate through original points in order (prefer first point)
            for orig_pt in original_pts:
                orig_x, orig_y = orig_pt[0], orig_pt[1]
                orig_type = orig_pt[2] if len(orig_pt) > 2 else 0.0

                # Skip if not type=0
                if orig_type != 0.0:
                    continue

                # Check if this point exists in coords within TOLERANCE
                for i, (cx, cy) in enumerate(coords):
                    distance = np.sqrt((cx - orig_x) ** 2 + (cy - orig_y) ** 2)
                    if distance <= TOLERANCE:
                        return i

            # No matching type=0 point found, return 0 (no rotation)
            return 0

        # TODO: check why this is needed!!!
        def remove_duplicate_consecutive_points_from_coords(
            coords: List[Tuple[float, float]],
        ) -> List[Tuple[float, float]]:
            """Remove duplicate consecutive points from a coordinate list.

            Args:
                coords: List of (x, y) coordinates

            Returns:
                List of coordinates with duplicates removed
            """
            if len(coords) <= 1:
                return coords

            # Use the same tolerance as in glyph.validate()
            TOLERANCE = 1e-9  # pylint: disable=C0103

            # Build a new list without duplicate consecutive points
            clean_coords = [coords[0]]

            for i in range(1, len(coords)):
                # Check if current point is the same as the previous one (within tolerance)
                prev_x, prev_y = clean_coords[-1]
                curr_x, curr_y = coords[i]

                # Calculate distance between points
                distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5

                # Only add the point if it's not within tolerance of the previous one
                if distance > TOLERANCE:
                    clean_coords.append(coords[i])

            return clean_coords

        def rotate_coords(coords: List[Tuple[float, float]], idx: int) -> List[Tuple[float, float]]:
            """Rotate coordinate list to start from given index.

            This function correctly handles all edge cases including:
            - idx = 0 (no rotation needed)
            - idx = len(coords) - 1 (rotation to last point)
            - Empty coordinate lists

            Args:
                coords: List of (x, y) coordinates
                idx: Index to rotate to (0-based)

            Returns:
                Rotated coordinate list starting at idx
            """
            if idx == 0 or not coords:
                return coords
            return coords[idx:] + coords[:idx]

        def convert_polygon_to_paths(polygon: shapely.geometry.Polygon) -> None:
            """Convert a single polygon to paths (exterior + interiors)."""
            # Convert exterior
            exterior_coords = list(polygon.exterior.coords)
            if len(exterior_coords) >= 4:
                exterior_coords = exterior_coords[:-1]  # Remove closing point
                # TODO: check why this is needed!!!
                exterior_coords = remove_duplicate_consecutive_points_from_coords(exterior_coords)
                if len(exterior_coords) >= 3:  # Need at least 3 points for a polygon
                    # Enforce CCW for exterior rings (positive polygons)
                    if not AvPolygon.is_ccw(np.asarray(exterior_coords)):
                        exterior_coords = list(reversed(exterior_coords))
                    # Rotate to first matching type=0 point from original
                    rotation_idx = find_rotation_target(exterior_coords, original_points_with_types)
                    exterior_coords = rotate_coords(exterior_coords, rotation_idx)
                    exterior_cmds = ["M"] + ["L"] * (len(exterior_coords) - 1) + ["Z"]
                    cleaned_paths.append(AvPath(exterior_coords, exterior_cmds))

            # Convert interiors (holes)
            for interior in polygon.interiors:
                interior_coords = list(interior.coords)
                if len(interior_coords) >= 4:
                    interior_coords = interior_coords[:-1]  # Remove closing point
                    # TODO: check why this is needed!!!
                    interior_coords = remove_duplicate_consecutive_points_from_coords(interior_coords)
                    if len(interior_coords) >= 3:  # Need at least 3 points for a polygon
                        # Enforce CW for interior rings (holes)
                        if AvPolygon.is_ccw(np.asarray(interior_coords)):
                            interior_coords = list(reversed(interior_coords))
                        # Rotate to first matching type=0 point from original
                        rotation_idx = find_rotation_target(interior_coords, original_points_with_types)
                        interior_coords = rotate_coords(interior_coords, rotation_idx)
                        interior_cmds = ["M"] + ["L"] * (len(interior_coords) - 1) + ["Z"]
                        cleaned_paths.append(AvPath(interior_coords, interior_cmds))

        if isinstance(result, shapely.geometry.Polygon) and not result.is_empty:
            convert_polygon_to_paths(result)
        elif isinstance(result, shapely.geometry.MultiPolygon):
            # Handle MultiPolygon result
            for poly in result.geoms:
                if not poly.is_empty:
                    convert_polygon_to_paths(poly)

        return cleaned_paths

    @staticmethod
    def _add_polygon_to_result(
        result_polygons: List[shapely.geometry.Polygon],
        new_polygon: shapely.geometry.Polygon,
    ) -> List[shapely.geometry.Polygon]:
        """Add a CCW polygon to result: union with overlapping, keep separate if non-overlapping.

        Only unions polygons that have actual geometric overlap (shared interior area),
        not just touching boundaries. This preserves holes in polygons.

        Args:
            result_polygons: Current list of result polygons
            new_polygon: New polygon to add

        Returns:
            Updated list of result polygons
        """
        if new_polygon.is_empty:
            return result_polygons

        # Check if new polygon has actual geometric overlap with any existing result polygon
        merged = False
        updated_result = []

        for existing in result_polygons:
            if not merged:
                # Check for actual overlap (shared area), not just touching
                # Use intersection to check if there's shared interior area
                intersection = existing.intersection(new_polygon)
                has_overlap = not intersection.is_empty and intersection.area > 1e-10  # Meaningful shared area

                if has_overlap:
                    # Union overlapping polygons
                    union_result = existing.union(new_polygon)

                    # Handle MultiPolygon results from union
                    if isinstance(union_result, shapely.geometry.MultiPolygon):
                        updated_result.extend([p for p in union_result.geoms if not p.is_empty])
                    elif not union_result.is_empty:
                        updated_result.append(union_result)

                    merged = True
                else:
                    # No overlap - keep existing polygon as-is
                    updated_result.append(existing)
            else:
                # Keep existing polygon as-is
                updated_result.append(existing)

        if not merged:
            # New polygon doesn't overlap with any existing - keep as separate
            updated_result.append(new_polygon)

        return updated_result

    @staticmethod
    def _subtract_polygon_from_result(
        result_polygons: List[shapely.geometry.Polygon],
        hole_polygon: shapely.geometry.Polygon,
    ) -> List[shapely.geometry.Polygon]:
        """Subtract a CW polygon from its containing polygon(s) in the result.

        Args:
            result_polygons: Current list of result polygons
            hole_polygon: Hole polygon to subtract

        Returns:
            Updated list of result polygons with hole subtracted
        """
        if hole_polygon.is_empty:
            return result_polygons

        updated_result = []

        for poly in result_polygons:
            # Check if hole is fully contained within this polygon
            # Use robust containment check
            centroid_inside = poly.contains(hole_polygon.centroid)
            repr_inside = poly.contains(hole_polygon.representative_point())
            hole_smaller = hole_polygon.area < poly.area

            if centroid_inside and repr_inside and hole_smaller:
                # Subtract hole from this polygon
                diff_result = poly.difference(hole_polygon)

                # Handle MultiPolygon results from difference
                if isinstance(diff_result, shapely.geometry.MultiPolygon):
                    updated_result.extend([p for p in diff_result.geoms if not p.is_empty])
                elif isinstance(diff_result, shapely.geometry.Polygon) and not diff_result.is_empty:
                    updated_result.append(diff_result)
            else:
                # Hole not contained in this polygon - keep polygon as-is
                updated_result.append(poly)

        return updated_result

    @staticmethod
    def resolve_polygonized_path_intersections(path: AvMultiPolylinePath) -> AvMultiPolygonPath:
        """Resolve self-intersections in a polygonized path with winding direction rules.

        The input path consists of 0..n segments. Segments that are not explicitly closed
        will be automatically closed by appending a 'Z' command. Segments follow the
        standard winding rule where:
        - Counter-clockwise (CCW) segments represent positive/additive polygons
        - Clockwise (CW) segments represent subtractive polygons (holes)

        Algorithm Strategy:
        The function resolves complex path intersections by processing polygonized segments
        through Shapely geometric operations, carefully handling winding directions and
        deferring CW polygons until the first CCW polygon is found.

        Step-by-step process:
        1. Split input path into individual segments (sub-paths) using path.split_into_single_paths()
        2. Store all original points with types for rotation target selection
        3. Convert each segment to closed path and then to polygonized format:
            - Use _analyze_segment() for segment validation and analysis
            - Handle invalid segments with warnings and continue processing
            - Store CCW orientation from each closed path for later processing
        4. Apply buffer(0) operation to each polygon to remove self-intersections:
            - buffer(0) cleans up topology and resolves intersections
            - Handles Polygon, MultiPolygon, and GeometryCollection results
            - Skips invalid or empty geometries with warnings
        5. Perform sequential boolean operations with special ordering:
            - Wait for first CCW polygon to initialize result
            - Defer all CW polygons encountered before first CCW
            - Once first CCW is found, process deferred CW polygons as holes
            - Subsequent CCW polygons are unioned (additive)
            - Subsequent CW polygons are differenced (subtractive)
            - Use _process_cleaned_geometry() helper for geometry processing:
              * Calls _process_single_geometry() for individual polygon processing
              * Handles MultiPolygon by processing each sub-polygon
              * Handles GeometryCollection by extracting Polygon types
        6. Handle different geometry types from buffer(0):
            - Polygon: processed directly via _process_single_geometry()
            - MultiPolygon: each sub-polygon processed with same orientation
            - GeometryCollection: Polygon types extracted and processed
            - LineString/Point: skipped with warning
        7. Convert final Shapely geometry back to AvMultiPolygonPath format:
            - Use _convert_shapely_to_paths() helper for geometry conversion:
              * Calls convert_polygon_to_paths() for individual polygon conversion
              * Uses find_rotation_target() to find first matching type=0 point
              * Rotates each polygon to start at first matching type=0 point
              * Enforces CCW for exterior rings, CW for interior rings
            - Extract exterior rings as closed paths with 'Z' command
            - Extract interior rings (holes) as separate paths
            - Join all paths using AvPath.join_paths
            - Return result with MULTI_POLYGON_CONSTRAINTS

        Key technical details:
        - Uses orientation from closed path's is_ccw() to determine winding
        - Implements deferred processing for CW polygons before first CCW
        - Comprehensive error handling with empty path fallback
        - Removes duplicate closing points when converting coordinates
        - Rotation algorithm: finds first type=0 point from original that matches
            coordinates in result (within TOLERANCE=1e-10), preferring earlier points
        - Rotation correctly handles all positions including last point (index n-1)
        - Uses helper methods: _analyze_segment(), _process_cleaned_geometry(), _convert_shapely_to_paths()

        The function handles the following cases:
        - Empty input paths: returns empty AvMultiPolygonPath
        - Invalid segment structure: skips with warning using _analyze_segment() validation
        - Degenerate polygons (< 3 points): skips with warning
        - Invalid geometries after buffer(0): skips with warning
        - Different geometry types from buffer(0):
            - Polygon: processed directly
            - MultiPolygon: each sub-polygon processed with same orientation
            - GeometryCollection: Polygon types extracted and processed
            - LineString/Point: skipped with warning
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

        # Store the original points with types to preserve after Shapely operations
        original_points_with_types = None
        if path.points.shape[0] > 0:
            original_points_with_types = path.points.copy()

        # Process each segment to ensure it's closed
        polygons: List[AvSinglePolygonPath] = []
        orientations: List[bool] = []  # Store CCW orientation from closed paths

        for segment in segments:
            # Create closed path, then get polygonized path
            try:
                # Analyze segment using PathCommandProcessor
                analysis = AvPathCleaner._analyze_segment(segment)

                if not analysis["is_valid"]:
                    print(f"Warning: Invalid segment structure. Skipping. Error: {analysis.get('error', 'Unknown')}")
                    continue

                # Optional: Log segment analysis for debugging
                if analysis["has_curves"]:
                    # Curves will be polygonized by make_closed_single()
                    pass

                closed_path = AvPath.make_closed_single(segment)
                polygonized = closed_path.polygonized_path()
                polygons.append(polygonized)
                orientations.append(closed_path.is_ccw)  # Store orientation from closed path
            except (TypeError, ValueError) as e:
                print(f"Error processing segment: {e}. Skipping.")
                continue

        if not polygons:
            return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)

        # Group holes with parents BEFORE buffer(0) to preserve structure
        try:
            # Step 1: Convert to raw Shapely polygons (no buffer yet)
            raw_polygons: List[Tuple[shapely.geometry.Polygon, bool]] = []

            for polygon, is_ccw in zip(polygons, orientations):
                if polygon.points.shape[0] < 3:
                    print("Warning: Contour has fewer than 3 points. Skipping.")
                    continue
                shapely_poly = shapely.geometry.Polygon(polygon.points[:, :2].tolist())
                raw_polygons.append((shapely_poly, is_ccw))

            if not raw_polygons:
                print("Warning: No valid polygons to process. Returning empty path.")
                return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)

            # Step 2: Pre-assign holes to CCW polygons using proximity
            # Build mapping: hole_index -> parent_ccw_index
            ccw_indices = [idx for idx, (_, is_ccw) in enumerate(raw_polygons) if is_ccw]
            cw_indices = [idx for idx, (_, is_ccw) in enumerate(raw_polygons) if not is_ccw]

            hole_to_parent: Dict[int, int] = {}  # Maps CW index -> CCW index

            for cw_idx in cw_indices:
                cw_poly, _ = raw_polygons[cw_idx]
                cw_centroid = cw_poly.centroid

                # Find nearest CCW polygon by centroid distance
                best_ccw_idx = None
                min_dist = float("inf")

                for ccw_idx in ccw_indices:
                    ccw_poly, _ = raw_polygons[ccw_idx]
                    ccw_centroid = ccw_poly.centroid
                    distance = cw_centroid.distance(ccw_centroid)

                    if distance < min_dist:
                        min_dist = distance
                        best_ccw_idx = ccw_idx

                if best_ccw_idx is not None:
                    hole_to_parent[cw_idx] = best_ccw_idx

            # Step 3: Process polygons in exact input order
            # Track which holes belong to each CCW, and defer holes until parent is encountered
            ccw_to_holes: Dict[int, List[shapely.geometry.Polygon]] = {idx: [] for idx in ccw_indices}

            for cw_idx, parent_ccw_idx in hole_to_parent.items():
                cw_poly, _ = raw_polygons[cw_idx]
                ccw_to_holes[parent_ccw_idx].append(cw_poly)

            # Process in exact input order, creating complete polygons when encountering CCW
            shapely_polygons: List[shapely.geometry.base.BaseGeometry] = []

            for idx, (poly, is_ccw) in enumerate(raw_polygons):
                if is_ccw:
                    # This is a CCW polygon - create it with all its assigned holes
                    holes = ccw_to_holes.get(idx, [])

                    try:
                        # Create polygon with exterior and holes
                        if holes:
                            complete_poly = shapely.geometry.Polygon(
                                poly.exterior.coords, [hole.exterior.coords for hole in holes]
                            )
                        else:
                            complete_poly = poly

                        # Now apply buffer(0) to the complete polygon (with holes)
                        cleaned = complete_poly.buffer(0)
                        if not cleaned.is_empty and cleaned.is_valid:
                            shapely_polygons.append(cleaned)
                    except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
                        print(f"Warning: Failed to create/clean polygon with holes: {e}")
                        continue
                # else: CW polygon (hole) - skip it, already incorporated into parent

            if not shapely_polygons:
                print("Warning: No valid polygons after cleaning. Returning empty path.")
                return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)

            # Step 4: Process polygons sequentially - union overlapping, keep separate if non-overlapping
            # Holes are already incorporated into their parent polygons
            result_polygons: List[shapely.geometry.Polygon] = []

            for idx, poly_geom in enumerate(shapely_polygons):
                # Handle MultiPolygon from buffer(0)
                current_polys = (
                    [poly_geom] if isinstance(poly_geom, shapely.geometry.Polygon) else list(poly_geom.geoms)
                )

                for poly in current_polys:
                    if poly.is_empty:
                        continue

                    # Add polygon (union with overlapping, keep separate if non-overlapping)
                    result_polygons = AvPathCleaner._add_polygon_to_result(result_polygons, poly)

            # Step 5: Convert list of polygons to MultiPolygon or Polygon
            if not result_polygons:
                print("Warning: No valid polygons after processing. Returning empty path.")
                return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)
            elif len(result_polygons) == 1:
                result = result_polygons[0]
            else:
                result = shapely.geometry.MultiPolygon(result_polygons)

            if result.is_empty:
                print("Warning: Result is empty after operations. Returning empty path.")
                return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)

        except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
            print(f"Error during polygon processing: {e}. Returning empty path.")
            return AvMultiPolygonPath(constraints=MULTI_POLYGON_CONSTRAINTS)

        # Convert final Shapely geometry back to AvMultiPolygonPath
        try:
            cleaned_paths = AvPathCleaner._convert_shapely_to_paths(result, original_points_with_types)
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


###############################################################################
# AvPathCurveRebuilder
###############################################################################


class AvPathCurveRebuilder:
    """Rebuild polygon paths by replacing point clusters with SVG Bezier curves.

    This class provides functionality to convert polygonized paths back to
    smooth Bezier curves using least-squares fitting. Input paths must be
    AvMultiPolygonPath or AvSinglePolygonPath with properly closed segments.
    """

    # Constants
    TOLERANCE = 1e-9  # pylint: disable=C0103

    @staticmethod
    def _distance(p1: NDArray[np.float64], p2: NDArray[np.float64]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def _join_segments(segments: List[AvPath]) -> AvPath:
        """Join multiple segments into a single path."""
        if not segments:
            return AvPath()
        if len(segments) == 1:
            return segments[0]
        return AvPath.join_paths(*segments)

    @staticmethod
    def rebuild_curve_path(path: AvMultiPolygonPath) -> AvPath:
        """Rebuild a polygon path by replacing point clusters with Bezier curves.

        Iterates through the input path linearly, replacing clusters of sampled
        curve points (type=2 for quadratic, type=3 for cubic) with proper SVG
        Bezier curve commands (Q, C) using least-squares fitting.

        The algorithm proactively avoids degenerate Z lines by pre-analyzing
        each segment and rotating if a curve cluster would end at Z with the
        same endpoint as the segment start.

        Args:
            path: Input polygon path with point type annotations.
                    Must have segments starting with M and ending with Z.

        Returns:
            New AvPath with Bezier curves replacing point clusters.
            Non-curve vertices are preserved exactly.
        """
        # Split into segments and process each with potential pre-rotation
        segments = path.split_into_single_paths()
        if not segments:
            return AvPath()

        processed_segments: List[AvPath] = []
        for segment in segments:
            # Pre-rotate segment if it would create degenerate Z (when possible)
            rotated_segment = AvPathCurveRebuilder._rotate_if_degenerate_z(segment)
            # Process the (possibly rotated) segment
            processed = AvPathCurveRebuilder._rebuild_single_segment(rotated_segment)
            if processed.points.shape[0] > 0:
                processed_segments.append(processed)

        if not processed_segments:
            return AvPath()

        # Join segments
        result = AvPathCurveRebuilder._join_segments(processed_segments)

        # Post-process: fix degenerate curves (Q/C commands with coinciding control points)
        result = AvPathCurveRebuilder._fix_degenerate_curves(result)

        # Post-process: fix any remaining degenerate Z lines that couldn't be
        # handled by pre-rotation (e.g., segments with only one type=0 point)
        return AvPathCurveRebuilder._fix_degenerate_z_lines(result)

    @staticmethod
    def _rebuild_single_segment(segment: AvPath) -> AvPath:
        """Rebuild curves for a single segment."""
        points = segment.points
        commands = segment.commands

        if points.shape[0] == 0:
            return AvPath()

        out_points: List[NDArray[np.float64]] = []
        out_commands: List[str] = []

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

    @staticmethod
    def _rotate_if_degenerate_z(segment: AvPath) -> AvPath:
        """Pre-rotate segment if curve reconstruction would create degenerate Z.

        Analyzes the segment to detect if a curve cluster ends at Z, which would
        make the reconstructed curve's endpoint equal to segment_start_point.
        If so, rotates the segment to start at a different type=0 point.

        Args:
            segment: Input segment to analyze and potentially rotate.

        Returns:
            Rotated segment if degenerate Z would occur, otherwise original.
        """
        pts = segment.points
        cmds = segment.commands

        if len(pts) < 3 or len(cmds) < 3:
            return segment
        if cmds[-1] != "Z":
            return segment

        # Check if there's a curve cluster ending at Z
        # A curve cluster ends at Z if the last L commands before Z have type 2 or 3
        last_l_idx = -1
        for i in range(len(cmds) - 2, -1, -1):
            if cmds[i] == "L":
                last_l_idx = i
                break

        if last_l_idx < 0:
            return segment

        # Count points up to last_l_idx to find corresponding point index
        pt_idx = 0
        for i in range(last_l_idx + 1):
            if cmds[i] == "M":
                pt_idx += 1
            elif cmds[i] == "L":
                pt_idx += 1

        # Check if the last point before Z is a curve type (2 or 3)
        last_pt_idx = pt_idx - 1
        if last_pt_idx < 0 or last_pt_idx >= len(pts):
            return segment

        last_pt_type = pts[last_pt_idx, 2] if pts.shape[1] > 2 else 0.0

        # If last point is type 0 or -1 (not a curve sample), no degenerate Z issue
        if last_pt_type not in (2.0, 3.0):
            return segment

        # Curve cluster ends at Z - the curve endpoint will be segment_start_point
        # Find all type=0 points that could be rotation targets
        type0_indices = []
        for i, pt in enumerate(pts):
            pt_type = pt[2] if len(pt) > 2 else 0.0
            if pt_type == 0.0:
                type0_indices.append(i)

        if len(type0_indices) < 2:
            return segment  # Can't rotate with fewer than 2 type=0 points

        # Find a type=0 point where rotating to it avoids degenerate Z
        # After rotation to index i, first_pt becomes pts[i], last_pt becomes pts[i-1]
        for target_idx in type0_indices[1:]:  # Skip index 0 (current start)
            prev_idx = target_idx - 1
            if prev_idx < 0:
                prev_idx = len(pts) - 1

            target_pt = pts[target_idx, :2]
            prev_pt = pts[prev_idx, :2]
            dist = AvPathCurveRebuilder._distance(target_pt, prev_pt)

            if dist > AvPathCurveRebuilder.TOLERANCE:
                # Found non-degenerate rotation target - rotate segment
                return AvPathCurveRebuilder._rotate_segment_points(segment, target_idx)

        return segment  # No suitable rotation found

    @staticmethod
    def _rotate_segment_points(segment: AvPath, target_idx: int) -> AvPath:
        """Rotate segment points to start at target_idx.

        Args:
            segment: Segment to rotate.
            target_idx: Index of point to become new start.

        Returns:
            Rotated segment.
        """
        pts = segment.points
        cmds = segment.commands

        if target_idx <= 0 or target_idx >= len(pts):
            return segment

        # Rotate points
        rotated_pts = np.concatenate([pts[target_idx:], pts[:target_idx]])

        # Commands stay the same (M L L ... L Z)
        return AvPath(rotated_pts, cmds)

    @staticmethod
    def _fix_degenerate_curves(path: AvPath) -> AvPath:
        """Fix degenerate curve commands by simplifying or removing them.

        A curve is degenerate if control points coincide with start/end points
        or if all points are collinear. This method detects and fixes:

        Quadratic curves (Q):
        - Control point coincides with start: Replace Q with L to end
        - Control point coincides with end: Replace Q with L to end
        - Start coincides with end: Remove Q entirely
        - All three points collinear: Replace Q with L to end

        Cubic curves (C):
        - ctrl1 coincides with start AND ctrl2 coincides with end: Replace C with L
        - ctrl1 coincides with start: Simplify to Q(ctrl2, end)
        - ctrl2 coincides with end: Simplify to Q(ctrl1, end)
        - ctrl1 coincides with ctrl2: Simplify to Q(ctrl1, end)
        - Start coincides with end: Remove C entirely
        - All four points collinear: Replace C with L to end

        Args:
            path: Path that may contain degenerate curves.

        Returns:
            Path with degenerate curves fixed.
        """
        points = path.points
        commands = path.commands

        if len(points) == 0:
            return path

        new_points = []
        new_commands = []
        point_idx = 0

        for cmd in commands:
            if cmd == "M":
                new_points.append(points[point_idx].copy())
                new_commands.append("M")
                point_idx += 1

            elif cmd == "L":
                new_points.append(points[point_idx].copy())
                new_commands.append("L")
                point_idx += 1

            elif cmd == "Q":
                # Quadratic: start (last emitted), control, end
                if not new_points:
                    # No start point - skip this Q
                    point_idx += 2
                    continue

                start_pt = new_points[-1][:2]
                ctrl_pt = points[point_idx]
                end_pt = points[point_idx + 1]

                # Calculate distances
                d_start_ctrl = np.linalg.norm(ctrl_pt[:2] - start_pt)
                d_ctrl_end = np.linalg.norm(end_pt[:2] - ctrl_pt[:2])
                d_start_end = np.linalg.norm(end_pt[:2] - start_pt)

                # Check for degeneracy
                if d_start_end < AvPathCurveRebuilder.TOLERANCE:
                    # Start and end coincide - remove this Q entirely
                    point_idx += 2
                    continue

                elif d_start_ctrl < AvPathCurveRebuilder.TOLERANCE or d_ctrl_end < AvPathCurveRebuilder.TOLERANCE:
                    # Control point coincides with start or end - replace with L
                    new_points.append(end_pt.copy())
                    new_commands.append("L")
                    point_idx += 2

                else:
                    # Check collinearity: if control point lies on line from start to end
                    # Use cross product to check if points are collinear
                    v1 = ctrl_pt[:2] - start_pt
                    v2 = end_pt[:2] - start_pt
                    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])

                    # Normalize by the area of bounding box
                    max_coord = max(abs(v1).max(), abs(v2).max())
                    if max_coord > 0:
                        normalized_cross = cross / (max_coord**2)
                    else:
                        normalized_cross = 0

                    if normalized_cross < AvPathCurveRebuilder.TOLERANCE:
                        # Collinear - replace with L
                        new_points.append(end_pt.copy())
                        new_commands.append("L")
                        point_idx += 2
                    else:
                        # Valid Q curve - keep it
                        new_points.append(ctrl_pt.copy())
                        new_points.append(end_pt.copy())
                        new_commands.append("Q")
                        point_idx += 2

            elif cmd == "C":
                # Cubic: start (last emitted), ctrl1, ctrl2, end
                if not new_points:
                    # No start point - skip this C
                    point_idx += 3
                    continue

                start_pt = new_points[-1][:2]
                ctrl1_pt = points[point_idx]
                ctrl2_pt = points[point_idx + 1]
                end_pt = points[point_idx + 2]

                # Calculate distances
                d_start_ctrl1 = np.linalg.norm(ctrl1_pt[:2] - start_pt)
                d_ctrl1_ctrl2 = np.linalg.norm(ctrl2_pt[:2] - ctrl1_pt[:2])
                d_ctrl2_end = np.linalg.norm(end_pt[:2] - ctrl2_pt[:2])
                d_start_end = np.linalg.norm(end_pt[:2] - start_pt)
                # d_start_ctrl2 = np.linalg.norm(ctrl2_pt[:2] - start_pt)
                # d_ctrl1_end = np.linalg.norm(end_pt[:2] - ctrl1_pt[:2])

                # Check for degeneracy
                if d_start_end < AvPathCurveRebuilder.TOLERANCE:
                    # Start and end coincide - remove this C entirely
                    point_idx += 3
                    continue

                # Case 1: Both control points coincide with their respective endpoints
                if d_start_ctrl1 < AvPathCurveRebuilder.TOLERANCE and d_ctrl2_end < AvPathCurveRebuilder.TOLERANCE:
                    # C collapses to a line
                    new_points.append(end_pt.copy())
                    new_commands.append("L")
                    point_idx += 3

                # Case 2: ctrl1 coincides with start - simplify to Q(ctrl2, end)
                elif d_start_ctrl1 < AvPathCurveRebuilder.TOLERANCE:
                    if d_ctrl2_end < AvPathCurveRebuilder.TOLERANCE:
                        # ctrl2 also coincides with end - use L
                        new_points.append(end_pt.copy())
                        new_commands.append("L")
                    else:
                        # Valid Q curve with ctrl2 as control point
                        new_points.append(ctrl2_pt.copy())
                        new_points.append(end_pt.copy())
                        new_commands.append("Q")
                    point_idx += 3

                # Case 3: ctrl2 coincides with end - simplify to Q(ctrl1, end)
                elif d_ctrl2_end < AvPathCurveRebuilder.TOLERANCE:
                    if d_start_ctrl1 < AvPathCurveRebuilder.TOLERANCE:
                        # ctrl1 also coincides with start - use L
                        new_points.append(end_pt.copy())
                        new_commands.append("L")
                    else:
                        # Valid Q curve with ctrl1 as control point
                        new_points.append(ctrl1_pt.copy())
                        new_points.append(end_pt.copy())
                        new_commands.append("Q")
                    point_idx += 3

                # Case 4: ctrl1 and ctrl2 coincide - simplify to Q
                elif d_ctrl1_ctrl2 < AvPathCurveRebuilder.TOLERANCE:
                    # Use ctrl1 (or ctrl2, they're the same) as control point
                    new_points.append(ctrl1_pt.copy())
                    new_points.append(end_pt.copy())
                    new_commands.append("Q")
                    point_idx += 3

                else:
                    # Check collinearity: all four points on a line
                    # Use cross products to check
                    v1 = ctrl1_pt[:2] - start_pt
                    v2 = ctrl2_pt[:2] - start_pt
                    v3 = end_pt[:2] - start_pt

                    cross1 = abs(v1[0] * v2[1] - v1[1] * v2[0])
                    cross2 = abs(v1[0] * v3[1] - v1[1] * v3[0])
                    cross3 = abs(v2[0] * v3[1] - v2[1] * v3[0])

                    # Normalize
                    max_coord = max(abs(v1).max(), abs(v2).max(), abs(v3).max())
                    if max_coord > 0:
                        norm_cross1 = cross1 / (max_coord**2)
                        norm_cross2 = cross2 / (max_coord**2)
                        norm_cross3 = cross3 / (max_coord**2)
                    else:
                        norm_cross1 = norm_cross2 = norm_cross3 = 0

                    if (
                        norm_cross1 < AvPathCurveRebuilder.TOLERANCE
                        and norm_cross2 < AvPathCurveRebuilder.TOLERANCE
                        and norm_cross3 < AvPathCurveRebuilder.TOLERANCE
                    ):
                        # All collinear - replace with L
                        new_points.append(end_pt.copy())
                        new_commands.append("L")
                        point_idx += 3
                    else:
                        # Valid C curve - keep it
                        new_points.append(ctrl1_pt.copy())
                        new_points.append(ctrl2_pt.copy())
                        new_points.append(end_pt.copy())
                        new_commands.append("C")
                        point_idx += 3

            elif cmd == "Z":
                new_commands.append("Z")

            else:
                # Unknown command - skip
                pass

        # Build output path
        if new_points:
            points_array = np.array(new_points, dtype=np.float64)
            return AvPath(points_array, new_commands, path.constraints)
        return AvPath()

    @staticmethod
    def _fix_degenerate_z_lines(path: AvPath) -> AvPath:
        """Fix degenerate Z lines by rotating reconstructed curve segments.

        This is a fallback for cases where pre-rotation wasn't possible
        (e.g., segments with only one type=0 point). It operates on the
        reconstructed path which has Q/C curve commands.

        Args:
            path: Path that may contain degenerate Z lines.

        Returns:
            Path with segments rotated to avoid degenerate Z lines.
        """
        segments = path.split_into_single_paths()
        if not segments:
            return path

        fixed_segments: List[AvPath] = []

        for seg in segments:
            pts = seg.points
            cmds = seg.commands

            # Check if closed and degenerate
            if len(cmds) < 2 or cmds[-1] != "Z" or len(pts) < 2:
                fixed_segments.append(seg)
                continue

            first_pt = pts[0, :2]
            last_pt = pts[-1, :2]
            dist = AvPathCurveRebuilder._distance(first_pt, last_pt)

            if dist > AvPathCurveRebuilder.TOLERANCE:
                # Not degenerate
                fixed_segments.append(seg)
                continue

            # Degenerate - check if we can rotate (has L commands)
            has_l_commands = any(cmd == "L" for cmd in cmds)
            if has_l_commands:
                # Try to rotate to L command endpoint
                rotated = AvPathCurveRebuilder._rotate_reconstructed_segment(seg)
                fixed_segments.append(rotated)
            else:
                # No L commands - can't rotate, accept degenerate Z
                fixed_segments.append(seg)

        return AvPathCurveRebuilder._join_segments(fixed_segments)

    @staticmethod
    def _rotate_reconstructed_segment(seg: AvPath) -> AvPath:
        """Rotate a reconstructed segment (with Q/C curves) to fix degenerate Z.

        Finds an L command endpoint to rotate to, which is simpler than
        rotating to curve endpoints.

        Args:
            seg: Reconstructed segment with degenerate Z line.

        Returns:
            Rotated segment, or original if no suitable rotation found.
        """
        pts = seg.points
        cmds = seg.commands

        if len(pts) < 3:
            return seg

        # Build command groups: (cmd, point_indices, endpoint_index)
        groups: List[Tuple[str, List[int], int]] = []
        pt_idx = 0

        for cmd in cmds:
            if cmd == "M":
                groups.append(("M", [pt_idx], pt_idx))
                pt_idx += 1
            elif cmd == "L":
                groups.append(("L", [pt_idx], pt_idx))
                pt_idx += 1
            elif cmd == "Q":
                groups.append(("Q", [pt_idx, pt_idx + 1], pt_idx + 1))
                pt_idx += 2
            elif cmd == "C":
                groups.append(("C", [pt_idx, pt_idx + 1, pt_idx + 2], pt_idx + 2))
                pt_idx += 3
            elif cmd == "Z":
                groups.append(("Z", [], -1))

        # Get middle groups (skip M and Z)
        middle = [(i, g) for i, g in enumerate(groups) if g[0] not in ("M", "Z")]

        if len(middle) < 2:
            return seg

        # Find L commands as rotation candidates (simpler than curves)
        l_indices = [(list_idx, grp_idx) for list_idx, (grp_idx, g) in enumerate(middle) if g[0] == "L"]

        for list_idx, _ in l_indices:
            # Check if rotating here creates non-degenerate Z
            curr_endpoint = middle[list_idx][1][2]
            prev_list_idx = list_idx - 1 if list_idx > 0 else len(middle) - 1
            prev_endpoint = middle[prev_list_idx][1][2]

            if curr_endpoint < 0 or prev_endpoint < 0:
                continue
            if curr_endpoint >= len(pts) or prev_endpoint >= len(pts):
                continue

            curr_pt = pts[curr_endpoint, :2]
            prev_pt = pts[prev_endpoint, :2]
            d = AvPathCurveRebuilder._distance(curr_pt, prev_pt)

            if d > AvPathCurveRebuilder.TOLERANCE:
                # Found non-degenerate rotation - rebuild path
                rotated_middle = middle[list_idx:] + middle[:list_idx]

                new_pts: List[np.ndarray] = []
                new_cmds: List[str] = ["M"]

                # M point is the endpoint of first rotated command
                m_endpoint = rotated_middle[0][1][2]
                new_pts.append(pts[m_endpoint].copy())

                # Add remaining commands (skip first L which became M)
                for i, (_, grp) in enumerate(rotated_middle):
                    cmd, pt_indices, _ = grp
                    if i == 0 and cmd == "L":
                        continue  # First L became M
                    new_cmds.append(cmd)
                    for pi in pt_indices:
                        new_pts.append(pts[pi].copy())

                new_cmds.append("Z")

                if new_pts:
                    try:
                        return AvPath(np.array(new_pts), new_cmds)
                    except ValueError:
                        continue
        return seg


class AvPathCreator:
    """Utility class for creating common geometric paths.

    Provides static methods to create AvPath objects for basic shapes
    like circles and rectangles with proper SVG path commands.
    """

    @staticmethod
    def circle(cx: float, cy: float, diameter: float) -> AvPath:
        """Create a circular path using 4 cubic Bezier curves.

        The circle is approximated using 4 cubic Bezier curve segments.
        Each segment uses two control points to approximate a 90-degree arc.
        The circle starts and ends on the perimeter.

        Args:
            cx (float): X coordinate of the circle center
            cy (float): Y coordinate of the circle center
            diameter (float): Diameter of the circle

        Returns:
            AvPath: A closed circular path

        Note:
            - The circle is centered at (cx, cy)
            - Uses 8 control points for 4 curve segments
            - Path starts and ends on perimeter at (cx + r, cy)
            - Uses magic constant k = 0.552284749831 for optimal approximation
            - Total commands: M + 4*C + Z = 6
            - Total points: 13 (start + 4*3 points for curves, Z has no point)
        """
        r = diameter / 2.0
        k = 0.552284749831  # Magic constant for 4-curve circle approximation

        # Generate points for 4 cubic Bezier curves
        # Each curve has: start (current position), control point 1, control point 2, end point
        pts = np.array(
            [
                [cx + r, cy],  # Start on perimeter (rightmost point)
                # First quadrant (bottom-right)
                [cx + r, cy - k * r],  # Control point 1
                [cx + k * r, cy - r],  # Control point 2
                [cx, cy - r],  # End of curve 1
                # Second quadrant (bottom-left)
                [cx - k * r, cy - r],  # Control point 1
                [cx - r, cy - k * r],  # Control point 2
                [cx - r, cy],  # End of curve 2
                # Third quadrant (top-left)
                [cx - r, cy + k * r],  # Control point 1
                [cx - k * r, cy + r],  # Control point 2
                [cx, cy + r],  # End of curve 3
                # Fourth quadrant (top-right)
                [cx + k * r, cy + r],  # Control point 1
                [cx + r, cy + k * r],  # Control point 2
                [cx + r, cy],  # End of curve 4 at start point (Z will close cleanly)
            ]
        )

        # Build command list: M + 4*C + Z
        cmds = ["M"]
        for _ in range(4):
            cmds.append("C")  # One cubic curve per quadrant
        cmds.append("Z")  # Close path

        return AvPath(pts, cmds)

    @staticmethod
    def rectangle(x1: float, y1: float, x2: float, y2: float) -> AvPath:
        """Create a rectangular path from corner coordinates.

        Creates a rectangle using the specified corner coordinates.
        The rectangle is drawn starting from (x1, y1), proceeding
        counter-clockwise, and returning to the start point.

        Args:
            x1 (float): X coordinate of first corner (typically bottom-left)
            y1 (float): Y coordinate of first corner (typically bottom-left)
            x2 (float): X coordinate of opposite corner (typically top-right)
            y2 (float): Y coordinate of opposite corner (typically top-right)

        Returns:
            AvPath: A closed rectangular path

        Note:
            - Rectangle corners are (x1,y1), (x2,y1), (x2,y2), (x1,y2)
            - Starts at (x1, y1) and proceeds counter-clockwise
            - Uses linear commands (L) for straight edges
            - 'Z' command automatically closes the path back to start
            - Total commands: M + 3*L + Z = 5
            - Total points: 4 (Z command has no point)
        """
        # Define rectangle corners in counter-clockwise order
        # Start at (x1, y1). The 'Z' command will close the path automatically.
        pts = np.array(
            [
                [x1, y1],  # First corner (start)
                [x2, y1],  # Second corner
                [x2, y2],  # Third corner
                [x1, y2],  # Fourth corner
            ]
        )

        # Build command list: Move + 3 Lines + Close
        cmds = ["M"]  # Move to start point
        for _ in range(3):
            cmds.append("L")  # Line to next corner
        cmds.append("Z")  # Close path

        return AvPath(pts, cmds)

    @staticmethod
    def rect_by_avbox(avbox: AvBox) -> AvPath:
        """Create a rectangular path from an AvBox.

        Convenience function that extracts coordinates from an AvBox
        and creates a rectangular path using those coordinates.

        Args:
            avbox: An AvBox object with xmin, ymin, xmax, ymax properties

        Returns:
            AvPath: A closed rectangular path matching the AvBox bounds

        Note:
            - Delegates to rectangle() for the actual path creation
            - Uses AvBox extent: (xmin, ymin, xmax, ymax)
        """
        return AvPathCreator.rectangle(avbox.xmin, avbox.ymin, avbox.xmax, avbox.ymax)
