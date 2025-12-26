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
from ave.path_support import PathCommandProcessor


###############################################################################
# AvPathCleaner
###############################################################################
class AvPathCleaner:
    """Collection of static path-cleaning utilities."""

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
        TOLERANCE: float = 1e-10
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
                result, first_ccw_found = AvPathCleaner._process_cleaned_geometry(
                    cleaned, is_ccw, result, first_ccw_found, deferred_cw_polygons
                )

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
