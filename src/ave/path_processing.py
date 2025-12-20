"""Path cleaning and manipulation utilities for vector graphics processing."""

from __future__ import annotations

from typing import List, Optional

import shapely.geometry

from ave.path import (
    MULTI_POLYLINE_CONSTRAINTS,
    AvMultiPolylinePath,
    AvPath,
    AvSinglePolygonPath,
)


###############################################################################
# AvPathCleaner
###############################################################################
class AvPathCleaner:
    """Collection of static path-cleaning utilities."""

    @staticmethod
    def resolve_polygonized_path_intersections(path: AvMultiPolylinePath) -> AvMultiPolylinePath:
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
        7. Convert final Shapely geometry back to AvMultiPolylinePath format:
            - Extract exterior rings as closed paths with 'Z' command
            - Extract interior rings (holes) as separate paths
            - Join all paths using AvPath.join_paths
            - Return result with MULTI_POLYLINE_CONSTRAINTS

        Key technical details:
        - Uses orientation from closed path's is_ccw() to determine winding
        - Implements deferred processing for CW polygons before first CCW
        - Comprehensive error handling with fallback to original path
        - Removes duplicate closing points when converting coordinates

        The function handles the following cases:
        - Empty input paths: returns empty AvMultiPolylinePath
        - Degenerate polygons (< 3 points): skips with warning
        - Invalid geometries after buffer(0): skips with warning
        - Different geometry types from buffer(0):
            - Polygon: processed directly
            - MultiPolygon: each sub-polygon processed with same orientation
            - GeometryCollection: Polygon types extracted and processed
        - CW polygons before first CCW: deferred until first CCW is found
        - No CCW polygon found: returns empty path with warning
        - Empty result after operations: returns original path with warning
        - Shapely errors during processing: returns original path with warning
        - Errors during geometry conversion: returns original path with warning
        - Errors during path joining: returns original path with warning

        Args:
            path: An AvMultiPolylinePath containing the segments to process

        Returns:
            AvMultiPolylinePath: A new path with resolved intersections and proper winding,
                                or the original path if errors occur
        """

        # Split path into individual segments
        segments = path.split_into_single_paths()

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
            return AvMultiPolylinePath(constraints=MULTI_POLYLINE_CONSTRAINTS)

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
                return AvMultiPolylinePath(constraints=MULTI_POLYLINE_CONSTRAINTS)

            if result.is_empty:
                print("Warning: Result is empty after operations. Returning original path.")
                return path

        except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
            print(f"Error during polygon processing: {e}. Returning original path.")
            return path  # Return original path on error

        # Convert final Shapely geometry back to AvMultiPolylinePath
        cleaned_paths: List[AvSinglePolygonPath] = []

        try:
            if isinstance(result, shapely.geometry.Polygon) and not result.is_empty:
                # Convert exterior
                exterior_coords = list(result.exterior.coords)
                if len(exterior_coords) >= 4:
                    exterior_coords = exterior_coords[:-1]  # Remove closing point
                    if len(exterior_coords) >= 3:  # Need at least 3 points for a polygon
                        exterior_cmds = ["M"] + ["L"] * (len(exterior_coords) - 1) + ["Z"]
                        cleaned_paths.append(AvPath(exterior_coords, exterior_cmds))

                # Convert interiors (holes)
                for interior in result.interiors:
                    interior_coords = list(interior.coords)
                    if len(interior_coords) >= 4:
                        interior_coords = interior_coords[:-1]  # Remove closing point
                        if len(interior_coords) >= 3:  # Need at least 3 points for a polygon
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
                                exterior_cmds = ["M"] + ["L"] * (len(exterior_coords) - 1) + ["Z"]
                                cleaned_paths.append(AvPath(exterior_coords, exterior_cmds))

                        # Convert interiors
                        for interior in poly.interiors:
                            interior_coords = list(interior.coords)
                            if len(interior_coords) >= 4:
                                interior_coords = interior_coords[:-1]
                                if len(interior_coords) >= 3:  # Need at least 3 points for a polygon
                                    interior_cmds = ["M"] + ["L"] * (len(interior_coords) - 1) + ["Z"]
                                    cleaned_paths.append(AvPath(interior_coords, interior_cmds))
        except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
            print(f"Error during geometry conversion: {e}. Returning original path.")
            return path

        # Join all paths and return as AvMultiPolylinePath with MULTI_POLYLINE_CONSTRAINTS
        if cleaned_paths:
            try:
                joined = AvPath.join_paths(*cleaned_paths)
                # Return with MULTI_POLYLINE_CONSTRAINTS
                return AvMultiPolylinePath(joined.points, joined.commands, MULTI_POLYLINE_CONSTRAINTS)
            except (TypeError, ValueError) as e:
                print(f"Error during path joining: {e}. Returning original path.")
                return path
        else:
            print("Warning: No valid paths to join. Returning original path.")
            return path
