"""Path cleaning and manipulation utilities for vector graphics processing."""

from typing import List

import shapely.geometry

from ave.path import AvClosedPath, AvPath, AvPolygonPath, AvSinglePath


class AvPathCleaner:
    """Collection of static path-cleaning utilities."""

    @staticmethod
    def resolve_path_intersections(path: AvPath) -> AvPath:
        """Resolve self-intersections in an AvPath using sequential Shapely boolean operations.

        Algorithm Strategy:
        The function resolves complex path intersections by converting vector paths to Shapely
        geometric objects, performing topological operations, then converting back to paths.

        Step-by-step process:
        1. Split input path into individual contours (sub-paths)
        2. Ensure all contours are properly closed by adding 'Z' command if missing
        3. Convert each closed contour to a polygonized path format
        4. Apply buffer(0) operation to each polygon to remove self-intersections
            - buffer(0) is a Shapely technique that cleans up topology
            - Handles Polygon, MultiPolygon, and GeometryCollection results
        5. Perform sequential boolean operations based on contour orientation:
            - CCW contours (exterior): union operation (additive)
            - CW contours (interior/hole): difference operation (subtractive)
            - This reconstructs the proper fill rule for complex glyphs
        6. Convert final Shapely geometry back to AvPath format:
            - Extract exterior rings as closed paths
            - Extract interior rings (holes) as separate paths
            - Join all paths into final result

        Key technical details:
        - Uses Shapely's is_ccw property to determine contour orientation
        - Handles edge cases: empty contours, degenerate polygons, processing errors
        - Falls back to original path if boolean operations fail
        - Removes duplicate closing points when converting coordinates back to paths

        Args:
            path: Input AvPath that may contain self-intersections

        Returns:
            AvPath: Cleaned path with intersections resolved, or empty path if no valid result
        """
        # Step 1: Split into individual contours
        contours: List[AvSinglePath] = path.split_into_single_paths()

        # Step 2: Ensure all contours are properly closed by adding 'Z' command if missing
        closed_contours: List[AvClosedPath] = []
        for contour in contours:
            # Skip empty contours
            if not contour.commands:
                continue

            # Check if contour is closed, considering both explicit 'Z' and implicit closure
            if contour.commands[-1] == "Z":
                # Already closed, use as is
                closed_path = AvClosedPath(contour.points.copy(), list(contour.commands))
            else:
                # Add 'Z' to close the contour
                new_commands = list(contour.commands) + ["Z"]
                closed_path = AvClosedPath(contour.points.copy(), new_commands)

            closed_contours.append(closed_path)

        if not closed_contours:
            return AvPath()

        # Apply buffer(0) operation to each polygon to remove self-intersections and store cleaned polygons
        cleaned_polygons: List[shapely.geometry.Polygon] = []

        for closed_path in closed_contours:
            # Step 3: Convert each closed contour to a polygonized path format
            polygonized: AvPolygonPath = closed_path.polygonized_path()

            # Skip degenerate polygons
            if polygonized.points.shape[0] < 3:
                print("Warning: Contour has fewer than 3 points. Skipping.")
                continue

            try:
                # Step 4: Create Shapely polygon and remove self-intersections
                shapely_poly = shapely.geometry.Polygon(polygonized.points[:, :2].tolist())
                cleaned_poly = shapely_poly.buffer(0)

                # Handle different geometry types returned by buffer(0)
                if isinstance(cleaned_poly, shapely.geometry.Polygon) and not cleaned_poly.is_empty:
                    cleaned_polygons.append(cleaned_poly)

                elif isinstance(cleaned_poly, shapely.geometry.MultiPolygon):
                    # Add all sub-polygons
                    for poly in cleaned_poly.geoms:
                        if not poly.is_empty:
                            cleaned_polygons.append(poly)

                elif isinstance(cleaned_poly, shapely.geometry.GeometryCollection):
                    # Extract Polygon types
                    for geom in cleaned_poly.geoms:
                        if isinstance(geom, shapely.geometry.Polygon) and not geom.is_empty:
                            cleaned_polygons.append(geom)

            except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
                print(f"Warning: Failed to process contour {e}. Skipping.")
                continue

        if not cleaned_polygons:
            return AvPath()

        # Step 5: Perform sequential boolean operations based on contour orientation
        try:
            # Start with the first polygon
            result = cleaned_polygons[0]

            # Process remaining polygons sequentially
            for poly in cleaned_polygons[1:]:
                # Check orientation using Shapely's built-in orientation
                # Shapely uses CCW for exterior rings by convention
                if poly.exterior.is_ccw:
                    # Union additive polygons
                    result = result.union(poly)
                else:
                    # Subtract hole polygons
                    result = result.difference(poly)

        except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
            print(f"Error during sequential boolean operations: {e}")
            return path  # Return original path on error

        # Step 6: Convert final Shapely geometry back to AvPath format
        cleaned_paths: List[AvPath] = []

        if isinstance(result, shapely.geometry.Polygon) and not result.is_empty:
            # Convert exterior
            exterior_coords = list(result.exterior.coords)
            if len(exterior_coords) >= 4:
                exterior_coords = exterior_coords[:-1]  # Remove closing point
                exterior_cmds = ["M"] + ["L"] * (len(exterior_coords) - 1) + ["Z"]
                cleaned_paths.append(AvPath(exterior_coords, exterior_cmds))

            # Convert interiors (holes)
            for interior in result.interiors:
                interior_coords = list(interior.coords)
                if len(interior_coords) >= 4:
                    interior_coords = interior_coords[:-1]  # Remove closing point
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
                        exterior_cmds = ["M"] + ["L"] * (len(exterior_coords) - 1) + ["Z"]
                        cleaned_paths.append(AvPath(exterior_coords, exterior_cmds))

                    # Convert interiors
                    for interior in poly.interiors:
                        interior_coords = list(interior.coords)
                        if len(interior_coords) >= 4:
                            interior_coords = interior_coords[:-1]
                            interior_cmds = ["M"] + ["L"] * (len(interior_coords) - 1) + ["Z"]
                            cleaned_paths.append(AvPath(interior_coords, interior_cmds))

        # Join all paths
        if cleaned_paths:
            return AvPath.join_paths(*cleaned_paths)
        else:
            return AvPath()
