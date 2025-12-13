"""Path cleaning and manipulation utilities for vector graphics processing."""

from typing import List, Tuple

import numpy as np
import shapely.geometry
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ave.path import AvClosedPath, AvPath, AvPolygonPath, AvSinglePath


###############################################################################
# AvPointMatcher
###############################################################################
class AvPointMatcher:
    """Utilities for matching points between two 2D point sets and transferring attributes.

    This class provides efficient nearest-neighbor matching between point arrays,
    primarily used to transfer point type information (on-curve vs control points)
    from an original path to a transformed or resampled path.

    Point types in AvPath:
        0.0 = on-curve point (M, L, or endpoint of Q/C)
        2.0 = quadratic Bezier control point (Q)
        3.0 = cubic Bezier control point (C)
    """

    @staticmethod
    def transfer_point_types(
        points_org: NDArray[np.float64],
        points_new: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Transfer point types from original points to new points via nearest neighbor matching.

        For each point in points_new, finds the nearest point in points_org and copies
        its type value. Uses a KD-tree for O(M log N) performance.

        Args:
            points_org: Original points array, shape (N, 3) with columns (x, y, type).
            points_new: New points array, shape (M, 2) or (M, 3) with columns (x, y, [type]).
                        If 3 columns, the type column will be overwritten.

        Returns:
            Array of shape (M, 3) with (x, y, type) where type is transferred from
            the nearest point in points_org.

        Example:
            >>> org = np.array([[0, 0, 0.0], [1, 1, 2.0], [2, 2, 3.0]])
            >>> new = np.array([[0.1, 0.1], [1.9, 1.9]])
            >>> result = AvPointMatcher.transfer_point_types(org, new)
            >>> result[:, 2]  # types: [0.0, 3.0]
        """
        # Handle empty arrays
        if points_org.shape[0] == 0:
            if points_new.shape[0] == 0:
                return np.empty((0, 3), dtype=np.float64)
            # No source points - default to on-curve type (0.0)
            result = np.zeros((points_new.shape[0], 3), dtype=np.float64)
            result[:, :2] = points_new[:, :2]
            return result

        if points_new.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)

        # Extract 2D coordinates for matching
        org_xy = points_org[:, :2]
        new_xy = points_new[:, :2]

        # Build KD-tree from original points
        tree = cKDTree(org_xy)

        # Query nearest neighbors for all new points
        _, indices = tree.query(new_xy, k=1)

        # Build result array
        result = np.zeros((points_new.shape[0], 3), dtype=np.float64)
        result[:, :2] = new_xy
        result[:, 2] = points_org[indices, 2]

        return result

    @staticmethod
    def transfer_point_types_ordered(
        points_org: NDArray[np.float64],
        points_new: NDArray[np.float64],
        search_window: int = 10,
    ) -> NDArray[np.float64]:
        """Transfer point types assuming roughly ordered/aligned point sequences.

        Optimized O(M * W) algorithm for cases where points_new and points_org
        are in similar order (possibly reversed). Uses a sliding window search
        instead of a full KD-tree.

        Args:
            points_org: Original points array, shape (N, 3) with columns (x, y, type).
            points_new: New points array, shape (M, 2) or (M, 3).
            search_window: Number of points to search around the expected position.
                            Larger values handle more deviation but are slower.

        Returns:
            Array of shape (M, 3) with transferred types.
        """
        n_org = points_org.shape[0]
        n_new = points_new.shape[0]

        # Handle empty arrays
        if n_org == 0:
            if n_new == 0:
                return np.empty((0, 3), dtype=np.float64)
            result = np.zeros((n_new, 3), dtype=np.float64)
            result[:, :2] = points_new[:, :2]
            return result

        if n_new == 0:
            return np.empty((0, 3), dtype=np.float64)

        # Extract coordinates
        org_xy = points_org[:, :2]
        new_xy = points_new[:, :2]

        # Determine if sequences are forward or reverse aligned
        # Compare first and last points to detect reversal
        forward_dist = np.sum((new_xy[0] - org_xy[0]) ** 2) + np.sum((new_xy[-1] - org_xy[-1]) ** 2)
        reverse_dist = np.sum((new_xy[0] - org_xy[-1]) ** 2) + np.sum((new_xy[-1] - org_xy[0]) ** 2)
        is_reversed = reverse_dist < forward_dist

        # Build result array
        result = np.zeros((n_new, 3), dtype=np.float64)
        result[:, :2] = new_xy

        # Linear index mapping ratio
        if n_new > 1:
            ratio = (n_org - 1) / (n_new - 1) if n_new > 1 else 0
        else:
            ratio = 0

        for j in range(n_new):
            # Expected index in original array
            if is_reversed:
                expected_i = int((n_org - 1) - j * ratio)
            else:
                expected_i = int(j * ratio)

            # Search window bounds
            i_start = max(0, expected_i - search_window)
            i_end = min(n_org, expected_i + search_window + 1)

            # Find nearest within window
            window_xy = org_xy[i_start:i_end]
            dists_sq = np.sum((window_xy - new_xy[j]) ** 2, axis=1)
            best_local = np.argmin(dists_sq)
            best_i = i_start + best_local

            result[j, 2] = points_org[best_i, 2]

        return result

    @staticmethod
    def find_nearest_indices(
        points_org: NDArray[np.float64],
        points_new: NDArray[np.float64],
    ) -> Tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Find the index of the nearest point in points_org for each point in points_new.

        Args:
            points_org: Original points array, shape (N, 2) or (N, 3).
            points_new: New points array, shape (M, 2) or (M, 3).

        Returns:
            Tuple of (indices, distances):
                - indices: Array of shape (M,) with index into points_org for each point_new
                - distances: Array of shape (M,) with Euclidean distance to nearest point
        """
        if points_org.shape[0] == 0 or points_new.shape[0] == 0:
            return (
                np.empty(points_new.shape[0], dtype=np.intp),
                np.empty(points_new.shape[0], dtype=np.float64),
            )

        org_xy = points_org[:, :2]
        new_xy = points_new[:, :2]

        tree = cKDTree(org_xy)
        distances, indices = tree.query(new_xy, k=1)

        return indices, distances

    @staticmethod
    def transfer_types_two_stage(
        points_org: NDArray[np.float64],
        points_new: NDArray[np.float64],
        search_window: int = 8,
        max_residual: float = 5e-3,
    ) -> NDArray[np.float64]:
        """Transfer point types using a two-stage approach for optimal performance.

        Stage 1: Use fast ordered matching for all points.
        Stage 2: For points with poor matches (high residual distance), use KD-tree fallback.

        This hybrid approach is ideal when most points follow the expected order but
        some may be displaced or reordered due to path transformations.

        Args:
            points_org: Original points array, shape (N, 3) with columns (x, y, type).
            points_new: New points array, shape (M, 2) or (M, 3).
            search_window: Window size for the fast ordered matching stage.
            max_residual: Maximum distance threshold to consider a match as "good".
                        Points with residuals above this trigger KD-tree fallback.

        Returns:
            Array of shape (M, 3) with transferred types.
        """
        n_org = points_org.shape[0]
        n_new = points_new.shape[0]

        # Handle empty arrays
        if n_org == 0:
            if n_new == 0:
                return np.empty((0, 3), dtype=np.float64)
            result = np.zeros((n_new, 3), dtype=np.float64)
            result[:, :2] = points_new[:, :2]
            return result

        if n_new == 0:
            return np.empty((0, 3), dtype=np.float64)

        # Stage 1: Fast ordered matching
        result = AvPointMatcher.transfer_point_types_ordered(points_org, points_new, search_window=search_window)

        # Stage 2: Identify and fix poor matches
        # Compute the residuals (distances) for the ordered matches
        org_xy = points_org[:, :2]
        new_xy = points_new[:, :2]

        # Reconstruct which original points were matched by the ordered algorithm
        chosen_idx = np.zeros(n_new, dtype=np.intp)

        # Determine if sequences are forward or reverse aligned
        forward_dist = np.sum((new_xy[0] - org_xy[0]) ** 2) + np.sum((new_xy[-1] - org_xy[-1]) ** 2)
        reverse_dist = np.sum((new_xy[0] - org_xy[-1]) ** 2) + np.sum((new_xy[-1] - org_xy[0]) ** 2)
        is_reversed = reverse_dist < forward_dist

        # Linear index mapping ratio
        if n_new > 1:
            ratio = (n_org - 1) / (n_new - 1)
        else:
            ratio = 0

        for j in range(n_new):
            # Expected index in original array
            if is_reversed:
                expected_i = int((n_org - 1) - j * ratio)
            else:
                expected_i = int(j * ratio)

            # Search window bounds
            i_start = max(0, expected_i - search_window)
            i_end = min(n_org, expected_i + search_window + 1)

            # Find nearest within window
            window_xy = org_xy[i_start:i_end]
            dists_sq = np.sum((window_xy - new_xy[j]) ** 2, axis=1)
            best_local = np.argmin(dists_sq)
            chosen_idx[j] = i_start + best_local

        # Compute residuals (actual distances)
        residuals = np.linalg.norm(new_xy - org_xy[chosen_idx], axis=1)

        # Find points that need fixing (poor matches)
        to_fix = np.where(residuals > max_residual)[0]

        # Stage 3: Fix outliers using KD-tree
        if to_fix.size > 0:
            # Get the problematic points
            problem_points = points_new[to_fix]

            # Use KD-tree matching for these points
            fixed = AvPointMatcher.transfer_point_types(points_org, problem_points)

            # Update only the problematic rows in the result
            result[to_fix] = fixed

        return result

    @staticmethod
    def find_exact_matches(
        points_org: NDArray[np.float64],
        points_new: NDArray[np.float64],
        tolerance: float = 1e-12,
    ) -> List[Tuple[int, int]]:
        """Find exact coordinate matches between two point sets.

        Args:
            points_org: Original points array, shape (N, 2) or (N, 3).
            points_new: New points array, shape (M, 2) or (M, 3).
            tolerance: Maximum distance to consider as exact match.

        Returns:
            List of (j, i) tuples where points_new[j] matches points_org[i] exactly.
        """
        if points_org.shape[0] == 0 or points_new.shape[0] == 0:
            return []

        indices, distances = AvPointMatcher.find_nearest_indices(points_org, points_new)

        exact_matches = []
        for j, (i, d) in enumerate(zip(indices, distances)):
            if d <= tolerance:
                exact_matches.append((j, int(i)))

        return exact_matches


###############################################################################
# AvPathCleaner
###############################################################################
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
