"""Path cleaning and manipulation utilities for vector graphics processing."""

from typing import List, Optional, Tuple

import numpy as np
import shapely.geometry
import shapely.ops
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ave.path import AvClosedPath, AvPath, AvPolygonPath, AvPolylinesPath, AvSinglePath


###############################################################################
# AvPointMatcher
###############################################################################
class AvPointMatcher:
    """Utilities for matching points between two 2D point sets and transferring attributes.

    This class provides efficient nearest-neighbor matching between point arrays,
    primarily used to transfer point type information (on-curve vs control points)
    from an original path to a transformed or resampled path.

    Point types in AvPath:
        Exact matches (within epsilon):
            0.0 = on-curve point (M, L, or endpoint of Q/C)
            2.0 = quadratic Bezier control point (Q)
            3.0 = cubic Bezier control point (C)
        Non-exact matches (nearest neighbor beyond epsilon):
            -1.0 = derived from on-curve point (0.0)
            -2.0 = derived from quadratic control point (2.0)
            -3.0 = derived from cubic control point (3.0)
    """

    # Default epsilon for exact match detection (in coordinate units)
    DEFAULT_EPSILON: float = 1e-6

    @staticmethod
    def _map_type_for_distance(
        org_type: float,
        distance: float,
        epsilon: float,
    ) -> float:
        """Map original type to result type based on match distance.

        Args:
            org_type: Original point type (0.0, 2.0, or 3.0).
            distance: Euclidean distance between matched points.
            epsilon: Threshold for exact match.

        Returns:
            Original type if exact match, otherwise negated type
            (0.0 -> -1.0, 2.0 -> -2.0, 3.0 -> -3.0).
        """
        if distance <= epsilon:
            return org_type
        # Non-exact match: map to negative type
        if org_type == 0.0:
            return -1.0
        elif org_type == 2.0:
            return -2.0
        elif org_type == 3.0:
            return -3.0
        else:
            # Unknown type - return negated value
            return -abs(org_type) if org_type >= 0 else org_type

    @staticmethod
    def transfer_point_types(
        points_org: NDArray[np.float64],
        points_new: NDArray[np.float64],
        epsilon: float | None = None,
    ) -> NDArray[np.float64]:
        """Transfer point types from original points to new points via nearest neighbor matching.

        For each point in points_new, finds the nearest point in points_org and assigns
        the type based on match distance:
        - Exact match (distance <= epsilon): original type (0.0, 2.0, 3.0)
        - Non-exact match: mapped type (-1.0, -2.0, -3.0)

        Uses a KD-tree for O(M log N) performance.

        Args:
            points_org: Original points array, shape (N, 3) with columns (x, y, type).
            points_new: New points array, shape (M, 2) or (M, 3) with columns (x, y, [type]).
                        If 3 columns, the type column will be overwritten.
            epsilon: Distance threshold for exact match. Defaults to DEFAULT_EPSILON.

        Returns:
            Array of shape (M, 3) with (x, y, type) where type depends on match distance.

        Example:
            >>> org = np.array([[0, 0, 0.0], [1, 1, 2.0], [2, 2, 3.0]])
            >>> new = np.array([[0.0, 0.0], [1.9, 1.9]])  # first exact, second not
            >>> result = AvPointMatcher.transfer_point_types(org, new)
            >>> result[:, 2]  # types: [0.0, -3.0]
        """
        if epsilon is None:
            epsilon = AvPointMatcher.DEFAULT_EPSILON

        # Handle empty arrays
        if points_org.shape[0] == 0:
            if points_new.shape[0] == 0:
                return np.empty((0, 3), dtype=np.float64)
            # No source points - default to non-exact on-curve type (-1.0)
            result = np.full((points_new.shape[0], 3), -1.0, dtype=np.float64)
            result[:, :2] = points_new[:, :2]
            return result

        if points_new.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)

        # Extract 2D coordinates for matching
        org_xy = points_org[:, :2]
        new_xy = points_new[:, :2]

        # Build KD-tree from original points
        tree = cKDTree(org_xy)

        # Query nearest neighbors for all new points (get distances too)
        distances, indices = tree.query(new_xy, k=1)

        # Build result array
        result = np.zeros((points_new.shape[0], 3), dtype=np.float64)
        result[:, :2] = new_xy

        # Assign types based on distance
        for j, org_idx in enumerate(indices):
            distance = distances[j]
            org_type = points_org[org_idx, 2]
            result[j, 2] = AvPointMatcher._map_type_for_distance(org_type, distance, epsilon)

        return result

    @staticmethod
    def _ordered_match_indices(
        org_xy: NDArray[np.float64],
        new_xy: NDArray[np.float64],
        search_window: int,
        epsilon: float,
        is_closed: bool | None = None,
    ) -> Tuple[NDArray[np.intp], NDArray[np.float64]]:
        n_org = org_xy.shape[0]
        n_new = new_xy.shape[0]

        if n_new == 0:
            return (
                np.empty(0, dtype=np.intp),
                np.empty(0, dtype=np.float64),
            )

        chosen_idx = np.empty(n_new, dtype=np.intp)
        chosen_dist = np.empty(n_new, dtype=np.float64)

        tree = cKDTree(org_xy)
        offsets = np.arange(-search_window, search_window + 1, dtype=np.intp)

        if n_org > 1:
            step_lengths = np.linalg.norm(org_xy - np.roll(org_xy, 1, axis=0), axis=1)
            typical_step = float(np.median(step_lengths))
        else:
            typical_step = 0.0

        if not np.isfinite(typical_step) or typical_step <= 0.0:
            typical_step = 1.0

        # Auto-detect if sequence is closed unless explicitly specified
        if is_closed is None:
            is_closed_like = False
            if n_org >= 3:
                first_last_dist = float(np.linalg.norm(org_xy[0] - org_xy[-1]))
                is_closed_like = first_last_dist < 2.0 * typical_step
        else:
            is_closed_like = is_closed

        if is_closed_like:
            ratio = (n_org / n_new) if n_new > 0 else 0.0
        else:
            ratio = (n_org - 1) / (n_new - 1) if n_new > 1 else 0.0

        # For open sequences, anchor at start (index 0) instead of using KD-tree
        if is_closed_like:
            i0 = int(tree.query(new_xy[0], k=1)[1])
        else:
            i0 = 0

        if n_new <= 1:
            chosen_idx[0] = i0
            chosen_dist[0] = float(tree.query(new_xy[0], k=1)[0])
            return chosen_idx, chosen_dist

        sample_count = min(5, n_new)
        sample_js = np.unique(np.linspace(0, n_new - 1, num=sample_count, dtype=int))

        def _sample_cost(sign: float) -> float:
            anchor = float(i0)
            cost = 0.0
            for j in sample_js:
                expected_i = int(round(anchor + sign * j * ratio))
                if is_closed_like:
                    cand = (expected_i + offsets) % n_org
                else:
                    cand = expected_i + offsets
                    cand = cand[(cand >= 0) & (cand < n_org)]
                    if cand.size == 0:
                        cand = np.array([int(np.clip(expected_i, 0, n_org - 1))], dtype=np.intp)
                d2 = np.sum((org_xy[cand] - new_xy[j]) ** 2, axis=1)
                cost += float(np.min(d2))
            return cost

        is_reversed = _sample_cost(-1.0) < _sample_cost(1.0)
        sign = -1.0 if is_reversed else 1.0

        anchor = float(i0)
        hard_threshold = max(10.0 * epsilon, 3.0 * typical_step)

        for j in range(n_new):
            expected_i = int(round(anchor + sign * j * ratio))
            if is_closed_like:
                cand = (expected_i + offsets) % n_org
                cand_offsets = offsets
            else:
                cand = expected_i + offsets
                mask = (cand >= 0) & (cand < n_org)
                cand = cand[mask]
                cand_offsets = offsets[mask]
                if cand.size == 0:
                    cand = np.array([int(np.clip(expected_i, 0, n_org - 1))], dtype=np.intp)
                    cand_offsets = np.array([0], dtype=np.intp)

            d2 = np.sum((org_xy[cand] - new_xy[j]) ** 2, axis=1)
            best_local = int(np.argmin(d2))
            best_i = int(cand[best_local])
            best_dist = float(np.sqrt(d2[best_local]))
            best_offset = int(cand_offsets[best_local])

            if (abs(best_offset) == search_window and best_dist > typical_step) or best_dist > hard_threshold:
                i_global = int(tree.query(new_xy[j], k=1)[1])
                anchor = float(i_global) - sign * j * ratio
                expected_i = int(round(anchor + sign * j * ratio))
                if is_closed_like:
                    cand = (expected_i + offsets) % n_org
                else:
                    cand = expected_i + offsets
                    cand = cand[(cand >= 0) & (cand < n_org)]
                    if cand.size == 0:
                        cand = np.array([int(np.clip(expected_i, 0, n_org - 1))], dtype=np.intp)
                d2 = np.sum((org_xy[cand] - new_xy[j]) ** 2, axis=1)
                best_local = int(np.argmin(d2))
                best_i = int(cand[best_local])
                best_dist = float(np.sqrt(d2[best_local]))

            chosen_idx[j] = best_i
            chosen_dist[j] = best_dist

        return chosen_idx, chosen_dist

    @staticmethod
    def transfer_point_types_ordered(
        points_org: NDArray[np.float64],
        points_new: NDArray[np.float64],
        search_window: int = 10,
        epsilon: float | None = None,
        is_closed: bool | None = None,
    ) -> NDArray[np.float64]:
        """Transfer point types assuming roughly ordered/aligned point sequences.

        Optimized O(M * W) algorithm for cases where points_new and points_org
        are in similar order (possibly reversed). Uses a sliding window search
        instead of a full KD-tree.

        Types are assigned based on match distance:
        - Exact match (distance <= epsilon): original type (0.0, 2.0, 3.0)
        - Non-exact match: mapped type (-1.0, -2.0, -3.0)

        Args:
            points_org: Original points array, shape (N, 3) with columns (x, y, type).
            points_new: New points array, shape (M, 2) or (M, 3).
            search_window: Number of points to search around the expected position.
                            Larger values handle more deviation but are slower.
            epsilon: Distance threshold for exact match. Defaults to DEFAULT_EPSILON.
            is_closed: Whether the path is closed (loop). If None, auto-detects.
                        For closed paths, the search wraps around at the ends.
                        For open paths, search is bounded within array indices.

        Returns:
            Array of shape (M, 3) with transferred types.
        """
        if epsilon is None:
            epsilon = AvPointMatcher.DEFAULT_EPSILON

        n_org = points_org.shape[0]
        n_new = points_new.shape[0]

        # Handle empty arrays
        if n_org == 0:
            if n_new == 0:
                return np.empty((0, 3), dtype=np.float64)
            result = np.full((n_new, 3), -1.0, dtype=np.float64)
            result[:, :2] = points_new[:, :2]
            return result

        if n_new == 0:
            return np.empty((0, 3), dtype=np.float64)

        # Extract coordinates
        org_xy = points_org[:, :2]
        new_xy = points_new[:, :2]

        # Determine alignment (including possible reversal) and windowed matches
        # Uses initial anchoring + re-alignment when local matching becomes unreliable
        indices, distances = AvPointMatcher._ordered_match_indices(org_xy, new_xy, search_window, epsilon, is_closed)

        # Build result array
        result = np.zeros((n_new, 3), dtype=np.float64)
        result[:, :2] = new_xy

        # Linear index mapping ratio
        for j, org_idx in enumerate(indices):
            org_type = points_org[org_idx, 2]
            result[j, 2] = AvPointMatcher._map_type_for_distance(org_type, distances[j], epsilon)

        return result

    @staticmethod
    def transfer_types_two_stage(
        points_org: NDArray[np.float64],
        points_new: NDArray[np.float64],
        search_window: int = 8,
        max_residual: float = 5e-3,
        epsilon: float | None = None,
        is_closed: bool | None = None,
    ) -> NDArray[np.float64]:
        """Transfer point types using a two-stage approach for optimal performance.

        Stage 1: Use fast ordered matching for all points.
        Stage 2: For points with poor matches (high residual distance), use KD-tree fallback.

        This hybrid approach is ideal when most points follow the expected order but
        some may be displaced or reordered due to path transformations.

        Types are assigned based on match distance:
        - Exact match (distance <= epsilon): original type (0.0, 2.0, 3.0)
        - Non-exact match: mapped type (-1.0, -2.0, -3.0)

        Args:
            points_org: Original points array, shape (N, 3) with columns (x, y, type).
            points_new: New points array, shape (M, 2) or (M, 3).
            search_window: Window size for the fast ordered matching stage.
            max_residual: Maximum distance threshold to consider a match as "good".
                        Points with residuals above this trigger KD-tree fallback.
            epsilon: Distance threshold for exact match. Defaults to DEFAULT_EPSILON.

        Returns:
            Array of shape (M, 3) with transferred types.
        """
        if epsilon is None:
            epsilon = AvPointMatcher.DEFAULT_EPSILON

        n_org = points_org.shape[0]
        n_new = points_new.shape[0]

        # Handle empty arrays
        if n_org == 0:
            if n_new == 0:
                return np.empty((0, 3), dtype=np.float64)
            result = np.full((n_new, 3), -1.0, dtype=np.float64)
            result[:, :2] = points_new[:, :2]
            return result

        if n_new == 0:
            return np.empty((0, 3), dtype=np.float64)

        # Stage 1: Fast ordered matching (with epsilon for exact/non-exact distinction)
        result = AvPointMatcher.transfer_point_types_ordered(
            points_org, points_new, search_window=search_window, epsilon=epsilon, is_closed=is_closed
        )

        # Stage 2: Identify and fix poor matches
        # Compute the residuals (distances) for the ordered matches
        org_xy = points_org[:, :2]
        new_xy = points_new[:, :2]

        # Reconstruct which original points were matched by the ordered algorithm
        _, residuals = AvPointMatcher._ordered_match_indices(org_xy, new_xy, search_window, epsilon, is_closed)

        # Find points that need fixing (poor matches)
        to_fix = np.where(residuals > max_residual)[0]

        # Stage 3: Fix outliers using KD-tree
        if to_fix.size > 0:
            # Get the problematic points
            problem_points = points_new[to_fix]

            # Use KD-tree matching for these points (with epsilon)
            fixed = AvPointMatcher.transfer_point_types(points_org, problem_points, epsilon=epsilon)

            # Update only the problematic rows in the result
            result[to_fix] = fixed

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
        hole_polygons: List[shapely.geometry.Polygon] = []

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

                # Check if this is a hole by examining the original contour orientation
                # CCW = exterior, CW = hole
                is_hole = not closed_path.is_ccw

                # Handle different geometry types returned by buffer(0)
                if isinstance(cleaned_poly, shapely.geometry.Polygon) and not cleaned_poly.is_empty:
                    if is_hole:
                        hole_polygons.append(cleaned_poly)
                    else:
                        cleaned_polygons.append(cleaned_poly)

                elif isinstance(cleaned_poly, shapely.geometry.MultiPolygon):
                    # Add all sub-polygons
                    for poly in cleaned_poly.geoms:
                        if not poly.is_empty:
                            if is_hole:
                                hole_polygons.append(poly)
                            else:
                                cleaned_polygons.append(poly)

                elif isinstance(cleaned_poly, shapely.geometry.GeometryCollection):
                    # Extract Polygon types
                    for geom in cleaned_poly.geoms:
                        if isinstance(geom, shapely.geometry.Polygon) and not geom.is_empty:
                            if is_hole:
                                hole_polygons.append(geom)
                            else:
                                cleaned_polygons.append(geom)

            except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
                print(f"Warning: Failed to process contour {e}. Skipping.")
                continue

        if not cleaned_polygons:
            return AvPath()

        # Step 5: Union all cleaned polygons together and subtract holes
        try:
            # First union all exterior polygons
            if cleaned_polygons:
                result = shapely.ops.unary_union(cleaned_polygons)
            else:
                return AvPath()

            # Then subtract all hole polygons
            if hole_polygons:
                holes_union = shapely.ops.unary_union(hole_polygons)
                result = result.difference(holes_union)

            if result.is_empty:
                return path  # Return original path if result is empty

        except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
            print(f"Error during union operation: {e}")
            return path  # Return original path on error

        # Step 6: Convert final Shapely geometry back to AvPath format
        cleaned_paths: List[AvPath] = []

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

        # Join all paths
        if cleaned_paths:
            joined = AvPath.join_paths(*cleaned_paths)
            return joined
        else:
            return AvPath()

    @staticmethod
    def resolve_polygonized_path_intersections(path: AvPolylinesPath) -> AvPolylinesPath:
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
        7. Convert final Shapely geometry back to AvPolylinesPath format:
            - Extract exterior rings as closed paths with 'Z' command
            - Extract interior rings (holes) as separate paths
            - Join all paths using AvPath.join_paths
            - Convert result to AvPolylinesPath explicitly

        Key technical details:
        - Uses orientation from AvClosedPath.is_ccw to determine winding
        - Implements deferred processing for CW polygons before first CCW
        - Comprehensive error handling with fallback to original path
        - Removes duplicate closing points when converting coordinates

        The function handles the following cases:
        - Empty input paths: returns empty AvPolylinesPath
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
            path: An AvPolylinesPath containing the segments to process

        Returns:
            AvPolylinesPath: A new path with resolved intersections and proper winding,
                            or the original path if errors occur
        """
        # Split path into individual segments
        segments = path.split_into_single_paths()

        # Process each segment to ensure it's closed
        polygons: List[AvPolygonPath] = []
        orientations: List[bool] = []  # Store CCW orientation from closed paths

        for segment in segments:
            # Create AvClosedPath first, then get polygonized path
            try:
                closed_path = AvClosedPath.from_single_path(segment)
                polygonized = closed_path.polygonized_path()
                polygons.append(polygonized)
                orientations.append(closed_path.is_ccw)  # Store orientation from closed path
            except (TypeError, ValueError) as e:
                print(f"Error processing segment: {e}. Skipping.")
                continue

        if not polygons:
            return AvPolylinesPath()

        # Sequentially combine polygons using the first CCW polygon as base
        # Store early CW polygons to defer them until we find the first CCW
        deferred_cw_polygons: List[shapely.geometry.base.BaseGeometry] = []
        result: Optional[shapely.geometry.base.BaseGeometry] = None
        first_ccw_found = False

        try:
            for i, polygon in enumerate(polygons):
                # Use stored orientation from closed path
                is_ccw = orientations[i]

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
                return AvPolylinesPath()

            if result.is_empty:
                print("Warning: Result is empty after operations. Returning original path.")
                return path

        except (shapely.errors.ShapelyError, ValueError, TypeError) as e:
            print(f"Error during polygon processing: {e}. Returning original path.")
            return path  # Return original path on error

        # Convert final Shapely geometry back to AvPolylinesPath
        cleaned_paths: List[AvPath] = []

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

        # Join all paths and return as AvPolylinesPath
        if cleaned_paths:
            try:
                joined = AvPath.join_paths(*cleaned_paths)
                # Convert to AvPolylinesPath explicitly
                return AvPolylinesPath(joined.points, joined.commands)
            except (TypeError, ValueError) as e:
                print(f"Error during path joining: {e}. Returning original path.")
                return path
        else:
            print("Warning: No valid paths to join. Returning original path.")
            return path
