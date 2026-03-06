"""Path exterior processing utilities.

This module contains functionality for processing exterior polygon paths,
including silhouette generation and undercut removal algorithms.
"""

from __future__ import annotations

from typing import List

import numpy as np
from shapely.geometry import LineString

from ave.path import SINGLE_POLYGON_CONSTRAINTS, AvSinglePolygonPath


class AvPathExterior:
    """Utilities for processing exterior polygon paths."""

    @staticmethod
    def left_exterior_silhouette_list(
        exteriors: List[AvSinglePolygonPath],
    ) -> List[AvSinglePolygonPath]:
        """Create left orthographic silhouette polygons with right blocking edge.

        Transforms simple CCW polygons into new polygons representing the left
        orthographic silhouette (undercut-free projection in +x direction) closed
        on the right by a vertical blocking edge at x = max_x.

        The left silhouette consists of all boundary edges whose outward normal
        has a negative x-component. For a CCW polygon, this means edges where
        dy = y2 - y1 < 0 (downward-going edges).

        Args:
            exteriors: List of simple CCW polygons (AvSinglePolygonPath) without curves

        Returns:
            List[AvSinglePolygonPath]: New polygons with:
                - Left boundary: left orthographic silhouette (dy < 0 edges)
                - Right boundary: vertical segment at x = max_x
                - Top/bottom edges: connecting silhouette to blocking edge

        Note:
            - Input must be simple polygons (no self-intersections)
            - Result is x-monotone and suitable for mold pull applications
            - Uses Shapely for robust geometric operations
            - Runs in O(n) time complexity per polygon"""
        result_silhouettes = []

        for exterior in exteriors:
            silhouette = AvPathExterior.left_exterior_silhouette(exterior)
            result_silhouettes.append(silhouette)

        return result_silhouettes

    @staticmethod
    def left_exterior_silhouette(
        exterior: AvSinglePolygonPath,
    ) -> AvSinglePolygonPath:
        """Compute left exterior silhouette for a single polygon.

        Simple vertex-tracing algorithm: traces polygon vertices from the
        point with maximum y-coordinate to the first point with minimum y-coordinate,
        then closes with vertical edge at max x. Undercuts are also removed.

        Args:
            exterior: Simple CCW polygon (AvSinglePolygonPath) without curves

        Returns:
            AvSinglePolygonPath: Silhouette polygon
        """
        if len(exterior.points) == 0:
            return AvSinglePolygonPath(np.empty((0, 3), dtype=np.float64), [], SINGLE_POLYGON_CONSTRAINTS)

        # Get polygon points (x, y only)
        points = exterior.points[:, :2]
        n = len(points)

        if n < 3:
            return AvSinglePolygonPath(np.empty((0, 3), dtype=np.float64), [], SINGLE_POLYGON_CONSTRAINTS)

        # Get bounding box
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])

        if max_y - min_y < 1e-10:
            return AvSinglePolygonPath(np.empty((0, 3), dtype=np.float64), [], SINGLE_POLYGON_CONSTRAINTS)

        # Find all points with maximum y-coordinate
        max_y_indices = np.where(np.abs(points[:, 1] - max_y) < 1e-10)[0]

        # Among points with max y, find the one with minimum x
        max_y_points = points[max_y_indices]
        min_x_among_max_y = np.min(max_y_points[:, 0])
        start_candidates = max_y_indices[np.abs(max_y_points[:, 0] - min_x_among_max_y) < 1e-10]
        start_idx = start_candidates[0]

        # Trace polygon from start_idx FORWARD until we reach a point with min_y
        # Going forward traces the left side for CCW polygons
        silhouette_points = []
        current_idx = start_idx
        found_min_y = False

        for _ in range(n):
            x, y = points[current_idx]
            silhouette_points.append((x, y))

            # Check if we've reached min_y
            if abs(y - min_y) < 1e-10:
                found_min_y = True
                break

            # Move to next point (wrap around if at end)
            current_idx = (current_idx + 1) % n

        if not found_min_y or len(silhouette_points) < 2:
            pts = np.array([[max_x, min_y], [max_x, max_y], [max_x, min_y], [max_x, min_y]])
            return AvSinglePolygonPath(pts, ["M", "L", "L", "L", "Z"], SINGLE_POLYGON_CONSTRAINTS)

        # Close polygon: add (max_x, min_y) and (max_x, max_y)
        final_coords = []
        final_coords.extend(silhouette_points)
        final_coords.append((max_x, min_y))
        final_coords.append((max_x, max_y))

        # Remove consecutive duplicates
        cleaned_coords = [final_coords[0]]
        for i in range(1, len(final_coords)):
            if not (
                abs(final_coords[i][0] - cleaned_coords[-1][0]) < 1e-9
                and abs(final_coords[i][1] - cleaned_coords[-1][1]) < 1e-9
            ):
                cleaned_coords.append(final_coords[i])

        if len(cleaned_coords) < 3:
            pts = np.array([[max_x, min_y], [max_x, max_y], [max_x, min_y], [max_x, min_y]])
            return AvSinglePolygonPath(pts, ["M", "L", "L", "L", "Z"], SINGLE_POLYGON_CONSTRAINTS)

        # Create result
        result_coords = np.array(cleaned_coords)

        # Convert to 3D format for undercut removal
        result_coords_3d = np.zeros((len(result_coords), 3), dtype=np.float64)
        result_coords_3d[:, :2] = result_coords

        # Post-process: remove undercuts to get clean left projection
        result_coords_3d = AvPathExterior._remove_left_upward_undercuts(result_coords_3d)
        result_coords_3d = AvPathExterior._remove_left_downward_undercuts(result_coords_3d)

        n_points = len(result_coords_3d)
        commands = ["M"] + ["L"] * (n_points - 1) + ["Z"]

        return AvSinglePolygonPath(result_coords_3d, commands, SINGLE_POLYGON_CONSTRAINTS)

    @staticmethod
    def _remove_left_upward_undercuts(coords: np.ndarray) -> np.ndarray:
        """Remove inward undercuts from polygon coordinates.

        Removes points that create inward undercuts where the path moves
        right AND upward simultaneously. The algorithm starts from the
        top-leftmost point and processes until the first bottom-most point
        is reached (following CCW order).

        When a point pt(n) is found where the next point pt(n+1) moves
        both right (x increases) AND up (y increases), the algorithm
        searches forward for a point pt(m) where the vector from pt(n) to
        pt(m) moves right AND downward. Instead of jumping directly to pt(m-1),
        the algorithm calculates point s as the intersection between the
        line segment pt(m-1)-pt(m) and a horizontal line at pt(n).y.
        The resulting polygon jumps from pt(n) to point s, then to pt(m).

        Args:
            coords: Array of polygon coordinates as (x, y, z) triples in CCW order

        Returns:
            np.ndarray: Filtered coordinates with upward undercuts removed in CCW order
        """
        if len(coords) < 3:
            return coords

        # Convert to 2D for easier processing
        points_2d = coords[:, :2]
        n_points = len(points_2d)

        # Find top-leftmost point (max y, then min x)
        max_y = np.max(points_2d[:, 1])
        max_y_indices = np.where(np.abs(points_2d[:, 1] - max_y) < 1e-10)[0]
        max_y_points = points_2d[max_y_indices]
        min_x_among_max_y = np.min(max_y_points[:, 0])
        start_candidates = max_y_indices[np.abs(max_y_points[:, 0] - min_x_among_max_y) < 1e-10]
        start_idx = start_candidates[0]

        # Find first bottom-most point (min y)
        min_y = np.min(points_2d[:, 1])

        # Process points from start_idx until we reach min_y
        result_points = []
        current_idx = start_idx
        reached_min_y = False

        for _ in range(n_points):
            if reached_min_y:
                break

            current_point = points_2d[current_idx]
            result_points.append(current_point)

            # Check if we've reached min_y
            if abs(current_point[1] - min_y) < 1e-10:
                reached_min_y = True
                # Add remaining points as-is
                next_idx = (current_idx + 1) % n_points
                while next_idx != start_idx:
                    result_points.append(points_2d[next_idx])
                    next_idx = (next_idx + 1) % n_points
                break

            # Get next point
            next_idx = (current_idx + 1) % n_points
            next_point = points_2d[next_idx]

            # Check for upward undercut condition
            if next_point[1] > current_point[1] + 1e-10 and next_point[0] > current_point[0] + 1e-10:
                # Found upward undercut, search for point with right-downward vector
                search_idx = next_idx
                found_jump_target = False

                for _ in range(n_points - 1):  # Prevent infinite loop
                    search_idx = (search_idx + 1) % n_points
                    search_point = points_2d[search_idx]

                    # Check if vector from current_point to search_point is right-downward
                    dx = search_point[0] - current_point[0]
                    dy = search_point[1] - current_point[1]

                    if dx > 1e-10 and dy < -1e-10:
                        # Found right-downward vector at pt(m)
                        # Calculate intersection point s between pt(m-1) and pt(m)
                        # with same y-coordinate as pt(n)
                        prev_idx = (search_idx - 1) % n_points
                        pt_m_minus_1 = points_2d[prev_idx]
                        pt_m = points_2d[search_idx]
                        pt_n = current_point

                        # Calculate intersection using shapely
                        line_segment = LineString([pt_m_minus_1, pt_m])
                        horizontal_line = LineString([[-1e6, pt_n[1]], [1e6, pt_n[1]]])
                        intersection = line_segment.intersection(horizontal_line)

                        if not intersection.is_empty and intersection.geom_type == "Point":
                            # Found intersection point s
                            point_s = np.array([intersection.x, intersection.y])
                            result_points.append(point_s)
                            result_points.append(pt_m)
                            current_idx = search_idx
                            found_jump_target = True
                            break
                        else:
                            # Fallback: if no intersection found, jump to pt(m-1)
                            current_idx = prev_idx
                            found_jump_target = True
                            break

                    # Also stop if we've looped back
                    if search_idx == start_idx:
                        break

                if not found_jump_target:
                    # If no suitable jump found, just move to next point
                    current_idx = next_idx
            else:
                # No undercut, move to next point
                current_idx = next_idx

        # Convert back to 3D format
        result = np.zeros((len(result_points), 3), dtype=np.float64)
        for i, (x, y) in enumerate(result_points):
            result[i, 0] = x
            result[i, 1] = y
            result[i, 2] = 0.0

        return result

    @staticmethod
    def _remove_left_downward_undercuts(coords: np.ndarray) -> np.ndarray:
        """Remove inward undercuts from polygon coordinates.

        Removes points that create inward undercuts where the path moves
        right AND downward simultaneously. The algorithm starts from the
        bottom-leftmost point and processes until the first top-most point
        is reached (following CW order, i.e. the given coords are iterated in inverse direction)

        When a point pt(n) is found where the next point pt(n+1) moves
        both right (x increases) AND down (y decreases), the algorithm
        searches forward for a point pt(m) where the vector from pt(n) to
        pt(m) moves right AND upward. Instead of jumping directly to pt(m-1),
        the algorithm calculates point s as the intersection between the
        line segment pt(m-1)-pt(m) and a horizontal line at pt(n).y.
        The resulting polygon jumps from pt(n) to point s, then to pt(m).

        Args:
            coords: Array of polygon coordinates as (x, y, z) triples in CCW order

        Returns:
            np.ndarray: Filtered coordinates with downward undercuts removed in CCW order
        """
        if len(coords) < 3:
            return coords

        mirrored_coords = coords[::-1].copy()
        mirrored_coords[:, 1] *= -1.0

        mirrored_result = AvPathExterior._remove_left_upward_undercuts(mirrored_coords)
        result = mirrored_result[::-1]
        result[:, 1] *= -1.0

        return result
