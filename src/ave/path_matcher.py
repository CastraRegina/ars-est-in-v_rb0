"""Path matching module for transferring type information between polygon paths."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from ave.path import AvMultiPolygonPath, AvPath
from ave.path_support import MULTI_POLYGON_CONSTRAINTS


class AvPathMatcher:
    """Transfer type information between polygon paths using segment-wise matching.

    This class matches segments between two polygon paths by:
    1. Splitting both paths into individual segments
    2. Finding segments with overlapping bounding boxes
    3. Selecting the best match for each segment based on spatial proximity
    4. Using KD-tree to match points within matched segments
    5. Handling intersecting segments that may have been reordered or split

    The matching uses segment-wise correspondence with KD-tree spatial search
    to efficiently find matching points within a small tolerance.
    """

    TOLERANCE: float = 1e-10
    UNMATCHED_TYPE: float = -1.0

    @classmethod
    def match_paths(cls, path_org: AvMultiPolygonPath, path_new: AvMultiPolygonPath) -> AvMultiPolygonPath:
        """Transfer type information between polygon paths using segment-wise matching.

        This method matches segments between two polygon paths by:
        1. Splitting both paths into individual segments
        2. Finding segments with overlapping bounding boxes
        3. Selecting the best match for each segment based on spatial proximity
        4. Using KD-tree to match points within matched segments
        5. Handling intersecting segments that may have been reordered or split

        Args:
            path_org: Original path with type information (3rd column of points).
            path_new: New path to receive type information.

        Returns:
            New AvMultiPolygonPath with matched type information.
        """
        # Split both paths into segments
        segments_org = path_org.split_into_single_paths()
        segments_new = path_new.split_into_single_paths()

        # Build bounding boxes for all segments
        bboxes_org = [seg.bounding_box() for seg in segments_org]
        bboxes_new = [seg.bounding_box() for seg in segments_new]

        # Track point ranges for each segment in the new path
        segment_ranges = cls._get_segment_point_ranges(segments_new)

        # Initialize result points with unmatched types
        result_points = path_new.points.copy()
        result_points[:, 2] = cls.UNMATCHED_TYPE

        # For each new segment, find the best matching original segment
        for new_idx, (new_segment, new_bbox, point_range) in enumerate(zip(segments_new, bboxes_new, segment_ranges)):
            # Find all original segments with overlapping bounding boxes
            overlapping_indices = []
            for org_idx, org_bbox in enumerate(bboxes_org):
                if new_bbox.overlaps(org_bbox):
                    overlapping_indices.append(org_idx)

            if not overlapping_indices:
                # No overlapping segments, keep as unmatched
                continue

            # Find the best matching segment
            best_match_idx = cls._find_best_segment_match(new_segment, [segments_org[i] for i in overlapping_indices])

            if best_match_idx is not None:
                # Match points within the best matching segments
                start, end = point_range
                segment_result = result_points[start:end]
                segment_result[:, 2] = cls.UNMATCHED_TYPE

                cls._match_segment_points(
                    segments_org[overlapping_indices[best_match_idx]], new_segment, segment_result
                )

                result_points[start:end] = segment_result

        return AvPath(result_points, list(path_new.commands), MULTI_POLYGON_CONSTRAINTS)

    @classmethod
    def _get_segment_point_ranges(cls, segments: List[AvPath]) -> List[Tuple[int, int]]:
        """Get the start and end indices for each segment in the original path.

        Args:
            segments: List of segments from split_into_single_paths()

        Returns:
            List of (start_idx, end_idx) tuples for each segment
        """
        ranges = []
        current_idx = 0

        for segment in segments:
            start_idx = current_idx
            end_idx = current_idx + len(segment.points)
            ranges.append((start_idx, end_idx))
            current_idx = end_idx

        return ranges

    @classmethod
    def _find_best_segment_match(cls, new_segment: AvPath, candidate_segments: List[AvPath]) -> Optional[int]:
        """Find the best matching segment from candidates.

        Args:
            new_segment: The segment to find a match for.
            candidate_segments: List of candidate segments to match against.

        Returns:
            Index of best matching segment, or None if no good match.
        """
        if not candidate_segments:
            return None
        if len(candidate_segments) == 1:
            return 0

        # Score each candidate based on multiple factors
        best_score = -np.inf
        best_idx = 0

        for idx, candidate in enumerate(candidate_segments):
            score = 0

            # Factor 1: Bounding box overlap area (larger is better)
            if new_segment.bounding_box().overlaps(candidate.bounding_box()):
                # Calculate overlap area
                new_bbox = new_segment.bounding_box()
                cand_bbox = candidate.bounding_box()

                overlap_xmin = max(new_bbox.xmin, cand_bbox.xmin)
                overlap_ymin = max(new_bbox.ymin, cand_bbox.ymin)
                overlap_xmax = min(new_bbox.xmax, cand_bbox.xmax)
                overlap_ymax = min(new_bbox.ymax, cand_bbox.ymax)

                overlap_area = max(0, overlap_xmax - overlap_xmin) * max(0, overlap_ymax - overlap_ymin)
                score += overlap_area

                # Factor 1b: For nested polygons, prefer similar sizes
                new_area = new_bbox.area
                cand_area = cand_bbox.area
                area_ratio = min(new_area, cand_area) / max(new_area, cand_area)
                score += area_ratio * 100  # Weight this heavily

            # Factor 2: Similarity in point count (closer is better)
            point_diff = abs(len(new_segment.points) - len(candidate.points))
            score -= point_diff * 0.1

            # Factor 3: Centroid distance (closer is better)
            centroid_dist = np.linalg.norm(np.array(new_segment.centroid) - np.array(candidate.centroid))
            score -= centroid_dist * 0.01

            # Factor 4: For holes (CW polygons), match with other CW polygons
            new_is_ccw = new_segment.is_ccw
            cand_is_ccw = candidate.is_ccw
            if new_is_ccw == cand_is_ccw:
                score += 50  # Bonus for matching winding direction

            if score > best_score:
                best_score = score
                best_idx = idx

        # Only return match if score is reasonable
        if best_score > -100:  # Threshold to avoid bad matches
            return best_idx
        return None

    @classmethod
    def _match_segment_points(
        cls, org_segment: AvPath, new_segment: AvPath, segment_result: NDArray[np.float64]
    ) -> None:
        """Match points between two segments using KD-tree.

        Args:
            org_segment: Original segment with type information.
            new_segment: New segment to receive type information.
            segment_result: Result array for this segment to update with matched types.
        """
        # Get points for matching
        org_points = org_segment.points[:, :2]  # Use only x, y for matching
        new_points = new_segment.points[:, :2]
        org_types = org_segment.points[:, 2]

        if len(org_points) == 0 or len(new_points) == 0:
            return

        # Build KD-tree for original points
        tree = KDTree(org_points)

        # Find nearest original point for each new point
        distances, indices = tree.query(new_points, k=1)

        # Match types for points within tolerance
        matched_types = np.where(distances < cls.TOLERANCE, org_types[indices], cls.UNMATCHED_TYPE)

        # Update segment result
        segment_result[:, 2] = matched_types
