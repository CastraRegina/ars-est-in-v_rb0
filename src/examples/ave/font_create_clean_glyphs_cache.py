"""Single font glyph details SVG page example."""

from typing import Dict, Optional

import numpy as np
from fontTools.ttLib import TTFont

from ave.font import AvFont
from ave.fonttools import FontHelper
from ave.geom import AvBox
from ave.glyph import (
    AvGlyph,
    AvGlyphCachedSourceFactory,
    AvGlyphFromTTFontFactory,
    AvGlyphPersistentFactory,
    AvGlyphPolygonizeFactory,
    AvLetter,
)
from ave.page import AvSvgPage
from ave.path import AvPath
from ave.path_processing import AvPathCleaner, AvPathCurveRebuilder, AvPathMatcher


def draw_viewbox_border(svg_page, vb_scale, viewbox_width, viewbox_height):
    """Draw the viewbox border."""
    svg_page.add(
        svg_page.drawing.path(
            d=(
                f"M 0 0 "
                f"L {vb_scale * viewbox_width} 0 "  # = (1.0, 0.0)
                f"L {vb_scale * viewbox_width} {vb_scale * viewbox_height} "
                f"L 0 {vb_scale * viewbox_height} "
                f"Z"
            ),
            stroke="blue",
            stroke_width=0.1 * vb_scale,
            fill="none",
        ),
        False,
    )


def setup_avfont(ttfont_filename: str, axes_values: Optional[Dict[str, float]] = None):
    """
    Setup an AvFont object from a given TrueType font file and optional axes values.
    """

    if axes_values is None:
        ttfont = TTFont(ttfont_filename)
    else:
        ttfont = FontHelper.instantiate_ttfont(TTFont(ttfont_filename), axes_values)

    # polygonize_steps=0 => no polygonization
    polygonize_steps = 0
    glyph_factory_ttfont = AvGlyphFromTTFontFactory(ttfont)
    glyph_factory_polygonized = AvGlyphPolygonizeFactory(glyph_factory_ttfont, polygonize_steps)
    glyph_factory_cached = AvGlyphCachedSourceFactory(glyph_factory_polygonized)
    avfont = AvFont(glyph_factory_cached)
    return avfont


def print_text(svg_page: AvSvgPage, xpos: float, ypos: float, text: str, avfont: AvFont, font_size: float) -> None:
    """Print text on the given svg_page at the given position with the given font and font size."""
    current_xpos = xpos
    for character in text:
        glyph = avfont.get_glyph(character)
        letter = AvLetter.from_font_size_units_per_em(glyph, font_size, avfont.props.units_per_em, current_xpos, ypos)
        svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
        svg_page.add(svg_path)
        current_xpos += letter.width


def create_new_q_tail(bbox: AvBox, dash_thickness: float) -> AvPath:
    """Create a new Q-tail segment with quadratic curves.

    Args:
        bbox: Bounding box of the shifted Q-tail segment
        dash_thickness: Thickness of the dash stroke

    Returns:
        New Q-tail segment with quadratic curves at top-left and bottom-right
    """
    # Calculate points for a diagonal beam (top-left to bottom-right) with thickness dash_thickness
    # The beam is a parallelogram centered on the main diagonal

    # Calculate the diagonal direction vector (from top-left to bottom-right)
    diag_x = bbox.xmax - bbox.xmin
    diag_y = bbox.ymin - bbox.ymax
    diag_length = np.sqrt(diag_x**2 + diag_y**2)

    # Normalize diagonal direction
    diag_dir_x = diag_x / diag_length
    diag_dir_y = diag_y / diag_length

    # Perpendicular direction (90° rotation for thickness)
    perp_x = -diag_dir_y
    perp_y = diag_dir_x

    # Half thickness - the beam is centered on the diagonal
    half_t = dash_thickness / 2

    # The main diagonal passes through (bbox.xmin, bbox.ymax) to (bbox.xmax, bbox.ymin)
    # Line 1 (upper edge): offset by +half_t in perpendicular direction
    # Line 2 (lower edge): offset by -half_t in perpendicular direction

    # Reference point on the diagonal (top-left corner)
    ref_x = bbox.xmin
    ref_y = bbox.ymax

    # Line 1: upper edge of beam (offset +half_t perpendicular)
    line1_x = ref_x + perp_x * half_t
    line1_y = ref_y + perp_y * half_t

    # Line 2: lower edge of beam (offset -half_t perpendicular)
    line2_x = ref_x - perp_x * half_t
    line2_y = ref_y - perp_y * half_t

    # Find intersection of line 1 with top edge (y = bbox.ymax)
    # Point 1: on top edge
    if abs(diag_dir_y) > 1e-10:
        t1 = (bbox.ymax - line1_y) / diag_dir_y
        p1_x = line1_x + t1 * diag_dir_x
    else:
        p1_x = bbox.xmin
    p1_y = bbox.ymax

    # Find intersection of line 1 with right edge (x = bbox.xmax)
    # Point 2: on right edge
    if abs(diag_dir_x) > 1e-10:
        t2 = (bbox.xmax - line1_x) / diag_dir_x
        p2_y = line1_y + t2 * diag_dir_y
    else:
        p2_y = bbox.ymin
    p2_x = bbox.xmax

    # Find intersection of line 2 with bottom edge (y = bbox.ymin)
    # Point 3: on bottom edge
    if abs(diag_dir_y) > 1e-10:
        t3 = (bbox.ymin - line2_y) / diag_dir_y
        p3_x = line2_x + t3 * diag_dir_x
    else:
        p3_x = bbox.xmax
    p3_y = bbox.ymin

    # Find intersection of line 2 with left edge (x = bbox.xmin)
    # Point 4: on left edge
    if abs(diag_dir_x) > 1e-10:
        t4 = (bbox.xmin - line2_x) / diag_dir_x
        p4_y = line2_y + t4 * diag_dir_y
    else:
        p4_y = bbox.ymax
    p4_x = bbox.xmin

    # CCW order: 1 (top) -> 4 (left) -> 3 (bottom) -> 2 (right)
    # Add control points at bbox corners for quadratic curves
    points = np.array(
        [
            [p1_x, p1_y, 0],  # 1: top edge
            [bbox.xmin, bbox.ymax, 2.0],  # 2: control point (top-left corner)
            [p4_x, p4_y, 0],  # 3: left edge
            [p3_x, p3_y, 0],  # 4: bottom edge
            [bbox.xmax, bbox.ymin, 2.0],  # 5: control point (bottom-right corner)
            [p2_x, p2_y, 0],  # 6: right edge
        ]
    )

    segment = AvPath(points, np.array(["M", "Q", "L", "Q", "Z"]))
    return segment


def customize_glyph(glyph: AvGlyph, dash_thickness: float) -> AvGlyph:
    """Update a glyph after the revise step.

    Algorithm Steps (for 'Q' glyph):
    1. Split the glyph path into individual segments using split_into_single_paths()
        - This separates the outer circle, inner hole, and tail into distinct paths

    2. Filter for CCW segments (exterior polygons):
        - Use segment.is_ccw to identify exterior contours (counter-clockwise)
        - Exclude interior holes (clockwise) from processing
        - The Q glyph typically has 2 CCW segments: outer circle and tail

    3. Identify the tail segment:
        - Use scoring function: score = centroid_x - centroid_y
        - Higher score = more right (high x) and more bottom (low y)
        - The segment with highest score is the tail

    4. Calculate bounding boxes:
        - Get bounding box of the tail segment
        - Calculate combined bounding box of all other segments

    5. Compute shift to align tail:
        - shift_x = combined_bbox.xmax - tail_bbox.xmax (align right edges)
        - shift_y = combined_bbox.ymin - tail_bbox.ymin (align bottom edges)
        - This positions the tail at the bottom-right of the main body

    6. Apply transformation:
        - Shift all points in the tail segment by (shift_x, shift_y)
        - Create new segment with shifted coordinates

    7. Replace line with quadratic curve:
        - Find two most bottom-right points in the shifted tail
        - Replace the line between them with a quadratic curve
        - Control point positioned at max-x, min-y of the segment

    8. Reconstruct the glyph:
        - Replace original tail with shifted, curved tail
        - Preserve original segment order
        - Join all segments back into a single path

    Args:
        glyph: The glyph to potentially update

    Returns:
        A new glyph if modifications were made, or the same glyph if no changes
    """

    if glyph.character == "Q":
        path = glyph.path

        # 1. Split the path into segments
        segments = path.split_into_single_paths()

        # 2. Filter for CCW segments (positive/exterior segments)
        ccw_segments = []
        for segment in segments:
            if segment.area > 0 and hasattr(segment, "is_ccw") and segment.is_ccw:
                ccw_segments.append(segment)

        # If we have less than 2 CCW segments, return original glyph
        if len(ccw_segments) < 2:
            return glyph

        # 3. Identify the tail segment by centroid position (most bottom-right)
        # Use evaluation function: maximize x, minimize y
        # Score = x - y (higher x increases score, lower y increases score)
        tail_segment = max(ccw_segments, key=lambda s: s.bounding_box().centroid[0] - s.bounding_box().centroid[1])

        # 4. Get the bounding box of the tail segment
        small_bbox = tail_segment.bounding_box()

        # 5. Create a list of all other segments (excluding the tail)
        other_segments = [s for s in segments if s is not tail_segment]

        # 6. Calculate the combined bounding box of all other segments
        if other_segments:
            # Start with the first segment's bbox
            combined_bbox = other_segments[0].bounding_box()

            # Expand to include all other segments
            for segment in other_segments[1:]:
                seg_bbox = segment.bounding_box()
                combined_bbox = AvBox(
                    min(combined_bbox.xmin, seg_bbox.xmin),
                    min(combined_bbox.ymin, seg_bbox.ymin),
                    max(combined_bbox.xmax, seg_bbox.xmax),
                    max(combined_bbox.ymax, seg_bbox.ymax),
                )

            # 7. Calculate the shift needed to position tail at bottom-right
            # Align tail so it extends from the bottom-right of the main body
            shift_x = combined_bbox.xmax - small_bbox.xmax  # Align right edges
            shift_y = combined_bbox.ymin - small_bbox.ymin  # Align bottom edges

            # Apply the shift to all points in the tail segment
            shifted_points = tail_segment.points.copy()
            shifted_points[:, 0] += shift_x  # Shift x coordinates
            shifted_points[:, 1] += shift_y  # Shift y coordinates

            # Create a new segment with shifted points
            shifted_segment = AvPath(shifted_points[:, :2], tail_segment.commands)  # Use only x, y coordinates

            # # 8. Ensure the segment starts with the topmost point for proper L-command connections
            # # Find the index of the point with maximum y-coordinate
            # points = shifted_segment.points[:, :2]
            # max_y_idx = np.argmax(points[:, 1])

            # # Rotate sequence so the topmost point becomes the first point
            # if max_y_idx != 0:
            #     # Copy of rotate_segment_points logic to avoid calling private method
            #     pts = shifted_segment.points
            #     rotated_pts = np.concatenate([pts[max_y_idx:], pts[:max_y_idx]])
            #     shifted_segment = AvPath(rotated_pts, shifted_segment.commands)

            # # IMPORTANT: Get the bounding box NOW, before any curve modifications
            # # This ensures both control points are calculated from the original polygon
            # original_bbox = shifted_segment.bounding_box()

            # # Find top-left points
            # points_tl = shifted_segment.points[:, :2]
            # scores_tl = -points_tl[:, 0] + points_tl[:, 1]
            # sorted_indices_tl = np.argsort(scores_tl)[::-1]
            # tl_idx1, tl_idx2 = sorted_indices_tl[0], sorted_indices_tl[1]

            # # Find bottom-right points (before any modifications)
            # scores_br = points_tl[:, 0] - points_tl[:, 1]
            # sorted_indices_br = np.argsort(scores_br)[::-1]
            # br_idx1, br_idx2 = sorted_indices_br[0], sorted_indices_br[1]

            # # 9. Apply top-left curve (control point at min-x, max-y)
            # tl_control = np.array([original_bbox.xmin, original_bbox.ymax])
            # shifted_segment = replace_line_with_curve(shifted_segment, tl_idx1, tl_idx2, tl_control)

            # # Adjust bottom-right indices if they shifted due to control point insertion
            # # The top-left curve inserts a point at tl_idx2 (the larger index after sorting)
            # insert_idx = max(tl_idx1, tl_idx2)
            # if br_idx1 >= insert_idx:
            #     br_idx1 += 1
            # if br_idx2 >= insert_idx:
            #     br_idx2 += 1

            # # 10. Apply bottom-right curve (control point at max-x, min-y)
            # br_control = np.array([original_bbox.xmax, original_bbox.ymin])
            # shifted_segment = replace_line_with_curve(shifted_segment, br_idx1, br_idx2, br_control)

            shifted_segment = create_new_q_tail(shifted_segment.bounding_box(), dash_thickness)

            # 11. Rebuild the path with the modified segment, preserving original order
            new_segments = []
            for segment in segments:
                if segment is tail_segment:
                    new_segments.append(shifted_segment)
                else:
                    new_segments.append(segment)

            # Join all segments back together in the same sequence
            if len(new_segments) > 0:
                new_path = new_segments[0]
                for segment in new_segments[1:]:
                    new_path = new_path.append(segment)
            else:
                new_path = path

            return AvGlyph(glyph.character, glyph.width(), new_path)
        return glyph

    return glyph


def replace_line_with_curve(segment: AvPath, idx1: int, idx2: int, control_point: np.ndarray) -> AvPath:
    """Replace the line between two points with a quadratic curve.

    Args:
        segment: The path segment to modify
        idx1, idx2: Indices of the two points to connect with a curve
        control_point: The control point for the quadratic curve

    Returns:
        Modified segment with quadratic curve
    """
    if len(segment.points) < 2:
        return segment

    # Ensure idx1 < idx2
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1

    # Must be consecutive
    if abs(idx1 - idx2) != 1:
        return segment

    new_points = segment.points.copy()
    new_commands = segment.commands.copy()

    if idx2 < len(new_commands) and new_commands[idx2] in ("L", "Z"):
        is_closing = new_commands[idx2] == "Z"
        new_commands[idx2] = "Q"
        control_with_type = np.array([control_point[0], control_point[1], 2.0])
        new_points = np.insert(new_points, idx2, control_with_type, axis=0)
        if is_closing:
            new_commands = np.append(new_commands, "Z")
            start_point = new_points[0].copy()
            new_points = np.vstack([new_points, start_point])

    return AvPath(new_points, new_commands)


def replace_line_with_quadratic_curve_top_left_simple(segment: AvPath, idx1: int, idx2: int) -> AvPath:
    """Replace the line between two specified points (top-left) with a quadratic curve.

    Simplified version - assumes points are consecutive and no Z-connection handling needed.

    Args:
        segment: The path segment to modify
        idx1, idx2: Indices of the two points to connect with a curve (must be consecutive)

    Returns:
        Modified segment with quadratic curve
    """
    if len(segment.points) < 2:
        return segment

    # Ensure idx1 < idx2
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1

    # Must be consecutive
    if abs(idx1 - idx2) != 1:
        return segment

    # Control point at min-x, max-y
    segment_bbox = segment.bounding_box()
    control_point = np.array([segment_bbox.xmin, segment_bbox.ymax])

    # Replace L with Q and insert control point
    new_points = segment.points.copy()
    new_commands = segment.commands.copy()

    if idx2 < len(new_commands) and new_commands[idx2] == "L":
        new_commands[idx2] = "Q"
        control_with_type = np.array([control_point[0], control_point[1], 2.0])
        new_points = np.insert(new_points, idx2, control_with_type, axis=0)

    return AvPath(new_points, new_commands)


def replace_line_with_quadratic_curve_bottom_right_simple(segment: AvPath, idx1: int, idx2: int) -> AvPath:
    """Replace the line between two specified points (bottom-right) with a quadratic curve.

    Simplified version - assumes points are consecutive and no Z-connection handling needed.

    Args:
        segment: The path segment to modify
        idx1, idx2: Indices of the two points to connect with a curve (must be consecutive)

    Returns:
        Modified segment with quadratic curve
    """
    if len(segment.points) < 2:
        return segment

    # Ensure idx1 < idx2
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1

    # Must be consecutive
    if abs(idx1 - idx2) != 1:
        return segment

    # Control point at max-x, min-y
    segment_bbox = segment.bounding_box()
    control_point = np.array([segment_bbox.xmax, segment_bbox.ymin])

    # Replace L or Z with Q and insert control point
    new_points = segment.points.copy()
    new_commands = segment.commands.copy()

    if idx2 < len(new_commands) and new_commands[idx2] in ("L", "Z"):
        is_closing = new_commands[idx2] == "Z"
        new_commands[idx2] = "Q"
        control_with_type = np.array([control_point[0], control_point[1], 2.0])
        new_points = np.insert(new_points, idx2, control_with_type, axis=0)
        # If it was a Z command, add L for the endpoint and Z to close the path
        if is_closing:
            # The Q command now uses ctrl and the original endpoint
            # Add L command for the closing line and Z for close
            new_commands = np.append(new_commands, "Z")
            # Z needs a point - use the starting point
            start_point = new_points[0].copy()
            new_points = np.vstack([new_points, start_point])

    return AvPath(new_points, new_commands)


def clean_chars_and_render_steps_on_page(
    svg_page: AvSvgPage,
    xpos: float,
    ypos: float,
    characters: str,
    avfont: AvFont,
    font_size: float,
    stroke_width: float,
) -> AvFont:
    """Clean the characters and render the steps on the page.

    Args:
        svg_page (AvSvgPage): The SVG page to render on.
        xpos (float): The x-coordinate of the starting position.
        ypos (float): The y-coordinate of the starting position.
        characters (str): The characters to clean and render.
        avfont (AvFont): The font to use.
        font_size (float): The font size to use.
        stroke_width (float): The stroke width to use.

    Returns:
        AvFont: The cleaned AvFont object used.
    """

    def print_glyph_path(
        glyph: AvGlyph, current_xpos: float, ypos: float, color: str, filled: bool, stroke_width: float
    ) -> float:
        letter = AvLetter.from_font_size_units_per_em(glyph, font_size, avfont.props.units_per_em, current_xpos, ypos)
        if filled:
            svg_path = svg_page.drawing.path(letter.svg_path_string(), fill=color, stroke="none")
        else:
            svg_path = svg_page.drawing.path(
                letter.svg_path_string(), fill="none", stroke=color, stroke_width=stroke_width
            )
        svg_page.add(svg_path)
        svg_path_debug = svg_page.drawing.path(
            letter.svg_path_string_debug_polyline(stroke_width),
            fill="none",
            stroke="blue",
            stroke_width=stroke_width * 0.2,
        )
        svg_page.add(svg_path_debug, True)
        return letter.width

    INFO_SIZE = font_size * 0.2  # pylint: disable=invalid-name

    clean_glyphs = {}
    current_xpos = xpos
    for char in characters:
        print(f"{char}", end="", flush=True)
        current_ypos = ypos
        original_glyph = avfont.get_glyph(char)
        glyph = original_glyph

        # Step 0: Original glyph
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S0-original", avfont, INFO_SIZE)
        print_glyph_path(glyph, current_xpos, current_ypos, "black", True, stroke_width)
        current_ypos += font_size

        # Step 1: Revise direction
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S1-revise-direction", avfont, INFO_SIZE)
        glyph = glyph.revise_direction()
        delta_xpos = print_glyph_path(glyph, current_xpos, current_ypos, "black", False, stroke_width)
        current_ypos += font_size

        # Step 2: Update glyph (custom modifications)
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S2-customize-glyph", avfont, INFO_SIZE)
        glyph = customize_glyph(glyph, avfont.props.dash_thickness)
        path = glyph.path
        delta_xpos = print_glyph_path(glyph, current_xpos, current_ypos, "black", False, stroke_width)
        current_ypos += font_size

        # Step 3: Polygonize
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S3-polygonize", avfont, INFO_SIZE)
        polygonized_path = path.polygonize(50)
        glyph = AvGlyph(character=original_glyph.character, width=original_glyph.width(), path=polygonized_path)
        print_glyph_path(glyph, current_xpos, current_ypos, "black", False, stroke_width)
        current_ypos += font_size

        # Step 4: Resolve intersections
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S4-resolve-intersections", avfont, INFO_SIZE)
        path = AvPathCleaner.resolve_polygonized_path_intersections(polygonized_path)
        glyph = AvGlyph(character=original_glyph.character, width=original_glyph.width(), path=path)
        print_glyph_path(glyph, current_xpos, current_ypos, "black", False, stroke_width)
        current_ypos += font_size

        # Step 5: Match paths
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S5-match-paths", avfont, INFO_SIZE)
        path = AvPathMatcher.match_paths(polygonized_path, path)
        glyph = AvGlyph(character=original_glyph.character, width=original_glyph.width(), path=path)
        print_glyph_path(glyph, current_xpos, current_ypos, "black", False, stroke_width)
        current_ypos += font_size

        # Step 6: Rebuild curve path
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S6-rebuild-curve-path", avfont, INFO_SIZE)
        path = AvPathCurveRebuilder.rebuild_curve_path(path)
        glyph = AvGlyph(character=original_glyph.character, width=original_glyph.width(), path=path)
        clean_glyphs[char] = glyph
        delta_xpos = print_glyph_path(glyph, current_xpos, current_ypos, "black", False, stroke_width)
        current_ypos += font_size

        # cleaning steps finished - now print overlays to check results

        # Step 7: Print overlay with stroke-border
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S7-overlay-border-check", avfont, INFO_SIZE)
        print_glyph_path(original_glyph, current_xpos, current_ypos, "red", False, stroke_width)
        print_glyph_path(glyph, current_xpos, current_ypos, "black", True, stroke_width)
        current_ypos += font_size

        # Step 8: Print overlay: original on top
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S8-overlay-original-on-top", avfont, INFO_SIZE)
        print_glyph_path(glyph, current_xpos, current_ypos, "red", True, stroke_width)
        print_glyph_path(original_glyph, current_xpos, current_ypos, "black", True, stroke_width)
        current_ypos += font_size

        # Step 9: Print overlay: cleaned on top
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S9-overlay-cleaned-on-top", avfont, INFO_SIZE)
        print_glyph_path(original_glyph, current_xpos, current_ypos, "red", True, stroke_width)
        print_glyph_path(glyph, current_xpos, current_ypos, "black", True, stroke_width)
        current_ypos += font_size

        # printing overlays done - now validate additionally
        glyph.validate()

        # after last step: move to next glyph
        current_xpos += delta_xpos
    print("")

    # Serialize to dict and deserialize back
    print("Creating cleaned glyphs factory... ", end="", flush=True)
    clean_glyphs_factory = AvGlyphPersistentFactory(clean_glyphs, avfont.props)
    print("done.")

    print("Serializing cleaned glyphs to dict... ", end="", flush=True)
    clean_glyphs_dict = clean_glyphs_factory.to_cache_dict()
    print("done.")

    # Deserialize from dict
    print("Deserializing cleaned glyphs from dict... ", end="", flush=True)
    clean_glyphs_factory = AvGlyphPersistentFactory.from_cache_dict(clean_glyphs_dict)
    clean_font = AvFont(clean_glyphs_factory)
    print("done.")

    # Step 10: Print characters again using loaded glyphs
    print("Processing deserialized characters...")
    current_xpos = xpos
    for char in characters:
        print(f"{char}", end="", flush=True)
        glyph = clean_font.get_glyph(char)
        original_glyph = avfont.get_glyph(char)
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S10-(de)serialized-font", avfont, INFO_SIZE)
        # Print original glyph filled and cleaned glyph as stroke
        delta_xpos = print_glyph_path(original_glyph, current_xpos, current_ypos, "black", True, stroke_width)
        delta_xpos = print_glyph_path(glyph, current_xpos, current_ypos, "red", False, stroke_width)
        current_xpos += delta_xpos
    print("")

    # Return Font
    return clean_font


def process_font_to_svg(avfont: AvFont, svg_filename: str, characters: str) -> AvFont:
    """Process font glyphs and save to SVG file.

    Args:
        avfont: The AvFont object to process
        svg_filename: Path where the SVG file will be saved
        characters: String of characters to process

    Returns:
        The cleaned font after processing
    """
    # Setup the page with A4 dimensions
    viewbox_width = 180  # viewbox width in mm
    viewbox_height = 120  # viewbox height in mm
    vb_scale = 1.0 / viewbox_width  # scale viewbox so that x-coordinates are between 0 and 1
    font_size = vb_scale * 2.7  # in mm
    stroke_width = 0.1 * avfont.props.dash_thickness * font_size / avfont.props.units_per_em

    # Create the SVG page using the factory method
    svg_page = AvSvgPage.create_page_a4(viewbox_width, viewbox_height, vb_scale)

    # Draw the viewbox border
    draw_viewbox_border(svg_page, vb_scale, viewbox_width, viewbox_height)

    clean_font = clean_chars_and_render_steps_on_page(svg_page, 0.05, 0.01, characters, avfont, font_size, stroke_width)

    # Save the SVG
    print(f"Saving to {svg_filename} ...")
    svg_page.save_as(svg_filename, include_debug_layer=True, pretty=True, compressed=True)
    print(f"Saved to  {svg_filename}")

    return clean_font


###############################################################################
# Main
###############################################################################


def main():
    """Main function to demonstrate glyph details."""

    font_in_fn = "fonts/RobotoFlex-VariableFont_GRAD,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"
    font_out_name = "RobotoFlex-VariableFont_AA_"
    font_num_wghts = 2
    font_wghts_min = 100
    font_wghts_max = 1000
    font_out_fn_base = f"fonts/cache/{font_out_name}"  # XX_wghtYYYY.json.zip
    svg_out_fn_base = f"data/output/fonts/cache/{font_out_name}"  # XX_wghtYYYY.svgz

    # -------------------------------------------------------------------------
    # Characters to display
    characters = ""
    characters += "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    characters += "abcdefghijklmnopqrstuvwxyz "
    characters += "0123456789 "
    characters += ',.;:+-*#_<> !"§$%&/()=?{}[] '
    # NON-ASCII EXCEPTION: German characters and special symbols for comprehensive font testing
    characters += "ÄÖÜ äöü ß€µ@²³~^°\\ '`"
    # characters += "\u00ff \u0066 \u0069 \u006c"

    # Print some individual character details
    detail_chars = "AKXf"  # intersection
    detail_chars += "e&46"  # self-intersection
    detail_chars += "QR§$"  # intersection and hole
    # NON-ASCII EXCEPTION: German characters for font testing
    detail_chars += "Ä"  # intersection and hole, several polygons
    # NON-ASCII EXCEPTION: German characters for font testing
    detail_chars += "BOÖä"  # holes
    detail_chars += 'i:%"'  # several polygons
    detail_chars += "€#"  # several intersections

    characters = detail_chars + "-"

    # -------------------------------------------------------------------------

    # Create fonts with different weights
    wght_range = range(font_wghts_min, font_wghts_max + 1, (font_wghts_max - font_wghts_min) // (font_num_wghts - 1))
    for idx, wght in enumerate(wght_range, 1):
        print(f"Processing weight {wght} ...")

        font_out_fn = f"{font_out_fn_base}{idx:02d}_wght{wght:04d}.json.zip"
        svg_out_fn = f"{svg_out_fn_base}{idx:02d}_wght{wght:04d}.svgz"

        avfont = setup_avfont(font_in_fn, {"wght": wght})

        # Save SVG:
        clean_font = process_font_to_svg(avfont, svg_out_fn, characters)

        # Save font:
        print(f"Saving to {font_out_fn} ...")
        clean_font.glyph_factory.save_to_file(font_out_fn)
        print(f"Saved to  {font_out_fn}")

        print("-----------------------------------------------------------------------")


if __name__ == "__main__":
    main()
