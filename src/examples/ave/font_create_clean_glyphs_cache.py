"""Single font glyph details SVG page example."""

from typing import Dict, Optional

import numpy as np
from fontTools.ttLib import TTFont

from ave.font import AvFont
from ave.fonttools import FontHelper
from ave.geom import AvBox
from ave.glyph import (
    AvFontProperties,
    AvGlyph,
    AvGlyphCachedSourceFactory,
    AvGlyphFromTTFontFactory,
    AvGlyphPersistentFactory,
    AvGlyphPolygonizeFactory,
)
from ave.letter import AvLetter
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
        letter = AvLetter(glyph, font_size / avfont.props.units_per_em, current_xpos, ypos)
        svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
        svg_page.add(svg_path)
        current_xpos += letter.width()


def create_new_q_tail(bbox: AvBox, dash_thickness: float) -> AvPath:
    """Create a new Q-tail segment with quadratic curves.

    Args:
        bbox: Bounding box of the shifted Q-tail segment
        dash_thickness: Thickness of the dash stroke

    Returns:
        New Q-tail segment with quadratic curves at top-left and bottom-right
    """
    # Calculate diagonal length and offsets for the beam thickness
    width = bbox.xmax - bbox.xmin
    height = bbox.ymax - bbox.ymin
    diag_length = np.sqrt(width**2 + height**2)
    half_t = dash_thickness / 2

    # Offset along each edge from corner (derived from perpendicular thickness)
    offset_x = half_t * diag_length / height  # offset along horizontal edges
    offset_y = half_t * diag_length / width  # offset along vertical edges

    # 6 points in CCW order with control points at corners
    points = np.array(
        [
            [bbox.xmin + offset_x, bbox.ymax, 0.0],  # 1: top edge
            [bbox.xmin, bbox.ymax, 2.0],  # 2: control point (top-left corner)
            [bbox.xmin, bbox.ymax - offset_y, 0.0],  # 3: left edge
            [bbox.xmax - offset_x, bbox.ymin, 0.0],  # 4: bottom edge
            [bbox.xmax, bbox.ymin, 2.0],  # 5: control point (bottom-right corner)
            [bbox.xmax, bbox.ymin + offset_y, 0.0],  # 6: right edge
        ]
    )

    return AvPath(points, ["M", "Q", "L", "Q", "Z"])


def customize_glyph(glyph: AvGlyph, props: AvFontProperties) -> AvGlyph:
    """Update a glyph after the revise step.

    Algorithm Steps (for 'Q' glyph):
    1. Split the glyph path into individual segments using split_into_single_paths()
        - This separates the outer circle, inner hole, and tail into distinct paths

    2. Filter for CCW segments (exterior polygons):
        - Use segment.area > 0 and segment.is_ccw to identify exterior contours
        - Exclude interior holes from processing
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

    6. Apply transformation and create new tail:
        - Shift all points in the tail segment by (shift_x, shift_y)
        - Use shifted bounding box to create a new stylized tail via
            create_new_q_tail() with the specified dash_thickness

    7. Reconstruct the glyph:
        - Replace original tail with the new stylized tail
        - Preserve original segment order
        - Join all segments back into a single path

    Args:
        glyph: The glyph to potentially update.
        props: Font properties containing dash_thickness and font information.

    Returns:
        A new glyph if modifications were made, or the same glyph if no changes.
    """

    if props.full_name == "Roboto Flex Regular":
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
            tail_segment: AvPath = max(
                ccw_segments, key=lambda s: s.bounding_box().centroid[0] - s.bounding_box().centroid[1]
            )

            # 4. Get the bounding box of the tail segment
            tail_bbox = tail_segment.bounding_box()

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
                shift_x = combined_bbox.xmax - tail_bbox.xmax  # Align right edges
                shift_y = combined_bbox.ymin - tail_bbox.ymin  # Align bottom edges

                # Apply the shift to all points in the tail segment
                shifted_points = tail_segment.points.copy()
                shifted_points[:, 0] += shift_x
                shifted_points[:, 1] += shift_y

                # Create a temporary segment to get the shifted bounding box
                shifted_segment = AvPath(shifted_points[:, :2], tail_segment.commands)

                # 8. Create a new stylized tail using the shifted bounding box
                shifted_segment = create_new_q_tail(shifted_segment.bounding_box(), props.dash_thickness)

                # 9. Rebuild the path with the new tail, preserving original order
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

    return glyph  # no change, return original glyph


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
        letter = AvLetter(glyph, font_size / avfont.props.units_per_em, current_xpos, ypos)
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

        # Add bounding box in yellow
        bbox = letter.bounding_box()
        svg_bbox = svg_page.drawing.path(
            f"M {bbox.xmin:g} {bbox.ymin:g} L {bbox.xmax:g} {bbox.ymin:g} L {bbox.xmax:g} {bbox.ymax:g} L {bbox.xmin:g} {bbox.ymax:g} Z",
            fill="none",
            stroke="yellow",
            stroke_width=stroke_width * 0.1,
        )
        svg_page.add(svg_bbox, True)

        # Add letter box in green
        lbox = letter.letter_box()
        svg_lbox = svg_page.drawing.path(
            f"M {lbox.xmin:g} {lbox.ymin:g} L {lbox.xmax:g} {lbox.ymin:g} L {lbox.xmax:g} {lbox.ymax:g} L {lbox.xmin:g} {lbox.ymax:g} Z",
            fill="none",
            stroke="green",
            stroke_width=stroke_width * 0.1,
        )
        svg_page.add(svg_lbox, True)

        return letter.width()

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
        glyph = customize_glyph(glyph, avfont.props)
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
    ypos = current_ypos
    for char in characters:
        print(f"{char}", end="", flush=True)
        current_ypos = ypos
        glyph = clean_font.get_glyph(char)
        original_glyph = avfont.get_glyph(char)
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S10-(de)serialized-font", avfont, INFO_SIZE)
        # Print original glyph filled and cleaned glyph as stroke
        delta_xpos = print_glyph_path(original_glyph, current_xpos, current_ypos, "red", True, stroke_width)
        delta_xpos = print_glyph_path(glyph, current_xpos, current_ypos, "black", True, stroke_width)
        current_ypos += font_size

        # Step 11: Print clean end result
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S11-end-result", avfont, INFO_SIZE)
        print_glyph_path(glyph, current_xpos, current_ypos, "black", True, stroke_width)
        current_ypos += font_size

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
    font_num_wghts = 3
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

    # characters = detail_chars + "-"

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

    print(avfont.props.info_string())


if __name__ == "__main__":
    main()


# Support point patterns: numbered points with fractional spacing
# Each line shows N points creating N-1 intervals, where fraction × (N-1) = 1 total width
#   1               2               3    1/2  --> font_num_wghts=3
#   1       2       3       4       5    1/4  --> font_num_wghts=5
#   1     2   3     4     5   6     7    1/6  --> font_num_wghts=7
#   1   2   3   4   5   6   7   8   9    1/8  --> font_num_wghts=9
#   1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7    1/16 --> font_num_wghts=17
