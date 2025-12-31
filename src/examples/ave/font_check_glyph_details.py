"""Single font glyph details SVG page example."""

from fontTools.ttLib import TTFont

from ave.font import AvFont
from ave.glyph import (
    AvGlyph,
    AvGlyphCachedSourceFactory,
    AvGlyphFromTTFontFactory,
    AvGlyphPersistentFactory,
    AvGlyphPolygonizeFactory,
    AvLetter,
)
from ave.page import AvSvgPage
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


def setup_avfont(ttfont_filename: str):
    """Setup an AvFont object from a given TrueType font file."""
    ttfont = TTFont(ttfont_filename)

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
        path = glyph.path
        delta_xpos = print_glyph_path(glyph, current_xpos, current_ypos, "black", False, stroke_width)
        current_ypos += font_size

        # Step 2: Polygonize
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S2-polygonize", avfont, INFO_SIZE)
        polygonized_path = path.polygonize(50)
        glyph = AvGlyph(character=original_glyph.character, width=original_glyph.width(), path=polygonized_path)
        print_glyph_path(glyph, current_xpos, current_ypos, "black", False, stroke_width)
        current_ypos += font_size

        # Step 3: Resolve intersections
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S3-resolve-intersections", avfont, INFO_SIZE)
        path = AvPathCleaner.resolve_polygonized_path_intersections(polygonized_path)
        glyph = AvGlyph(character=original_glyph.character, width=original_glyph.width(), path=path)
        print_glyph_path(glyph, current_xpos, current_ypos, "black", False, stroke_width)
        current_ypos += font_size

        # Step 4: Match paths
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S4-match-paths", avfont, INFO_SIZE)
        path = AvPathMatcher.match_paths(polygonized_path, path)
        glyph = AvGlyph(character=original_glyph.character, width=original_glyph.width(), path=path)
        print_glyph_path(glyph, current_xpos, current_ypos, "black", False, stroke_width)
        current_ypos += font_size

        # Step 5: Rebuild curve path
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S5-rebuild-curve-path", avfont, INFO_SIZE)
        path = AvPathCurveRebuilder.rebuild_curve_path(path)
        glyph = AvGlyph(character=original_glyph.character, width=original_glyph.width(), path=path)
        clean_glyphs[char] = glyph
        delta_xpos = print_glyph_path(glyph, current_xpos, current_ypos, "black", False, stroke_width)
        current_ypos += font_size

        # cleaning steps finished - now print overlays to check results

        # Step 6: Print overlay with stroke-border
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S6-overlay-border-check", avfont, INFO_SIZE)
        print_glyph_path(original_glyph, current_xpos, current_ypos, "red", False, stroke_width)
        print_glyph_path(glyph, current_xpos, current_ypos, "black", True, stroke_width)
        current_ypos += font_size

        # Step 7: Print overlay: original on top
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S7-overlay-original-on-top", avfont, INFO_SIZE)
        print_glyph_path(glyph, current_xpos, current_ypos, "red", True, stroke_width)
        print_glyph_path(original_glyph, current_xpos, current_ypos, "black", True, stroke_width)
        current_ypos += font_size

        # Step 8: Print overlay: cleaned on top
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S8-overlay-cleaned-on-top", avfont, INFO_SIZE)
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

    # print characters again using loaded glyphs
    print("Processing deserialized characters...")
    current_xpos = xpos
    for char in characters:
        print(f"{char}", end="", flush=True)
        glyph = clean_font.get_glyph(char)
        original_glyph = avfont.get_glyph(char)
        if char == characters[0]:
            print_text(svg_page, 0.005, current_ypos, "S9-(de)serialized-font", avfont, INFO_SIZE)
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
    svg_filename = "data/output/example/svg/ave/example_glyph_details.svgz"

    # Font setup
    font_filename = "fonts/RobotoFlex-VariableFont_GRAD,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"
    avfont = setup_avfont(font_filename)

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

    # characters = detail_chars

    # Process font and save to SVG
    clean_font = process_font_to_svg(avfont, svg_filename, characters)

    # Print font info
    print(clean_font.get_info_string(False))


if __name__ == "__main__":
    main()
