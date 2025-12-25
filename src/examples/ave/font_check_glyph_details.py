"""Single font glyph details SVG page example."""

from fontTools.ttLib import TTFont

from ave.font import AvFont, AvFontProperties
from ave.glyph import (
    AvGlyph,
    AvGlyphCachedFactory,
    AvGlyphFromTTFontFactory,
    AvGlyphPolygonizeFactory,
    AvLetter,
)
from ave.page import AvSvgPage
from ave.path_processing import AvPathCleaner, AvPathCurveRebuilder, AvPathMatcher


def setup_avfont(ttfont_filename: str):
    """Setup an AvFont object from a given TrueType font file."""
    ttfont = TTFont(ttfont_filename)

    # polygonize_steps=0 => no polygonization
    polygonize_steps = 0
    glyph_factory_ttfont = AvGlyphFromTTFontFactory(ttfont)
    glyph_factory_polygonized = AvGlyphPolygonizeFactory(glyph_factory_ttfont, polygonize_steps)

    avfont = AvFont(glyph_factory_polygonized, AvFontProperties.from_ttfont(ttfont))
    return avfont


def print_text_on_page(
    svg_page: AvSvgPage, xpos: float, ypos: float, text: str, avfont: AvFont, font_size: float
) -> None:
    """Print text on the given svg_page at the given position with the given font size and font."""
    current_xpos = xpos
    for character in text:
        glyph = avfont.get_glyph(character)
        letter = AvLetter.from_font_size_units_per_em(glyph, font_size, avfont.props.units_per_em, current_xpos, ypos)
        svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
        svg_page.add(svg_path)
        current_xpos += letter.width


def create_cleaned_font(characters: str, original_font: AvFont) -> AvFont:
    """Create a cleaned AvFont for the specified characters using resolve_path_intersections.

    Args:
        characters: String of characters to include in the cleaned font
        original_font: Original AvFont to get glyphs from

    Returns:
        AvFont: New font with cleaned glyphs
    """
    # Create dictionary for cleaned glyphs
    cleaned_glyphs = {}

    for char in characters:
        try:
            # Get original glyph
            original_glyph = original_font.get_glyph(char)

            # Clean the glyph
            cleaned_path = AvPathCleaner.resolve_polygonized_path_intersections(original_glyph.path.polygonized_path())
            cleaned_glyph = AvGlyph(character=original_glyph.character, width=original_glyph.width(), path=cleaned_path)
            cleaned_glyphs[char] = cleaned_glyph

        except (ValueError, TypeError, AttributeError) as e:
            print(f"Failed to clean glyph for '{char}': {e}")
            cleaned_glyphs[char] = original_glyph

    # Create cleaned font using AvGlyphCachedFactory
    cleaned_glyph_factory = AvGlyphCachedFactory(glyphs=cleaned_glyphs, source_factory=None)
    cleaned_font = AvFont(cleaned_glyph_factory, original_font.props)

    return cleaned_font


def create_cleaned_font2(characters: str, original_font: AvFont) -> AvFont:
    """Create a cleaned AvFont for the specified characters using resolve_path_intersections.

    Args:
        characters: String of characters to include in the cleaned font
        original_font: Original AvFont to get glyphs from

    Returns:
        AvFont: New font with cleaned glyphs
    """
    # Create dictionary for cleaned glyphs
    cleaned_glyphs = {}

    for char in characters:
        try:
            # Get original glyph
            original_glyph = original_font.get_glyph(char)

            # Clean the glyph
            cleaned_path = AvPathCleaner.resolve_polygonized_path_intersections(original_glyph.path)
            cleaned_glyph = AvGlyph(character=original_glyph.character, width=original_glyph.width(), path=cleaned_path)
            cleaned_glyphs[char] = cleaned_glyph

        except (ValueError, TypeError, AttributeError) as e:
            print(f"Failed to clean glyph for '{char}': {e}")
            cleaned_glyphs[char] = original_glyph

    # Create cleaned font using AvGlyphCachedFactory
    cleaned_glyph_factory = AvGlyphCachedFactory(glyphs=cleaned_glyphs, source_factory=None)
    cleaned_font = AvFont(cleaned_glyph_factory, original_font.props)

    return cleaned_font


def create_cleaned_font3(characters: str, original_font: AvFont) -> AvFont:
    """Create a cleaned AvFont for the specified characters using resolve_path_intersections.

    Args:
        characters: String of characters to include in the cleaned font
        original_font: Original AvFont to get glyphs from

    Returns:
        AvFont: New font with cleaned glyphs
    """
    # Create dictionary for cleaned glyphs
    cleaned_glyphs = {}

    for char in characters:
        try:
            original_glyph = original_font.get_glyph(char)

            original_path = original_glyph.path  # step 0
            print(f"Original path for '{char}':")
            print(original_path)

            revised_glyph = original_glyph.revise_direction()  # step 1
            revised_path = revised_glyph.path
            print(f"Revised path for '{char}':")
            print(revised_path)

            polygonized_path = revised_path.polygonize(2)  # step 2
            print("After polygonization:")
            print(polygonized_path)

            resolved_intersects_path = AvPathCleaner.resolve_polygonized_path_intersections(polygonized_path)  # step 3
            print("After resolve_polygonized_path_intersections:")
            print(resolved_intersects_path)

            matched_path = AvPathMatcher.match_paths(polygonized_path, resolved_intersects_path)  # step 4
            print("After matching:")
            print(matched_path)

            rebuild_path = AvPathCurveRebuilder.rebuild_curve_path(matched_path)  # step 5
            print("After rebuild:")
            print(rebuild_path)

            cleaned_glyph = AvGlyph(character=original_glyph.character, width=original_glyph.width(), path=rebuild_path)
            cleaned_glyphs[char] = cleaned_glyph
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Failed to clean glyph for '{char}': {e}")
            cleaned_glyphs[char] = original_glyph

    # Create cleaned font using AvGlyphCachedFactory
    cleaned_glyph_factory = AvGlyphCachedFactory(glyphs=cleaned_glyphs, source_factory=None)
    cleaned_font = AvFont(cleaned_glyph_factory, original_font.props)

    return cleaned_font


def main():
    """Main function to demonstrate glyph details."""
    output_filename = "data/output/example/svg/ave/example_glyph_details.svg"

    # Font setup
    font_filename = "fonts/RobotoFlex-VariableFont_GRAD,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"
    avfont = setup_avfont(font_filename)

    # Setup the page with A4 dimensions
    viewbox_width = 180  # viewbox width in mm
    viewbox_height = 120  # viewbox height in mm
    vb_scale = 1.0 / viewbox_width  # scale viewbox so that x-coordinates are between 0 and 1
    font_size = vb_scale * 2.7  # in mm
    stroke_width = 0.1 * avfont.props.dash_thickness * font_size / avfont.props.units_per_em

    # Create the SVG page using the factory method
    svg_page = AvSvgPage.create_page_a4(viewbox_width, viewbox_height, vb_scale)

    # Draw the viewbox border
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

    # Characters to display
    characters = ""
    characters += "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    characters += "abcdefghijklmnopqrstuvwxyz "
    characters += "0123456789 "
    characters += ',.;:+-*#_<> !"§$%&/()=?{}[] '
    # NON-ASCII EXCEPTION: German characters and special symbols for comprehensive font testing
    characters += "ÄÖÜ äöü ß€µ@²³~^°\\ "

    # Print characters on the page
    print_text_on_page(svg_page, 0.05, 0.01, characters, avfont, font_size)

    # Print some individual character details
    detail_chars = "AKXf"  # intersection
    detail_chars += "€#"  # several intersections
    detail_chars += "e&46"  # self-intersection
    detail_chars += "QR§$"  # intersection and hole
    # NON-ASCII EXCEPTION: German characters for font testing
    detail_chars += "Ä"  # intersection and hole, several polygons
    # NON-ASCII EXCEPTION: German characters for font testing
    detail_chars += "BOÖä"  # holes
    detail_chars += 'i:%"'  # several polygons

    xpos = 0.05
    ypos = 0.15
    for character in detail_chars:
        glyph = avfont.get_glyph(character)
        letter = AvLetter.from_font_size_units_per_em(glyph, font_size, avfont.props.units_per_em, xpos, ypos)
        svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
        svg_page.add(svg_path)
        ypos += 0.02

    # Print characters again without fill at different position
    xpos = 0.075
    ypos = 0.15
    for character in detail_chars:
        glyph = avfont.get_glyph(character)
        letter = AvLetter.from_font_size_units_per_em(glyph, font_size, avfont.props.units_per_em, xpos, ypos)
        svg_path = svg_page.drawing.path(
            letter.svg_path_string(), fill="none", stroke="black", stroke_width=stroke_width
        )
        svg_page.add(svg_path)
        ypos += 0.02

    # Create cleaned font
    cleaned_font = create_cleaned_font(detail_chars, avfont)

    # Print cleaned characters on the page
    xpos = 0.1
    ypos = 0.15
    for character in detail_chars:
        glyph = cleaned_font.get_glyph(character)
        letter = AvLetter.from_font_size_units_per_em(glyph, font_size, cleaned_font.props.units_per_em, xpos, ypos)
        svg_path = svg_page.drawing.path(
            letter.svg_path_string(), fill="none", stroke="black", stroke_width=stroke_width
        )
        svg_page.add(svg_path)
        ypos += 0.02

    # Create polygonized font with 4 steps for detailed character analysis
    polygonized_factory = AvGlyphPolygonizeFactory(avfont.glyph_factory, polygonize_steps=2)
    polygonized_font = AvFont(polygonized_factory, avfont.props)

    # Create cleaned font based on polygonized font
    cleaned_font = create_cleaned_font(detail_chars, polygonized_font)
    xpos = 0.125
    ypos = 0.15
    for character in detail_chars:
        glyph = cleaned_font.get_glyph(character)
        letter = AvLetter.from_font_size_units_per_em(glyph, font_size, cleaned_font.props.units_per_em, xpos, ypos)
        svg_path = svg_page.drawing.path(
            letter.svg_path_string(), fill="none", stroke="black", stroke_width=stroke_width
        )
        svg_page.add(svg_path)
        ypos += 0.02

    cleaned_font2 = create_cleaned_font2(detail_chars, polygonized_font)
    xpos = 0.15
    ypos = 0.15
    for character in detail_chars:
        glyph = cleaned_font2.get_glyph(character)
        letter = AvLetter.from_font_size_units_per_em(glyph, font_size, cleaned_font2.props.units_per_em, xpos, ypos)
        svg_path = svg_page.drawing.path(
            letter.svg_path_string(), fill="none", stroke="black", stroke_width=stroke_width
        )
        svg_page.add(svg_path)
        ypos += 0.02

    ###########################################################################
    # clean font 3
    ###########################################################################
    characters = detail_chars
    ttfont = TTFont(font_filename)
    glyph_factory = AvGlyphFromTTFontFactory(ttfont)
    avfont = AvFont(glyph_factory, AvFontProperties.from_ttfont(ttfont))
    cleaned_font3 = create_cleaned_font3(characters, avfont)
    xpos = 0.175
    ypos = 0.15
    for character in characters:
        glyph = cleaned_font3.get_glyph(character)
        letter = AvLetter.from_font_size_units_per_em(glyph, font_size, cleaned_font3.props.units_per_em, xpos, ypos)
        svg_path = svg_page.drawing.path(
            letter.svg_path_string(), fill="none", stroke="brown", stroke_width=stroke_width
        )
        svg_page.add(svg_path)
        ypos += 0.02

    # Save the SVG
    svg_page.save_as(output_filename, include_debug_layer=True, pretty=True)
    print(f"Saved to {output_filename}")
    # print(avfont.get_info_string())


if __name__ == "__main__":
    main()
