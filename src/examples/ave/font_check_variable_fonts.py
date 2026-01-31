"""Variable font handling utilities and Roboto Flex font analysis example."""

from typing import Dict, List, Optional

from fontTools.ttLib import TTFont

from ave.font import AvFont
from ave.fonttools import FontHelper
from ave.glyph import (
    AvGlyphCachedSourceFactory,
    AvGlyphFromTTFontFactory,
    AvGlyphPolygonizeFactory,
)
from ave.letter import AvSingleGlyphLetter
from ave.page import AvSvgPage

CHARACTERS = ""
CHARACTERS += "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
CHARACTERS += "abcdefghijklmnopqrstuvwxyz "
CHARACTERS += "0123456789 "
CHARACTERS += ',.;:+-*#_<> !"§$%&/()=?{}[] '
# NON-ASCII EXCEPTION: German characters and special symbols for comprehensive font testing
CHARACTERS += "ÄÖÜ äöü ß€µ@²³~^°\\ "


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
    glpyh_factory_polygonized = AvGlyphPolygonizeFactory(glyph_factory_ttfont, polygonize_steps)
    glyph_factory_cached = AvGlyphCachedSourceFactory(glpyh_factory_polygonized)
    avfont = AvFont(glyph_factory_cached)

    return avfont


def print_text_on_page(
    svg_page: AvSvgPage, xpos: float, ypos: float, text: str, avfont: AvFont, font_size: float
) -> None:
    """Print text on the given svg_page at the given position with the given font size and font."""
    current_xpos = xpos
    for character in text:
        glyph = avfont.get_glyph(character)
        letter = AvSingleGlyphLetter(glyph, font_size / avfont.props.units_per_em, current_xpos, ypos)
        svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
        svg_page.add(svg_path)
        current_xpos += letter.advance_width


def print_font_example_page(output_filename: str, avfonts: List[AvFont]) -> None:
    # create a page with A4 dimensions
    """
    Create a page with a given AvFont object and print all characters defined in CHARACTERS on the page.

    Args:
        avfont (AvFontNIM): The AvFont object to use for printing the characters.
        output_filename (str): The filename of the SVG file to save.
    """
    vb_width_mm = 180  # viewbox width in mm
    vb_height_mm = 120  # viewbox height in mm
    vb_scale = 1.0 / vb_width_mm  # scale viewbox so that x-coordinates are between 0 and 1
    font_size = vb_scale * 2.7  # in mm

    svg_page = AvSvgPage.create_page_a4(vb_width_mm, vb_height_mm, vb_scale)

    # define a path that describes the outline of the viewbox
    svg_page.add(
        svg_page.drawing.path(
            d=(
                f"M 0 0 "
                f"L {vb_scale * vb_width_mm} 0 "  # = (1.0, 0.0)
                f"L {vb_scale * vb_width_mm} {vb_scale * vb_height_mm} "
                f"L 0 {vb_scale * vb_height_mm} "
                f"Z"
            ),
            stroke="blue",
            stroke_width=0.1 * vb_scale,
            fill="none",
        ),
        False,
    )

    xpos = 0.0
    ypos = 0.01
    for font in avfonts:
        print_text_on_page(svg_page, xpos, ypos, CHARACTERS, font, font_size)
        ypos += 0.05

    xpos = -0.04
    for ypos in [round(i * 0.1, 1) for i in range(7)]:
        print_text_on_page(svg_page, xpos, ypos, str(ypos), avfonts[0], font_size)

    # Save the SVG file
    print(f"save file {output_filename} ...")
    svg_page.save_as(output_filename, include_debug_layer=True, pretty=True, indent=2, compressed=True)
    print("save done.")


def main():
    """Main"""

    font_filename = "fonts/RobotoFlex-VariableFont_GRAD,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"
    output_filename = "data/output/example/svg/ave/example_font_RobotoFlex_variable.svgz"
    print_font_example_page(
        output_filename,
        [
            setup_avfont(font_filename, {"wght": 100.0}),
            setup_avfont(font_filename, {"wght": 400.0}),
            setup_avfont(font_filename, {"wght": 1000.0}),
        ],
    )
    font = setup_avfont(font_filename, {"wght": 400.0})
    print(font.get_info_string())

    # font_filename = "fonts/RobotoMono-VariableFont_wght.ttf"
    # output_filename = "data/output/example/svg/ave/example_font_RobotoMono_variable.svgz"
    # print_font_example_page(
    #     output_filename,
    #     [
    #         setup_avfont(font_filename, {"wght": 100.0}),
    #         setup_avfont(font_filename, {"wght": 400.0}),
    #         setup_avfont(font_filename, {"wght": 700.0}),
    #     ],
    # )

    # font_filename = "fonts/Petrona-VariableFont_wght.ttf"
    # output_filename = "data/output/example/svg/ave/example_font_Petrona_variable.svgz"
    # print_font_example_page(
    #     output_filename,
    #     [
    #         setup_avfont(font_filename, {"wght": 100.0}),
    #         setup_avfont(font_filename, {"wght": 400.0}),
    #         setup_avfont(font_filename, {"wght": 900.0}),
    #     ],
    # )

    # font_filename = "fonts/Caveat-VariableFont_wght.ttf"
    # output_filename = "data/output/example/svg/ave/example_font_Caveat_variable.svgz"
    # print_font_example_page(
    #     output_filename,
    #     [
    #         setup_avfont(font_filename, {"wght": 400.0}),
    #         setup_avfont(font_filename, {"wght": 400.0}),
    #         setup_avfont(font_filename, {"wght": 700.0}),
    #     ],
    # )


if __name__ == "__main__":
    main()


#   1               2               3
#   1       2       3       4       5
#   1     2   3     4     5   6     7
#   1   2   3   4   5   6   7   8   9
#   1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7
