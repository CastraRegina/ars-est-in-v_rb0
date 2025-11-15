"""Module to check how to handle the font Petrona"""

# from typing import Dict

from fontTools.ttLib import TTFont

from ave.fonttools import FontHelper
from ave.glyph import AvFont, AvGlyph, AvGlyphFactory, AvLetter
from ave.page import AvSvgPage


def main():
    """Main"""
    font_filename = "fonts/Petrona-VariableFont_wght.ttf"
    output_filename = "data/output/example/svg/ave/example_font_Petrona.svgz"

    # create a page with A4 dimensions
    vb_width_mm = 150  # 105  # viewbox width in mm
    vb_height_mm = 150  # 148.5  # viewbox height in mm
    vb_scale = 1.0 / vb_width_mm  # scale viewbox so that x-coordinates are between 0 and 1
    font_size = vb_scale * 3  # in mm

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
            stroke="black",
            stroke_width=0.1 * vb_scale,
            fill="none",
        ),
        True,
    )

    ttfont_w100 = FontHelper.instantiate_ttfont(TTFont(font_filename), {"wght": 100})
    ttfont_w900 = FontHelper.instantiate_ttfont(TTFont(font_filename), {"wght": 900})
    avfont_w100 = AvFont(ttfont_w100, AvGlyphFactory())
    avfont_w900 = AvFont(ttfont_w900, AvGlyphFactory())
    font_scale_w100 = font_size / avfont_w100.units_per_em
    font_scale_w900 = font_size / avfont_w900.units_per_em

    glyph: AvGlyph = avfont_w100.fetch_glyph(" ")
    letter = AvLetter(glyph, 0.0, 0.9, font_scale_w100)
    svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
    svg_page.add(svg_path)

    text = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        + "abcdefghijklmnopqrstuvwxyz "
        + "ÄÖÜ äöü ß€µ@²³~^°\\ 1234567890 "
        + ',.;:+-*#_<> !"§$%&/()=?{}[]'
    )

    x_pos = 0
    y_pos = 0.8
    for character in text:
        glyph: AvGlyph = avfont_w100.fetch_glyph(character)
        letter = AvLetter(glyph, x_pos, y_pos, font_scale_w100)
        svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
        svg_page.add(svg_path)
        x_pos += letter.width

    x_pos = 0
    y_pos = 0.75
    for character in text:
        glyph: AvGlyph = avfont_w900.fetch_glyph(character)
        letter = AvLetter(glyph, x_pos, y_pos, font_scale_w900)
        svg_path = svg_page.drawing.path(letter.svg_path_string(), fill="black", stroke="none")
        svg_page.add(svg_path)
        x_pos += letter.width

    # Save the SVG file
    print(f"save file {output_filename} ...")
    svg_page.save_as(output_filename, include_debug_layer=True, pretty=True, indent=2, compressed=True)
    print("save done.")

    print(avfont_w100.info_string())


if __name__ == "__main__":
    main()


#   1               2               3
#   1       2       3       4       5
#   1     2   3     4     5   6     7
#   1   2   3   4   5   6   7   8   9
#   1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7
