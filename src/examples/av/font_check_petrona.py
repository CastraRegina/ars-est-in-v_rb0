"""Module to check how to handle the font Petrona"""

from typing import Dict

from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

from av.glyph import AvFont, AvGlyph
from av.page import AvPageSvg


def main():
    """Main"""
    output_filename = "data/output/example/svg/example_font_Petrona.svg"

    canvas_width = 210  # DIN A4 page width in mm
    canvas_height = 297  # DIN A4 page height in mm

    rect_vb_width = 150  # rectangle viewbox width in mm
    rect_vb_height = 150  # rectangle viewbox height in mm

    vb_ratio = 1 / rect_vb_width  # multiply each dimension with this ratio

    # Center the rectangle horizontally and vertically on the page
    vb_w = vb_ratio * canvas_width
    vb_h = vb_ratio * canvas_height
    vb_x = -vb_ratio * (canvas_width - rect_vb_width) / 2
    vb_y = -vb_ratio * (canvas_height - rect_vb_height) / 2

    # Set up the SVG canvas:
    #   Define viewBox so that "1" is the width of the rectangle
    #   Multiply a dimension with "vb_ratio" to get the size regarding viewBox
    svg_page_output = AvPageSvg(canvas_width, canvas_height, vb_x, vb_y, vb_w, vb_h)

    # Draw the rectangle
    svg_page_output.add(
        svg_page_output.drawing.rect(
            insert=(0, 0),
            size=(vb_ratio * rect_vb_width, vb_ratio * rect_vb_height),  # = (1.0, xxxx)
            stroke="black",
            stroke_width=0.1 * vb_ratio,
            fill="none",
        )
    )

    def instantiate_font(ttfont: TTFont, values: Dict[str, float]) -> AvFont:
        # values {"wght": 700, "wdth": 25, "GRAD": 100}
        axes_values = AvFont.default_axes_values(ttfont)
        axes_values.update(values)
        ttfont = instancer.instantiateVariableFont(ttfont, axes_values)
        return AvFont(ttfont)

    # prepare variables for fonts
    font_size = vb_ratio * 3  # in mm
    text = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        + "abcdefghijklmnopqrstuvwxyz "
        + "ÄÖÜ äöü ß€µ@²³~^°\\ 1234567890 "
        + ',.;:+-*#_<> !"§$%&/()=?{}[]'
    )
    font_filename = "fonts/Petrona-VariableFont_wght.ttf"

    avfont_w100 = instantiate_font(TTFont(font_filename), {"wght": 100})
    avfont_w900 = instantiate_font(TTFont(font_filename), {"wght": 900})

    x_pos = 0
    y_pos = 0.1
    for character in text:
        glyph: AvGlyph = avfont_w100.glyph(character)
        svg_page_output.add_glyph(glyph, x_pos, y_pos, font_size)
        x_pos += glyph.real_width(font_size)

    x_pos = 0
    y_pos = 0.1 + 2 * font_size
    for character in text:
        glyph: AvGlyph = avfont_w900.glyph(character)
        svg_page_output.add_glyph(glyph, x_pos, y_pos, font_size)
        x_pos += glyph.real_width(font_size)

    # Save the SVG file
    print("save...")
    svg_page_output.save_as(output_filename + "z", include_debug_layer=True, pretty=True, indent=2, compressed=True)
    print("save done.")

    glyph: AvGlyph = avfont_w100.glyph(" ")
    print(glyph.real_width(font_size))


if __name__ == "__main__":
    main()


#   1               2               3
#   1       2       3       4       5
#   1     2   3     4     5   6     7
#   1   2   3   4   5   6   7   8   9
#   1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7
