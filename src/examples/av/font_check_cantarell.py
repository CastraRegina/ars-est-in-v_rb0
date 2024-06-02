"""Module to check how to handle the font Cantarell"""

from fontTools.ttLib import TTFont

from av.glyph import AvFont, AvGlyph
from av.page import AvPageSvg


def main():
    """Main"""
    output_filename = "data/output/example/svg/example_font_Cantarell.svg"

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

    # prepare variables for fonts
    font_size = vb_ratio * 3  # in mm
    text = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        + "abcdefghijklmnopqrstuvwxyz "
        + "ÄÖÜ äöü ß€µ@²³~^°\\ 1234567890 "
        + ',.;:+-*#_<> !"§$%&/()=?{}[]'
    )
    font_filename_regular = "fonts/Cantarell-Regular.ttf"
    # font_filename_regular = "fonts/Cantarell-Bold.ttf"

    ttfont_regular = TTFont(font_filename_regular)
    avfont_regular = AvFont(ttfont_regular)

    x_pos = 0
    y_pos = 0.1
    for character in text:
        glyph: AvGlyph = avfont_regular.glyph(character)
        svg_page_output.add_glyph(glyph, x_pos, y_pos, font_size)
        x_pos += glyph.real_width(font_size)

    # Save the SVG file
    print("save...")
    svg_page_output.save_as(output_filename + "z", include_debug_layer=True, pretty=True, indent=2, compressed=True)
    print("save done.")


if __name__ == "__main__":
    main()
