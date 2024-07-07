"""Module to check how to handle the font Cantarell
Actually Cantarell has different polylines for different weights.
Therefore interpolation is not easy.
"""

from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from av.glyph import AvFont, AvGlyph
from av.page import AvPageSvg


class AvFontGen:

    @staticmethod
    def interpolate_glyph(glyph0: AvGlyph, glyph1: AvGlyph, factor: float) -> AvGlyph:

        def interpolate_points(points0, points1, t):
            """Interpolate between two sets of points."""
            # return [(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])) for p0, p1 in zip(points0, points1)]

            return [
                (x1 * (1 - factor) + x2 * factor, y1 * (1 - factor) + y2 * factor)
                for (x1, y1), (x2, y2) in zip(points0, points1)
            ]

        # Create recording pens
        pen0 = RecordingPen()
        pen1 = RecordingPen()

        # Draw glyphs into pens
        glyph0._glyph_set.draw(pen0)
        glyph1._glyph_set.draw(pen1)

        # Get the recorded glyph paths
        glyph0_commands = pen0.value
        glyph1_commands = pen1.value

        print(glyph0_commands)
        print("-------------------")
        print(glyph1_commands)
        print("-------------------")
        print()

        # Create a new pen to construct the interpolated glyph
        pen = TTGlyphPen(None)

        # Interpolate the paths
        for command0, command1 in zip(glyph0_commands, glyph1_commands):

            cmd_type0 = command0[0]
            cmd_type1 = command1[0]
            points0 = command0[1]
            points1 = command1[1]

            print("cmd0:", cmd_type0)
            print("cmd1:", cmd_type1)
            print("points0:", points0)
            print("points1:", points1)

            if cmd_type0 == "moveTo":
                pen.moveTo(interpolate_points(points0, points1, factor)[0])
            elif cmd_type0 == "lineTo":
                pen.lineTo(interpolate_points(points0, points1, factor)[0])
            elif cmd_type0 == "qCurveTo":
                pen.qCurveTo(*interpolate_points(points0, points1, factor))
            elif cmd_type0 == "curveTo":
                pen.curveTo(*interpolate_points(points0, points1, factor))
            elif cmd_type0 == "closePath":
                pen.closePath()
            else:
                raise ValueError(f"Unsupported command type: {cmd_type0}")

        return None  # TODO !!!


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
    font_filename_bold = "fonts/Cantarell-Bold.ttf"

    avfont_regular = AvFont(TTFont(font_filename_regular))
    avfont_bold = AvFont(TTFont(font_filename_bold))

    x_pos = 0
    y_pos = 0.1
    for character in text:
        glyph: AvGlyph = avfont_regular.glyph(character)
        svg_page_output.add_glyph(glyph, x_pos, y_pos, font_size)
        x_pos += glyph.real_width(font_size)

    x_pos = 0
    y_pos = 0.1 + 1.5 * font_size
    for character in text:
        glyph: AvGlyph = avfont_bold.glyph(character)
        svg_page_output.add_glyph(glyph, x_pos, y_pos, font_size)
        x_pos += glyph.real_width(font_size)

    character = "B"
    glyph = AvFontGen.interpolate_glyph(AvGlyph(avfont_regular, character), AvGlyph(avfont_bold, character), 0.5)
    svg_page_output.add_glyph(glyph, x_pos, y_pos + 1.0, font_size)

    # Save the SVG file
    print("save...")
    svg_page_output.save_as(output_filename + "z", include_debug_layer=True, pretty=True, indent=2, compressed=True)
    print("save done.")


if __name__ == "__main__":
    main()


#   1               2               3
#   1       2       3       4       5
#   1     2   3     4     5   6     7
#   1   2   3   4   5   6   7   8   9
#   1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7
