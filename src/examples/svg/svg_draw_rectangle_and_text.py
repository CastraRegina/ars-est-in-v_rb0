import typing
import svgwrite
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen

OUTPUT_FILE = "data/output/example/svg/din_a4_page_rectangle_and_text.svg"

CANVAS_UNIT = "mm"  # Units for CANVAS dimensions
CANVAS_WIDTH = 210  # DIN A4 page width in mm
CANVAS_HEIGHT = 297  # DIN A4 page height in mm

RECT_WIDTH = 140  # rectangle width in mm
RECT_HEIGHT = 100  # rectangle height in mm

VB_RATIO = 1 / RECT_WIDTH  # multiply each dimension with this ratio

FONT_FILENAME = "fonts/NotoSansMono-VariableFont_wdth,wght.ttf"
FONT_SIZE = VB_RATIO * 5  # in mm


class AVGlyph:
    def __init__(self, font: TTFont, character: str):
        self._font = font
        self.character = character
        self.width = 0  # value set later
        self.bounding_box = (0, 0, 0, 0)  # value set later
        self._glyph_set = None  # value set later

        glyph_name = self._font.getBestCmap()[ord(character)]
        self._glyph_set = font.getGlyphSet()[glyph_name]
        bounds_pen = BoundsPen(font.getGlyphSet())
        self._glyph_set.draw(bounds_pen)
        self.bounding_box = bounds_pen.bounds

        self.width = self._glyph_set.width

    def scaled_width(self, font_size: float, units_per_em: float) -> float:
        return font_size * self.width * 1. / units_per_em

    def scaled_position(self, x_pos: float, y_pos: float, font_size: float,
                        units_per_em: float) -> typing.Tuple[float, float]:
        if self.bounding_box:
            return (x_pos + font_size * self.bounding_box[0] / units_per_em,
                    y_pos - font_size * self.bounding_box[3] / units_per_em)
        return (0.0, 0.0)

    def scaled_size(self, font_size: float,
                    units_per_em: float) -> typing.Tuple[float, float]:
        if self.bounding_box:
            (x_min, y_min, x_max, y_max) = self.bounding_box
            return (font_size * (x_max - x_min) * 1. / units_per_em,
                    font_size * (y_max - y_min) * 1. / units_per_em)
        return (0.0, 0.0)


class AVFont:
    def __init__(self, font_filename: str):
        self._font_filename = font_filename
        self.font = TTFont(self._font_filename)
        self.ascent = self.font['hhea'].ascent
        self.descent = self.font['hhea'].descent
        self.line_gap = self.font['hhea'].lineGap
        self.units_per_em = self.font['head'].unitsPerEm
        self._glyph_cache: typing.Dict[str, AVGlyph] = {}
        self.family_name = self.font['name'].getDebugName(1)
        self.subfamily_name = self.font['name'].getDebugName(2)
        self.full_name = self.font['name'].getDebugName(4)
        self.license_description = self.font['name'].getDebugName(13)

    def glyph(self, character: str) -> AVGlyph:
        glyph = self._glyph_cache.get(character, None)
        if not glyph:
            glyph = AVGlyph(self.font, character)
            self._glyph_cache[character] = glyph
        return glyph


def main():
    # Center the rectangle horizontally and vertically on the page
    vb_w = VB_RATIO * CANVAS_WIDTH
    vb_h = VB_RATIO * CANVAS_HEIGHT
    vb_x = -VB_RATIO * (CANVAS_WIDTH - RECT_WIDTH) / 2
    vb_y = -VB_RATIO * (CANVAS_HEIGHT - RECT_HEIGHT) / 2

    # Set up the SVG canvas:
    # Define viewBox so that "1" is the width of the rectangle
    # Multiply a dimension with "VB_RATIO" to get the size regarding viewBox
    dwg = svgwrite.Drawing(OUTPUT_FILE,
                           size=(f"{CANVAS_WIDTH}mm", f"{CANVAS_HEIGHT}mm"),
                           viewBox=(f"{vb_x} {vb_y} {vb_w} {vb_h}")
                           )

    # Draw the rectangle
    dwg.add(
        dwg.rect(
            insert=(0, 0),
            size=(VB_RATIO*RECT_WIDTH, VB_RATIO*RECT_HEIGHT),  # = (1.0, xxxx)
            stroke="black",
            stroke_width=0.1*VB_RATIO,
            fill="none"
        )
    )

    font = AVFont(FONT_FILENAME)

    x_pos = VB_RATIO * 10  # in mm
    y_pos = VB_RATIO * 15  # in mm

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ " + \
           "abcdefghijklmnopqrstuvwxyz " + \
           "ÄÖÜ äöü 1234567890 " + \
           ',.;:+-*#_<> !"§$%&/()=?{}[]'

    c_x_pos = x_pos
    for character in text:
        glyph = font.glyph(character)
        pos = glyph.scaled_position(
            c_x_pos, y_pos, FONT_SIZE, font.units_per_em)
        size = glyph.scaled_size(FONT_SIZE, font.units_per_em)

        dwg.add(
            dwg.rect(
                insert=pos,
                size=size,
                stroke="red",
                stroke_width=0.05*VB_RATIO,
                fill="none"
            )
        )
        c_x_pos += glyph.scaled_width(FONT_SIZE, font.units_per_em)

    dwg.add(
        dwg.text(
            text,
            insert=(x_pos, y_pos-FONT_SIZE),
            font_family=font.family_name,
            font_size=FONT_SIZE
        )
    )

    dwg.add(
        dwg.text(
            text,
            insert=(x_pos, y_pos),
            font_family=font.family_name,
            font_size=FONT_SIZE
        )
    )

    dwg.add(
        dwg.text(
            text,
            insert=(x_pos, y_pos+FONT_SIZE),
            font_family=font.family_name,
            font_size=FONT_SIZE
        )
    )

    # Save the SVG file
    dwg.save()


if __name__ == "__main__":
    main()
