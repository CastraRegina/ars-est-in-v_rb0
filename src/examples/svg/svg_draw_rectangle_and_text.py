# import typing
from typing import Dict, Tuple
# from dataclasses import dataclass
import svgwrite
# from svgwrite.elementfactory import ElementBuilder
from fontTools.ttLib import TTFont
# from fontTools.ufoLib.glifLib import GlyphSet
from fontTools.pens.boundsPen import BoundsPen

OUTPUT_FILE = "data/output/example/svg/din_a4_page_rectangle_and_text.svg"

CANVAS_UNIT = "mm"  # Units for CANVAS dimensions
CANVAS_WIDTH = 210  # DIN A4 page width in mm
CANVAS_HEIGHT = 297  # DIN A4 page height in mm

RECT_WIDTH = 140  # rectangle width in mm
RECT_HEIGHT = 100  # rectangle height in mm

VB_RATIO = 1 / RECT_WIDTH  # multiply each dimension with this ratio

FONT_FILENAME = "fonts/RobotoFlex-VariableFont_GRAD,XTRA," + \
                "YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"
# FONT_FILENAME = "fonts/Recursive-VariableFont_CASL,CRSV,MONO,slnt,wght.ttf"
# FONT_FILENAME = "fonts/NotoSansMono-VariableFont_wdth,wght.ttf"

FONT_SIZE = VB_RATIO * 3  # in mm


class AVGlyph:
    pass


class AVFont:
    def __init__(self, ttfont: TTFont,
                 axes_values: Dict[str, float] = None):
        # ttfont is already configured with the given axes_values
        self.ttfont = ttfont
        self.axes_values = axes_values if axes_values else {}
        self.ascender = self.ttfont['hhea'].ascender  # in unitsPerEm
        self.descender = self.ttfont['hhea'].descender  # in unitsPerEm
        self.line_gap = self.ttfont['hhea'].lineGap  # in unitsPerEm
        self.x_height = self.ttfont["OS/2"].sxHeight  # in unitsPerEm
        self.cap_height = self.ttfont["OS/2"].sCapHeight  # in unitsPerEm
        self.units_per_em = self.ttfont['head'].unitsPerEm
        self.family_name = self.ttfont['name'].getDebugName(1)
        self.subfamily_name = self.ttfont['name'].getDebugName(2)
        self.full_name = self.ttfont['name'].getDebugName(4)
        self.license_description = self.ttfont['name'].getDebugName(13)
        self._glyph_cache: Dict[str, AVGlyph] = {}  # character->AVGlyph

    def real_ascender(self, font_size: float) -> float:
        return self.ascender * font_size / self.units_per_em

    def real_descender(self, font_size: float) -> float:
        return self.descender * font_size / self.units_per_em

    def real_line_gap(self, font_size: float) -> float:
        return self.line_gap * font_size / self.units_per_em

    def real_x_height(self, font_size: float) -> float:
        return self.x_height * font_size / self.units_per_em

    def real_cap_height(self, font_size: float) -> float:
        return self.cap_height * font_size / self.units_per_em

    def glyph(self, character: str) -> AVGlyph:
        glyph = self._glyph_cache.get(character, None)
        if not glyph:
            glyph = AVGlyph(self, character)
            self._glyph_cache[character] = glyph
        return glyph

    def glyph_ascent_descent_of(self, characters: str) -> Tuple[float, float]:
        (ascent, descent) = (0.0, 0.0)
        for char in characters:
            if bounding_box := self.glyph(char).bounding_box:
                (_, descent, _, ascent) = bounding_box
                break
        for char in characters:
            if bounding_box := self.glyph(char).bounding_box:
                (_, y_min, _, y_max) = bounding_box
                ascent = max(ascent, y_max)
                descent = min(descent, y_min)
        return (ascent, descent)

    @staticmethod
    def default_axes_values(ttfont: TTFont) -> Dict[str, float]:
        axes_values: Dict[str, float] = {}
        for axis in ttfont['fvar'].axes:
            axes_values[axis.axisTag] = axis.defaultValue
        return axes_values

    @staticmethod
    def real_value(ttfont: TTFont, font_size: float, value: float) -> float:
        # value in unitsPerEm
        units_per_em = ttfont['head'].unitsPerEm
        return value * font_size / units_per_em


class AVGlyph:  # pylint: disable=function-redefined
    def __init__(self, avfont: AVFont, character: str):
        self._avfont: AVFont = avfont
        self.character: str = character
        glyph_name = self._avfont.ttfont.getBestCmap()[ord(character)]
        self._glyph_set = self._avfont.ttfont.getGlyphSet()[glyph_name]
        bounds_pen = BoundsPen(self._avfont.ttfont.getGlyphSet())
        self._glyph_set.draw(bounds_pen)
        self.bounding_box = bounds_pen.bounds  # (x_min, y_min, x_max, y_max)
        self.width = self._glyph_set.width

    def real_width(self, font_size: float) -> float:
        return self.width * font_size / self._avfont.units_per_em

    def svg_text(self, dwg: svgwrite.Drawing,
                 x_pos: float, y_pos: float,
                 font_size: float, **svg_properties) \
            -> svgwrite.elementfactory.ElementBuilder:
        text_properties = {"insert": (x_pos, y_pos),
                           "font_family": self._avfont.family_name,
                           "font_size": font_size}
        text_properties.update(svg_properties)
        # for (a, v) in text_properties.items():
        #     print(a, v)
        ret_text = dwg.text(self.character, **text_properties)
        return ret_text

    def rect_em_width(self, x_pos: float, y_pos: float,
                      ascent: float, descent: float,
                      font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        units_per_em = self._avfont.units_per_em
        middle_of_em = 0.5 * (ascent + descent) * font_size / units_per_em

        rect = (x_pos,
                y_pos - middle_of_em - 0.5 * font_size,
                self.real_width(font_size),
                font_size)
        return rect

    def rect_given_ascent_descent(self, x_pos: float, y_pos: float,
                                  ascent: float, descent: float,
                                  font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        units_per_em = self._avfont.units_per_em
        rect = (x_pos,
                y_pos - ascent * font_size / units_per_em,
                self.real_width(font_size),
                font_size - descent * font_size / units_per_em)
        return rect

    def rect_font_ascent_descent(self, x_pos: float, y_pos: float,
                                 font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        ascent = self._avfont.ascender
        descent = self._avfont.descender
        return self.rect_given_ascent_descent(x_pos, y_pos,
                                              ascent, descent,
                                              font_size)

    def rect_bounding_box(self, x_pos: float, y_pos: float, font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        rect = (0.0, 0.0, 0.0, 0.0)
        if self.bounding_box:
            units_per_em = self._avfont.units_per_em
            (x_min, y_min, x_max, y_max) = self.bounding_box
            rect = (x_pos + x_min * font_size / units_per_em,
                    y_pos - y_max * font_size / units_per_em,
                    (x_max - x_min) * font_size / units_per_em,
                    (y_max - y_min) * font_size / units_per_em)
        return rect

    def svg_rect(self, dwg: svgwrite.Drawing,
                 rect: Tuple[float, float, float, float],
                 stroke: str, stroke_width: float, **svg_properties) \
            -> svgwrite.elementfactory.ElementBuilder:
        (x_pos, y_pos, width, height) = rect
        rect_properties = {"insert": (x_pos, y_pos),
                           "size": (width, height),
                           "stroke": stroke,
                           "stroke_width": stroke_width,
                           "fill": "none"}
        rect_properties.update(svg_properties)
        return dwg.rect(**rect_properties)


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

    ttfont = TTFont(FONT_FILENAME)
    axes_values = AVFont.default_axes_values(ttfont)
    font = AVFont(ttfont, axes_values)

    x_pos = VB_RATIO * 10  # in mm
    y_pos = VB_RATIO * 10  # in mm

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ " + \
           "abcdefghijklmnopqrstuvwxyz " + \
           "ÄÖÜ äöü ß€µ@²³~^°\\ 1234567890 " + \
           ',.;:+-*#_<> !"§$%&/()=?{}[]'

    (ascent, descent) = font.glyph_ascent_descent_of(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ " +
        "abcdefghijklmnopqrstuvwxyz ")

    c_x_pos = x_pos
    c_y_pos = y_pos
    for character in text:
        glyph: AVGlyph = font.glyph(character)

        rect = glyph.rect_font_ascent_descent(c_x_pos, c_y_pos, FONT_SIZE)
        dwg.add(glyph.svg_rect(dwg, rect, "green", 0.05*VB_RATIO))

        rect = glyph.rect_em_width(
            c_x_pos, c_y_pos, ascent, descent, FONT_SIZE)
        dwg.add(glyph.svg_rect(dwg, rect, "blue", 0.05*VB_RATIO))

        rect = glyph.rect_bounding_box(c_x_pos, c_y_pos, FONT_SIZE)
        dwg.add(glyph.svg_rect(dwg, rect, "red", 0.025*VB_RATIO))

        dwg.add(glyph.svg_text(dwg, c_x_pos, c_y_pos, FONT_SIZE))

        c_x_pos += glyph.real_width(FONT_SIZE)

    c_x_pos = x_pos
    c_y_pos = y_pos - FONT_SIZE
    for character in text:
        glyph: AVGlyph = font.glyph(character)

        # rect = glyph.rect_font_ascent_descent(c_x_pos, c_y_pos, FONT_SIZE)
        # dwg.add(glyph.svg_rect(dwg, rect, "green", 0.05*VB_RATIO))

        rect = glyph.rect_em_width(
            c_x_pos, c_y_pos, ascent, descent, FONT_SIZE)
        dwg.add(glyph.svg_rect(dwg, rect, "blue", 0.05*VB_RATIO))

        rect = glyph.rect_bounding_box(c_x_pos, c_y_pos, FONT_SIZE)
        dwg.add(glyph.svg_rect(dwg, rect, "red", 0.025*VB_RATIO))

        dwg.add(glyph.svg_text(dwg, c_x_pos, c_y_pos, FONT_SIZE))

        c_x_pos += glyph.real_width(FONT_SIZE)

    c_x_pos = x_pos
    c_y_pos = y_pos + FONT_SIZE
    for character in text:
        glyph: AVGlyph = font.glyph(character)

        # rect = glyph.rect_font_ascent_descent(c_x_pos, c_y_pos, FONT_SIZE)
        # dwg.add(glyph.svg_rect(dwg, rect, "green", 0.05*VB_RATIO))

        rect = glyph.rect_em_width(
            c_x_pos, c_y_pos, ascent, descent, FONT_SIZE)
        dwg.add(glyph.svg_rect(dwg, rect, "blue", 0.05*VB_RATIO))

        rect = glyph.rect_bounding_box(c_x_pos, c_y_pos, FONT_SIZE)
        dwg.add(glyph.svg_rect(dwg, rect, "red", 0.025*VB_RATIO))

        dwg.add(glyph.svg_text(dwg, c_x_pos, c_y_pos, FONT_SIZE))

        c_x_pos += glyph.real_width(FONT_SIZE)

    # print(font.glyph_ascent_descent_of('"'))
    # print(font.glyph_ascent_descent_of(' "'))
    # print(font.glyph_ascent_descent_of(' " '))
    # print(font.glyph_ascent_descent_of('_'))
    # print(font.glyph_ascent_descent_of(' _'))
    # print(font.glyph_ascent_descent_of(' _ '))
    # print(font.glyph_ascent_descent_of('a'))
    # print(font.glyph_ascent_descent_of(' a'))
    # print(font.glyph_ascent_descent_of('  a'))
    # print(font.glyph_ascent_descent_of(' a '))
    # print(font.glyph_ascent_descent_of('  a '))
    # print(font.glyph_ascent_descent_of(''))
    # print(font.glyph_ascent_descent_of(' '))
    # print(font.glyph_ascent_descent_of('  '))
    # print(font.glyph_ascent_descent_of('Ä'))

    # Save the SVG file
    dwg.save()


if __name__ == "__main__":
    main()
