"""Handling Glyphs for SVG"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import svgwrite
import svgwrite.base
import svgwrite.container
import svgwrite.elementfactory
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

import av.consts
import av.helper
import av.path


class AvGlyph:
    """Representation of a Glyph (single character of a certain font) for SVG.
    A Glyph is independent from font_size.
    To get the *real* dimensions call the functions by providing *font_size*.
    """

    def __init__(self, avfont: AvFont, character: str):
        self._avfont: AvFont = avfont
        self.character: str = character
        bounds_pen = BoundsPen(self._avfont.ttfont.getGlyphSet())
        glyph_name = self._avfont.ttfont.getBestCmap()[ord(character)]
        self._glyph_set = self._avfont.ttfont.getGlyphSet()[glyph_name]
        self._glyph_set.draw(bounds_pen)
        self.bounding_box = bounds_pen.bounds  # (0:x_min, 1:y_min, 2:x_max, 3:y_max)
        self.width = self._glyph_set.width
        # create and store a polygonized_path_string:
        svg_pen = SVGPathPen(self._avfont.ttfont.getGlyphSet())
        self._glyph_set.draw(svg_pen)
        self.path_string = svg_pen.getCommands()
        self.polygonized_path_string = AvGlyph.polygonize_path_string(self.path_string)

    def font_ascender(self) -> float:
        """Returns the ascender of the font

        Returns:
            float: ascender in unitsPerEm
        """
        return self._avfont.ascender

    def font_descender(self) -> float:
        """Returns the descender of the font

        Returns:
            float: descender in unitsPerEm
        """
        return self._avfont.descender

    def real_width(self, font_size: float, align: Optional[av.consts.Align] = None) -> float:
        real_width = self.width * font_size / self._avfont.units_per_em
        if not align:
            return real_width
        (bb_x_pos, _, bb_width, _) = self.rect_bounding_box(0, 0, font_size)

        if align == av.consts.Align.LEFT:
            return real_width - bb_x_pos
        elif align == av.consts.Align.RIGHT:
            return bb_x_pos + bb_width
        elif align == av.consts.Align.BOTH:
            return bb_width
        else:
            print("ERROR in real_width(): align-value not implemented", align)
            return real_width

    def real_dash_thickness(self, font_size: float) -> float:
        glyph = self._avfont.glyph("-")
        if glyph.bounding_box:
            thickness = glyph.bounding_box[3] - glyph.bounding_box[1]
            return thickness * font_size / self._avfont.units_per_em
        return 0.0

    def real_sidebearing_left(self, font_size: float) -> float:
        if self.bounding_box:
            return self.bounding_box[0] * font_size / self._avfont.units_per_em
        return 0.0

    def real_sidebearing_right(self, font_size: float) -> float:
        if self.bounding_box:
            sidebearing_right = self.width - self.bounding_box[2]
            return sidebearing_right * font_size / self._avfont.units_per_em
        return 0.0

    def real_path_string(self, x_pos: float, y_pos: float, font_size: float) -> str:
        scale = font_size / self._avfont.units_per_em
        path_string = av.path.AvSvgPath.transform_path_string(
            self.polygonized_path_string, [scale, 0, 0, -scale, x_pos, y_pos]
        )
        return path_string

    def svg_path(
        self,
        dwg: svgwrite.Drawing,
        x_pos: float,
        y_pos: float,
        font_size: float,
        **svg_properties,
    ) -> svgwrite.elementfactory.ElementBuilder:
        path_string = self.real_path_string(x_pos, y_pos, font_size)
        svg_path = dwg.path(path_string, **svg_properties)
        return svg_path

    # def svg_text(
    #     self,
    #     dwg: svgwrite.Drawing,
    #     x_pos: float,
    #     y_pos: float,
    #     font_size: float,
    #     **svg_properties,
    # ) -> svgwrite.elementfactory.ElementBuilder:
    #     text_properties = {
    #         "insert": (x_pos, y_pos),
    #         "font_family": self._avfont.family_name,
    #         "font_size": font_size,
    #     }
    #     text_properties.update(svg_properties)
    #     ret_text = dwg.text(self.character, **text_properties)
    #     return ret_text

    def rect_em(
        self,
        x_pos: float,
        y_pos: float,
        ascent: float,
        descent: float,
        real_width: float,
        font_size: float,
    ) -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        units_per_em = self._avfont.units_per_em
        middle_of_em = 0.5 * (ascent + descent) * font_size / units_per_em

        rect = (x_pos, y_pos - middle_of_em - 0.5 * font_size, real_width, font_size)
        return rect

    def rect_em_width(
        self,
        x_pos: float,
        y_pos: float,
        ascent: float,
        descent: float,
        font_size: float,
    ) -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        return self.rect_em(x_pos, y_pos, ascent, descent, self.real_width(font_size), font_size)

    def rect_given_ascent_descent(
        self,
        x_pos: float,
        y_pos: float,
        ascent: float,
        descent: float,
        font_size: float,
    ) -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        units_per_em = self._avfont.units_per_em
        rect = (
            x_pos,
            y_pos - ascent * font_size / units_per_em,
            self.real_width(font_size),
            font_size - descent * font_size / units_per_em,
        )
        return rect

    def rect_font_ascent_descent(
        self, x_pos: float, y_pos: float, font_size: float
    ) -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        ascent = self._avfont.ascender
        descent = self._avfont.descender
        return self.rect_given_ascent_descent(x_pos, y_pos, ascent, descent, font_size)

    def rect_bounding_box(self, x_pos: float, y_pos: float, font_size: float) -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        rect = (0.0, 0.0, 0.0, 0.0)
        if self.bounding_box:
            units_per_em = self._avfont.units_per_em
            (x_min, y_min, x_max, y_max) = self.bounding_box
            rect = (
                x_pos + x_min * font_size / units_per_em,
                y_pos - y_max * font_size / units_per_em,
                (x_max - x_min) * font_size / units_per_em,
                (y_max - y_min) * font_size / units_per_em,
            )
        return rect

    def area_coverage(self, ascent: float, descent: float, font_size: float) -> float:
        glyph_string = self.real_path_string(0, 0, font_size)
        glyph_polygon = av.path.AvPathPolygon()
        glyph_polygon.add_path_string(glyph_string)

        rect = self.rect_em_width(0, 0, ascent, descent, font_size)
        rect_string = av.helper.HelperSvg.rect_to_path(rect)
        rect_polygon = av.path.AvPathPolygon()
        rect_polygon.add_path_string(rect_string)

        inter = rect_polygon.multipolygon.intersection(glyph_polygon.multipolygon)
        rect_area = rect_polygon.multipolygon.area

        return inter.area / rect_area

    @staticmethod
    def svg_rect(
        dwg: svgwrite.Drawing,
        rect: Tuple[float, float, float, float],
        stroke: str,
        stroke_width: float,
        **svg_properties,
    ) -> svgwrite.elementfactory.ElementBuilder:
        (x_pos, y_pos, width, height) = rect
        rect_properties = {
            "insert": (x_pos, y_pos),
            "size": (width, height),
            "stroke": stroke,  # color
            "stroke_width": stroke_width,
            "fill": "none",
        }
        rect_properties.update(svg_properties)
        return dwg.rect(**rect_properties)

    @staticmethod
    def polygonize_path_string(path_string: str) -> str:
        if not path_string:
            path_string = "M 0 0"
        else:
            polygon = av.path.AvPathPolygon()
            poly_func = None
            match av.consts.POLYGONIZE_TYPE:
                case av.consts.Polygonize.UNIFORM:
                    poly_func = av.path.AvPathPolygon.polygonize_uniform
                case av.consts.Polygonize.BY_ANGLE:
                    poly_func = av.path.AvPathPolygon.polygonize_by_angle
            path_string = av.path.AvPathPolygon.polygonize_path(path_string, poly_func)

            polygon.add_path_string(path_string)
            path_strings = polygon.path_strings()
            path_string = " ".join(path_strings)
        return path_string


# pyright: reportAttributeAccessIssue=false
class AvFont:
    """Representation of a Font used by Glyph-class"""

    def __init__(self, ttfont: TTFont):
        # ttfont is already configured with the given axes_values
        self.ttfont: TTFont = ttfont
        self.ascender: float = self.ttfont["hhea"].ascender  # in unitsPerEm
        self.descender: float = self.ttfont["hhea"].descender  # in unitsPerEm
        self.line_gap: float = self.ttfont["hhea"].lineGap  # in unitsPerEm
        self.x_height: float = self.ttfont["OS/2"].sxHeight  # in unitsPerEm
        self.cap_height: float = self.ttfont["OS/2"].sCapHeight  # in unitsPerEm
        self.units_per_em: float = self.ttfont["head"].unitsPerEm
        self.family_name: str = self.ttfont["name"].getDebugName(1)
        self.subfamily_name: str = self.ttfont["name"].getDebugName(2)
        self.full_name: str = self.ttfont["name"].getDebugName(4)
        self.license_description: str = self.ttfont["name"].getDebugName(13)
        self._glyph_cache: Dict[str, AvGlyph] = {}  # character->AVGlyph

    def glyph(self, character: str) -> AvGlyph:
        glyph = self._glyph_cache.get(character, None)
        if not glyph:
            glyph = AvGlyph(self, character)
            self._glyph_cache[character] = glyph
        return glyph

    def glyph_ascent_descent_of(self, characters: str) -> Tuple[float, float]:
        (ascent, descent) = (0.0, 0.0)
        for char in characters:  # get "first" char to initialize
            if bounding_box := self.glyph(char).bounding_box:
                (_, descent, _, ascent) = bounding_box
                break
        for char in characters:  # iterate over all characters
            if bounding_box := self.glyph(char).bounding_box:
                (_, y_min, _, y_max) = bounding_box
                ascent = max(ascent, y_max)
                descent = min(descent, y_min)
        return (ascent, descent)

    @staticmethod
    def default_axes_values(ttfont: TTFont) -> Dict[str, float]:
        axes_values: Dict[str, float] = {}
        for axis in ttfont["fvar"].axes:
            axes_values[axis.axisTag] = axis.defaultValue
        return axes_values

    @staticmethod
    def real_value(ttfont: TTFont, font_size: float, value: float) -> float:
        units_per_em = ttfont["head"].unitsPerEm
        return value * font_size / units_per_em


def main():
    """Main"""


if __name__ == "__main__":
    main()
