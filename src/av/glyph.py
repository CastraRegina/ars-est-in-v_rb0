"""Handling Glyphs for SVG"""

from __future__ import annotations

from abc import ABC, abstractmethod
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


class AvGlyphABC(ABC):
    """Representation of a Glyph (single character of a certain font).
    Purpose of class which implements this abstract class is to provide data regarding
    - the extensions of the Glyph like
      width, height, font-ascent, font-descent, left-/right-sidebearings, ...)
    - a SVG-path representation of the Glyph
    A Glyph returns *real* dimensions.
    Therefore *font_size* is used internally to provide the *real* dimensions.
    """

    @abstractmethod
    def character(self) -> str:
        """Returns the character which this Glyph represents."""

    @abstractmethod
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns bounding box (tightest box around Glyph) as
        (0:x_min, 1:y_min, 2:x_max, 3:y_max)
        relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top

        Returns:
            Tuple[float, float, float, float]: (0:x_min, 1:y_min, 2:x_max, 3:y_max)
        """

    # bb_width # bounding box width (w/o LSB and RSB) of Glyph
    # bb_height # bounding box height of Glyph

    # advance_width # width to the next Glyph origin, i.e. LSB + bb_width + RSB

    # left_sidebearing
    # right_sidebearing

    # font_ascent  # baseline to top-line of font
    # font_descent # baseline to bottom-line of font

    # def area_coverage(self) -> float:
    #   using advance_width and font_ascent / font_descent

    # svg_path


class AvGlyph:
    """Representation of a Glyph (single character of a certain font).
    Purpose of this class is to provide data regarding
    - the extensions of the Glyph (width, height, ascender, descender, sidebearings, ...)
    - a SVG-path representation of the Glyph
    A Glyph is independent from font_size.
    To get the *real* dimensions call the functions by providing *font_size*.
    *real* means the dimension in real-coordinates, i.e. the dimension in the SVG-output.
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
        self.polygonized_path_string = av.path.AvSvgPath.polygonize_svg_path_string(self.path_string)

    def font_ascender(self) -> float:
        """Returns the ascender of the font in unitsPerEm

        Returns:
            float: ascender in unitsPerEm
        """
        return self._avfont.ascender

    def font_descender(self) -> float:
        """Returns the descender of the font in unitsPerEm

        Returns:
            float: descender in unitsPerEm
        """
        return self._avfont.descender

    def real_width(self, font_size: float, align: Optional[av.consts.Align] = None) -> float:
        """Returns the "real" width calculated by using _font_size_.

        Args:
            font_size (float): font_size
            align (Optional[av.consts.Align], optional): LEFT, RIGHT, BOTH. Defaults to None.
                None: official width (i.e. including LSB and RSB) from _glyph_set.width
                LEFT: official width - bounding_box.x_pos      = official width - LSB
                RIGHT: bounding_box.width + bounding_box.x_pos = official width - RSB
                BOTH: bounding_box.width                       = official width - LSB - RSB

        Returns:
            float: width
        """
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
        """Return the thickness of a dash-line.
        For example this value can be used as a reference value.

        Args:
            font_size (float): font_size in order to calculate the real thickness.

        Returns:
            float: the thickness of a dash-line
        """
        glyph = self._avfont.glyph("-")
        if glyph.bounding_box:
            thickness = glyph.bounding_box[3] - glyph.bounding_box[1]
            return thickness * font_size / self._avfont.units_per_em
        return 0.0

    def real_sidebearing_left(self, font_size: float) -> float:
        """The left side bearing (LSB) refers to the horizontal space on the left side of
        an individual character or glyph. The LSB, along with the Right Side Bearing (RSB),
        ensures that characters sit beside one another with an even appearance.
        A negative LSB means that the left sidebearing extends beyond the left edge of the glyph,
        resulting in a character that appears visually shifted to the left.

        Args:
            font_size (float): font_size

        Returns:
            float: left side bearing LSB
        """
        if self.bounding_box:
            return self.bounding_box[0] * font_size / self._avfont.units_per_em
        return 0.0

    def real_sidebearing_right(self, font_size: float) -> float:
        """The right side bearing (RSB) refers to the horizontal space on the right side of
        an individual character or glyph. The RSB, along with the Left Side Bearing (LSB),
        ensures that characters sit beside one another with an even appearance.
        A negative RSB means that the right sidebearing extends beyond the right edge of the glyph,
        resulting in a character that appears visually shifted to the right.

        Args:
            font_size (float): font_size

        Returns:
            float: right side bearing RSB
        """
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
        **svg_properties: str,
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

    # def rect_em(
    #     self,
    #     x_pos: float,
    #     y_pos: float,
    #     ascent: float,
    #     descent: float,
    #     real_width: float,
    #     font_size: float,
    # ) -> Tuple[float, float, float, float]:
    #     # returns (x_pos_left_corner, y_pos_top_corner, width, height)
    #     units_per_em = self._avfont.units_per_em
    #     middle_of_em = 0.5 * (ascent + descent) * font_size / units_per_em

    #     rect = (x_pos, y_pos - middle_of_em - 0.5 * font_size, real_width, font_size)
    #     return rect

    def rect_em_width(
        self,
        x_pos: float,
        y_pos: float,
        ascent: float,
        descent: float,
        font_size: float,
    ) -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        units_per_em = self._avfont.units_per_em
        middle_of_em = 0.5 * (ascent + descent) * font_size / units_per_em
        rect = (x_pos, y_pos - middle_of_em - 0.5 * font_size, self.real_width(font_size), font_size)
        return rect

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


# pyright: reportAttributeAccessIssue=false
class AvFont:
    """Representation of a Font used by Glyph-class"""

    def __init__(self, ttfont: TTFont):
        # remark: ttfont is already configured with the given axes_values

        def get_char_height(char: str) -> float:
            bounds_pen = BoundsPen(ttfont.getGlyphSet())
            glyph_name = ttfont.getBestCmap()[ord(char)]
            glyph_set = ttfont.getGlyphSet()[glyph_name]
            glyph_set.draw(bounds_pen)
            (_, _, _, char_height) = bounds_pen.bounds
            return char_height

        self.ttfont: TTFont = ttfont
        self.ascender: float = self.ttfont["hhea"].ascender  # in unitsPerEm
        self.descender: float = self.ttfont["hhea"].descender  # in unitsPerEm
        self.line_gap: float = self.ttfont["hhea"].lineGap  # in unitsPerEm
        # self.x_height: float = self.ttfont["OS/2"].sxHeight  # in unitsPerEm
        self.x_height: float = get_char_height("x")  # in unitsPerEm
        # self.cap_height: float = self.ttfont["OS/2"].sCapHeight  # in unitsPerEm
        self.cap_height: float = get_char_height("H")  # in unitsPerEm
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
        """Retrieve the real ascent and descent values for the given *characters*
           based on values (i.e. min(y_min), max(y_max)) of the bounding boxes

        Args:
            characters (str): String of characters

        Returns:
            Tuple[float, float]: (ascent, descent)
        """
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

    # @staticmethod
    # def real_value(ttfont: TTFont, font_size: float, value: float) -> float:
    #     units_per_em = ttfont["head"].unitsPerEm
    #     return value * font_size / units_per_em


def main():
    """Main"""


if __name__ == "__main__":
    main()
