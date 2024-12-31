"""Handling Glyphs and Fonts"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

# from fontTools.pens.basePen import BasePen
from fontTools.pens.boundsPen import BoundsPen

# from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

import ave.consts
from ave.fonttools import AvPolylinePen
from ave.geom import AvBox
from ave.svgpath import AvSvgPath

# from fontTools.varLib import instancer


# ==============================================================================
# Glyphs
# ==============================================================================
@dataclass
class AvGlyph:
    """
    Representation of a Glyph, i.e. a single character of a certain font.
    Uses dimensions in unitsPerEm, i.e. independent from font_size.
    Provides
    - geometric dimensions of the Glyph (bounding_box, ascender, descender, sidebearings, ...)
    - svg_path_string (str): a SVG path representation of the Glyph
    """

    _font: TTFont
    _character: str
    _bounding_box: Optional[AvBox] = None
    _svg_path_string: str = ""

    def __init__(self, font: TTFont, character: str) -> None:
        self._font = font
        self._character = character

    @property
    def font(self) -> TTFont:
        """
        The font of this glyph.
        """
        return self._font

    @property
    def character(self) -> str:
        """
        The character of this glyph.
        """
        return self._character

    def width(self, align: Optional[ave.consts.Align] = None) -> float:
        """
        Returns the width calculated considering the alignment.

        Args:
            align (Optional[av.consts.Align], optional): LEFT, RIGHT, BOTH. Defaults to None.
                None: official glyph_width (i.e. including LSB and RSB), i.e. glyph_set[glyph_name].width
                LEFT: official glyph_width - bounding_box.xmin == official width - LSB
                RIGHT: bounding_box.width + bounding_box.xmin  == official width - RSB
                BOTH: bounding_box.width                       == official width - LSB - RSB
        """
        glyph_set = self.font.getGlyphSet()
        glyph_name = self.font.getBestCmap()[ord(self.character)]
        glyph_width = glyph_set[glyph_name].width
        bounding_box = self.bounding_box()

        if align is None:
            return glyph_width
        elif align == ave.consts.Align.LEFT:
            return glyph_width - bounding_box.xmin
        elif align == ave.consts.Align.RIGHT:
            return bounding_box.xmin + bounding_box.width
        elif align == ave.consts.Align.BOTH:
            return bounding_box.width
        else:
            raise ValueError(f"Invalid align value: {align}")

    def height(self) -> float:
        """
        The height of the glyph, i.e. the height of the bounding box.
        """
        return self.bounding_box().height

    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a glyph (positive value).
        """
        return self.bounding_box().ymax

    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a glyph (negative value).
        """
        return self.bounding_box().ymin

    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of a glyph.
        """
        return self.bounding_box().xmin

    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of a glyph.
        """
        return self.width(None) - self.bounding_box().xmax

    def bounding_box(self) -> AvBox:
        """
        The bounding box of the glyph.
        A bounding box is a rectangle which fully contains a glyph.
        The coordinates of the bounding box are relative to the baseline
        (0,0) with orientation left-to-right, bottom-to-top.
        Uses dimensions in unitsPerEm.
        """
        if not self._bounding_box:
            glyph_name = self._font.getBestCmap()[ord(self._character)]
            glyph_set = self._font.getGlyphSet()
            bounds_pen = BoundsPen(glyph_set)
            glyph_set[glyph_name].draw(bounds_pen)
            if bounds_pen.bounds:
                self._bounding_box = AvBox(*bounds_pen.bounds)
            else:
                glyph_width = glyph_set[glyph_name].width
                self._bounding_box = AvBox(0, 0, glyph_width, 0)
        return self._bounding_box

    def svg_path_string(self) -> str:
        """
        Returns the SVG path representation (absolute coordinates) of the glyph.
        The SVG path is a string that defines the outline of the glyph using
        SVG path commands. This path can be used to render the glyph as a
        vector graphic.

        Returns:
            str: The SVG path string representing the glyph.
                The path is absolute, so that it can be easily transformed.
        """
        if not self._svg_path_string:
            glyph_name = self._font.getBestCmap()[ord(self._character)]
            glyph_set = self._font.getGlyphSet()
            svg_path_pen = SVGPathPen(glyph_set)
            glyph_set[glyph_name].draw(svg_path_pen)
            svg_path_string = svg_path_pen.getCommands()
            # print(f'svg_path_string:"{svg_path_string}"')
            if not svg_path_string:
                svg_path_string = "M 0 0"
            self._svg_path_string = AvSvgPath.convert_relative_to_absolute(svg_path_string)
        return self._svg_path_string


@dataclass
class AvPolygonizedGlyph(AvGlyph):
    """
    Representation of a polygonized Glyph.
    Uses dimensions in unitsPerEm, i.e. independent from font_size.
    Provides
    - geometric dimensions of the Glyph (bounding_box, ascender, descender, sidebearings, ...)
    - polygonized_path (shapely.MultiPolygon): a polygonized representation of the Glyph
        with several shapely.Polygons (contained by shapely.MultiPolygon) representing the Glyph's contours:
        - first Polygon is always the outer contour.
        - exterior rings (shells) are always counter-clockwise, positive.
        - interior rings (holes) are always clockwise, negative.
    - svg_path_string (str): a SVG path representation of the Glyph
    """

    def __init__(self, font: TTFont, character: str) -> None:
        # pylint: disable=useless-super-delegation
        super().__init__(font, character)

    # TODO: create bounding box (property) only if called
    # TODO: create polygonized_path (property) only if called
    # TODO: create svg_path (property) only if called

    # create and store a polygonized_path:
    # pen = RecordingPen()
    # glyph_name = self._font.font.getBestCmap()[ord(self._character)]
    # glyph_set = self._font.font.getGlyphSet()
    # glyph_set[glyph_name].draw(pen)
    # for command in pen.value:
    #     print(command)

    def bounding_box(self) -> AvBox:
        return super().bounding_box()  # TODO: calculate bounding box based on polygonized_path

    def svg_path_string(self) -> str:
        # if not self._svg_path_string:
        #     glyph_name = self._font.getBestCmap()[ord(self._character)]
        #     glyph_set = self._font.getGlyphSet()
        #     path_pen = SVGPathPen(glyph_set)
        #     glyph_set[glyph_name].draw(path_pen)
        #     svg_path_string = path_pen.getCommands()
        #     self._svg_path_string = AvSvgPath.convert_relative_to_absolute(svg_path_string)
        # return self._svg_path_string
        #
        # steps = 10
        # path_pen = AvPolylinePen(glyph_set, steps)
        # glyph_set[glyph_name].draw(path_pen)
        #
        # self._svg_path_string = AvSvgPath.convert_relative_to_absolute(svg_path_string)
        return ""  # TODO


# ==============================================================================
# Glyph factories
# ==============================================================================
class AvGlyphFactoryABC(ABC):
    """
    Abstract base class for glyph factories.

    A glyph factory is responsible for creating glyph representations for a given font and character.
    """

    @abstractmethod
    def create_glyph(self, font: TTFont, character: str) -> AvGlyph:
        """
        Creates and returns a glyph representation for the specified character
        and font.

        Args:
            font (TTFont): The font object associated with the glyph.
            character (str): The character to create a glyph for.

        Returns:
            AvGlyphABC: An instance representing the glyph of the specified
            character in the given font.
        """


class AvGlyphFactory(AvGlyphFactoryABC):
    """Factory class for creating glyph instances."""

    def create_glyph(self, font: TTFont, character: str) -> AvGlyph:
        return AvGlyph(font, character)


class AvPolygonizedGlyphFactory(AvGlyphFactoryABC):
    """Factory for creating polygonized glyph instances."""

    def create_glyph(self, font: TTFont, character: str) -> AvPolygonizedGlyph:
        return AvPolygonizedGlyph(font, character)


# ==============================================================================
# Letters
# ==============================================================================
@dataclass
class AvLetter:
    """
    A letter is a Glyph which is sclaled to real dimensions with a position and a font size.
    """

    _xpos: float  # left-to-right
    _ypos: float  # bottom-to-top
    _font_size: float
    _glyph: AvGlyph

    def __init__(self, xpos: float, ypos: float, font_size: float, glyph: AvGlyph) -> None:
        self._xpos = xpos
        self._ypos = ypos
        self._font_size = font_size
        self._glyph = glyph

    @property
    def xpos(self) -> float:
        """The x position of the letter in real dimensions."""
        return self._xpos

    @property
    def ypos(self) -> float:
        """The y position of the letter in real dimensions."""
        return self._ypos

    @property
    def font_size(self) -> float:
        """The font size of the letter in real dimensions."""
        return self._font_size

    @property
    def glyph(self) -> AvGlyph:
        """The glyph of the letter."""
        return self._glyph

    @property
    def units_per_em(self) -> float:
        """The units per em of the letter's font."""
        return self._glyph.font["head"].unitsPerEm  # type: ignore

    @property
    def scale(self) -> float:
        """Returns the scale factor for the letter which is used to transform the glyph to real dimensions."""
        return self.font_size / self.units_per_em

    @property
    def trafo(self) -> List[float]:
        """
        Returns the affine transformation matrix for the letter to transform the glyph to real dimensions.
        Returns: [scale, 0, 0, scale, xpos, ypos].
        """
        return [self.scale, 0, 0, self.scale, self.xpos, self.ypos]

    def width(self, align: Optional[ave.consts.Align] = None) -> float:
        """Returns the width of the letter in real dimensions, considering the alignment."""
        glyph_width = self.glyph.width(align)
        return glyph_width * self.scale

    def bounding_box(self) -> AvBox:
        """
        Returns the bounding box of the letter in real dimensions.
        The bounding box of a letter is the smallest rectangle that completely
        contains the letter's outline. The bounding box is aligned with the
        baseline of the letter and its coordinates are relative to the baseline
        origin (0,0) with orientation left-to-right, bottom-to-top.
        Returns:
            AvBox: The bounding box of the letter.
        """
        return self._glyph.bounding_box().transform_affine(self.trafo)

    def svg_path_string(self) -> str:
        """
        Returns the SVG path representation of the letter in real dimensions.
        The SVG path is a string that defines the outline of the letter using
        SVG path commands. This path can be used to render the letter as a
        vector graphic.
        Returns:
            str: The SVG path string representing the letter.
        """
        return AvSvgPath.transform_path_string(self._glyph.svg_path_string(), self.trafo)


# ==============================================================================
# Fonts
# ==============================================================================
@dataclass
class AvFont:
    """
    Representation of a Font.
    Uses dimensions in unitsPerEm, i.e. independent from font_size.
    Holds a Dictionary of glyphs which can be accessed by get_glyph().
    """

    font: TTFont
    glyph_factory: AvGlyphFactoryABC
    glyphs: Dict[str, AvGlyph] = field(default_factory=lambda: {})
    # The maximum distance above the baseline, i.e. the highest y-coordinate (positive value).
    ascender: float = 0
    # The maximum distance below the baseline, i.e. the lowest y-coordinate (negative value).
    descender: float = 0
    line_gap: float = 0  # Additional spacing between lines of text.
    line_height: float = 0  # = ascender - descender + line_gap
    x_height: float = 0
    cap_height: float = 0
    units_per_em: float = 0
    family_name: str = ""
    subfamily_name: str = ""
    full_name: str = ""
    license_description: str = ""

    def __init__(self, font: TTFont, glyph_factory: AvGlyphFactoryABC) -> None:
        """
        Initialize the AvFont class.
        In case font is a "variable font", the font is already configured with the correct axes_values.
        """

        def calculate_glyph_height(font: TTFont, character: str) -> float:
            """Calculate the height of a glyph in a font."""
            glyph_name = font.getBestCmap()[ord(character)]
            glyph_set = font.getGlyphSet()
            bounds_pen = BoundsPen(glyph_set)
            glyph_set[glyph_name].draw(bounds_pen)
            if bounds_pen.bounds:
                (_, _, _, height) = bounds_pen.bounds
                return height
            else:
                return 0

        self.font = font
        self.glyph_factory = glyph_factory
        self.glyphs = {}
        self.ascender = self.font["hhea"].ascender  # type: ignore
        self.descender = self.font["hhea"].descender  # type: ignore
        self.line_gap = self.font["hhea"].lineGap  # type: ignore
        self.line_height = self.ascender - self.descender + self.line_gap
        self.x_height = calculate_glyph_height(self.font, "x")
        self.cap_height = calculate_glyph_height(self.font, "H")
        self.units_per_em = self.font["head"].unitsPerEm  # type: ignore
        self.family_name = self.font["name"].getDebugName(1)  # type: ignore
        self.subfamily_name = self.font["name"].getDebugName(2)  # type: ignore
        self.full_name = self.font["name"].getDebugName(4)  # type: ignore
        self.license_description = self.font["name"].getDebugName(13)  # type: ignore

    def info_string(self):
        """
        Return a string containing information about the font.

        The string is formatted for display in a text box or similar.
        It contains the following information:

        - ascender: The maximum distance above the baseline, i.e. the highest y-coordinate of a glyph (positive value).
        - descender: The maximum distance below the baseline, i.e. the lowest y-coordinate of a glyph (negative value).
        - line_gap: Additional spacing between lines of text.
        - line_height: = ascender - descender + line_gap
        - x_height: The height of a lowercase "x" in the font.
        - cap_height: The height of an uppercase "H" in the font.
        - units_per_em: The number of units per EM in the font.
        - family_name: The family name of the font.
        - subfamily_name: The subfamily name of the font.
        - full_name: The full name of the font.
        - license_description: The license description of the font.

        :return: A string containing information about the font.
        """
        info_string = (
            "-----Font Information:-----\n"
            f"ascender: {self.ascender}\n"
            f"descender: {self.descender}\n"
            f"line_gap: {self.line_gap}\n"
            f"line_height: {self.line_height}\n"
            f"x_height: {self.x_height}\n"
            f"cap_height: {self.cap_height}\n"
            f"units_per_em: {self.units_per_em}\n"
            f"family_name: {self.family_name}\n"
            f"subfamily_name: {self.subfamily_name}\n"
            f"full_name: {self.full_name}\n"
            f"license_description: {self.license_description}\n"
        )

        # Add list of glyphs in cache dictioneary
        info_string += "-----Glyphs in cache:-----\n"
        glyph_count = 0
        for glyph_character in self.glyphs:
            glyph_count += 1
            info_string += f'"{glyph_character}" '
            if glyph_count % 20 == 0:
                info_string += "\n"
        if info_string[-1] != "\n":
            info_string += "\n"
        return info_string

    def fetch_glyph(self, character: str) -> AvGlyph:
        """Returns the AvGlyph for the given character from the caching dictionary."""
        if character not in self.glyphs:
            self.glyphs[character] = self.glyph_factory.create_glyph(self.font, character)
        return self.glyphs[character]

    def overall_ascender(self):
        """Returns the overall maximum ascender by iterating over all glyphs in the cache."""
        return self.max_ascender(self.glyphs.values())

    def overall_descender(self):
        """Returns the overall minimum descender by iterating over all glyphs in the cache."""
        return self.min_descender(self.glyphs.values())

    @staticmethod
    def max_ascender(glyphs: Iterable[AvGlyph]):
        """Calculates the overall maximum ascender by iterating over the given glyphs."""
        ascender: float = 0
        for glyph in glyphs:
            ascender = max(ascender, glyph.ascender())
        return ascender

    @staticmethod
    def min_descender(glyphs: Iterable[AvGlyph]):
        """Calculates the overall minimum descender by iterating over the given glyphs."""
        descender: float = 0
        for glyph in glyphs:
            descender = min(descender, glyph.descender())
        return descender


def polygonize_glyph(font_path, character, steps=10) -> AvPolylinePen:
    """
    Polygonizes a glyph from a font file into line segments.
    """
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    glyph_name = font.getBestCmap()[ord(character)]
    pen = AvPolylinePen(glyph_set, steps)
    glyph_set[glyph_name].draw(pen)
    print(type(glyph_set[glyph_name]))
    print(dir(glyph_set[glyph_name]))
    print(glyph_set[glyph_name].glyphSet)
    print(dir(glyph_set[glyph_name].glyphSet))

    return pen


def main():
    """Main"""
    font_filename = "fonts/RobotoFlex-VariableFont_GRAD,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"

    # avfont_w100 = AvFont.instantiate(TTFont(font_filename), {"wght": 100})
    # glyph = avfont_w100.get_glyph("U")  # I

    # Example usage:
    glyph_name = "U"  # Replace with the desired glyph name
    polyline_pen = polygonize_glyph(font_filename, glyph_name)

    recording_pen = polyline_pen.recording_pen
    print(recording_pen.value)

    # svg_pen = SVGPathPen(TTFont(font_filename).getGlyphSet())
    # polyline_pen.drawPoints(svg_pen)
    # svg_path_data = svg_pen.getCommands()
    # print(svg_path_data)

    # # Example usage with SVGPathPen:
    # svg_pen = SVGPathPen(TTFont(font_filename).getGlyphSet())
    # polyline_pen.draw(svg_pen)
    # svg_path_data = svg_pen.getCommands()
    # print(svg_path_data)

    # # Create an instance of RecordingPen and pass the PolylinePen instance to it
    # recording_pen = RecordingPen()

    # # Draw the glyph using the RecordingPen
    # glyph.draw(polyline_pen)


if __name__ == "__main__":
    main()
