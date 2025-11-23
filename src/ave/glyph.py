"""Handling Glyphs and Fonts"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont
from numpy.typing import NDArray

import ave.consts
from ave.consts import AvGlyphCmds
from ave.fonttools import AvGlyphPtsCmdsPen, AvPolylinePen
from ave.geom import AvBox

# ==============================================================================
# Glyphs
# ==============================================================================


class AvGlyphABC(ABC):
    """
    Abstract base class for glyphs.

    A glyph is a geometric shape that represents a character in a font.
    It is composed of a set of points and a set of commands that define how to draw the shape.

    Glyphs are used to render text in a page.
    """

    @property
    @abstractmethod
    def character(self) -> str:
        """Returns the character which this Glyph represents."""
        raise NotImplementedError

    @property
    @abstractmethod
    def points(self) -> NDArray[np.float64]:
        """Returns the points of the glyph as a numpy array of shape (n_points, 2)"""
        raise NotImplementedError

    @property
    @abstractmethod
    def commands(self) -> List[AvGlyphCmds]:
        """Returns the commands of the glyph as a list of SvgPathCmds"""
        raise NotImplementedError

    @abstractmethod
    def width(self, align: Optional[ave.consts.Align] = None) -> float:
        """
        Returns the width calculated considering the alignment.
        Returns the official glyph_width of this glyph if align is None.

        Args:
            align (Optional[av.consts.Align], optional): LEFT, RIGHT, BOTH. Defaults to None.
                None:  official glyph_width (i.e. including LSB and RSB)
                LEFT:  official glyph_width - bounding_box.xmin == official width - LSB
                RIGHT: bounding_box.width + bounding_box.xmin   == official width - RSB
                BOTH:  bounding_box.width                       == official width - LSB - RSB
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def height(self) -> float:
        """
        The height of the glyph, i.e. the height of the bounding box.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a glyph (positive value).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a glyph (negative value).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of a glyph.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of a glyph.
        """
        raise NotImplementedError

    @abstractmethod
    def bounding_box(self) -> AvBox:
        """Returns bounding box (tightest box around Glyph) as
        (0:x_min, 1:y_min, 2:x_max, 3:y_max)
        relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top

        Returns:
            Tuple[float, float, float, float]: (0:x_min, 1:y_min, 2:x_max, 3:y_max)
        """
        raise NotImplementedError


@dataclass
class AvGlyph(AvGlyphABC):
    """
    Representation of a Glyph, i.e. a single character of a certain font.
    Uses dimensions in unitsPerEm, i.e. independent from font_size.
    Provides
    - geometric dimensions of the Glyph (bounding_box, ascender, descender, sidebearings, ...)
    - svg_path_string (str): a SVG path representation of the Glyph
    """

    _character: str
    _width: float
    _points: NDArray[np.float64]  # shape (n_points, 2)
    _commands: List[AvGlyphCmds]  # shape (n_commands, 1)

    _cache_bounding_box: AvBox = AvBox(0, 0, 0, 0)  # caching variable

    def __init__(self, character: str, width: float, points: NDArray[np.float64], commands: List[AvGlyphCmds]) -> None:
        """
        Initialize an AvGlyph.

        Args:
            character (str): A single character.
            width (float): The width of the glyph in unitsPerEm.
            points (NDArray[np.float64]): A numpy array of shape (n_points, 2) containing the points of the glyph.
            commands (List[AvGlyphCmds]): A list of SvgPathCmds containing the commands of the glyph.
        """
        self._character = character
        self._width = width
        self._points = points
        self._commands = commands

    @classmethod
    def from_ttfont_character(cls, ttfont: TTFont, character: str) -> AvGlyph:
        """
        Factory method to create an AvGlyph from a TTFont and character.
        """
        glyph_name = ttfont.getBestCmap()[ord(character)]
        glyph_set = ttfont.getGlyphSet()
        pen = AvGlyphPtsCmdsPen(glyph_set)
        glyph_set[glyph_name].draw(pen)
        width = glyph_set[glyph_name].width
        return cls(character, width, pen.points, pen.commands)

    @property
    def character(self) -> str:
        """
        The character of this glyph.
        """
        return self._character

    @property
    def points(self) -> NDArray[np.float64]:
        """
        The points of this glyph as a numpy array of shape (n_points, 2).
        """
        return self._points

    @property
    def commands(self) -> List[AvGlyphCmds]:
        """
        The commands of this glyph as a list of SVG path commands.
        """
        return self._commands

    def width(self, align: Optional[ave.consts.Align] = None) -> float:
        """
        Returns the width calculated considering the alignment.
        Returns the official glyph_width of this glyph if align is None.

        Args:
            align (Optional[av.consts.Align], optional): LEFT, RIGHT, BOTH. Defaults to None.
                None:  official glyph_width (i.e. including LSB and RSB)
                LEFT:  official glyph_width - bounding_box.xmin == official width - LSB
                RIGHT: bounding_box.width + bounding_box.xmin   == official width - RSB
                BOTH:  bounding_box.width                       == official width - LSB - RSB
        """
        bounding_box = self.bounding_box()
        if align is None:
            return self._width
        elif align == ave.consts.Align.LEFT:
            return self._width - bounding_box.xmin
        elif align == ave.consts.Align.RIGHT:
            return bounding_box.xmin + bounding_box.width
        elif align == ave.consts.Align.BOTH:
            return bounding_box.width
        else:
            raise ValueError(f"Invalid align value: {align}")

    @property
    def height(self) -> float:
        """
        The height of the glyph, i.e. the height of the bounding box.
        """
        return self.bounding_box().height

    @property
    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a glyph (positive value).
        """
        return self.bounding_box().ymax

    @property
    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a glyph (negative value).
        """
        return self.bounding_box().ymin

    @property
    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of a glyph.
        """
        return self.bounding_box().xmin

    @property
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of a glyph.
        """
        return self._width - self.bounding_box().xmax

    def bounding_box(self) -> AvBox:
        """
        The bounding box of the glyph.
        A bounding box is a rectangle which fully contains a glyph.
        The coordinates of the bounding box are relative to the baseline
        (0,0) with orientation left-to-right, bottom-to-top.
        Uses dimensions in unitsPerEm.
        """

        # TODO: calculate bounding box taking into account the curves and not the control points

        # If there are no points, the bounding box is already set to its default value
        # and there is no need to recalculate it.
        if not self._points.size:
            return self._cache_bounding_box

        # TODO: check which one is faster:
        # x_min, y_min = self._points.min(axis=0)
        # x_max, y_max = self._points.max(axis=0)
        # ---------------------------------------------------------------------
        # x_min, x_max = self._points[:, 0].min(), self._points[:, 0].max()
        # y_min, y_max = self._points[:, 1].min(), self._points[:, 1].max()
        # (my check so far: the sliced version is faster):
        points_x = self._points[:, 0]
        points_y = self._points[:, 1]
        x_min, x_max, y_min, y_max = points_x.min(), points_x.max(), points_y.min(), points_y.max()

        self._cache_bounding_box = AvBox(x_min, y_min, x_max, y_max)
        return self._cache_bounding_box


# ==============================================================================
# Glyph factories
# ==============================================================================


class AvGlyphFactoryABC(ABC):
    """
    Abstract base class for glyph factories.
    A glyph factory is responsible for creating glyph representations for a character.
    """

    @abstractmethod
    def create_glyph(self, character: str) -> AvGlyphABC:
        """
        Creates and returns a glyph representation for the specified character.
        Args:
            character (str): The character to create a glyph for.
        Returns:
            AvGlyph: An instance representing the glyph of the specified character.
        """


@dataclass
class AvGlyphFromTTFontFactory(AvGlyphFactoryABC):
    """Factory class for creating glyph instances."""

    _font: TTFont

    def __init__(self, font: TTFont) -> None:
        """
        Initializes the glyph factory.
        """
        self._font = font

    @property
    def font(self) -> TTFont:
        """
        Returns the TTFont instance associated with this glyph factory.
        """
        return self._font

    def create_glyph(self, character: str) -> AvGlyph:
        return AvGlyph.from_ttfont_character(self._font, character)


# ==============================================================================
# Letters
# ==============================================================================


@dataclass
class AvLetter:
    """
    A Letter is a Glyph which is scaled to real dimensions with a position and alignment.
    """

    _xpos: float  # left-to-right
    _ypos: float  # bottom-to-top
    _scale: float  # = font_size / units_per_em
    _glyph: AvGlyphABC
    _align: Optional[ave.consts.Align] = None  # LEFT, RIGHT, BOTH. Defaults to None.

    def __init__(
        self, glyph: AvGlyphABC, xpos: float, ypos: float, scale: float, align: Optional[ave.consts.Align] = None
    ) -> None:
        self._glyph = glyph
        self._xpos = xpos
        self._ypos = ypos
        self._scale = scale
        self._align = align

    @classmethod
    def from_font_size_units_per_em(
        cls,
        glyph: AvGlyphABC,
        xpos: float,
        ypos: float,
        font_size: float,
        units_per_em: float,
        align: Optional[ave.consts.Align] = None,
    ) -> AvLetter:
        """
        Factory method to create an AvLetter from font_size and units_per_em.
        """

        return cls(glyph, xpos, ypos, font_size / units_per_em, align)

    @property
    def xpos(self) -> float:
        """The x position of the letter in real dimensions."""
        return self._xpos

    @property
    def ypos(self) -> float:
        """The y position of the letter in real dimensions."""
        return self._ypos

    @property
    def scale(self) -> float:
        """Returns the scale factor for the letter which is used to transform the glyph to real dimensions."""
        return self._scale

    @property
    def align(self) -> Optional[ave.consts.Align]:
        """The alignment of the letter; None, LEFT, RIGHT, BOTH."""
        return self._align

    @property
    def trafo(self) -> List[float]:
        """
        Returns the affine transformation matrix for the letter to transform the glyph to real dimensions.
        Returns: [scale, 0, 0, scale, xpos, ypos] or [scale, 0, 0, scale, xpos-lsb, ypos] if alignment is LEFT or BOTH.
        """
        if self.align == ave.consts.Align.LEFT or self.align == ave.consts.Align.BOTH:
            lsb_scaled = self.scale * self._glyph.left_side_bearing
            return [self.scale, 0, 0, self.scale, self.xpos - lsb_scaled, self.ypos]
        return [self.scale, 0, 0, self.scale, self.xpos, self.ypos]

    @property
    def width(self) -> float:
        """
        Returns the width calculated considering the alignment.
        """
        return self.scale * self._glyph.width(self.align)

    @property
    def height(self) -> float:
        """
        The height of the Letter, i.e. the height of the bounding box.
        """
        return self.scale * self._glyph.height

    @property
    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a Letter (positive value).
        """
        return self.scale * self._glyph.ascender

    @property
    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a Letter (negative value).
        """
        return self.scale * self._glyph.descender

    @property
    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of the Letter taking alignment into account.
        """
        if self.align == ave.consts.Align.LEFT or self.align == ave.consts.Align.BOTH:
            return 0
        return self.scale * self._glyph.left_side_bearing

    @property
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of the Letter taking alignment into account.
        """
        if self.align == ave.consts.Align.RIGHT or self.align == ave.consts.Align.BOTH:
            return 0
        return self.scale * self._glyph.right_side_bearing

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
        points = self._glyph.points
        commands = self._glyph.commands
        scale, _, _, _, translate_x, translate_y = self.trafo
        return AvLetter._svg_path_string(points, commands, scale, translate_x, translate_y)

    @classmethod
    def _svg_path_string(
        cls,
        points: NDArray[np.float64],
        commands: List[AvGlyphCmds],
        scale: float = 1.0,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
    ) -> str:
        """
        Returns the SVG path representation (absolute coordinates) of the glyph.
        The SVG path is a string that defines the outline of the glyph using
        SVG path commands. This path can be used to render the glyph as a
        vector graphic.

        Supported commands:
            M (move-to), L (line-to),
            C (cubic bezier), Q (quadratic bezier),
            Z (close-path).

        Args:
            points (NDArray[np.float64]): The array of points defining the outline of the glyph.
            commands (List[AvGlyphCmds]): The list of commands defining the outline of the glyph.
            scale (float): The scale factor to apply to the points before generating the SVG path string.
            translate_x (float): X-coordinate translation before generating the SVG path string.
            translate_y (float): Y-coordinate translation before generating the SVG path string.

        Returns:
            str: The SVG path string (absolute coordinates) representing the glyph.
                    Returns "M 0 0" if there are no points.
        """
        # Apply the scale and translation to the points
        points_transformed = points * scale
        points_transformed[:, 0] += translate_x
        points_transformed[:, 1] += translate_y

        parts: List[str] = []
        p_idx = 0
        for cmd in commands:
            if cmd == "M" or cmd == "L":
                # Move-to or Line-to: one point (x,y)
                if p_idx >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x, y = points_transformed[p_idx]
                parts.append(f"{cmd} {x:g} {y:g}")
                p_idx += 1
            elif cmd == "Q":
                # Quadratic bezier: control point + end point (2 points total)
                if p_idx + 1 >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x1, y1 = points_transformed[p_idx]
                x2, y2 = points_transformed[p_idx + 1]
                parts.append(f"{cmd} {x1:g} {y1:g} {x2:g} {y2:g}")
                p_idx += 2
            elif cmd == "C":
                # Cubic bezier: control1 + control2 + end point (3 points total)
                if p_idx + 2 >= points_transformed.shape[0]:
                    raise ValueError(f"Not enough points for command {cmd}")
                x1, y1 = points_transformed[p_idx]
                x2, y2 = points_transformed[p_idx + 1]
                x3, y3 = points_transformed[p_idx + 2]
                parts.append(f"{cmd} {x1:g} {y1:g} {x2:g} {y2:g} {x3:g} {y3:g}")
                p_idx += 3
            elif cmd == "Z":
                # Close-path: no coordinates
                parts.append("Z")
            else:
                # Unsupported command (should not occur from AvPointCommandPen)
                raise ValueError(f"Unsupported SVG command in AvGlyph: {cmd}")

        # Return the composed absolute-path string or "M 0 0" string if parts is empty
        return " ".join(parts) if parts else "M 0 0"


# ==============================================================================
# Fonts
# ==============================================================================
@dataclass
class AvFontProperties:
    """
    Holds the basic properties of a font.
    """

    ascender: float = 0  # Highest y-coordinate above the baseline (positive value).
    descender: float = 0  # Lowest y-coordinate below the baseline (negative value).
    line_gap: float = 0  # Additional spacing between lines of text.
    line_height: float = 0  # = ascender - descender + line_gap
    x_height: float = 0  # height of a lowercase "x".
    cap_height: float = 0  # height of an uppercase "H".
    units_per_em: float = 0  # Number of units per EM.
    family_name: str = ""  # Family name.
    subfamily_name: str = ""  # Subfamily name.
    full_name: str = ""  # Full name.
    license_description: str = ""  # License description.
    # TODO: add dash_thickness: Return the thickness of a dash-line, e.g. as a reference value for stroke-width.
    # TODO: add em_width: Return the width of an "em"

    @classmethod
    def calculate_glyph_height(cls, font: TTFont, character: str) -> float:
        """
        Calculate the height of a glyph in a font.
        """
        glyph_name = font.getBestCmap()[ord(character)]
        glyph_set = font.getGlyphSet()
        bounds_pen = BoundsPen(glyph_set)
        glyph_set[glyph_name].draw(bounds_pen)
        if bounds_pen.bounds:
            (_, _, _, height) = bounds_pen.bounds
            return height
        else:
            return 0

    @classmethod
    def from_ttfont(cls, font: TTFont) -> AvFontProperties:
        """
        Factory method to create an AvFontProperties object from a TTFont.
        "Variable font" already configured with correct axes_values.
        """
        font_properties = cls()
        font_properties.ascender = font["hhea"].ascender  # type: ignore
        font_properties.descender = font["hhea"].descender  # type: ignore
        font_properties.line_gap = font["hhea"].lineGap  # type: ignore
        font_properties.line_height = font_properties.ascender - font_properties.descender + font_properties.line_gap
        font_properties.x_height = cls.calculate_glyph_height(font, "x")
        font_properties.cap_height = cls.calculate_glyph_height(font, "H")
        font_properties.units_per_em = font["head"].unitsPerEm  # type: ignore
        font_properties.family_name = font["name"].getDebugName(1)  # type: ignore
        font_properties.subfamily_name = font["name"].getDebugName(2)  # type: ignore
        font_properties.full_name = font["name"].getDebugName(4)  # type: ignore
        font_properties.license_description = font["name"].getDebugName(13)  # type: ignore
        return font_properties

    def info_string(self) -> str:
        """
        Return a string containing information about the font properties.
        The string is formatted for display in a text box or similar.
        """

        info_string = (
            "-----Font Information:-----\n"
            f"ascender:     {self.ascender:>5.0f} (max distance above baseline = highest y-coord, positive value)\n"
            f"descender:    {self.descender:>5.0f} (max distance below baseline = lowest y-coord, negative value)\n"
            f"line_gap:     {self.line_gap:>5.0f} (additional spacing between lines of text)\n"
            f"line_height:  {self.line_height:>5.0f} (ascender - descender + line_gap)\n"
            f"x_height:     {self.x_height:>5.0f} (height of lowercase 'x')\n"
            f"cap_height:   {self.cap_height:>5.0f} (height of uppercase 'H')\n"
            f"units_per_em: {self.units_per_em:>5.0f} (number of units per EM)\n"
            f"family_name:         {self.family_name}\n"
            f"subfamily_name:      {self.subfamily_name}\n"
            f"full_name:           {self.full_name}\n"
            f"license_description: {self.license_description}\n"
        )
        return info_string


@dataclass
class AvFont:
    """
    Representation of a Font.
    Holds a Dictionary of glyphs which can be accessed by get_glyph().
    """

    _glyph_factory: AvGlyphFactoryABC
    _font_properties: AvFontProperties = field(default_factory=AvFontProperties)
    _glyphs: Dict[str, AvGlyphABC] = field(default_factory=lambda: {})

    def __init__(self, glyph_factory: AvGlyphFactoryABC, font_properties: AvFontProperties) -> None:
        """Initialize the AvFont class."""
        self.glyph_factory = glyph_factory
        self._font_properties = font_properties
        self._glyphs = {}

    @property
    def props(self) -> AvFontProperties:
        """Returns the AvFontProperties object associated with this font."""
        return self._font_properties

    def get_glyph(self, character: str) -> AvGlyphABC:
        """Returns the AvGlyph for the given character from the caching dictionary."""
        if character not in self._glyphs:
            self._glyphs[character] = self.glyph_factory.create_glyph(character)
        return self._glyphs[character]

    def get_info_string(self) -> str:
        """
        Return a string containing information about the font.
        The string is formatted for display in a text box or similar.
        """
        font_info_string = self._font_properties.info_string()
        font_info_string += "-----Glyphs in cache:-----\n"
        glyph_count = 0
        for glyph_character in self._glyphs:
            glyph_count += 1
            font_info_string += f"{glyph_character}"
            if glyph_count % 20 == 0:
                font_info_string += "\n"
        if font_info_string[-1] != "\n":
            font_info_string += "\n"
        return font_info_string

    def actual_ascender(self):
        """Returns the overall maximum ascender by iterating over all glyphs in the cache (positive value)."""
        ascender: float = 0
        for glyph in self._glyphs.values():
            ascender = max(ascender, glyph.ascender)
        return ascender

    def actual_descender(self):
        """Returns the overall minimum descender by iterating over all glyphs in the cache (negative value)."""
        descender: float = 0
        for glyph in self._glyphs.values():
            descender = min(descender, glyph.descender)
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
