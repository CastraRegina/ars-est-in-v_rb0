"""Handling Glyphs and Fonts"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont
from numpy.typing import NDArray

import ave.common
from ave.common import AvGlyphCmds
from ave.fonttools import AvGlyphPtsCmdsPen
from ave.geom import AvBox, AvPath

###############################################################################
# Glyph
###############################################################################


@dataclass
class AvGlyph:
    """
    Representation of a Glyph, i.e. a single character of a certain font.
    Uses dimensions in unitsPerEm, i.e. independent from font_size.
    It is composed of a set of points and a set of commands that define how to draw the shape.
    Glyphs are used to render text in a page.
    """

    _character: str
    _width: float
    _path: AvPath

    def __init__(
        self,
        character: str,
        width: float,
        path: AvPath,
    ) -> None:
        """
        Initialize an AvGlyph.

        Args:
            character (str): A single character.
            width (float): The width of the glyph in unitsPerEm.
            path (AvPath): The path object containing points and commands for the glyph.
        """
        super().__init__()
        self._character = character
        self._width = width
        self._path = path

    @classmethod
    def from_dict(cls, data: dict) -> AvGlyph:
        """Create an AvGlyph instance from a dictionary."""

        path_dict = data.get("path")
        if path_dict is None:
            path = AvPath()
        else:
            path = AvPath.from_dict(path_dict)

        return cls(
            character=data.get("character", ""),
            width=data.get("width", 0.0),
            path=path,
        )

    def to_dict(self) -> dict:
        """Convert the AvGlyph instance to a dictionary."""

        return {
            "character": self._character,
            "width": self._width,
            "path": self._path.to_dict(),
        }

    @classmethod
    def from_ttfont_character(cls, ttfont: TTFont, character: str, polygonize_steps: int = 0) -> AvGlyph:
        """
        Factory method to create an AvGlyph from a TTFont and character.

        Parameters:
            ttfont (TTFont): The TTFont to use.
            character (str): The character to use.
            polygonize_steps (int, optional): The number of steps to use for polygonization.
                Defaults to 0 = no polygonization.

        Notes:
            If polygonize_steps is 0 the commands could contain also curves
            If polygonize_steps is greater than 0, then curves will be polygonized
        """
        glyph_name = ttfont.getBestCmap()[ord(character)]
        glyph_set = ttfont.getGlyphSet()
        pen = AvGlyphPtsCmdsPen(glyph_set, polygonize_steps=polygonize_steps)
        glyph_set[glyph_name].draw(pen)
        width = glyph_set[glyph_name].width
        # Create AvPath first, then create AvGlyph
        path = AvPath(pen.points, pen.commands)
        return cls(character, width, path)

    @property
    def character(self) -> str:
        """
        The character of this glyph.
        """
        return self._character

    @property
    def path(self) -> AvPath:
        """
        The path object of this glyph containing points and commands.
        """
        return self._path

    def width(self, align: Optional[ave.common.Align] = None) -> float:
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
        elif align == ave.common.Align.LEFT:
            return self._width - bounding_box.xmin
        elif align == ave.common.Align.RIGHT:
            return bounding_box.xmin + bounding_box.width
        elif align == ave.common.Align.BOTH:
            return bounding_box.width
        else:
            raise ValueError(f"Invalid align value: {align}")

    @property
    def height(self) -> float:
        """
        The height of the glyph, i.e. the height of the bounding box (positive value).
        """
        return self.bounding_box().height

    @property
    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a glyph (mostly positive value).
        """
        return self.bounding_box().ymax

    @property
    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a glyph (usually negative value).
        """
        return self.bounding_box().ymin

    @property
    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of a glyph (sign varies +/-).
        """
        return self.bounding_box().xmin

    @property
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of a glyph (sign varies +/-).
        """
        return self._width - self.bounding_box().xmax

    def bounding_box(self) -> AvBox:
        """
        Returns bounding box (tightest box around Glyph)
        Coordinates are relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top
        Uses dimensions in unitsPerEm.
        """
        # Delegate entirely to AvPath's bounding box implementation
        return self._path.bounding_box()


###############################################################################
# Glyph Factory
###############################################################################


class AvGlyphFactory:
    """
    Abstract base class for glyph factories.
    A glyph factory is responsible for creating glyph representations for a character.
    """

    @abstractmethod
    def create_glyph(self, character: str) -> AvGlyph:
        """
        Creates and returns a glyph representation for the specified character.
        Args:
            character (str): The character to create a glyph for.
        Returns:
            AvGlyph: An instance representing the glyph of the specified character.
        """


@dataclass
class AvGlyphFromTTFontFactory(AvGlyphFactory):
    """Factory class for creating glyph instances."""

    _ttfont: TTFont

    def __init__(self, ttfont: TTFont) -> None:
        """
        Initializes the glyph factory.
        """
        self._ttfont = ttfont

    @property
    def ttfont(self) -> TTFont:
        """
        Returns the TTFont instance associated with this glyph factory.
        """
        return self._ttfont

    def create_glyph(self, character: str) -> AvGlyph:
        return AvGlyph.from_ttfont_character(self._ttfont, character)


@dataclass
class AvPolylineGlyphFromTTFontFactory(AvGlyphFromTTFontFactory):
    """
    A factory class for creating glyph instances which polygonizes the curves.
    """

    _polygonize_steps: int

    def __init__(self, ttfont: TTFont, polygonize_steps: int = 50) -> None:
        """
        Initializes the glyph factory.

        Parameters:
            ttfont (TTFont): The TTFont to use.
            polygonize_steps (int, optional): The number of steps to use for polygonization.
                Defaults to 0 = no polygonization.
        """
        super().__init__(ttfont)
        self._polygonize_steps = polygonize_steps

    @property
    def polygonize_steps(self) -> int:
        """
        The number of steps used for polygonization when creating glyphs.
        """
        return self._polygonize_steps

    def create_glyph(self, character: str) -> AvGlyph:
        return AvGlyph.from_ttfont_character(self._ttfont, character, self._polygonize_steps)


###############################################################################
# Letter
###############################################################################


@dataclass
class AvLetter:
    """
    A Letter is a Glyph which is scaled to real dimensions with a position and alignment.
    """

    _glyph: AvGlyph
    _scale: float  # = font_size / units_per_em
    _xpos: float  # left-to-right
    _ypos: float  # bottom-to-top
    _align: Optional[ave.common.Align] = None  # LEFT, RIGHT, BOTH. Defaults to None.

    def __init__(
        self,
        glyph: AvGlyph,
        scale: float,
        xpos: float = 0.0,
        ypos: float = 0.0,
        align: Optional[ave.common.Align] = None,
    ) -> None:
        self._glyph = glyph
        self._scale = scale
        self._xpos = xpos
        self._ypos = ypos
        self._align = align

    @classmethod
    def from_font_size_units_per_em(
        cls,
        glyph: AvGlyph,
        font_size: float,
        units_per_em: float,
        xpos: float = 0.0,
        ypos: float = 0.0,
        align: Optional[ave.common.Align] = None,
    ) -> AvLetter:
        """
        Factory method to create an AvLetter from font_size and units_per_em.
        """

        return cls(glyph, font_size / units_per_em, xpos, ypos, align)

    @property
    def xpos(self) -> float:
        """The x position of the letter in real dimensions."""
        return self._xpos

    @xpos.setter
    def xpos(self, xpos: float) -> None:
        """Sets the x position of the letter in real dimensions."""
        self._xpos = xpos

    @property
    def ypos(self) -> float:
        """The y position of the letter in real dimensions."""
        return self._ypos

    @ypos.setter
    def ypos(self, ypos: float) -> None:
        """Sets the y position of the letter in real dimensions."""
        self._ypos = ypos

    @property
    def scale(self) -> float:
        """Returns the scale factor for the letter which is used to transform the glyph to real dimensions."""
        return self._scale

    @property
    def align(self) -> Optional[ave.common.Align]:
        """The alignment of the letter; None, LEFT, RIGHT, BOTH."""
        return self._align

    @property
    def trafo(self) -> List[float]:
        """
        Returns the affine transformation matrix for the letter to transform the glyph to real dimensions.
        Returns: [scale, 0, 0, scale, xpos, ypos] or [scale, 0, 0, scale, xpos-lsb, ypos] if alignment is LEFT or BOTH.
        """
        if self.align == ave.common.Align.LEFT or self.align == ave.common.Align.BOTH:
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
        if self.align == ave.common.Align.LEFT or self.align == ave.common.Align.BOTH:
            return 0
        return self.scale * self._glyph.left_side_bearing

    @property
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of the Letter taking alignment into account.
        """
        if self.align == ave.common.Align.RIGHT or self.align == ave.common.Align.BOTH:
            return 0
        return self.scale * self._glyph.right_side_bearing

    def bounding_box(self) -> AvBox:
        """
        Returns bounding box (tightest box around Letter) in real dimensions.
        Coordinates are relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top
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
        points = self._glyph.path.points
        commands = self._glyph.path.commands
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
        # Apply scale and translation to the points, make points to be 2 dimensions (x, y)
        points_transformed = points[:, :2] * scale + (translate_x, translate_y)

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


###############################################################################
# Font
###############################################################################


@dataclass
class AvFontProperties:
    """
    Represents the properties of a font.

    The properties are as follows:

    - `ascender`: The highest y-coordinate above the baseline (mostly positive value).
    - `descender`: The lowest y-coordinate below the baseline (usually negative value).
    - `line_gap`: Additional spacing between lines.
    - `x_height`: Height of lowercase 'x'.
    - `cap_height`: Height of uppercase 'H'.
    - `units_per_em`: Units per em.
    - `family_name`: Font family name.
    - `subfamily_name`: Style name (Regular, Bold, etc.).
    - `full_name`: Full font name.
    - `license_description`: License text.
    - `line_height`: Computed line height of the font (ascender - descender + line_gap).
    """

    ascender: float
    descender: float
    line_gap: float

    x_height: float
    cap_height: float
    units_per_em: float

    family_name: str
    subfamily_name: str
    full_name: str
    license_description: str

    @property
    def line_height(self) -> float:
        """Computed line height of the font (ascender - descender + line_gap)."""
        return self.ascender - self.descender + self.line_gap

    def __init__(
        self,
        ascender: float = 0,
        descender: float = 0,
        line_gap: float = 0,
        x_height: float = 0,
        cap_height: float = 0,
        units_per_em: float = 1000,
        family_name: str = "",
        subfamily_name: str = "",
        full_name: str = "",
        license_description: str = "",
    ) -> None:
        super().__init__()
        self.ascender = ascender
        self.descender = descender
        self.line_gap = line_gap
        self.x_height = x_height
        self.cap_height = cap_height
        self.units_per_em = units_per_em
        self.family_name = family_name
        self.subfamily_name = subfamily_name
        self.full_name = full_name
        self.license_description = license_description

    @classmethod
    def from_dict(cls, data: dict) -> AvFontProperties:
        """Create an AvFontProperties instance from a dictionary."""
        return cls(
            ascender=data.get("ascender", 0),
            descender=data.get("descender", 0),
            line_gap=data.get("line_gap", 0),
            x_height=data.get("x_height", 0),
            cap_height=data.get("cap_height", 0),
            units_per_em=data.get("units_per_em", 1000),
            family_name=data.get("family_name", ""),
            subfamily_name=data.get("subfamily_name", ""),
            full_name=data.get("full_name", ""),
            license_description=data.get("license_description", ""),
        )

    def to_dict(self) -> dict:
        """Convert the AvFontProperties instance to a dictionary."""
        return {
            "ascender": self.ascender,
            "descender": self.descender,
            "line_gap": self.line_gap,
            "x_height": self.x_height,
            "cap_height": self.cap_height,
            "units_per_em": self.units_per_em,
            "family_name": self.family_name,
            "subfamily_name": self.subfamily_name,
            "full_name": self.full_name,
            "license_description": self.license_description,
        }

    @classmethod
    def _glyph_visual_height(cls, font: TTFont, char: str) -> float:
        """Return yMax - yMin of the glyph for `char`, or 0.0 if missing/empty."""
        cmap = font.getBestCmap()
        if not cmap or ord(char) not in cmap:
            return 0.0

        glyph_name = cmap[ord(char)]
        glyph_set = font.getGlyphSet()

        pen = BoundsPen(glyph_set)
        try:
            glyph_set[glyph_name].draw(pen)
        except KeyError:
            return 0.0

        if pen.bounds is None:
            return 0.0

        _, y_min, _, y_max = pen.bounds
        return max(0.0, float(y_max - y_min))

    @classmethod
    def _get_name_safe(cls, name_table, name_id: int) -> str:
        """
        Extract a name string with multiple fallbacks.
        This is the most robust way to read name records from real-world fonts.
        """
        # 1. Fast path - getDebugName handles most cases correctly
        name = name_table.getDebugName(name_id)
        if name is not None:
            return name

        # 2. Try Windows Unicode English (platform 3, encoding 1, lang 0x409)
        record = name_table.getName(name_id, 3, 1, 0x409)
        if record is not None:
            try:
                return record.toUnicode()
            except (UnicodeDecodeError, ValueError):
                pass

        # 3. Any Windows record
        record = name_table.getName(name_id, 3, 1)
        if record is not None:
            try:
                return record.toUnicode()
            except (UnicodeDecodeError, ValueError):
                pass

        # 4. Last resort: iterate all records manually
        for record in name_table.names:
            if record.nameID == name_id:
                try:
                    return record.toUnicode()
                except (UnicodeDecodeError, ValueError, AttributeError):
                    continue

        return ""

    @classmethod
    def from_ttfont(cls, ttfont: TTFont) -> "AvFontProperties":
        """
        Create AvFontProperties from a fontTools TTFont object.
        Works with static and variable fonts (as long as the correct variation is active).
        """
        hhea = ttfont["hhea"]
        head = ttfont["head"]
        name_table = ttfont["name"]

        return cls(
            ascender=float(hhea.ascender),  # type: ignore
            descender=float(hhea.descender),  # type: ignore
            line_gap=float(hhea.lineGap),  # type: ignore
            # line_height is automatically computed via @computed_field
            x_height=cls._glyph_visual_height(ttfont, "x"),
            cap_height=cls._glyph_visual_height(ttfont, "H"),
            units_per_em=float(head.unitsPerEm),  # type: ignore
            family_name=cls._get_name_safe(name_table, 1),
            subfamily_name=cls._get_name_safe(name_table, 2),
            full_name=cls._get_name_safe(name_table, 4),
            license_description=cls._get_name_safe(name_table, 13),
        )

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

    def __repr__(self) -> str:
        return f"AvFontProperties({self.family_name} {self.subfamily_name}, {self.units_per_em}upem)"


@dataclass
class AvFont:
    """
    Representation of a Font.
    Holds a Dictionary of glyphs which can be accessed by get_glyph().
    """

    _glyph_factory: AvGlyphFactory
    _font_properties: AvFontProperties = field(default_factory=AvFontProperties)
    _glyphs: Dict[str, AvGlyph] = field(default_factory=dict)

    def __init__(
        self,
        glyph_factory: AvGlyphFactory,
        font_properties: AvFontProperties,
        glyphs: Optional[Dict[str, AvGlyph]] = None,
    ) -> None:
        self._glyph_factory = glyph_factory
        self._font_properties = font_properties
        if glyphs is None:
            glyphs = {}
        self._glyphs = glyphs

    @property
    def props(self) -> AvFontProperties:
        """Returns the AvFontProperties object associated with this font."""
        return self._font_properties

    def get_glyph(self, character: str) -> AvGlyph:
        """Returns the AvGlyph for the given character from the caching dictionary."""
        if character not in self._glyphs:
            self._glyphs[character] = self._glyph_factory.create_glyph(character)
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
        ascender: float = 0.0
        for glyph in self._glyphs.values():
            ascender = max(ascender, glyph.ascender)
        return ascender

    def actual_descender(self):
        """Returns the overall minimum descender by iterating over all glyphs in the cache (negative value)."""
        descender: float = 0.0
        for glyph in self._glyphs.values():
            descender = min(descender, glyph.descender)
        return descender


###############################################################################
# Functions
###############################################################################


# def polygonize_glyph(font_path, character, steps=10) -> AvPolylinePen:
#     """
#     Polygonizes a glyph from a font file into line segments.
#     """
#     font = TTFont(font_path)
#     glyph_set = font.getGlyphSet()
#     glyph_name = font.getBestCmap()[ord(character)]
#     pen = AvPolylinePen(glyph_set, steps)
#     glyph_set[glyph_name].draw(pen)
#     print(type(glyph_set[glyph_name]))
#     print(dir(glyph_set[glyph_name]))
#     print(glyph_set[glyph_name].glyphSet)
#     print(dir(glyph_set[glyph_name].glyphSet))

#     return pen


###############################################################################
# Main
###############################################################################


def main():
    """Main"""
    # font_filename = "fonts/RobotoFlex-VariableFont_GRAD,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"

    # # avfont_w100 = AvFont.instantiate(TTFont(font_filename), {"wght": 100})
    # # glyph = avfont_w100.get_glyph("U")  # I

    # # Example usage:
    # glyph_name = "U"  # Replace with the desired glyph name
    # polyline_pen = polygonize_glyph(font_filename, glyph_name)

    # recording_pen = polyline_pen.recording_pen
    # print(recording_pen.value)

    # # svg_pen = SVGPathPen(TTFont(font_filename).getGlyphSet())
    # # polyline_pen.drawPoints(svg_pen)
    # # svg_path_data = svg_pen.getCommands()
    # # print(svg_path_data)

    # # # Example usage with SVGPathPen:
    # # svg_pen = SVGPathPen(TTFont(font_filename).getGlyphSet())
    # # polyline_pen.draw(svg_pen)
    # # svg_path_data = svg_pen.getCommands()
    # # print(svg_path_data)

    # # # Create an instance of RecordingPen and pass the PolylinePen instance to it
    # # recording_pen = RecordingPen()

    # # # Draw the glyph using the RecordingPen
    # # glyph.draw(polyline_pen)


if __name__ == "__main__":
    main()
