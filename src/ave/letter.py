"""Multi-glyph container for managing collections of AvGlyph objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ave.common import Align
from ave.geom import AvBox
from ave.glyph import AvGlyph
from ave.path import AvPath

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
    _align: Optional[Align] = None  # LEFT, RIGHT, BOTH. Defaults to None.

    def __init__(
        self,
        glyph: AvGlyph,
        scale: float,
        xpos: float = 0.0,
        ypos: float = 0.0,
        align: Optional[Align] = None,
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
        align: Optional[Align] = None,
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
    def align(self) -> Optional[Align]:
        """The alignment of the letter; None, LEFT, RIGHT, BOTH."""
        return self._align

    @property
    def trafo(self) -> List[float]:
        """
        Returns the affine transformation matrix for the letter to transform the glyph to real dimensions.
        Returns: [scale, 0, 0, scale, xpos, ypos] or [scale, 0, 0, scale, xpos-lsb, ypos] if alignment is LEFT or BOTH.
        """
        if self.align in (Align.LEFT, Align.BOTH):
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
        if self.align in (Align.LEFT, Align.BOTH):
            return 0
        return self.scale * self._glyph.left_side_bearing

    @property
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of the Letter taking alignment into account.
        """
        if self.align in (Align.RIGHT, Align.BOTH):
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
        scale, _, _, _, translate_x, translate_y = self.trafo
        return self._glyph.path.svg_path_string(scale, translate_x, translate_y)

    def svg_path_string_debug_polyline(self, stroke_width: float = 1.0) -> str:
        """
        Returns a debug SVG path representation of the letter using only polylines.
        This method converts curves (Q, C) to straight lines between control points,
        making it useful for debugging the path structure.

        Supported commands:
            M (move-to), L (line-to), Z (close-path)
            Q (quadratic bezier) -> converted to L commands
            C (cubic bezier) -> converted to L commands

        Args:
            stroke_width (float): The stroke width used to determine marker sizes.

        Returns:
            str: The debug SVG path string using only polylines with markers.
                Markers include:
                - Squares: Regular points (L commands)
                - Circles: Control points (intermediate points in Q and C commands)
                - Triangles (pointing right): M command points (segment starts)
                - Triangles (pointing left): Points before Z commands (segment ends)
                Note: Triangles are drawn in addition to the base markers (squares/circles).
        """
        scale, _, _, _, translate_x, translate_y = self.trafo
        return self._glyph.path.svg_path_string_debug_polyline(scale, translate_x, translate_y, stroke_width)


###############################################################################
# MultiGlyphLetter
###############################################################################


# @dataclass
# class AvMultiGlyphLetter:
#     """Container for multiple AvGlyph objects that extends AvGlyph.

#     Represents a collection of glyphs that can be managed together,
#     such as a multi-glyph representing the same character but with different weights.
#     Inherits from AvGlyph to provide compatibility with existing glyph systems.
#     """

#     _glyphs: List[AvGlyph] = field(default_factory=list)
#     _x_positions: List[float] = field(default_factory=list)  # X positions for each glyph

#     def __init__(self, character: str, glyphs: List[AvGlyph], x_positions: List[float]):
#         """Initialize AvMultiGlyph with list of glyphs.

#         Args:
#             character: The character of the multi-glyph
#             glyphs: List of AvGlyph objects to contain (can be empty list)
#             x_positions: List of x positions for each glyph. Must have same length as glyphs.
#         """
#         # Initialize parent AvGlyph with placeholder values
#         super().__init__(character=character, width=0.0, path=AvPath())

#         # Set and validate glyphs
#         self._glyphs = glyphs

#         # Validate and set x positions
#         if len(x_positions) != len(glyphs):
#             if len(x_positions) == 0 and len(glyphs) > 0:
#                 # Initialize with 0.0 for all glyphs if empty list provided
#                 self._x_positions = [0.0] * len(glyphs)
#             else:
#                 raise ValueError("x_positions must have the same length as glyphs")
#         else:
#             self._x_positions = x_positions

#         for i, glyph in enumerate(self._glyphs):
#             if not isinstance(glyph, AvGlyph):
#                 raise TypeError(f"Glyph at index {i} is not an AvGlyph object")

#     @property
#     def glyphs(self) -> List[AvGlyph]:
#         """Get the glyphs list."""
#         return self._glyphs

#     @property
#     def x_positions(self) -> List[float]:
#         """Get the x positions list."""
#         return self._x_positions

#     # TODO: implement correct width calculation
#     def width(self, align: Optional[Align] = None) -> float:
#         """Calculate total width of all glyphs."""
#         return sum(glyph.width(align) for glyph in self._glyphs)

#     def bounding_box(self):
#         """Get bounding box that encompasses all glyphs."""
#         if not self._glyphs:
#             return AvBox(0.0, 0.0, 0.0, 0.0)

#         if len(self._glyphs) == 1:
#             bbox = self._glyphs[0].bounding_box()
#             x_pos = self._x_positions[0]
#             return AvBox(bbox.xmin + x_pos, bbox.ymin, bbox.xmax + x_pos, bbox.ymax)

#         # Calculate combined bounding box using pre-calculated x positions
#         # Initialize with first glyph's bounding box
#         first_bbox = self._glyphs[0].bounding_box()
#         first_x = self._x_positions[0]
#         min_x = first_bbox.xmin + first_x
#         max_x = first_bbox.xmax + first_x
#         min_y = first_bbox.ymin
#         max_y = first_bbox.ymax

#         # Process remaining glyphs
#         for i, glyph in enumerate(self._glyphs[1:], 1):
#             bbox = glyph.bounding_box()
#             x_pos = self._x_positions[i]

#             # Transform bbox to its position
#             glyph_min_x = bbox.xmin + x_pos
#             glyph_max_x = bbox.xmax + x_pos
#             glyph_min_y = bbox.ymin
#             glyph_max_y = bbox.ymax

#             min_x = min(min_x, glyph_min_x)
#             max_x = max(max_x, glyph_max_x)
#             min_y = min(min_y, glyph_min_y)
#             max_y = max(max_y, glyph_max_y)

#         return AvBox(min_x, min_y, max_x, max_y)

#     @property
#     def left_side_bearing(self) -> float:
#         """Get left side bearing of the first glyph."""
#         if not self._glyphs:
#             return 0.0
#         return self._glyphs[0].left_side_bearing()

#     @property
#     def right_side_bearing(self) -> float:
#         """Get right side bearing of the last glyph."""
#         if not self._glyphs:
#             return 0.0
#         return self._glyphs[-1].right_side_bearing()
